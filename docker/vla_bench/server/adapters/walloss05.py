"""Adapter for Wall-OSS-0.5 (wall-x) — 7-D right-arm joint embodiment.

WHY THIS FILE EXISTS AND WHY IT DOES NOT USE THE SHIPPED INFERENCE PATH
----------------------------------------------------------------------
wall-x ships two standalone inference entry points and NEITHER fits this benchmark's embodiment:

* `scripts/fake_inference.py` -> `wall_x._vendor.harrix.envs.libero_common.encode_proprio`, which builds the
  proprioception vector from LIBERO end-effector fields (`eef_pos`, `eef_axisangle`, `gripper`) and decodes
  the model output with `decode_chunk`, which requires a `right_ee_cartesian_pos` slice in `dof_config`.
  This embodiment is 7-D **absolute joint positions**; there is no EEF slice at all, so `decode_chunk` raises
  and `encode_proprio` has nothing to fill the joint dims with.
* `scripts/run_serving.sh` only knows the authors' own robots (`desktop`, `turtle`, `ex001`).

There is also a substantive mismatch in the shipped path even if the LIBERO encoder were patched:
`harrix.adapters.qwen_vlact` builds the prompt with `_camera_label()`, which renders our cameras as
"front view" / "right wrist view", whereas the TRAINING collator renders the raw config names
"face_view" / "right_wrist_view" (`camera_name_mapping` is None in this run's yml). It also normalises the
full 26-D proprioception vector, which maps the 19 virtual `action_padding` dims to **-1**, while the
training collator normalises only the real 7 dims and then zero-pads — the padded dims were **0** in
training. Both differences reach the model (the padded dims are concatenated into `propri_proj`, they are
not masked away), so the shipped path would be evaluating a different input distribution.

WHAT THIS ADAPTER DOES INSTEAD
------------------------------
It reproduces the training pipeline by **calling it**, not by re-implementing it. The observation is turned
into a batch by the run's own `wall_x.data.backends.lerobot.loader.DataCollator`, constructed from the run's
own `right7_2view_ep20.yml`, so the text prompt, the image resize chain, the q01/q99 min-max normalisation,
the zero padding to 26 dims, `agent_pos_mask`, `dof_mask` and `dataset_names` are byte-for-byte the ones the
trainer produced. Only two things are substituted:

  * the images come from the wire instead of `LeRobotDataset.__getitem__` (they are fed through the same
    `_vision_preprocess` resize chain, reproduced verbatim below), and
  * the ground-truth action chunk is replaced by zeros. That is safe and exact: with
    `use_fast_tokenizer=false` the collator's `replace_action_token()` never reads the action values (it only
    strips the `<|action_fast|><|im_end|>\n` placeholder), `dof_mask` is derived from `isnan` and is therefore
    all-ones on the 7 real dims either way, and `action_chunk` is not passed on to the model.

The model itself is loaded through the authors' own inference helpers (`harrix.utils.ckpt_load`,
`harrix.utils.normalizer`, `harrix.utils.train_config`, `wall_x.trainer.trainer_utils.load_wallx_processors`)
— the same sequence `QwenVLActInferAdapter.__init__` uses — and driven through the authors' own
`generate_flow_action`. Nothing about the model or the flow solver is re-implemented here.

Everything that could silently diverge is asserted at warmup: the `<|action|>` / `<|propri|>` token ids seen
by the model must equal the ones the collator emits, the proprioception must be 26-D with a zero tail, and
the returned chunk must be (chunk_len, 26).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

from .base import PolicyAdapter


class WallOSS05Adapter(PolicyAdapter):
    name = "walloss05"

    def __init__(
        self,
        checkpoint_dir: str,
        repo_dir: str,
        train_config: str,
        chunk_len: int = 30,
        exec_len: int | None = 30,
        num_inference_timesteps: int = 10,
        device: str = "cuda",
        seed: int = 1000,
        norm_key: str | None = None,
    ):
        self.ckpt = str(Path(checkpoint_dir).resolve())
        self.repo_dir = str(Path(repo_dir).resolve())
        self.train_config_path = str(Path(train_config).resolve())
        self.chunk_len, self.exec_len = int(chunk_len), (int(exec_len) if exec_len else None)
        self.num_inference_timesteps = int(num_inference_timesteps)
        self.device = device
        self.seed = int(seed)
        self._norm_key_override = norm_key
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

    # ------------------------------------------------------------------ setup
    def build_pipeline(self) -> None:
        """Rebuild the TRAINING data pipeline (collator, resize chain, prompt builder). No GPU, no weights.

        Split out of warmup() so it can be diffed against the real training dataloader on CPU.
        """
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from wall_x.config.loader import load_config
        from wall_x.data.backends.lerobot.build import _build_flat_config
        from wall_x.data.backends.lerobot.config import LerobotConfig
        from wall_x.data.backends.lerobot.loader import DataCollator, get_data_configs
        from wall_x.data.backends.lerobot.utils import (
            get_wallx_normal_text,
            load_norm_stats,
            process_grounding_points,
        )
        self._torch = torch
        self._get_wallx_normal_text = get_wallx_normal_text
        self._process_grounding_points = process_grounding_points

        # ---- 1. the TRAINING data pipeline, rebuilt from the run's own yml -------------------------
        cfg = load_config(self.train_config_path)
        flat, lecfg = _build_flat_config(cfg)
        dl = get_data_configs(flat["data"])
        self._flat, self._lecfg, self._dl = flat, lecfg, dl

        # `get_data_configs` adds 1 to action_horizon; the loader then asks LeRobot for
        # `action_horizon - 1` future frames and the prompt gets that many <|action|> tokens.
        self._T = int(dl["action_horizon"]) - 1
        if self._T != self.chunk_len:
            raise RuntimeError(
                f"config says the trained chunk is {self._T} steps but this adapter was configured with "
                f"chunk_len={self.chunk_len}. Refusing to serve a horizon the model was not trained on.")

        norm_stats = load_norm_stats(
            flat["norm_stats_path"], dl["key_mappings"],
            dof_config=flat["dof_config"], agent_pos_config=flat["agent_pos_config"])
        self._collator = DataCollator(flat, dl, norm_stats, lecfg)

        self._action_active = sum(int(v) for k, v in flat["dof_config"].items() if k != "action_padding")
        self._state_active = sum(int(v) for k, v in flat["agent_pos_config"].items() if k != "action_padding")
        self._action_dim = int(flat["dof_total_dim"])
        if (self._action_active, self._state_active) != (7, 7):
            raise RuntimeError(
                f"this benchmark is a 7-D right-arm embodiment but the config resolves to "
                f"action={self._action_active}, state={self._state_active}")

        # image resize chain: same LerobotConfig the training PreprocessedDataset builds
        self._data_config = LerobotConfig().update(
            train_test_split=dl["train_test_split"], seed=dl["seed"],
            resolution=dl.get("resolution", None), priority_order=dl.get("priority_order", None),
            camera_name_mapping=dl.get("camera_name_mapping", None))
        self._cam_key_mapping = dl["key_mappings"]["camera"]
        # camera ORDER is `LeRobotDatasetMetadata.camera_keys`, which is what _vision_preprocess iterates.
        meta = LeRobotDatasetMetadata(lecfg["repo_id"], root=lecfg.get("root"))
        self._camera_keys = list(meta.camera_keys)
        if len(self._camera_keys) != 2:
            raise RuntimeError(f"expected 2 cameras, dataset meta reports {self._camera_keys}")

    # ------------------------------------------------------------------ setup
    def warmup(self) -> None:
        import torch
        from wall_x._vendor.harrix.utils.ckpt_load import load_state_dict, resolve_checkpoint_dir
        from wall_x._vendor.harrix.utils.normalizer import build_normalizers
        from wall_x._vendor.harrix.utils.train_config import (
            build_model_config,
            load_train_config_with_ckpt_overlay,
            normalize_train_config_for_inference,
            register_data_backend,
        )
        from wall_x.model.qact.qwen2_5.adapter import Qwen2_5Adapter
        from wall_x.trainer.trainer_utils import load_wallx_processors

        self.build_pipeline()
        lecfg = self._lecfg

        # ---- 2. the model, through the authors' own inference loading sequence ----------------------
        ckpt = resolve_checkpoint_dir(self.ckpt)
        self._resolved_ckpt = ckpt
        train_config = load_train_config_with_ckpt_overlay(self.train_config_path, ckpt)
        train_config = normalize_train_config_for_inference(train_config, self.train_config_path)
        register_data_backend(train_config)

        # The normalizer key IS the repo_id: build.py uses `dataset_name = repo_id` and the collator stamps
        # `dataset_names = [repo_id]`. Anything else silently applies the wrong action scale.
        norm_key = self._norm_key_override or str(lecfg["repo_id"])
        na, npr, resolved_key = build_normalizers(ckpt, train_config, norm_key)
        if resolved_key != norm_key:
            raise RuntimeError(
                f"normalizer key {norm_key!r} not found in the checkpoint; build_normalizers fell back to "
                f"{resolved_key!r}. Refusing to un-normalise actions with a different dataset's scale.")
        self._norm_key = resolved_key

        ConfigClass = Qwen2_5Adapter.config_class()
        ModelClass = Qwen2_5Adapter.inference_model_class()
        model_config = build_model_config(ConfigClass, ckpt, train_config, self.train_config_path)

        procs = load_wallx_processors(train_config, normalizer=na, device=self.device)
        processor, tokenizer_mixin = procs["processor"], procs.get("tokenizer_mixin")

        model = ModelClass(model_config, processor, tokenizer_mixin)
        model.resize_token_embeddings(len(processor.tokenizer))
        model.to_bfloat16_for_selected_params()
        state_dict = load_state_dict(ckpt, ModelClass)
        embed_key = "model.embed_tokens.weight"
        if embed_key in state_dict:
            ckpt_vocab = state_dict[embed_key].shape[0]
            if model.model.embed_tokens.weight.shape[0] != ckpt_vocab:
                model.resize_token_embeddings(ckpt_vocab)
        msg = model.load_state_dict(state_dict, strict=False)
        missing = [k for k in msg.missing_keys if "normalizer_" not in k]
        if missing:
            raise RuntimeError(f"checkpoint is missing {len(missing)} model tensors, e.g. {missing[:5]}")
        del state_dict
        model.set_normalizer(copy.deepcopy(na), copy.deepcopy(npr))
        model.eval()
        model.to(self.device)
        model.to_bfloat16_for_selected_params()
        self._model = model

        # ---- 3. fail loudly on anything that could silently diverge ---------------------------------
        ctok = self._collator.processor.tokenizer
        for tok_name, id_key in (("<|action|>", "action_token_id"), ("<|propri|>", "propri_token_id")):
            collator_id = ctok.convert_tokens_to_ids(tok_name)
            model_id = model.action_token_id_set[id_key]
            if collator_id != model_id:
                raise RuntimeError(
                    f"token id mismatch for {tok_name}: the training collator emits {collator_id} but the "
                    f"model was built with {model_id}. The scatter would land on the wrong tokens.")

        self._torch.manual_seed(self.seed)   # flow inference starts from torch.randn; make the run reproducible
        # one throwaway pass so the reported latency excludes lazy CUDA/kernel init
        self.predict({"primary": np.zeros((480, 640, 3), np.uint8),
                      "wrist": np.zeros((480, 640, 3), np.uint8),
                      "state": np.zeros(7, np.float32), "task": "warmup"})
        self._torch.manual_seed(self.seed)

    # ------------------------------------------------------------- preprocessing
    def _vision_preprocess(self, frames: dict):
        """Verbatim reproduction of PreprocessedDataset._vision_preprocess.

        Training decoded a LeRobot float CHW frame and did `(x * 255).to(uint8)`; that round trip is exact
        for uint8 input (verified), so starting from the wire's uint8 HWC array is identical.
        """
        from PIL import Image
        from qwen_vl_utils.vision_process import smart_resize

        processed_frames = []
        for key in self._camera_keys:
            img_pil = Image.fromarray(frames[key])
            orig_width, orig_height = img_pil.size
            target_size = self._data_config.resolution.get(self._cam_key_mapping[key], -1)
            if target_size != -1:
                if orig_width > orig_height:
                    new_width = target_size
                    new_height = int(target_size * orig_height / orig_width)
                else:
                    new_height = target_size
                    new_width = int(target_size * orig_width / orig_height)
                img_pil = img_pil.resize((new_width, new_height))

            current_width, current_height = img_pil.size
            resized_height, resized_width = smart_resize(
                current_height, current_width,
                factor=self._data_config.image_factor,
                min_pixels=self._data_config.min_pixels,
                max_pixels=self._data_config.max_pixels,
            )
            processed_frames.append(img_pil.resize((resized_width, resized_height)))

        return processed_frames, orig_height, orig_width, resized_height, resized_width

    # ------------------------------------------------------------------ inference
    def predict(self, obs: dict) -> np.ndarray:
        torch = self._torch
        frames = {self._camera_keys[0]: np.ascontiguousarray(obs["primary"], dtype=np.uint8),
                  self._camera_keys[1]: np.ascontiguousarray(obs["wrist"], dtype=np.uint8)}
        image_inputs, h, w, resize_h, resize_w = self._vision_preprocess(frames)

        # Same call the training __getitem__ makes. frame_idx only selects frame-ranged instructions, of
        # which this dataset has none (tasks.parquet carries one flat string per episode), and
        # generate_subtask_ratio is 0.0 in this run's config, so the action branch is always taken.
        complete_text, generate_subtask = self._get_wallx_normal_text(
            {"instruction": obs["task"]},
            self._dl["action_horizon"] - 1,
            0,
            self._data_config.priority_order,
            self._cam_key_mapping,
            generate_subtask_ratio=self._data_config.generate_subtask_ratio,
            camera_name_mapping=self._data_config.camera_name_mapping,
        )
        if generate_subtask:
            raise RuntimeError("get_wallx_normal_text took the subtask branch; this must never happen at "
                               "inference (generate_subtask_ratio should be 0).")
        text = self._process_grounding_points(complete_text, h, w, resize_h, resize_w,
                                              self._data_config.model_type)

        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)[: self._state_active]
        item = {
            "image_inputs": image_inputs,
            "text": text,
            # zeros: the collator never reads the values (use_fast_tokenizer=false) and dof_mask comes from
            # isnan, so this reproduces the training dof_mask exactly without inventing a ground truth.
            "action": torch.zeros((self._T, self._action_active), dtype=torch.float32),
            "agent_pos": torch.as_tensor(state, dtype=torch.float32),
            "frame_index": torch.tensor(0),
        }
        batch = self._collator([item])

        if tuple(batch["proprioception"].shape) != (1, 1, self._action_dim):
            raise RuntimeError(f"proprioception is {tuple(batch['proprioception'].shape)}, expected "
                               f"(1, 1, {self._action_dim})")
        tail = batch["proprioception"][0, 0, self._state_active:]
        if tail.abs().max().item() != 0.0:
            raise RuntimeError("the virtual action_padding tail of proprioception is not zero; the training "
                               "collator zero-pads it after normalisation and the model sees those dims.")

        keep = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "pixel_values_videos",
                "video_grid_thw", "proprioception", "agent_pos_mask", "dof_mask", "moe_token_types",
                "dataset_names")
        model_inputs = {k: batch[k] for k in keep if k in batch}
        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor):
                model_inputs[k] = v.to(self.device)

        with torch.no_grad():
            out = self._model.generate_flow_action(
                action_horizon=self._T,
                action_dim=self._action_dim,
                num_inference_timesteps=self.num_inference_timesteps,
                **model_inputs,
            )
        act = out["predict_action"]
        if isinstance(act, torch.Tensor):
            act = act.detach().float().cpu().numpy()
        act = np.asarray(act)
        if act.ndim == 3:
            act = act[0]
        if act.shape != (self._T, self._action_dim):
            raise RuntimeError(f"model returned {act.shape}, expected ({self._T}, {self._action_dim})")
        return act[:, : self._action_active].astype(np.float32)

    def info(self) -> dict:
        d = super().info()
        d.update({"checkpoint": getattr(self, "_resolved_ckpt", self.ckpt),
                  "norm_key": getattr(self, "_norm_key", None),
                  "num_inference_timesteps": self.num_inference_timesteps})
        return d
