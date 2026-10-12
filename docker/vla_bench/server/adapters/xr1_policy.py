"""Adapter for XR-1 / Xiaomi-Robotics-1-5B.

NOT a LeRobot policy. XR-1 is a Qwen3-VL-4B VLM whose KV cache conditions a flow-matching DiT; the observation is a
**chat payload**, not a tensor dict, and the action it emits is a 60-dim *relative* packed vector that has to be
un-packed against the current joint state. This adapter reproduces the repo's own serving path
(`mibot/server/runtime/{client,server}.py`, mirrored by `models/xr1/sanity_check.py`) for one observation:

  1. resize both views with the repo's `resize_image(im, factor=32, max_pixels=160000)` (640x480 -> 448x320);
  2. build the two-turn chat template — ego view, right-wrist view, "Generate robot actions for the task: <task>
     /no_cot", then an assistant turn "<cot></cot>" — with `AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")`
     and `images_kwargs={"do_resize": False}` (the images are already sized);
  3. compose the 60-D state vector (`compose_state`: our 6 joints at 8:14, gripper at 15, everything else zero) and
     normalise it to [-1, 1] with q01/q99 from the XR-1 derivative's `normalize.json`;
  4. `model.generate(batch)` with a zero action seed and the canonical action mask;
  5. `denormalize_action(out * mask, mean, std) * mask` -> (T, 60) packed;
  6. `unpack_action(packed, state7, wrist_encoding)` -> (T, 7) **absolute** joint targets.

Two things that are easy to get wrong and are handled here:
  * `XR1.forward` **pops** `action` / `action_mask` / `state` / `prefix_length` out of the batch dict, so the batch
    must be rebuilt from the cached payload on every call.
  * `generate` draws `torch.randn_like(action)`; the offline evaluation must be reproducible, so the RNG is seeded
    per call from a fixed base seed.

Checkpoint: the fine-tune is a DeepSpeed **ZeRO-2** Lightning checkpoint, which keeps an unsharded copy of every
parameter, so `checkpoint/mp_rank_00_model_states.pt` already holds the complete bf16 state dict under `"module"`
with a `model.` prefix — exactly what `mibot/server/deploy.py::load_model` reads. No `zero_to_fp32.py` conversion
(which would write ~22 GB of fp32 that is then cast back to bf16) and nothing is written next to the checkpoint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from .base import PolicyAdapter


class XR1Adapter(PolicyAdapter):
    def __init__(self, checkpoint: str, repo_dir: str, model_dir: str, data_dir: str, device: str = "cuda:0",
                 chunk_len: int = 30, exec_len: int | None = 30, seed: int = 1000,
                 processor_id: str = "Qwen/Qwen3-VL-4B-Instruct", max_pixels: int = 160000, resize_factor: int = 32,
                 build_kwargs: dict | None = None):
        for p in (repo_dir, model_dir):                     # repo for `mibot`, model dir for `export_xr1_dataset`
            if p not in sys.path:
                sys.path.insert(0, p)
        from PIL import Image
        from transformers import AutoProcessor
        from mibot.models import MIMODEL
        from mibot.utils.io import (ACTION_EPS, build_action_mask, compose_state, denormalize_action, resize_image,
                                    validate_quantiles, validate_stats)
        from export_xr1_dataset import unpack_action

        self._Image, self._resize_image, self._unpack = Image, resize_image, unpack_action
        self._denorm, self._eps = denormalize_action, ACTION_EPS
        self.name = f"xr1:{checkpoint}"
        self.device, self.seed = device, int(seed)
        self.max_pixels, self.resize_factor = int(max_pixels), int(resize_factor)

        data = Path(data_dir)
        norm = json.loads((data / "normalize.json").read_text())
        manifest = json.loads((data / "manifest.json").read_text())
        self.wrist_encoding = manifest["wrist_encoding"]
        L = int(norm["_meta"]["action_length"])
        self.chunk_len = int(chunk_len or L)
        self.exec_len = int(exec_len or self.chunk_len)
        assert L == self.chunk_len, f"normalize.json action_length={L} != chunk_len={self.chunk_len}"
        mean, std = validate_stats(norm["mean"], norm["std"], L)
        q01, q99 = validate_quantiles(norm["q01"], norm["q99"])
        self.mean_t = torch.tensor(mean, device=device)
        self.std_t = torch.tensor(std, device=device)
        self.q01_t = torch.tensor(q01, device=device)
        self.q99_t = torch.tensor(q99, device=device)
        self.valid = torch.from_numpy(q99 > q01).to(device)
        self.action_mask = torch.from_numpy(build_action_mask(L)).to(device)
        self._compose_state = compose_state

        # ffn_gradient_checkpointing is True in the training config; it only wraps mlp.forward with a checkpoint that
        # short-circuits outside training, changes no state-dict key, and is pointless under eval()/no_grad.
        kw = {"type": "xr1", "freq_coefficient": 1.0, "freq_excluded_dims": [17, 18, 19],
              "ffn_gradient_checkpointing": False, "async_train": True}
        kw.update(build_kwargs or {})
        self.build_kwargs = kw
        model = MIMODEL.build(dict(kw)).to(torch.bfloat16)
        blob = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        self.train_step = int(blob.get("global_step", blob.get("global_steps", -1)))
        sd = {k[len("model."):]: v for k, v in blob["module"].items() if k.startswith("model.")}
        info = model.load_state_dict(sd, strict=True)
        self.load_info = str(info)
        del blob, sd
        self.model = model.eval().to(device)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.num_steps = int(getattr(self.model, "num_steps", -1))

        self.processor = AutoProcessor.from_pretrained(processor_id)
        self.processor.tokenizer.padding_side = "right"
        self.processor_id = processor_id
        self._payload_cache: dict = {}

    # ------------------------------------------------------------------ prompt
    def _payload(self, primary, wrist, task: str):
        ego = self._resize_image(self._Image.fromarray(np.ascontiguousarray(primary)),
                                 factor=self.resize_factor, max_pixels=self.max_pixels)
        wr = self._resize_image(self._Image.fromarray(np.ascontiguousarray(wrist)),
                                factor=self.resize_factor, max_pixels=self.max_pixels)
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "The following observations are captured from multiple views.\n# Ego View\n"},
                {"type": "image", "image": ego},
                {"type": "text", "text": "\n# Right-Wrist View\n"},
                {"type": "image", "image": wr},
                {"type": "text", "text": f"\nGenerate robot actions for the task:\n{task} /no_cot"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]
        return self.processor.apply_chat_template([messages], tokenize=True, return_dict=True, return_tensors="pt",
                                                  padding=True, images_kwargs={"do_resize": False})

    def warmup(self):
        pass

    @torch.no_grad()
    def predict(self, obs: dict) -> np.ndarray:
        state7 = np.asarray(obs["state"], np.float32).reshape(7)
        payload = self._payload(obs["primary"], obs["wrist"], str(obs["task"]))
        state60 = self._compose_state(left_gripper=np.zeros(1, np.float32), left_joint=np.zeros(6, np.float32),
                                      right_gripper=state7[6:7], right_joint=state7[:6])
        # rebuild the batch every call: XR1.forward pops action / action_mask / state / prefix_length out of it
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in payload.items()}
        batch["state"] = torch.from_numpy(state60)[None].to(self.device)
        b = batch["input_ids"].shape[0]
        mask = self.action_mask.unsqueeze(0).expand(b, -1, -1)
        batch["action"] = torch.zeros((b, self.chunk_len, self.mean_t.shape[-1]), device=self.device, dtype=torch.bfloat16)
        batch["action_mask"] = mask
        st = batch["state"]
        ns = torch.zeros_like(st)
        v = self.valid[0]
        ns[..., v] = 2.0 * (st[..., v] - self.q01_t[..., v]) / (self.q99_t[..., v] - self.q01_t[..., v] + self._eps) - 1.0
        batch["state"] = ns.clamp(-1.0, 1.0)
        torch.manual_seed(self.seed)          # generate() draws randn_like(action); keep the evaluation reproducible
        out = self.model.generate(batch)
        out = self._denorm(out * mask, self.mean_t, self.std_t) * mask
        packed = out[0].float().cpu().numpy()
        a = self._unpack(packed, state7.astype(np.float64), self.wrist_encoding)
        a = np.asarray(a, np.float64)[: self.chunk_len, :7]
        assert a.shape == (self.chunk_len, 7), f"expected ({self.chunk_len}, 7), got {a.shape}"
        return a.astype(np.float32)

    def info(self) -> dict:
        d = super().info()
        d.update({"policy_type": "xr1 (Xiaomi-Robotics-1-5B)",
                  "cameras": {"primary": "ego view (cam_high)", "wrist": "right-wrist view (cam_right_wrist)"},
                  "state_key": "60-D composed state, right joints at 8:14 and gripper at 15, q01/q99 normalised",
                  "packed_action_dims": {"joints 0-2": "packed 8:11", "joints 3-5": "packed 0:3", "gripper": "packed 14"},
                  "wrist_encoding": self.wrist_encoding,
                  "flow_steps": self.num_steps, "params_b": round(self.n_params / 1e9, 3),
                  "checkpoint_global_step": self.train_step, "load_state_dict": self.load_info,
                  "processor": self.processor_id, "build_kwargs": self.build_kwargs, "seed": self.seed,
                  "dtype": "bfloat16"})
        return d
