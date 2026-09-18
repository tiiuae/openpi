"""Adapter for FastWAM (world-action model, Wan2.2-TI2V-5B video expert + ActionDiT action expert).

NOT a LeRobot policy. FastWAM is driven through `FastWAM.infer_action(...)`, which wants a single RGB *canvas*
(both camera views concatenated side by side), a min/max-normalised proprio vector, and a **precomputed T5 text
embedding** — the model is instantiated with `load_text_encoder: false`, exactly as it was trained, so the four
benchmark instructions are read from the embedding cache built by the pre-processing step of the training job.

**Horizon: this is the one model in the benchmark that does not predict 30 steps.** Its video tokenizer forces the
horizon to be a multiple of 4 (BLOCKERS D-FW1), so it trained at the author-native 32 and the deployment executes
only the FIRST 30 of the 32 predicted steps. Construct it with `chunk_len=32, exec_len=30`: `eval.py` strides by
`exec_len`, so the replan cadence is the same 1.00 s as every other model, and `make_deployment_guide.py` renders
"chunk 32 predicted, 30 executed".

Observation pipeline, reproducing `datasets/lerobot/robot_video_dataset.py` + `processors/fastwam_processor.py`:
  * per camera: uint8 HWC RGB -> CHW float32 /255 -> torchvision `Resize([224, 224])` (bilinear + antialias).
    This is a SQUASH of the 480x640 frame, not an aspect-preserving resize + crop — the repo's LIBERO helper
    `_center_crop_resize` is for square LIBERO frames and must NOT be used here.
  * `torch.cat([cam_high, cam_right_wrist], dim=-1)` -> [3, 224, 448] (`concat_multi_camera: horizontal`,
    `video_size: [224, 448]`, camera order = `shape_meta.images` order). The dataset's own
    ResizeSmallestSideAspectPreserving + CenterCrop to [224, 448] are then identities.
  * `(x - 0.5) / 0.5` -> [-1, 1]. `infer_action` takes the t=0 frame of that canvas as `input_image`.
Proprio: min/max-normalised to [-1, 1] with the run's `dataset_stats.json` `state.default.global_min/max`, then
clamped to [-5, 5] (`utils/normalizer.py::SingleFieldLinearNormalizer.forward`).
Action: `infer_action` returns the NORMALISED chunk; un-normalise with the same linear map built from
`action.default.global_min/max`. The dataset is absolute joint targets (`delta_action_dim_mask` all false), so the
result is directly the benchmark's 7-D absolute targets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .base import PolicyAdapter

# robot_video_dataset.py: DEFAULT_PROMPT
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def _linear_norm(stats_min, stats_max, range_tol: float = 1e-4):
    """SingleFieldLinearNormalizer("min/max") -> (scale, offset) mapping [min, max] onto [-1, 1]."""
    lo = np.asarray(stats_min, np.float64)
    hi = np.asarray(stats_max, np.float64)
    rng = hi - lo
    ignore = rng < range_tol
    rng = np.where(ignore, 2.0, rng)          # output_max - output_min
    scale = 2.0 / rng
    offset = -1.0 - scale * lo
    offset = np.where(ignore, 0.0 - lo, offset)  # (output_max + output_min)/2 - input_min
    return scale, offset


class FastWAMAdapter(PolicyAdapter):
    def __init__(self, checkpoint: str, run_dir: str, text_embed_cache: str, device: str = "cuda:0",
                 dtype: str = "bfloat16", chunk_len: int = 32, exec_len: int = 30,
                 num_inference_steps: int | None = None, seed: int | None = None,
                 dataset_stats: str | None = None, compile_action_infer: bool = False,
                 context_len: int = 128):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        self.name = f"fastwam:{checkpoint}"
        run = Path(run_dir)
        cfg = OmegaConf.load(run / "config.yaml")          # already fully resolved by the training job
        self.torch_dtype = getattr(torch, dtype)
        self.model = instantiate(cfg.model, model_dtype=self.torch_dtype, device=device)

        payload = torch.load(checkpoint, map_location="cpu")
        # trainer.py saves {"mot": ..., "proprio_encoder": ..., "step", "torch_dtype"}; model.load_checkpoint uses
        # strict=False for `mot`, which would silently accept a partial load — assert instead.
        missing, unexpected = self.model.mot.load_state_dict(payload["mot"], strict=False)
        assert not missing and not unexpected, f"mot load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
        self.model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        self.train_step = int(payload.get("step", -1))
        self.model = self.model.to(device).eval()

        stats_path = Path(dataset_stats) if dataset_stats else run / "dataset_stats.json"
        st = json.load(open(stats_path))
        self.a_scale, self.a_offset = _linear_norm(st["action"]["default"]["global_min"], st["action"]["default"]["global_max"])
        self.s_scale, self.s_offset = _linear_norm(st["state"]["default"]["global_min"], st["state"]["default"]["global_max"])
        self.stats_path = str(stats_path)

        self.device = device
        self.chunk_len = int(chunk_len)
        self.exec_len = int(exec_len)
        self.steps = int(num_inference_steps if num_inference_steps is not None else cfg.eval_num_inference_steps)
        self.seed = int(seed if seed is not None else cfg.seed)
        self.compile_action_infer = bool(compile_action_infer)
        self.context_len = int(context_len)
        self.cache = Path(text_embed_cache)
        self._ctx: dict[str, tuple] = {}
        # the canvas geometry is fixed by the training config, not guessed
        self.video_size = [int(x) for x in cfg.data.train.video_size]                 # [H, W] = [224, 448]
        self.cam_order = [c.key for c in cfg.data.train.shape_meta.images]            # cam_high, cam_right_wrist
        self.cam_shape = [int(x) for x in cfg.data.train.shape_meta.images[0].shape]  # [3, 224, 224]
        assert cfg.data.train.concat_multi_camera == "horizontal", cfg.data.train.concat_multi_camera
        assert self.video_size == [self.cam_shape[1], self.cam_shape[2] * len(self.cam_order)], \
            f"canvas {self.video_size} != {len(self.cam_order)}x{self.cam_shape[1:]} side by side"
        import torchvision.transforms as T
        self.resize = T.Resize(size=list(self.cam_shape[1:]))

    # ---------------------------------------------------------------- observation construction
    def _canvas(self, primary, wrist):
        views = []
        for img in (primary, wrist):                       # cam_high LEFT, cam_right_wrist RIGHT
            x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)   # HWC uint8 -> CHW uint8
            assert x.dtype == torch.uint8 and x.shape[0] == 3, (x.dtype, x.shape)
            views.append(self.resize(x.to(torch.float32) / 255.0))             # ToTensor + Resize (squash)
        c = torch.cat(views, dim=-1)                                            # [3, 224, 448]
        c = (c - 0.5) / 0.5                                                     # Normalize(mean .5, std .5) -> [-1, 1]
        return c.unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)

    def _context(self, task: str):
        if task not in self._ctx:
            h = hashlib.sha256(DEFAULT_PROMPT.format(task=task).encode("utf-8")).hexdigest()
            p = self.cache / f"{h}.t5_len{self.context_len}.wan22ti2v5b.pt"
            if not p.exists():
                raise FileNotFoundError(f"no cached T5 embedding for task {task!r} at {p} — the model is built with "
                                        f"load_text_encoder=false and cannot embed a prompt at run time")
            payload = torch.load(p, map_location="cpu")
            ctx, mask = payload["context"].clone(), payload["mask"].bool()
            ctx[~mask] = 0.0                                # robot_video_dataset.py: zero the padding …
            self._ctx[task] = (ctx.unsqueeze(0), torch.ones_like(mask).unsqueeze(0))  # … and hand the model an all-ones mask
        return self._ctx[task]

    def warmup(self):
        pass

    @torch.no_grad()
    def predict(self, obs: dict) -> np.ndarray:
        ctx, mask = self._context(str(obs["task"]))
        s = np.asarray(obs["state"], np.float64)
        proprio = np.clip(s * self.s_scale + self.s_offset, -5.0, 5.0).astype(np.float32)
        out = self.model.infer_action(
            prompt=None,
            input_image=self._canvas(obs["primary"], obs["wrist"]),
            action_horizon=self.chunk_len,
            proprio=torch.from_numpy(proprio).unsqueeze(0),
            context=ctx, context_mask=mask,
            num_inference_steps=self.steps,
            seed=self.seed, rand_device="cpu",
            compile_action_infer=self.compile_action_infer,
        )["action"]
        a = np.asarray(out.float().cpu().numpy(), np.float64)
        a = (a - self.a_offset) / self.a_scale              # SingleFieldLinearNormalizer.backward
        a = a[: self.chunk_len, :7]
        assert a.shape == (self.chunk_len, 7), f"expected ({self.chunk_len}, 7), got {a.shape}"
        return a.astype(np.float32)

    def info(self) -> dict:
        d = super().info()
        d.update({"policy_type": "fastwam",
                  "cameras": {"primary": "cam_high (canvas left half)", "wrist": "cam_right_wrist (canvas right half)"},
                  "canvas_hw": self.video_size, "camera_order": self.cam_order,
                  "state_key": "proprio (min/max normalised, clamped to [-5, 5])",
                  "num_inference_steps": self.steps, "seed": self.seed,
                  "compile_action_infer": self.compile_action_infer,
                  "train_step_in_checkpoint": self.train_step,
                  "dataset_stats": self.stats_path,
                  "text_embed_cache": str(self.cache),
                  "dtype": str(self.torch_dtype),
                  "horizon_note": "chunk_len 32 predicted (video tokenizer needs a multiple of 4; BLOCKERS D-FW1); "
                                  "exec_len 30 executed, so the replan cadence matches the rest of the benchmark"})
        return d
