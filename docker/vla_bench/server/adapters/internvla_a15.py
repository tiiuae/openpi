"""Adapter for InternVLA-A1.5 (`policy.type = internvla_a1_5`).

InternVLA-A1.5 ships inside a *fork* of LeRobot (1.0.0) whose inference contract is NOT the plain LeRobot one:
`predict_action_chunk` expects a fully tokenised Qwen3.5-VL chat batch (prompt text with "Task: ...; Control Mode:
<joint>; State: <256-bin discretised state>", image-pad tokens, attention mask, 32-D padded state), and the
normalisation lives in the *dataset* transform pipeline rather than in a LeRobot pre/post-processor. So
`adapters/lerobot_policy.py` cannot drive it.

Rather than re-implement that pipeline, this adapter drives the repo's OWN deployment backend,
`evaluation/LIBERO/policy_server/backends/policy_backend_internvla_a1_5.py::InternVLAA15Backend`, which is exactly
the code path the authors use to serve the policy to a robot from raw camera frames. It:

  1. maps our two views onto the schema slots (`dataset_schemas/configs/trossen_ai_mobile.yaml`:
     cam_high -> image0, cam_right_wrist -> image1; slot image2 is filled with white and masked off, as
     `RemapImageKeyTransformFn` does at training time),
  2. letterboxes both views to 224x224 (`ResizeImagesWithPadFn`, bilinear, 224x168 + 28-px bars for 640x480) --
     see `letterbox` below: the backend alone does NOT do this,
  3. normalises the 7-D state with the checkpoint's own `stats.json` (mean/std),
  4. builds the eval-mode chat prompt with `InternVLAA15ChatProcessorTransformFn(mode="eval")`,
  5. calls `predict_action_chunk`, then un-normalises with mean/std (mode read from the checkpoint's
     `train_config.json`) and clips to the training action min/max.

Two changes to the backend, everything else is the authors' code:
  * `_sample_to_inputs`, overridden to cast float tensors to the compute dtype so the weights and the batch agree —
    the same thing `models/internvla_a15/sanity_check.py::make_batch` does;
  * `letterbox` (default on, 2026-09-29): the backend constructs `ResizeImagesWithPadFn(224, 224)` with an EMPTY
    `mapping` -- it is never hydrated -- and that transform resizes only the keys in its mapping, so the backend passes
    every frame through at the size it arrives. Training hydrated the same transform from the dataset schema
    (trossen_ai_mobile: cam_high -> image0, cam_right_wrist -> image1) and letterboxed both views before the remap.
    On the old 224x224 client wire this went unnoticed (right token count, squashed geometry); on the native 640x480
    wire the model received unresized frames (a 30x40 Qwen grid instead of training's 16x16). The adapter gives the
    backend the same transform with the image0/image1 slot keys, which is where `build_base_sample` holds the frames
    when it calls it; image2 (the masked white filler) stays unresized, as in training. Checked against the trainer's
    own dataset samples: `_scratch_newwire/internvla_verify_inputs.json`. `letterbox=False` reproduces the adapter as
    it was before (the legacy-wire numbers).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from .base import PolicyAdapter


class InternVLAA15Adapter(PolicyAdapter):
    def __init__(self, checkpoint: str, repo_dir: str, device: str = "cuda", dtype: str = "bfloat16",
                 chunk_len: int = 30, exec_len: int | None = 30, robot_type: str = "trossen_ai_mobile",
                 resize_size: int = 224, max_prompt_length: int = 650, inference_backend: str = "standard",
                 action_loss_only: bool = True, vlm_model_path: str | None = None, stats_key: str | None = None,
                 letterbox: bool = True):
        backends_root = str(Path(repo_dir) / "evaluation" / "LIBERO")
        if backends_root not in sys.path:
            sys.path.insert(0, backends_root)
        from policy_server.backends.policy_backend_internvla_a1_5 import InternVLAA15Backend  # noqa: E402

        torch_dtype = getattr(torch, dtype)

        class _Backend(InternVLAA15Backend):
            """Same backend, but the batch is cast to the model's compute dtype (sanity_check.py::make_batch)."""

            def _sample_to_inputs(self, sample):
                inputs = {}
                for key, value in sample.items():
                    if key == "task":
                        inputs[key] = [value]
                        continue
                    if isinstance(value, bool) or not isinstance(value, torch.Tensor):
                        continue
                    v = value.unsqueeze(0).to(self.device)
                    if v.dtype.is_floating_point:
                        v = v.to(torch_dtype)
                    inputs[key] = v
                return inputs

        self.name = f"internvla_a1_5:{checkpoint}"
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        self.backend = _Backend(ckpt_path=checkpoint, device=device, stats_key=stats_key, robot_type=robot_type,
                                resize_size=resize_size, max_prompt_length=max_prompt_length,
                                vlm_model_path=vlm_model_path, action_loss_only=action_loss_only,
                                inference_backend=inference_backend)
        self.letterbox = bool(letterbox)
        if self.letterbox:
            from lerobot.transforms.core import ResizeImagesWithPadFn
            self.backend.resize = ResizeImagesWithPadFn(height=resize_size, width=resize_size, mapping={
                "observation.images.image0": "observation.images.image0",
                "observation.images.image1": "observation.images.image1"})
        # config.dtype is bfloat16 for this checkpoint; the backend leaves the weights as loaded and relies on
        # autocast. Cast explicitly so the reported VRAM and latency describe the bf16 deployment recipe the
        # README prescribes for the real robot (and the one sanity_check.py measured).
        self.backend.policy.to(dtype=torch_dtype)
        self.backend.compute_dtype = torch_dtype
        cfg = self.backend.policy.config
        self.cfg = cfg
        self.chunk_len = int(chunk_len or cfg.chunk_size)
        self.exec_len = int(exec_len or cfg.n_action_steps)
        self.device, self.torch_dtype = device, torch_dtype
        assert int(cfg.chunk_size) == self.chunk_len, (
            f"checkpoint chunk_size={cfg.chunk_size} != requested chunk_len={self.chunk_len}")

    def warmup(self) -> None:
        pass

    def predict(self, obs: dict) -> np.ndarray:
        example = {
            # slot order follows dataset_schemas/configs/trossen_ai_mobile.yaml: image0 = cam_high, image1 = cam_right_wrist
            "image": [np.ascontiguousarray(obs["primary"]), np.ascontiguousarray(obs["wrist"])],
            "state": np.asarray(obs["state"], np.float32),
            "lang": str(obs["task"]),
        }
        out = self.backend.infer({"examples": [example]})
        a = np.asarray(out["actions"], np.float64)
        if a.ndim == 3:
            a = a[0]
        a = a[: self.chunk_len, :7]
        assert a.shape == (self.chunk_len, 7), f"expected ({self.chunk_len}, 7), got {a.shape}"
        return a.astype(np.float32)

    def info(self) -> dict:
        d = super().info()
        d.update({"policy_type": "internvla_a1_5",
                  "cameras": {"primary": "observation.images.image0 (cam_high)",
                              "wrist": "observation.images.image1 (cam_right_wrist)",
                              "image2": "masked white filler"},
                  "state_key": "observation.state",
                  "robot_type": self.backend.robot_type,
                  "stats_key": self.backend.stats_key,
                  "action_denorm_mode": self.backend.action_denorm_mode,
                  "inference_backend": getattr(self.cfg, "inference_backend", "?"),
                  "action_loss_only": bool(getattr(self.cfg, "action_loss_only", True)),
                  "letterbox": self.letterbox,
                  "num_inference_steps": int(getattr(self.cfg, "num_inference_steps", -1)),
                  "dtype": str(self.torch_dtype)})
        return d
