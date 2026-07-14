"""ACT policy wrapped as an openpi-style `infer(obs) -> {"actions": ...}`.

The robot client (openpi `examples/trossen_ai/main.py`) sends, each query:

    observation = {
        "state":  np.ndarray (state_dim,)            # joint positions, RLDS order
        "images": {robot_cam_name: img, ...}         # uint8, CHW or HWC, RGB
        "prompt": str                                # IGNORED by ACT (visuomotor)
    }

and expects back:

    {"actions": np.ndarray (chunk_size, action_dim)}  # robot-space joint targets

This adapter:
  * maps the model's `camera_names` (e.g. primary/secondary/wrist) onto the robot
    camera keys via `camera_map`,
  * coerces each image to HWC uint8 and resizes to the TRAINING resolution
    (the ResNet backbone is size-agnostic but was trained at a fixed scale, so
    matching it matters),
  * runs `model.predict_action(state, stack, denormalize=True)`.

NOTE on ordering: `state` and the returned `actions` are in whatever joint order
the RLDS dataset used at training time. The robot client must send state — and
apply actions — in that SAME order. Verify this against your dataset spec; a
permutation here silently produces garbage motion.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from PIL import Image


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """Accept (H,W,3) or (3,H,W), float[0,1] or uint8 → return (H,W,3) uint8 RGB."""
    a = np.asarray(img)
    if a.ndim != 3:
        raise ValueError(f"expected a 3D image, got shape {a.shape}")
    # CHW -> HWC (channels-first if first axis is 1/3 and last axis isn't)
    if a.shape[0] in (1, 3) and a.shape[2] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    if a.dtype != np.uint8:
        a = (a * 255.0 if float(a.max()) <= 1.5 else a).clip(0, 255).astype(np.uint8)
    return a


class ActPolicy:
    """openpi `BasePolicy`-compatible wrapper around an ACT HF snapshot."""

    def __init__(
        self,
        model,
        camera_names: List[str],
        camera_map: Dict[str, str],
        image_height: int = 480,
        image_width: int = 640,
    ):
        self.model = model
        self.camera_names = list(camera_names)          # model's expected cameras, in order
        self.camera_map = dict(camera_map)              # model_cam -> robot_cam key
        self.size = (image_height, image_width)
        missing = [c for c in self.camera_names if c not in self.camera_map]
        if missing:
            raise ValueError(f"camera_map is missing model cameras: {missing}")

    def _build_stack(self, images: Dict[str, np.ndarray]) -> np.ndarray:
        h, w = self.size
        frames = []
        for cam in self.camera_names:
            robot_key = self.camera_map[cam]
            if robot_key not in images:
                raise KeyError(
                    f"model camera '{cam}' maps to robot camera '{robot_key}', "
                    f"which is not in the observation. Got: {list(images.keys())}"
                )
            hwc = _to_hwc_uint8(images[robot_key])
            if hwc.shape[:2] != (h, w):
                hwc = np.asarray(Image.fromarray(hwc).resize((w, h), Image.BILINEAR))
            frames.append(hwc)
        return np.stack(frames, axis=0)                 # (K, H, W, 3) uint8

    def infer(self, obs: Dict) -> Dict:
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        images = obs.get("images", obs)                 # tolerate flat dicts too
        stack = self._build_stack(images)
        actions = self.model.predict_action(state, stack, denormalize=True)
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        return {"actions": np.asarray(actions, dtype=np.float32)}

    def reset(self) -> None:
        pass
