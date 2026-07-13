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
  * prepares `state` for the model: pins static dims (qpos_std below threshold) and
    any extra dims beyond what the robot sends to the training mean, so a single-arm
    checkpoint stays in-distribution when served to a bimanual robot; trims model
    actions back to `robot_action_dim` (default 14) for the client.

NOTE on ordering: `state` and the returned `actions` are in whatever joint order
the RLDS dataset used at training time. The robot client must send state — and
apply actions — in that SAME order. Verify this against your dataset spec; a
permutation here silently produces garbage motion.

NOTE on static proprio (single-arm checkpoints): some checkpoints were collected
with one arm held static (e.g. a right-arm-only dataset → left arm dims 0-6 have
qpos_std clamped at the 0.01 floor). At deploy the physical static arm sits at an
arbitrary pose; on a near-degenerate [q01..q99] range that live value normalizes
to a saturated ±N — a value the model never saw — which corrupts the conditioning
for the WHOLE state and makes the policy orient-but-not-descend. To stay
in-distribution we pin any state dim whose qpos_std < `static_state_threshold`
to its training qpos_mean before the model normalizes. Extra dims beyond what the
robot sends are likewise filled with the training mean (not zeros).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


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
        robot_action_dim: Optional[int] = 14,
        static_state_threshold: float = 0.03,
    ):
        self.model = model
        self.camera_names = list(camera_names)          # model's expected cameras, in order
        self.camera_map = dict(camera_map)              # model_cam -> robot_cam key
        self.size = (image_height, image_width)
        self._model_state_dim = int(model.config.state_dim)
        self._model_action_dim = int(model.config.action_dim)
        self.robot_action_dim = robot_action_dim
        missing = [c for c in self.camera_names if c not in self.camera_map]
        if missing:
            raise ValueError(f"camera_map is missing model cameras: {missing}")

        # Training norm stats (numpy, on CPU) — used to pin static dims and pad extras.
        self._state_mean = np.asarray(model.qpos_mean.detach().cpu().numpy(), dtype=np.float32)
        self._state_std = np.asarray(model.qpos_std.detach().cpu().numpy(), dtype=np.float32)
        # A state dim is "static" if its training std is below threshold (clamped at the
        # 0.01 floor → that joint/feature never moved in collection). Pin those to the
        # training mean so the model sees in-distribution conditioning at deploy time.
        self._static_mask = self._state_std < float(static_state_threshold)
        self._static_threshold = float(static_state_threshold)
        self._infer_count = 0

    def _prepare_state(self, raw_state: np.ndarray) -> np.ndarray:
        """Build a full state_dim vector for the model.

        Starts from the training mean (so static + extra dims are in-distribution) and
        overwrites the ACTIVE dims (std >= threshold, within the robot's sent range) with
        the live proprio. The model normalizes internally, so static/extra dims → 0.
        """
        raw = np.asarray(raw_state, dtype=np.float32).reshape(-1)
        if raw.shape[0] > self._model_state_dim:
            raise ValueError(
                f"observation state has {raw.shape[0]} dims but model expects state_dim={self._model_state_dim}"
            )
        state = self._state_mean.copy()
        n = min(raw.shape[0], self._model_state_dim)
        active_idxs = np.where(~self._static_mask[:n])[0]
        state[active_idxs] = raw[active_idxs]
        return state

    def _trim_actions_for_robot(self, actions: np.ndarray) -> np.ndarray:
        """Return only the leading robot dims when the model predicts a wider action vector."""
        actions = np.asarray(actions, dtype=np.float32)
        if self.robot_action_dim is not None and actions.shape[-1] > self.robot_action_dim:
            actions = actions[..., : self.robot_action_dim]
        return actions

    @staticmethod
    def _fmt_row(arr: np.ndarray) -> str:
        return np.array2string(np.asarray(arr), precision=4, max_line_width=120, suppress_small=True)

    def _log_inference(self, raw_state, prepared_state, raw_actions, trimmed_actions):
        static_dims = np.where(self._static_mask)[0].tolist()
        active_dims = [i for i in range(self._model_state_dim) if not self._static_mask[i]]
        level = logging.INFO if self._infer_count == 0 else logging.DEBUG
        if not logger.isEnabledFor(level):
            return
        logger.log(level, "ACT infer #%d", self._infer_count)
        logger.log(level, "  raw state (%d): %s", raw_state.shape[0], self._fmt_row(raw_state))
        logger.log(level, "  prepared state (%d): %s", prepared_state.shape[0], self._fmt_row(prepared_state))
        logger.log(
            level,
            "  pinned dims (static/padded, →training mean): %s",
            static_dims if static_dims else "none",
        )
        logger.log(level, "  active dims (live proprio): %s", active_dims if active_dims else "none")
        # Per-dim span across the chunk = how much the model wants to move each joint.
        # Large span on the active arm dims (e.g. 8,9,10) => model plans a descent.
        # Span ~0 on a dim => model holds it (static or near-static).
        ra = np.asarray(raw_actions)
        ta = np.asarray(trimmed_actions)
        logger.log(level, "  raw action %s  span: %s", ra.shape, self._fmt_row(ra.max(0) - ra.min(0)))
        logger.log(level, "  raw action            min: %s", self._fmt_row(ra.min(0)))
        logger.log(level, "  raw action            max: %s", self._fmt_row(ra.max(0)))
        logger.log(level, "  trimmed action %s  span: %s", ta.shape, self._fmt_row(ta.max(0) - ta.min(0)))
        logger.log(level, "  trimmed action        min: %s", self._fmt_row(ta.min(0)))
        logger.log(level, "  trimmed action        max: %s", self._fmt_row(ta.max(0)))

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
        raw_state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        state = self._prepare_state(raw_state)
        images = obs.get("images", obs)                 # tolerate flat dicts too
        stack = self._build_stack(images)
        actions = self.model.predict_action(state, stack, denormalize=True)
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        raw_actions = np.asarray(actions, dtype=np.float32)
        trimmed = self._trim_actions_for_robot(raw_actions)
        self._log_inference(raw_state, state, raw_actions, trimmed)
        self._infer_count += 1
        return {"actions": trimmed}

    def reset(self) -> None:
        self._infer_count = 0
