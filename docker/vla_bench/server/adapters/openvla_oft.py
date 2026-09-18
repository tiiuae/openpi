"""Adapter for OpenVLA-OFT+ (11th benchmark model).

Runs the policy IN-PROCESS rather than through the authors' FastAPI `vla-scripts/deploy.py` server, so the
evaluation harness measures model latency and not HTTP round-trips. The loading sequence mirrors
`experiments/robot/openvla_utils.py`: base VLA + merged LoRA, then the two side-car modules the OFT+ recipe
trains alongside it (the L1 regression action head and the proprio projector), then the FiLM-wrapped vision
backbone that ships inside the checkpoint.

Must be imported with "right7" somewhere in sys.argv, or `prismatic.vla.constants` silently falls back to the
LIBERO block (chunk 8, proprio 8) and every prediction is the wrong shape. `_ensure_right7_constants()` below
enforces that rather than trusting the caller.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
from .base import PolicyAdapter

OPENVLA_IMAGE_SIZE = 224   # openvla_utils.OPENVLA_IMAGE_SIZE


def _ensure_right7_constants() -> None:
    """The upstream constants module picks its block by sniffing sys.argv. Make the trigger present."""
    if not any("right7" in a.lower() for a in sys.argv):
        sys.argv.append("--dataset_name=right7_2view_v1")


class OpenVLAOFTAdapter(PolicyAdapter):
    name = "openvla_oft"

    def __init__(self, checkpoint_dir: str, repo_dir: str, chunk_len: int = 30,
                 exec_len: int | None = None, center_crop: bool = True, unnorm_key: str = "right7_2view_v1"):
        _ensure_right7_constants()
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        self.ckpt = Path(checkpoint_dir)
        self.chunk_len, self.exec_len = chunk_len, exec_len
        self.center_crop, self.unnorm_key = center_crop, unnorm_key
        self._model = None
        stats = self.ckpt / "dataset_statistics.json"
        if not stats.is_file():
            raise FileNotFoundError(
                f"{stats} is missing. It is written by finetune.py into every checkpoint dir and holds the "
                f"action/proprio normalisation bounds; without it predictions are unnormalised.")
        self.norm_stats = json.loads(stats.read_text())

    def warmup(self) -> None:
        import torch
        # openvla_utils imports tensorflow (it is used for the lanczos resize and the center crop) and TF
        # would otherwise claim the whole GPU before torch allocates. The repo does the same thing in
        # prismatic/vla/datasets/rlds/dataset.py:35; that module is not on this import path, so do it here.
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM
        if (NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM) != (self.chunk_len, 7, 7):
            raise RuntimeError(
                f"constants mismatch: the repo resolved chunk={NUM_ACTIONS_CHUNK}, action={ACTION_DIM}, "
                f"proprio={PROPRIO_DIM} but this embodiment is ({self.chunk_len}, 7, 7). The 'right7' argv "
                f"trigger did not take effect.")
        from experiments.robot.openvla_utils import (
            get_vla, get_processor, get_action_head, get_proprio_projector)

        class _Cfg:  # openvla_utils reads attributes off a config object
            # robot_utils.get_action() dispatches on this and raises ValueError for anything else.
            model_family = "openvla"
            pretrained_checkpoint = str(self.ckpt)
            use_l1_regression, use_diffusion, use_film = True, False, True
            num_images_in_input, use_proprio, center_crop = 2, True, True
            lora_rank, load_in_8bit, load_in_4bit = 32, False, False
            unnorm_key = self.unnorm_key

        cfg = _Cfg(); cfg.center_crop = self.center_crop; cfg.unnorm_key = self.unnorm_key
        self._cfg = cfg
        self._vla = get_vla(cfg)
        self._processor = get_processor(cfg)
        self._action_head = get_action_head(cfg, self._vla.llm_dim)
        self._proprio_projector = get_proprio_projector(cfg, self._vla.llm_dim, proprio_dim=PROPRIO_DIM)
        self._torch = torch
        # one throwaway pass so the reported latency excludes lazy CUDA/kernel init
        self.predict({"primary": np.zeros((480, 640, 3), np.uint8),
                      "wrist": np.zeros((480, 640, 3), np.uint8),
                      "state": np.zeros(7, np.float32), "task": "warmup"})

    def predict(self, obs: dict) -> np.ndarray:
        from experiments.robot.robot_utils import get_action
        payload = {
            "full_image": obs["primary"],
            "wrist_image": obs["wrist"],
            "state": np.asarray(obs["state"], dtype=np.float32),
        }
        actions = get_action(
            self._cfg, self._vla, payload, obs["task"],
            processor=self._processor, action_head=self._action_head,
            proprio_projector=self._proprio_projector, use_film=True,
        )
        a = np.asarray(actions, dtype=np.float32).reshape(-1, 7)
        if a.shape[0] != self.chunk_len:
            raise RuntimeError(f"expected a {self.chunk_len}-step chunk, got {a.shape[0]}")
        return a
