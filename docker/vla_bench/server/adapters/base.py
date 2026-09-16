"""Thin adapter interface every model must implement for the unified evaluation (plan §9) and the deployment guides.

obs = {"primary": HxWx3 uint8 RGB (cam_high), "wrist": HxWx3 uint8 RGB (cam_right_wrist), "state": float32[7], "task": str}
predict(obs) -> float32[T, 7] absolute joint targets (6 joints rad + gripper carriage m) at 30 fps, T = chunk length.
"""
from __future__ import annotations
import numpy as np

class PolicyAdapter:
    name: str = "base"
    chunk_len: int = 1          # T actions returned per call
    exec_len: int | None = None # actions consumed per call at deployment (None -> chunk_len)

    def warmup(self) -> None:
        pass

    def predict(self, obs: dict) -> np.ndarray:
        raise NotImplementedError

    def peak_vram_gb(self) -> float | None:
        try:
            import torch
            if torch.cuda.is_available():
                return float(torch.cuda.max_memory_allocated() / 1e9)
        except Exception:
            pass
        return None

    def info(self) -> dict:
        return {"name": self.name, "chunk_len": self.chunk_len, "exec_len": self.exec_len or self.chunk_len}
