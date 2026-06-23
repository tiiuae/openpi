"""CogACT / AAE ensemble (cosine-similarity weighting).

AAE-style (CogACT, arXiv 2411.19650): overlapping predictions are weighted by
their pairwise cosine similarity — predictions that "agree" with the rest receive
higher weight, suppressing outliers.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from .base import ActionEnsemble
from .config import EnsembleConfig
from .factory import register_ensemble

logger = logging.getLogger(__name__)


class CogACTEnsemble(ActionEnsemble):
    def __init__(
        self,
        max_buffer_size: int = 25,
        mode: str = "cogact",  # "cogact" | "latest" | "hybrid"
        lambda_mix: float = 0.5,
    ) -> None:
        self.max_buffer_size = max_buffer_size
        self.mode = mode
        self.lambda_mix = lambda_mix
        self._buffer: list[tuple[int, np.ndarray]] = []
        self._lock = threading.Lock()
        self._warned_depth = False

    def add_chunk(self, query_step: int, chunk: np.ndarray) -> None:
        if not self._warned_depth and len(chunk) > self.max_buffer_size:
            logger.warning(
                "CogACTEnsemble: chunk length %d > max_buffer_size %d; older "
                "overlaps for a step may be evicted before they are consumed.",
                len(chunk), self.max_buffer_size,
            )
            self._warned_depth = True
        with self._lock:
            self._buffer.append((query_step, chunk))
            if len(self._buffer) > self.max_buffer_size:
                self._buffer.pop(0)

    def get_action(self, current_step: int) -> np.ndarray | None:
        with self._lock:
            buf = list(self._buffer)  # snapshot under lock
        candidates = [chunk[current_step - qs] for qs, chunk in buf if 0 <= current_step - qs < len(chunk)]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0].copy()

        mat = np.array(candidates)  # (N, D)

        # normalize
        norms = np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-8)
        normed = mat / norms

        N = len(mat)

        # --- consensus term ---
        if self.mode in ("cogact", "hybrid"):
            sim = normed @ normed.T  # (N, N)
            consensus = sim.mean(axis=1)  # (N,)
        else:
            consensus = np.zeros(N)

        # --- latest-anchor term ---
        if self.mode in ("latest", "hybrid"):
            ref = normed[-1]  # latest prediction
            latest_sim = normed @ ref  # (N,)
        else:
            latest_sim = np.zeros(N)

        # --- combine ---
        if self.mode == "cogact":
            weights = consensus
        elif self.mode == "latest":
            weights = latest_sim
        else:  # hybrid
            weights = self.lambda_mix * consensus + (1 - self.lambda_mix) * latest_sim

        weights = weights.clip(min=0)

        w_sum = weights.sum()
        if w_sum < 1e-8:
            weights = np.ones(N) / N
        else:
            weights /= w_sum

        return np.average(mat, axis=0, weights=weights)

    def get_overlap_count(self, current_step: int) -> int:
        with self._lock:
            buf = list(self._buffer)
        return sum(1 for qs, chunk in buf if 0 <= current_step - qs < len(chunk))

    def reset(self) -> None:
        with self._lock:
            self._buffer.clear()


@register_ensemble("cogact")
def _build_cogact(cfg: EnsembleConfig) -> CogACTEnsemble:
    return CogACTEnsemble(max_buffer_size=cfg.max_buffer_size, mode=cfg.cogact_mode)
