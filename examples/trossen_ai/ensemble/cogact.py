"""CogACT / AAE ensemble (cosine-similarity weighting).

AAE-style (CogACT, arXiv 2411.19650): overlapping predictions are weighted by
their pairwise cosine similarity — predictions that "agree" with the rest receive
higher weight, suppressing outliers.
"""
from __future__ import annotations

import numpy as np

from .base import ActionEnsemble
from .config import EnsembleConfig
from .factory import register_ensemble


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

    def add_chunk(self, query_step: int, chunk: np.ndarray) -> None:
        self._buffer.append((query_step, chunk))
        if len(self._buffer) > self.max_buffer_size:
            self._buffer.pop(0)

    def get_action(self, current_step: int) -> np.ndarray | None:
        candidates = [chunk[current_step - qs] for qs, chunk in self._buffer if 0 <= current_step - qs < len(chunk)]
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
        return sum(1 for qs, chunk in self._buffer if 0 <= current_step - qs < len(chunk))

    def reset(self) -> None:
        self._buffer.clear()


@register_ensemble("cogact")
def _build_cogact(cfg: EnsembleConfig) -> CogACTEnsemble:
    # FIXME(Task 4): honor cfg.cogact_mode instead of forcing "latest".
    return CogACTEnsemble(max_buffer_size=cfg.max_buffer_size, mode="latest")
