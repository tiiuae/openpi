"""Exponential-decay ensemble (ACT / original behaviour)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .base import ActionEnsemble
from .config import EnsembleConfig
from .factory import register_ensemble


class ExponentialEnsemble(ActionEnsemble):
    """Blend overlapping chunk predictions with exponential-decay weights.

    For a given timestep the buffer holds one entry per past inference call
    that predicted that step.  Entries are appended in arrival order (oldest
    first), and weights follow ``exp(-decay * k)`` where k=0 is the oldest
    prediction.  This matches the original ACT paper formulation.

    Args:
        decay: Exponential decay rate (>0).  Higher = steeper fall-off
               toward newer predictions.  Default 1.0 matches the original
               ``temporal_ensemble_coefficient = True`` behaviour.
    """

    def __init__(self, decay: float = 1.0) -> None:
        self.decay = decay
        self._buffer: dict[int, list[np.ndarray]] = defaultdict(list)

    def add_chunk(self, query_step: int, chunk: np.ndarray) -> None:
        for k, action in enumerate(chunk):
            self._buffer[query_step + k].append(action)

    def get_action(self, current_step: int) -> np.ndarray | None:
        candidates = self._buffer.get(current_step)
        if not candidates:
            return None
        mat = np.array(candidates)  # (N, D)
        weights = np.exp(-self.decay * np.arange(len(mat)))
        weights /= weights.sum()
        return np.average(mat, axis=0, weights=weights)

    def get_overlap_count(self, current_step: int) -> int:
        return len(self._buffer.get(current_step, []))

    def reset(self) -> None:
        self._buffer.clear()


@register_ensemble("exp")
def _build_exp(cfg: EnsembleConfig) -> ExponentialEnsemble:
    return ExponentialEnsemble(decay=cfg.decay)
