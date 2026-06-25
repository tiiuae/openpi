"""Exponential-decay ensemble (ACT / original behaviour)."""
from __future__ import annotations

import threading
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
        self._lock = threading.Lock()
        self._last_weights: np.ndarray | None = None

    def add_chunk(self, query_step: int, chunk: np.ndarray) -> None:
        with self._lock:
            for k, action in enumerate(chunk):
                self._buffer[query_step + k].append(action)

    def get_action(self, current_step: int) -> np.ndarray | None:
        with self._lock:
            candidates = self._buffer.pop(current_step, None)  # consume + evict
            # drop any stragglers strictly older than the step we just consumed
            for stale in [k for k in self._buffer if k < current_step]:
                del self._buffer[stale]
        if not candidates:
            return None
        mat = np.array(candidates)  # (N, D)
        weights = np.exp(-self.decay * np.arange(len(mat)))
        weights /= weights.sum()
        self._last_weights = weights
        return np.average(mat, axis=0, weights=weights)

    def get_overlap_count(self, current_step: int) -> int:
        with self._lock:
            return len(self._buffer.get(current_step, []))

    def reset(self) -> None:
        with self._lock:
            self._buffer.clear()

    def buffer_size(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._buffer.values())

    def last_weights(self) -> np.ndarray | None:
        return self._last_weights


@register_ensemble("exp")
def _build_exp(cfg: EnsembleConfig) -> ExponentialEnsemble:
    return ExponentialEnsemble(decay=cfg.decay)
