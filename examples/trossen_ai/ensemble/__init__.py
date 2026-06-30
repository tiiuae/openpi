"""Temporal action ensemble — one strategy, no registry.

A single policy inference returns a *chunk* of T future actions. As the control
loop re-queries (every ``rate_of_inference`` steps, or every step in async mode),
several chunks predict the same future timestep. :class:`TemporalEnsemble`
averages those overlapping predictions with exponential-decay weights — the
ACT-style temporal-ensembling that smooths jittery policy output.

Design notes (this is a deliberate rewrite of the old multi-strategy package):
- ONE class, ONE knob (``decay``). No registry, no cogact/latest/hybrid modes.
- Chunks are kept whole until fully consumed, so an overlap is never evicted
  before the step that needs it — the bug the old exact-step-pop buffer had.
- Thread-safe: the async worker adds chunks from a background thread while the
  control loop reads actions.

Usage::

    ens = make_ensemble(smoothing=True, decay=1.0)   # or None when smoothing off
    ens.add_chunk(query_step, chunk)   # after each inference; chunk is (T, D)
    a_t = ens.get_action(current_step) # blended action, or None if no overlap
    ens.reset()                        # at the start of each episode
"""
from __future__ import annotations

import threading

import numpy as np

__all__ = ["TemporalEnsemble", "make_ensemble"]


class TemporalEnsemble:
    """Blend overlapping chunk predictions with exponential-decay weights.

    For a queried timestep the ensemble holds every recent chunk that predicts
    it. Predictions are ordered oldest-query-first and weighted ``exp(-decay*k)``
    (k=0 = oldest), matching the original ACT temporal-ensemble formulation:
    larger ``decay`` trusts older, already-committed predictions more (smoother,
    less reactive); ``decay=0`` is a plain average.

    Args:
        decay:      Exponential decay rate (>=0). Default 1.0.
        max_chunks: Hard cap on retained chunks (memory bound). Fully-consumed
                    chunks are evicted eagerly; this only matters if inference
                    outruns consumption.
    """

    def __init__(self, decay: float = 1.0, max_chunks: int = 64) -> None:
        self.decay = float(decay)
        self.max_chunks = int(max_chunks)
        self._chunks: list[tuple[int, np.ndarray]] = []  # (query_step, (T, D))
        self._lock = threading.Lock()
        self._last_weights: np.ndarray | None = None

    def add_chunk(self, query_step: int, chunk: np.ndarray) -> None:
        """Register a predicted chunk queried at *query_step* (shape (T, D))."""
        arr = np.asarray(chunk, dtype=float)
        with self._lock:
            self._chunks.append((int(query_step), arr))
            if len(self._chunks) > self.max_chunks:
                self._chunks.pop(0)

    def get_action(self, current_step: int) -> np.ndarray | None:
        """Return the blended action for *current_step*, or None if no overlap."""
        with self._lock:
            # Evict chunks that can no longer predict the current (or any future)
            # step, then gather the still-overlapping predictions.
            self._chunks = [(qs, c) for qs, c in self._chunks
                            if qs + len(c) > current_step]
            overlaps = [(qs, c[current_step - qs]) for qs, c in self._chunks
                        if 0 <= current_step - qs < len(c)]
        if not overlaps:
            return None
        overlaps.sort(key=lambda t: t[0])  # oldest query first
        mat = np.array([a for _, a in overlaps], dtype=float)  # (N, D)
        weights = np.exp(-self.decay * np.arange(len(mat)))
        weights /= weights.sum()
        self._last_weights = weights
        return np.average(mat, axis=0, weights=weights)

    def get_overlap_count(self, current_step: int) -> int:
        """How many buffered chunks predict *current_step* (telemetry)."""
        with self._lock:
            return sum(1 for qs, c in self._chunks
                       if 0 <= current_step - qs < len(c))

    def reset(self) -> None:
        """Clear all buffered chunks. Call at the start of each episode."""
        with self._lock:
            self._chunks.clear()
            self._last_weights = None

    def buffer_size(self) -> int:
        """Number of retained chunks (telemetry / memory-bound check)."""
        with self._lock:
            return len(self._chunks)

    def last_weights(self) -> np.ndarray | None:
        """Blend weights from the most recent ``get_action`` (telemetry only)."""
        return self._last_weights


def make_ensemble(smoothing: bool = True, *, decay: float = 1.0) -> TemporalEnsemble | None:
    """Return a :class:`TemporalEnsemble`, or ``None`` when smoothing is off.

    ``None`` means "no smoothing" — the control loop uses the latest chunk's
    action directly. Async inference requires smoothing (a non-None ensemble).
    """
    return TemporalEnsemble(decay=decay) if smoothing else None
