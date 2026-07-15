"""Temporal action ensembles — two weighting strategies, no registry.

A single policy inference returns a *chunk* of T future actions. As the control
loop re-queries (every ``rate_of_inference`` steps, or every step in async mode),
several chunks predict the same future timestep. An ensemble averages those
overlapping predictions into one commanded action; the two differ only in how
the per-prediction *weights* are computed:

- :class:`TemporalEnsemble` — ACT-style exponential-decay-by-age weights
  (``exp(-decay*k)``). Older, already-committed predictions dominate; smooths
  jittery output. One knob: ``decay``.
- :class:`CogACTEnsemble` — CogACT / AAE cosine-similarity *consensus* weights
  (arXiv 2411.19650). Each prediction is weighted by how much it agrees with the
  others, so outliers are suppressed. Modes ``cogact`` | ``latest`` | ``hybrid``.

Design notes (shared buffer semantics for both classes):
- Chunks are kept whole until fully consumed, so an overlap is never evicted
  before the step that needs it — the bug the old exact-step-pop buffer had.
- Thread-safe: the async worker adds chunks from a background thread while the
  control loop reads actions.

Usage::

    ens = make_ensemble(smoothing=True, decay=1.0)                 # temporal
    ens = make_ensemble(smoothing=True, method="cogact")           # cogact
    ens.add_chunk(query_step, chunk)   # after each inference; chunk is (T, D)
    a_t = ens.get_action(current_step) # blended action, or None if no overlap
    ens.reset()                        # at the start of each episode
"""
from __future__ import annotations

import threading

import numpy as np

__all__ = ["TemporalEnsemble", "CogACTEnsemble", "make_ensemble"]


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


class CogACTEnsemble:
    """Blend overlapping chunk predictions with CogACT / AAE cosine-similarity
    consensus weighting (arXiv 2411.19650).

    Where :class:`TemporalEnsemble` weights each overlapping prediction by its
    age, CogACT weights it by how much it *agrees* with the other overlapping
    predictions — the mean pairwise cosine similarity — so a prediction that
    disagrees with the consensus is down-weighted (outlier suppression):

    - ``mode="cogact"``  weight_i = mean_j cos(a_i, a_j)      (pure consensus)
    - ``mode="latest"``  weight_i = cos(a_i, a_newest)        (anchor to newest)
    - ``mode="hybrid"``  lambda_mix*consensus + (1-lambda_mix)*latest

    Buffer semantics are identical to :class:`TemporalEnsemble` (whole chunks
    retained until consumed, thread-safe), so it is a drop-in alternative.

    Args:
        mode:       "cogact" | "latest" | "hybrid".
        lambda_mix: Consensus/latest mix for "hybrid" (0..1). Ignored otherwise.
        max_chunks: Hard cap on retained chunks (memory bound).
    """

    def __init__(self, mode: str = "cogact", lambda_mix: float = 0.5,
                 max_chunks: int = 64) -> None:
        self.mode = str(mode)
        self.lambda_mix = float(lambda_mix)
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
        """Return the consensus-blended action for *current_step*, or None."""
        with self._lock:
            self._chunks = [(qs, c) for qs, c in self._chunks
                            if qs + len(c) > current_step]
            overlaps = [(qs, c[current_step - qs]) for qs, c in self._chunks
                        if 0 <= current_step - qs < len(c)]
        if not overlaps:
            return None
        overlaps.sort(key=lambda t: t[0])           # oldest query first, newest last
        mat = np.array([a for _, a in overlaps], dtype=float)  # (N, D)
        if len(mat) == 1:
            self._last_weights = np.array([1.0])
            return mat[0].copy()

        # Cosine-similarity weighting on unit-normalized predictions.
        normed = mat / np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-8)
        n = len(mat)
        consensus = ((normed @ normed.T).mean(axis=1)
                     if self.mode in ("cogact", "hybrid") else np.zeros(n))
        latest = (normed @ normed[-1]
                  if self.mode in ("latest", "hybrid") else np.zeros(n))
        if self.mode == "cogact":
            weights = consensus
        elif self.mode == "latest":
            weights = latest
        else:  # hybrid
            weights = self.lambda_mix * consensus + (1 - self.lambda_mix) * latest

        weights = weights.clip(min=0)
        total = weights.sum()
        weights = np.ones(n) / n if total < 1e-8 else weights / total
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


def make_ensemble(smoothing: bool = True, *, decay: float = 1.0,
                  method: str = "temporal", cogact_mode: str = "cogact",
                  lambda_mix: float = 0.5):
    """Return an ensemble instance, or ``None`` when smoothing is off.

    ``None`` means "no smoothing" — the control loop uses the latest chunk's
    action directly. Async inference requires smoothing (a non-None ensemble).

    Args:
        smoothing:   Master on/off. ``False`` returns ``None``.
        method:      "temporal" (exp-decay, default) or "cogact" (consensus).
        decay:       Exp-decay rate for the temporal method.
        cogact_mode: "cogact" | "latest" | "hybrid" for the cogact method.
        lambda_mix:  Consensus/latest mix for cogact "hybrid" mode.
    """
    if not smoothing:
        return None
    if method == "temporal":
        return TemporalEnsemble(decay=decay)
    if method == "cogact":
        return CogACTEnsemble(mode=cogact_mode, lambda_mix=lambda_mix)
    raise ValueError(f"Unknown smoothing method {method!r}; expected 'temporal' or 'cogact'.")
