"""
Client-side action ensembling strategies for temporal smoothing.

Two strategies are provided:

  ExponentialEnsemble  — ACT-style: overlapping chunk predictions for a given
                         timestep are blended with exponential-decay weights,
                         giving the oldest prediction the highest weight.

  CogACTEnsemble       — AAE-style (CogACT, arXiv 2411.19650): overlapping
                         predictions are weighted by their pairwise cosine
                         similarity — predictions that "agree" with the rest
                         receive higher weight, suppressing outliers.

Both share the same interface:

    ensemble.add_chunk(query_step, chunk)   # call after each policy inference
    ensemble.get_action(current_step)       # call to obtain the blended action
    ensemble.reset()                        # call at the start of each episode
"""

from __future__ import annotations

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class ActionEnsemble(ABC):
    @abstractmethod
    def add_chunk(self, query_step: int, chunk: np.ndarray) -> None:
        """Register a new predicted action chunk.

        Args:
            query_step: The episode step at which this chunk was queried.
            chunk:      Array of shape (T, D) — T predicted actions of dim D.
        """

    @abstractmethod
    def get_action(self, current_step: int) -> np.ndarray | None:
        """Return the blended action for *current_step*.

        Returns None if no predictions are available for that step.
        """

    @abstractmethod
    def get_latest_raw(self, current_step: int) -> np.ndarray | None:
        """Return the most recent chunk's direct prediction for *current_step*."""

    @abstractmethod
    def get_overlap_count(self, current_step: int) -> int:
        """Return how many overlapping predictions are available for *current_step*."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state. Call at the beginning of each episode."""


# ---------------------------------------------------------------------------
# Exponential-decay ensemble  (ACT / original behaviour)
# ---------------------------------------------------------------------------


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

    def get_latest_raw(self, current_step: int) -> np.ndarray | None:
        candidates = self._buffer.get(current_step)
        return candidates[-1].copy() if candidates else None

    def get_overlap_count(self, current_step: int) -> int:
        return len(self._buffer.get(current_step, []))

    def reset(self) -> None:
        self._buffer.clear()


# ---------------------------------------------------------------------------
# CogACT / AAE ensemble  (cosine-similarity weighting)
# ---------------------------------------------------------------------------


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

    def get_latest_raw(self, current_step: int) -> np.ndarray | None:
        for qs, chunk in reversed(self._buffer):
            offset = current_step - qs
            if 0 <= offset < len(chunk):
                return chunk[offset].copy()
        return None

    def get_overlap_count(self, current_step: int) -> int:
        return sum(1 for qs, chunk in self._buffer if 0 <= current_step - qs < len(chunk))

    def reset(self) -> None:
        self._buffer.clear()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_ensemble(
    ensemble_type: str,
    *,
    decay: float = 1.0,
    max_buffer_size: int = 25,
) -> ActionEnsemble | None:
    """Return an ensemble instance or None.

    Args:
        ensemble_type:   ``"exp"`` | ``"cogact"`` | ``"none"``.
        decay:           Decay rate for ExponentialEnsemble.
        max_buffer_size: Buffer depth for CogACTEnsemble.
    """
    if ensemble_type == "none":
        return None
    if ensemble_type == "exp":
        return ExponentialEnsemble(decay=decay)
    if ensemble_type == "cogact":
        return CogACTEnsemble(max_buffer_size=max_buffer_size, mode="latest")
    raise ValueError(f"Unknown ensemble_type {ensemble_type!r}. Choose from: exp, cogact, none")


# ---------------------------------------------------------------------------
# Action logger
# ---------------------------------------------------------------------------


class ActionLogger:
    """Records the number of overlapping predictions used at each episode step
    and saves a JSON file at episode end (or on manual stop / Ctrl+C).

    Call ``save()`` inside a ``finally`` block so it fires on normal
    completion, timeout, KeyboardInterrupt, and 'r'-restart alike.

    Usage::

        logger = ActionLogger(save_dir="./action_logs")
        logger.reset()                             # start of episode
        logger.log(step, ensemble.get_overlap_count(step))  # each step
        logger.save(tag="episode_0")               # end / interrupt

    Output JSON structure::

        {
            "tag": "episode_243steps",
            "total_steps": 243,
            "steps": [
                {"step": 0, "n_overlaps": 1},
                {"step": 1, "n_overlaps": 3},
                ...
            ]
        }
    """

    def __init__(self, save_dir: str = "./action_logs") -> None:
        self._save_dir = Path(save_dir)
        self._records: list[dict] = []

    def reset(self) -> None:
        self._records.clear()

    def log(self, step: int, n_overlaps: int) -> None:
        self._records.append({"step": step, "n_overlaps": n_overlaps})

    def save(self, tag: str = "episode") -> None:
        if not self._records:
            logger.info("ActionLogger: nothing to save.")
            return

        self._save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        path = self._save_dir / f"{tag}_{timestamp}.json"

        payload = {
            "tag": tag,
            "total_steps": len(self._records),
            "steps": self._records,
        }
        path.write_text(json.dumps(payload, indent=2))
        logger.info("ActionLogger: saved to %s", path)


# ---------------------------------------------------------------------------
# Async inference worker
# ---------------------------------------------------------------------------


class AsyncPolicyWorker:
    """Runs policy inference in a background thread so the control loop never
    blocks on network / GPU latency.

    The worker always processes the *latest* submitted observation — if the
    control loop submits a new one before the previous inference finishes, the
    stale observation is silently dropped (latest-wins semantics).  Each
    completed chunk is added directly to the ensemble.

    This pairs naturally with CogACTEnsemble (query every step, blend by
    agreement) but works with any ActionEnsemble.

    Usage::

        worker = AsyncPolicyWorker(policy_client, ensemble, action_dim)
        worker.start()

        # inside control loop (non-blocking):
        worker.submit(obs, episode_step)

        # block once at the very start of an episode:
        worker.wait_for_first()

        # get blended action (always non-blocking):
        a_t = ensemble.get_action(step)

        worker.stop()
    """

    def __init__(self, policy_client, ensemble: ActionEnsemble, action_dim: int) -> None:
        self._client = policy_client
        self._ensemble = ensemble
        self._action_dim = action_dim
        self._pending: tuple | None = None  # (obs_dict, query_step)
        self._lock = threading.Lock()
        self._first_result = threading.Event()
        self._need_first = True
        self._generation = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._first_result.clear()
        self._need_first = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="policy-worker")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def submit(self, obs: dict, query_step: int) -> None:
        """Submit a fresh observation. Non-blocking. Overwrites any pending stale obs."""
        with self._lock:
            self._pending = (obs, query_step)

    def flush(self) -> None:
        """Drop the pending observation and discard any in-flight inference.

        Call this whenever the control loop stops trusting predictions already
        in flight — the operator took the arm over with a scripted motion, or
        the task instruction changed. Without it, a chunk requested under the
        old instruction/pose lands in the ensemble seconds later and is blended
        into the new behaviour. ``wait_for_first()`` is re-armed, so the caller
        can block for a genuinely fresh chunk before moving again.
        """
        with self._lock:
            self._pending = None
            self._generation += 1
            self._need_first = True
        self._first_result.clear()

    def wait_for_first(self, timeout: float = 30.0) -> bool:
        """Block until the first inference chunk has been added to the ensemble."""
        return self._first_result.wait(timeout=timeout)

    def _loop(self) -> None:
        while self._running:
            with self._lock:
                item = self._pending
                generation = self._generation
                if item is not None:
                    self._pending = None  # consume
            if item is None:
                time.sleep(0.001)
                continue
            obs, query_step = item
            try:
                response = self._client.infer(obs)
                chunk = np.asarray(response["actions"])[:, : self._action_dim]
                with self._lock:
                    if generation != self._generation:
                        continue  # flushed while this inference was running — drop it
                    self._ensemble.add_chunk(query_step, chunk)
                    first = self._need_first
                    self._need_first = False
                if first:
                    self._first_result.set()
            except Exception:
                logger.exception("AsyncPolicyWorker: inference error")
