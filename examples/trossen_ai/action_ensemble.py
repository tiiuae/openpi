"""
Client-side action ensembling strategies for temporal smoothing.

Three strategies are provided:

  ExponentialEnsemble  — ACT-style: overlapping chunk predictions for a given
                         timestep are blended with exponential-decay weights,
                         giving the oldest prediction the highest weight.

  CogACTEnsemble       — AAE-style (CogACT, arXiv 2411.19650): overlapping
                         predictions are weighted by their pairwise cosine
                         similarity — predictions that "agree" with the rest
                         receive higher weight, suppressing outliers.

  LatestChunkEnsemble  — RTC execution: a newly returned chunk atomically
                         replaces the previous one and is indexed by the
                         control step at which its observation was captured.

All share the same interface:

    ensemble.add_chunk(query_step, chunk)   # call after each policy inference
    ensemble.get_action(current_step)       # call to obtain the blended action
    ensemble.reset()                        # call at the start of each episode
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from collections import deque
from datetime import UTC
from datetime import datetime
import json
import logging
from numbers import Integral
from pathlib import Path
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

RTC_PROTOCOL_VERSION = 2


def validate_rtc_server_metadata(
    metadata: dict,
    *,
    rate_of_inference: int | None = None,
    async_inference: bool = False,
) -> bool:
    """Return whether RTC is enabled, rejecting the unsafe legacy protocol.

    The first RTC deployment advertised ``rtc_enabled`` but did not carry
    query-step, delay, or reset metadata. Running an RTC client against that
    server silently aligns chunks to stale actions, so protocol v2 is required
    before any robot hardware is connected.
    """
    if not bool(metadata.get("rtc_enabled", False)):
        return False

    version = metadata.get("rtc_protocol_version")
    if version != RTC_PROTOCOL_VERSION:
        raise RuntimeError(
            "RTC server protocol is unsafe or unsupported: "
            f"expected rtc_protocol_version={RTC_PROTOCOL_VERSION}, got {version!r}. "
            "Update/restart the deployment server before connecting the robot."
        )

    action_horizon = metadata.get("action_horizon")
    if (
        isinstance(action_horizon, bool)
        or not isinstance(action_horizon, Integral)
        or action_horizon <= 0
    ):
        raise RuntimeError(
            "RTC protocol v2 metadata must include a positive integer action_horizon; "
            f"got {action_horizon!r}."
        )
    if rate_of_inference is not None:
        if (
            isinstance(rate_of_inference, bool)
            or not isinstance(rate_of_inference, Integral)
            or rate_of_inference <= 0
        ):
            raise ValueError(f"rate_of_inference must be a positive integer, got {rate_of_inference!r}")
        if not async_inference and rate_of_inference >= action_horizon:
            raise ValueError(
                "Synchronous RTC requires rate_of_inference < server action_horizon "
                f"({action_horizon}); got {rate_of_inference}."
            )
    return True


def validate_rtc_response_query(response: dict, expected_query_step: int) -> None:
    """Reject a response that cannot be associated with its RTC request."""
    response_query_step = response.get("rtc_query_step")
    if response_query_step is None:
        raise RuntimeError("RTC protocol v2 response is missing rtc_query_step")
    if isinstance(response_query_step, bool) or not isinstance(response_query_step, Integral):
        raise RuntimeError(
            "RTC protocol v2 response has an invalid rtc_query_step: "
            f"expected an integer, got {response_query_step!r}"
        )
    if int(response_query_step) != int(expected_query_step):
        raise RuntimeError(
            "RTC response/query mismatch: "
            f"requested step {expected_query_step}, received {response_query_step!r}"
        )


def measured_pose_hold(observation: dict, action_dim: int) -> np.ndarray:
    """Return a finite copy of measured joint state for a safe action hold."""
    state = np.asarray(observation["state"])
    if state.shape != (action_dim,):
        raise ValueError(f"Measured state must have shape ({action_dim},), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("Measured state contains NaN or infinite values; refusing to command it")
    return state.copy()


def validated_action_chunk(actions, action_dim: int | None = None) -> np.ndarray:
    """Return a finite 2-D action chunk, optionally truncated to ``action_dim``."""
    chunk = np.asarray(actions)
    if chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] == 0:
        raise ValueError(f"Policy actions must have non-empty shape (horizon, action_dim), got {chunk.shape}")
    if action_dim is not None and chunk.shape[1] < action_dim:
        raise ValueError(
            f"Policy actions need at least {action_dim} dimensions, got shape {chunk.shape}"
        )
    if not np.issubdtype(chunk.dtype, np.number) or np.issubdtype(chunk.dtype, np.complexfloating):
        raise ValueError(f"Policy actions must be real numeric values, got dtype {chunk.dtype}")
    try:
        finite = np.isfinite(chunk).all()
    except TypeError as exc:
        raise ValueError(f"Policy actions must be numeric, got dtype {chunk.dtype}") from exc
    if not finite:
        raise ValueError("Policy actions contain NaN or infinite values")
    if action_dim is not None:
        chunk = chunk[:, :action_dim]
    return chunk.copy()


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
# Atomic latest-chunk queue (RTC)
# ---------------------------------------------------------------------------


class LatestChunkEnsemble(ActionEnsemble):
    """Execute exactly one time-aligned chunk, replacing it atomically.

    RTC conditions the new model chunk on the previously committed chunk.
    Blending that result again with older chunks (for example with CogACT)
    produces a command stream different from the one RTC conditioned on. This
    queue instead commits the newest response as a unit and indexes it using
    ``current_step - query_step`` so inference latency is skipped, not replayed.
    """

    def __init__(self) -> None:
        self._latest: tuple[int, np.ndarray] | None = None
        self._lock = threading.Lock()

    def add_chunk(self, query_step: int, chunk: np.ndarray) -> None:
        chunk = validated_action_chunk(chunk)
        query_step = int(query_step)
        with self._lock:
            if self._latest is not None and query_step < self._latest[0]:
                logger.warning(
                    "Ignoring out-of-order action chunk for query step %d; latest is %d",
                    query_step,
                    self._latest[0],
                )
                return
            self._latest = (query_step, chunk)

    def _get_action(self, current_step: int) -> np.ndarray | None:
        with self._lock:
            if self._latest is None:
                return None
            query_step, chunk = self._latest
            offset = int(current_step) - query_step
            if offset < 0:
                return None
            if offset >= len(chunk):
                # Do not leave an expired chunk looking live to diagnostics or
                # later callers. The control loop will hold measured position.
                self._latest = None
                return None
            return chunk[offset].copy()

    def get_action(self, current_step: int) -> np.ndarray | None:
        return self._get_action(current_step)

    def get_latest_raw(self, current_step: int) -> np.ndarray | None:
        return self._get_action(current_step)

    def get_overlap_count(self, current_step: int) -> int:
        with self._lock:
            if self._latest is None:
                return 0
            query_step, chunk = self._latest
            offset = int(current_step) - query_step
            if offset >= len(chunk):
                self._latest = None
                return 0
            return int(offset >= 0)

    def reset(self) -> None:
        with self._lock:
            self._latest = None


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
# Async inference workers
# ---------------------------------------------------------------------------


class AsyncPolicyWorker:
    """Original async worker used by every non-RTC checkpoint.

    Keep this path deliberately unchanged. RTC has different queueing and wire
    semantics and lives in :class:`RTCAsyncPolicyWorker` below.
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
        """Drop the pending observation and discard any in-flight inference."""
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
                    self._pending = None
            if item is None:
                time.sleep(0.001)
                continue
            obs, query_step = item
            try:
                response = self._client.infer(obs)
                chunk = np.asarray(response["actions"])[:, : self._action_dim]
                with self._lock:
                    if generation != self._generation:
                        continue
                    self._ensemble.add_chunk(query_step, chunk)
                    first = self._need_first
                    self._need_first = False
                if first:
                    self._first_result.set()
            except Exception:
                logger.exception("AsyncPolicyWorker: inference error")


class _RoundTripDelayEstimator:
    """Conservative request-to-response latency forecast in control steps."""

    def __init__(self, control_frequency: float, window_size: int = 10) -> None:
        if control_frequency <= 0:
            raise ValueError("control_frequency must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self._control_frequency = float(control_frequency)
        self._samples: deque[float] = deque(maxlen=window_size)

    def observe(self, round_trip_s: float) -> None:
        if np.isfinite(round_trip_s) and round_trip_s >= 0:
            self._samples.append(float(round_trip_s))

    def steps(self, pending_age_s: float = 0.0) -> int:
        if not np.isfinite(pending_age_s) or pending_age_s < 0:
            raise ValueError("pending_age_s must be finite and non-negative")
        if not self._samples:
            return 0
        # The window mean tracks the typical round trip. Using max() instead
        # pins the forecast to the single slowest recent sample for the whole
        # window: when latency is bimodal (e.g. alternating fast/slow
        # inference cycles), that keeps the estimate near the slow mode even
        # on fast calls, which needlessly exhausts the RTC overlap horizon
        # and disables guidance (rtc_skip_reason=inference_delay_exhausts_
        # overlap) on every fast call. Include time the captured observation
        # waited behind the preceding request; its query-step clock starts
        # before websocket infer() does.
        mean_round_trip_s = sum(self._samples) / len(self._samples)
        return int(np.ceil((mean_round_trip_s + pending_age_s) * self._control_frequency))


class RTCAsyncPolicyWorker:
    """Runs policy inference in a background thread so the control loop never
    blocks on network / GPU latency.

    The worker always processes the *latest* submitted observation — if the
    control loop submits a new one before the previous inference finishes, the
    stale observation is silently dropped (latest-wins semantics).  Each
    completed chunk is added directly to the ensemble.

    With RTC, use :class:`LatestChunkEnsemble`: each request is tagged with the
    control step at which its observation was captured, and the returned chunk
    atomically replaces the old one at that same query step. The worker also
    forecasts request latency and propagates reset epochs to the server.

    Usage::

        worker = RTCAsyncPolicyWorker(policy_client, ensemble, action_dim)
        worker.start()

        # inside control loop (non-blocking):
        worker.submit(obs, episode_step)

        # block once at the very start of an episode:
        worker.wait_for_first()

        # get blended action (always non-blocking):
        a_t = ensemble.get_action(step)

        worker.stop()
    """

    def __init__(
        self,
        policy_client,
        ensemble: ActionEnsemble,
        action_dim: int,
        *,
        control_frequency: float = 30.0,
    ) -> None:
        self._client = policy_client
        self._ensemble = ensemble
        self._action_dim = action_dim
        self._rtc_enabled = True
        self._delay_estimator = _RoundTripDelayEstimator(control_frequency)
        self._pending: tuple | None = None  # (obs_dict, query_step, submitted_at)
        self._lock = threading.Lock()
        self._first_result = threading.Event()
        self._need_first = True
        self._generation = 0
        self._last_submitted_query_step: int | None = None
        self._remote_reset_pending = True
        self._restart_required = False
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                if self._thread.is_alive():
                    raise RuntimeError("Previous AsyncPolicyWorker thread is still running")
                self._thread = None
            if self._running:
                raise RuntimeError("AsyncPolicyWorker is already running")
            self._pending = None
            self._generation += 1
            self._last_submitted_query_step = None
            self._remote_reset_pending = self._rtc_enabled
            self._restart_required = False
            self._need_first = True
            self._running = True
            self._ensemble.reset()
            self._first_result.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="policy-worker")
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            self._running = False
            thread = self._thread
        if thread is None:
            return
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.error(
                "AsyncPolicyWorker did not stop within %.1fs; refusing to start another worker on this websocket",
                timeout,
            )
            return
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def submit(self, obs: dict, query_step: int) -> None:
        """Submit a fresh observation. Non-blocking. Overwrites any pending stale obs."""
        query_step = int(query_step)
        with self._lock:
            if self._last_submitted_query_step is not None and query_step < self._last_submitted_query_step:
                raise ValueError(
                    "query_step must be monotonic within an RTC/reset epoch: "
                    f"got {query_step} after {self._last_submitted_query_step}"
                )
            self._last_submitted_query_step = query_step
            # The control loop owns the original observation. Use a shallow
            # copy so adding wire metadata cannot mutate it under the caller.
            self._pending = (dict(obs), query_step, time.perf_counter())

    def flush(self) -> None:
        """Drop the pending observation and discard any in-flight inference.

        Call this whenever the control loop stops trusting predictions already
        in flight — the operator took the arm over with a scripted motion, or
        the task instruction changed. Without it, a chunk requested under the
        old instruction/pose can land seconds later and enter the new behaviour.
        ``wait_for_first()`` is re-armed, so the caller can block for a genuinely
        fresh chunk before moving again.
        """
        with self._lock:
            self._pending = None
            self._generation += 1
            self._need_first = True
            self._last_submitted_query_step = None
            self._remote_reset_pending = self._rtc_enabled
            self._restart_required = False
            self._ensemble.reset()
            self._first_result.clear()

    def consume_restart_required(self) -> bool:
        """Return and clear whether an inference failure requires a fresh ramp."""
        with self._lock:
            restart_required = self._restart_required
            self._restart_required = False
            return restart_required

    def wait_for_first(self, timeout: float = 30.0) -> bool:
        """Block until a first result arrives or a failed attempt requests a restart."""
        return self._first_result.wait(timeout=timeout)

    def _loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    break
                item = self._pending
                generation = self._generation
                send_reset = False
                if item is not None:
                    self._pending = None  # consume
                    if self._rtc_enabled and self._remote_reset_pending:
                        send_reset = True
                        self._remote_reset_pending = False
            if item is None:
                time.sleep(0.001)
                continue
            obs, query_step, submitted_at = item
            request_start = time.perf_counter()
            if self._rtc_enabled:
                obs["rtc_query_step"] = query_step
                obs["rtc_inference_delay"] = self._delay_estimator.steps(
                    pending_age_s=max(request_start - submitted_at, 0.0)
                )
                obs["rtc_reset"] = send_reset

            try:
                response = self._client.infer(obs)
                round_trip_s = time.perf_counter() - request_start
                self._delay_estimator.observe(round_trip_s)

                if self._rtc_enabled:
                    validate_rtc_response_query(response, query_step)

                chunk = validated_action_chunk(response["actions"], self._action_dim)
                with self._lock:
                    if generation != self._generation or not self._running:
                        continue  # flushed while this inference was running — drop it
                    self._ensemble.add_chunk(query_step, chunk)
                    first = self._need_first
                    self._need_first = False
                    if first:
                        self._first_result.set()
            except Exception:
                with self._lock:
                    if generation == self._generation and self._running:
                        # The request may have reached the server before failing.
                        # Invalidate its whole local epoch, then reset remotely on
                        # the next request so the committed streams converge.
                        self._pending = None
                        self._generation += 1
                        self._last_submitted_query_step = None
                        self._remote_reset_pending = self._rtc_enabled
                        self._restart_required = True
                        self._need_first = True
                        self._ensemble.reset()
                        # Wake a control loop already waiting for its first
                        # result. It will observe restart_required, flush this
                        # failed epoch, and submit a fresh reset request.
                        self._first_result.set()
                logger.exception("AsyncPolicyWorker: inference error")
