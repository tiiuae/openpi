"""Background inference worker.

Runs policy inference off the control-loop thread so the loop never blocks on
network / GPU latency. Depends only on the :class:`TemporalEnsemble`.
"""

from __future__ import annotations

import logging
import threading
import time

from ensemble import TemporalEnsemble
import numpy as np
from webapp.telemetry import NullSink
from webapp.telemetry import TelemetrySink

logger = logging.getLogger(__name__)


class AsyncPolicyWorker:
    """Runs policy inference in a background thread so the control loop never
    blocks on network / GPU latency.

    The worker always processes the *latest* submitted observation — if the
    control loop submits a new one before the previous inference finishes, the
    stale observation is silently dropped (latest-wins semantics).  Each
    completed chunk is added directly to the ensemble.

    The control loop submits a fresh observation every step and the worker adds
    each completed chunk to the temporal ensemble, which blends overlaps.

    Usage::

        worker = AsyncPolicyWorker(policy_client, ensemble, adapter)
        worker.start()

        # inside control loop (non-blocking):
        worker.submit(obs, episode_step, joints14)

        # block once at the very start of an episode:
        worker.wait_for_first()

        # get blended action (always non-blocking):
        a_t = ensemble.get_action(step)

        worker.stop()
    """

    def __init__(
        self,
        policy_client,
        ensemble: TemporalEnsemble,
        adapter,
        sink: TelemetrySink = NullSink(),  # noqa
    ) -> None:
        self._client = policy_client
        self._ensemble = ensemble
        self._adapter = adapter
        self._sink = sink
        self._pending: tuple | None = None  # (obs_dict, query_step, joints14)
        self._lock = threading.Lock()
        self._first_result = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._first_result.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="policy-worker")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def submit(self, obs: dict, query_step: int, joints14: np.ndarray) -> None:
        """Submit a fresh observation. Non-blocking. Overwrites any pending stale obs."""
        with self._lock:
            self._pending = (obs, query_step, joints14)

    def wait_for_first(self, timeout: float = 30.0) -> bool:
        """Block until the first inference chunk has been added to the ensemble."""
        return self._first_result.wait(timeout=timeout)

    def _loop(self) -> None:
        first = True
        while self._running:
            with self._lock:
                item = self._pending
                if item is not None:
                    self._pending = None  # consume
            if item is None:
                time.sleep(0.001)
                continue
            obs, query_step, joints14 = item
            try:
                t0 = time.perf_counter()
                response = self._client.infer(obs)
                rtt_ms = (time.perf_counter() - t0) * 1e3
                self._sink.on_inference(rtt_ms, time.time())
                chunk = self._adapter.decode_chunk(response["actions"], joints14)
                self._ensemble.add_chunk(query_step, chunk)
                self._sink.on_chunk(query_step, chunk, time.time())
                ee = self._adapter.ee_chunk(response["actions"])
                if ee is not None:
                    self._sink.on_ee_chunk(query_step, ee, time.time())
                if first:
                    self._first_result.set()
                    first = False
            except Exception:
                logger.exception("AsyncPolicyWorker: inference error")
