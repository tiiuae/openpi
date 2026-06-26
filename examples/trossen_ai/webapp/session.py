"""Run exactly one control/replay session in a daemon thread.

The runner factory is injected so the hardware-bound runners (bridge / replay)
can be swapped for fakes in tests. A Runner must expose run() (blocking),
stop(), and estop().
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Protocol

logger = logging.getLogger(__name__)


class Runner(Protocol):
    def run(self) -> None: ...
    def stop(self) -> None: ...
    def estop(self) -> None: ...


RunnerFactory = Callable[[str, dict, object], Runner]


class SessionManager:
    def __init__(self, runner_factory: RunnerFactory) -> None:
        self._factory = runner_factory
        self._runner: Runner | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop_requested = False

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, kind: str, config: dict, sink: object) -> None:
        with self._lock:
            if self.is_running():
                raise RuntimeError("A session is already running")
            self._stop_requested = False
            self._runner = None
            self._thread = threading.Thread(
                target=self._run, args=(kind, config, sink),
                daemon=True, name=f"session-{kind}",
            )
            self._thread.start()

    def _run(self, kind: str, config: dict, sink) -> None:
        # Forward all Python logging to the browser for the whole session so the
        # UI sees arm connections, errors, and lifecycle — not just the inner
        # episode loop (which is where the handler used to be attached).
        from webapp.log_bridge import SinkLogHandler

        handler = SinkLogHandler(sink)
        handler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        prev_level = root.level
        root.addHandler(handler)
        if prev_level == logging.NOTSET or prev_level > logging.INFO:
            root.setLevel(logging.INFO)
        logger.info("Session starting: kind=%s", kind)
        try:
            runner = self._factory(kind, config, sink)
            with self._lock:
                self._runner = runner
            if self._stop_requested:
                logger.info("Session %s: stop requested before run; cancelling", kind)
                runner.stop()
                return
            runner.run()
            logger.info("Session %s: finished", kind)
        except Exception as exc:  # noqa: BLE001 — surface to UI, never crash the thread
            logger.exception("Session %s crashed: %s", kind, exc)
            try:
                sink.on_status("error", {"message": str(exc)})
            except Exception:
                pass
        finally:
            root.removeHandler(handler)
            root.setLevel(prev_level)

    def stop(self, timeout: float = 15.0) -> None:
        self._stop_requested = True
        with self._lock:
            runner, thread = self._runner, self._thread
        if runner is not None:
            runner.stop()
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(
                    "Session thread did not stop within %ss; it may still be running",
                    timeout,
                )
        with self._lock:
            self._runner, self._thread = None, None

    def estop(self, timeout: float = 15.0) -> None:
        self._stop_requested = True
        with self._lock:
            runner, thread = self._runner, self._thread
        if runner is not None:
            runner.estop()
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(
                    "Session thread did not stop within %ss; it may still be running",
                    timeout,
                )
        with self._lock:
            self._runner, self._thread = None, None
