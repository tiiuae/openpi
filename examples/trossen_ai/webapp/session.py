"""Run exactly one control/replay session in a daemon thread.

The runner factory is injected so the hardware-bound runners (bridge / replay)
can be swapped for fakes in tests. A Runner must expose run() (blocking),
stop(), and estop().
"""
from __future__ import annotations

import threading
from typing import Callable, Protocol


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
        try:
            runner = self._factory(kind, config, sink)
            self._runner = runner
            if self._stop_requested:
                runner.stop()
                return
            runner.run()
        except Exception as exc:  # noqa: BLE001 — surface to UI, never crash the thread
            try:
                sink.on_status("error", {"message": str(exc)})
            except Exception:
                pass

    def stop(self, timeout: float = 15.0) -> None:
        self._stop_requested = True
        runner, thread = self._runner, self._thread
        if runner is not None:
            runner.stop()
        if thread is not None:
            thread.join(timeout=timeout)
        self._runner, self._thread = None, None

    def estop(self, timeout: float = 15.0) -> None:
        self._stop_requested = True
        runner, thread = self._runner, self._thread
        if runner is not None:
            runner.estop()
        if thread is not None:
            thread.join(timeout=timeout)
        self._runner, self._thread = None, None
