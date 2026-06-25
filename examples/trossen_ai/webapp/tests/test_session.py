import time

import pytest

from webapp.session import SessionManager


class _NoopSink:
    def on_status(self, *a): ...


class FakeRunner:
    """Stands in for a bridge/replay run. run() blocks until stop()."""

    def __init__(self, kind, config, sink):
        self.kind = kind
        self.config = config
        self.sink = sink
        self._running = False
        self.estopped = False

    def run(self):
        self._running = True
        self.sink.on_status("started", {"kind": self.kind})
        while self._running:
            time.sleep(0.005)
        self.sink.on_status("stopped", {})

    def stop(self):
        self._running = False

    def estop(self):
        self.estopped = True
        self._running = False


def _factory(created):
    def make(kind, config, sink):
        r = FakeRunner(kind, config, sink)
        created.append(r)
        return r
    return make


def test_start_then_stop():
    created = []
    sm = SessionManager(_factory(created))
    sm.start("live", {"x": 1}, sink=_NoopSink())
    time.sleep(0.02)
    assert sm.is_running()
    sm.stop()
    assert not sm.is_running()
    assert created[0].kind == "live"


def test_single_session_guard():
    created = []
    sm = SessionManager(_factory(created))
    sm.start("live", {}, sink=_NoopSink())
    time.sleep(0.02)
    with pytest.raises(RuntimeError):
        sm.start("replay", {}, sink=_NoopSink())
    sm.stop()


def test_estop_calls_runner_estop():
    created = []
    sm = SessionManager(_factory(created))
    sm.start("live", {}, sink=_NoopSink())
    time.sleep(0.02)
    sm.estop()
    assert created[0].estopped
    assert not sm.is_running()
