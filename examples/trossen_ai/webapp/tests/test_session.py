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


import threading


def test_start_returns_immediately_when_construction_blocks():
    """Runner construction must not block the caller (the WS event loop)."""
    def slow_factory(kind, config, sink):
        time.sleep(0.5)  # simulate a blocking policy-server connect
        return FakeRunner(kind, config, sink)

    sm = SessionManager(slow_factory)
    t0 = time.perf_counter()
    sm.start("live", {}, sink=_NoopSink())
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, f"start() blocked for {elapsed:.3f}s"
    sm.stop()


def test_stop_during_construction_skips_run():
    """If stop is requested before construction finishes, run() is never entered."""
    gate = threading.Event()
    ran = []

    class GatedRunner:
        def __init__(self, kind, config, sink):
            gate.wait(2.0)

        def run(self):
            ran.append(True)

        def stop(self):
            ...

        def estop(self):
            ...

    sm = SessionManager(lambda k, c, s: GatedRunner(k, c, s))
    sm.start("live", {}, sink=_NoopSink())
    time.sleep(0.05)  # thread now blocked inside __init__
    stopper = threading.Thread(target=sm.stop)
    stopper.start()
    time.sleep(0.05)
    gate.set()  # let construction finish; _run should see stop flag
    stopper.join(2.0)
    assert ran == []


def test_construction_error_emits_error_status():
    statuses = []

    class RecordingSink:
        def on_status(self, kind, payload):
            statuses.append((kind, payload))

    def boom_factory(kind, config, sink):
        raise RuntimeError("connect failed")

    sm = SessionManager(boom_factory)
    sm.start("live", {}, sink=RecordingSink())
    time.sleep(0.1)
    assert any(k == "error" for k, _ in statuses)
    assert not sm.is_running()


def test_session_calls_sink_close_on_finish():
    closed = []

    class _Sink:
        def on_status(self, kind, payload): pass
        def close(self): closed.append(True)

    class _Runner:
        def __init__(self, kind, config, sink): pass
        def run(self): pass
        def stop(self): pass
        def estop(self): pass

    mgr = SessionManager(lambda k, c, s: _Runner(k, c, s))
    sink = _Sink()
    mgr.start("live", {}, sink)
    mgr._thread.join(timeout=5.0)
    assert closed == [True]


def test_session_close_called_even_when_runner_raises():
    closed = []

    class _Sink:
        def on_status(self, kind, payload): pass
        def close(self): closed.append(True)

    class _Boom:
        def __init__(self, kind, config, sink): pass
        def run(self): raise RuntimeError("crash")
        def stop(self): pass
        def estop(self): pass

    mgr = SessionManager(lambda k, c, s: _Boom(k, c, s))
    mgr.start("live", {}, _Sink())
    mgr._thread.join(timeout=5.0)
    assert closed == [True]
