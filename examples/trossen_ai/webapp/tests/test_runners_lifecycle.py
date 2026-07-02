"""Lifecycle of LiveRunner: cheap construction, stop honored during connect.

These tests stub the policy-connect wait and the bridge so no hardware/server
is needed. They assert the runner constructs without blocking and that stop
prevents the bridge from ever being built/run.
"""
import time

import runners


class _Sink:
    def __init__(self):
        self.statuses = []

    def on_status(self, kind, payload):
        self.statuses.append((kind, payload))


def test_live_runner_construction_is_cheap(monkeypatch):
    called = {"wait": False}

    def fake_wait(*a, **k):
        called["wait"] = True

    monkeypatch.setattr(runners, "wait_for_policy_server", fake_wait, raising=False)
    r = runners.LiveRunner("live", {"policy_host": "h", "policy_port": 1}, _Sink())
    # No connect / bridge build happened during construction.
    assert called["wait"] is False
    assert r._bridge is None


def test_live_runner_stop_before_run_skips_bridge(monkeypatch):
    built = {"bridge": False}

    def fake_wait(host, port, timeout, should_stop, interval=0.5):
        # Simulate a slow connect that observes the stop flag.
        for _ in range(100):
            if should_stop():
                from policy_connect import ConnectStopped
                raise ConnectStopped()
            time.sleep(0.005)

    def fake_make_bridge(self, cfg):
        built["bridge"] = True
        raise AssertionError("bridge must not be built after stop")

    monkeypatch.setattr(runners, "wait_for_policy_server", fake_wait, raising=False)
    monkeypatch.setattr(runners.LiveRunner, "_make_bridge", fake_make_bridge, raising=False)

    sink = _Sink()
    r = runners.LiveRunner("live", {"policy_host": "h", "policy_port": 1}, sink)
    r.stop()  # request stop before run
    r.run()   # should raise ConnectStopped internally, caught, no bridge
    assert built["bridge"] is False


def test_live_runner_emits_model_metadata(monkeypatch):
    import runners

    events = []

    class _Sink:
        def on_status(self, kind, payload): events.append((kind, payload))
        def on_log(self, *a): pass

    class _Client:
        def get_server_metadata(self): return {"policy": "pi0", "ckpt": "step_40000"}

    class _Bridge:
        def __init__(self): self.policy_client = _Client()
        def run_episode(self, task_prompt=""): pass
        def cleanup(self): pass

    r = runners.LiveRunner("live", {"policy_host": "h", "policy_port": 1,
                            "connect_timeout": 0.1}, _Sink())
    monkeypatch.setattr(runners, "wait_for_policy_server", lambda *a, **k: None)
    monkeypatch.setattr(r, "_make_bridge", lambda cfg: _Bridge())
    r.run()
    assert ("model", {"policy": "pi0", "ckpt": "step_40000"}) in events
