import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from webapp.server import create_app

_URDF = (Path(__file__).resolve().parents[2] / "external" / "joint_to_ee"
         / "trossen_arm_description" / "urdf" / "generated" / "mobile_ai.urdf")


def test_index_served():
    client = TestClient(create_app())
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_presets_round_trip(tmp_path):
    client = TestClient(create_app(presets_dir=tmp_path))
    assert client.get("/api/presets").json() == []
    client.post("/api/presets", json={"name": "rig-a", "config": {"control_freq": 25}})
    assert client.get("/api/presets").json() == ["rig-a"]
    assert client.get("/api/presets/rig-a").json()["control_freq"] == 25
    client.delete("/api/presets/rig-a")
    assert client.get("/api/presets").json() == []


def test_health_returns_json():
    client = TestClient(create_app())
    r = client.get("/api/health")
    assert r.status_code == 200
    assert "session_running" in r.json()


def test_feedback_endpoint_writes_file(tmp_path):
    client = TestClient(create_app(presets_dir=tmp_path, feedback_dir=tmp_path / "fb"))
    r = client.post("/api/feedback", json={"name": "Ada", "email": "a@x.com", "feedback": "hi"})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    files = list((tmp_path / "fb").glob("*.md"))
    assert len(files) == 1
    assert "Ada" in files[0].read_text()


def test_files_endpoint_lists_dirs(tmp_path):
    (tmp_path / "sub").mkdir()
    client = TestClient(create_app())
    r = client.get("/api/files", params={"path": str(tmp_path)})
    assert r.status_code == 200
    assert [e["name"] for e in r.json()["entries"]] == ["sub"]


def test_shutdown_hook_runs_clean():
    app = create_app()
    with TestClient(app):  # entering/exiting triggers startup/shutdown
        pass
    # Exiting the context ran the shutdown hook (session.stop()) without raising.
    assert True


def test_replay_page_served():
    client = TestClient(create_app())
    r = client.get("/replay")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_episode_trajectory_route(monkeypatch):
    from webapp import episode_preview as ep

    captured = {}

    def fake_build(dataset_dir, episode_index, control_freq, max_joint_speed,
                   seed_joints=None, **kw):
        captured.update(dataset_dir=dataset_dir, episode_index=episode_index,
                        control_freq=control_freq, max_joint_speed=max_joint_speed,
                        seed_joints=seed_joints)
        return {"n_frames": 2, "spikes": [1]}

    monkeypatch.setattr(ep, "build_trajectory", fake_build)
    client = TestClient(create_app())
    r = client.get("/api/episode_trajectory", params={
        "dataset_dir": "/data/ds", "episode_index": 3,
        "control_freq": 25, "max_joint_speed": 3.0})
    assert r.status_code == 200
    assert r.json()["spikes"] == [1]
    assert captured["dataset_dir"] == "/data/ds"
    assert captured["episode_index"] == 3
    assert captured["control_freq"] == 25
    assert captured["seed_joints"] is None


def test_teleop_page_served():
    client = TestClient(create_app())
    r = client.get("/teleop")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_ws_start_teleop_uses_factory():
    started = {}

    class FakeRunner:
        def __init__(self, kind, config, sink):
            started["kind"] = kind
            started["config"] = config
            self._sink = sink
        def run(self):
            self._sink.on_status("teleop_started", {"detached": True})
        def stop(self): pass
        def estop(self): pass

    app = create_app(runner_factory=lambda k, c, s: FakeRunner(k, c, s))
    client = TestClient(app)
    with client.websocket_connect("/ws/telemetry") as ws:
        ws.send_json({"action": "start_teleop",
                      "config": {"mode": "detached", "control_freq": 10}})
        for _ in range(20):
            evt = ws.receive_json()
            if evt.get("type") == "status" and evt.get("kind") == "teleop_started":
                break
        assert started["kind"] == "teleop"
        assert started["config"]["mode"] == "detached"


def test_ws_teleop_input_and_switch_arm_routed_to_runner():
    import threading

    calls = []
    release = threading.Event()

    class FakeInput:
        def update(self, payload):
            calls.append(payload)

    class FakeRunner:
        def __init__(self, kind, config, sink):
            self.input = FakeInput()
            self._sink = sink
        def run(self):
            # Reachable as session._runner while the test sends input, then
            # block until the test releases so run() doesn't hang the suite.
            self._sink.on_status("teleop_started", {"detached": True})
            release.wait(timeout=5.0)
        def stop(self): release.set()
        def estop(self): release.set()

    app = create_app(runner_factory=lambda k, c, s: FakeRunner(k, c, s))
    client = TestClient(app)
    with client.websocket_connect("/ws/telemetry") as ws:
        ws.send_json({"action": "start_teleop", "config": {"mode": "detached"}})
        for _ in range(20):
            evt = ws.receive_json()
            if evt.get("type") == "status" and evt.get("kind") == "teleop_started":
                break
        ws.send_json({"action": "teleop_input",
                      "payload": {"axes": [1, 0, 0, 0, 0, 0], "grip": 0.0}})
        ws.send_json({"action": "switch_arm"})
        # Spin the WS so both dispatch branches run before we assert.
        ws.send_json({"action": "stop"})
        release.set()

    assert {"axes": [1, 0, 0, 0, 0, 0], "grip": 0.0} in calls
    assert {"switch_arm": True} in calls


@pytest.mark.skipif(not _URDF.is_file(), reason="URDF tree not checked out")
def test_robot_urdf_and_mesh_served():
    client = TestClient(create_app())
    r = client.get("/robot.urdf")
    assert r.status_code == 200
    assert "robot" in r.text[:500].lower()  # URDF root tag
    # A mesh referenced as package://trossen_arm_description/meshes/... resolves.
    m = client.get("/pkg/trossen_arm_description/meshes/wxai/base_link.stl")
    assert m.status_code == 200
    assert len(m.content) > 0


# add to webapp/tests/test_server.py
import json as _json


def test_live_run_records_to_runs_dir(tmp_path):
    class FakeRunner:
        def __init__(self, kind, config, sink):
            self._sink = sink
        def run(self):
            import numpy as np
            self._sink.on_status("started", {})
            self._sink.on_action(0, np.array([1.0]), 1.0)
            self._sink.on_overlap(0, 2)
            self._sink.on_status("stopped", {"reason": "finished"})
        def stop(self): pass
        def estop(self): pass

    app = create_app(runs_dir=tmp_path, runner_factory=lambda k, c, s: FakeRunner(k, c, s))
    client = TestClient(app)
    with client.websocket_connect("/ws/telemetry") as ws:
        ws.send_json({"action": "start_live", "config": {"model_name": "pi0-test"}})
        run_id = None
        for _ in range(40):
            evt = ws.receive_json()
            if evt.get("type") == "status" and evt.get("kind") == "run_started":
                run_id = evt["payload"]["run_id"]
            if evt.get("type") == "status" and evt.get("kind") == "stopped":
                break
        assert run_id is not None
    # RecordingSink wrote a run dir with manifest + events (+ summary via close()).
    run_dir = tmp_path / run_id
    manifest = _json.loads((run_dir / "manifest.json").read_text())
    assert manifest["model_name"] == "pi0-test"
    assert (run_dir / "events.jsonl").read_text().strip() != ""


def test_non_live_run_does_not_record(tmp_path):
    class FakeRunner:
        def __init__(self, kind, config, sink): self._sink = sink
        def run(self): self._sink.on_status("stopped", {"reason": "finished"})
        def stop(self): pass
        def estop(self): pass

    app = create_app(runs_dir=tmp_path, runner_factory=lambda k, c, s: FakeRunner(k, c, s))
    client = TestClient(app)
    with client.websocket_connect("/ws/telemetry") as ws:
        ws.send_json({"action": "start_replay", "config": {"dataset_dir": "/x"}})
        for _ in range(20):
            evt = ws.receive_json()
            if evt.get("type") == "status" and evt.get("kind") == "stopped":
                break
    assert list(tmp_path.iterdir()) == []  # nothing recorded for replay
