from fastapi.testclient import TestClient

from webapp.server import create_app


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
