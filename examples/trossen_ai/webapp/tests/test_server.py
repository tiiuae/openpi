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
