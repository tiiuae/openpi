from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from gcp_ckpt_downloader import config
from gcp_ckpt_downloader import download_store
from gcp_ckpt_downloader import server


class FakeStore:
    """Minimal stand-in for DownloadStore: real state, no background threads.

    server.py's routes only ever call `.create()`/`.get()` on the store, so
    this fake implements exactly that surface -- with genuine state (a jobs
    dict) instead of a Mock, so assertions exercise real request/response
    wiring rather than call-count bookkeeping. `create()` can be told to
    raise `JobAlreadyRunningError` to drive the 409 path without needing a
    real background thread race.
    """

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.create_calls: list[tuple[str, list]] = []
        self.raise_already_running = False

    def create(self, dest, items):
        self.create_calls.append((dest, items))
        if self.raise_already_running:
            raise download_store.JobAlreadyRunningError("a download job is already running")
        job = {"id": "dl-test1234", "dest": dest, "created": 0, "state": "running", "items": items}
        self.jobs[job["id"]] = job
        return job

    def get(self, job_id):
        return self.jobs.get(job_id)


@pytest.fixture
def fake_store():
    return FakeStore()


@pytest.fixture
def client(fake_store):
    app = server.create_app(store=fake_store)
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /api/config
# ---------------------------------------------------------------------------


def test_get_config_returns_defaults_from_config_module(client):
    resp = client.get("/api/config")

    assert resp.status_code == 200
    assert resp.json() == {"bucket": config.DEFAULT_BUCKET, "dest": config.DEFAULT_DEST}


# ---------------------------------------------------------------------------
# GET /api/gcs/auth
# ---------------------------------------------------------------------------


def test_gcs_auth_returns_gcs_auth_status_result(client, monkeypatch):
    monkeypatch.setattr(
        server.gcs,
        "auth_status",
        lambda: {"active_account": "me@example.com", "adc_present": True, "can_list": True},
    )

    resp = client.get("/api/gcs/auth")

    assert resp.status_code == 200
    assert resp.json() == {"active_account": "me@example.com", "adc_present": True, "can_list": True}


# ---------------------------------------------------------------------------
# GET /api/gcs/ls
# ---------------------------------------------------------------------------


def test_gcs_ls_defaults_to_default_bucket_when_prefix_omitted(client, monkeypatch):
    seen = {}

    def fake_list_prefix(prefix):
        seen["prefix"] = prefix
        return {"prefix": prefix, "parent": None, "folders": [], "objects": []}

    monkeypatch.setattr(server.gcs, "list_prefix", fake_list_prefix)

    resp = client.get("/api/gcs/ls")

    assert resp.status_code == 200
    assert seen["prefix"] == config.DEFAULT_BUCKET


def test_gcs_ls_passes_through_explicit_prefix_and_result(client, monkeypatch):
    def fake_list_prefix(prefix):
        return {
            "prefix": prefix,
            "parent": "gs://bucket/",
            "folders": [{"name": "step_10000", "uri": prefix + "step_10000/"}],
            "objects": [],
        }

    monkeypatch.setattr(server.gcs, "list_prefix", fake_list_prefix)

    resp = client.get("/api/gcs/ls", params={"prefix": "gs://bucket/sub/"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["prefix"] == "gs://bucket/sub/"
    assert body["folders"] == [{"name": "step_10000", "uri": "gs://bucket/sub/step_10000/"}]


def test_gcs_ls_rejects_prefix_that_does_not_start_with_gs_scheme(client, monkeypatch):
    called = []
    monkeypatch.setattr(server.gcs, "list_prefix", called.append)

    resp = client.get("/api/gcs/ls", params={"prefix": "not-a-gs-uri"})

    assert resp.status_code == 400
    assert "detail" in resp.json()
    assert called == []  # rejected before ever reaching gcs.list_prefix


# ---------------------------------------------------------------------------
# GET /api/local/ls
# ---------------------------------------------------------------------------


def test_local_ls_defaults_to_home_tilde_when_path_omitted(client, monkeypatch):
    seen = {}

    def fake_list_directory(path_str):
        seen["path_str"] = path_str
        return {"path": "/home/someone", "parent": "/home", "entries": []}

    monkeypatch.setattr(server.local_fs, "list_directory", fake_list_directory)

    resp = client.get("/api/local/ls")

    assert resp.status_code == 200
    assert seen["path_str"] == "~"


def test_local_ls_passes_through_explicit_path_and_result(client, monkeypatch):
    monkeypatch.setattr(
        server.local_fs,
        "list_directory",
        lambda path_str: {
            "path": path_str,
            "parent": "/",
            "entries": [{"name": "sub", "type": "dir", "path": "/tmp/sub"}],
        },
    )

    resp = client.get("/api/local/ls", params={"path": "/tmp"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == "/tmp"
    assert body["entries"] == [{"name": "sub", "type": "dir", "path": "/tmp/sub"}]


# ---------------------------------------------------------------------------
# GET /api/local/existing
# ---------------------------------------------------------------------------


def test_local_existing_splits_comma_separated_names_and_returns_wrapped_result(client, monkeypatch):
    seen = {}

    def fake_existing_names(dest, names):
        seen["dest"] = dest
        seen["names"] = names
        return ["step_10000"]

    monkeypatch.setattr(server.local_fs, "existing_names", fake_existing_names)

    resp = client.get("/api/local/existing", params={"dest": "/tmp/dest", "names": "step_10000,step_20000"})

    assert resp.status_code == 200
    assert resp.json() == {"existing": ["step_10000"]}
    assert seen["dest"] == "/tmp/dest"
    assert seen["names"] == ["step_10000", "step_20000"]


def test_local_existing_rejects_relative_dest(client, monkeypatch):
    called = []
    monkeypatch.setattr(server.local_fs, "existing_names", lambda dest, names: called.append(dest))

    resp = client.get("/api/local/existing", params={"dest": "relative/dest", "names": "a"})

    assert resp.status_code == 400
    assert called == []


# ---------------------------------------------------------------------------
# POST /api/gcs/download
# ---------------------------------------------------------------------------


def test_start_download_calls_store_create_and_returns_the_job(client, fake_store):
    resp = client.post(
        "/api/gcs/download",
        json={"dest": "/tmp/dest", "items": [{"uri": "gs://bucket/step_10000/", "name": "step_10000"}]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "dl-test1234"
    assert body["dest"] == "/tmp/dest"
    assert fake_store.create_calls == [("/tmp/dest", [{"uri": "gs://bucket/step_10000/", "name": "step_10000"}])]


def test_start_download_rejects_relative_dest_without_calling_store(client, fake_store):
    resp = client.post("/api/gcs/download", json={"dest": "relative/dest", "items": []})

    assert resp.status_code == 400
    assert fake_store.create_calls == []


def test_start_download_returns_409_when_a_job_is_already_running(client, fake_store):
    fake_store.raise_already_running = True

    resp = client.post("/api/gcs/download", json={"dest": "/tmp/dest", "items": []})

    assert resp.status_code == 409
    assert "detail" in resp.json()


# ---------------------------------------------------------------------------
# GET /api/gcs/download/{job_id}
# ---------------------------------------------------------------------------


def test_get_download_job_returns_the_stored_job(client, fake_store):
    fake_store.jobs["dl-existing"] = {
        "id": "dl-existing",
        "dest": "/tmp/dest",
        "created": 0,
        "state": "done",
        "items": [],
    }

    resp = client.get("/api/gcs/download/dl-existing")

    assert resp.status_code == 200
    assert resp.json()["state"] == "done"


def test_get_download_job_404s_for_unknown_job_id(client):
    resp = client.get("/api/gcs/download/dl-does-not-exist")

    assert resp.status_code == 404
    assert "detail" in resp.json()


# ---------------------------------------------------------------------------
# GET / and /static mount
# ---------------------------------------------------------------------------


def test_index_serves_static_index_html(monkeypatch, tmp_path, fake_store):
    (tmp_path / "index.html").write_text("<html>hello</html>")
    monkeypatch.setattr(server, "STATIC_DIR", tmp_path)

    app = server.create_app(store=fake_store)
    client = TestClient(app)

    resp = client.get("/")

    assert resp.status_code == 200
    assert "hello" in resp.text


def test_static_mount_serves_files_from_static_dir(monkeypatch, tmp_path, fake_store):
    (tmp_path / "index.html").write_text("<html>hello</html>")
    (tmp_path / "theme.css").write_text("body { color: red; }")
    monkeypatch.setattr(server, "STATIC_DIR", tmp_path)

    app = server.create_app(store=fake_store)
    client = TestClient(app)

    resp = client.get("/static/theme.css")

    assert resp.status_code == 200
    assert "color: red" in resp.text
