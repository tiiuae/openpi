"""FastAPI app: static UI mount + REST routes wiring config/gcs/local_fs/download_store together.

Route table (see the plan / task brief for the authoritative spec):
    GET  /                                       -> static/index.html
    GET  /api/config                             -> {bucket, dest}
    GET  /api/gcs/auth                           -> gcs.auth_status()
    GET  /api/gcs/ls?prefix=                     -> gcs.list_prefix(prefix or DEFAULT_BUCKET)
    GET  /api/local/ls?path=                     -> local_fs.list_directory(path or "~")
    GET  /api/local/existing?dest=&names=a,b,c   -> {"existing": [...]}
    POST /api/gcs/download                       -> store.create(dest, items)
    GET  /api/gcs/download/{job_id}              -> store.get(job_id) or 404

This module only wires HTTP request/response shapes to Tasks 1-2's modules;
it holds no business logic of its own beyond minimal input validation (a
`prefix` must look like a `gs://` URI, a `dest` must be an absolute local
path) and translating `download_store.JobAlreadyRunningError` into a 409.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from . import download_store
from . import gcs
from . import local_fs

STATIC_DIR = Path(__file__).parent / "static"


def _validated_prefix(raw_prefix: str) -> str:
    """Fall back to the default bucket when omitted; reject anything else
    that doesn't look like a `gs://` URI."""
    if not raw_prefix:
        return config.DEFAULT_BUCKET
    if not raw_prefix.startswith("gs://"):
        raise HTTPException(status_code=400, detail=f"prefix must start with gs://: {raw_prefix!r}")
    return raw_prefix


def _validated_dest(raw_dest: str) -> str:
    """Require a non-empty, absolute local path -- there's no sensible
    default destination to fall back to."""
    if not raw_dest or not Path(raw_dest).expanduser().is_absolute():
        raise HTTPException(status_code=400, detail=f"dest must be an absolute path: {raw_dest!r}")
    return raw_dest


def create_app(store: download_store.DownloadStore | None = None) -> FastAPI:
    if store is None:
        store = download_store.DownloadStore()

    app = FastAPI(title="GCP Checkpoint Downloader")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        # static/index.html doesn't exist until Task 5 populates static/; a
        # missing file here is expected until then, so this deliberately
        # doesn't special-case its absence.
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/config")
    def get_config():
        return {"bucket": config.DEFAULT_BUCKET, "dest": config.DEFAULT_DEST}

    @app.get("/api/gcs/auth")
    def gcs_auth():
        return gcs.auth_status()

    @app.get("/api/gcs/ls")
    def gcs_ls(prefix: str = ""):
        return gcs.list_prefix(_validated_prefix(prefix))

    @app.get("/api/local/ls")
    def local_ls(path: str = ""):
        return local_fs.list_directory(path or "~")

    @app.get("/api/local/existing")
    def local_existing(dest: str = "", names: str = ""):
        name_list = [n.strip() for n in names.split(",") if n.strip()]
        return {"existing": local_fs.existing_names(_validated_dest(dest), name_list)}

    @app.post("/api/gcs/download")
    def start_download(body: dict):
        dest = _validated_dest(body.get("dest", ""))
        items = body.get("items", [])
        try:
            return store.create(dest, items)
        except download_store.JobAlreadyRunningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/gcs/download/{job_id}")
    def get_download_job(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job id: {job_id}")
        return job

    return app


app = create_app()  # for `uvicorn gcp_ckpt_downloader.server:app`
