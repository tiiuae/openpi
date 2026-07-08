"""In-memory download-job registry with a sequential background worker.

Each `DownloadStore.create()` call starts one background thread that copies
`items` (GCS checkpoint folders) into `dest` one at a time, by running the
argv `gcs.download_cmd(uri, dest)` builds (`gcloud storage cp -r ...`) as a
subprocess. All job state lives in a plain dict guarded by `_lock`; `get()`
hands back a snapshot copy so a caller polling from another thread never
observes a half-mutated item.

v1 allows only one active ("running") job at a time: `create()` raises
`JobAlreadyRunningError` if called while a job is still running. This is a
plain Python exception (this module has no HTTP concept) -- callers (e.g. a
web server) translate it into whatever status code makes sense for them.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid

from . import gcs

# Keep only the tail of gcloud's stderr in memory -- a large checkpoint copy
# can produce a long progress stream and the UI only needs enough of it for
# context, not the full transcript.
_LOG_TAIL_MAX_CHARS = 2048


class JobAlreadyRunningError(Exception):
    """Raised by `create()` when another job is still `running`."""


class DownloadStore:
    """In-memory registry of download jobs, one background thread per job."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def create(self, dest: str, items: list[dict]) -> dict:
        """Start a new download job copying `items` into `dest`.

        `items` is a list of `{"uri": "gs://...", "name": "..."}` dicts.
        Creates `dest` (and parents) if it doesn't already exist. Raises
        `JobAlreadyRunningError` if another job is currently `running` --
        v1 supports only one active job at a time.
        """
        with self._lock:
            if any(job["state"] == "running" for job in self._jobs.values()):
                raise JobAlreadyRunningError("a download job is already running")

            os.makedirs(dest, exist_ok=True)

            job_id = f"dl-{uuid.uuid4().hex[:8]}"
            job = {
                "id": job_id,
                "dest": dest,
                "created": time.time(),
                "state": "running",
                "items": [
                    {
                        "uri": item["uri"],
                        "name": item["name"],
                        "state": "queued",
                        "message": "",
                        "log_tail": "",
                    }
                    for item in items
                ],
            }
            self._jobs[job_id] = job
            snapshot = self._snapshot_locked(job)

        threading.Thread(target=self._run, args=(job_id,), daemon=True).start()
        return snapshot

    def get(self, job_id: str) -> dict | None:
        """Return a snapshot of job `job_id`, or None if it doesn't exist."""
        with self._lock:
            job = self._jobs.get(job_id)
            return self._snapshot_locked(job) if job is not None else None

    @staticmethod
    def _snapshot_locked(job: dict) -> dict:
        # Copy the job dict and each item dict so the caller's copy can't be
        # mutated by the background thread after it's handed back. Must be
        # called with `_lock` held.
        return {**job, "items": [dict(item) for item in job["items"]]}

    def _update_item(self, job_id: str, index: int, **fields) -> None:
        with self._lock:
            self._jobs[job_id]["items"][index].update(fields)

    def _finish_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["state"] = "error" if any(item["state"] == "error" for item in job["items"]) else "done"

    def _run(self, job_id: str) -> None:
        with self._lock:
            dest = self._jobs[job_id]["dest"]
            item_count = len(self._jobs[job_id]["items"])

        for index in range(item_count):
            with self._lock:
                uri = self._jobs[job_id]["items"][index]["uri"]

            self._update_item(job_id, index, state="downloading")
            tail, returncode = self._download_one(uri, dest, job_id, index)

            if returncode == 0:
                self._update_item(job_id, index, state="done", log_tail=tail)
            else:
                self._update_item(job_id, index, state="error", message=tail, log_tail=tail)

        self._finish_job(job_id)

    def _download_one(self, uri: str, dest: str, job_id: str, index: int) -> tuple[str, int]:
        """Run one `gcloud storage cp` for `uri`, streaming stderr into a
        rolling ~2KB tail (updated live so polling sees progress) and
        returning `(final_tail, returncode)`.
        """
        argv = gcs.download_cmd(uri, dest)
        tail = ""
        try:
            # stdout is discarded (unused): gcloud storage cp writes its
            # progress to stderr, and leaving stdout as PIPE without ever
            # reading it risks the child blocking once that pipe fills.
            proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            for line in proc.stderr:
                tail += line
                if len(tail) > _LOG_TAIL_MAX_CHARS:
                    tail = tail[-_LOG_TAIL_MAX_CHARS:]
                self._update_item(job_id, index, log_tail=tail)
            returncode = proc.wait()
        except OSError as exc:
            return str(exc), 1
        return tail, returncode
