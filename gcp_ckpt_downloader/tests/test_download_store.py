from __future__ import annotations

import threading
import time

import pytest

from gcp_ckpt_downloader import download_store
from gcp_ckpt_downloader.download_store import DownloadStore
from gcp_ckpt_downloader.download_store import JobAlreadyRunningError

ITEMS = [
    {"uri": "gs://bucket/a/step_10000/", "name": "step_10000"},
    {"uri": "gs://bucket/a/step_20000/", "name": "step_20000"},
]


def _wait_until(predicate, timeout=2.0, interval=0.01):
    """Poll `predicate` until it's truthy or `timeout` elapses.

    Used instead of a fixed sleep to synchronize with the store's background
    worker thread without making the test slow or flaky.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class _FakeProcess:
    """Stand-in for subprocess.Popen's return value.

    `stderr` is an iterable of lines (mimicking a text-mode pipe); `wait()`
    returns the canned returncode. Both are driven entirely from in-memory
    data -- no real subprocess/network involved.
    """

    def __init__(self, order_log, name, lines, returncode):
        self._order_log = order_log
        self._name = name
        self._lines = lines
        self.returncode = returncode

    @property
    def stderr(self):
        return iter(self._lines)

    def wait(self):
        self._order_log.append(f"end:{self._name}")
        return self.returncode


def _make_fake_popen(order_log, script):
    """Build a fake `subprocess.Popen` driven by `script: {name: (lines, returncode)}`.

    The item name is recovered from argv, which gcs.download_cmd shapes as
    [..., src_uri, dest_parent] -- argv[-2] is the (slash-stripped) source uri.
    """

    def fake_popen(argv, **kwargs):
        name = argv[-2].rsplit("/", 1)[-1]
        order_log.append(f"start:{name}")
        lines, returncode = script[name]
        return _FakeProcess(order_log, name, lines, returncode)

    return fake_popen


# ---------------------------------------------------------------------------
# create() / _run(): happy path, sequential ordering, item + overall state
# ---------------------------------------------------------------------------


def test_create_returns_job_with_id_dest_and_queued_items(monkeypatch, tmp_path):
    # This test only inspects create()'s synchronous return value, but
    # create() still spawns a real background thread that would otherwise
    # call the real subprocess.Popen -- an actual `gcloud storage cp` attempt
    # against a fake bucket. Fake it out like every other test in this file
    # so the suite stays fully off-network.
    order_log: list[str] = []
    script = {
        "step_10000": (["ok\n"], 0),
        "step_20000": (["ok\n"], 0),
    }
    monkeypatch.setattr(download_store.subprocess, "Popen", _make_fake_popen(order_log, script))

    store = DownloadStore()

    job = store.create(str(tmp_path), ITEMS)

    assert job["id"].startswith("dl-")
    assert job["dest"] == str(tmp_path)
    assert job["state"] == "running"
    assert [item["name"] for item in job["items"]] == ["step_10000", "step_20000"]
    assert all(item["state"] in ("queued", "downloading", "done") for item in job["items"])


def test_create_makes_dest_directory_if_missing(tmp_path):
    dest = tmp_path / "nested" / "dest"
    store = DownloadStore()

    store.create(str(dest), [])

    assert dest.is_dir()


def test_two_items_download_sequentially_and_all_succeed(monkeypatch, tmp_path):
    order_log: list[str] = []
    script = {
        "step_10000": (["Copying step_10000...\n", "Completed 1/1\n"], 0),
        "step_20000": (["Copying step_20000...\n", "Completed 1/1\n"], 0),
    }
    monkeypatch.setattr(download_store.subprocess, "Popen", _make_fake_popen(order_log, script))

    store = DownloadStore()
    job = store.create(str(tmp_path), ITEMS)

    assert _wait_until(lambda: store.get(job["id"])["state"] != "running")

    final = store.get(job["id"])
    assert final["state"] == "done"
    assert [item["state"] for item in final["items"]] == ["done", "done"]
    assert "Completed 1/1" in final["items"][0]["log_tail"]

    # The crux of the sequencing requirement: item 2 must not start until
    # item 1's process has fully finished (order interleaves, not batches).
    assert order_log == [
        "start:step_10000",
        "end:step_10000",
        "start:step_20000",
        "end:step_20000",
    ]


def test_second_item_error_marks_item_and_overall_job_as_error(monkeypatch, tmp_path):
    order_log: list[str] = []
    script = {
        "step_10000": (["ok\n"], 0),
        "step_20000": (["ERROR: permission denied\n"], 1),
    }
    monkeypatch.setattr(download_store.subprocess, "Popen", _make_fake_popen(order_log, script))

    store = DownloadStore()
    job = store.create(str(tmp_path), ITEMS)

    assert _wait_until(lambda: store.get(job["id"])["state"] != "running")

    final = store.get(job["id"])
    assert final["state"] == "error"
    assert final["items"][0]["state"] == "done"
    assert final["items"][1]["state"] == "error"
    assert "permission denied" in final["items"][1]["message"]


def test_log_tail_is_capped_to_roughly_2kb(monkeypatch, tmp_path):
    order_log: list[str] = []
    long_lines = [f"progress line {i}\n" for i in range(500)]  # far more than 2KB
    script = {"step_10000": (long_lines, 0)}
    monkeypatch.setattr(download_store.subprocess, "Popen", _make_fake_popen(order_log, script))

    store = DownloadStore()
    job = store.create(str(tmp_path), [ITEMS[0]])

    assert _wait_until(lambda: store.get(job["id"])["state"] != "running")

    final = store.get(job["id"])
    tail = final["items"][0]["log_tail"]
    assert len(tail) <= download_store._LOG_TAIL_MAX_CHARS  # noqa: SLF001 -- checking the documented cap
    assert tail.endswith("progress line 499\n")


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


def test_get_returns_none_for_unknown_job_id(tmp_path):
    store = DownloadStore()

    assert store.get("dl-doesnotexist") is None


def test_get_returns_independent_snapshot_not_live_reference(monkeypatch, tmp_path):
    # A fake process that blocks until released, so the job is deterministically
    # still "running" (item "downloading") when we grab our snapshot -- a fake
    # that finishes instantly would race with the assertions below.
    release = threading.Event()

    class _BlockingProcess:
        returncode = 0

        @property
        def stderr(self):
            release.wait(timeout=2)
            return iter(["ok\n"])

        def wait(self):
            return 0

    monkeypatch.setattr(download_store.subprocess, "Popen", lambda argv, **kwargs: _BlockingProcess())

    store = DownloadStore()
    job = store.create(str(tmp_path), [ITEMS[0]])
    assert _wait_until(lambda: store.get(job["id"])["items"][0]["state"] == "downloading")
    snapshot = store.get(job["id"])
    assert snapshot["state"] == "running"
    assert snapshot["items"][0]["state"] == "downloading"

    release.set()
    assert _wait_until(lambda: store.get(job["id"])["state"] != "running")

    # The earlier snapshot must not have been mutated in place by the
    # background thread after it was handed back.
    assert snapshot["state"] == "running"
    assert snapshot["items"][0]["state"] == "downloading"


# ---------------------------------------------------------------------------
# Concurrency guard: one active job at a time
# ---------------------------------------------------------------------------


def test_create_rejects_a_second_job_while_one_is_running(monkeypatch, tmp_path):
    release = threading.Event()

    class _BlockingProcess:
        returncode = 0

        @property
        def stderr(self):
            release.wait(timeout=2)
            return iter(["done\n"])

        def wait(self):
            return 0

    monkeypatch.setattr(download_store.subprocess, "Popen", lambda argv, **kwargs: _BlockingProcess())

    store = DownloadStore()
    first_job = store.create(str(tmp_path), [ITEMS[0]])
    assert first_job["state"] == "running"

    with pytest.raises(JobAlreadyRunningError):
        store.create(str(tmp_path), [ITEMS[1]])

    release.set()
    assert _wait_until(lambda: store.get(first_job["id"])["state"] != "running")


def test_create_allows_a_new_job_after_the_previous_one_finished(monkeypatch, tmp_path):
    script = {
        "step_10000": (["ok\n"], 0),
        "step_20000": (["ok\n"], 0),
    }
    order_log: list[str] = []
    monkeypatch.setattr(download_store.subprocess, "Popen", _make_fake_popen(order_log, script))

    store = DownloadStore()
    first_job = store.create(str(tmp_path), [ITEMS[0]])
    assert _wait_until(lambda: store.get(first_job["id"])["state"] != "running")

    second_job = store.create(str(tmp_path), [ITEMS[1]])
    assert second_job["state"] == "running"
    assert _wait_until(lambda: store.get(second_job["id"])["state"] != "running")
    assert store.get(second_job["id"])["state"] == "done"
