from __future__ import annotations

import pathlib
import subprocess

from gcp_ckpt_downloader import config
from gcp_ckpt_downloader import gcs

# ---------------------------------------------------------------------------
# _run: the subprocess boundary. Must never raise, always returns a 3-tuple.
# ---------------------------------------------------------------------------


def test_run_returns_returncode_stdout_stderr_without_raising_on_nonzero(monkeypatch):
    class FakeCompleted:
        returncode = 3
        stdout = "some output\n"
        stderr = "some error\n"

    def fake_subprocess_run(args, *, capture_output, text, timeout, check):
        assert capture_output is True
        assert text is True
        assert check is False
        return FakeCompleted()

    monkeypatch.setattr(gcs.subprocess, "run", fake_subprocess_run)

    result = gcs._run(["gcloud", "storage", "ls"])  # noqa: SLF001 -- testing the subprocess boundary itself

    assert result == (3, "some output\n", "some error\n")


def test_run_turns_timeout_into_nonzero_return_instead_of_raising(monkeypatch):
    def fake_subprocess_run(args, *, capture_output, text, timeout, check):
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr(gcs.subprocess, "run", fake_subprocess_run)

    returncode, _stdout, stderr = gcs._run(["gcloud", "storage", "ls"], timeout=5)  # noqa: SLF001

    assert returncode != 0
    assert isinstance(stderr, str)
    assert stderr


# ---------------------------------------------------------------------------
# list_prefix
# ---------------------------------------------------------------------------


def test_list_prefix_splits_folders_and_objects_and_normalizes_prefix(monkeypatch):
    calls = []

    def fake_run(args, timeout=None):
        calls.append(args)
        stdout = "gs://bucket/a/b/step_10000/\ngs://bucket/a/b/step_20000/\ngs://bucket/a/b/info.json\n"
        return 0, stdout, ""

    monkeypatch.setattr(gcs, "_run", fake_run)

    # Note: no trailing slash on input -- normalization must add it.
    result = gcs.list_prefix("gs://bucket/a/b")

    assert result["prefix"] == "gs://bucket/a/b/"
    assert result["parent"] == "gs://bucket/a/"
    assert result["folders"] == [
        {"name": "step_10000", "uri": "gs://bucket/a/b/step_10000/"},
        {"name": "step_20000", "uri": "gs://bucket/a/b/step_20000/"},
    ]
    assert result["objects"] == [
        {"name": "info.json", "uri": "gs://bucket/a/b/info.json"},
    ]
    assert calls == [["gcloud", "storage", "ls", "--project", config.GCP_PROJECT, "gs://bucket/a/b/"]]


def test_list_prefix_at_bucket_root_has_no_parent(monkeypatch):
    monkeypatch.setattr(gcs, "_run", lambda args, timeout=None: (0, "gs://bucket/a/\n", ""))

    result = gcs.list_prefix("gs://bucket/")

    assert result["prefix"] == "gs://bucket/"
    assert result["parent"] is None


def test_list_prefix_excludes_the_prefix_itself_if_echoed_back(monkeypatch):
    # Some gcloud versions echo the queried prefix as its own line; it must
    # not be double-counted as a child folder of itself.
    stdout = "gs://bucket/a/\ngs://bucket/a/c/\n"
    monkeypatch.setattr(gcs, "_run", lambda args, timeout=None: (0, stdout, ""))

    result = gcs.list_prefix("gs://bucket/a/")

    names = [f["name"] for f in result["folders"]]
    assert names == ["c"]


def test_list_prefix_returns_error_dict_on_nonzero_returncode(monkeypatch):
    monkeypatch.setattr(gcs, "_run", lambda args, timeout=None: (1, "", "PERMISSION_DENIED: no access\n"))

    result = gcs.list_prefix("gs://bucket/a/")

    assert result == {"error": "PERMISSION_DENIED: no access"}


# ---------------------------------------------------------------------------
# auth_status
# ---------------------------------------------------------------------------


def test_auth_status_reports_ok_state(monkeypatch):
    def fake_run(args, timeout=None):
        if "auth" in args:
            return 0, "user@example.com\n", ""
        return 0, "", ""  # storage ls (can_list check) succeeds

    monkeypatch.setattr(gcs, "_run", fake_run)
    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)

    result = gcs.auth_status()

    assert result["active_account"] == "user@example.com"
    assert result["adc_present"] is True
    assert result["can_list"] is True
    assert "error" not in result


def test_auth_status_can_list_false_when_storage_ls_fails(monkeypatch):
    def fake_run(args, timeout=None):
        if "auth" in args:
            return 0, "user@example.com\n", ""
        return 1, "", "PERMISSION_DENIED\n"

    monkeypatch.setattr(gcs, "_run", fake_run)
    monkeypatch.setattr(pathlib.Path, "exists", lambda self: False)

    result = gcs.auth_status()

    assert result["active_account"] == "user@example.com"
    assert result["adc_present"] is False
    assert result["can_list"] is False
    assert result["error"] == "PERMISSION_DENIED"


def test_auth_status_no_active_account_when_auth_list_empty(monkeypatch):
    monkeypatch.setattr(gcs, "_run", lambda args, timeout=None: (0, "", ""))
    monkeypatch.setattr(pathlib.Path, "exists", lambda self: False)

    result = gcs.auth_status()

    assert result["active_account"] is None


# ---------------------------------------------------------------------------
# download_cmd
# ---------------------------------------------------------------------------


def test_download_cmd_builds_argv_for_gcloud_storage_cp():
    argv = gcs.download_cmd("gs://bucket/a/step_10000/", "/dest/")

    assert argv == [
        "gcloud",
        "storage",
        "cp",
        "-r",
        "--project",
        config.GCP_PROJECT,
        "gs://bucket/a/step_10000",
        "/dest/",
    ]


def test_download_cmd_strips_trailing_slash_only_from_source():
    argv = gcs.download_cmd("gs://bucket/a/step_10000", "/dest")

    assert argv[-2:] == ["gs://bucket/a/step_10000", "/dest"]
