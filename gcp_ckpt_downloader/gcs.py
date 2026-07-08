"""Thin wrapper around the `gcloud storage` / `gcloud auth` CLIs.

All GCS access goes through subprocess calls to the `gcloud` CLI rather than
a Python client library, since the target machine already has `gcloud`
installed and authenticated. `_run` is the single subprocess boundary: it
never raises (including on nonzero exit or timeout), so callers always get a
plain `(returncode, stdout, stderr)` tuple to reason about.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

from . import config

# Short timeout for the connectivity probe in auth_status(), so a hung/slow
# network doesn't block the UI's auth chip for long.
_AUTH_CHECK_TIMEOUT_SECONDS = 10

_ADC_PATH = Path("~/.config/gcloud/application_default_credentials.json")


def _run(args: list[str], timeout: float | None = None) -> tuple[int, str, str]:
    """Run a CLI command, returning (returncode, stdout, stderr).

    Never raises: a timeout or a missing/unrunnable executable is reported as
    a nonzero returncode with a descriptive stderr message, same as any other
    command failure. Callers decide what a nonzero returncode means.
    """
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"timed out after {timeout}s: {' '.join(args)}"
    except OSError as e:
        return 1, "", str(e)


def _normalize_prefix(prefix: str) -> str:
    if not prefix.startswith("gs://"):
        prefix = "gs://" + prefix.lstrip("/")
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix


def _parent_of(normalized_prefix: str) -> str | None:
    """Return the parent gs:// prefix, or None if already at the bucket root."""
    trimmed = normalized_prefix[:-1]  # drop trailing "/"
    path_part = trimmed[len("gs://") :]
    if "/" not in path_part:
        return None  # trimmed is just "gs://bucket" -- no parent above the bucket
    return trimmed.rsplit("/", 1)[0] + "/"


def list_prefix(prefix: str) -> dict:
    """List the immediate children of `prefix` (one level, not recursive).

    Returns {"prefix", "parent", "folders": [...], "objects": [...]} on
    success, or {"error": <stderr>} on failure.
    """
    normalized = _normalize_prefix(prefix)
    returncode, stdout, stderr = _run(["gcloud", "storage", "ls", "--project", config.GCP_PROJECT, normalized])
    if returncode != 0:
        return {"error": stderr.strip()}

    folders: list[dict] = []
    objects: list[dict] = []
    for raw_line in stdout.splitlines():
        uri = raw_line.strip()
        if not uri or uri == normalized:
            continue
        if uri.endswith("/"):
            name = uri.rstrip("/").rsplit("/", 1)[-1]
            folders.append({"name": name, "uri": uri})
        else:
            name = uri.rsplit("/", 1)[-1]
            objects.append({"name": name, "uri": uri})

    return {
        "prefix": normalized,
        "parent": _parent_of(normalized),
        "folders": folders,
        "objects": objects,
    }


def auth_status() -> dict:
    """Report whether gcloud is authenticated and can reach the bucket."""
    result: dict = {}

    returncode, stdout, stderr = _run(["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"])
    accounts = [line.strip() for line in stdout.splitlines() if line.strip()]
    result["active_account"] = accounts[0] if accounts else None
    if returncode != 0 and stderr.strip():
        result["error"] = stderr.strip()

    result["adc_present"] = _ADC_PATH.expanduser().exists()

    can_list_returncode, _, can_list_stderr = _run(
        ["gcloud", "storage", "ls", "--project", config.GCP_PROJECT, config.DEFAULT_BUCKET],
        timeout=_AUTH_CHECK_TIMEOUT_SECONDS,
    )
    result["can_list"] = can_list_returncode == 0
    if can_list_returncode != 0 and can_list_stderr.strip():
        result.setdefault("error", can_list_stderr.strip())

    return result


def download_cmd(src_uri: str, dest_parent: str) -> list[str]:
    """Build the argv (not a shell string) for one `gcloud storage cp` item.

    Copying `gs://.../step_10000` into `/dest/` creates `/dest/step_10000/`.
    """
    return [
        "gcloud",
        "storage",
        "cp",
        "-r",
        "--project",
        config.GCP_PROJECT,
        src_uri.rstrip("/"),
        dest_parent,
    ]
