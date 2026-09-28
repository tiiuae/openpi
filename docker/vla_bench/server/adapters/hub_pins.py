"""Pin the Hugging Face Hub repos a policy resolves BY NAME to the revision it was trained and verified with.

Several policies rebuild a frozen backbone, tokenizer or processor from a Hub repo id every time they load
(`AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")`). Offline, the Hub library resolves that name
through the cache's `refs/main`. Online, it resolves the Hub's current `main`. Either way a newer upstream
revision changes tokenisation or image processing without any error. The deployment audit (2026-09-25)
found six models exposed to this.

`apply(pins)` makes the pinned revision the only one a by-name load can see, without touching the model
code or the repo-id strings:

  * It builds an overlay hub cache under /tmp. Each pinned repo exposes ONLY its pinned snapshot, and its
    `refs/main` names that snapshot. Every other entry of the real cache is symlinked through unchanged.
    `HF_HUB_CACHE` then points at the overlay. The real cache (usually the read-only /hf mount) is never
    written.
  * A pinned snapshot missing from the cache is a hard error at startup, naming the exact
    `hf download ... --revision <sha>` command. Online (HF_HUB_OFFLINE unset or 0), the pinned revision is
    downloaded first. The process is then switched to HF_HUB_OFFLINE=1, so nothing re-resolves `main`.
  * `hf download --revision <sha>` writes no `refs/main`, so before this, a cache seeded that way failed
    offline for a by-name load. The overlay supplies the ref, so both ways of seeding the cache now work.

It must run before `huggingface_hub` is imported, because the library reads HF_HUB_CACHE and
HF_HUB_OFFLINE once, at import. vla_server.build_adapter() calls apply_from_env() before it imports the
adapter module. Pins come from VLA_BENCH_HUB_PINS (set per model in its Dockerfile), a JSON object:
    {"<repo_id>": "<40-hex revision>"}
    {"<repo_id>": {"revision": "<sha>", "allow_patterns": [...], "ignore_patterns": [...]}}
The patterns only shape an online download (the same files the model's README tells a workstation to fetch).

`pinned_snapshot()` is the single-repo form, for an adapter that needs a local directory rather than a name
(gr00t's backbone).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ENV = "VLA_BENCH_HUB_PINS"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_applied: dict | None = None


def _spec(repo: str, v) -> dict:
    d = {"revision": v} if isinstance(v, str) else dict(v)
    if not _SHA.match(str(d.get("revision", ""))):
        raise ValueError(f"{ENV}: {repo} must be pinned to a 40-hex commit, got {d.get('revision')!r}")
    return d


def hub_cache_dir() -> Path:
    """Where huggingface_hub would look, by its own precedence (HF_HUB_CACHE > HUGGINGFACE_HUB_CACHE > HF_HOME/hub)."""
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(var):
            return Path(os.environ[var])
    home = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "huggingface")
    return Path(home) / "hub"


def offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "0").strip().upper() in ("1", "ON", "YES", "TRUE")


def _folder(repo: str) -> str:
    return "models--" + repo.replace("/", "--")


def download_command(repo: str, spec: dict) -> str:
    # One --include pattern per command: hf >= 1.0 takes one value per flag (a second bare value is read as a
    # file name) and hf < 1.0 keeps only the last of repeated flags; one pattern per call works on both.
    base = f"hf download {repo} --revision {spec['revision']}"
    base += "".join(f' --exclude "{p}"' for p in spec.get("ignore_patterns") or [])
    includes = spec.get("allow_patterns") or []
    return " && ".join(f'{base} --include "{p}"' for p in includes) if includes else base


_DOWNLOAD = ("import json, sys\n"
             "from huggingface_hub import snapshot_download\n"
             "a = json.loads(sys.argv[1])\n"
             "print(snapshot_download(a['repo'], revision=a['revision'], allow_patterns=a.get('allow_patterns'),"
             " ignore_patterns=a.get('ignore_patterns')))\n")


def _fetch(repo: str, spec: dict, cache: Path) -> None:
    """Download exactly the pinned revision into `cache`. In a subprocess, so this process still has not
    imported huggingface_hub when HF_HUB_CACHE is redirected afterwards."""
    env = {**os.environ, "HF_HUB_OFFLINE": "0", "HF_HUB_CACHE": str(cache)}
    r = subprocess.run([sys.executable, "-c", _DOWNLOAD, json.dumps({"repo": repo, **spec})], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"hub pin: could not download {repo}@{spec['revision']} into {cache}:\n"
                         f"{r.stderr.strip()[-2000:]}\nOn a machine with network access run:  "
                         f"{download_command(repo, spec)}  and mount that cache at /hf.")


def pinned_snapshot(repo: str, revision: str, allow_patterns=None, ignore_patterns=None) -> Path:
    """Local directory holding `repo` at exactly `revision`: the cache's snapshot, downloaded first if online."""
    spec = _spec(repo, {"revision": revision, "allow_patterns": allow_patterns, "ignore_patterns": ignore_patterns})
    cache = hub_cache_dir()
    snap = cache / _folder(repo) / "snapshots" / revision
    if not snap.is_dir():
        if offline():
            raise SystemExit(f"{repo} at the pinned revision {revision} is not in the Hugging Face cache "
                             f"({cache}). On the workstation run:  {download_command(repo, spec)}")
        _fetch(repo, spec, cache)
    if not snap.is_dir():
        raise SystemExit(f"{repo}@{revision}: download reported success but {snap} does not exist")
    return snap


def apply(pins: dict) -> dict:
    """Redirect HF_HUB_CACHE to an overlay that exposes only the pinned revision of each pinned repo.
    Idempotent. Returns {repo: revision} plus where each came from, for the startup log."""
    global _applied
    if _applied is not None:
        return _applied
    if not pins:
        _applied = {}
        return _applied
    if "huggingface_hub" in sys.modules:
        raise RuntimeError(f"{ENV} must be applied before huggingface_hub is imported: the library reads "
                           f"HF_HUB_CACHE and HF_HUB_OFFLINE once, at import")
    specs = {repo: _spec(repo, v) for repo, v in pins.items()}
    src = hub_cache_dir()
    was_offline = offline()
    missing = [r for r, s in specs.items() if not (src / _folder(r) / "snapshots" / s["revision"]).is_dir()]
    if missing and was_offline:
        raise SystemExit("pinned Hugging Face revisions missing from the cache at " f"{src} (HF_HUB_OFFLINE=1):\n"
                         + "\n".join(f"  {download_command(r, specs[r])}" for r in missing)
                         + "\nRun these on the workstation and mount that cache at /hf.")
    for r in missing:
        _fetch(r, specs[r], src)

    overlay = Path(tempfile.mkdtemp(prefix="vla_bench_hub_pins_"))
    pinned_folders = {_folder(r): r for r in specs}
    if src.is_dir():
        for child in src.iterdir():
            if child.name not in pinned_folders:
                (overlay / child.name).symlink_to(child)
    report = {}
    for folder, repo in pinned_folders.items():
        rev = specs[repo]["revision"]
        real, fake = src / folder, overlay / folder
        (fake / "snapshots").mkdir(parents=True)
        (fake / "refs").mkdir()
        (fake / "snapshots" / rev).symlink_to(real / "snapshots" / rev)
        (fake / "refs" / "main").write_text(rev)
        for child in real.iterdir():          # blobs/ (the snapshot's symlinks point there), .no_exist/, ...
            if child.name not in ("snapshots", "refs"):
                (fake / child.name).symlink_to(child)
        report[repo] = {"revision": rev, "from": "downloaded" if repo in missing else "cache"}
    os.environ["HF_HUB_CACHE"] = str(overlay)
    if not was_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"    # every pinned repo is local now; nothing may re-resolve `main`
    report["_overlay"] = str(overlay)
    report["_real_cache"] = str(src)
    _applied = report
    return report


def apply_from_env() -> dict:
    text = os.environ.get(ENV, "").strip()
    if not text:
        return apply({})
    try:
        pins = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"{ENV} is not valid JSON ({e}): {text!r}")
    if not isinstance(pins, dict):
        raise SystemExit(f"{ENV} must be a JSON object of repo_id -> revision, got {text!r}")
    return apply(pins)
