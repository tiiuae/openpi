# webapp/run_store.py
"""Read/write access to recorded Live runs under a runs directory."""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class RunStore:
    def __init__(self, runs_dir) -> None:
        self._dir = Path(runs_dir)

    def _run_path(self, run_id: str) -> Path:
        # Reject path traversal / nested ids; run dirs are flat single segments.
        if not _RUN_ID_RE.match(run_id or ""):
            raise ValueError(f"invalid run id: {run_id!r}")
        return self._dir / run_id

    @staticmethod
    def _read_json(path: Path):
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def list(self) -> list[dict]:
        if not self._dir.is_dir():
            return []
        rows = []
        for d in self._dir.iterdir():
            if not d.is_dir():
                continue
            manifest = self._read_json(d / "manifest.json")
            if manifest is None:
                continue
            summary = self._read_json(d / "summary.json") or {}
            rating = self._read_json(d / "rating.json")
            rows.append({
                "run_id": manifest.get("run_id", d.name),
                "started_at": manifest.get("started_at"),
                "model_name": manifest.get("model_name"),
                "config": manifest.get("config", {}),
                "steps": summary.get("steps"),
                "end_reason": summary.get("end_reason", "unknown"),
                "rtt_mean": (summary.get("rtt_ms") or {}).get("mean"),
                "overlap_mean": summary.get("overlap_mean"),
                "smoothing_delta_mean": summary.get("smoothing_delta_mean"),
                "rating": rating,
            })
        rows.sort(key=lambda r: r["run_id"], reverse=True)
        return rows

    def get(self, run_id: str, with_events: bool = False) -> dict:
        d = self._run_path(run_id)
        manifest = self._read_json(d / "manifest.json")
        if manifest is None:
            raise KeyError(run_id)
        out = {
            "manifest": manifest,
            "summary": self._read_json(d / "summary.json"),
            "rating": self._read_json(d / "rating.json"),
        }
        if with_events:
            out["events"] = self._read_events(d / "events.jsonl")
        return out

    @staticmethod
    def _read_events(path: Path) -> list[dict]:
        try:
            text = path.read_text()
        except OSError:
            return []
        events = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from a crash mid-write
        return events

    def set_rating(self, run_id: str, rating: dict) -> None:
        d = self._run_path(run_id)
        if not (d / "manifest.json").exists():
            raise KeyError(run_id)
        payload = dict(rating or {})
        payload["rated_at"] = time.time()
        (d / "rating.json").write_text(json.dumps(payload, indent=2))

    def delete(self, run_id: str) -> None:
        d = self._run_path(run_id)
        if d.is_symlink() or not d.is_dir() or not (d / "manifest.json").exists():
            raise KeyError(run_id)
        shutil.rmtree(d)
