# webapp/tests/test_run_store.py
import json
from pathlib import Path

from webapp.run_store import RunStore


def _make_run(base: Path, run_id: str, *, model_name="m", end_reason="completed",
              extra_events=None):
    d = base / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "started_at": 1.0, "kind": "live",
        "config": {"model_name": model_name, "smoothing": True}, "model_name": model_name,
        "model": None}))
    (d / "summary.json").write_text(json.dumps({
        "run_id": run_id, "steps": 10, "end_reason": end_reason,
        "rtt_ms": {"mean": 40.0}, "overlap_mean": 3.0, "smoothing_delta_mean": 0.1}))
    events = extra_events or [{"type": "overlap", "step": 0, "count": 2}]
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    return d


def test_list_returns_summaries_newest_first(tmp_path):
    _make_run(tmp_path, "2026-07-02-100000", model_name="old")
    _make_run(tmp_path, "2026-07-02-120000", model_name="new")
    store = RunStore(tmp_path)
    runs = store.list()
    assert [r["run_id"] for r in runs] == ["2026-07-02-120000", "2026-07-02-100000"]
    assert runs[0]["model_name"] == "new"
    assert runs[0]["steps"] == 10
    assert runs[0]["end_reason"] == "completed"


def test_list_tolerates_run_without_summary(tmp_path):
    d = tmp_path / "half"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"run_id": "half", "started_at": 1.0,
                                                 "config": {}, "model_name": None}))
    store = RunStore(tmp_path)
    runs = store.list()
    assert runs[0]["run_id"] == "half"
    assert runs[0]["end_reason"] == "unknown"


def test_get_returns_manifest_summary_rating(tmp_path):
    _make_run(tmp_path, "r1")
    store = RunStore(tmp_path)
    got = store.get("r1")
    assert got["manifest"]["run_id"] == "r1"
    assert got["summary"]["steps"] == 10
    assert got["rating"] is None
    assert "events" not in got  # events only when requested


def test_get_with_events(tmp_path):
    _make_run(tmp_path, "r1", extra_events=[
        {"type": "overlap", "step": 0, "count": 2},
        {"type": "action", "step": 0, "action": [1.0], "raw": [0.0], "ts": 1.0}])
    store = RunStore(tmp_path)
    got = store.get("r1", with_events=True)
    assert len(got["events"]) == 2


def test_set_rating_writes_file(tmp_path):
    _make_run(tmp_path, "r1")
    store = RunStore(tmp_path)
    store.set_rating("r1", {"success": "partial", "score": 3, "note": "ok"})
    saved = json.loads((tmp_path / "r1" / "rating.json").read_text())
    assert saved["success"] == "partial"
    assert saved["score"] == 3
    assert "rated_at" in saved
    assert store.get("r1")["rating"]["note"] == "ok"


def test_delete_removes_run_dir(tmp_path):
    _make_run(tmp_path, "r1")
    store = RunStore(tmp_path)
    store.delete("r1")
    assert not (tmp_path / "r1").exists()
    assert store.list() == []


def test_delete_missing_run_raises_keyerror(tmp_path):
    store = RunStore(tmp_path)
    try:
        store.delete("nope")
    except KeyError:
        return
    assert False, "expected KeyError"


def test_get_missing_run_raises_keyerror(tmp_path):
    store = RunStore(tmp_path)
    try:
        store.get("nope")
    except KeyError:
        return
    assert False, "expected KeyError"


def test_run_id_traversal_rejected(tmp_path):
    store = RunStore(tmp_path)
    for bad in ("../secret", "a/b", ".."):
        try:
            store.get(bad)
        except (KeyError, ValueError):
            continue
        assert False, f"expected rejection for {bad!r}"
