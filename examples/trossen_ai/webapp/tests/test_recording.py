# webapp/tests/test_recording.py
import numpy as np

from webapp.recording import TeeSink


class _Spy:
    def __init__(self):
        self.calls = []
    def on_log(self, level, msg, ts): self.calls.append(("log", level, msg, ts))
    def on_action(self, step, action, ts): self.calls.append(("action", step, ts))
    def on_inference(self, rtt_ms, ts): self.calls.append(("inference", rtt_ms, ts))
    def on_chunk(self, query_step, chunk, ts): self.calls.append(("chunk", query_step, ts))
    def on_overlap(self, step, count): self.calls.append(("overlap", step, count))
    def on_weights(self, step, weights, ts): self.calls.append(("weights", step, ts))
    def on_images(self, images, ts): self.calls.append(("images", ts))
    def on_status(self, kind, payload): self.calls.append(("status", kind))


class _Boom(_Spy):
    def on_overlap(self, step, count):
        raise RuntimeError("inner sink failed")


def test_tee_fans_out_to_all_sinks():
    a, b = _Spy(), _Spy()
    tee = TeeSink([a, b])
    tee.on_action(3, np.zeros(14), 1.0)
    tee.on_overlap(3, 2)
    tee.on_status("started", {})
    assert ("action", 3, 1.0) in a.calls and ("action", 3, 1.0) in b.calls
    assert ("overlap", 3, 2) in a.calls and ("overlap", 3, 2) in b.calls
    assert ("status", "started") in a.calls and ("status", "started") in b.calls


def test_tee_one_raising_sink_does_not_starve_others():
    boom, ok = _Boom(), _Spy()
    tee = TeeSink([boom, ok])
    tee.on_overlap(1, 1)  # boom raises internally; must not propagate
    assert ("overlap", 1, 1) in ok.calls


def test_tee_close_calls_close_on_inner_sinks_that_have_it():
    closed = []

    class _Closable(_Spy):
        def close(self): closed.append(True)

    tee = TeeSink([_Closable(), _Spy()])  # second has no close()
    tee.close()  # must not raise despite one sink lacking close()
    assert closed == [True]


# add to webapp/tests/test_recording.py
import json
from pathlib import Path

from webapp.recording import RecordingSink


def _read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def test_recording_writes_manifest_on_construction(tmp_path):
    run_dir = tmp_path / "2026-07-02-181500"
    RecordingSink(run_dir, run_id="2026-07-02-181500",
                  config={"model_name": "pi0-v3", "control_freq": 25})
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["run_id"] == "2026-07-02-181500"
    assert manifest["kind"] == "live"
    assert manifest["config"]["control_freq"] == 25
    assert manifest["model_name"] == "pi0-v3"     # copied from config
    assert manifest["model"] is None              # no server metadata yet
    assert "started_at" in manifest


def test_recording_appends_events_to_jsonl(tmp_path):
    import numpy as np
    run_dir = tmp_path / "r"
    sink = RecordingSink(run_dir, run_id="r", config={})
    sink.on_action(0, np.array([1.0, 2.0]), 1.0)
    sink.on_overlap(0, 3)
    sink.on_weights(0, np.array([0.6, 0.4]), 1.0)
    rows = _read_jsonl(run_dir / "events.jsonl")
    assert rows[0] == {"type": "action", "step": 0, "action": [1.0, 2.0], "raw": None, "ts": 1.0}
    assert {"type": "overlap", "step": 0, "count": 3} in rows
    assert {"type": "weights", "step": 0, "weights": [0.6, 0.4], "ts": 1.0} in rows


def test_recording_does_not_record_images(tmp_path):
    run_dir = tmp_path / "r"
    sink = RecordingSink(run_dir, run_id="r", config={})
    sink.on_images({"cam": b"xxx"}, 1.0)
    rows = _read_jsonl(run_dir / "events.jsonl")
    assert all(r["type"] != "images" for r in rows)


def test_recording_action_raw_from_latest_chunk(tmp_path):
    import numpy as np
    run_dir = tmp_path / "r"
    sink = RecordingSink(run_dir, run_id="r", config={})
    sink.on_chunk(5, np.array([[10.0], [11.0], [12.0]]), 1.0)
    sink.on_action(6, np.array([99.0]), 1.1)
    rows = _read_jsonl(run_dir / "events.jsonl")
    action = next(r for r in rows if r["type"] == "action")
    assert action["raw"] == [11.0]  # chunk[6-5]


def test_recording_folds_model_metadata_into_manifest(tmp_path):
    run_dir = tmp_path / "r"
    sink = RecordingSink(run_dir, run_id="r", config={"model_name": "user-typed"})
    sink.on_status("model", {"policy": "pi0", "ckpt": "step_40000"})
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["model"] == {"policy": "pi0", "ckpt": "step_40000"}
    assert manifest["model_name"] == "user-typed"  # user field stays authoritative


def test_recording_disk_error_degrades_to_noop(tmp_path):
    # Point the run dir at a path whose parent is a file -> mkdir fails.
    bad_parent = tmp_path / "afile"
    bad_parent.write_text("x")
    sink = RecordingSink(bad_parent / "run", run_id="run", config={})  # must not raise
    sink.on_overlap(0, 1)  # must not raise
    sink.close()           # must not raise
