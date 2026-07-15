import json
import queue

import numpy as np

from webapp.recording import RecordingSink
from webapp.telemetry import QueueSink


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_recording_sink_writes_ee_event(tmp_path):
    sink = RecordingSink(tmp_path / "run1", run_id="run1", config={})
    # EE chunk of 3 steps, 16-D each, added for query_step=0
    ee_chunk = np.arange(3 * 16, dtype=float).reshape(3, 16)
    sink.on_ee_chunk(0, ee_chunk, ts=1.0)
    # executed step 1 -> should log ee row 1
    sink.on_action(1, np.zeros(14), ts=2.0)
    sink.close()

    lines = (tmp_path / "run1" / "events.jsonl").read_text().splitlines()
    ee_events = [json.loads(x) for x in lines if json.loads(x).get("type") == "ee"]
    assert len(ee_events) == 1
    assert ee_events[0]["step"] == 1
    assert ee_events[0]["ee"] == list(range(16, 32))


def test_queue_sink_joins_ee_chunk_to_action():
    q = queue.Queue()
    s = QueueSink(q)
    # EE chunk queried at step 5 covers steps 5..7
    ee_chunk = np.arange(3 * 16, dtype=float).reshape(3, 16)
    s.on_ee_chunk(5, ee_chunk, ts=1.0)
    s.on_action(6, np.zeros(14), ts=1.1)  # step 6 -> ee row 6-5 = 1
    ee_evts = [e for e in _drain(q) if e["type"] == "ee"]
    assert len(ee_evts) == 1
    assert ee_evts[0]["step"] == 6
    assert ee_evts[0]["ee"] == list(range(16, 32))  # ee_chunk[1]
    assert ee_evts[0]["ts"] == 1.1


def test_queue_sink_no_ee_event_when_out_of_range():
    q = queue.Queue()
    s = QueueSink(q)
    ee_chunk = np.arange(3 * 16, dtype=float).reshape(3, 16)  # covers steps 0..2
    s.on_ee_chunk(0, ee_chunk, ts=1.0)
    s.on_action(5, np.zeros(14), ts=1.1)  # offset 5 >= len 3 -> no ee event
    assert [e for e in _drain(q) if e["type"] == "ee"] == []


def test_recording_sink_no_ee_event_when_out_of_range(tmp_path):
    sink = RecordingSink(tmp_path / "run1", run_id="run1", config={})
    ee_chunk = np.arange(3 * 16, dtype=float).reshape(3, 16)  # covers steps 0..2
    sink.on_ee_chunk(0, ee_chunk, ts=1.0)
    sink.on_action(5, np.zeros(14), ts=2.0)  # offset 5 >= len 3 -> no ee event
    sink.close()

    lines = (tmp_path / "run1" / "events.jsonl").read_text().splitlines()
    ee_events = [json.loads(x) for x in lines if json.loads(x).get("type") == "ee"]
    assert ee_events == []
