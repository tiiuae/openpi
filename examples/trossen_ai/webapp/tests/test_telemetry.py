import queue

import numpy as np

from webapp.telemetry import NullSink, QueueSink


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_null_sink_accepts_all_calls():
    s = NullSink()
    s.on_log("INFO", "hi", 1.0)
    s.on_action(0, np.zeros(14), 1.0)
    s.on_inference(12.3, 1.0)
    s.on_chunk(0, np.ones((5, 14)), 1.0)
    s.on_overlap(0, 3)
    s.on_images({"cam": b"x"}, 1.0)
    s.on_status("started", {})  # no exception = pass


def test_queue_sink_emits_typed_events():
    q = queue.Queue()
    s = QueueSink(q)
    s.on_log("WARNING", "careful", 2.0)
    s.on_overlap(7, 4)
    events = _drain(q)
    assert {"type": "log", "level": "WARNING", "msg": "careful", "ts": 2.0} in events
    assert {"type": "overlap", "step": 7, "count": 4} in events


def test_queue_sink_action_includes_raw_from_latest_chunk():
    q = queue.Queue()
    s = QueueSink(q)
    # chunk queried at step 5 covers steps 5..7
    s.on_chunk(5, np.array([[10.0], [11.0], [12.0]]), 1.0)
    s.on_action(6, np.array([99.0]), 1.1)  # blended output for step 6
    events = _drain(q)
    action_evt = next(e for e in events if e["type"] == "action")
    assert action_evt["step"] == 6
    assert action_evt["action"] == [99.0]
    assert action_evt["raw"] == [11.0]  # chunk[6 - 5]


def test_queue_sink_action_raw_none_when_out_of_range():
    q = queue.Queue()
    s = QueueSink(q)
    s.on_chunk(5, np.array([[10.0]]), 1.0)  # covers only step 5
    s.on_action(9, np.array([1.0]), 1.1)
    action_evt = next(e for e in _drain(q) if e["type"] == "action")
    assert action_evt["raw"] is None


def test_queue_sink_throttles_images():
    q = queue.Queue()
    s = QueueSink(q, image_min_interval_s=10.0)
    s.on_images({"cam": b"abc"}, ts=100.0)
    s.on_images({"cam": b"def"}, ts=100.5)  # within interval -> dropped
    s.on_images({"cam": b"ghi"}, ts=200.0)  # after interval -> sent
    imgs = [e for e in _drain(q) if e["type"] == "images"]
    assert len(imgs) == 2
