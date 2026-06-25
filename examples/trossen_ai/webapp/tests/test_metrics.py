import numpy as np

from webapp.metrics import Metrics


def test_rtt_percentiles():
    m = Metrics(window=100)
    for v in range(1, 101):  # 1..100 ms
        m.add_rtt(float(v))
    snap = m.snapshot()
    assert snap["rtt_last"] == 100.0
    assert 49 <= snap["rtt_p50"] <= 52
    assert 94 <= snap["rtt_p95"] <= 96


def test_loop_hz_from_action_timestamps():
    m = Metrics(window=100)
    for i in range(11):  # 10 intervals of 0.02s -> 50 Hz
        m.add_action(ts=i * 0.02, action=np.zeros(14))
    assert abs(m.snapshot()["loop_hz"] - 50.0) < 1.0


def test_jitter_is_rolling_stddev_of_action_deltas():
    m = Metrics(window=100)
    for i in range(5):  # constant action -> zero jitter
        m.add_action(ts=i * 0.02, action=np.ones(14))
    assert m.snapshot()["jitter"] == 0.0
    m.add_action(ts=0.12, action=np.ones(14) * 5)  # introduce variation
    assert m.snapshot()["jitter"] > 0.0


def test_drops_counter():
    m = Metrics(window=10)
    assert m.snapshot()["drops"] == 0
    m.add_drop(); m.add_drop()
    assert m.snapshot()["drops"] == 2


def test_empty_snapshot_is_safe():
    snap = Metrics(window=10).snapshot()
    assert snap["rtt_last"] is None
    assert snap["loop_hz"] is None
    assert snap["jitter"] is None
    assert snap["drops"] == 0
