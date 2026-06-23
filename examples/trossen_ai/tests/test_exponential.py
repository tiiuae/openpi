import threading

import numpy as np

from ensemble import ExponentialEnsemble


def test_exp_blend_weights_oldest_highest():
    e = ExponentialEnsemble(decay=1.0)
    e.add_chunk(0, np.array([[0.0], [10.0]]))  # step1 <- 10 (older, k=0)
    e.add_chunk(1, np.array([[20.0]]))         # step1 <- 20 (newer, k=1)
    a = e.get_action(1)
    assert 10.0 < a[0] < 15.0


def test_exp_buffer_evicts_consumed_steps():
    e = ExponentialEnsemble(decay=1.0)
    for s in range(100):
        e.add_chunk(s, np.ones((5, 2)))  # each chunk covers s..s+4
        e.get_action(s)                  # consume step s
    assert all(k > 99 for k in e._buffer), f"stale keys: {sorted(e._buffer)[:5]}"
    assert len(e._buffer) <= 5


def test_exp_concurrent_add_and_get_no_corruption():
    e = ExponentialEnsemble(decay=1.0)
    errors = []

    def producer():
        try:
            for s in range(2000):
                e.add_chunk(s, np.ones((10, 4)) * s)
        except Exception as ex:  # noqa: BLE001
            errors.append(ex)

    def consumer():
        try:
            for s in range(2000):
                e.get_action(s)
        except Exception as ex:  # noqa: BLE001
            errors.append(ex)

    t1, t2 = threading.Thread(target=producer), threading.Thread(target=consumer)
    t1.start(); t2.start(); t1.join(); t2.join()
    assert not errors, errors
