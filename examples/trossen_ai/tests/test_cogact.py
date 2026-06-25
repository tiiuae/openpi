import logging
import threading

import numpy as np

from ensemble import CogACTEnsemble


def test_cogact_concurrent_add_and_get_no_corruption():
    e = CogACTEnsemble(max_buffer_size=25, mode="cogact")
    errors = []

    def producer():
        try:
            for s in range(2000):
                e.add_chunk(s, np.ones((10, 4)) * (s + 1))
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


def test_cogact_last_weights_single_candidate_is_unit():
    e = CogACTEnsemble(max_buffer_size=25, mode="cogact")
    e.add_chunk(5, np.array([[1.0, 2.0]]))  # one prediction for step 5
    e.get_action(5)
    w = e.last_weights()
    assert w is not None and len(w) == 1
    assert np.isclose(w[0], 1.0)


def test_cogact_last_weights_normalized_for_multiple(caplog):
    e = CogACTEnsemble(max_buffer_size=25, mode="cogact")
    e.add_chunk(0, np.array([[1.0, 0.0], [1.0, 0.0]]))
    e.add_chunk(1, np.array([[1.0, 0.0]]))
    e.get_action(1)
    w = e.last_weights()
    assert w is not None and len(w) == 2
    assert np.isclose(w.sum(), 1.0)


def test_cogact_warns_when_buffer_shorter_than_chunk(caplog):
    e = CogACTEnsemble(max_buffer_size=2, mode="cogact")
    with caplog.at_level(logging.WARNING):
        e.add_chunk(0, np.ones((10, 4)))  # chunk len 10 > buffer 2
    assert any("max_buffer_size" in r.message for r in caplog.records)
