"""Tests for the temporal action ensemble (single-strategy rewrite)."""
import numpy as np
import pytest

from ensemble import TemporalEnsemble, make_ensemble


def test_make_ensemble_off_returns_none():
    assert make_ensemble(False) is None


def test_make_ensemble_on_returns_instance():
    ens = make_ensemble(True, decay=0.5)
    assert isinstance(ens, TemporalEnsemble)
    assert ens.decay == 0.5


def test_single_chunk_returns_its_actions():
    ens = TemporalEnsemble(decay=1.0)
    chunk = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ens.add_chunk(0, chunk)
    np.testing.assert_allclose(ens.get_action(0), [1.0, 2.0])
    np.testing.assert_allclose(ens.get_action(1), [3.0, 4.0])
    np.testing.assert_allclose(ens.get_action(2), [5.0, 6.0])


def test_no_overlap_returns_none():
    ens = TemporalEnsemble()
    ens.add_chunk(0, np.zeros((2, 3)))
    assert ens.get_action(5) is None


def test_zero_decay_is_plain_average():
    ens = TemporalEnsemble(decay=0.0)
    ens.add_chunk(0, np.array([[0.0]]))        # predicts step 0 = 0
    ens.add_chunk(0, np.array([[4.0]]))        # another chunk predicts step 0 = 4
    np.testing.assert_allclose(ens.get_action(0), [2.0])  # mean(0, 4)


def test_decay_weights_older_predictions_more():
    # Two overlaps for step 1: oldest query (step 0) predicts 0, newer (step 1)
    # predicts 10. With decay>0 the oldest gets more weight -> result < midpoint.
    ens = TemporalEnsemble(decay=1.0)
    ens.add_chunk(0, np.array([[0.0], [0.0]]))   # covers steps 0,1 -> 0 at step 1
    ens.add_chunk(1, np.array([[10.0]]))          # covers step 1 -> 10
    out = float(ens.get_action(1)[0])
    assert 0.0 < out < 5.0


def test_overlap_not_evicted_before_consumed():
    # The old exact-step-pop buffer dropped overlaps early; here a long chunk and
    # a later chunk must both still contribute at a shared future step.
    ens = TemporalEnsemble(decay=0.0)
    ens.add_chunk(0, np.ones((5, 1)) * 2.0)   # predicts steps 0..4 = 2
    ens.get_action(0)                          # consume step 0
    ens.add_chunk(2, np.ones((1, 1)) * 4.0)   # predicts step 2 = 4
    np.testing.assert_allclose(ens.get_action(2), [3.0])  # mean(2, 4), overlap kept


def test_overlap_count_and_buffer_size():
    ens = TemporalEnsemble()
    ens.add_chunk(0, np.ones((3, 1)))
    ens.add_chunk(1, np.ones((3, 1)))
    assert ens.buffer_size() == 2
    assert ens.get_overlap_count(2) == 2   # both chunks predict step 2
    assert ens.get_overlap_count(0) == 1   # only first chunk predicts step 0


def test_reset_clears_state():
    ens = TemporalEnsemble()
    ens.add_chunk(0, np.ones((3, 1)))
    ens.get_action(0)
    ens.reset()
    assert ens.buffer_size() == 0
    assert ens.last_weights() is None
    assert ens.get_action(0) is None


def test_last_weights_sum_to_one():
    ens = TemporalEnsemble(decay=1.0)
    ens.add_chunk(0, np.ones((2, 1)))
    ens.add_chunk(1, np.ones((2, 1)))
    ens.get_action(1)
    w = ens.last_weights()
    assert w is not None
    assert pytest.approx(1.0) == float(w.sum())


def test_get_action_evicts_fully_past_chunks():
    ens = TemporalEnsemble()
    ens.add_chunk(0, np.ones((2, 1)))   # covers steps 0,1
    ens.get_action(5)                    # past the chunk -> evicted
    assert ens.buffer_size() == 0
