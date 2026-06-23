"""Unit test for the hold-last-action fallback helper (no hardware)."""
import numpy as np

from action_fallback import HoldLastAction


def test_first_call_with_none_returns_none():
    h = HoldLastAction()
    assert h.resolve(None) is None  # nothing to hold yet -> caller skips


def test_real_action_is_remembered_and_repeated():
    h = HoldLastAction()
    a = np.array([1.0, 2.0, 3.0])
    assert np.allclose(h.resolve(a), a)        # passes through, stored
    assert np.allclose(h.resolve(None), a)     # None -> repeat last
    b = np.array([4.0, 5.0, 6.0])
    assert np.allclose(h.resolve(b), b)        # updates last
    assert np.allclose(h.resolve(None), b)


def test_reset_clears_held_action():
    h = HoldLastAction()
    h.resolve(np.array([1.0, 2.0]))
    h.reset()
    assert h.resolve(None) is None
