"""Offline tests for policy-reply validation.

Table-driven over the malformed replies a backend can actually send. numpy and
pytest only — no transport, no server.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policy_reply import (  # noqa: E402
    MAX_PLAUSIBLE_HORIZON,
    PolicyReplyError,
    validate_actions,
)

DIM = 14


def _chunk(horizon: int = 25, width: int = DIM) -> np.ndarray:
    return np.arange(horizon * width, dtype=np.float32).reshape(horizon, width)


def test_a_good_reply_comes_back_as_float_of_the_right_width():
    actions = validate_actions({"actions": _chunk()}, action_dim=DIM)
    assert actions.shape == (25, DIM)
    assert actions.dtype == np.float64


def test_extra_columns_are_truncated():
    """Servers padded to a wider action space are supported; the robot takes
    the leading columns it has joints for."""
    actions = validate_actions({"actions": _chunk(10, 32)}, action_dim=DIM)
    assert actions.shape == (10, DIM)


def test_the_returned_array_does_not_alias_the_reply():
    reply = {"actions": _chunk(4)}
    actions = validate_actions(reply, action_dim=DIM)
    actions[0, 0] = 999.0
    assert reply["actions"][0, 0] != 999.0


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ("a traceback string", "expected a dict"),
        (None, "expected a dict"),
        ({"result": _chunk()}, "no 'actions' key"),
        ({"actions": np.zeros((0, DIM))}, "is empty"),
        ({"actions": np.zeros((25, 13))}, "13 columns"),
        ({"actions": np.zeros((1, 25, DIM))}, "must be 2-D"),
        ({"actions": np.zeros(DIM)}, "must be 2-D"),
        ({"actions": np.zeros((MAX_PLAUSIBLE_HORIZON + 1, DIM))}, "exceeds the plausible maximum"),
        ({"actions": np.zeros((5, DIM), dtype=bool)}, "must be numeric"),
        ({"actions": np.full((5, DIM), "x")}, "must be numeric"),
    ],
)
def test_malformed_replies_are_refused(reply, message):
    with pytest.raises(PolicyReplyError) as excinfo:
        validate_actions(reply, action_dim=DIM)
    assert message in str(excinfo.value)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_refused(bad):
    """np.clip does not remove NaN, so the position clamp downstream is no
    defence — a NaN has to be stopped here or it reaches the motors."""
    chunk = _chunk(6)
    chunk[3, 7] = bad
    with pytest.raises(PolicyReplyError) as excinfo:
        validate_actions({"actions": chunk}, action_dim=DIM)
    assert "non-finite" in str(excinfo.value)
    assert "[3]" in str(excinfo.value)


def test_a_non_finite_value_outside_the_used_columns_is_ignored():
    """Only the columns this robot executes matter; a padded server column full
    of NaN is the server's business."""
    chunk = _chunk(6, 20)
    chunk[0, 18] = np.nan
    assert validate_actions({"actions": chunk}, action_dim=DIM).shape == (6, DIM)


def test_a_single_step_chunk_is_valid():
    assert validate_actions({"actions": _chunk(1)}, action_dim=DIM).shape == (1, DIM)


def test_integer_actions_are_accepted_and_converted():
    actions = validate_actions({"actions": np.ones((3, DIM), dtype=np.int32)}, action_dim=DIM)
    assert actions.dtype == np.float64
