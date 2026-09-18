"""Validation for what comes back from the policy server.

Pure numpy, no transport: the same check runs in the synchronous control path,
in the async inference worker, and offline in ``replay_request.py``, so a reply
that the live client would refuse cannot pass a compatibility test.

The client used to take the server at its word — ``response["actions"][:, :D]``
and then index row by row. Nothing checked rank, horizon, dtype or finiteness,
so an empty array still satisfied the "first result arrived" wait while
providing no action, a 3-D array failed much later with an index error, and
NaN travelled straight into the ensemble average and out to the motors.
``np.clip`` does not remove NaN, so the position clamp was no defence.

A reply that fails these checks is a fault of the inference source, not an
action: callers drop it, which leaves the ensemble starved, which makes the
control loop hold its pose. There is no repair path that invents an action.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

# A horizon longer than this is a schema mismatch (a batch axis mistaken for
# time, a transposed array), not a genuinely long chunk: at 25 Hz even 1000
# steps is 40 seconds of open-loop motion from a single observation.
MAX_PLAUSIBLE_HORIZON = 1000


class PolicyReplyError(ValueError):
    """A reply that cannot be turned into actions to execute."""


def validate_actions(
    reply: object,
    *,
    action_dim: int,
    max_horizon: int = MAX_PLAUSIBLE_HORIZON,
) -> np.ndarray:
    """Return the reply's action chunk as a checked ``(T, action_dim)`` float array.

    Raises ``PolicyReplyError`` with a message naming what was wrong, which is
    what the operator needs to fix the server rather than the robot.
    """
    if not isinstance(reply, Mapping):
        raise PolicyReplyError(f"expected a dict from the policy server, got {type(reply).__name__}")
    if "actions" not in reply:
        keys = sorted(str(key) for key in reply)
        raise PolicyReplyError(f"reply has no 'actions' key (got keys: {keys})")

    actions = reply["actions"]
    try:
        array = np.asarray(actions)
    except Exception as error:  # pragma: no cover - asarray is extremely permissive
        raise PolicyReplyError(f"'actions' is not array-like: {error}") from error

    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise PolicyReplyError(f"'actions' must be numeric, got dtype {array.dtype}")
    if array.ndim != 2:
        raise PolicyReplyError(
            f"'actions' must be 2-D (horizon, action_dim), got shape {array.shape}. "
            "A leading batch axis must be squeezed by the server, not by the robot."
        )

    horizon, width = array.shape
    if horizon < 1:
        raise PolicyReplyError(f"'actions' is empty (shape {array.shape}) — no action to execute")
    if horizon > max_horizon:
        raise PolicyReplyError(f"'actions' horizon {horizon} exceeds the plausible maximum {max_horizon}")
    if width < action_dim:
        raise PolicyReplyError(f"'actions' has {width} columns, need at least {action_dim} for this robot")

    array = array[:, :action_dim].astype(float, copy=True)
    if not np.all(np.isfinite(array)):
        rows = np.unique(np.where(~np.isfinite(array))[0]).tolist()
        raise PolicyReplyError(f"'actions' contains non-finite values on rows {rows[:8]} — clamping cannot fix NaN")
    return array


def server_timing(reply: object) -> dict:
    """The server's own timing block, or an empty dict when it sends none.

    openpi's websocket server reports how long the model itself took; other
    backends may not. Absent timing is normal, never an error.
    """
    if isinstance(reply, Mapping):
        timing = reply.get("server_timing")
        if isinstance(timing, Mapping):
            return dict(timing)
    return {}
