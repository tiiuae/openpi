"""Joint-space limit arithmetic for the Trossen client.

Pure numpy. Nothing here imports lerobot, scipy or touches hardware, so every
rule the arm is held to can be unit-tested offline (``tests/test_motion_limits.py``)
instead of on the robot.

Two ideas the rest of the client leans on:

**Scale a step, don't reshape it.** When a commanded step is too fast, the whole
step is scaled by one factor. Scaling each over-limit joint by its own ratio
(leaving the others at full speed) would slow the arm *and* send it somewhere
else: the commanded delta stops being parallel to the one the policy asked for,
so the tool leaves the intended path. One factor is a pure time scaling — same
line through joint space, traversed at the fastest safe speed, every joint
arriving together.

**Derive durations from distance.** A canned "4 seconds to home" is only safe
for the move it was tuned on. ``ramp_duration`` sizes a move from how far each
joint has to travel and how fast the driver says that joint may go.

**Profile.** ``scipy``'s PchipInterpolator over two points is *linear* (scipy
special-cases n==2 to a constant derivative), so the arm is commanded to jump
from rest to full speed and back: velocity is bounded but acceleration is not.
``minimum_jerk_progress`` is used instead — zero velocity and zero acceleration
at both ends, peaking at 1.875x the average speed in the middle, which is the
factor durations are derived against.
"""

from __future__ import annotations

import numpy as np

# Peak/average speed ratio of the minimum-jerk profile below: max of
# s'(t) = 30t^2 - 60t^3 + 30t^4 is 1.875, at t = 0.5.
MIN_JERK_PEAK_FACTOR = 1.875

# Derived durations target this fraction of the driver's own velocity limit, so
# tracking error or a calibration difference cannot push a joint over the edge.
DEFAULT_PEAK_FRACTION = 0.7

# Below this, a position-limit overshoot is floating-point or calibration noise
# rather than a meaningful commanded position (a quantile-normalized gripper
# output landing ~1e-4 past the stop, a held pose echoing the arm's own measured
# position ~1e-6 past it). Clamping always applies at full precision; this only
# decides whether it is worth saying out loud.
CLAMP_WARNING_TOLERANCE = 1e-3


class JointLimitError(ValueError):
    """A limit table that cannot be trusted to constrain motion.

    Raised rather than returned: a table that fails these checks must stop the
    arm from moving, not silently disable the check that was protecting it.
    """


def validate_position_limits(limits, action_dim: int) -> np.ndarray:
    """Return *limits* as a checked ``(action_dim, 2)`` array of ``[min, max]``.

    ``np.clip`` with ``min > max`` silently pins the joint to ``max`` on every
    step regardless of what was commanded, which looks exactly like a broken
    checkpoint, so an inverted pair is an error here rather than a warning.
    """
    if limits is None:
        raise JointLimitError("no position limits available")
    array = np.asarray(limits, dtype=float)
    if array.shape != (action_dim, 2):
        raise JointLimitError(f"expected ({action_dim}, 2) position limits, got {array.shape}")
    if not np.all(np.isfinite(array)):
        bad = np.where(~np.isfinite(array).all(axis=1))[0].tolist()
        raise JointLimitError(f"position limits are not finite on joints {bad}")
    inverted = np.where(array[:, 0] > array[:, 1])[0]
    if inverted.size:
        pairs = ", ".join(f"joint {i}: [{array[i, 0]:.6g}, {array[i, 1]:.6g}]" for i in inverted)
        raise JointLimitError(f"position limits have min > max ({pairs})")
    return array


def validate_velocity_limits(limits, action_dim: int) -> np.ndarray:
    """Return *limits* as a checked ``(action_dim,)`` array of max speeds."""
    if limits is None:
        raise JointLimitError("no velocity limits available")
    array = np.asarray(limits, dtype=float)
    if array.shape != (action_dim,):
        raise JointLimitError(f"expected ({action_dim},) velocity limits, got {array.shape}")
    if not np.all(np.isfinite(array)):
        bad = np.where(~np.isfinite(array))[0].tolist()
        raise JointLimitError(f"velocity limits are not finite on joints {bad}")
    non_positive = np.where(array <= 0.0)[0]
    if non_positive.size:
        raise JointLimitError(f"velocity limits are not positive on joints {non_positive.tolist()}")
    return array


def scale_to_velocity_limits(
    last: np.ndarray,
    target: np.ndarray,
    dt: float,
    velocity_limits: np.ndarray,
) -> tuple[np.ndarray, float, list[int]]:
    """Slow one commanded step down to the fastest speed the driver allows.

    Returns ``(sent, factor, over)`` where *sent* is ``last + (target - last) *
    factor``, *factor* is in ``(0, 1]``, and *over* lists the joints that were
    over their limit before scaling. The returned delta is always parallel to
    the requested one, so the arm keeps the path and only loses speed.
    """
    delta = np.asarray(target, dtype=float) - np.asarray(last, dtype=float)
    speed = np.abs(delta) / dt
    over = np.where(speed > velocity_limits)[0]
    if over.size == 0:
        return np.asarray(target, dtype=float), 1.0, []
    factor = float(np.min(velocity_limits[over] / speed[over]))
    return np.asarray(last, dtype=float) + delta * factor, factor, over.tolist()


def clamp_to_position_limits(pose: np.ndarray, position_limits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clamp *pose* into range. Returns ``(clamped, deviation_per_joint)``."""
    pose = np.asarray(pose, dtype=float)
    clamped = np.clip(pose, position_limits[:, 0], position_limits[:, 1])
    return clamped, np.abs(pose - clamped)


def ramp_duration(
    start: np.ndarray,
    goal: np.ndarray,
    velocity_limits: np.ndarray | None,
    *,
    minimum: float,
    peak_fraction: float = DEFAULT_PEAK_FRACTION,
) -> float:
    """Seconds a minimum-jerk move from *start* to *goal* should take.

    Sized so the profile's *peak* speed stays at *peak_fraction* of each joint's
    limit. A move of a few millimetres takes *minimum*; a full-workspace move
    takes as long as its slowest joint needs.

    There is deliberately no upper bound. A duration cap that shortens a move
    below what the velocity limit allows would quietly put back the over-speed
    the derivation exists to prevent — a cap that overrides a safety limit is
    not a safety mechanism. A slow joint simply takes a long time, and the
    caller is expected to say so when the number is surprising.

    With no velocity limits there is nothing to derive from, so *minimum* is
    returned: the caller's tuned floor, i.e. exactly the old fixed-duration
    behaviour, which is only as safe as the move it was tuned for.
    """
    if velocity_limits is None:
        return minimum
    travel = np.abs(np.asarray(goal, dtype=float) - np.asarray(start, dtype=float))
    needed = float(np.max(MIN_JERK_PEAK_FACTOR * travel / (peak_fraction * velocity_limits)))
    return max(needed, minimum)


def minimum_jerk_progress(tau: float) -> float:
    """Fraction of the move completed at normalised time *tau* in ``[0, 1]``.

    ``10t^3 - 15t^4 + 6t^5``: starts and ends at rest with zero acceleration, so
    the arm is never commanded to change speed instantaneously.
    """
    tau = min(max(tau, 0.0), 1.0)
    return tau * tau * tau * (10.0 - 15.0 * tau + 6.0 * tau * tau)


def minimum_jerk_pose(start: np.ndarray, goal: np.ndarray, elapsed: float, duration: float) -> np.ndarray:
    """Interpolated pose *elapsed* seconds into a *duration*-long move."""
    if duration <= 0:
        return np.asarray(goal, dtype=float)
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    return start + (goal - start) * minimum_jerk_progress(elapsed / duration)


def safe_oscillation_period(
    amplitude: float,
    period: float,
    joint_indices: list[int],
    velocity_limits: np.ndarray | None,
    peak_fraction: float = DEFAULT_PEAK_FRACTION,
) -> float:
    """Stretch an oscillation's period until its peak speed is within limits.

    ``A*sin(2*pi*t/P)`` peaks at ``A*2*pi/P``, so a period tuned for one joint is
    not automatically safe on another with a lower limit.
    """
    if velocity_limits is None or not joint_indices:
        return period
    slowest = float(np.min(velocity_limits[joint_indices]))
    needed = 2.0 * np.pi * abs(amplitude) / (peak_fraction * slowest)
    return float(max(period, needed))
