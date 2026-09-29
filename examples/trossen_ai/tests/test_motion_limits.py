"""Offline tests for the joint-limit arithmetic.

numpy and pytest only — no lerobot, no robot, no policy server — so these run
anywhere and are meant to be the gate that hardware-facing changes pass first.

    uv run pytest tests/            # from examples/trossen_ai
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motion_limits import (  # noqa: E402
    DEFAULT_PEAK_FRACTION,
    MIN_JERK_PEAK_FACTOR,
    JointLimitError,
    clamp_to_position_limits,
    minimum_jerk_pose,
    minimum_jerk_progress,
    ramp_duration,
    safe_oscillation_period,
    scale_to_velocity_limits,
    validate_position_limits,
    validate_velocity_limits,
)

DIM = 14
DT = 1.0 / 25.0


def _velocity_limits(value: float = 1.0) -> np.ndarray:
    return np.full(DIM, value)


def _position_limits(low: float = -3.0, high: float = 3.0) -> np.ndarray:
    return np.tile(np.array([low, high]), (DIM, 1))


# -- validation ----------------------------------------------------------------


def test_position_limits_accepts_a_good_table():
    limits = _position_limits()
    assert validate_position_limits(limits, DIM).shape == (DIM, 2)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda t: None, "no position limits"),
        (lambda t: t[:5], "expected (14, 2)"),
        (lambda t: _with(t, (3, 1), np.nan), "not finite"),
        (lambda t: _with(t, (6, 0), 10.0), "min > max"),
    ],
)
def test_position_limits_rejects_bad_tables(mutate, message):
    """A table that cannot constrain motion must raise, never quietly disable
    the clamp: `np.clip` with min > max pins the joint to max on every step."""
    with pytest.raises(JointLimitError) as excinfo:
        validate_position_limits(mutate(_position_limits()), DIM)
    assert message in str(excinfo.value)


def _with(table: np.ndarray, index, value):
    table = table.copy()
    table[index] = value
    return table


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        (None, "no velocity limits"),
        (np.ones(7), "expected (14,)"),
        (_with(np.ones(DIM), 2, np.inf), "not finite"),
        (_with(np.ones(DIM), 9, 0.0), "not positive"),
        (_with(np.ones(DIM), 9, -1.0), "not positive"),
    ],
)
def test_velocity_limits_rejects_bad_tables(limits, message):
    with pytest.raises(JointLimitError) as excinfo:
        validate_velocity_limits(limits, DIM)
    assert message in str(excinfo.value)


# -- velocity scaling ----------------------------------------------------------


def test_a_step_within_limits_is_passed_through_untouched():
    last = np.zeros(DIM)
    target = np.full(DIM, 0.01)  # 0.25 rad/s at 25 Hz, limit 1.0
    sent, factor, over = scale_to_velocity_limits(last, target, DT, _velocity_limits())
    assert factor == 1.0
    assert over == []
    np.testing.assert_array_equal(sent, target)


def test_an_over_fast_step_is_scaled_to_exactly_the_limit():
    last = np.zeros(DIM)
    target = np.zeros(DIM)
    target[3] = 1.0  # 25 rad/s at 25 Hz against a 1.0 rad/s limit
    sent, factor, over = scale_to_velocity_limits(last, target, DT, _velocity_limits())
    assert over == [3]
    assert factor == pytest.approx(1.0 / 25.0)
    assert abs(sent[3] - last[3]) / DT == pytest.approx(1.0)


def test_scaling_keeps_the_step_parallel_so_the_arm_stays_on_its_path():
    """The whole point of one factor over per-joint factors: a slowed elbow with
    a full-speed shoulder is a different move, not a slower one."""
    rng = np.random.default_rng(0)
    limits = rng.uniform(0.2, 2.0, DIM)
    for _ in range(200):
        last = rng.uniform(-1.0, 1.0, DIM)
        delta = rng.uniform(-2.0, 2.0, DIM)
        sent, factor, _ = scale_to_velocity_limits(last, last + delta, DT, limits)
        sent_delta = sent - last
        assert 0.0 < factor <= 1.0
        # Parallel: every component shrank by the same factor.
        np.testing.assert_allclose(sent_delta, delta * factor, atol=1e-12)
        # And the result is actually within limits.
        assert np.all(np.abs(sent_delta) / DT <= limits + 1e-9)


def test_a_joint_at_its_limit_is_not_slowed():
    last = np.zeros(DIM)
    target = np.zeros(DIM)
    target[0] = 1.0 * DT  # exactly the limit
    _, factor, over = scale_to_velocity_limits(last, target, DT, _velocity_limits())
    assert factor == 1.0
    assert over == []


# -- position clamping ---------------------------------------------------------


def test_clamping_reports_deviation_per_joint():
    pose = np.zeros(DIM)
    pose[2] = 5.0
    clamped, deviation = clamp_to_position_limits(pose, _position_limits(-3.0, 3.0))
    assert clamped[2] == 3.0
    assert deviation[2] == pytest.approx(2.0)
    assert deviation.sum() == pytest.approx(2.0)


def test_clamping_leaves_in_range_joints_alone():
    pose = np.linspace(-2.0, 2.0, DIM)
    clamped, deviation = clamp_to_position_limits(pose, _position_limits())
    np.testing.assert_array_equal(clamped, pose)
    assert not deviation.any()


# -- durations and profile -----------------------------------------------------


def test_a_tiny_move_takes_the_minimum():
    start = np.zeros(DIM)
    assert ramp_duration(start, start + 1e-4, _velocity_limits(), minimum=0.3) == pytest.approx(0.3)


def test_a_long_move_is_never_capped_below_what_the_limit_allows():
    """A duration cap that shortens a move below its velocity limit would put
    back the over-speed the derivation exists to prevent, so there is no cap."""
    start = np.zeros(DIM)
    slow = _velocity_limits(0.2)
    duration = ramp_duration(start, start + 2.5, slow, minimum=4.0)
    assert duration > 30.0
    peak = MIN_JERK_PEAK_FACTOR * 2.5 / duration
    assert peak <= 0.2 * DEFAULT_PEAK_FRACTION + 1e-9


@pytest.mark.parametrize("seed", range(8))
def test_a_derived_duration_keeps_the_peak_speed_under_the_limit(seed):
    """Sample the profile and check the fastest commanded step, which is what
    the driver actually sees."""
    rng = np.random.default_rng(seed)
    limits = rng.uniform(0.1, 2.0, DIM)
    start = rng.uniform(-1.0, 1.0, DIM)
    goal = start + rng.uniform(-3.0, 3.0, DIM)
    duration = ramp_duration(start, goal, limits, minimum=0.3)

    times = np.arange(0.0, duration, DT)
    poses = np.array([minimum_jerk_pose(start, goal, t, duration) for t in times])
    speeds = np.abs(np.diff(poses, axis=0)) / DT
    assert np.all(speeds.max(axis=0) <= limits * DEFAULT_PEAK_FRACTION + 1e-6)


def test_minimum_jerk_starts_and_ends_at_rest():
    progress = np.array([minimum_jerk_progress(t) for t in np.linspace(0.0, 1.0, 1001)])
    assert progress[0] == 0.0
    assert progress[-1] == pytest.approx(1.0)
    assert np.all(np.diff(progress) >= -1e-12)  # monotonic
    speed = np.diff(progress) * 1000.0
    # At rest at both ends, fastest in the middle, at the documented ratio.
    assert speed[0] < 0.01
    assert speed[-1] < 0.01
    assert speed.max() == pytest.approx(MIN_JERK_PEAK_FACTOR, rel=1e-3)


def test_progress_is_clamped_outside_the_window():
    assert minimum_jerk_progress(-1.0) == 0.0
    assert minimum_jerk_progress(2.0) == 1.0


def test_a_zero_length_move_lands_on_the_goal():
    goal = np.full(DIM, 0.5)
    np.testing.assert_array_equal(minimum_jerk_pose(np.zeros(DIM), goal, 0.0, 0.0), goal)


def test_oscillation_period_is_stretched_for_a_slow_joint():
    limits = _velocity_limits(2.0)
    limits[5] = 0.1
    # 0.5 rad at a 2 s period peaks at 1.57 rad/s, far over the 0.1 rad/s joint.
    stretched = safe_oscillation_period(0.5, 2.0, [5], limits)
    assert stretched > 2.0
    assert 2.0 * np.pi * 0.5 / stretched == pytest.approx(0.1 * DEFAULT_PEAK_FRACTION)


def test_oscillation_period_is_left_alone_when_already_safe():
    assert safe_oscillation_period(0.01, 2.0, [0], _velocity_limits(5.0)) == 2.0


def test_missing_limits_fall_back_to_the_tuned_floor():
    """Nothing to derive from, so the caller's floor stands — i.e. exactly the
    old fixed-duration behaviour, and no pretence of a check that isn't there."""
    start = np.zeros(DIM)
    assert ramp_duration(start, start + 100.0, None, minimum=4.0) == 4.0
    assert safe_oscillation_period(1.0, 2.0, [0], None) == 2.0
