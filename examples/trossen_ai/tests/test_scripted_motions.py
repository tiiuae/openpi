"""Offline tests for the canned motions.

``ScriptedMotions`` takes ``get_pose``/``send_pose`` callables, so the whole
motion path can be exercised against a fake arm that just records what it was
told to do. numpy and pytest only — no robot, no lerobot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripted_motions as sm  # noqa: E402

DIM = sm.BIMANUAL_DIM
CONTROL_HZ = 25
DT = 1.0 / CONTROL_HZ


class FakeClock:
    """Advances only when the code under test sleeps.

    The motion loops stream targets in real time, so a derived 20 s ramp would
    really take 20 s to test. Time is driven by the loop itself instead, which
    makes the tests instant and deterministic — and is the same fixture the
    control-loop tests need.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 0.0)


@pytest.fixture(autouse=True)
def fake_clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(sm, "time", clock)
    return clock


class FakeArm:
    """Records every commanded pose. Reports the last one as its measured pose,
    which is what a perfectly-tracking arm would do."""

    def __init__(self, start: np.ndarray | None = None):
        self.pose = np.zeros(DIM) if start is None else np.asarray(start, dtype=float)
        self.sent: list[np.ndarray] = []

    def get_pose(self) -> np.ndarray:
        return self.pose.copy()

    def send_pose(self, pose: np.ndarray) -> None:
        pose = np.asarray(pose, dtype=float).copy()
        self.sent.append(pose)
        self.pose = pose

    @property
    def commanded_speeds(self) -> np.ndarray:
        """Per-joint speed implied by each commanded step, shape (n-1, DIM)."""
        return np.abs(np.diff(np.array(self.sent), axis=0)) / DT


def _motions(arm: FakeArm, velocity_limits: np.ndarray | None, **kwargs) -> sm.ScriptedMotions:
    return sm.ScriptedMotions(
        get_pose=arm.get_pose,
        send_pose=arm.send_pose,
        control_frequency=CONTROL_HZ,
        joint_limits=kwargs.pop("joint_limits", np.tile(np.array([-4.0, 4.0]), (DIM, 1))),
        velocity_limits=velocity_limits,
        **kwargs,
    )


@pytest.mark.parametrize("limit", [0.2, 0.5, 2.0])
def test_home_never_commands_a_step_over_the_velocity_limit(limit):
    """The bug this guards: a fixed 4 s home from a far pose implies a speed the
    driver refuses, and the firmware fault was the only thing catching it."""
    arm = FakeArm(np.full(DIM, 2.5))  # a long way from the home pose
    _motions(arm, np.full(DIM, limit)).home(sm.ARMS)
    assert arm.commanded_speeds.max() <= limit + 1e-9


def test_a_longer_move_takes_longer():
    near = FakeArm(sm.HOME_ARM_POSE.tolist() * 2)
    far = FakeArm(np.full(DIM, 2.5))
    limits = np.full(DIM, 0.5)
    _motions(near, limits).home(sm.ARMS)
    _motions(far, limits).home(sm.ARMS)
    assert len(far.sent) > len(near.sent)


def test_duration_floor_is_respected_for_a_tiny_move():
    arm = FakeArm(sm.HOME_ARM_POSE.tolist() * 2)
    _motions(arm, np.full(DIM, 10.0)).home(sm.ARMS)
    assert len(arm.sent) >= sm.MIN_HOME_DURATION_S * CONTROL_HZ * 0.9


def test_home_ends_on_the_home_pose_for_the_named_arm_only():
    arm = FakeArm(np.full(DIM, 1.0))
    _motions(arm, np.full(DIM, 1.0)).home(("left",))
    np.testing.assert_allclose(arm.sent[-1][:sm.JOINTS_PER_ARM], sm.HOME_ARM_POSE)
    np.testing.assert_allclose(arm.sent[-1][sm.JOINTS_PER_ARM :], np.full(sm.JOINTS_PER_ARM, 1.0))


def test_sleep_goes_through_home_before_folding_down():
    """A direct ramp to zero from an arbitrary pose can drag the elbow through
    the workspace; home is the known-safe waypoint."""
    arm = FakeArm(np.full(DIM, 2.0))
    _motions(arm, np.full(DIM, 1.0)).sleep(sm.ARMS)
    poses = np.array(arm.sent)
    reached_home = np.isclose(poses[:, : sm.JOINTS_PER_ARM], sm.HOME_ARM_POSE, atol=1e-6).all(axis=1)
    assert reached_home.any(), "sleep never passed through the home pose"
    assert not reached_home[-1], "sleep ended at home instead of the folded pose"
    np.testing.assert_allclose(arm.sent[-1], np.zeros(DIM), atol=1e-9)


def test_oscillation_period_stretches_for_a_slow_joint():
    slow = np.full(DIM, 5.0)
    slow[sm.WRIST_ROTATE_IDX] = 0.05
    arm = FakeArm()
    _motions(arm, slow).twist_wrist(("left",))
    assert arm.commanded_speeds.max() <= 0.05 + 1e-9


def test_oscillation_returns_to_its_starting_pose():
    start = np.full(DIM, 0.3)
    arm = FakeArm(start)
    _motions(arm, np.full(DIM, 2.0)).wave(sm.ARMS)
    np.testing.assert_allclose(arm.sent[-1], start, atol=1e-9)


def test_every_commanded_pose_stays_inside_the_position_limits():
    limits = np.tile(np.array([-0.2, 0.2]), (DIM, 1))
    arm = FakeArm()
    _motions(arm, np.full(DIM, 1.0), joint_limits=limits).home(sm.ARMS)
    poses = np.array(arm.sent)
    assert np.all(poses >= limits[:, 0] - 1e-9)
    assert np.all(poses <= limits[:, 1] + 1e-9)


def test_commands_for_a_disabled_arm_are_refused_not_retargeted():
    arm = FakeArm()
    motions = _motions(arm, np.full(DIM, 1.0), enabled_arms=("left",))
    assert motions.run(sm.Command("home", ("right",))) is False
    assert arm.sent == []


def test_hold_moves_nothing():
    arm = FakeArm()
    assert _motions(arm, np.full(DIM, 1.0)).run(sm.Command("hold", sm.ARMS)) is True
    assert arm.sent == []


def test_without_velocity_limits_motions_still_run():
    """Falling back to the fixed floors is worse, but it must not crash."""
    arm = FakeArm(np.full(DIM, 1.0))
    _motions(arm, None).home(sm.ARMS)
    np.testing.assert_allclose(arm.sent[-1][: sm.JOINTS_PER_ARM], sm.HOME_ARM_POSE)


# -- the command parser: a task instruction must never reach the gripper --------


@pytest.mark.parametrize(
    "text",
    [
        "close the drawer",
        "move the arm to the left",
        "open the fridge and take the cup",
        "pick up the blue cup and place it in the orange basket",
    ],
)
def test_task_instructions_are_not_parsed_as_commands(text):
    assert sm.parse_command(text) is None


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("home", "home"),
        ("return to the home position", "home"),
        ("stop", "hold"),
        ("freeze", "hold"),
        ("open", "open_gripper"),
        ("close left gripper", "close_gripper"),
        ("quit", "quit"),
    ],
)
def test_keyword_commands_are_recognised(text, name):
    command = sm.parse_command(text)
    assert command is not None and command.name == name
