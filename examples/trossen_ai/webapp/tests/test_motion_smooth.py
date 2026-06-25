"""Off-robot tests for the smooth-streaming command path and firmware guard.

A FakeArm captures set_all_positions calls; a FakeRobot exposes left_arm /
right_arm and the observation/joint-feature surface RobotController uses.
"""
import numpy as np
import pytest

import robot_control
from robot_control import RobotController, limit_joint_velocity, send_action_smooth


class FakeArm:
    def __init__(self, names):
        self.config = type("C", (), {"joint_names": names})()
        self.driver = self
        self.calls = []

    def set_all_positions(self, goal_positions, goal_time, blocking, goal_feedforward_velocities=None):
        self.calls.append(dict(goal=list(goal_positions), goal_time=goal_time,
                               blocking=blocking, ff=list(goal_feedforward_velocities)))


class FakeRobot:
    def __init__(self):
        names = [f"j{i}.pos" for i in range(7)]
        self.left_arm = FakeArm(names)
        self.right_arm = FakeArm(names)
        self._obs = {f"left_j{i}.pos": 0.0 for i in range(7)}
        self._obs.update({f"right_j{i}.pos": 0.0 for i in range(7)})

    @property
    def _joint_ft(self):
        return {k: float for k in self._obs}

    def get_observation(self):
        return dict(self._obs)

    def send_action(self, action_dict):
        self._sent = action_dict

    def disconnect(self):
        ...


def test_send_action_smooth_sets_feedforward_velocity():
    robot = FakeRobot()
    target = np.arange(14, dtype=float) * 0.1  # left 0..0.6, right 0.7..1.3
    send_action_smooth(robot, target, dt=0.04)
    assert len(robot.left_arm.calls) == 1
    call = robot.left_arm.calls[0]
    assert call["goal_time"] == pytest.approx(0.04)
    assert call["blocking"] is False
    # feedforward velocity == (target - current) / dt, current is zeros
    assert call["ff"][0] == pytest.approx(0.0)
    assert call["ff"][1] == pytest.approx(0.1 / 0.04)


def test_execute_action_smooth_path_used_when_enabled():
    robot = FakeRobot()
    ctrl = RobotController(robot, control_frequency=25, test_mode="autonomous", smooth_streaming=True)
    assert ctrl.execute_action(np.zeros(14)) is True
    assert robot.left_arm.calls, "smooth path should call driver.set_all_positions"


def test_execute_action_firmware_error_triggers_sleep(monkeypatch):
    robot = FakeRobot()
    ctrl = RobotController(robot, control_frequency=25, test_mode="autonomous")

    def boom(action_dict):
        raise RuntimeError("firmware: joint velocity exceeded")

    monkeypatch.setattr(robot, "send_action", boom)
    slept = {"n": 0}
    monkeypatch.setattr(ctrl, "move_to_sleep_position", lambda duration=10.0: slept.__setitem__("n", slept["n"] + 1))
    assert ctrl.execute_action(np.zeros(14)) is False
    assert slept["n"] == 1


def test_home_position_constant_shape():
    assert robot_control.HOME_POSITION.shape == (14,)
    assert robot_control.HOME_POSITION[1] == pytest.approx(np.pi / 3)


def test_limit_joint_velocity_clamps_branch_flip():
    # An IK branch-flip: right joint-3 (index 10) jumps 0.42 rad in one 20ms step
    # (~21 rad/s), well over the firmware limit. It must be clamped to max_speed*dt.
    prev = np.zeros(14)
    target = np.zeros(14)
    target[10] = 0.42
    out = limit_joint_velocity(prev, target, dt=0.02, max_speed=3.0)
    assert out[10] == pytest.approx(0.06)  # 3.0 rad/s * 0.02 s
    assert abs(out[10] - prev[10]) <= 3.0 * 0.02 + 1e-9


def test_limit_joint_velocity_passes_small_step():
    # Normal recorded motion (well under the cap) is unchanged.
    prev = np.zeros(14)
    target = np.full(14, 0.01)
    out = limit_joint_velocity(prev, target, dt=0.02, max_speed=3.0)
    assert np.allclose(out, target)


def test_limit_joint_velocity_caps_every_joint_velocity():
    rng = np.random.default_rng(0)
    prev = rng.standard_normal(14)
    target = prev + rng.standard_normal(14)  # arbitrary, some over-limit deltas
    dt = 0.02
    out = limit_joint_velocity(prev, target, dt, max_speed=3.0)
    assert np.all(np.abs(out - prev) <= 3.0 * dt + 1e-9)
