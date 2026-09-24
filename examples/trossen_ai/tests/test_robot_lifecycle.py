"""Offline tests for connect/release without unintended motion.

Fakes stand in for the driver and cameras and record every call, so the
question "did anything command the arm to move?" is answered by inspection
rather than by watching the hardware. No robot, no lerobot.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# robot_lifecycle imports trossen_arm for Mode.idle; stub it so the module is
# importable in a plain numpy/pytest environment.
if "trossen_arm" not in sys.modules:
    stub = types.ModuleType("trossen_arm")
    stub.Mode = types.SimpleNamespace(idle="idle", position="position")
    sys.modules["trossen_arm"] = stub

import robot_lifecycle  # noqa: E402


class FakeDriver:
    def __init__(self, log: list, name: str, fail_cleanup: bool = False):
        self.log = log
        self.name = name
        self.fail_cleanup = fail_cleanup
        self.modes: list = []

    def set_all_modes(self, mode):
        self.modes.append(mode)
        self.log.append(f"{self.name}:mode")

    def set_all_positions(self, positions, goal_time=2.0, blocking=True):
        self.log.append(f"{self.name}:MOVE")

    def cleanup(self, reboot_controller=False):
        if self.fail_cleanup:
            raise RuntimeError("controller not responding")
        self.log.append(f"{self.name}:cleanup")


class FakeCamera:
    def __init__(self, log: list, name: str, fail: bool = False):
        self.log = log
        self.name = name
        self.fail = fail

    def connect(self):
        self.log.append(f"cam {self.name}:connect")

    def disconnect(self):
        if self.fail:
            raise RuntimeError("device busy")
        self.log.append(f"cam {self.name}:disconnect")


class FakeArm:
    """Mirrors WidowXAIFollower: connect() ends by calling configure(), which
    sets position mode and drives to the staged pose."""

    def __init__(self, log: list, name: str, fail_cleanup: bool = False):
        self.log = log
        self.name = name
        self.driver = FakeDriver(log, name, fail_cleanup)
        self.cameras = {}

    def connect(self, calibrate: bool = True):
        self.log.append(f"{self.name}:configure-driver")
        self.configure()

    def configure(self):
        self.driver.set_all_modes("position")
        self.driver.set_all_positions([0.0] * 7)

    def disconnect(self):
        self.configure()
        self.driver.set_all_positions([0.0] * 7)
        self.driver.cleanup()


class FakeRobot:
    def __init__(self, fail_left_cleanup: bool = False, fail_camera: bool = False):
        self.log: list[str] = []
        self.left_arm = FakeArm(self.log, "left", fail_left_cleanup)
        self.right_arm = FakeArm(self.log, "right")
        self.cameras = {
            "cam_high": FakeCamera(self.log, "cam_high", fail_camera),
            "cam_wrist": FakeCamera(self.log, "cam_wrist"),
        }

    def connect(self, calibrate: bool = True):
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)
        for camera in self.cameras.values():
            camera.connect()

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()
        for camera in self.cameras.values():
            camera.disconnect()


# -- connect -------------------------------------------------------------------


def test_the_normal_connect_does_move_the_arms():
    """Establishes the baseline the fix is measured against."""
    robot = FakeRobot()
    robot.connect()
    assert "left:MOVE" in robot.log
    assert "right:MOVE" in robot.log


def test_motion_free_connect_issues_no_position_goal():
    """--mode test says "no movement"; this is what makes that true."""
    robot = FakeRobot()
    robot_lifecycle.connect_without_motion(robot)
    assert not [entry for entry in robot.log if "MOVE" in entry]


def test_motion_free_connect_brakes_both_arms():
    """Idle is braked, not limp, so an observation-only session holds its pose
    instead of the arm dropping.

    Compared against whichever Mode.idle robot_lifecycle actually bound: the
    stub above in a bare environment, or the real trossen_arm enum when another
    test has already imported the driver. Comparing strings made this pass or
    fail depending on test order.
    """
    robot = FakeRobot()
    robot_lifecycle.connect_without_motion(robot)
    idle = robot_lifecycle.trossen_arm.Mode.idle
    assert robot.left_arm.driver.modes == [idle]
    assert robot.right_arm.driver.modes == [idle]


def test_motion_free_connect_still_opens_the_cameras():
    robot = FakeRobot()
    robot_lifecycle.connect_without_motion(robot)
    assert "cam cam_high:connect" in robot.log
    assert "cam cam_wrist:connect" in robot.log


def test_configure_is_restored_after_a_motion_free_connect():
    """The swap is scoped: a later deliberate park must still work normally."""
    robot = FakeRobot()
    robot_lifecycle.connect_without_motion(robot)
    robot.log.clear()
    robot.left_arm.configure()
    assert "left:MOVE" in robot.log


def test_configure_is_restored_even_if_connect_raises():
    robot = FakeRobot()
    original = robot.left_arm.configure

    def boom(calibrate=True):
        raise RuntimeError("no route to host")

    robot.connect = boom
    with pytest.raises(RuntimeError):
        robot_lifecycle.connect_without_motion(robot)
    assert robot.left_arm.configure == original


# -- release -------------------------------------------------------------------


def test_release_without_parking_commands_no_motion():
    """The whole point: a fault must not be answered with a new trajectory."""
    robot = FakeRobot()
    robot_lifecycle.release_without_parking(robot)
    assert not [entry for entry in robot.log if "MOVE" in entry]
    assert "left:cleanup" in robot.log
    assert "right:cleanup" in robot.log


def test_release_frees_every_camera():
    robot = FakeRobot()
    robot_lifecycle.release_without_parking(robot)
    assert "cam cam_high:disconnect" in robot.log
    assert "cam cam_wrist:disconnect" in robot.log


def test_one_failing_arm_does_not_strand_the_other_or_the_cameras():
    """The driver releases left, then right, then cameras, so a raising left
    arm used to leave everything after it held open."""
    robot = FakeRobot(fail_left_cleanup=True)
    robot_lifecycle.release_without_parking(robot)
    assert "right:cleanup" in robot.log
    assert "cam cam_high:disconnect" in robot.log
    assert "cam cam_wrist:disconnect" in robot.log


def test_one_failing_camera_does_not_strand_the_others():
    robot = FakeRobot(fail_camera=True)
    robot_lifecycle.release_without_parking(robot)
    assert "cam cam_wrist:disconnect" in robot.log
    assert "left:cleanup" in robot.log


def test_release_never_raises_so_a_fault_path_can_always_finish():
    robot = FakeRobot(fail_left_cleanup=True, fail_camera=True)
    robot_lifecycle.release_without_parking(robot)  # must not raise


# -- deliberate park -----------------------------------------------------------


def test_park_and_release_does_move_the_arms():
    """Parking is still available; it is just no longer the answer to a fault."""
    robot = FakeRobot()
    robot_lifecycle.park_and_release(robot)
    assert "left:MOVE" in robot.log
    assert "left:cleanup" in robot.log


def test_a_failed_park_still_releases_the_hardware():
    robot = FakeRobot()

    def boom():
        raise RuntimeError("arm blocked")

    robot.disconnect = boom
    robot_lifecycle.park_and_release(robot)
    assert "left:cleanup" in robot.log
    assert "cam cam_high:disconnect" in robot.log
