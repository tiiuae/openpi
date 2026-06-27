"""Robot construction + motion/safety, decoupled from any policy or dataset.

`build_stationary_robot` is the single source for the Trossen bimanual follower
config (IPs, cameras), so the bridge and the dataset-replay tool stay in sync.
`RobotController` wraps a connected robot with firmware-fault detection and
smooth-streaming motion used by the autonomous control loop, operating on full
14-D joint vectors (left arm 0:7, right arm 7:14).
"""

from __future__ import annotations

import logging
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

logger = logging.getLogger(__name__)

# Home/"stage" pose: arms up & open, ready for task start (left arm only; right at 0).
# Mirrors the commented stage_pose in main.py on the trossen-ai branch.
HOME_POSITION = np.array([0, np.pi / 3, np.pi / 6, np.pi / 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)


def send_action_smooth(
    robot, action14: np.ndarray, dt: float, feedforward_velocity: np.ndarray | None = None
) -> None:
    """Stream a 14-D joint target with feed-forward velocity for natural motion.

    Bypasses robot.send_action (which sends zero feed-forward velocity, planning
    to arrive at rest at every waypoint -> stutter). The arm carries velocity
    through each waypoint, and goal_time = dt so the firmware does not
    over/under-shoot the control period.

    ``feedforward_velocity`` is the intended per-joint velocity (rad/s). When the
    caller already knows the commanded trajectory (dataset replay), it should
    pass ``(target - prev_command)/dt`` here: that is the *smooth* trajectory
    velocity. Falling back to ``(target - measured)/dt`` (when None) reads the
    live pose, which lags and carries encoder jitter; dividing that by the small
    control period dt amplifies the noise into a shaky, audible command. The
    autonomous loop, which has no prior commanded target, uses the fallback.
    """
    action14 = np.asarray(action14, dtype=float).flatten()
    if feedforward_velocity is not None:
        ff = np.asarray(feedforward_velocity, dtype=float).flatten()
    else:
        obs = robot.get_observation()
        joint_pos_keys = [k for k in obs if k.endswith(".pos")]
        current = np.array([obs[k] for k in joint_pos_keys], dtype=float)
        ff = (action14 - current) / dt
    ff = np.nan_to_num(ff, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(robot.left_arm.config.joint_names)
    robot.left_arm.driver.set_all_positions(
        list(action14[:n]),
        goal_time=dt,
        blocking=False,
        goal_feedforward_velocities=list(ff[:n]),
    )
    robot.right_arm.driver.set_all_positions(
        list(action14[n : n * 2]),
        goal_time=dt,
        blocking=False,
        goal_feedforward_velocities=list(ff[n : n * 2]),
    )


# Conservative per-joint speed cap (rad/s) for replayed/streamed joint targets.
# Well under the arm's firmware velocity limit (~3*pi ≈ 9.42 rad/s) and ~10x the
# speed of normal recorded motion, so it only bites on a spurious IK branch-flip
# (which would otherwise command a one-step velocity spike and fault the firmware).
MAX_JOINT_SPEED = 3.0


def limit_joint_velocity(
    prev14: np.ndarray, target14: np.ndarray, dt: float, max_speed: float = MAX_JOINT_SPEED
) -> np.ndarray:
    """Clamp a 14-D joint target so no joint moves faster than ``max_speed`` rad/s.

    Bounds ``|target - prev|`` per joint to ``max_speed * dt``. A joint-space
    discontinuity (e.g. the IK solver jumping to an alternate configuration for
    nearly the same end-effector pose) is then spread across several control
    steps instead of commanding an over-limit velocity. Returns the clamped
    target; the caller should feed it back as ``prev`` on the next step so the
    arm keeps migrating toward the true target.
    """
    prev = np.asarray(prev14, dtype=float).flatten()
    target = np.asarray(target14, dtype=float).flatten()
    max_step = float(max_speed) * float(dt)
    delta = np.clip(target - prev, -max_step, max_step)
    return prev + delta


def build_stationary_robot(
    *, connect: bool = True, with_cameras: bool = True, min_time_to_move_multiplier: float = 10.0, loop_rate: int = 30
):
    """Build (and optionally connect) the Trossen bimanual follower robot.

    Args:
        connect:                     Call ``robot.connect()`` before returning.
        with_cameras:                Include the three OpenCV cameras. Replay
                                     needs no images, so it can skip them; the
                                     autonomous loop keeps them.
        min_time_to_move_multiplier: Passed to BiWidowXAIFollowerRobotConfig.
        loop_rate:                   Passed to BiWidowXAIFollowerRobotConfig.
    """
    # Lazy imports: these hardware packages are not installed in the test env.
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa
    from lerobot.robots import make_robot_from_config  # noqa
    from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig  # noqa

    cameras = {}
    if with_cameras:
        cameras = {
            "cam_high": OpenCVCameraConfig(index_or_path=Path("/dev/video16"), width=640, height=480, fps=30),
            "cam_right_wrist": OpenCVCameraConfig(index_or_path=Path("/dev/video10"), width=640, height=480, fps=30),
            "cam_left_wrist": OpenCVCameraConfig(index_or_path=Path("/dev/video4"), width=640, height=480, fps=30),
        }
    robot_config = BiWidowXAIFollowerRobotConfig(
        id="bimanual_follower",
        left_arm_ip_address="192.168.1.5",
        right_arm_ip_address="192.168.1.4",
        min_time_to_move_multiplier=min_time_to_move_multiplier,
        loop_rate=loop_rate,
        cameras=cameras,
    )
    robot = make_robot_from_config(robot_config)
    if connect:
        robot.connect()
    return robot


class RobotController:
    """Joint-space motion + safety for the bimanual follower (14-D actions)."""

    SLEEP_POSITION = np.zeros(14)

    def __init__(
        self,
        robot,
        control_frequency: int = 50,
        test_mode: str = "autonomous",
        smooth_streaming: bool = False,  # noqa
    ) -> None:
        self.robot = robot
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.test_mode = test_mode
        self.smooth_streaming = smooth_streaming

    # ---- state ----
    def current_joints14(self) -> np.ndarray:
        obs = self.robot.get_observation()
        joint_pos_keys = [k for k in obs if k.endswith(".pos")]
        return np.array([obs[k] for k in joint_pos_keys])

    # ---- motion ----
    def execute_action(self, action: np.ndarray, feedforward_velocity: np.ndarray | None = None) -> bool:
        """Send a 14-D joint action. Returns True on success, False if the
        firmware faulted (in which case the arm is moved to sleep).

        ``feedforward_velocity`` (rad/s) is forwarded to the smooth-streaming
        path; pass the commanded trajectory velocity so the feed-forward term is
        smooth instead of derived from the noisy measured pose. Ignored when
        ``smooth_streaming`` is off."""
        full_action = np.asarray(action).flatten()

        if self.test_mode == "test":
            logger.info(f"TEST MODE: Would execute action: {full_action}")
            return True

        try:
            if self.smooth_streaming:
                send_action_smooth(self.robot, full_action, self.dt, feedforward_velocity)
            else:
                joint_features = list(self.robot._joint_ft.keys())  # noqa: SLF001
                action_dict = {k: full_action[i] for i, k in enumerate(joint_features)}
                self.robot.send_action(action_dict)
        except Exception as exc:  # firmware fault halts the arm
            logger.error(f"Firmware error executing action: {exc}. Moving to sleep position.")
            try:
                self.move_to_sleep_position(duration=10.0)
            except Exception as sleep_exc:
                logger.error(f"Failed to reach sleep position: {sleep_exc}")
            return False
        return True

    def move_to_start_position(self, goal_position: np.ndarray, duration: float = 5.0) -> None:
        """Smoothly move from the current pose to *goal_position* (PCHIP)."""
        start_pose = self.current_joints14()
        waypoints = np.array([start_pose, np.asarray(goal_position).flatten()])
        timepoints = np.array([0, duration])
        interpolator = PchipInterpolator(timepoints, waypoints, axis=0)

        start_time = time.time()
        end_time = start_time + duration
        while time.time() < end_time:
            loop_start = time.perf_counter()
            positions = interpolator(time.time() - start_time)
            self.execute_action(positions)
            elapsed = time.perf_counter() - loop_start
            if self.dt - elapsed > 0:
                time.sleep(self.dt - elapsed)

    def move_to_sleep_position(self, duration: float = 10.0) -> None:
        joint_features = list(self.robot._joint_ft.keys())  # noqa: SLF001
        current_pose = self.current_joints14()
        waypoints = np.array([current_pose, self.SLEEP_POSITION])
        timepoints = np.array([0, duration])
        interpolator = PchipInterpolator(timepoints, waypoints, axis=0)

        logger.info(f"Moving to sleep position over {duration}s...")
        start_time = time.time()
        end_time = start_time + duration
        while time.time() < end_time:
            loop_start = time.perf_counter()
            positions = interpolator(time.time() - start_time)
            action_dict = {k: positions[i] for i, k in enumerate(joint_features)}
            self.robot.send_action(action_dict)
            elapsed = time.perf_counter() - loop_start
            if self.dt - elapsed > 0:
                time.sleep(self.dt - elapsed)
        logger.info("Reached sleep position.")

    def disconnect(self) -> None:
        self.robot.disconnect()
