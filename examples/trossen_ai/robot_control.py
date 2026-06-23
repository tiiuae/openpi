"""Robot construction + motion/safety, decoupled from any policy or dataset.

`build_stationary_robot` is the single source for the Trossen bimanual follower
config (IPs, cameras), so the bridge and the dataset-replay tool stay in sync.
`RobotController` wraps a connected robot with the joint-velocity safety check and
PCHIP-smoothed motion used by the autonomous control loop, operating on full 14-D
joint vectors (left arm 0:7, right arm 7:14).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig
from scipy.interpolate import PchipInterpolator

logger = logging.getLogger(__name__)


def build_stationary_robot(*, connect: bool = True, with_cameras: bool = True):
    """Build (and optionally connect) the Trossen bimanual follower robot.

    Args:
        connect:      Call ``robot.connect()`` before returning.
        with_cameras: Include the three OpenCV cameras. Replay needs no images,
                      so it can skip them; the autonomous loop keeps them.
    """
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
        min_time_to_move_multiplier=3.0,
        loop_rate=30,
        cameras=cameras,
    )
    robot = make_robot_from_config(robot_config)
    if connect:
        robot.connect()
    return robot


class RobotController:
    """Joint-space motion + safety for the bimanual follower (14-D actions)."""

    # Per-joint commanded-velocity bounds (rad/s), left arm then right arm.
    JOINT_LIMIT = np.array(
        [
            [-6.283185, 6.283185],
            [-6.283185, 6.283185],
            [-6.283185, 6.283185],
            [-9.424778, 9.424778],
            [-9.424778, 9.424778],
            [-9.424778, 9.424778],
            [-9.424778, 9.424778],
            [-6.283185, 6.283185],
            [-6.283185, 6.283185],
            [-6.283185, 6.283185],
            [-9.424778, 9.424778],
            [-9.424778, 9.424778],
            [-9.424778, 9.424778],
            [-9.424778, 9.424778],
        ]
    )
    SLEEP_POSITION = np.zeros(14)

    def __init__(self, robot, control_frequency: int = 50, test_mode: str = "autonomous") -> None:
        self.robot = robot
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.test_mode = test_mode

    # ---- state ----
    def current_joints14(self) -> np.ndarray:
        obs = self.robot.get_observation()
        joint_pos_keys = [k for k in obs if k.endswith(".pos")]
        return np.array([obs[k] for k in joint_pos_keys])

    # ---- safety ----
    def is_action_within_limits(self, target_pose: np.ndarray) -> bool:
        target_pose = np.asarray(target_pose).flatten()
        if target_pose.shape[0] != 14:
            logger.error(f"Expected 14D action, got {target_pose.shape}")
            return False
        try:
            current_pose = self.current_joints14()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to get observation: {e}")
            return False

        commanded_velocity = (target_pose - current_pose) / self.dt
        for i, vel in enumerate(commanded_velocity):
            min_vel, max_vel = self.JOINT_LIMIT[i]
            if vel < min_vel or vel > max_vel:
                logger.warning(f"Joint {i} velocity exceeded: {vel:.4f} not in [{min_vel:.4f}, {max_vel:.4f}]")
                return False
        return True

    # ---- motion ----
    def execute_action(self, action: np.ndarray) -> bool:
        """Send a 14-D joint action. Returns False if it violated limits (and
        moved the arm to sleep), True otherwise."""
        full_action = np.asarray(action).flatten()

        if self.test_mode == "test":
            logger.info(f"TEST MODE: Would execute action: {full_action}")
            return True

        joint_features = list(self.robot._joint_ft.keys())  # noqa: SLF001
        action_dict = {k: full_action[i] for i, k in enumerate(joint_features)}
        self.robot.send_action(action_dict)

        if not self.is_action_within_limits(full_action):
            logger.warning("Action exceeds limits. Moving to sleep position.")
            self.move_to_sleep_position(duration=10.0)
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
