"""Robot construction + motion/safety, decoupled from any policy or dataset.

`build_stationary_robot` is the single source for the Trossen bimanual follower
config (IPs, cameras), so the bridge and the dataset-replay tool stay in sync.
`RobotController` wraps a connected robot with firmware-fault detection and
smooth-streaming motion used by the autonomous control loop, operating on full
14-D joint vectors (left arm 0:7, right arm 7:14).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

logger = logging.getLogger(__name__)

# Home/"stage" pose: arms up & open, ready for task start (left arm only; right at 0).
# Mirrors the commented stage_pose in main.py on the trossen-ai branch.
HOME_POSITION = np.array([0, np.pi / 3, np.pi / 6, np.pi / 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)


# The Trossen firmware chooses its interpolation from goal_time
# (set_all_positions docs): goal_time > 0.2s -> quintic polynomial, which is the
# ONLY mode that uses goal_feedforward_velocities; 0.001-0.2s -> linear, which
# IGNORES the feed-forward terms; below that -> applied immediately. So a
# feed-forward velocity only does anything when goal_time clears 0.2s.
_QUINTIC_MIN_GOAL_TIME = 0.2


def send_action_smooth(
    robot,
    action14: np.ndarray,
    dt: float,
    feedforward_velocity: np.ndarray | None = None,
    goal_time: float | None = None,
) -> None:
    """Stream a 14-D joint target with feed-forward velocity for natural motion.

    How the firmware uses this (verified against the trossen_arm SDK, not just
    its docstring): set_all_positions plans a trajectory from the arm's current
    (position, velocity) to ``(goal_position, goal_feedforward_velocity)`` over
    ``goal_time``. With the feed-forward velocity set to the streaming
    trajectory velocity, the planned path *passes through* the waypoint at speed
    instead of decelerating to rest at it — that is what removes the stutter.

    The catch the old implementation missed: the firmware only runs the quintic
    interpolation that honours the feed-forward velocity when ``goal_time`` is
    above ~0.2s; at the old ``goal_time = dt`` (0.04s @ 25Hz) it silently used
    linear interpolation, dropped the feed-forward entirely, and re-planned a
    fresh linear segment every control period — the "accelerate then brake"
    chatter. So we plan over a horizon longer than that threshold while still
    issuing a new command every ``dt``: the arm follows only the smooth early
    part of each quintic toward the (continuously updated) target, and the next
    command supersedes the plan long before it would brake at the goal. A larger
    ``goal_time`` is smoother but lags the target more.

    ``feedforward_velocity`` is the per-joint velocity (rad/s) the arm should
    carry through the waypoint. Prefer the *commanded* trajectory velocity
    ``(target - prev_command)/dt``. The ``None`` fallback derives it from the
    measured pose ``(target - measured)/dt``, which lags and carries encoder
    jitter — usable only when no commanded history exists.
    """
    action14 = np.asarray(action14, dtype=float).flatten()
    # Horizon: caller's goal_time (e.g. min_time_to_move_multiplier * dt) but
    # never below the quintic threshold, else the feed-forward is ignored.
    horizon = goal_time if goal_time is not None else max(0.25, 5.0 * dt)
    horizon = max(horizon, _QUINTIC_MIN_GOAL_TIME + 0.01)
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
        goal_time=horizon,
        blocking=False,
        goal_feedforward_velocities=list(ff[:n]),
    )
    robot.right_arm.driver.set_all_positions(
        list(action14[n : n * 2]),
        goal_time=horizon,
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
        from realsense_settings import apply_realsense_settings  # noqa

        apply_realsense_settings(Path(__file__).with_name("realsense_settings.json"))
        cameras = {
            "cam_high": OpenCVCameraConfig(
                index_or_path=Path(os.environ.get("TROSSEN_CAMERA_HIGH", "/dev/video40")),
                width=640,
                height=480,
                fps=30,
            ),
            "cam_right_wrist": OpenCVCameraConfig(
                index_or_path=Path(os.environ.get("TROSSEN_CAMERA_RIGHT_WRIST", "/dev/video41")),
                width=640,
                height=480,
                fps=30,
            ),
            "cam_left_wrist": OpenCVCameraConfig(
                index_or_path=Path(os.environ.get("TROSSEN_CAMERA_LEFT_WRIST", "/dev/video42")),
                width=640,
                height=480,
                fps=30,
            ),
        }
    robot_config = BiWidowXAIFollowerRobotConfig(
        id="bimanual_follower",
        left_arm_ip_address=os.environ.get("TROSSEN_LEFT_ARM_IP", "192.168.1.5"),
        right_arm_ip_address=os.environ.get("TROSSEN_RIGHT_ARM_IP", "192.168.1.4"),
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
