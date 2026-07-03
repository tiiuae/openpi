#!/usr/bin/env python3
"""
Trossen Arm <-> OpenPI Policy Server Bridge (Bimanual Version)

Bridge between a bimanual widowx and the OpenPI policy server.
Handles:
1. Collecting observations from the arm (joint positions, images)
2. Sending observations to the policy server via WebSocket
3. Receiving action predictions
4. Executing actions on the arm

Usage:
    python main.py --mode autonomous --task_prompt "Pick up the blue cup and place it in the orange basket"

    Test mode (no movement):
    python main.py --mode test --task_prompt "Pick up the blue cup and place it in the orange basket"
"""

import argparse
import logging
import time
from pathlib import Path

from action_ensemble import ActionLogger
from action_ensemble import AsyncPolicyWorker
from action_ensemble import make_ensemble
import cv2
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig
import numpy as np
from openpi_client import websocket_client_policy
from PIL import Image
from scipy.interpolate import PchipInterpolator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TRAINING_SIZE = (224, 224)


class TrossenOpenPIBridge:
    """Bridge between a Trossen AI Stationary Kit and OpenPI policy server."""

    def __init__(
        self,
        policy_server_host: str = "localhost",
        policy_server_port: int = 8080,
        control_frequency: int = 10,
        test_mode: str = "autonomous",  # "autonomous" or "test"
        max_steps: int = 1000,
        action_chunk_size: int = 10,
        rate_of_inference: int = 10,
        ensemble_type: str = "exp",
        async_inference: bool = False,
        log_dir: str | None = None,
        use_left_arm_only: bool = False,
        use_right_arm_only: bool = False,
        starvla: bool = False,
        raw_gripper: bool = True,
        gripper_indices: list[int] | None = None,
    ):
        self.starvla = starvla
        self.control_frequency = control_frequency
        self.max_steps = max_steps
        self.dt = 1.0 / control_frequency
        self.test_mode = test_mode

        logger.info(f"Connecting to policy server at {policy_server_host}:{policy_server_port}")
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_server_host, port=policy_server_port
        )

        robot_config = BiWidowXAIFollowerRobotConfig(
            id="bimanual_follower",
            left_arm_ip_address="192.168.1.5",
            right_arm_ip_address="192.168.1.4",
            min_time_to_move_multiplier=3.0,
            loop_rate=30,
            cameras={
                "cam_high": OpenCVCameraConfig(index_or_path=Path("/dev/cam_high"), width=640, height=480, fps=30),
                # "cam_low": RealSenseCameraConfig(
                #     serial_number_or_name="130322272628", width=640, height=480, fps=30, use_depth=False
                # ),
                "cam_right_wrist": OpenCVCameraConfig(
                    index_or_path=Path("/dev/cam_wrist_right"), width=640, height=480, fps=30
                ),
                "cam_left_wrist": OpenCVCameraConfig(
                    index_or_path=Path("/dev/cam_wrist_left"), width=640, height=480, fps=30
                ),
            },
        )
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()

        self.current_action_chunk = None
        self.action_chunk_idx = 0

        self.action_chunk_size = action_chunk_size
        self.episode_step = 0
        self.is_running = False

        self.rate_of_inference = rate_of_inference  # Number of control steps per policy inference
        self.action_dim = len(self.robot._joint_ft)  # 7 joints per arm * 2 arms
        self.ensemble = make_ensemble(ensemble_type)

        if async_inference and self.ensemble is None:
            raise ValueError("--async_inference requires an ensemble (ensemble_type cannot be 'none')")
        self.async_inference = async_inference
        self._policy_worker = (
            AsyncPolicyWorker(self.policy_client, self.ensemble, self.action_dim) if async_inference else None
        )

        self.action_logger = ActionLogger(log_dir) if log_dir else None

        self.use_left_arm_only = use_left_arm_only
        self.use_right_arm_only = use_right_arm_only

        # Gripper channels are near-binary, so temporal averaging makes them mushy and
        # laggy. When an ensemble is active we bypass it for the gripper dims and use the
        # latest raw prediction instead (joints stay smoothed). For a 14-dim bimanual
        # ALOHA layout ([6 joints + gripper] per arm) the grippers are at indices 6 and 13.
        self.raw_gripper = raw_gripper
        if gripper_indices is not None:
            self.gripper_indices = list(gripper_indices)
        elif self.action_dim == 14:
            self.gripper_indices = [6, 13]
        else:
            self.gripper_indices = []

    def execute_action(self, action: np.ndarray):
        """Execute action on the arm."""
        full_action = action.copy()

        if self.use_left_arm_only or self.use_right_arm_only:
            if self.use_right_arm_only:
                # Freeze left arm (indices 0:7) at current pose; only right arm (7:14) moves
                full_action[:7] = self._frozen_arm_pose[:7]
            elif self.use_left_arm_only:
                # Freeze right arm (indices 7:14) at current pose; only left arm (0:7) moves
                full_action[7:] = self._frozen_arm_pose[7:]

        if self.test_mode == "test":
            logger.info(f"TEST MODE: Would execute action: {full_action}")
            return
        if self.test_mode == "autonomous":
            joint_features = list(self.robot._joint_ft.keys())
            action_dict = {k: full_action[i] for i, k in enumerate(joint_features)}

            self.robot.send_action(action_dict)
        else:
            logger.error(f"Unknown mode: {self.test_mode}. No action executed.")

    def _build_observation(self, observation_dict: dict, task_prompt: str) -> dict:
        joint_pos_keys = [k for k in observation_dict if k.endswith(".pos")]
        joint_positions = np.array([observation_dict[k] for k in joint_pos_keys])
        cameras = list(self.robot._cameras_ft.keys())
        images = {}
        for cam in cameras:
            image_hwc = observation_dict[cam]
            if self.starvla:
                image_rgb = np.array(Image.fromarray(cv2.cvtColor(image_hwc, cv2.COLOR_BGR2RGB)).resize((224, 224)))
            else:
                image_resized = cv2.resize(image_hwc, DEFAULT_TRAINING_SIZE, interpolation=cv2.INTER_LANCZOS4)
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            images[cam] = np.transpose(image_rgb, (2, 0, 1))
        return {"state": joint_positions, "images": images, "prompt": task_prompt}

    def move_to_start_position(self, goal_position: np.ndarray, duration: float = 5.0):
        """The first position queried from the policy depends on the training data.
        Assuming the first position is a "stage" position will result in a large jump if the arm is not already there.
        To avoid this, we smoothly move the arm to a first action/position before sending the rest of the actions.
        We use PCHIP interpolation for smooth trajectory generation and give it enough time to reach the position to prevent
        jumps and triggering safety stops (velocity limits)."""

        joint_pos_keys = [k for k in self.robot.get_observation().keys() if k.endswith(".pos")]
        self._frozen_arm_pose = np.array([self.robot.get_observation()[k] for k in joint_pos_keys])
        # Example stage_pose for bimanual WidowX arms.
        # Each value corresponds to a joint position (in radians) for the 14 joints:
        # [left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5, left_left_carriage_joint,
        #  right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5, right_left_carriage_joint]
        # The values below represent a "stage" pose, e.g. arms up and open, ready for task start.
        # stage_pose = np.array([0, np.pi/3, np.pi/6, np.pi/5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        waypoints = np.array([self._frozen_arm_pose, goal_position])
        timepoints = np.array([0, duration])  # Use the provided duration
        interpolator_position = PchipInterpolator(timepoints, waypoints, axis=0)

        start_time = time.time()
        end_time = start_time + timepoints[-1]

        while time.time() < end_time:
            loop_start_time = time.time()
            current_time = loop_start_time - start_time
            positions = interpolator_position(current_time)
            self.execute_action(positions)

    def run_episode(self, task_prompt: str = "look down"):
        """Run a single episode of policy execution."""
        logger.info(f"Starting episode with prompt: '{task_prompt}'")
        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.is_running = True
        if self.ensemble is not None:
            self.ensemble.reset()
        if self.action_logger is not None:
            self.action_logger.reset()
        is_first_step = True

        if self.use_left_arm_only or self.use_right_arm_only:
            _obs = self.robot.get_observation()
            _joint_pos_keys = [k for k in _obs.keys() if k.endswith(".pos")]
            self._frozen_arm_pose = np.array([_obs[k] for k in _joint_pos_keys])

        if self.async_inference:
            self._policy_worker.start()

        try:
            while self.is_running and self.episode_step < self.max_steps:
                start_loop_time = time.perf_counter()

                if self.async_inference:
                    # Submit fresh observation every step — non-blocking
                    obs = self._build_observation(self.robot.get_observation(), task_prompt)
                    self._policy_worker.submit(obs, self.episode_step)
                    if is_first_step:
                        logger.info("Waiting for first inference result...")
                        if not self._policy_worker.wait_for_first(timeout=30.0):
                            logger.error("Timed out waiting for first inference — aborting")
                            break
                    a_t = self.ensemble.get_action(self.episode_step)
                    if a_t is None:
                        a_t = np.zeros(self.action_dim)

                else:
                    # Synchronous: request new chunk every rate_of_inference steps
                    if self.current_action_chunk is None or self.action_chunk_idx >= self.rate_of_inference:
                        observation = self._build_observation(self.robot.get_observation(), task_prompt)
                        logger.info(f"Step {self.episode_step}: Requesting new action chunk")
                        response = self.policy_client.infer(observation)
                        self.current_action_chunk = response["actions"][:, : self.action_dim]
                        if self.ensemble is not None:
                            self.ensemble.add_chunk(self.episode_step, self.current_action_chunk)
                        self.action_chunk_idx = 0
                        logger.info(f"Received action chunk: {self.current_action_chunk.shape}")

                    if self.ensemble is not None:
                        a_t = self.ensemble.get_action(self.episode_step)
                        if a_t is None:
                            a_t = np.zeros(self.action_dim)
                    else:
                        a_t = self.current_action_chunk[self.action_chunk_idx]

                if self.action_logger is not None and self.ensemble is not None:
                    self.action_logger.log(
                        self.episode_step,
                        self.ensemble.get_overlap_count(self.episode_step),
                    )

                # Use the latest raw prediction for the gripper channels to avoid the
                # ensemble averaging/lag that leaves the gripper half-open.
                if self.raw_gripper and self.ensemble is not None and self.gripper_indices:
                    raw = self.ensemble.get_latest_raw(self.episode_step)
                    if raw is not None:
                        for gi in self.gripper_indices:
                            if gi < len(a_t) and gi < len(raw):
                                a_t[gi] = raw[gi]

                if is_first_step:
                    logger.info("Moving to start position to avoid large jumps...")
                    self.move_to_start_position(a_t, duration=5.0)
                    is_first_step = False
                else:
                    self.execute_action(a_t)

                self.action_chunk_idx += 1
                self.episode_step += 1

                dt_s = time.perf_counter() - start_loop_time
                if self.dt - dt_s > 0:
                    time.sleep(self.dt - dt_s)
                loop_s = time.perf_counter() - start_loop_time
                logger.info(f"time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        finally:
            if self.async_inference:
                self._policy_worker.stop()
            if self.action_logger is not None:
                self.action_logger.save(tag=f"episode_{self.episode_step}steps")

        self.is_running = False
        logger.info(f"Episode completed after {self.episode_step} steps")

    def autonomous_mode(self, task_prompt: str = "look down"):
        """Run in autonomous mode where the arm executes policy predictions."""
        logger.info("Starting autonomous mode")
        self.run_episode(task_prompt=task_prompt)

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trossen AI Stationary Kit <-> OpenPI Policy Server Bridge")
    parser.add_argument("--policy_host", default="192.168.50.174", help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=8800, help="Policy server port")
    parser.add_argument("--control_freq", type=int, default=25, help="Control frequency in Hz")
    parser.add_argument(
        "--mode",
        choices=["autonomous", "test"],
        default="autonomous",
        help="Operation mode: autonomous (execute) or test (no movement)",
    )
    parser.add_argument("--task_prompt", default="move the arm to the left", help="Task description for the policy")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument(
        "--action_chunk_size", type=int, default=25, help="Number of actions predicted per inference call"
    )
    parser.add_argument(
        "--rate_of_inference", type=int, default=20, help="Control steps between policy inference calls"
    )
    parser.add_argument(
        "--ensemble_type",
        choices=["exp", "cogact", "none"],
        default="exp",
        help="Action ensemble strategy: 'exp' (exponential decay), 'cogact' (cosine-similarity AAE), 'none' (disabled)",
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Directory to save per-episode overlap count JSON. Omit to disable.",
    )
    parser.add_argument(
        "--async_inference",
        action="store_true",
        help="Run inference in a background thread — control loop never blocks. "
        "Requires an ensemble (not 'none'). Recommended with --ensemble_type cogact.",
    )
    parser.add_argument(
        "--starvla", action="store_true", help="Use StarVLA image resizing (224x224 via PIL) instead of default"
    )
    parser.add_argument(
        "--use_left_arm_only", action="store_true", help="Only move the left arm; right arm stays at current pose"
    )
    parser.add_argument(
        "--use_right_arm_only", action="store_true", help="Only move the right arm; left arm stays at current pose"
    )
    parser.add_argument(
        "--ensemble_gripper",
        action="store_true",
        help="Also temporally ensemble the gripper channels. By default the gripper uses "
        "the latest raw prediction to avoid mushy/laggy open-close behaviour.",
    )
    args = parser.parse_args()

    bridge = TrossenOpenPIBridge(
        policy_server_host=args.policy_host,
        policy_server_port=args.policy_port,
        control_frequency=args.control_freq,
        test_mode=args.mode,
        max_steps=args.max_steps,
        action_chunk_size=args.action_chunk_size,
        rate_of_inference=args.rate_of_inference,
        ensemble_type=args.ensemble_type,
        async_inference=args.async_inference,
        log_dir=args.log_dir,
        use_left_arm_only=args.use_left_arm_only,
        use_right_arm_only=args.use_right_arm_only,
        starvla=args.starvla,
        raw_gripper=not args.ensemble_gripper,
    )

    bridge.autonomous_mode(task_prompt=args.task_prompt)

    bridge.cleanup()
