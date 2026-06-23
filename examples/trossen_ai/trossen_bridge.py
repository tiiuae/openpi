import logging
from pathlib import Path
import time

from action_logger import ActionLogger
from async_worker import AsyncPolicyWorker
from ensemble import make_ensemble
import cv2
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig
import numpy as np
from openpi_client import websocket_client_policy
from PIL import Image
from scipy.interpolate import PchipInterpolator

from adapters import ActionSpaceAdapter, JointAdapter, extract_joints

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TRAINING_SIZE = (224, 224)


class TrossenOpenPIBridge:
    """Bridge between a Trossen AI Stationary Kit and OpenPI policy server."""

    ######## JOINT LIMIT VALUES ##########
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

    SLEEP_POSITION = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

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
        async_inference: bool = False,  # noqa
        log_dir: str | None = None,
        use_left_arm_only: bool = False,  # noqa
        use_right_arm_only: bool = False,  # noqa
        starvla: bool = False,  # noqa
        adapter: "ActionSpaceAdapter | None" = None,
    ):
        self.adapter = adapter if adapter is not None else JointAdapter()
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
                "cam_high": OpenCVCameraConfig(index_or_path=Path("/dev/video16"), width=640, height=480, fps=30),
                # "cam_low": RealSenseCameraConfig(
                #     serial_number_or_name="130322272628", width=640, height=480, fps=30, use_depth=False
                # ),
                "cam_right_wrist": OpenCVCameraConfig(
                    index_or_path=Path("/dev/video10"), width=640, height=480, fps=30
                ),
                "cam_left_wrist": OpenCVCameraConfig(index_or_path=Path("/dev/video4"), width=640, height=480, fps=30),
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
        self.action_dim = len(self.robot._joint_ft)  # 7 joints per arm * 2 arms # noqa
        self.state_dim = self.adapter.state_dim
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

    ##################################################
    ###### When Joint limits are exceeded, robot is sent to sleep position
    def is_action_within_limits(self, target_pose: np.ndarray) -> bool:

        target_pose = np.asarray(target_pose).flatten()

        if target_pose.shape[0] != 14:
            logger.error(f"Expected 14D action, got {target_pose.shape}")
            return False

        try:
            obs = self.robot.get_observation()
        except Exception as e:
            logger.error(f"Failed to get observation: {e}")
            return False

        joint_pos_keys = [k for k in obs if k.endswith(".pos")]

        current_pose = np.array([obs[k] for k in joint_pos_keys])

        commanded_velocity = (target_pose - current_pose) / self.dt

        for i, vel in enumerate(commanded_velocity):
            min_vel, max_vel = self.JOINT_LIMIT[i]

            if vel < min_vel or vel > max_vel:
                logger.warning(f"Joint {i} velocity exceeded: {vel:.4f} not in [{min_vel:.4f}, {max_vel:.4f}]")

                return False

        return True

        #######################################

    def expand_action_to_full(self, action: np.ndarray) -> np.ndarray:
        """
        Convert 7D or 14D action into full 14D action.
        """

        action = np.asarray(action).flatten()

        # Already full bimanual action
        if action.shape[0] == 14:
            return action.copy()

        # Single arm action
        if action.shape[0] == 7:
            full_action = self._frozen_arm_pose.copy()

            if self.use_left_arm_only:
                full_action[:7] = action

            elif self.use_right_arm_only:
                full_action[7:] = action

            else:
                raise ValueError("Received 7D action but no single-arm mode enabled.")

            return full_action

        raise ValueError(f"Unexpected action dimension: {action.shape}")

    def execute_action(self, action: np.ndarray):
        """Execute action on the arm."""

        full_action = self.expand_action_to_full(action)
        """
        if self.use_left_arm_only or self.use_right_arm_only:
            if self.use_right_arm_only:
                # Freeze left arm (indices 0:7) at current pose; only right arm (7:14) moves
                full_action[:7] = self._frozen_arm_pose[:7]
            elif self.use_left_arm_only:
                # Freeze right arm (indices 7:14) at current pose; only left arm (0:7) moves
                full_action[7:] = self._frozen_arm_pose[7:]
        """

        if self.test_mode == "test":
            logger.info(f"TEST MODE: Would execute action: {full_action}")
            return
        if self.test_mode == "autonomous":
            joint_features = list(self.robot._joint_ft.keys())  # noqa
            action_dict = {k: full_action[i] for i, k in enumerate(joint_features)}

            self.robot.send_action(action_dict)
        else:
            logger.error(f"Unknown mode: {self.test_mode}. No action executed.")

        ####################################
        ### Check action value before sending to robot
        ## FIXME: double check the limit
        if not self.is_action_within_limits(full_action):
            logger.warning("Action exceeds limits. Moving to sleep position.")
            self.move_to_sleep_position(duration=10.0)
            self.is_running = False
            return

        ##################################

    def _build_observation(self, observation_dict: dict, task_prompt: str) -> dict:
        state = self.adapter.build_state(observation_dict)
        cameras = list(self.robot._cameras_ft.keys())  # noqa
        images = {}
        for cam in cameras:
            image_hwc = observation_dict[cam]
            if self.starvla:
                image_rgb = np.array(Image.fromarray(cv2.cvtColor(image_hwc, cv2.COLOR_BGR2RGB)).resize((224, 224)))
            else:
                image_resized = cv2.resize(image_hwc, DEFAULT_TRAINING_SIZE, interpolation=cv2.INTER_LANCZOS4)
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            images[cam] = np.transpose(image_rgb, (2, 0, 1))
        return {"state": state, "images": images, "prompt": task_prompt}

    def move_to_start_position(self, goal_position: np.ndarray, duration: float = 5.0):
        """Smoothly move the arm to a start position using PCHIP interpolation."""

        joint_pos_keys = [k for k in self.robot.get_observation().keys() if k.endswith(".pos")]  # noqa
        self._frozen_arm_pose = np.array([self.robot.get_observation()[k] for k in joint_pos_keys])

        waypoints = np.array([self._frozen_arm_pose, goal_position])
        timepoints = np.array([0, duration])

        interpolator_position = PchipInterpolator(timepoints, waypoints, axis=0)

        start_time = time.time()
        end_time = start_time + duration

        while time.time() < end_time:
            loop_start_time = time.time()
            current_time = loop_start_time - start_time
            positions = interpolator_position(current_time)
            self.execute_action(positions)

    def move_to_sleep_position(self, duration: float = 10.0):
        joint_features = list(self.robot._joint_ft.keys())  # noqa
        joint_pos_keys = [k for k in self.robot.get_observation() if k.endswith(".pos")]
        current_pose = np.array([self.robot.get_observation()[k] for k in joint_pos_keys])

        waypoints = np.array([current_pose, self.SLEEP_POSITION])
        timepoints = np.array([0, duration])
        interpolator = PchipInterpolator(timepoints, waypoints, axis=0)

        print(f"Moving to sleep position over {duration}s...")
        dt = 1.0 / self.control_frequency
        start_time = time.time()
        end_time = start_time + duration

        while time.time() < end_time:
            loop_start = time.perf_counter()
            current_time = time.time() - start_time
            positions = interpolator(current_time)
            action_dict = {k: positions[i] for i, k in enumerate(joint_features)}
            self.robot.send_action(action_dict)
            elapsed = time.perf_counter() - loop_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

        print("Reached sleep position.")

    def run_episode(self, task_prompt: str = "look down"):
        """Run a single episode of policy execution."""
        logger.info(f"Starting episode with prompt: '{task_prompt}'")
        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.is_running = True
        if self.async_inference and not isinstance(self.adapter, JointAdapter):
            raise NotImplementedError("Async inference with EE decoding is not supported yet.")
        if self.ensemble is not None:
            self.ensemble.reset()
        if self.action_logger is not None:
            self.action_logger.reset()
        is_first_step = True

        if self.use_left_arm_only or self.use_right_arm_only:
            _obs = self.robot.get_observation()
            _joint_pos_keys = [k for k in _obs.keys() if k.endswith(".pos")]  # noqa
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
                        obs_raw = self.robot.get_observation()
                        observation = self._build_observation(obs_raw, task_prompt)
                        logger.info(f"Step {self.episode_step}: Requesting new action chunk")
                        response = self.policy_client.infer(observation)
                        current_joints14 = extract_joints(obs_raw)
                        self.current_action_chunk = self.adapter.decode_chunk(response["actions"], current_joints14)
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

                # FIXME: temporary adaptation for the case of the model with 7 dof
                """
                if a_t.shape[0] == 7:
                    a_t = np.concatenate([self._frozen_arm_pose[:7], a_t])
                """
                if a_t.shape[0] == 7:
                    full_action = self._frozen_arm_pose.copy()

                    if self.use_left_arm_only:
                        full_action[:7] = a_t

                    elif self.use_right_arm_only:
                        full_action[7:] = a_t

                    else:
                        raise ValueError("Received 7D action but no single-arm mode enabled.")

                    a_t = full_action

                if is_first_step:
                    logger.info("Moving to start position to avoid large jumps...")
                    self.move_to_start_position(a_t, duration=5.0)
                    is_first_step = False
                else:
                    self.execute_action(a_t)
                    if not self.is_running:  ######## is_runnning is false when there is a joint limit
                        break

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
