import logging
import time

from action_fallback import HoldLastAction
from action_logger import ActionLogger
from adapters import ActionSpaceAdapter
from adapters import JointAdapter
from adapters import extract_joints
from async_worker import AsyncPolicyWorker
import cv2
from ensemble import make_ensemble
import numpy as np
from openpi_client import websocket_client_policy
from PIL import Image
from robot_control import build_stationary_robot
from robot_control import limit_joint_velocity
from scipy.interpolate import PchipInterpolator
from webapp.telemetry import NullSink
from webapp.telemetry import TelemetrySink

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TRAINING_SIZE = (224, 224)


class TrossenOpenPIBridge:
    """Bridge between a Trossen AI Stationary Kit and OpenPI policy server."""

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
        cogact_mode: str = "cogact",
        async_inference: bool = False,  # noqa
        log_dir: str | None = None,
        use_left_arm_only: bool = False,  # noqa
        use_right_arm_only: bool = False,  # noqa
        starvla: bool = False,  # noqa
        adapter: "ActionSpaceAdapter | None" = None,
        sink: TelemetrySink = NullSink(),  # noqa
        smooth_streaming: bool = False,  # noqa
        min_time_to_move_multiplier: float = 3.0,
        loop_rate: int = 30,
        max_joint_speed: float = 3.0,
    ):
        self.adapter = adapter if adapter is not None else JointAdapter()
        self.sink = sink
        self.starvla = starvla
        self.control_frequency = control_frequency
        self.max_steps = max_steps
        self.dt = 1.0 / control_frequency
        self.test_mode = test_mode

        logger.info(f"Connecting to policy server at {policy_server_host}:{policy_server_port}")
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_server_host, port=policy_server_port
        )

        self.robot = build_stationary_robot(
            min_time_to_move_multiplier=min_time_to_move_multiplier,
            loop_rate=loop_rate,
        )
        self.smooth_streaming = smooth_streaming
        # Per-joint velocity cap (rad/s) applied to every commanded action so a
        # policy/IK discontinuity can't trip the firmware velocity limit. <=0 disables.
        self.max_joint_speed = max_joint_speed
        self._last_commanded = None

        self.current_action_chunk = None
        self.action_chunk_idx = 0

        self.action_chunk_size = action_chunk_size
        self.episode_step = 0
        self.is_running = False

        self.rate_of_inference = rate_of_inference  # Number of control steps per policy inference
        self.action_dim = len(self.robot._joint_ft)  # 7 joints per arm * 2 arms # noqa
        self.state_dim = self.adapter.state_dim
        self.ensemble = make_ensemble(ensemble_type, cogact_mode=cogact_mode)

        if async_inference and self.ensemble is None:
            raise ValueError("--async_inference requires an ensemble (ensemble_type cannot be 'none')")
        self.async_inference = async_inference
        self._policy_worker = (
            AsyncPolicyWorker(self.policy_client, self.ensemble, self.action_dim, sink=self.sink)
            if async_inference
            else None
        )

        self.action_logger = ActionLogger(log_dir) if log_dir else None

        self.use_left_arm_only = use_left_arm_only
        self.use_right_arm_only = use_right_arm_only

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
            # Velocity-limit the commanded action: cap per-joint delta vs the last
            # command so a policy/IK jump is spread over several control steps
            # instead of faulting the firmware ("joint velocity limit exceeded").
            if self.max_joint_speed > 0 and self._last_commanded is not None:
                full_action = limit_joint_velocity(self._last_commanded, full_action, self.dt, self.max_joint_speed)
            self._last_commanded = np.asarray(full_action, dtype=float).flatten()

            joint_features = list(self.robot._joint_ft.keys())  # noqa
            action_dict = {k: full_action[i] for i, k in enumerate(joint_features)}

            try:
                if self.smooth_streaming:
                    from robot_control import send_action_smooth  # noqa

                    send_action_smooth(self.robot, full_action, self.dt)
                else:
                    self.robot.send_action(action_dict)
            except Exception as exc:
                logger.error(f"Firmware error executing action: {exc}. Moving to sleep position.")
                if getattr(self, "sink", None):
                    self.sink.on_status("firmware_error", {"message": str(exc)})
                try:
                    self.move_to_sleep_position(duration=10.0)
                finally:
                    self.is_running = False
                return
        else:
            logger.error(f"Unknown mode: {self.test_mode}. No action executed.")

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
        self.sink.on_images(
            {cam: cv2.imencode(".jpg", observation_dict[cam])[1].tobytes() for cam in cameras}, time.time()
        )
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

        logger.info(f"Moving to sleep position over {duration}s...")
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

        logger.info("Reached sleep position.")

    def run_episode(self, task_prompt: str = "look down"):
        """Run a single episode of policy execution."""
        logger.info(f"Starting episode with prompt: '{task_prompt}'")
        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.is_running = True
        fallback = HoldLastAction()
        # Log forwarding is attached at the session level (webapp.session) for the
        # whole run; attaching here too would double every line in the UI.
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
                    a_t = fallback.resolve(self.ensemble.get_action(self.episode_step))
                    if a_t is None:
                        # no prediction yet and no prior action — skip this step
                        self.episode_step += 1
                        continue

                else:
                    # Synchronous: request new chunk every rate_of_inference steps
                    if self.current_action_chunk is None or self.action_chunk_idx >= self.rate_of_inference:
                        obs_raw = self.robot.get_observation()
                        observation = self._build_observation(obs_raw, task_prompt)
                        logger.info(f"Step {self.episode_step}: Requesting new action chunk")
                        _t0 = time.perf_counter()
                        response = self.policy_client.infer(observation)
                        self.sink.on_inference((time.perf_counter() - _t0) * 1e3, time.time())
                        current_joints14 = extract_joints(obs_raw)
                        self.current_action_chunk = self.adapter.decode_chunk(response["actions"], current_joints14)
                        if self.ensemble is not None:
                            self.ensemble.add_chunk(self.episode_step, self.current_action_chunk)
                            self.sink.on_chunk(self.episode_step, self.current_action_chunk, time.time())
                        self.action_chunk_idx = 0
                        logger.info(f"Received action chunk: {self.current_action_chunk.shape}")

                    if self.ensemble is not None:
                        a_t = fallback.resolve(self.ensemble.get_action(self.episode_step))
                        if a_t is None:
                            self.episode_step += 1
                            continue
                    else:
                        a_t = self.current_action_chunk[self.action_chunk_idx]

                if self.action_logger is not None and self.ensemble is not None:
                    self.action_logger.log(
                        self.episode_step,
                        self.ensemble.get_overlap_count(self.episode_step),
                    )

                if self.ensemble is not None:
                    self.sink.on_overlap(self.episode_step, self.ensemble.get_overlap_count(self.episode_step))
                    self.sink.on_status("buffer", {"size": self.ensemble.buffer_size()})
                    _w = self.ensemble.last_weights()
                    if _w is not None:
                        self.sink.on_weights(self.episode_step, _w, time.time())
                self.sink.on_action(self.episode_step, np.asarray(a_t), time.time())

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
