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
from collections.abc import Callable
import logging
from pathlib import Path
import time

from action_ensemble import ActionLogger
from action_ensemble import AsyncPolicyWorker
from action_ensemble import LatestChunkEnsemble
from action_ensemble import RTCAsyncPolicyWorker
from action_ensemble import make_ensemble
from action_ensemble import measured_pose_hold
from action_ensemble import validated_action_chunk
from action_ensemble import validate_rtc_response_query
from action_ensemble import validate_rtc_server_metadata
import cv2
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig
import numpy as np
from openpi_client import websocket_client_policy
from PIL import Image
from realsense_settings import apply_realsense_settings
from scipy.interpolate import PchipInterpolator
from scripted_motions import ARMS
from scripted_motions import BIMANUAL_DIM
from scripted_motions import HELP_ROWS
from scripted_motions import ScriptedMotions
from scripted_motions import parse_command
import terminal_ui

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TRAINING_SIZE = (224, 224)
DEFAULT_REALSENSE_SETTINGS = Path(__file__).parent / "realsense_settings.json"
# How often the control loop logs its rate when there is no pinned status line.
# Anything faster competes with the operator typing instructions into the terminal.
RATE_SUMMARY_PERIOD_S = 5.0
# The pinned status line costs no log lines, so it can refresh briskly.
RATE_STATUS_PERIOD_S = 1.0
FIRST_INFERENCE_MAX_ATTEMPTS = 3


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
        rtc: bool = False,
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
        self.rtc_enabled = bool(rtc)

        logger.info(f"Connecting to policy server at {policy_server_host}:{policy_server_port}")
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_server_host, port=policy_server_port
        )
        if self.rtc_enabled:
            server_metadata = self.policy_client.get_server_metadata() or {}
            # --rtc is explicit opt-in. Validate before constructing or
            # connecting the robot; without --rtc no checkpoint behavior or
            # metadata handling changes.
            if not validate_rtc_server_metadata(
                server_metadata,
                rate_of_inference=rate_of_inference,
                async_inference=async_inference,
            ):
                raise RuntimeError(
                    "--rtc was requested, but the policy server did not advertise RTC. "
                    "Start deployment.model_server.server_policy with --rtc."
                )
            logger.info(
                "RTC protocol v2 enabled (server action horizon: %s)",
                server_metadata.get("action_horizon", "unknown"),
            )

        robot_config = BiWidowXAIFollowerRobotConfig(
            id="bimanual_follower",
            left_arm_ip_address="192.168.1.5",
            right_arm_ip_address="192.168.1.4",
            min_time_to_move_multiplier=3.0,
            loop_rate=30,
            cameras={
                "cam_high": OpenCVCameraConfig(index_or_path=Path("/dev/video40"), width=640, height=480, fps=30),
                # "cam_low": RealSenseCameraConfig(
                #     serial_number_or_name="130322272628", width=640, height=480, fps=30, use_depth=False
                # ),
                "cam_right_wrist": OpenCVCameraConfig(
                    index_or_path=Path("/dev/video41"), width=640, height=480, fps=30
                ),
                "cam_left_wrist": OpenCVCameraConfig(index_or_path=Path("/dev/video42"), width=640, height=480, fps=30),
            },
        )
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()

        # Push the recorded exposure/gain/white balance onto the cameras so inference sees the
        # same image the policy was trained on, instead of whatever auto-exposure settles on.
        apply_realsense_settings(DEFAULT_REALSENSE_SETTINGS)

        self.current_action_chunk = None
        self.action_chunk_idx = 0

        self._rate_window: list[float] = []
        self._last_rate_log = time.perf_counter()
        self._last_action: np.ndarray | None = None
        self._prompt_listener: terminal_ui.BasePromptListener | None = None

        self.action_chunk_size = action_chunk_size
        self.episode_step = 0
        self.is_running = False

        self.rate_of_inference = rate_of_inference  # Number of control steps per policy inference
        self.action_dim = len(self.robot._joint_ft)  # 7 joints per arm * 2 arms
        if self.rtc_enabled:
            if ensemble_type != "none":
                logger.info(
                    "RTC controls temporal consistency; replacing --ensemble_type %s "
                    "with an atomic latest-chunk queue",
                    ensemble_type,
                )
            self.ensemble = LatestChunkEnsemble()
        else:
            self.ensemble = make_ensemble(ensemble_type)

        if async_inference and self.ensemble is None:
            raise ValueError("--async_inference requires an ensemble (ensemble_type cannot be 'none')")
        self.async_inference = async_inference
        if async_inference and self.rtc_enabled:
            self._policy_worker = RTCAsyncPolicyWorker(
                self.policy_client,
                self.ensemble,
                self.action_dim,
                control_frequency=self.control_frequency,
            )
        elif async_inference:
            # Preserve the original worker for every non-RTC checkpoint.
            self._policy_worker = AsyncPolicyWorker(
                self.policy_client,
                self.ensemble,
                self.action_dim,
            )
        else:
            self._policy_worker = None
        self._rtc_sync_reset_pending = self.rtc_enabled

        self.action_logger = ActionLogger(log_dir) if log_dir else None

        self.use_left_arm_only = use_left_arm_only
        self.use_right_arm_only = use_right_arm_only

        # Keyword-triggered canned motions (home / grippers / wrist twist / wave).
        # They assume the 14-dim bimanual layout, so they stay off for anything else.
        if self.action_dim == BIMANUAL_DIM:
            enabled_arms = ARMS
            if use_left_arm_only:
                enabled_arms = ("left",)
            elif use_right_arm_only:
                enabled_arms = ("right",)
            self.motions = ScriptedMotions(
                get_pose=self._read_joint_pose,
                send_pose=self._send_scripted_pose,
                control_frequency=control_frequency,
                joint_limits=self._read_joint_limits(),
                enabled_arms=enabled_arms,
            )
        else:
            self.motions = None
            logger.warning("action_dim is %d, not %d — keyword motions disabled", self.action_dim, BIMANUAL_DIM)

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

    def _log_rate_summary(self, loop_s: float) -> None:
        """Report the control rate without one line per control step.

        With the pinned UI the rate goes to the status line, which costs no log
        lines at all. Otherwise it is logged at most once every
        RATE_SUMMARY_PERIOD_S — per-step logging makes the terminal unusable for
        typing instructions.
        """
        # A step that ran a blocking ramp or scripted motion took seconds, not
        # milliseconds — keeping it would wreck the average for the whole window.
        if loop_s < 5 * self.dt:
            self._rate_window.append(loop_s)
        pinned = self._prompt_listener is not None and self._prompt_listener.pinned
        period = RATE_STATUS_PERIOD_S if pinned else RATE_SUMMARY_PERIOD_S
        now = time.perf_counter()
        if now - self._last_rate_log < period or not self._rate_window:
            return
        mean_s = sum(self._rate_window) / len(self._rate_window)
        if pinned:
            self._prompt_listener.set_rate(1 / mean_s)
        else:
            message = f"step {self.episode_step}: {mean_s * 1e3:.1f}ms/step ({1 / mean_s:.0f} Hz avg)"
            if self.test_mode == "test" and self._last_action is not None:
                action = np.array2string(self._last_action, precision=3, max_line_width=np.inf, suppress_small=True)
                message += f" | TEST MODE, last action: {action}"
            logger.info(message)
        self._reset_rate_window()

    def _reset_rate_window(self) -> None:
        """Drop collected timings. Called after a blocking move so the multi-second
        ramp doesn't get averaged into the control rate."""
        self._rate_window.clear()
        self._last_rate_log = time.perf_counter()

    def _read_joint_pose(self) -> np.ndarray:
        """Current measured joint pose, in the same order as `robot._joint_ft`."""
        observation = self.robot.get_observation()
        return np.array([v for k, v in observation.items() if k.endswith(".pos")])

    def _read_joint_limits(self) -> np.ndarray | None:
        """Per-joint [min, max] straight from the driver, so scripted motions can
        clamp instead of hardcoding a gripper stroke that varies per end effector."""
        try:
            limits = [
                (joint.position_min, joint.position_max)
                for arm in (self.robot.left_arm, self.robot.right_arm)
                for joint in arm.driver.get_joint_limits()
            ]
            array = np.array(limits, dtype=float)
            if array.shape != (self.action_dim, 2):
                raise ValueError(f"expected ({self.action_dim}, 2) limits, got {array.shape}")
        except Exception:
            logger.warning("Could not read joint limits from the driver — using fallback values", exc_info=True)
            return None
        return array

    def _send_scripted_pose(self, pose: np.ndarray):
        """Send one pose from a scripted motion.

        Bypasses execute_action()'s arm freezing on purpose: the operator asked
        for this motion explicitly, and ScriptedMotions already refuses commands
        for an arm disabled by --use_left_arm_only / --use_right_arm_only.
        """
        if self.test_mode == "test":
            logger.debug(f"TEST MODE: Would send scripted pose: {pose}")
            return
        joint_features = list(self.robot._joint_ft.keys())
        self.robot.send_action({k: pose[i] for i, k in enumerate(joint_features)})

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
            # Per-step, so it stays at DEBUG — at 25 Hz it buries the log (and any
            # instruction you are typing). Run with --debug to see every action.
            logger.debug(f"TEST MODE: Would execute action: {full_action}")
            self._last_action = full_action
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

    def _dispatch_rtc_action(
        self,
        action: np.ndarray,
        prompt_listener: terminal_ui.BasePromptListener | None,
    ) -> bool:
        """Dispatch one RTC action atomically with respect to prompt arrival."""
        if prompt_listener is None:
            self.execute_action(action)
            return True
        return prompt_listener.dispatch_if_current(lambda: self.execute_action(action))

    def move_to_start_position(
        self,
        goal_position: np.ndarray,
        duration: float = 5.0,
        should_stop: Callable[[], bool] | None = None,
        dispatch_action: Callable[[np.ndarray], bool] | None = None,
    ) -> bool:
        """The first position queried from the policy depends on the training data.
        Assuming the first position is a "stage" position will result in a large jump if the arm is not already there.
        To avoid this, we smoothly move the arm to a first action/position before sending the rest of the actions.
        We use PCHIP interpolation for smooth trajectory generation and give it enough time to reach the position to prevent
        jumps and triggering safety stops (velocity limits).

        Returns ``False`` when newly submitted operator input invalidates the
        target and interrupts the ramp.
        """

        self._frozen_arm_pose = self._read_joint_pose()
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
            if should_stop is not None and should_stop():
                return False
            loop_start_time = time.perf_counter()
            current_time = time.time() - start_time
            positions = interpolator_position(current_time)
            if dispatch_action is None:
                self.execute_action(positions)
            elif not dispatch_action(positions):
                return False
            # Rate-limit to the control frequency: without this the ramp hammers the
            # driver as fast as the loop spins, which now happens on every resume.
            remaining = self.dt - (time.perf_counter() - loop_start_time)
            if remaining > 0:
                time.sleep(remaining)
        if should_stop is not None and should_stop():
            return False
        return True

    def run_episode(self, task_prompt: str = "look down"):
        """Run a single episode of policy execution."""
        logger.info(f"Starting episode with prompt: '{task_prompt}'")
        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.is_running = True
        self._rtc_sync_reset_pending = self.rtc_enabled
        if self.ensemble is not None:
            self.ensemble.reset()
        if self.action_logger is not None:
            self.action_logger.reset()
        is_first_step = True

        if self.use_left_arm_only or self.use_right_arm_only:
            _obs = self.robot.get_observation()
            _joint_pos_keys = [k for k in _obs.keys() if k.endswith(".pos")]
            self._frozen_arm_pose = np.array([_obs[k] for k in _joint_pos_keys])

        prompt_listener = None
        paused = False
        if self.async_inference:
            self._policy_worker.start()
            if self.motions is not None:
                terminal_ui.print_help(HELP_ROWS)

            if self.rtc_enabled:
                def invalidate_rtc_on_submit(typed: str) -> bool:
                    """Invalidate RTC predictions in the input thread, before poll()."""
                    command = parse_command(typed) if self.motions is not None else None
                    interrupts_policy = (
                        command is None
                        or command.moves_arm
                        or command.name == "quit"
                    )
                    if interrupts_policy:
                        self._policy_worker.flush()
                    return interrupts_policy

                prompt_listener = terminal_ui.make_prompt_listener(
                    task_prompt,
                    on_submit=invalidate_rtc_on_submit,
                )
            else:
                # Original listener semantics for every other checkpoint.
                prompt_listener = terminal_ui.make_prompt_listener(task_prompt)
            prompt_listener.start()
        self._prompt_listener = prompt_listener

        try:
            while self.is_running and self.episode_step < self.max_steps:
                start_loop_time = time.perf_counter()

                if self.async_inference:
                    # Pick up whatever was typed on stdin since the last step: either
                    # a keyword command (scripted motion, pauses the policy) or a new
                    # task instruction.
                    typed = prompt_listener.poll() if prompt_listener is not None else None
                    if typed is not None:
                        command = parse_command(typed) if self.motions is not None else None
                        if command is not None and command.name == "quit":
                            # Leave the loop so the finally below stops the worker and
                            # listener; cleanup() then parks the arms and closes the
                            # cameras through robot.disconnect().
                            logger.info("Quit requested — shutting down")
                            self.is_running = False
                            break
                        if command is not None and command.name == "help":
                            terminal_ui.print_help(HELP_ROWS)
                        elif command is not None:
                            if command.moves_arm:
                                # Stop feeding policy actions before driving the arm
                                # ourselves, and throw away predictions made for the
                                # pose/instruction the motion is about to invalidate.
                                paused = True
                                if not self.rtc_enabled:
                                    self.ensemble.reset()
                                self._policy_worker.flush()
                            prompt_listener.set_paused(paused)
                            self.motions.run(command)
                            self._reset_rate_window()
                            if paused:
                                logger.info("Policy paused — type an instruction (Enter alone = default) to resume")
                        else:
                            task_prompt = typed
                            prompt_listener.set_task(task_prompt)
                            logger.info(f"Task instruction: '{task_prompt}'")
                            if self.rtc_enabled:
                                # Explicit RTC mode makes every prompt a new
                                # epoch. Clear committed/in-flight chunks now,
                                # hold measured pose, and reset the server on
                                # the next request.
                                self._policy_worker.flush()
                                self._dispatch_rtc_action(
                                    self._read_joint_pose(), prompt_listener
                                )
                                is_first_step = True
                                # Log this on every path, including resuming from a
                                # scripted motion (e.g. "home" then a new prompt) —
                                # that is exactly the sequence where confirming the
                                # server-side RTC cache was actually cleared matters
                                # most, since the arm's pose just changed underneath
                                # whatever chunk was previously in flight.
                                logger.info(
                                    "Cleared RTC prompt cache; waiting for a fresh chunk"
                                )
                                if paused:
                                    logger.info("Resuming policy")
                                    paused = False
                                    prompt_listener.set_paused(paused)
                            else:
                                # Preserve the original non-RTC behavior: only
                                # a scripted-motion pause starts a fresh epoch;
                                # a live prompt switch keeps temporal blending.
                                if paused:
                                    logger.info("Resuming policy")
                                    paused = False
                                    is_first_step = True
                                    prompt_listener.set_paused(paused)

                        # Preserve and process every queued line in order. This
                        # is what makes rapid `home` then `<new task>` reliable.
                        if self.rtc_enabled and prompt_listener.has_pending():
                            continue

                    if paused:
                        time.sleep(0.05)
                        continue

                    # The input thread invalidates the worker as soon as an
                    # instruction is submitted. Do not issue another action
                    # while the control loop is waiting to poll that line.
                    if self.rtc_enabled and prompt_listener.has_interrupting_input():
                        continue

                    if self.rtc_enabled and self._policy_worker.consume_restart_required():
                        # A failed/malformed inference invalidates both client
                        # and server RTC epochs. Wait for a reset response and
                        # ramp to it instead of switching to it mid-stream.
                        logger.warning("Policy inference failed; waiting for a fresh reset chunk")
                        self._policy_worker.flush()
                        is_first_step = True

                    # Submit fresh observation every step — non-blocking
                    obs = self._build_observation(self.robot.get_observation(), task_prompt)
                    self._policy_worker.submit(obs, self.episode_step)
                    if is_first_step:
                        logger.info("Waiting for first inference result...")
                        if not self.rtc_enabled:
                            if not self._policy_worker.wait_for_first(timeout=30.0):
                                logger.error("Timed out waiting for first inference — aborting")
                                break
                        else:
                            fresh_chunk_ready = False
                            interrupted_by_input = False
                            for attempt in range(1, FIRST_INFERENCE_MAX_ATTEMPTS + 1):
                                deadline = time.monotonic() + 30.0
                                first_result_arrived = False
                                while time.monotonic() < deadline:
                                    if prompt_listener.has_interrupting_input():
                                        interrupted_by_input = True
                                        break
                                    remaining = deadline - time.monotonic()
                                    if self._policy_worker.wait_for_first(
                                        timeout=min(0.05, max(remaining, 0.0))
                                    ):
                                        if prompt_listener.has_interrupting_input():
                                            interrupted_by_input = True
                                        else:
                                            first_result_arrived = True
                                        break

                                if interrupted_by_input:
                                    logger.info("Operator input interrupted the pending RTC start")
                                    break
                                if not first_result_arrived:
                                    logger.error("Timed out waiting for first RTC inference")
                                    break
                                if not self._policy_worker.consume_restart_required():
                                    fresh_chunk_ready = True
                                    break

                                self._policy_worker.flush()
                                if attempt == FIRST_INFERENCE_MAX_ATTEMPTS:
                                    logger.error("First RTC inference failed %d times", attempt)
                                    break
                                logger.warning(
                                    "Retrying first RTC inference with a server reset (%d/%d)",
                                    attempt + 1,
                                    FIRST_INFERENCE_MAX_ATTEMPTS,
                                )
                                obs = self._build_observation(
                                    self.robot.get_observation(), task_prompt
                                )
                                self._policy_worker.submit(obs, self.episode_step)

                            if interrupted_by_input:
                                continue
                            if not fresh_chunk_ready:
                                logger.error("Could not obtain a fresh RTC chunk — aborting")
                                break
                    if self.rtc_enabled and prompt_listener.has_interrupting_input():
                        continue
                    a_t = self.ensemble.get_action(self.episode_step)
                    if a_t is None:
                        if self.rtc_enabled:
                            # Absolute zero is a motion command, not a neutral
                            # RTC fallback. Hold the measured request pose.
                            a_t = measured_pose_hold(obs, self.action_dim)
                        else:
                            a_t = np.zeros(self.action_dim)

                else:
                    # Synchronous: request new chunk every rate_of_inference steps
                    if self.current_action_chunk is None or self.action_chunk_idx >= self.rate_of_inference:
                        observation = self._build_observation(self.robot.get_observation(), task_prompt)
                        if self.rtc_enabled:
                            # Synchronous inference blocks the control loop, so
                            # zero action steps execute between request and reply.
                            observation["rtc_query_step"] = self.episode_step
                            observation["rtc_inference_delay"] = 0
                            observation["rtc_reset"] = self._rtc_sync_reset_pending
                        logger.info(f"Step {self.episode_step}: Requesting new action chunk")
                        response = self.policy_client.infer(observation)
                        if self.rtc_enabled:
                            validate_rtc_response_query(response, self.episode_step)
                            self._rtc_sync_reset_pending = False
                            self.current_action_chunk = validated_action_chunk(
                                response["actions"], self.action_dim
                            )
                        else:
                            self.current_action_chunk = response["actions"][:, : self.action_dim]
                        if self.ensemble is not None:
                            self.ensemble.add_chunk(self.episode_step, self.current_action_chunk)
                        self.action_chunk_idx = 0
                        logger.info(f"Received action chunk: {self.current_action_chunk.shape}")

                    if self.ensemble is not None:
                        a_t = self.ensemble.get_action(self.episode_step)
                        if a_t is None:
                            if self.rtc_enabled:
                                a_t = measured_pose_hold(
                                    {"state": self._read_joint_pose()}, self.action_dim
                                )
                            else:
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
                if (
                    self.raw_gripper
                    and not self.rtc_enabled
                    and self.ensemble is not None
                    and self.gripper_indices
                ):
                    raw = self.ensemble.get_latest_raw(self.episode_step)
                    if raw is not None:
                        for gi in self.gripper_indices:
                            if gi < len(a_t) and gi < len(raw):
                                a_t[gi] = raw[gi]

                if is_first_step:
                    logger.info("Moving to start position to avoid large jumps...")
                    if self.rtc_enabled:
                        completed = self.move_to_start_position(
                            a_t,
                            duration=5.0,
                            should_stop=(
                                prompt_listener.has_interrupting_input
                                if prompt_listener is not None
                                else None
                            ),
                            dispatch_action=lambda positions: self._dispatch_rtc_action(
                                positions, prompt_listener
                            ),
                        )
                        if not completed or (
                            prompt_listener is not None
                            and prompt_listener.has_interrupting_input()
                        ):
                            logger.info("Operator input interrupted the RTC start ramp")
                            continue
                    else:
                        self.move_to_start_position(a_t, duration=5.0)
                    is_first_step = False
                    self._reset_rate_window()
                else:
                    if self.rtc_enabled:
                        if not self._dispatch_rtc_action(a_t, prompt_listener):
                            continue
                    else:
                        self.execute_action(a_t)

                self.action_chunk_idx += 1
                self.episode_step += 1

                dt_s = time.perf_counter() - start_loop_time
                if self.dt - dt_s > 0:
                    time.sleep(self.dt - dt_s)
                loop_s = time.perf_counter() - start_loop_time
                # One line per control step is 25 lines/s — it drowns the log and
                # scrolls away whatever you are typing. Summarise instead.
                logger.debug(f"time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
                self._log_rate_summary(loop_s)

        finally:
            if self.async_inference:
                self._policy_worker.stop()
            if prompt_listener is not None:
                prompt_listener.stop()
            if self.action_logger is not None:
                self.action_logger.save(tag=f"episode_{self.episode_step}steps")

        self.is_running = False
        logger.info(f"Episode completed after {self.episode_step} steps")

    def autonomous_mode(self, task_prompt: str = "look down"):
        """Run in autonomous mode where the arm executes policy predictions."""
        logger.info("Starting autonomous mode")
        self.run_episode(task_prompt=task_prompt)

    def cleanup(self):
        """Disconnect the arms and release the cameras.

        robot.disconnect() parks both arms (staged pose, then all joints to zero)
        before closing the cameras — but it closes them *after* the arms, so a
        failing arm would otherwise leave the cameras held open and the next run
        unable to grab them.
        """
        logger.info("Cleaning up...")
        try:
            self.robot.disconnect()
        except Exception:
            logger.exception("Robot disconnect failed — releasing cameras directly")
            for name, camera in self.robot.cameras.items():
                try:
                    camera.disconnect()
                except Exception:
                    logger.warning("Could not release camera %s", name, exc_info=True)


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
        "Without --rtc this requires an ensemble (not 'none'); CogACT is recommended.",
    )
    parser.add_argument(
        "--rtc",
        action="store_true",
        help=(
            "Opt in to RTC v2 timing, cache reset, and atomic chunk execution. "
            "The policy server must also be started with --rtc."
        ),
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log every control step (per-step timing and, in test mode, every action). "
        "Off by default because the stream makes typing instructions impossible.",
    )
    args = parser.parse_args()

    terminal_ui.setup_logging(debug=args.debug)

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
        rtc=args.rtc,
        log_dir=args.log_dir,
        use_left_arm_only=args.use_left_arm_only,
        use_right_arm_only=args.use_right_arm_only,
        starvla=args.starvla,
        raw_gripper=not args.ensemble_gripper,
    )

    try:
        bridge.autonomous_mode(task_prompt=args.task_prompt)
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")
    finally:
        # Ctrl+C and crashes must still park the arms and release the cameras,
        # otherwise the next run finds the devices busy.
        bridge.cleanup()
