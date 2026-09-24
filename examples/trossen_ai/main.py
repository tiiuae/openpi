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
from pathlib import Path
import time

from action_ensemble import ActionLogger
from action_ensemble import AsyncPolicyWorker
from action_ensemble import make_ensemble
import cv2
from episode_recorder import EpisodeRecorder
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig
import motion_limits
from motion_limits import JointLimitError
import numpy as np
from openpi_client import websocket_client_policy
from PIL import Image
from policy_reply import validate_actions
from realsense_settings import apply_realsense_settings
from scipy.interpolate import PchipInterpolator
from scripted_motions import ARMS
from scripted_motions import BIMANUAL_DIM
from scripted_motions import GRIPPER_IDX
from scripted_motions import HELP_ROWS
from scripted_motions import JOINTS_PER_ARM
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
# How long the control loop will hold its pose waiting for the policy to produce
# an action for the current step before it stops driving and pauses. One chunk at
# 25 Hz is about a second, so half a second of nothing means the policy is behind
# or the server is gone, not that a step was merely late.
STARVED_HOLD_BUDGET_S = 0.5


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
        allow_unlimited_motion: bool = False,
        starvla: bool = False,
        raw_gripper: bool = True,
        gripper_indices: list[int] | None = None,
        record_dir: str | None = None,
        record_repo_id: str = "local/trossen_eval",
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
        self._starved_steps = 0
        self._prompt_listener: terminal_ui.BasePromptListener | None = None

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

        # Episode recording is driven by typed commands (record/hold/save/reject),
        # which only the async prompt listener delivers.
        self._current_task = ""
        self._recorder = None
        if record_dir is not None:
            if not async_inference:
                raise ValueError("--record_dir requires --async_inference (recording is controlled by typed commands)")
            self._recorder = EpisodeRecorder(self.robot, fps=control_frequency, root=record_dir, repo_id=record_repo_id)

        self.use_left_arm_only = use_left_arm_only
        self.use_right_arm_only = use_right_arm_only

        # Per-joint [min, max] position and max velocity, straight from the driver.
        # Every pose that reaches the arm is checked against both by _limit_and_send(),
        # whichever source it came from (policy step, start-position ramp, scripted
        # motion). A table that fails validation does not silently disable the check
        # it was protecting: motion is refused instead (see _motion_permitted below).
        self.joint_limits = self._read_joint_limits()
        self.joint_velocity_limits = self._read_joint_velocity_limits()
        self._limit_warn_deadline = 0.0

        self._motion_permitted = True
        if self.joint_limits is None or self.joint_velocity_limits is None:
            if allow_unlimited_motion:
                logger.error(
                    "MOTION LIMITS UNAVAILABLE and --allow_unlimited_motion was passed — the arm will be "
                    "commanded with NO velocity or position checks. The firmware fault is the only thing "
                    "left between a bad action and the hardware."
                )
            else:
                self._motion_permitted = False
                logger.error(
                    "MOTION LIMITS UNAVAILABLE — refusing to command the arm. Observation, inference and "
                    "recording still run, so this session is read-only. Fix the driver connection, or pass "
                    "--allow_unlimited_motion to command the arm without limit checks anyway."
                )

        # Keyword-triggered canned motions (home / grippers / wrist twist / wave).
        # They assume the 14-dim bimanual layout, so they stay off for anything else.
        # The arm-only flags gate the POLICY stream, not operator-commanded
        # scripted motions: "sleep" and "home" must park BOTH arms whichever
        # arm the policy is restricted to, so all arms stay enabled here.
        if self.action_dim == BIMANUAL_DIM:
            self.motions = ScriptedMotions(
                get_pose=self._read_joint_pose,
                send_pose=self._send_scripted_pose,
                control_frequency=control_frequency,
                joint_limits=self.joint_limits,
                velocity_limits=self.joint_velocity_limits,
                enabled_arms=ARMS,
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
            if self._recorder is not None and self._recorder.is_recording:
                message += f" | ● REC {self._recorder.steps} steps"
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
        """Per-joint [min, max] position, straight from the driver and validated.

        None means "do not trust any position limit": the caller refuses to move
        the arm rather than carrying on with the clamp quietly switched off.
        """
        try:
            limits = [
                (joint.position_min, joint.position_max)
                for arm in (self.robot.left_arm, self.robot.right_arm)
                for joint in arm.driver.get_joint_limits()
            ]
            array = motion_limits.validate_position_limits(limits, self.action_dim)
        except JointLimitError as error:
            logger.error("Driver position limits are unusable: %s", error)
            return None
        except Exception:
            logger.error("Could not read position limits from the driver", exc_info=True)
            return None
        # The gripper carriages are the pair most likely to be miscalibrated per
        # end effector, so their stroke is worth seeing in the startup log.
        for index in (GRIPPER_IDX, JOINTS_PER_ARM + GRIPPER_IDX):
            if index < len(array):
                logger.info("Driver joint %d position limits: [%.6g, %.6g]", index, *array[index])
        return array

    def _read_joint_velocity_limits(self) -> np.ndarray | None:
        """Per-joint max velocity (rad/s; m/s for the gripper carriage), validated."""
        try:
            limits = [
                joint.velocity_max
                for arm in (self.robot.left_arm, self.robot.right_arm)
                for joint in arm.driver.get_joint_limits()
            ]
            return motion_limits.validate_velocity_limits(limits, self.action_dim)
        except JointLimitError as error:
            logger.error("Driver velocity limits are unusable: %s", error)
            return None
        except Exception:
            logger.error("Could not read velocity limits from the driver", exc_info=True)
            return None

    def _warn_throttled(self, message: str, *args) -> None:
        """Log a limit warning at most once a second.

        The limiter can fire on every step of a 25 Hz loop, and a warning per
        step buries the log and scrolls away whatever the operator is typing.
        """
        now = time.perf_counter()
        if now >= self._limit_warn_deadline:
            self._limit_warn_deadline = now + 1.0
            logger.warning(message, *args)

    def _limit_and_send(self, pose: np.ndarray, *, freeze_disabled_arms: bool) -> np.ndarray:
        """The one path from a desired pose to the driver.

        Policy steps, start-position ramps and scripted motions all come through
        here, so a limit applies to the arm rather than to one code path: a
        scripted "home" cannot be faster than a policy step is allowed to be.
        Returns the pose actually commanded, which is what the episode recorder
        must store — a clamped action that is recorded unclamped teaches the
        trajectory a move the arm never made.
        """
        target = np.asarray(pose, dtype=float).copy()

        if freeze_disabled_arms and (self.use_left_arm_only or self.use_right_arm_only):
            if self.use_right_arm_only:
                # Freeze left arm (indices 0:7) at current pose; only right arm (7:14) moves
                target[:7] = self._frozen_arm_pose[:7]
            elif self.use_left_arm_only:
                # Freeze right arm (indices 7:14) at current pose; only left arm (0:7) moves
                target[7:] = self._frozen_arm_pose[7:]

        # Velocity first, then position: scaling a step down can only shorten it,
        # so it can never push a joint back out of the range the clamp just fixed.
        if self.joint_velocity_limits is not None and self._last_action is not None:
            target, factor, over = motion_limits.scale_to_velocity_limits(
                self._last_action, target, self.dt, self.joint_velocity_limits
            )
            if over:
                self._warn_throttled(
                    "Step exceeds the velocity limit on joints %s — slowing the whole step to %.0f%% "
                    "(scaling only the fast joints would bend the path)",
                    over,
                    factor * 100.0,
                )

        if self.joint_limits is not None:
            target, deviation = motion_limits.clamp_to_position_limits(target, self.joint_limits)
            out_of_range = np.where(deviation > motion_limits.CLAMP_WARNING_TOLERANCE)[0]
            if out_of_range.size:
                self._warn_throttled(
                    "Step exceeds position limits on joints %s (by up to %.4g) — clamping to range",
                    out_of_range.tolist(),
                    deviation[out_of_range].max(),
                )

        if not self._motion_permitted:
            # No trustworthy limits: hold. Reported once per second rather than
            # per step so the reason stays visible without burying the log.
            self._warn_throttled("Motion refused: no validated joint limits (see startup log)")
            return self._last_action if self._last_action is not None else target

        if self.test_mode == "test":
            # Per-step, so it stays at DEBUG — at 25 Hz it buries the log (and any
            # instruction you are typing). Run with --debug to see every action.
            logger.debug(f"TEST MODE: Would command pose: {target}")
            self._last_action = target
            return target
        if self.test_mode != "autonomous":
            logger.error(f"Unknown mode: {self.test_mode}. No action executed.")
            return target

        joint_features = list(self.robot._joint_ft.keys())
        self.robot.send_action({k: target[i] for i, k in enumerate(joint_features)})
        self._last_action = target
        return target

    def _hold_pose(self, *, paused: bool) -> tuple[np.ndarray, bool]:
        """Pose to command when the policy has nothing for this step.

        This used to be ``np.zeros(action_dim)``. Zero is not "do nothing": it
        is an absolute pose — every joint folded down and both grippers shut —
        so a starved step commanded a full sweep to the rest pose from wherever
        the arm was, closing the gripper on whatever it was holding. Velocity
        scaling and position clamping slow that move down; they do not change
        where it is going.

        Holding the last commanded pose is always safe, and it keeps the
        driver's goal stream fed while the policy catches up. If the starvation
        lasts past STARVED_HOLD_BUDGET_S the loop pauses instead, so the arm
        sits still under a visible reason rather than silently stalling.

        Returns ``(pose, paused)``.
        """
        self._starved_steps += 1
        pose = self._last_action if self._last_action is not None else self._read_joint_pose()

        budget = max(int(STARVED_HOLD_BUDGET_S * self.control_frequency), 1)
        if not paused and self._starved_steps >= budget:
            logger.warning(
                "No policy action for %d consecutive steps (%.1fs) — holding position and pausing. "
                "Check the policy server, then type an instruction to resume.",
                self._starved_steps,
                self._starved_steps * self.dt,
            )
            if self.async_inference:
                self.ensemble.reset()
                self._policy_worker.flush()
            return pose, True

        if self._starved_steps == 1:
            logger.info("No policy action for this step — holding the last commanded pose")
        return pose, paused

    def _send_scripted_pose(self, pose: np.ndarray):
        """Send one pose from a scripted motion.

        Skips the arm-only freeze on purpose: the operator asked for this motion
        explicitly, and ScriptedMotions already refuses commands for an arm
        disabled by --use_left_arm_only / --use_right_arm_only. The velocity and
        position limits in _limit_and_send() are NOT skipped.
        """
        # Scripted motions are recorded like policy steps (labeled with the
        # current task instruction) — a gripper fix mid-take must appear in the
        # episode or the saved trajectory teleports.
        recording = self._recorder is not None and self._recorder.is_recording
        observation = self.robot.get_observation() if recording else None
        sent = self._limit_and_send(pose, freeze_disabled_arms=False)
        if recording:
            self._recorder.add(observation, sent, self._current_task)

    def execute_action(self, action: np.ndarray) -> np.ndarray:
        """Execute one policy action. Returns the pose actually sent (after arm
        freezing, velocity scaling and position clamping), which is what the
        episode recorder must store."""
        return self._limit_and_send(action, freeze_disabled_arms=True)

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

        self._frozen_arm_pose = self._read_joint_pose()
        # Anchor the velocity-limit reference to the arm's real current pose, so the
        # very first ramp sample (and thus the whole ramp) is velocity-checked too.
        # Without this the first sample is compared against nothing and goes out
        # unlimited, which is exactly the jump this ramp exists to avoid.
        self._last_action = self._frozen_arm_pose
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
            loop_start_time = time.perf_counter()
            current_time = time.time() - start_time
            positions = interpolator_position(current_time)
            # A ramp can run mid-take (resuming the policy after a scripted
            # motion) — record it like any other motion, or the episode jumps.
            recording = self._recorder is not None and self._recorder.is_recording
            raw_obs = self.robot.get_observation() if recording else None
            executed = self.execute_action(positions)
            if recording:
                self._recorder.add(raw_obs, executed, self._current_task)
            # Rate-limit to the control frequency: without this the ramp hammers the
            # driver as fast as the loop spins, which now happens on every resume.
            remaining = self.dt - (time.perf_counter() - loop_start_time)
            if remaining > 0:
                time.sleep(remaining)

    def _handle_recording_command(self, name: str, *, paused: bool) -> bool:
        """Handle a typed record/save/reject. Returns the loop's paused flag:
        save/reject close the current take, so they pause the policy — the
        operator is deciding about the take, not driving the arm. 'record'
        starts a take without interrupting the running policy."""
        recorder = self._recorder
        if recorder is None:
            logger.warning("Recording is disabled — relaunch with --record_dir to enable it")
            return paused
        if name == "record":
            if recorder.is_pending:
                logger.warning("A take is pending — 'save' or 'reject' it before recording again")
            elif recorder.is_recording:
                logger.info("Already recording (%d steps)", recorder.steps)
            else:
                recorder.start()
            return paused
        if not (recorder.is_recording or recorder.is_pending):
            logger.warning("No recorded take to %s", name)
            return paused
        if not paused:
            paused = True
            self.ensemble.reset()
            self._policy_worker.flush()
            if self._prompt_listener is not None:
                self._prompt_listener.set_paused(True)
            logger.info("Policy paused — type an instruction (Enter alone = default) to resume")
        if name == "save":
            recorder.save()
        else:
            recorder.reject()
        return paused

    def run_episode(self, task_prompt: str = "look down"):
        """Run a single episode of policy execution."""
        logger.info(f"Starting episode with prompt: '{task_prompt}'")
        self._current_task = task_prompt
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

        prompt_listener = None
        paused = False
        if self.async_inference:
            self._policy_worker.start()
            if self.motions is not None:
                terminal_ui.print_help(HELP_ROWS)
            prompt_listener = terminal_ui.make_prompt_listener(task_prompt)
            prompt_listener.start()
        self._prompt_listener = prompt_listener
        if self._recorder is not None and prompt_listener is not None:
            self._recorder.status_cb = prompt_listener.set_recording
            self._recorder.save_cb = prompt_listener.set_saving

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
                        elif command is not None and command.name in ("record", "save", "reject"):
                            paused = self._handle_recording_command(command.name, paused=paused)
                        elif command is not None:
                            if command.moves_arm:
                                # Stop feeding policy actions before driving the arm
                                # ourselves, and throw away predictions made for the
                                # pose/instruction the motion is about to invalidate.
                                paused = True
                                self.ensemble.reset()
                                self._policy_worker.flush()
                                if command.name == "hold" and self._recorder is not None:
                                    # Pausing in place is the end of the take; other
                                    # motions keep recording (their frames are captured
                                    # in _send_scripted_pose).
                                    self._recorder.end()
                            prompt_listener.set_paused(paused)
                            self.motions.run(command)
                            self._reset_rate_window()
                            if paused:
                                logger.info("Policy paused — type an instruction (Enter alone = default) to resume")
                        else:
                            task_prompt = typed
                            self._current_task = task_prompt
                            prompt_listener.set_task(task_prompt)
                            logger.info(f"Task instruction: '{task_prompt}'")
                            if paused:
                                # Ensemble was cleared on pause, so treat this like a
                                # fresh episode: block for a new chunk, then ramp to it
                                # from wherever the scripted motion left the arm.
                                logger.info("Resuming policy")
                                paused = False
                                is_first_step = True
                                prompt_listener.set_paused(paused)
                            # Not paused: the ensemble still holds chunks for the old
                            # instruction, so the switch blends in over the next
                            # ~action_chunk_size steps instead of resetting (a reset
                            # would leave get_action() empty and drive the arm to zeros).

                    if paused:
                        time.sleep(0.05)
                        continue

                    # Submit fresh observation every step — non-blocking
                    raw_obs = self.robot.get_observation()
                    obs = self._build_observation(raw_obs, task_prompt)
                    self._policy_worker.submit(obs, self.episode_step)
                    if is_first_step:
                        logger.info("Waiting for first inference result...")
                        if not self._policy_worker.wait_for_first(timeout=30.0):
                            logger.error("Timed out waiting for first inference — aborting")
                            break
                    a_t = self.ensemble.get_action(self.episode_step)
                    if a_t is None:
                        a_t, paused = self._hold_pose(paused=paused)
                        if paused:
                            prompt_listener.set_paused(True)
                            continue
                    else:
                        self._starved_steps = 0

                else:
                    # Synchronous: request new chunk every rate_of_inference steps
                    if self.current_action_chunk is None or self.action_chunk_idx >= self.rate_of_inference:
                        observation = self._build_observation(self.robot.get_observation(), task_prompt)
                        logger.info(f"Step {self.episode_step}: Requesting new action chunk")
                        response = self.policy_client.infer(observation)
                        self.current_action_chunk = validate_actions(response, action_dim=self.action_dim)
                        if self.ensemble is not None:
                            self.ensemble.add_chunk(self.episode_step, self.current_action_chunk)
                        self.action_chunk_idx = 0
                        logger.info(f"Received action chunk: {self.current_action_chunk.shape}")

                    if self.ensemble is not None:
                        a_t = self.ensemble.get_action(self.episode_step)
                        if a_t is None:
                            a_t, _ = self._hold_pose(paused=False)
                        else:
                            self._starved_steps = 0
                    elif self.action_chunk_idx < len(self.current_action_chunk):
                        a_t = self.current_action_chunk[self.action_chunk_idx]
                    else:
                        # The server returned a shorter horizon than --rate_of_inference
                        # asks us to consume before the next request. Hold rather than
                        # index past the end of the chunk.
                        a_t, _ = self._hold_pose(paused=False)

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
                    self._reset_rate_window()
                else:
                    executed = self.execute_action(a_t)
                    if self._recorder is not None and self.async_inference:
                        # raw_obs is the observation this action was computed for
                        # (fetched at the top of the async branch). No-op unless
                        # a take is running.
                        self._recorder.add(raw_obs, executed, task_prompt)

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
        if self._recorder is not None:
            # Before disconnecting: an unsaved take is discarded (with a warning)
            # and the parquet writers closed, or the dataset cannot be reloaded.
            try:
                self._recorder.close()
            except Exception:
                logger.exception("Could not finalize the recording dataset")
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
        "--allow_unlimited_motion",
        action="store_true",
        help="Command the arm even when the driver's joint limits could not be read or failed validation. "
        "Off by default: without limits there is no velocity or position check between a bad action and "
        "the hardware, so the client stays read-only instead.",
    )
    parser.add_argument(
        "--ensemble_gripper",
        action="store_true",
        help="Also temporally ensemble the gripper channels. By default the gripper uses "
        "the latest raw prediction to avoid mushy/laggy open-close behaviour.",
    )
    parser.add_argument(
        "--record_dir",
        default=None,
        help="Record evaluation episodes into a LeRobot dataset at this directory (appends if it exists). "
        "Requires --async_inference. Type 'record' to start a take, 'hold'/'stop' to end it, then 'save' or 'reject'.",
    )
    parser.add_argument(
        "--record_repo_id",
        default="local/trossen_eval",
        help="repo_id stored in the recorded dataset's metadata",
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
        log_dir=args.log_dir,
        use_left_arm_only=args.use_left_arm_only,
        use_right_arm_only=args.use_right_arm_only,
        allow_unlimited_motion=args.allow_unlimited_motion,
        starvla=args.starvla,
        raw_gripper=not args.ensemble_gripper,
        record_dir=args.record_dir,
        record_repo_id=args.record_repo_id,
    )

    try:
        bridge.autonomous_mode(task_prompt=args.task_prompt)
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")
    finally:
        # Ctrl+C and crashes must still park the arms and release the cameras,
        # otherwise the next run finds the devices busy.
        bridge.cleanup()
