#!/usr/bin/env python3
"""Trossen AI × OpenPI — unified command-line entry point.

Single Typer app with three subcommands, replacing the old ``main.py`` /
``main_ee.py`` / ``replay_ee_dataset.py`` scripts:

    cli.py live-joint   joint-space policy bridge (policy server -> arm)
    cli.py live-ee      end-effector-space policy bridge (FK obs / IK actions)
    cli.py replay       replay an episode's EE actions from a dataset (no server)

Heavy / hardware-bound imports (``trossen_bridge``, ``robot_control``, the IK
converter) are done *inside* each command so ``--help`` works off-robot without
the hardware packages installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    help="Trossen AI <-> OpenPI control CLI (live policy + dataset replay).",
    no_args_is_help=True,
    add_completion=False,
)

# repo-root/dataset/converted_to_EE  (this file: repo/examples/trossen_ai/)
DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset" / "converted_to_EE"


@app.command("live-joint")
def live_joint(
    policy_host: str = typer.Option("192.168.50.174", help="Policy server host"),
    policy_port: int = typer.Option(8800, help="Policy server port"),
    control_freq: int = typer.Option(25, help="Control frequency in Hz"),
    mode: str = typer.Option("autonomous", help="autonomous (execute) or test (no movement)"),
    task_prompt: str = typer.Option("move the arm to the left", help="Task description"),
    max_steps: int = typer.Option(1000, help="Maximum steps per episode"),
    action_chunk_size: int = typer.Option(25, help="Actions predicted per inference"),
    rate_of_inference: int = typer.Option(20, help="Control steps between inferences (ignored with --async-inference)"),
    smoothing: bool = typer.Option(True, help="Action smoothing (blend overlapping chunks)"),
    smoothing_decay: float = typer.Option(1.0, help="Exp-decay for --smoothing-method temporal (higher = trust older predictions more)"),
    smoothing_method: str = typer.Option("temporal", help="Blend weighting: 'temporal' (exp-decay) or 'cogact' (cosine-similarity consensus)"),
    cogact_mode: str = typer.Option("cogact", help="For --smoothing-method cogact: 'cogact' | 'latest' | 'hybrid'"),
    log_dir: Optional[str] = typer.Option(None, help="Directory for per-episode overlap JSON"),
    async_inference: bool = typer.Option(False, help="Background-thread inference (needs --smoothing)"),
    use_left_arm_only: bool = typer.Option(False, help="Only move the left arm"),
    use_right_arm_only: bool = typer.Option(False, help="Only move the right arm"),
) -> None:
    """Run the joint-space policy bridge against the OpenPI policy server."""
    from adapters import JointAdapter
    from trossen_bridge import TrossenOpenPIBridge

    bridge = TrossenOpenPIBridge(
        policy_server_host=policy_host,
        policy_server_port=policy_port,
        control_frequency=control_freq,
        test_mode=mode,
        max_steps=max_steps,
        action_chunk_size=action_chunk_size,
        rate_of_inference=rate_of_inference,
        smoothing=smoothing,
        smoothing_decay=smoothing_decay,
        smoothing_method=smoothing_method,
        cogact_mode=cogact_mode,
        async_inference=async_inference,
        log_dir=log_dir,
        use_left_arm_only=use_left_arm_only,
        use_right_arm_only=use_right_arm_only,
        adapter=JointAdapter(),
    )
    bridge.autonomous_mode(task_prompt=task_prompt)
    bridge.cleanup()


@app.command("live-ee")
def live_ee(
    policy_host: str = typer.Option("192.168.50.174", help="Policy server host"),
    policy_port: int = typer.Option(8800, help="Policy server port"),
    control_freq: int = typer.Option(25, help="Control frequency in Hz"),
    mode: str = typer.Option("autonomous", help="autonomous (execute) or test (no movement)"),
    task_prompt: str = typer.Option("move the arm to the left", help="Task description"),
    max_steps: int = typer.Option(1000, help="Maximum steps per episode"),
    action_chunk_size: int = typer.Option(25, help="Actions predicted per inference"),
    rate_of_inference: int = typer.Option(20, help="Control steps between inferences"),
    smoothing: bool = typer.Option(True, help="Action smoothing (blend overlapping chunks)"),
    smoothing_decay: float = typer.Option(1.0, help="Exp-decay for --smoothing-method temporal (higher = trust older predictions more)"),
    smoothing_method: str = typer.Option("temporal", help="Blend weighting: 'temporal' (exp-decay) or 'cogact' (cosine-similarity consensus)"),
    cogact_mode: str = typer.Option("cogact", help="For --smoothing-method cogact: 'cogact' | 'latest' | 'hybrid'"),
    log_dir: Optional[str] = typer.Option(None, help="Directory for per-episode overlap JSON"),
    async_inference: bool = typer.Option(False, help="Background-thread inference (not supported in EE mode)"),
    use_left_arm_only: bool = typer.Option(False, help="Only move the left arm"),
    use_right_arm_only: bool = typer.Option(False, help="Only move the right arm"),
    ik_orientation_weight: float = typer.Option(
        0.01, help="placo IK orientation weight (raise for tighter rotation tracking)"
    ),
    ik_pos_tol_m: float = typer.Option(
        1e-3,
        help="IK convergence + failure tolerance (m): refines until FK error <= this; above it after ik_max_iters -> hold last",
    ),
) -> None:
    """Run the end-effector-space policy bridge (FK observations / IK actions).

    The policy outputs absolute 8-D EE poses per arm ([x,y,z,qw,qx,qy,qz,grip],
    robot-base frame). See docs/end_effector_support.md (Option 1). Synchronous
    inference only (async EE decoding is not yet supported).
    """
    if async_inference:
        raise typer.BadParameter("--async-inference is not supported in EE mode yet.")

    from adapters import EEAdapter
    from external.joint_to_ee.ee_to_joints import EEToJointsConverter
    from external.joint_to_ee.kinematics import make_kinematics
    from trossen_bridge import TrossenOpenPIBridge

    converter = EEToJointsConverter(
        make_kinematics(),
        orientation_weight=ik_orientation_weight,
        pos_tol_m=ik_pos_tol_m,
    )
    adapter = EEAdapter(converter)

    bridge = TrossenOpenPIBridge(
        policy_server_host=policy_host,
        policy_server_port=policy_port,
        control_frequency=control_freq,
        test_mode=mode,
        max_steps=max_steps,
        action_chunk_size=action_chunk_size,
        rate_of_inference=rate_of_inference,
        smoothing=smoothing,
        smoothing_decay=smoothing_decay,
        smoothing_method=smoothing_method,
        cogact_mode=cogact_mode,
        async_inference=False,
        log_dir=log_dir,
        use_left_arm_only=use_left_arm_only,
        use_right_arm_only=use_right_arm_only,
        adapter=adapter,
    )
    bridge.autonomous_mode(task_prompt=task_prompt)
    bridge.cleanup()


@app.command("replay")
def replay(
    dataset_dir: str = typer.Option(str(DEFAULT_DATASET_DIR), help="LeRobot v3.0 dataset directory"),
    episode_index: int = typer.Option(0, help="Episode to replay"),
    control_freq: Optional[int] = typer.Option(None, help="Stream rate in Hz (default: dataset fps)"),
    mode: str = typer.Option("autonomous", help="autonomous (move arms) or test (decode + log, no movement)"),
    start_duration: float = typer.Option(5.0, help="Seconds for the PCHIP move to the episode's first pose"),
    ik_orientation_weight: float = typer.Option(
        0.01, help="placo IK orientation weight (raise for tighter rotation tracking)"
    ),
    ik_pos_tol_m: float = typer.Option(
        1e-3, help="IK convergence + failure tolerance (m); above it -> hold last joints"
    ),
    ik_max_joint_jump_deg: float = typer.Option(
        0.0,
        help="Branch-flip guard: >0 rejects a frame whose IK jumps any joint more "
        "than this (deg) vs the previous frame and holds the last good pose; 0 disables",
    ),
) -> None:
    """Replay one episode's EE actions from a LeRobot v3.0 dataset on the robot.

    Reads ``action.ee_left`` + ``action.ee_right`` (absolute 8-D poses, robot-base
    frame), IK-decodes them to 14-D joint targets, then streams them at the dataset
    fps — PCHIP move-to-start first, then per-step joint-velocity safety. No policy
    server is involved.
    """
    import logging
    import time

    from dataset_replay import EpisodeReader
    from external.joint_to_ee.ee_to_joints import EEToJointsConverter
    from external.joint_to_ee.kinematics import make_kinematics
    from robot_control import RobotController, build_stationary_robot

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    reader = EpisodeReader(dataset_dir)
    episode = reader.read_episode(episode_index)
    freq = control_freq or episode.fps
    logger.info(
        "Episode %d: %d frames @ %d fps (task=%r); streaming at %d Hz",
        episode.episode_index,
        episode.n_frames,
        episode.fps,
        episode.task,
        freq,
    )

    converter = EEToJointsConverter(
        make_kinematics(),
        orientation_weight=ik_orientation_weight,
        pos_tol_m=ik_pos_tol_m,
        max_joint_jump_deg=(ik_max_joint_jump_deg if ik_max_joint_jump_deg > 0 else None),
    )

    robot = build_stationary_robot(with_cameras=False)
    controller = RobotController(robot, control_frequency=freq, test_mode=mode)

    try:
        # Seed IK from the robot's actual current joints, then decode the whole
        # episode (decode_chunk chains the seed frame-to-frame for continuity).
        current_joints14 = controller.current_joints14()
        logger.info("IK-decoding %d EE frames -> joints...", episode.n_frames)
        joints = converter.decode_chunk(episode.ee_chunk16, current_joints14)  # (N, 14)

        logger.info("Moving to episode start pose over %.1fs...", start_duration)
        controller.move_to_start_position(joints[0], duration=start_duration)

        dt = 1.0 / freq
        for step, a_t in enumerate(joints[1:], start=1):
            loop_start = time.perf_counter()
            if not controller.execute_action(a_t):
                logger.error("Step %d exceeded joint limits — aborted (arm sent to sleep).", step)
                break
            elapsed = time.perf_counter() - loop_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)
        else:
            logger.info("Replay completed: %d frames.", episode.n_frames)
    finally:
        controller.disconnect()


@app.command("teleop")
def teleop(
    mode: str = typer.Option("detached", help="detached (3D only, no robot) | test (cameras, no motion) | autonomous (real motion)"),
    control_freq: int = typer.Option(25, help="Control loop frequency in Hz"),
    arm: str = typer.Option("left", help="Initial active arm: left | right"),
    device: Optional[str] = typer.Option(None, help="evdev gamepad path (default: first /dev/input device)"),
    max_lin: float = typer.Option(0.05, help="Max EE linear velocity (m/s) at full stick"),
    max_ang: float = typer.Option(0.5, help="Max EE angular velocity (rad/s) at full stick"),
    grip_rate: float = typer.Option(0.5, help="Gripper rate (normalized 1/s) while held"),
    deadzone: float = typer.Option(0.1, help="Analog stick deadzone"),
    max_joint_speed: float = typer.Option(3.0, help="Per-joint velocity cap (rad/s)"),
    ik_orientation_weight: float = typer.Option(0.01, help="placo IK orientation weight"),
    ik_pos_tol_m: float = typer.Option(1e-3, help="IK convergence/failure tolerance (m)"),
) -> None:
    """Joystick/keyboard end-effector teleoperation through the current EE IK.

    Detached mode runs off-robot (3D model only) — useful to dry-test IK before
    touching hardware. test/autonomous build the real robot (cameras on);
    autonomous moves it for real. Same control core as the /teleop web page.
    """
    import logging
    import signal

    import numpy as np

    from external.joint_to_ee.ee_to_joints import EEToJointsConverter
    from external.joint_to_ee.kinematics import make_kinematics
    from robot_control import HOME_POSITION
    from teleop import TeleopController
    from teleop_evdev import EvdevTeleopInput
    from webapp.telemetry import NullSink

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    converter = EEToJointsConverter(
        make_kinematics(), orientation_weight=ik_orientation_weight, pos_tol_m=ik_pos_tol_m)

    controller = None
    cam_keys: list[str] = []
    start14 = np.asarray(HOME_POSITION, float)
    if mode != "detached":
        from robot_control import RobotController, build_stationary_robot
        robot = build_stationary_robot(with_cameras=True)
        cam_keys = list(robot._cameras_ft.keys())  # noqa: SLF001
        controller = RobotController(robot, control_frequency=control_freq, test_mode=mode)
        start14 = controller.current_joints14()

    inp = EvdevTeleopInput(device_path=device, max_lin=max_lin, max_ang=max_ang,
                           grip_rate=grip_rate, deadzone=deadzone)
    inp.start()
    teleop_ctrl = TeleopController(
        converter, NullSink(), inp, controller=controller, start14=start14,
        control_freq=control_freq, max_lin=max_lin, max_ang=max_ang,
        grip_rate=grip_rate, max_joint_speed=max_joint_speed, cam_keys=cam_keys)
    teleop_ctrl.active_arm = arm

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    typer.echo(f"Teleop running (mode={mode}, arm={arm}). Ctrl-C to stop.")
    try:
        teleop_ctrl.run(should_stop=lambda: stop["flag"])
    finally:
        inp.stop()
        if controller is not None:
            controller.disconnect()


if __name__ == "__main__":
    app()
