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
    rate_of_inference: int = typer.Option(20, help="Control steps between inferences"),
    ensemble_type: str = typer.Option("exp", help="Action ensemble strategy: exp|cogact|none"),
    cogact_mode: str = typer.Option(
        "cogact", help="CogACT weighting mode (only with --ensemble-type cogact): cogact|latest|hybrid"
    ),
    log_dir: Optional[str] = typer.Option(None, help="Directory for per-episode overlap JSON"),
    async_inference: bool = typer.Option(False, help="Background-thread inference (needs ensemble)"),
    starvla: bool = typer.Option(False, help="StarVLA 224x224 PIL resizing"),
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
        ensemble_type=ensemble_type,
        cogact_mode=cogact_mode,
        async_inference=async_inference,
        log_dir=log_dir,
        use_left_arm_only=use_left_arm_only,
        use_right_arm_only=use_right_arm_only,
        starvla=starvla,
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
    ensemble_type: str = typer.Option("exp", help="Action ensemble strategy: exp|cogact|none"),
    cogact_mode: str = typer.Option(
        "cogact", help="CogACT weighting mode (only with --ensemble-type cogact): cogact|latest|hybrid"
    ),
    log_dir: Optional[str] = typer.Option(None, help="Directory for per-episode overlap JSON"),
    async_inference: bool = typer.Option(False, help="Background-thread inference (not supported in EE mode)"),
    starvla: bool = typer.Option(False, help="StarVLA 224x224 PIL resizing"),
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
        ensemble_type=ensemble_type,
        cogact_mode=cogact_mode,
        async_inference=False,
        log_dir=log_dir,
        use_left_arm_only=use_left_arm_only,
        use_right_arm_only=use_right_arm_only,
        starvla=starvla,
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


if __name__ == "__main__":
    app()
