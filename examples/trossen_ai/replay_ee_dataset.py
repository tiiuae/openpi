#!/usr/bin/env python3
"""Replay one episode's EE actions from a LeRobot v3.0 dataset on the robot.

Reads ``action.ee_left`` + ``action.ee_right`` (absolute 8-D poses, robot-base
frame) for the chosen episode, IK-decodes them to 14-D joint targets with the same
converter ``main_ee.py`` uses, then streams them to the arms at the dataset fps —
PCHIP move-to-start first, then per-step joint-velocity safety on every frame.

No policy server is involved.

Example::

    python replay_ee_dataset.py --episode_index 0
    python replay_ee_dataset.py --episode_index 3 --mode test   # no movement
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

from dataset_replay import EpisodeReader
from external.joint_to_ee.ee_to_joints import EEToJointsConverter
from external.joint_to_ee.kinematics import make_kinematics
from robot_control import RobotController
from robot_control import build_stationary_robot

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# repo-root/dataset/converted_to_EE  (this file: repo/examples/trossen_ai/)
DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset" / "converted_to_EE"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay EE actions from a LeRobot v3.0 dataset episode")
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATASET_DIR), help="LeRobot v3.0 dataset directory")
    parser.add_argument("--episode_index", type=int, default=0, help="Episode to replay")
    parser.add_argument("--control_freq", type=int, default=None, help="Stream rate in Hz (default: dataset fps)")
    parser.add_argument(
        "--mode",
        choices=["autonomous", "test"],
        default="autonomous",
        help="autonomous (move arms) or test (decode + log, no movement)",
    )
    parser.add_argument(
        "--start_duration", type=float, default=5.0, help="Seconds for the PCHIP move to the episode's first pose"
    )
    parser.add_argument(
        "--ik_orientation_weight",
        type=float,
        default=0.01,
        help="placo IK orientation weight (raise for tighter rotation tracking)",
    )
    parser.add_argument(
        "--ik_pos_tol_m",
        type=float,
        default=1e-3,
        help="IK convergence + failure tolerance (m); above it -> hold last joints",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    reader = EpisodeReader(args.dataset_dir)
    episode = reader.read_episode(args.episode_index)
    control_freq = args.control_freq or episode.fps
    logger.info(
        "Episode %d: %d frames @ %d fps (task=%r); streaming at %d Hz",
        episode.episode_index,
        episode.n_frames,
        episode.fps,
        episode.task,
        control_freq,
    )

    converter = EEToJointsConverter(
        make_kinematics(),
        orientation_weight=args.ik_orientation_weight,
        pos_tol_m=args.ik_pos_tol_m,
    )

    robot = build_stationary_robot(with_cameras=False)
    controller = RobotController(robot, control_frequency=control_freq, test_mode=args.mode)

    try:
        # Seed IK from the robot's actual current joints, then decode the whole
        # episode (decode_chunk chains the seed frame-to-frame for continuity).
        current_joints14 = controller.current_joints14()
        logger.info("IK-decoding %d EE frames -> joints...", episode.n_frames)
        joints = converter.decode_chunk(episode.ee_chunk16, current_joints14)  # (N, 14)

        logger.info("Moving to episode start pose over %.1fs...", args.start_duration)
        controller.move_to_start_position(joints[0], duration=args.start_duration)

        dt = 1.0 / control_freq
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
    main()
