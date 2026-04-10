#!/usr/bin/env python3
"""
Capture one observation from the robot and save it to disk.

This script connects to the robot hardware, collects a single observation
(joint positions + camera images), formats it exactly as main.py does,
and saves it as a msgpack file — without contacting the policy server.

Usage:
    python capture_request.py --output captured_request.msgpack \
        --task_prompt "Pick up the blue cup and place it in the orange basket"
"""

import argparse
import logging

import cv2
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig
import numpy as np

from openpi_client import msgpack_numpy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def capture(task_prompt: str, output_path: str) -> None:
    robot_config = BiWidowXAIFollowerRobotConfig(
        id="bimanual_follower",
        left_arm_ip_address="192.168.1.5",
        right_arm_ip_address="192.168.1.4",
        min_time_to_move_multiplier=4.0,
        loop_rate=30,
        cameras={
            "cam_high": OpenCVCameraConfig(index_or_path=16, width=640, height=480, fps=30),
            "cam_right_wrist": OpenCVCameraConfig(index_or_path=10, width=640, height=480, fps=30),
            "cam_left_wrist": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=30),
        },
    )

    logger.info("Connecting to robot...")
    robot = make_robot_from_config(robot_config)
    robot.connect()

    try:
        logger.info("Reading one observation from robot...")
        observation_dict = robot.get_observation()

        # Extract joint positions
        joint_pos_keys = [k for k in observation_dict.keys() if k.endswith(".pos")]
        joint_positions = np.array([observation_dict[k] for k in joint_pos_keys])

        # Resize and convert camera images (BGR -> RGB, HWC -> CHW)
        import os
        debug_dir = "debug"
        os.makedirs(debug_dir, exist_ok=True)

        cameras = list(robot._cameras_ft.keys())
        images = {}
        for cam in cameras:
            image_hwc = observation_dict[cam]
            cv2.imwrite(os.path.join(debug_dir, f"{cam}_01_raw_bgr.png"), image_hwc)

            image_resized = cv2.resize(image_hwc, (224, 224))
            cv2.imwrite(os.path.join(debug_dir, f"{cam}_02_resized_bgr.png"), image_resized)

            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(debug_dir, f"{cam}_03_resized_rgb.png"), image_resized)

            image_chw = np.transpose(image_rgb, (2, 0, 1))
            cv2.imwrite(os.path.join(debug_dir, f"{cam}_04_final_chw_as_hwc.png"), np.transpose(image_chw, (1, 2, 0))[:, :, ::-1])

            images[cam] = image_chw

        logger.info(f"Debug images saved to '{debug_dir}/'  ({len(cameras) * 4} files)")

        observation = {
            "state": joint_positions,
            "images": images,
            "prompt": task_prompt,
        }

        packed = msgpack_numpy.packb(observation)
        with open(output_path, "wb") as f:
            f.write(packed)

        logger.info(f"Saved observation to '{output_path}'")
        logger.info(f"  state shape : {joint_positions.shape}")
        logger.info(f"  cameras     : {list(images.keys())}")
        logger.info(f"  image shape : {next(iter(images.values())).shape}")
        logger.info(f"  prompt      : {task_prompt!r}")
        logger.info(f"  file size   : {len(packed):,} bytes")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture one robot observation to a msgpack file")
    parser.add_argument("--output", default="captured_request.msgpack", help="Output file path")
    parser.add_argument("--task_prompt", default="move the arm to the left", help="Task prompt to embed")
    args = parser.parse_args()

    capture(task_prompt=args.task_prompt, output_path=args.output)
