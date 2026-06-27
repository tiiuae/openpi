#!/usr/bin/env python3
"""
Example script to smoothly move the Bi-WidowX AI Follower to a "sleep" position over a specified duration.
Useful after doing tests and the arms are in a random pose, to move them back to a safe position before disconnecting.
"""

import time

from lerobot.robots import make_robot_from_config
from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig
import numpy as np
from scipy.interpolate import PchipInterpolator

SLEEP_POSITION = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
CONTROL_FREQUENCY = 30
DURATION = 5.0  # seconds to smoothly reach sleep position

robot = make_robot_from_config(
    BiWidowXAIFollowerRobotConfig(
        id="bimanual_follower",
        left_arm_ip_address="192.168.1.5",
        right_arm_ip_address="192.168.1.4",
        min_time_to_move_multiplier=4.0,
        loop_rate=CONTROL_FREQUENCY,
        cameras={},
    )
)

robot.connect()

joint_features = list(robot._joint_ft.keys())  # noqa
joint_pos_keys = [k for k in robot.get_observation() if k.endswith(".pos")]
current_pose = np.array([robot.get_observation()[k] for k in joint_pos_keys])

waypoints = np.array([current_pose, SLEEP_POSITION])
timepoints = np.array([0, DURATION])
interpolator = PchipInterpolator(timepoints, waypoints, axis=0)

print(f"Moving to sleep position over {DURATION}s...")
dt = 1.0 / CONTROL_FREQUENCY
start_time = time.time()
end_time = start_time + DURATION

while time.time() < end_time:
    loop_start = time.perf_counter()
    current_time = time.time() - start_time
    positions = interpolator(current_time)
    action_dict = {k: positions[i] for i, k in enumerate(joint_features)}
    robot.send_action(action_dict)
    elapsed = time.perf_counter() - loop_start
    if dt - elapsed > 0:
        time.sleep(dt - elapsed)

print("Reached sleep position.")
robot.disconnect()
