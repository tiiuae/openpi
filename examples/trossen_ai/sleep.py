import numpy as np
from lerobot.robots import make_robot_from_config
from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig


move_to_home_position = np.array(
    [0, np.pi / 4, np.pi / 6, np.pi / 5, 0, 0, 0, 0, np.pi / 4, np.pi / 6, np.pi / 5, 0, 0, 0]
)

move_to_sleep_position = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

robot = make_robot_from_config(
    BiWidowXAIFollowerRobotConfig(
        id="bimanual_follower",
        left_arm_ip_address="192.168.1.5",
        right_arm_ip_address="192.168.1.4",
        min_time_to_move_multiplier=4.0,
        loop_rate=30,
        cameras={},
    )
)

robot.connect()

robot.send_action(move_to_sleep_position)
print("Sent to sleep position.")

robot.disconnect()
