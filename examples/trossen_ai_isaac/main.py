#!/usr/bin/env python3
"""
Trossen Arm <-> OpenPI Policy Server Bridge (Isaac Sim Version)

Drop-in simulation replacement for trossen_ai/main.py.
Identical logic, observation format, and action format — only the
robot backend is swapped: instead of a real BiWidowXAI, the actions
are sent to a Mobile AI simulated in Isaac Sim and observations are
read back from it.

Handles:
1. Collecting observations from the sim (joint positions + virtual cameras)
2. Sending observations to the policy server via WebSocket
3. Receiving action predictions
4. Executing actions on the simulated robot

Usage:
    ~/isaacsim/python.sh openpi/examples/trossen_ai_isaac/main.py \
        --task_prompt "grab and handover red cube"

    Test mode (no movement, just prints actions):
    ~/isaacsim/python.sh openpi/examples/trossen_ai_isaac/main.py \
        --mode test --task_prompt "grab and handover red cube"

Note:
    This script must be run with Isaac Sim's Python interpreter
    (~/isaacsim/python.sh), which must also have openpi_client installed:
        ~/isaacsim/python.sh -m pip install -e /path/to/openpi/packages/openpi-client
"""

import argparse
from collections import defaultdict
import logging
import os
import sys
import time

# ── Isaac Sim MUST be initialized before any other Isaac/Omni imports ──────────
from isaacsim import SimulationApp

# Support WebRTC streaming when LIVESTREAM=2 is set (e.g. from docker-compose)
_livestream = int(os.environ.get("LIVESTREAM", 0))
_sim_config: dict = {"headless": _livestream > 0}
if _livestream > 0:
    _sim_config["livestream"] = _livestream

simulation_app = SimulationApp(_sim_config)
# ────────────────────────────────────────────────────────────────────────────────

import cv2  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim, XformPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))

ROBOT_USD_PATH = os.path.join(_REPO_ROOT, "trossen_ai_isaac", "assets", "robots", "mobile_ai", "mobile_ai.usd")
ROBOT_SCENE_PATH = "/World/mobile_ai"
ENVIRONMENT_SCENE_PATH = "/World/Environment"

# ── Mobile AI DOF layout (26 DOFs total) ────────────────────────────────────────
#   [0-3]   caster wheel / swivel joints
#   [4-5]   left/right drive wheels
#   [6-9]   additional base joints
#   [10,12,14,16,18,20]  left arm (6 joints)
#   [11,13,15,17,19,21]  right arm (6 joints)
#   [22,23] left gripper fingers
#   [24,25] right gripper fingers
LEFT_ARM_DOF_INDICES = [10, 12, 14, 16, 18, 20]
LEFT_GRIPPER_DOF_INDEX = 22
RIGHT_ARM_DOF_INDICES = [11, 13, 15, 17, 19, 21]
RIGHT_GRIPPER_DOF_INDEX = 24

# DOF indices that make up the policy state / action vector.
# Order mirrors the real robot's _joint_ft key order:
#   left_joint_0..5, left_gripper, right_joint_0..5, right_gripper  → 14 values
ACTION_DOF_INDICES = (
    LEFT_ARM_DOF_INDICES
    + [LEFT_GRIPPER_DOF_INDEX]
    + RIGHT_ARM_DOF_INDICES
    + [RIGHT_GRIPPER_DOF_INDEX]
)
ACTION_DIM = len(ACTION_DOF_INDICES)  # 14

MOBILE_AI_DEFAULT_DOF_POSITIONS = (
    [0.0] * 10 + [0.0, 0.0] * 6 + [0.044, 0.044, 0.044, 0.044]
)

# Robot spawn pose (from mobile_ai_pick_place.py)
ROBOT_SPAWN_POSITION = np.array([0.0, 2.1, -0.70])
ROBOT_SPAWN_ORIENTATION = np.array([0.7071068, 0.0, 0.0, -0.7071068])  # -90° around Z

# ── Camera configuration ────────────────────────────────────────────────────────
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Fixed-world cameras.  Adjust positions/orientations for your scene.
# Orientation: quaternion [w, x, y, z].  Helpers at the bottom of this block.
def _euler_to_quat_wxyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Euler angles (degrees, XYZ) → Isaac Sim quaternion [w, x, y, z]."""
    r = Rotation.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True)
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z])


CAM_CONFIGS = {
    # Overhead camera looking down at the robot workspace
    "cam_high": {
        "prim_path": "/World/cam_high",
        "position": np.array([0.0, 0.0, 2.5]),
        "orientation": _euler_to_quat_wxyz(-90.0, 0.0, 0.0),
    },
    # Low-angle front camera
    "cam_low": {
        "prim_path": "/World/cam_low",
        "position": np.array([0.0, -1.5, 0.6]),
        "orientation": _euler_to_quat_wxyz(-15.0, 0.0, 0.0),
    },
    # Wrist cameras: prim path under the wrist link → moves with the arm
    "cam_right_wrist": {
        "prim_path": f"{ROBOT_SCENE_PATH}/follower_right_link_6/cam_right_wrist",
        "position": np.array([0.05, 0.0, 0.0]),   # local offset from link_6
        "orientation": _euler_to_quat_wxyz(0.0, -90.0, 0.0),
    },
    "cam_left_wrist": {
        "prim_path": f"{ROBOT_SCENE_PATH}/follower_left_link_6/cam_left_wrist",
        "position": np.array([0.05, 0.0, 0.0]),   # local offset from link_6
        "orientation": _euler_to_quat_wxyz(0.0, -90.0, 0.0),
    },
}


class TrossenOpenPIBridgeIsaac:
    """Bridge between Isaac Sim Mobile AI and the OpenPI policy server.

    Identical public interface to TrossenOpenPIBridge (real-robot version) so
    the two can be swapped without changing any higher-level code.
    """

    def __init__(
        self,
        policy_server_host: str = "localhost",
        policy_server_port: int = 8000,
        control_frequency: int = 30,
        test_mode: str = "autonomous",
        max_steps: int = 1000,
    ):
        self.control_frequency = control_frequency
        self.max_steps = max_steps
        self.dt = 1.0 / control_frequency
        self.test_mode = test_mode

        self._policy_host = policy_server_host
        self._policy_port = policy_server_port
        self.policy_client = None  # connected lazily after the scene is up

        # ── Build Isaac Sim scene ───────────────────────────────────────────
        # Must update BEFORE any stage operations (matches mobile_ai_pick_place.py pattern)
        simulation_app.update()
        self._setup_scene()
        self._start_simulation()

        # ── Wait for policy server while keeping the simulation alive ───────
        # Polling here (instead of in the Dockerfile CMD) lets us call
        # simulation_app.update() every iteration so the WebRTC viewer renders.
        import socket
        logger.info(f"Waiting for policy server at {policy_server_host}:{policy_server_port}…")
        while simulation_app.is_running():
            try:
                with socket.create_connection((policy_server_host, policy_server_port), timeout=1):
                    break
            except OSError:
                logger.info("  policy server not ready, retrying in 2 s…")
                for _ in range(60):  # 2 s at ~30 fps
                    simulation_app.update()
                    time.sleep(1 / 30)
        logger.info("Policy server is ready — connecting…")
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_server_host, port=policy_server_port
        )

        # ── Shared episode state (identical to real bridge) ─────────────────
        self.current_action_chunk = None
        self.action_chunk_idx = 0
        self.action_chunk_size = 50
        self.episode_step = 0
        self.is_running = False
        self.rate_of_inference = 50

        self.temporal_ensemble_coefficient = None

        self.action_buffer = defaultdict(list)
        self.action_buffer_size = self.max_steps + self.action_chunk_size

        self.action_dim = ACTION_DIM  # 14 (matches real robot's _joint_ft length)

    # ── Scene setup ────────────────────────────────────────────────────────────

    def _setup_scene(self) -> None:
        """Load the Mobile AI USD and create virtual cameras."""
        logger.info("Setting up Isaac Sim scene…")

        stage_utils.create_new_stage(template="sunlight")
        # Commit the new stage to the renderer immediately — in headless/WebRTC
        # mode the renderer stays attached to the previous empty stage unless
        # we tick it right after create_new_stage.
        simulation_app.update()

        # Ground/environment
        assets_root = get_assets_root_path()
        if assets_root:
            logger.info(f"Loading environment from {assets_root}")
            stage_utils.add_reference_to_stage(
                usd_path=assets_root + "/Isaac/Environments/Simple_Room/simple_room.usd",
                path=ENVIRONMENT_SCENE_PATH,
            )
        else:
            logger.warning("Could not find Isaac assets root — skipping environment USD.")
        simulation_app.update()

        # Robot
        if not os.path.exists(ROBOT_USD_PATH):
            raise FileNotFoundError(f"Robot USD not found: {ROBOT_USD_PATH}")
        logger.info(f"Loading robot from {ROBOT_USD_PATH}")
        stage_utils.add_reference_to_stage(usd_path=ROBOT_USD_PATH, path=ROBOT_SCENE_PATH)
        simulation_app.update()

        robot_xform = XformPrim(ROBOT_SCENE_PATH)
        robot_xform.set_world_poses(
            positions=ROBOT_SPAWN_POSITION,
            orientations=ROBOT_SPAWN_ORIENTATION,
        )

        # Articulation wrapper (used for DOF position reads and writes)
        self._robot = Articulation(ROBOT_SCENE_PATH)
        self._robot.set_default_state(dof_positions=MOBILE_AI_DEFAULT_DOF_POSITIONS)

        # Cameras
        self._cameras: dict[str, Camera] = {}
        for name, cfg in CAM_CONFIGS.items():
            cam = Camera(
                prim_path=cfg["prim_path"],
                position=cfg["position"],
                orientation=cfg["orientation"],
                resolution=(IMAGE_WIDTH, IMAGE_HEIGHT),
            )
            self._cameras[name] = cam

        logger.info("Scene setup complete.")

    def _start_simulation(self) -> None:
        """Start the Isaac Sim timeline and initialise all prims."""
        omni.timeline.get_timeline_interface().play()
        simulation_app.update()

        # Set viewport camera AFTER play() so it takes effect in both GUI and WebRTC modes.
        # In headless/WebRTC mode the camera change needs several rendered frames to propagate
        # to the streaming encoder, so we run a short warmup render loop.
        set_camera_view(eye=[-3.0, 3.0, 2.5], target=[0.0, 0.5, 0.0])
        logger.info("Warming up renderer…")
        for _ in range(60):  # ~2 s at 30 fps
            simulation_app.update()

        # Initialise cameras after the first timeline tick
        for cam in self._cameras.values():
            cam.initialize()

        # Reset robot to default pose
        self._reset_robot()
        simulation_app.update()
        logger.info("Simulation started.")

    def _reset_robot(self) -> None:
        """Reset robot joints to the default (home) configuration."""
        default_pos = np.array([MOBILE_AI_DEFAULT_DOF_POSITIONS])
        self._robot.set_dof_positions(default_pos)
        self._robot.set_dof_position_targets(default_pos)

    # ── Observation ────────────────────────────────────────────────────────────

    def _get_robot_observation(self) -> dict:
        """Read joint positions and camera images from the simulation.

        Returns a dict with the same keys that the real robot's
        ``get_observation()`` would return, so the rest of the pipeline
        is unchanged.
        """
        # ── Joint positions (shape: [ACTION_DIM]) ──────────────────────────
        all_dof_pos = self._robot.get_dof_positions().numpy().flatten()
        joint_pos = all_dof_pos[ACTION_DOF_INDICES]  # (14,)

        # Pack in the same ".pos" key format the real bridge expects
        obs = {f"joint_{i}.pos": joint_pos[i] for i in range(ACTION_DIM)}

        # ── Camera images (H×W×3, uint8, RGB) ────────────────────────────
        for name, cam in self._cameras.items():
            frame = cam.get_current_frame()
            rgba = frame.get("rgba")
            if rgba is None:
                # Camera not ready yet — return a black frame
                rgb = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
            else:
                rgb = rgba[:, :, :3].astype(np.uint8)
            obs[name] = rgb  # HWC, RGB

        return obs

    # ── Action execution ───────────────────────────────────────────────────────

    def execute_action(self, action: np.ndarray) -> None:
        """Apply a 14-DOF joint position command to the simulated robot."""
        if self.test_mode == "test":
            logger.info(f"TEST MODE: Would execute action: {action}")
            simulation_app.update()
            return

        if self.test_mode == "autonomous":
            # Build a full-DOF position target array from the current positions
            all_dof_pos = self._robot.get_dof_positions().numpy()  # (1, 26)
            targets = all_dof_pos.copy()
            for i, dof_idx in enumerate(ACTION_DOF_INDICES):
                targets[0, dof_idx] = action[i]
            self._robot.set_dof_position_targets(targets)
            simulation_app.update()
        else:
            logger.error(f"Unknown mode: {self.test_mode}. No action executed.")

    def move_to_start_position(self, goal_position: np.ndarray, duration: float = 5.0) -> None:
        """Smoothly move the robot from its current pose to *goal_position*.

        Uses PCHIP interpolation — identical to the real-robot version — but
        advances the simulation clock instead of calling time.sleep().
        """
        all_dof_pos = self._robot.get_dof_positions().numpy().flatten()
        current_pose = all_dof_pos[ACTION_DOF_INDICES]

        waypoints = np.array([current_pose, goal_position])
        timepoints = np.array([0.0, duration])
        interpolator = PchipInterpolator(timepoints, waypoints, axis=0)

        n_steps = int(duration * self.control_frequency)
        for step in range(n_steps):
            t = step * self.dt
            positions = interpolator(t)
            self.execute_action(positions)

    # ── Episode loop (identical logic to real bridge) ─────────────────────────

    def run_episode(self, task_prompt: str = "look down") -> None:
        """Run a single episode of policy execution."""
        logger.info(f"Starting episode with prompt: '{task_prompt}'")
        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.is_running = True
        is_first_step = True

        while self.is_running and self.episode_step < self.max_steps:
            start_loop_time = time.perf_counter()

            if self.current_action_chunk is None or self.action_chunk_idx >= self.rate_of_inference:
                observation_dict = self._get_robot_observation()

                # Extract joint positions
                joint_pos_keys = [k for k in observation_dict if k.endswith(".pos")]
                joint_positions = np.array([observation_dict[k] for k in joint_pos_keys])

                # Resize + channel-transpose camera images (same as real bridge)
                cameras = list(CAM_CONFIGS.keys())
                for cam in cameras:
                    image_hwc = observation_dict[cam]
                    image_resized = cv2.resize(image_hwc, (224, 224))
                    # Images from Isaac Sim are already RGB — no BGR→RGB conversion needed
                    image_chw = np.transpose(image_resized, (2, 0, 1))
                    observation_dict[cam] = image_chw

                observation = {
                    "state": joint_positions,
                    "images": {cam: observation_dict[cam] for cam in cameras},
                    "prompt": task_prompt,
                }

                logger.info(f"Step {self.episode_step}: Requesting new action chunk")
                response = self.policy_client.infer(observation)
                self.current_action_chunk = response["actions"]

                for k in range(self.action_chunk_size):
                    future_t = self.episode_step + k
                    if future_t < self.action_buffer_size:
                        self.action_buffer[future_t].append(self.current_action_chunk[k])

                self.action_chunk_idx = 0
                logger.info(f"Received action chunk: {self.current_action_chunk.shape}")

            # Temporal ensembling (same logic as real bridge)
            if self.temporal_ensemble_coefficient is not None:
                if len(self.action_buffer[self.episode_step]) == 0:
                    a_t = np.zeros(self.action_dim)
                else:
                    candidates = np.array(self.action_buffer[self.episode_step])
                    weights = self._get_weights(len(candidates))
                    a_t = np.average(candidates, axis=0, weights=weights)
            else:
                a_t = self.current_action_chunk[self.action_chunk_idx]

            if is_first_step:
                logger.info("Moving to start position to avoid large jumps…")
                self.move_to_start_position(a_t, duration=5.0)
                is_first_step = False
            else:
                self.execute_action(a_t)

            self.action_chunk_idx += 1
            self.episode_step += 1

            # Always advance the simulation render every loop iteration
            # (mirrors the while loop pattern in mobile_ai_pick_place.py)
            simulation_app.update()

            dt_s = time.perf_counter() - start_loop_time
            busy_wait = self.dt - dt_s
            if busy_wait > 0:
                time.sleep(busy_wait)
            loop_s = time.perf_counter() - start_loop_time
            logger.info(f"time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        self.is_running = False
        logger.info(f"Episode completed after {self.episode_step} steps")

    def _get_weights(self, num_preds: int) -> np.ndarray:
        weights = np.exp(-self.temporal_ensemble_coefficient * np.arange(num_preds))
        return weights / weights.sum()

    def autonomous_mode(self, task_prompt: str = "look down") -> None:
        logger.info("Starting autonomous mode")
        self.run_episode(task_prompt=task_prompt)

    def cleanup(self) -> None:
        """Shut down the simulation."""
        logger.info("Cleaning up…")
        simulation_app.close()


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trossen AI Mobile AI (Isaac Sim) <-> OpenPI Policy Server Bridge"
    )
    parser.add_argument("--policy_host", default="localhost", help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--control_freq", type=int, default=30, help="Control frequency in Hz")
    parser.add_argument(
        "--mode",
        choices=["autonomous", "test"],
        default="autonomous",
        help="Operation mode: autonomous (execute) or test (no movement)",
    )
    parser.add_argument("--task_prompt", default="move the arm to the left", help="Task description for the policy")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    args = parser.parse_args()

    bridge = TrossenOpenPIBridgeIsaac(
        policy_server_host=args.policy_host,
        policy_server_port=args.policy_port,
        control_frequency=args.control_freq,
        test_mode=args.mode,
        max_steps=args.max_steps,
    )

    try:
        bridge.autonomous_mode(task_prompt=args.task_prompt)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        bridge.cleanup()
