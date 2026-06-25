"""Hardware-bound runners that adapt the bridge and replay tool to the Runner
interface used by SessionManager. Syntax-checked in CI; exercised on the rig.

`config` is the dict produced by the web form. The constructors forward only the
keys they declare; unknown keys are ignored.
"""
from __future__ import annotations

import time

import numpy as np

from adapters import EEAdapter, JointAdapter
from dataset_replay import EpisodeReader
from external.joint_to_ee.ee_to_joints import EEToJointsConverter
from external.joint_to_ee.kinematics import make_kinematics
from robot_control import RobotController, build_stationary_robot
from trossen_bridge import TrossenOpenPIBridge


class LiveRunner:
    """Runs TrossenOpenPIBridge.run_episode with a telemetry sink."""

    def __init__(self, kind: str, config: dict, sink) -> None:
        adapter = JointAdapter()
        if config.get("adapter") == "ee":
            conv = EEToJointsConverter(
                make_kinematics(),
                orientation_weight=float(config.get("ik_orientation_weight", 0.01)),
                pos_tol_m=float(config.get("ik_pos_tol_m", 1e-3)),
            )
            adapter = EEAdapter(conv)
        self._bridge = TrossenOpenPIBridge(
            policy_server_host=config.get("policy_host", "localhost"),
            policy_server_port=int(config.get("policy_port", 8000)),
            control_frequency=int(config.get("control_freq", 25)),
            test_mode=config.get("mode", "test"),
            max_steps=int(config.get("max_steps", 1000)),
            rate_of_inference=int(config.get("rate_of_inference", 20)),
            ensemble_type=config.get("ensemble_type", "exp"),
            cogact_mode=config.get("cogact_mode", "cogact"),
            async_inference=bool(config.get("async_inference", False)),
            use_left_arm_only=bool(config.get("use_left_arm_only", False)),
            use_right_arm_only=bool(config.get("use_right_arm_only", False)),
            starvla=bool(config.get("starvla", False)),
            adapter=adapter,
            sink=sink,
        )
        self._prompt = config.get("task_prompt", "")

    def run(self) -> None:
        self._bridge.run_episode(task_prompt=self._prompt)

    def stop(self) -> None:
        self._bridge.is_running = False

    def estop(self) -> None:
        self._bridge.is_running = False
        try:
            self._bridge.move_to_sleep_position(duration=10.0)
        finally:
            self._bridge.cleanup()


class ReplayRunner:
    """Replays a dataset episode (EE -> IK -> robot) with a telemetry sink."""

    def __init__(self, kind: str, config: dict, sink) -> None:
        self._sink = sink
        self._stopped = False
        reader = EpisodeReader(config["dataset_dir"])
        self._episode = reader.read_episode(int(config.get("episode_index", 0)))
        self._control_freq = int(config.get("control_freq") or self._episode.fps)
        self._converter = EEToJointsConverter(
            make_kinematics(),
            orientation_weight=float(config.get("ik_orientation_weight", 0.01)),
            pos_tol_m=float(config.get("ik_pos_tol_m", 1e-3)),
        )
        robot = build_stationary_robot(with_cameras=False)
        self._controller = RobotController(robot, control_frequency=self._control_freq,
                                           test_mode=config.get("mode", "test"))

    def run(self) -> None:
        cur = self._controller.current_joints14()
        joints = self._converter.decode_chunk(self._episode.ee_chunk16, cur)
        self._controller.move_to_start_position(joints[0], duration=5.0)
        dt = 1.0 / self._control_freq
        for step, a_t in enumerate(joints[1:], start=1):
            if self._stopped:
                break
            t0 = time.perf_counter()
            ok = self._controller.execute_action(a_t)
            self._sink.on_action(step, np.asarray(a_t), time.time())
            if not ok:
                self._sink.on_status("limit_violation", {"step": step})
                break
            elapsed = time.perf_counter() - t0
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)
        self._controller.disconnect()

    def stop(self) -> None:
        self._stopped = True

    def estop(self) -> None:
        self._stopped = True
        self._controller.move_to_sleep_position(duration=10.0)


def make_runner(kind: str, config: dict, sink):
    if kind == "live":
        return LiveRunner(kind, config, sink)
    if kind == "replay":
        return ReplayRunner(kind, config, sink)
    raise ValueError(f"Unknown session kind {kind!r}")
