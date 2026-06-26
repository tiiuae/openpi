"""Hardware-bound runners adapting the bridge and replay tool to the Runner
interface used by SessionManager.

Construction is intentionally cheap: all blocking work (policy-server connect,
robot build/connect, episode load) happens in run(), which executes on the
session thread. stop() flips a flag honored by the connect wait and the loops,
so a session can be cancelled even while still connecting.

`config` is the dict produced by the web form. Constructors forward only the
keys they use; unknown keys are ignored.
"""
from __future__ import annotations

import logging

from policy_connect import ConnectStopped, ConnectTimeout, wait_for_policy_server

logger = logging.getLogger(__name__)


class LiveRunner:
    """Runs TrossenOpenPIBridge.run_episode with a telemetry sink."""

    def __init__(self, kind: str, config: dict, sink) -> None:
        self._config = config
        self._sink = sink
        self._stopped = False
        self._bridge = None

    def _make_bridge(self, config: dict):
        # Imported lazily so this module loads off-robot.
        from adapters import EEAdapter, JointAdapter
        from external.joint_to_ee.ee_to_joints import EEToJointsConverter
        from external.joint_to_ee.kinematics import make_kinematics
        from trossen_bridge import TrossenOpenPIBridge

        adapter = JointAdapter()
        if config.get("adapter") == "ee":
            conv = EEToJointsConverter(
                make_kinematics(),
                orientation_weight=float(config.get("ik_orientation_weight", 0.01)),
                pos_tol_m=float(config.get("ik_pos_tol_m", 1e-3)),
            )
            adapter = EEAdapter(conv)
        return TrossenOpenPIBridge(
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
            smooth_streaming=bool(config.get("smooth_streaming", False)),
            min_time_to_move_multiplier=float(config.get("min_time_to_move_multiplier", 3.0)),
            loop_rate=int(config.get("loop_rate", config.get("control_freq", 25))),
            max_joint_speed=float(config.get("max_joint_speed", 3.0)),
            adapter=adapter,
            sink=self._sink,
        )

    def run(self) -> None:
        cfg = self._config
        host = cfg.get("policy_host", "localhost")
        port = int(cfg.get("policy_port", 8000))
        timeout = float(cfg.get("connect_timeout", 15.0))
        self._sink.on_status("connecting", {"host": host, "port": port})
        try:
            wait_for_policy_server(host, port, timeout, should_stop=lambda: self._stopped)
        except ConnectStopped:
            self._sink.on_status("stopped", {"reason": "cancelled before connect"})
            return
        except ConnectTimeout as exc:
            logger.error("%s", exc)
            self._sink.on_status("connect_failed", {"message": str(exc)})
            return
        if self._stopped:
            self._sink.on_status("stopped", {"reason": "cancelled"})
            return

        self._bridge = self._make_bridge(cfg)
        try:
            self._bridge.run_episode(task_prompt=cfg.get("task_prompt", ""))
        finally:
            self._bridge.cleanup()

    def stop(self) -> None:
        self._stopped = True
        if self._bridge is not None:
            self._bridge.is_running = False

    def estop(self) -> None:
        self._stopped = True
        if self._bridge is not None:
            self._bridge.is_running = False
            try:
                self._bridge.move_to_sleep_position(duration=10.0)
            finally:
                self._bridge.cleanup()


class ReplayRunner:
    """Replays a dataset episode (EE -> IK -> robot) with a telemetry sink."""

    def __init__(self, kind: str, config: dict, sink) -> None:
        self._config = config
        self._sink = sink
        self._stopped = False
        self._controller = None

    def run(self) -> None:
        import time

        import numpy as np
        from dataset_replay import EpisodeReader
        from external.joint_to_ee.ee_to_joints import EEToJointsConverter
        from external.joint_to_ee.kinematics import make_kinematics
        from robot_control import RobotController, build_stationary_robot, limit_joint_velocity

        cfg = self._config
        mode = cfg.get("mode", "test")
        reader = EpisodeReader(cfg["dataset_dir"])
        episode = reader.read_episode(int(cfg.get("episode_index", 0)))
        # NB: cfg values arrive as strings; "0" is truthy, so coerce to int
        # *before* falling back to the episode fps (else dt = 1/0).
        control_freq = int(cfg.get("control_freq") or 0) or int(episode.fps)
        max_joint_speed = float(cfg.get("max_joint_speed", 3.0))
        converter = EEToJointsConverter(
            make_kinematics(),
            orientation_weight=float(cfg.get("ik_orientation_weight", 0.01)),
            pos_tol_m=float(cfg.get("ik_pos_tol_m", 1e-3)),
        )
        if self._stopped:
            self._sink.on_status("stopped", {"reason": "cancelled"})
            return

        # Test mode is a pure dry run: NEVER connect the arm (connecting wakes/
        # homes it). Seed IK from the home pose and stream the decoded+clamped
        # trajectory to the UI without any hardware. Only autonomous mode builds
        # and moves the real robot.
        if mode != "autonomous":
            cur = np.zeros(14, dtype=float)
            joints = converter.decode_chunk(episode.ee_chunk16, cur)
            dt = 1.0 / control_freq
            last = np.asarray(joints[0], dtype=float).flatten()
            self._sink.on_action(0, last, time.time())
            for step, a_t in enumerate(joints[1:], start=1):
                if self._stopped:
                    break
                a_cmd = limit_joint_velocity(last, a_t, dt, max_joint_speed)
                last = a_cmd
                self._sink.on_action(step, np.asarray(a_cmd), time.time())
                time.sleep(dt)
            self._sink.on_status("stopped", {"reason": "test_complete"})
            return

        robot = build_stationary_robot(
            with_cameras=False,
            min_time_to_move_multiplier=float(cfg.get("min_time_to_move_multiplier", 3.0)),
            loop_rate=int(cfg.get("loop_rate", 30)),
        )
        self._controller = RobotController(
            robot, control_frequency=control_freq, test_mode=mode,
            smooth_streaming=bool(cfg.get("smooth_streaming", False)),
        )
        try:
            cur = self._controller.current_joints14()
            joints = converter.decode_chunk(episode.ee_chunk16, cur)
            self._controller.move_to_start_position(joints[0], duration=5.0)
            dt = 1.0 / control_freq
            last = np.asarray(joints[0], dtype=float).flatten()
            for step, a_t in enumerate(joints[1:], start=1):
                if self._stopped:
                    break
                # Cap per-step joint velocity so a spurious IK branch-flip in the
                # decoded trajectory can't command an over-limit jump (firmware
                # "joint velocity limit exceeded"). Feed the clamped target back
                # as `last` so the arm keeps migrating toward the true target.
                a_cmd = limit_joint_velocity(last, a_t, dt, max_joint_speed)
                last = a_cmd
                t0 = time.perf_counter()
                ok = self._controller.execute_action(a_cmd)
                self._sink.on_action(step, np.asarray(a_cmd), time.time())
                if not ok:
                    self._sink.on_status("firmware_error", {"step": step})
                    break
                elapsed = time.perf_counter() - t0
                if dt - elapsed > 0:
                    time.sleep(dt - elapsed)
        finally:
            self._controller.disconnect()

    def stop(self) -> None:
        self._stopped = True

    def estop(self) -> None:
        self._stopped = True
        if self._controller is not None:
            self._controller.move_to_sleep_position(duration=10.0)


def make_runner(kind: str, config: dict, sink):
    if kind == "live":
        return LiveRunner(kind, config, sink)
    if kind == "replay":
        return ReplayRunner(kind, config, sink)
    raise ValueError(f"Unknown session kind {kind!r}")
