"""Runner wrapping TeleopController for a web session.

Owns a WebTeleopInput so the WS layer can stream operator input into the live
session. Detached mode imports no hardware module. Cheap to construct; all
blocking work (robot connect, control loop) runs in run() on the session thread.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)


class TeleopRunner:
    def __init__(self, kind: str, config: dict, sink) -> None:
        from webapp.teleop_input import WebTeleopInput
        self._config = config
        self._sink = sink
        self._stopped = threading.Event()
        self._controller = None          # RobotController (test/autonomous)
        self._teleop = None              # TeleopController
        self.input = WebTeleopInput(
            max_lin=float(config.get("max_lin", 0.05)),
            max_ang=float(config.get("max_ang", 0.5)),
            grip_rate=float(config.get("grip_rate", 0.5)),
            deadzone=float(config.get("deadzone", 0.1)),
        )

    def _build_converter(self):
        from external.joint_to_ee.ee_to_joints import EEToJointsConverter
        from external.joint_to_ee.kinematics import make_kinematics
        return EEToJointsConverter(
            make_kinematics(),
            orientation_weight=float(self._config.get("ik_orientation_weight", 0.01)),
            pos_tol_m=float(self._config.get("ik_pos_tol_m", 1e-3)),
        )

    def run(self) -> None:
        from robot_control import HOME_POSITION
        from teleop import TeleopController

        cfg = self._config
        mode = cfg.get("mode", "detached")
        freq = int(cfg.get("control_freq", 25))
        converter = self._build_converter()

        controller = None
        cam_keys: list[str] = []
        start14 = np.asarray(HOME_POSITION, float)

        if mode != "detached":
            from robot_control import RobotController, build_stationary_robot
            robot = build_stationary_robot(with_cameras=True)
            cam_keys = list(robot._cameras_ft.keys())  # noqa: SLF001
            controller = RobotController(robot, control_frequency=freq, test_mode=mode)
            start14 = controller.current_joints14()
            if self._stopped.is_set():
                controller.disconnect()
                self._sink.on_status("stopped", {"reason": "cancelled"})
                return

        self._controller = controller
        self._teleop = TeleopController(
            converter, self._sink, self.input, controller=controller,
            start14=start14, control_freq=freq,
            max_lin=float(cfg.get("max_lin", 0.05)),
            max_ang=float(cfg.get("max_ang", 0.5)),
            grip_rate=float(cfg.get("grip_rate", 0.5)),
            max_joint_speed=float(cfg.get("max_joint_speed", 3.0)),
            cam_keys=cam_keys,
        )
        try:
            self._teleop.run(should_stop=self._stopped.is_set)
        finally:
            if controller is not None:
                controller.disconnect()

    def stop(self) -> None:
        self._stopped.set()

    def estop(self) -> None:
        self._stopped.set()
        if self._controller is not None:
            self._controller.move_to_sleep_position(duration=10.0)
