"""Standalone movers: send the arm(s) to a fixed pose, then disconnect.

Used by the Sleep/Home buttons. They implement the Runner interface (run/stop/
estop) so SessionManager runs them on its thread and the single-session guard
prevents them from racing an eval/replay session.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class _PoseMover:
    def __init__(self, kind: str, config: dict, sink) -> None:
        self._config = config
        self._sink = sink
        self._controller = None

    def _goal(self):  # overridden
        raise NotImplementedError

    def run(self) -> None:
        from robot_control import RobotController, build_stationary_robot

        self._sink.on_status("started", {"kind": "move"})
        robot = build_stationary_robot(with_cameras=False)
        self._controller = RobotController(
            robot, control_frequency=int(self._config.get("control_freq", 25)),
            test_mode=self._config.get("mode", "autonomous"),
        )
        try:
            self._controller.move_to_start_position(self._goal(), duration=5.0)
        finally:
            self._controller.disconnect()
            self._sink.on_status("stopped", {"kind": "move"})

    def stop(self) -> None:
        ...

    def estop(self) -> None:
        if self._controller is not None:
            self._controller.move_to_sleep_position(duration=10.0)


class SleepRunner(_PoseMover):
    def _goal(self):
        from robot_control import RobotController
        return RobotController.SLEEP_POSITION


class HomeRunner(_PoseMover):
    def _goal(self):
        import robot_control
        return robot_control.HOME_POSITION
