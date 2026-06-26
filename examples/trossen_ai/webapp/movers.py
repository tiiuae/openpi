"""Standalone movers: send the arm(s) to a fixed pose and HOLD it there.

Used by the Sleep/Home buttons. They implement the Runner interface (run/stop/
estop) so SessionManager runs them on its thread and the single-session guard
prevents them from racing an eval/replay session.

The mover moves to the target, then blocks until the user presses Stop. The
servos hold the commanded pose under torque while blocked, so no further
commands are sent. Disconnecting immediately after the move (the previous
behaviour) is what triggered the driver's per-arm park sequence — the "both
arms home, then sleep, then the right arm again" double-move. We disconnect
only once the user actually stops.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class _PoseMover:
    def __init__(self, kind: str, config: dict, sink) -> None:
        self._kind = kind
        self._config = config
        self._sink = sink
        self._controller = None
        self._stop = threading.Event()
        self._estop = False

    def _goal(self):  # overridden
        raise NotImplementedError

    def run(self) -> None:
        from robot_control import RobotController, build_stationary_robot

        logger.info("%s mover: connecting robot (no cameras)…", self._kind)
        self._sink.on_status("started", {"kind": self._kind})
        robot = build_stationary_robot(with_cameras=False)
        logger.info("%s mover: robot connected", self._kind)
        self._controller = RobotController(
            robot, control_frequency=int(self._config.get("control_freq", 25)),
            test_mode=self._config.get("mode", "test"),
        )
        try:
            if self._stop.is_set():
                logger.info("%s mover: stop requested before move; aborting", self._kind)
                return
            logger.info("%s mover: moving to target over 5s…", self._kind)
            self._controller.move_to_start_position(self._goal(), duration=5.0)
            logger.info("%s mover: target reached; holding until stop", self._kind)
            # Block (no further commands) until the user stops. The arm holds the
            # commanded pose via servo torque; disconnecting is deferred so the
            # driver's park sequence doesn't fire the moment we arrive.
            self._stop.wait()
            if self._estop:
                logger.warning("%s mover: E-STOP — moving to sleep before release", self._kind)
                self._controller.move_to_sleep_position(duration=10.0)
            else:
                logger.info("%s mover: stop received; releasing", self._kind)
        finally:
            self._controller.disconnect()
            logger.info("%s mover: robot disconnected", self._kind)
            self._sink.on_status("stopped", {"kind": self._kind})

    def stop(self) -> None:
        self._stop.set()

    def estop(self) -> None:
        # Defer the sleep move to run()'s own thread to avoid racing disconnect.
        self._estop = True
        self._stop.set()


class SleepRunner(_PoseMover):
    def _goal(self):
        from robot_control import RobotController
        return RobotController.SLEEP_POSITION


class HomeRunner(_PoseMover):
    def _goal(self):
        import robot_control
        return robot_control.HOME_POSITION
