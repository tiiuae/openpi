"""Thread-safe latch turning browser WS messages into TeleopCommands.

The WebSocket handler calls update() on every 'teleop_input'/'switch_arm'
message; the TeleopController thread calls poll() each control tick. Velocities
persist while held; edge events fire exactly once.
"""
from __future__ import annotations

import threading

from teleop import TeleopCommand, scale_axes


class WebTeleopInput:
    def __init__(self, max_lin: float = 0.05, max_ang: float = 0.5,
                 grip_rate: float = 0.5, deadzone: float = 0.1) -> None:
        self._lock = threading.Lock()
        self._axes = [0.0] * 6
        self._grip = 0.0
        self._switch = False
        self._home = False
        self._sleep = False
        self.max_lin, self.max_ang = max_lin, max_ang
        self.grip_rate, self.deadzone = grip_rate, deadzone

    def update(self, msg: dict) -> None:
        with self._lock:
            if "axes" in msg:
                a = list(msg["axes"])[:6]
                self._axes = (a + [0.0] * 6)[:6]
            if "grip" in msg:
                self._grip = float(msg["grip"])
            self._switch = self._switch or bool(msg.get("switch_arm"))
            self._home = self._home or bool(msg.get("go_home"))
            self._sleep = self._sleep or bool(msg.get("go_sleep"))

    def poll(self) -> TeleopCommand:
        with self._lock:
            lin, ang = scale_axes(self._axes, self.max_lin, self.max_ang, self.deadzone)
            grip = 0.0 if abs(self._grip) < self.deadzone else self._grip * self.grip_rate
            cmd = TeleopCommand(lin=lin, ang=ang, grip=grip,
                                switch_arm=self._switch, go_home=self._home,
                                go_sleep=self._sleep)
            self._switch = self._home = self._sleep = False  # consume edges
            return cmd
