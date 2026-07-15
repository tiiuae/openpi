"""CLI gamepad/keyboard input via evdev.

state_to_command (pure) maps a normalized axis/button snapshot to a
TeleopCommand and is fully unit-tested. EvdevTeleopInput runs a background
thread that maintains that snapshot from /dev/input events; evdev is imported
lazily inside the reader thread so this module loads without evdev present.
"""
from __future__ import annotations

import threading

from teleop import TeleopCommand, scale_axes

# Logical axis snapshot keys (analog in [-1,1], buttons 0/1):
#   LS_X, LS_Y, RS_X, RS_Y, LT, RT  (analog)
#   Z_UP, Z_DOWN, GRIP_CLOSE, GRIP_OPEN  (buttons)


def state_to_command(state: dict, *, max_lin: float, max_ang: float,
                     grip_rate: float, deadzone: float, arm: str,
                     switch_arm: bool = False, go_home: bool = False,
                     go_sleep: bool = False) -> TeleopCommand:
    # translation: LS up(-Y)=+X, LS right(+X)=+Y, bumpers=+/-Z
    raw6 = [
        -state.get("LS_Y", 0.0),                       # tx
        state.get("LS_X", 0.0),                        # ty
        float(state.get("Z_UP", 0)) - float(state.get("Z_DOWN", 0)),  # tz
        state.get("RT", 0.0) - state.get("LT", 0.0),   # roll
        -state.get("RS_Y", 0.0),                       # pitch
        state.get("RS_X", 0.0),                        # yaw
    ]
    lin, ang = scale_axes(raw6, max_lin, max_ang, deadzone)
    grip = (float(state.get("GRIP_OPEN", 0)) - float(state.get("GRIP_CLOSE", 0))) * grip_rate
    return TeleopCommand(lin=lin, ang=ang, grip=grip, arm=arm,
                         switch_arm=switch_arm, go_home=go_home, go_sleep=go_sleep)


class EvdevTeleopInput:
    """Background-thread gamepad reader. Lazy-imports evdev."""

    def __init__(self, device_path: str | None = None, *, max_lin: float = 0.05,
                 max_ang: float = 0.5, grip_rate: float = 0.5, deadzone: float = 0.1):
        self._lock = threading.Lock()
        self._state = {k: 0.0 for k in
                       ("LS_X", "LS_Y", "RS_X", "RS_Y", "LT", "RT",
                        "Z_UP", "Z_DOWN", "GRIP_CLOSE", "GRIP_OPEN")}
        self._arm = "left"
        self._edges = {"switch_arm": False, "go_home": False, "go_sleep": False}
        self._device_path = device_path
        self._cfg = dict(max_lin=max_lin, max_ang=max_ang,
                         grip_rate=grip_rate, deadzone=deadzone)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def poll(self) -> TeleopCommand:
        with self._lock:
            edges = dict(self._edges)
            self._edges = {k: False for k in self._edges}
            cmd = state_to_command(self._state, arm=self._arm,
                                   switch_arm=edges["switch_arm"],
                                   go_home=edges["go_home"],
                                   go_sleep=edges["go_sleep"], **self._cfg)
        if edges["switch_arm"]:
            self._arm = "right" if self._arm == "left" else "left"
        return cmd

    def _reader(self) -> None:  # pragma: no cover - needs a device
        from evdev import InputDevice, ecodes, list_devices

        path = self._device_path or (list_devices()[0] if list_devices() else None)
        if path is None:
            return
        dev = InputDevice(path)
        ABS = {ecodes.ABS_X: ("LS_X", 32767), ecodes.ABS_Y: ("LS_Y", 32767),
               ecodes.ABS_RX: ("RS_X", 32767), ecodes.ABS_RY: ("RS_Y", 32767),
               ecodes.ABS_Z: ("LT", 255), ecodes.ABS_RZ: ("RT", 255)}
        BTN = {ecodes.BTN_TR: "Z_UP", ecodes.BTN_TL: "Z_DOWN",
               ecodes.BTN_SOUTH: "GRIP_CLOSE", ecodes.BTN_EAST: "GRIP_OPEN"}
        for ev in dev.read_loop():
            if self._stop.is_set():
                break
            with self._lock:
                if ev.type == ecodes.EV_ABS and ev.code in ABS:
                    name, scale = ABS[ev.code]
                    self._state[name] = max(-1.0, min(1.0, ev.value / scale))
                elif ev.type == ecodes.EV_ABS and ev.code == ecodes.ABS_HAT0Y:
                    self._edges["go_home"] |= ev.value < 0
                    self._edges["go_sleep"] |= ev.value > 0
                elif ev.type == ecodes.EV_KEY and ev.code in BTN:
                    self._state[BTN[ev.code]] = float(ev.value)
                elif ev.type == ecodes.EV_KEY and ev.code == ecodes.BTN_WEST and ev.value == 1:
                    self._edges["switch_arm"] = True
