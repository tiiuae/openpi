"""Opt-in observation history for policies that condition on the recent past (`main.py --send_history N`).

OFF by default. Nothing here runs, and the request is byte-for-byte unchanged, unless `--send_history` is
given a positive N. Every model that is served from a single observation is unaffected.

Why the client and not the server. FLUX 3 Action conditions on the measured state at N = 8 consecutive
control ticks, on the command sent on the tick before each of them, and on the camera frames of the oldest
and the current of those ticks. The server sees an observation only when the client queries -- every
`--rate_of_inference` ticks in sync mode, roughly once per inference in async mode -- so it cannot rebuild
ticks t-7 .. t-1. Only the client sees every tick.

What is recorded matches the authors' own control loop
(`flux_action/processing/history.py::ObservationHistory`, driven by `select_action` at every tick):

  * one entry per control tick: the measured joint positions and the wire images of that tick, and the
    command sent to the robot on the PREVIOUS tick -- the last one that went through `execute_action`, i.e.
    after arm freezing and, in async mode, after the ensemble blend. That is what physically moved the arm;
  * the first tick of an episode (or after a pause or intervention) fills all N slots with that tick's
    observation and uses its measured state as the command, exactly as the authors pad.

What is sent, as ONE extra top-level key of the same flat request:

    request["history"] = {
        "state":         (N, D) float64  measured joint positions, ticks t-N+1 .. t (last row == request["state"])
        "command":       (N, D) float64  command sent on the tick before each of those ticks
        "oldest_images": {cam: (3, H, W) uint8}  the images of tick t-N+1, same preprocessing as request["images"]
        "ticks":         N
        "valid":         real ticks in the window (<= N; the rest repeat the first tick)
    }

Only the oldest tick's images are sent because FLUX encodes only two snapshots (oldest and current); the
current tick's images are request["images"] as always. That adds one image set (~450 KB for three 224x224
cameras) to each request.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class HistoryWindow:
    def __init__(self, ticks: int):
        if ticks < 1:
            raise ValueError("--send_history needs a positive number of ticks")
        self.ticks = int(ticks)
        self.reset()

    def reset(self) -> None:
        """Episode boundary, pause or intervention: the next recorded tick starts a fresh, padded window."""
        self._states: deque[np.ndarray] = deque(maxlen=self.ticks)
        self._commands: deque[np.ndarray] = deque(maxlen=self.ticks)
        self._images: deque[dict] = deque(maxlen=self.ticks)
        self._last_command: np.ndarray | None = None
        self._real = 0

    def record(self, observation: dict) -> None:
        """Once per control tick, with that tick's wire observation, before the tick's command is sent."""
        state = np.array(observation["state"], dtype=np.float64)
        command = state if self._last_command is None else self._last_command
        for _ in range(self.ticks if not self._states else 1):
            self._states.append(state)
            self._commands.append(command)
            self._images.append(observation["images"])
        self._real = min(self._real + 1, self.ticks)

    def command_sent(self, command: np.ndarray) -> None:
        """Every command that goes to the robot (execute_action); the last one before a tick is that tick's."""
        self._last_command = np.array(command, dtype=np.float64)

    def attach(self, observation: dict) -> dict:
        """A copy of the request with the current window under "history" (call after record())."""
        if not self._states:
            raise RuntimeError("attach() before any record()")
        return {
            **observation,
            "history": {
                "state": np.stack(self._states),
                "command": np.stack(self._commands),
                "oldest_images": dict(self._images[0]),
                "ticks": self.ticks,
                "valid": self._real,
            },
        }
