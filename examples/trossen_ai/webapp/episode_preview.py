"""Compute a dataset episode's joint trajectory for off-robot preview.

Pure-ish data layer: ``clamp_velocity_spikes`` is pure NumPy (no robot, no IK)
and carries the velocity-clamp + spike-detection logic. ``build_trajectory``
wires the dataset reader and the EE->joints IK decoder to it and returns a
JSON-able dict for the Replay preview UI. The robot is never moved.
"""
from __future__ import annotations

import numpy as np


# Mirror of the runner's clamp (robot_control.limit_joint_velocity) so the
# preview reflects exactly what the hardware will execute.
def clamp_velocity_spikes(raw14: np.ndarray, dt: float, max_speed: float):
    """Return (clamped (N,14), velocity (N,14), spikes list[int]).

    - clamped: each step's per-joint delta bounded to ``max_speed*dt``, fed
      forward so a jump migrates across frames (matches limit_joint_velocity).
    - velocity: per-step raw joint speed ``|raw[t]-raw[t-1]|/dt`` (row 0 = 0).
    - spikes: frame indices where any joint's raw velocity exceeds max_speed.
    """
    raw = np.asarray(raw14, dtype=float)
    n = len(raw)
    clamped = np.zeros_like(raw)
    velocity = np.zeros_like(raw)
    spikes: list[int] = []
    if n == 0:
        return clamped, velocity, spikes
    max_step = float(max_speed) * float(dt)
    clamped[0] = raw[0]
    last = raw[0].copy()
    for t in range(1, n):
        velocity[t] = np.abs(raw[t] - raw[t - 1]) / float(dt)
        if np.any(velocity[t] > float(max_speed)):
            spikes.append(t)
        delta = np.clip(raw[t] - last, -max_step, max_step)
        last = last + delta
        clamped[t] = last
    return clamped, velocity, spikes
