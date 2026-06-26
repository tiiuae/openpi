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


# 14-D home-pose seed for IK frame 0 when no live robot pose is supplied.
HOME_SEED14 = np.zeros(14, dtype=np.float32)

# EE position columns inside each arm's 8-D pose block (x,y,z,qw,qx,qy,qz,grip).
_EE_LEFT_POS = slice(0, 3)
_EE_RIGHT_POS = slice(8, 11)


def _default_reader_factory(dataset_dir):
    from dataset_replay import EpisodeReader  # lazy: pandas/dataset only
    return EpisodeReader(dataset_dir)


def _default_converter_factory():
    from external.joint_to_ee.ee_to_joints import EEToJointsConverter
    from external.joint_to_ee.kinematics import make_kinematics
    return EEToJointsConverter(make_kinematics())


def build_trajectory(dataset_dir, episode_index, control_freq, max_joint_speed,
                     seed_joints=None, reader_factory=None, converter_factory=None):
    """Return a JSON-able dict describing one episode's joint/EE trajectory.

    reader_factory(dataset_dir) -> reader with .read_episode(idx) -> episode
        having .ee_chunk16 (N,16) and .fps. converter_factory() -> object with
        .decode_chunk(chunk16, seed14) -> (N,14). Both default to the real,
        robot-free implementations; injected in tests to avoid placo/URDF.
    """
    reader = (reader_factory or _default_reader_factory)(dataset_dir)
    converter = (converter_factory or _default_converter_factory)()
    episode = reader.read_episode(int(episode_index))

    seed = (np.asarray(seed_joints, dtype=np.float32).flatten()
            if seed_joints is not None else HOME_SEED14.copy())
    raw = np.asarray(converter.decode_chunk(episode.ee_chunk16, seed), dtype=float)

    control_freq = int(control_freq or episode.fps)
    dt = 1.0 / control_freq
    clamped, velocity, spikes = clamp_velocity_spikes(raw, dt, float(max_joint_speed))

    ee = np.asarray(episode.ee_chunk16, dtype=float)
    return {
        "fps": int(episode.fps),
        "control_freq": control_freq,
        "n_frames": int(len(raw)),
        "max_joint_speed": float(max_joint_speed),
        "joints_raw": raw.astype(float).tolist(),
        "joints_clamped": clamped.astype(float).tolist(),
        "ee_pos": {
            "left": ee[:, _EE_LEFT_POS].tolist(),
            "right": ee[:, _EE_RIGHT_POS].tolist(),
        },
        "velocity": velocity.astype(float).tolist(),
        "spikes": spikes,
    }
