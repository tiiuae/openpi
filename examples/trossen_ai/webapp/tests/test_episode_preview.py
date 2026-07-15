from dataclasses import dataclass

import numpy as np

from episode_preview import clamp_velocity_spikes, build_trajectory


def test_no_spike_when_motion_under_limit():
    # Two frames, every joint moves 0.1 rad over dt=0.1s -> 1.0 rad/s < 3.0.
    raw = np.zeros((2, 14), dtype=np.float32)
    raw[1] = 0.1
    clamped, velocity, spikes = clamp_velocity_spikes(raw, dt=0.1, max_speed=3.0)
    assert spikes == []
    # Under the limit: clamped == raw.
    assert np.allclose(clamped, raw)
    # velocity[0] is 0 (no previous frame); velocity[1] == 1.0 rad/s.
    assert np.allclose(velocity[0], 0.0)
    assert np.allclose(velocity[1], 1.0)


def test_spike_detected_and_clamped():
    # Joint 0 jumps 1.0 rad over dt=0.1s -> 10 rad/s, well over 3.0.
    raw = np.zeros((2, 14), dtype=np.float32)
    raw[1, 0] = 1.0
    clamped, velocity, spikes = clamp_velocity_spikes(raw, dt=0.1, max_speed=3.0)
    assert spikes == [1]
    # Clamp caps the step at max_speed*dt = 0.3 rad.
    assert np.isclose(clamped[1, 0], 0.3)
    assert np.isclose(velocity[1, 0], 10.0)


def test_clamp_spreads_jump_over_multiple_frames():
    # A single 1.0 rad jump held flat afterwards migrates 0.3 rad/frame.
    raw = np.zeros((4, 14), dtype=np.float32)
    raw[1:, 0] = 1.0
    clamped, _, _ = clamp_velocity_spikes(raw, dt=0.1, max_speed=3.0)
    assert np.isclose(clamped[1, 0], 0.3)
    assert np.isclose(clamped[2, 0], 0.6)
    assert np.isclose(clamped[3, 0], 0.9)


def test_build_trajectory_shapes_and_spikes():
    # Fake episode: 3 frames, 16-D EE chunk (values irrelevant to the fake IK).
    @dataclass
    class FakeEpisode:
        ee_chunk16: np.ndarray
        fps: int
        task: str | None

    class FakeReader:
        def __init__(self, dataset_dir):
            pass

        def read_episode(self, idx):
            return FakeEpisode(ee_chunk16=np.zeros((3, 16), np.float32),
                               fps=30, task="demo")

    class FakeConverter:
        # Ignores EE, returns a raw joint trajectory with a spike on joint 0.
        def decode_chunk(self, chunk16, seed14):
            raw = np.zeros((len(chunk16), 14), np.float32)
            raw[2, 0] = 1.0  # 1.0 rad jump at frame 2
            return raw

    data = build_trajectory(
        dataset_dir="/unused", episode_index=0,
        control_freq=10, max_joint_speed=3.0,
        reader_factory=lambda d: FakeReader(d),
        converter_factory=lambda: FakeConverter(),
    )
    assert data["n_frames"] == 3
    assert data["fps"] == 30
    assert data["control_freq"] == 10
    assert data["max_joint_speed"] == 3.0
    assert len(data["joints_raw"]) == 3 and len(data["joints_raw"][0]) == 14
    assert len(data["joints_clamped"]) == 3
    assert len(data["ee_pos"]["left"]) == 3 and len(data["ee_pos"]["left"][0]) == 3
    assert len(data["ee_pos"]["right"]) == 3
    # dt = 1/10 = 0.1s; 1.0 rad jump -> 10 rad/s > 3.0 -> spike at frame 2.
    assert data["spikes"] == [2]
    # JSON-able: plain lists/numbers, no numpy.
    assert isinstance(data["joints_raw"][0][0], float)
