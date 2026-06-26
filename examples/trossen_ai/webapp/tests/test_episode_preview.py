import numpy as np

from episode_preview import clamp_velocity_spikes


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
