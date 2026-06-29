import numpy as np
from webapp.teleop_input import WebTeleopInput


def test_poll_returns_latest_velocity():
    src = WebTeleopInput(max_lin=0.05, max_ang=0.5, grip_rate=0.5, deadzone=0.1)
    src.update({"axes": [1.0, 0, 0, 0, 0, 0], "grip": 0.0})
    cmd = src.poll()
    assert cmd.lin[0] == 0.05
    # held: a second poll without update keeps the velocity
    assert src.poll().lin[0] == 0.05


def test_edges_fire_once_then_clear():
    src = WebTeleopInput()
    src.update({"axes": [0, 0, 0, 0, 0, 0], "grip": 0.0, "switch_arm": True})
    assert src.poll().switch_arm is True
    assert src.poll().switch_arm is False   # auto-cleared


def test_zero_input_when_never_updated():
    src = WebTeleopInput()
    cmd = src.poll()
    assert np.allclose(cmd.lin, 0) and np.allclose(cmd.ang, 0)
