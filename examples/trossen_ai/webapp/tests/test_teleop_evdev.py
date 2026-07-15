from teleop_evdev import state_to_command


def test_left_stick_maps_to_translation():
    # axis state normalized [-1,1]; stick up = -1 -> +X
    state = {"LS_Y": -1.0, "LS_X": 0.0, "RS_X": 0.0, "RS_Y": 0.0,
             "LT": 0.0, "RT": 0.0, "Z_UP": 0, "Z_DOWN": 0,
             "GRIP_CLOSE": 0, "GRIP_OPEN": 0}
    cmd = state_to_command(state, max_lin=0.05, max_ang=0.5, grip_rate=0.5,
                           deadzone=0.1, arm="left")
    assert cmd.lin[0] == 0.05      # +X
    assert cmd.lin[1] == 0.0


def test_bumpers_map_to_z_and_grip_buttons_to_rate():
    state = {"LS_Y": 0.0, "LS_X": 0.0, "RS_X": 0.0, "RS_Y": 0.0,
             "LT": 0.0, "RT": 0.0, "Z_UP": 1, "Z_DOWN": 0,
             "GRIP_CLOSE": 1, "GRIP_OPEN": 0}
    cmd = state_to_command(state, max_lin=0.05, max_ang=0.5, grip_rate=0.5,
                           deadzone=0.1, arm="right")
    assert cmd.lin[2] == 0.05      # +Z from Z_UP
    assert cmd.grip == -0.5        # close = negative grip rate
    assert cmd.arm == "right"
