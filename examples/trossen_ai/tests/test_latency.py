import time

import numpy as np
import pytest

from external.joint_to_ee import constants as C

pytestmark = pytest.mark.integration


def test_decode_chunk_latency_under_budget():
    from external.joint_to_ee.ee_to_joints import EEToJointsConverter
    from external.joint_to_ee.kinematics import make_kinematics

    conv = EEToJointsConverter(make_kinematics())
    q14 = np.zeros(14)
    q14[C.OBS_LEFT_JOINTS] = [0.1, 0.3, 0.4, 0.0, 0.2, 0.0]
    q14[C.OBS_RIGHT_JOINTS] = [-0.1, 0.3, 0.4, 0.0, 0.2, 0.0]
    chunk = np.repeat(conv.joints14_to_ee16(q14)[None, :], 25, axis=0)

    conv.decode_chunk(chunk[:1], q14)  # warm up placo
    t0 = time.perf_counter()
    conv.decode_chunk(chunk, q14)
    dt = time.perf_counter() - t0

    print(f"\ndecode_chunk(25) = {dt * 1e3:.1f} ms ({dt / 25 * 1e3:.2f} ms/step, 50 IK solves)")
    assert dt < 1.0  # generous ceiling; tighten after measuring on the rig
