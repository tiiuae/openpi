import numpy as np
from teleop import TeleopController, TeleopCommand


class FakeConverter:
    """Identity-ish stand-in: avoids placo. Records IK seeds it was given."""
    def __init__(self):
        self.seeds = []

    def joints14_to_ee16(self, joints14):
        p = np.zeros(16, dtype=float)
        p[3] = 1.0
        p[8 + 3] = 1.0
        # encode the first joint into x so we can detect re-seeding
        p[0] = float(joints14[0])
        return p

    def decode_chunk(self, chunk16, seed14):
        self.seeds.append(np.asarray(seed14, dtype=float).copy())
        # map pose16[0] (left x) back into joint 0 so motion is observable
        out = np.zeros((len(chunk16), 14), dtype=float)
        out[:, 0] = chunk16[:, 0]
        return out


class RecordingSink:
    def __init__(self):
        self.actions = []
        self.images = []
        self.status = []

    def on_action(self, step, action, ts): self.actions.append((step, np.asarray(action)))
    def on_images(self, images, ts): self.images.append(images)
    def on_status(self, kind, payload): self.status.append((kind, payload))
    def on_log(self, *a): pass


class OneShotInput:
    def __init__(self, cmd): self._cmd = cmd
    def poll(self): return self._cmd


def test_detached_step_streams_joints_and_no_images():
    conv, sink = FakeConverter(), RecordingSink()
    cmd = TeleopCommand(lin=np.array([1.0, 0, 0]), ang=np.zeros(3), grip=0.0, arm="left")
    ctrl = TeleopController(conv, sink, OneShotInput(cmd), controller=None,
                            start14=np.zeros(14), control_freq=10)
    ctrl.step(cmd, dt=1.0)
    assert len(sink.actions) == 1
    # left x advanced by lin*dt = 1.0 -> joint 0 ~ 1.0
    assert abs(sink.actions[0][1][0] - 1.0) < 1e-6
    assert sink.images == []          # detached: never streams camera frames


def test_detached_step_feeds_decoded_joints_back_as_next_seed():
    conv, sink = FakeConverter(), RecordingSink()
    cmd = TeleopCommand(lin=np.array([0.5, 0, 0]), ang=np.zeros(3), grip=0.0, arm="left")
    ctrl = TeleopController(conv, sink, OneShotInput(cmd), controller=None,
                            start14=np.zeros(14), control_freq=10)
    ctrl.step(cmd, dt=1.0)
    ctrl.step(cmd, dt=1.0)
    # second decode_chunk seed == first decoded joints (continuity)
    assert conv.seeds[1][0] == sink.actions[0][1][0]


class FakeRobot:
    def __init__(self):
        self._obs = {"cam_high": np.zeros((2, 2, 3), dtype=np.uint8), "j.pos": 0.0}
    def get_observation(self):
        return self._obs


class FakeRobotController:
    """Stands in for RobotController in test/autonomous paths."""
    def __init__(self, execute_ok=True):
        self.robot = FakeRobot()
        self.executed = []
        self.moves = []
        self._ok = execute_ok
    def execute_action(self, joints, feedforward_velocity=None):
        self.executed.append(np.asarray(joints)); return self._ok
    def move_to_start_position(self, goal, duration=5.0):
        self.moves.append(np.asarray(goal))


def test_go_home_reseeds_target_to_fk_of_home():
    import teleop
    conv, sink = FakeConverter(), RecordingSink()
    ctrl = TeleopController(conv, sink, OneShotInput(None), controller=None,
                            start14=np.zeros(14), control_freq=10)
    home_cmd = TeleopCommand(lin=np.zeros(3), ang=np.zeros(3), go_home=True)
    ctrl.step(home_cmd, dt=0.1)
    # seed14 now equals HOME_POSITION, target16 == FK(HOME)
    np.testing.assert_allclose(ctrl.seed14, teleop.HOME_POSITION)
    assert ctrl.target16[0] == float(teleop.HOME_POSITION[0])
    assert ("teleop_move", {"target": "home"}) in sink.status


def test_autonomous_step_executes_and_streams_images():
    conv, sink = FakeConverter(), RecordingSink()
    fc = FakeRobotController()
    cmd = TeleopCommand(lin=np.array([0.1, 0, 0]), ang=np.zeros(3), arm="left")
    ctrl = TeleopController(conv, sink, OneShotInput(cmd), controller=fc,
                            start14=np.zeros(14), control_freq=10,
                            cam_keys=["cam_high"])
    ctrl.step(cmd, dt=1.0)
    assert len(fc.executed) == 1            # real execute attempted
    assert len(sink.images) == 1            # camera frame streamed
    assert "cam_high" in sink.images[0]


def test_firmware_fault_emits_status():
    conv, sink = FakeConverter(), RecordingSink()
    fc = FakeRobotController(execute_ok=False)
    cmd = TeleopCommand(lin=np.array([0.1, 0, 0]), ang=np.zeros(3), arm="left")
    ctrl = TeleopController(conv, sink, OneShotInput(cmd), controller=fc,
                            start14=np.zeros(14), control_freq=10)
    ctrl.step(cmd, dt=1.0)
    assert any(k == "firmware_error" for k, _ in sink.status)
