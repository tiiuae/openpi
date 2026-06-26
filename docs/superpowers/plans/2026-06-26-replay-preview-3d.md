# Replay Episode Preview (3D + Charts + Safety Gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Replay page a look-before-run tool — precompute a dataset episode's full trajectory (pure IK, no robot), play it back as a 3D URDF animation + value charts with a media-player transport, and gate hardware runs behind a severe velocity-spike warning.

**Architecture:** A new pure-compute backend module (`episode_preview.py`) loads an episode, decodes EE→joints, applies the same per-step velocity clamp the runner uses, and reports raw/clamped trajectories + velocity-spike frames as JSON over a new REST route. The robot URDF + meshes are served statically. The frontend fetches that JSON on a **Preview** click, drives a trimmed Three.js URDF viewer (reused from the existing wizard viewer) and two Chart.js charts from one shared transport controller, and intercepts the hardware **Replay Episode** button with a spike modal.

**Tech Stack:** Python (FastAPI, NumPy, pandas), pytest; vanilla ES-module JS, Chart.js (vendored), Three.js + URDFLoader (CDN importmap + vendored loader).

**Working directory for all paths:** `/home/edgeai/openpi-client-test/examples/trossen_ai`

**Conventions observed in this codebase:**
- Backend tests live in `webapp/tests/`, run with `pytest` from `examples/trossen_ai`. `conftest.py` puts both `webapp/` and `examples/trossen_ai/` on `sys.path`, so tests do bare imports (`import episode_preview`, `import robot_control`).
- Routes that need robot-only deps lazy-import them *inside* the handler so the REST layer loads off-robot (see `/api/episodes` in `webapp/server.py:107`).
- No JS test runner exists. Frontend tasks use **manual browser verification**, not automated tests.
- Joint layout (14-D): indices 0–5 left arm, 6 left gripper, 7–12 right arm, 13 right gripper (`external/joint_to_ee/constants.py`). Matches the viewer's `STATE_IDX`.

---

## Task 1: Pure clamp/velocity/spike math in `episode_preview.py`

The truly testable core: given a raw `(N,14)` joint trajectory, compute the
velocity-clamped trajectory (mirroring `robot_control.limit_joint_velocity`),
the per-step joint velocity, and the spike frame list. No IK, no robot — fully
unit-testable.

**Files:**
- Create: `webapp/episode_preview.py`
- Test: `webapp/tests/test_episode_preview.py`

- [ ] **Step 1: Write the failing test**

Create `webapp/tests/test_episode_preview.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/edgeai/openpi-client-test/examples/trossen_ai && python -m pytest webapp/tests/test_episode_preview.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'episode_preview'`.

- [ ] **Step 3: Write minimal implementation**

Create `webapp/episode_preview.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/edgeai/openpi-client-test/examples/trossen_ai && python -m pytest webapp/tests/test_episode_preview.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /home/edgeai/openpi-client-test
git add examples/trossen_ai/webapp/episode_preview.py examples/trossen_ai/webapp/tests/test_episode_preview.py
git commit -m "feat(replay): clamp/velocity/spike math for episode preview"
```

---

## Task 2: `build_trajectory` wiring (reader + IK + math)

Assemble the JSON payload. The IK decoder and dataset reader are **injectable**
so the test runs without placo/URDF; production passes the real ones.

**Files:**
- Modify: `webapp/episode_preview.py`
- Test: `webapp/tests/test_episode_preview.py`

- [ ] **Step 1: Write the failing test**

Change the import line at the top of `webapp/tests/test_episode_preview.py` to:

```python
from episode_preview import clamp_velocity_spikes, build_trajectory
```

Append to the same file:

```python
from dataclasses import dataclass


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/edgeai/openpi-client-test/examples/trossen_ai && python -m pytest webapp/tests/test_episode_preview.py::test_build_trajectory_shapes_and_spikes -v`
Expected: FAIL — `ImportError: cannot import name 'build_trajectory'`.

- [ ] **Step 3: Write minimal implementation**

Append to `webapp/episode_preview.py`:

```python
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
            if seed_joints is not None else HOME_SEED14)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/edgeai/openpi-client-test/examples/trossen_ai && python -m pytest webapp/tests/test_episode_preview.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
cd /home/edgeai/openpi-client-test
git add examples/trossen_ai/webapp/episode_preview.py examples/trossen_ai/webapp/tests/test_episode_preview.py
git commit -m "feat(replay): build_trajectory assembles preview JSON payload"
```

---

## Task 3: `/api/episode_trajectory` route

**Files:**
- Modify: `webapp/server.py` (add route after the `/api/episodes` route at lines 107-111)
- Test: `webapp/tests/test_server.py`

- [ ] **Step 1: Write the failing test**

Append to `webapp/tests/test_server.py`:

```python
def test_episode_trajectory_route(monkeypatch):
    from webapp import episode_preview as ep

    captured = {}

    def fake_build(dataset_dir, episode_index, control_freq, max_joint_speed,
                   seed_joints=None, **kw):
        captured.update(dataset_dir=dataset_dir, episode_index=episode_index,
                        control_freq=control_freq, max_joint_speed=max_joint_speed)
        return {"n_frames": 2, "spikes": [1]}

    monkeypatch.setattr(ep, "build_trajectory", fake_build)
    client = TestClient(create_app())
    r = client.get("/api/episode_trajectory", params={
        "dataset_dir": "/data/ds", "episode_index": 3,
        "control_freq": 25, "max_joint_speed": 3.0})
    assert r.status_code == 200
    assert r.json()["spikes"] == [1]
    assert captured["dataset_dir"] == "/data/ds"
    assert captured["episode_index"] == 3
    assert captured["control_freq"] == 25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/edgeai/openpi-client-test/examples/trossen_ai && python -m pytest webapp/tests/test_server.py::test_episode_trajectory_route -v`
Expected: FAIL — 404 (route not defined).

- [ ] **Step 3: Write minimal implementation**

In `webapp/server.py`, immediately after the `episodes` route (which ends at line 111), add:

```python
    @app.get("/api/episode_trajectory")
    def episode_trajectory(dataset_dir: str, episode_index: int = 0,
                           control_freq: int = 0, max_joint_speed: float = 3.0):
        from webapp import episode_preview  # lazy: pulls IK/dataset deps on demand
        # Best-effort live seed: only when idle and a robot can be read cheaply.
        seed = None
        if not session.is_running():
            try:
                from robot_control import build_stationary_robot, RobotController
                robot = build_stationary_robot(with_cameras=False)
                seed = RobotController(robot).current_joints14().tolist()
                robot.disconnect()
            except Exception:  # noqa: BLE001 — no robot / busy: fall back to home seed
                seed = None
        return episode_preview.build_trajectory(
            dataset_dir=dataset_dir, episode_index=int(episode_index),
            control_freq=int(control_freq), max_joint_speed=float(max_joint_speed),
            seed_joints=seed,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/edgeai/openpi-client-test/examples/trossen_ai && python -m pytest webapp/tests/test_server.py::test_episode_trajectory_route -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/edgeai/openpi-client-test
git add examples/trossen_ai/webapp/server.py examples/trossen_ai/webapp/tests/test_server.py
git commit -m "feat(replay): /api/episode_trajectory route with best-effort live seed"
```

---

## Task 4: Serve URDF + meshes (`/robot.urdf`, `/pkg/...`)

**Files:**
- Modify: `webapp/server.py`
- Test: `webapp/tests/test_server.py`

- [ ] **Step 1: Write the failing test**

Append to `webapp/tests/test_server.py`:

```python
def test_robot_urdf_and_mesh_served():
    client = TestClient(create_app())
    r = client.get("/robot.urdf")
    assert r.status_code == 200
    assert "robot" in r.text[:500].lower()  # URDF root tag
    # A mesh referenced as package://trossen_arm_description/meshes/... resolves.
    m = client.get("/pkg/trossen_arm_description/meshes/wxai/base_link.stl")
    assert m.status_code == 200
    assert len(m.content) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/edgeai/openpi-client-test/examples/trossen_ai && python -m pytest webapp/tests/test_server.py::test_robot_urdf_and_mesh_served -v`
Expected: FAIL — 404 on `/robot.urdf`.

- [ ] **Step 3: Write minimal implementation**

In `webapp/server.py`, add near the top after the `STATIC_DIR` definition (line 23):

```python
URDF_PKG_DIR = (Path(__file__).parent.parent / "external" / "joint_to_ee"
                / "trossen_arm_description")
URDF_FILE = URDF_PKG_DIR / "urdf" / "generated" / "mobile_ai.urdf"
```

Then, immediately after the existing `app.mount("/static", ...)` line (line 59), add:

```python
    app.mount("/pkg/trossen_arm_description",
              StaticFiles(directory=URDF_PKG_DIR), name="urdf_pkg")

    @app.get("/robot.urdf")
    def robot_urdf():
        return FileResponse(URDF_FILE, media_type="application/xml")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/edgeai/openpi-client-test/examples/trossen_ai && python -m pytest webapp/tests/test_server.py -v`
Expected: PASS (all server tests).

- [ ] **Step 5: Commit**

```bash
cd /home/edgeai/openpi-client-test
git add examples/trossen_ai/webapp/server.py examples/trossen_ai/webapp/tests/test_server.py
git commit -m "feat(replay): serve mobile_ai URDF and mesh package"
```

---

## Task 5: Vendor URDFLoader.js

The CDN importmap covers `three` + addons, but `urdf-loader` is mapped to a
local file in the wizard. Copy that single file into the webapp vendor dir.

**Files:**
- Create: `webapp/static/vendor/URDFLoader.js`

- [ ] **Step 1: Locate the wizard's served loader**

Run: `cd /home/edgeai/openpi-client-test && find get_inspired_from_web_wizard -name 'URDFLoader.js' -not -path '*/node_modules/*'`
Expected: a path under the wizard (served at `/lib/URDFLoader.js`). If found, use it in Step 2. If nothing is found locally, fetch the matching version instead:
`curl -fsSL https://cdn.jsdelivr.net/npm/urdf-loader@0.12.6/src/URDFLoader.js -o /tmp/URDFLoader.js && head -3 /tmp/URDFLoader.js`
Expected: ES-module source importing from `three`.

- [ ] **Step 2: Copy into the webapp vendor dir**

Run (uses the wizard copy if present, else the `/tmp` download from Step 1):
```bash
cd /home/edgeai/openpi-client-test
SRC="$(find get_inspired_from_web_wizard -name 'URDFLoader.js' -not -path '*/node_modules/*' | head -1)"
cp "${SRC:-/tmp/URDFLoader.js}" examples/trossen_ai/webapp/static/vendor/URDFLoader.js
head -3 examples/trossen_ai/webapp/static/vendor/URDFLoader.js
```
Expected: file exists, begins with `three` imports.

- [ ] **Step 3: Verify it imports `three` bare**

Run: `cd /home/edgeai/openpi-client-test && grep -m1 "from 'three'" examples/trossen_ai/webapp/static/vendor/URDFLoader.js`
Expected: a line importing from `'three'` — the importmap (Task 8) maps this to the CDN so all Three.js code shares one instance.

- [ ] **Step 4: Commit**

```bash
cd /home/edgeai/openpi-client-test
git add examples/trossen_ai/webapp/static/vendor/URDFLoader.js
git commit -m "chore(replay): vendor URDFLoader.js for the 3D preview"
```

---

## Task 6: Trimmed URDF viewer module `urdf_view.js`

A single-panel viewer: load the URDF, set joints from a 14-D vector, orbit
camera, view presets. No EE/FK/trails/calibration. (No JS test runner — verified
in the browser in Task 9.)

**Files:**
- Create: `webapp/static/js/urdf_view.js`

- [ ] **Step 1: Write the module**

Create `webapp/static/js/urdf_view.js`:

```javascript
// Trimmed single-panel URDF viewer for the Replay preview. Loads /robot.urdf,
// drives 14-D joints, orbit + view presets. Adapted from the wizard viewer.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ColladaLoader } from 'three/examples/jsm/loaders/ColladaLoader.js';
import URDFLoader from 'urdf-loader';

// Dataset 14-D index -> URDF joint name (left 0-6, right 7-13).
const JOINT_MAP = {
  0: 'follower_left_joint_0',  1: 'follower_left_joint_1',  2: 'follower_left_joint_2',
  3: 'follower_left_joint_3',  4: 'follower_left_joint_4',  5: 'follower_left_joint_5',
  7: 'follower_right_joint_0', 8: 'follower_right_joint_1', 9: 'follower_right_joint_2',
  10: 'follower_right_joint_3', 11: 'follower_right_joint_4', 12: 'follower_right_joint_5',
};
const GRIPPER_MAP = {
  6:  ['follower_left_right_carriage_joint',  'follower_left_left_carriage_joint'],
  13: ['follower_right_right_carriage_joint', 'follower_right_left_carriage_joint'],
};

const VIEWS = {
  top:   { pos: [0, 0, 5],      up: [0, 1, 0] },
  front: { pos: [0, -4.5, 1.2], up: [0, 0, 1] },
  left:  { pos: [-4.5, 0, 1.2], up: [0, 0, 1] },
  right: { pos: [4.5, 0, 1.2],  up: [0, 0, 1] },
  back:  { pos: [0, 4.5, 1.2],  up: [0, 0, 1] },
  iso:   { pos: [3.0, -3.5, 2.0], up: [0, 0, 1] },
};

export class UrdfView {
  constructor(canvas) {
    this.canvas = canvas;
    this.robot = null;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#f0f2f5');
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(3, 2, 5);
    this.scene.add(sun);
    const grid = new THREE.GridHelper(6, 30, 0x999999, 0xcccccc);
    grid.rotation.x = Math.PI / 2;
    this.scene.add(grid);

    this.camera = new THREE.PerspectiveCamera(45, this._aspect(), 0.01, 50);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(...VIEWS.iso.pos);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0, 1);
    this.controls.enableDamping = true;
    this.controls.update();

    window.addEventListener('resize', () => this._resize());
    this._resize();
    this._animate();
  }

  _aspect() {
    const p = this.canvas.parentElement;
    return Math.max(1, p.clientWidth) / Math.max(1, p.clientHeight);
  }

  _resize() {
    const p = this.canvas.parentElement;
    this.renderer.setSize(p.clientWidth, p.clientHeight, false);
    this.camera.aspect = this._aspect();
    this.camera.updateProjectionMatrix();
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  _makeLoader() {
    const loader = new URDFLoader();
    loader.packages = { trossen_arm_description: '/pkg/trossen_arm_description' };
    loader.loadMeshCb = (path, manager, done) => {
      if (/\.dae$/i.test(path)) {
        new ColladaLoader(manager).load(path, dae => {
          dae.scene.updateMatrixWorld(true);
          done(dae.scene);
        }, null, err => done(null, err));
      } else {
        loader.defaultMeshLoader(path, manager, done);
      }
    };
    return loader;
  }

  async load() {
    const loader = this._makeLoader();
    this.robot = await new Promise((resolve, reject) => {
      loader.load('/robot.urdf', r => {
        r.traverse(c => {
          if (c.isMesh) {
            c.material = new THREE.MeshPhongMaterial({ color: 0x8899aa, shininess: 60 });
          }
        });
        resolve(r);
      }, null, reject);
    });
    this.scene.add(this.robot);
  }

  // Apply a 14-D joint vector (radians + gripper metres) to the URDF.
  setFrameJoints(joints14) {
    if (!this.robot || !joints14) return;
    for (const [idx, name] of Object.entries(JOINT_MAP)) {
      const v = joints14[+idx];
      if (v !== undefined) this.robot.setJointValue(name, v);
    }
    for (const [idx, names] of Object.entries(GRIPPER_MAP)) {
      const v = joints14[+idx];
      if (v !== undefined) for (const n of names) this.robot.setJointValue(n, v);
    }
  }

  snapView(name) {
    const v = VIEWS[name] || VIEWS.iso;
    this.camera.up.set(...v.up);
    this.camera.position.set(...v.pos);
    this.controls.target.set(0, 0, 1);
    this.controls.update();
  }
}
```

- [ ] **Step 2: Commit**

```bash
cd /home/edgeai/openpi-client-test
git add examples/trossen_ai/webapp/static/js/urdf_view.js
git commit -m "feat(replay): trimmed single-panel URDF viewer module"
```

---

## Task 7: Charts + transport module `trajectory.js`

Two static Chart.js charts (joints, EE position), a frame-cursor plugin, and a
`Transport` controller that advances both charts' cursor + the 3D viewer.

**Files:**
- Create: `webapp/static/js/trajectory.js`

- [ ] **Step 1: Write the module**

Create `webapp/static/js/trajectory.js`. (Reuses `PALETTE` and `JOINT_NAMES_14`
exported from the existing `charts.js`.)

```javascript
// Static trajectory charts + media-player transport for the Replay preview.
import { PALETTE, JOINT_NAMES_14 } from "./charts.js";

// Chart.js plugin: a vertical cursor line at chart.$frameX (data x value).
const cursorPlugin = {
  id: "frameCursor",
  afterDraw(chart) {
    const fx = chart.$frameX;
    if (fx == null) return;
    const x = chart.scales.x.getPixelForValue(fx);
    const { top, bottom } = chart.chartArea;
    const ctx = chart.ctx;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(x, top); ctx.lineTo(x, bottom);
    ctx.lineWidth = 1.5; ctx.strokeStyle = "#ff5555";
    ctx.stroke();
    ctx.restore();
  },
};
Chart.register(cursorPlugin);

const FONT = { size: 9, family: "Consolas,Menlo,Monaco,monospace" };
function baseOpts(title) {
  return {
    animation: false, maintainAspectRatio: false, responsive: true, parsing: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { labels: { color: "#8b949e", boxWidth: 10, font: FONT }, position: "bottom" },
      title: { display: true, text: title, color: "#8b949e", font: FONT },
    },
    scales: {
      x: { type: "linear", ticks: { color: "#484f58", font: FONT }, grid: { color: "#21262d" } },
      y: { ticks: { color: "#484f58", font: FONT }, grid: { color: "#21262d" } },
    },
  };
}

// series: [{label, raw:[y], clamped:[y]}]; spikeSeries: [{x,y}] or null.
function buildChart(canvas, title, series, spikeSeries) {
  const datasets = [];
  series.forEach((s, i) => {
    const color = PALETTE[i % PALETTE.length];
    datasets.push({
      label: s.label, borderColor: color, borderWidth: 1.5, pointRadius: 0, tension: 0,
      data: s.clamped.map((y, x) => ({ x, y })),
    });
    datasets.push({
      label: `${s.label} (raw)`, borderColor: color, borderWidth: 1, pointRadius: 0,
      borderDash: [4, 3], tension: 0,
      data: s.raw.map((y, x) => ({ x, y })),
    });
  });
  if (spikeSeries) {
    datasets.push({
      label: "⚠ spike", type: "scatter", showLine: false,
      pointRadius: 4, pointStyle: "triangle",
      backgroundColor: "#ff5555", borderColor: "#ff5555",
      data: spikeSeries,
    });
  }
  return new Chart(canvas, { type: "line", data: { datasets }, options: baseOpts(title) });
}

// Build the two preview charts from a /api/episode_trajectory payload.
// Returns { charts:[Chart], setCursor(frameIdx) }.
export function buildTrajectoryCharts(jointsCanvas, eeCanvas, data) {
  const N = data.n_frames;
  // Joints chart: 14 series, raw vs clamped. Spikes plotted on joint 0's clamped y.
  const jointSeries = JOINT_NAMES_14.map((label, j) => ({
    label,
    clamped: Array.from({ length: N }, (_, t) => data.joints_clamped[t][j]),
    raw: Array.from({ length: N }, (_, t) => data.joints_raw[t][j]),
  }));
  const spikePoints = data.spikes.map((t) => ({ x: t, y: data.joints_clamped[t][0] }));
  const jointsChart = buildChart(jointsCanvas, "Joint angles (rad)", jointSeries, spikePoints);

  // EE position chart: 6 series (L/R × x/y/z). Recorded EE, so clamped == raw.
  const axes = ["x", "y", "z"];
  const eeSeries = [];
  for (const arm of ["left", "right"]) {
    axes.forEach((ax, k) => {
      const vals = Array.from({ length: N }, (_, t) => data.ee_pos[arm][t][k]);
      eeSeries.push({ label: `${arm} ${ax}`, clamped: vals, raw: vals });
    });
  }
  const eeChart = buildChart(eeCanvas, "EE position (m)", eeSeries, null);

  const charts = [jointsChart, eeChart];
  return {
    charts,
    setCursor(frameIdx) {
      for (const c of charts) { c.$frameX = frameIdx; c.update("none"); }
    },
  };
}

// Media-player transport over a fixed-length trajectory.
// onFrame(i) is called for every frame change (drives 3D + chart cursor + readout).
export class Transport {
  constructor({ nFrames, fps, els, onFrame }) {
    this.n = nFrames;
    this.fps = fps || 30;
    this.els = els;            // { play, stop, slider, readout, speed }
    this.onFrame = onFrame;
    this.idx = 0;
    this.playing = false;
    this.timer = null;

    els.slider.min = 0; els.slider.max = Math.max(0, nFrames - 1); els.slider.value = 0;
    els.play.addEventListener("click", () => (this.playing ? this.pause() : this.play()));
    els.stop.addEventListener("click", () => this.stop());
    els.slider.addEventListener("input", (e) => { this.pause(); this.seek(+e.target.value); });
    this.seek(0);
  }

  _fmt(i) {
    const secs = i / this.fps;
    const m = Math.floor(secs / 60), s = Math.floor(secs % 60);
    return `${i + 1} / ${this.n} · ${m}:${String(s).padStart(2, "0")}`;
  }

  seek(i) {
    this.idx = Math.max(0, Math.min(i, this.n - 1));
    this.els.slider.value = this.idx;
    this.els.readout.textContent = this._fmt(this.idx);
    this.onFrame(this.idx);
  }

  play() {
    if (this.playing || this.n === 0) return;
    this.playing = true;
    this.els.play.textContent = "⏸ Pause";
    const step = () => {
      if (!this.playing) return;
      if (this.idx >= this.n - 1) { this.pause(); return; }
      this.seek(this.idx + 1);
      const speed = parseFloat(this.els.speed?.value || "1");
      this.timer = setTimeout(step, 1000 / (this.fps * speed));
    };
    step();
  }

  pause() {
    this.playing = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    this.els.play.textContent = "▶ Play";
  }

  stop() { this.pause(); this.seek(0); }
}
```

- [ ] **Step 2: Commit**

```bash
cd /home/edgeai/openpi-client-test
git add examples/trossen_ai/webapp/static/js/trajectory.js
git commit -m "feat(replay): trajectory charts + media-player transport"
```

---

## Task 8: Replay page markup — preview stack, importmap, spike modal

**Files:**
- Modify: `webapp/static/replay.html`

- [ ] **Step 1: Add the Three.js importmap to `<head>`**

In `webapp/static/replay.html`, immediately after the two vendor chart `<script>`
lines (lines 7-8), add:

```html
  <script type="importmap">
  {
    "imports": {
      "three":               "https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js",
      "three/addons/":       "https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/",
      "three/examples/jsm/": "https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/",
      "urdf-loader":         "/static/vendor/URDFLoader.js"
    }
  }
  </script>
```

- [ ] **Step 2: Replace the "Episode actions" card with the preview stack**

Replace the single line (line 36):

```html
      <section class="card"><h2>Episode actions</h2><div class="charts-grid" id="charts-box"></div></section>
```

with:

```html
      <section class="card">
        <div class="preview-hdr">
          <h2>Episode preview</h2>
          <button class="btn-outline" id="btn-preview">Preview</button>
        </div>
        <div class="spike-banner" id="spike-banner" style="display:none"></div>
        <div class="urdf-wrap"><canvas id="urdf-canvas"></canvas>
          <div class="cam-views" id="cam-views">
            <button class="btn-sm" data-view="iso">Iso</button>
            <button class="btn-sm" data-view="front">Front</button>
            <button class="btn-sm" data-view="top">Top</button>
            <button class="btn-sm" data-view="left">L</button>
            <button class="btn-sm" data-view="right">R</button>
          </div>
        </div>
        <div class="chart-cj-container"><canvas id="chart-joints"></canvas></div>
        <div class="chart-cj-container"><canvas id="chart-ee"></canvas></div>
        <div class="transport" id="transport">
          <button class="btn-sm" id="tp-play">▶ Play</button>
          <button class="btn-sm" id="tp-stop">⏹ Stop</button>
          <input type="range" id="tp-slider" min="0" max="0" value="0" style="flex:1">
          <span id="tp-readout" class="counter">— / —</span>
          <label>Speed
            <select id="tp-speed">
              <option value="0.25">0.25×</option><option value="0.5">0.5×</option>
              <option value="1" selected>1×</option><option value="2">2×</option>
            </select>
          </label>
        </div>
      </section>
```

- [ ] **Step 3: Add the spike gate modal**

Immediately after the filebrowser modal `</div>` (line 50), before
`<script type="module" ...>`, add:

```html
  <div class="modal-overlay" id="modal-spike" style="display:none">
    <div class="modal-box">
      <div class="modal-titlebar" style="background:#5a1111">⚠ Velocity spikes detected</div>
      <div style="padding:16px">
        <p id="spike-msg" style="color:#ffb4b4"></p>
        <p>Replaying this episode may command an aggressive movement and damage the robot.</p>
        <label style="display:flex;gap:8px;align-items:center;margin-top:12px">
          <input type="checkbox" id="spike-ack"> I understand the risk, run anyway
        </label>
      </div>
      <div class="modal-footer">
        <button class="btn-outline" id="spike-cancel">Cancel</button>
        <button class="btn-danger" id="spike-confirm" disabled>Replay Episode</button>
      </div>
    </div>
  </div>
```

- [ ] **Step 4: Verify the page still serves**

Run: `cd /home/edgeai/openpi-client-test/examples/trossen_ai && python -m pytest webapp/tests/test_server.py::test_replay_page_served -v`
Expected: PASS (markup is static; route unchanged).

- [ ] **Step 5: Commit**

```bash
cd /home/edgeai/openpi-client-test
git add examples/trossen_ai/webapp/static/replay.html
git commit -m "feat(replay): preview-stack markup, importmap, spike modal"
```

---

## Task 9: Wire it all in `replay.js` + theme styles

**Files:**
- Modify: `webapp/static/js/replay.js` (full rewrite)
- Modify: `webapp/static/theme.css` (append styles)

- [ ] **Step 1: Rewrite `replay.js`**

Replace the entire contents of `webapp/static/js/replay.js`:

```javascript
import { api } from "./api.js";
import { connect, send, onMessage, onOpen } from "./ws.js";
import { setupFileBrowser, openFileBrowser } from "./filebrowser.js";
import { setupLogs } from "./logs.js";
import { setupControls } from "./controls.js";
import { renderFeedback } from "./feedback.js";
import { UrdfView } from "./urdf_view.js";
import { buildTrajectoryCharts, Transport } from "./trajectory.js";

const $ = (id) => document.getElementById(id);

let view = null;          // UrdfView
let traj = null;          // last fetched trajectory payload
let charts = null;        // { charts, setCursor }
let transport = null;     // Transport

document.addEventListener("DOMContentLoaded", () => {
  setupFileBrowser();
  setupLogs();
  setupControls();
  renderFeedback($("feedback-card"));

  try {
    view = new UrdfView($("urdf-canvas"));
    view.load().catch((e) => console.warn("URDF load failed:", e));
  } catch (e) {
    console.warn("3D viewer unavailable:", e);
  }

  $("cam-views").addEventListener("click", (e) => {
    const v = e.target.dataset.view;
    if (v && view) view.snapView(v);
  });

  $("btn-browse-dataset").addEventListener("click", () => openFileBrowser($("dataset_dir")));
  $("dataset_dir").addEventListener("change", loadEpisodes);
  $("btn-preview").addEventListener("click", buildPreview);
  $("btn-replay").addEventListener("click", onReplayClick);

  // Spike modal wiring.
  $("spike-ack").addEventListener("change", (e) => { $("spike-confirm").disabled = !e.target.checked; });
  $("spike-cancel").addEventListener("click", () => ($("modal-spike").style.display = "none"));
  $("spike-confirm").addEventListener("click", () => { $("modal-spike").style.display = "none"; startReplay(); });

  // During a real hardware replay, advance the transport cursor from telemetry.
  onMessage("action", (e) => { if (transport && e.step != null) transport.seek(e.step); });
  onOpen(() => {});
  connect();
});

async function loadEpisodes() {
  const dir = $("dataset_dir").value; if (!dir) return;
  try {
    const info = await api(`/api/episodes?dataset_dir=${encodeURIComponent(dir)}`);
    $("episode-select").innerHTML = Array.from({ length: info.total_episodes }, (_, i) =>
      `<option value="${i}">Episode ${i}</option>`).join("");
  } catch (e) { /* surfaced via logs */ }
}

function trajUrl() {
  const p = new URLSearchParams({
    dataset_dir: $("dataset_dir").value,
    episode_index: $("episode-select").value || "0",
    control_freq: "0",      // 0 -> backend uses episode fps
    max_joint_speed: "3.0",
  });
  return `/api/episode_trajectory?${p}`;
}

async function fetchTrajectory() {
  const data = await api(trajUrl());
  data._dataset = $("dataset_dir").value;
  data._ep = $("episode-select").value;
  traj = data;
  return data;
}

async function buildPreview() {
  $("btn-preview").textContent = "Computing…"; $("btn-preview").disabled = true;
  try {
    const data = await fetchTrajectory();
    renderSpikeBanner(data);
    charts = buildTrajectoryCharts($("chart-joints"), $("chart-ee"), data);
    transport = new Transport({
      nFrames: data.n_frames, fps: data.fps,
      els: { play: $("tp-play"), stop: $("tp-stop"), slider: $("tp-slider"),
             readout: $("tp-readout"), speed: $("tp-speed") },
      onFrame: (i) => {
        charts.setCursor(i);
        if (view) view.setFrameJoints(data.joints_clamped[i]);
      },
    });
  } catch (e) {
    console.error("Preview failed:", e);
  } finally {
    $("btn-preview").textContent = "Preview"; $("btn-preview").disabled = false;
  }
}

function renderSpikeBanner(data) {
  const b = $("spike-banner");
  if (data.spikes && data.spikes.length) {
    b.textContent = `⚠ ${data.spikes.length} velocity spike(s) — replay may damage the robot`;
    b.style.display = "block";
  } else {
    b.style.display = "none";
  }
}

function worstVelocity(data) {
  let worst = 0;
  for (const t of data.spikes) for (const v of data.velocity[t]) if (v > worst) worst = v;
  return worst;
}

// Replay button: ensure spike data exists, gate if spiky, else go.
async function onReplayClick() {
  try {
    const stale = !traj || traj._dataset !== $("dataset_dir").value ||
                  String(traj._ep) !== String($("episode-select").value);
    if (stale) await fetchTrajectory();
  } catch (e) { console.warn("Spike pre-check failed; proceeding to confirm:", e); }

  if (traj && traj.spikes && traj.spikes.length) {
    $("spike-msg").textContent =
      `${traj.spikes.length} frame(s) exceed the ${traj.max_joint_speed} rad/s limit ` +
      `(worst ${worstVelocity(traj).toFixed(1)} rad/s).`;
    $("spike-ack").checked = false; $("spike-confirm").disabled = true;
    $("modal-spike").style.display = "flex";
    return;
  }
  startReplay();
}

function startReplay() {
  const cfg = { dataset_dir: $("dataset_dir").value, episode_index: $("episode-select").value,
                mode: $("mode-select").value };
  if (cfg.mode === "autonomous" && !confirm("Replay will move the REAL robot. Continue?")) return;
  send({ action: "start_replay", config: cfg });
}
```

- [ ] **Step 2: Append styles to `theme.css`**

Append to `webapp/static/theme.css`:

```css
/* Replay preview stack */
.preview-hdr { display:flex; align-items:center; justify-content:space-between; }
.spike-banner { margin:8px 0; padding:8px 12px; border-radius:6px;
  background:#5a1111; color:#ffb4b4; font-weight:600; }
.urdf-wrap { position:relative; height:340px; margin:8px 0;
  border:1px solid #21262d; border-radius:6px; overflow:hidden; }
.urdf-wrap canvas { width:100%; height:100%; display:block; }
.cam-views { position:absolute; top:6px; right:6px; display:flex; gap:4px; }
.transport { display:flex; align-items:center; gap:8px; margin-top:10px; }
.transport .counter { min-width:120px; text-align:center; }
```

- [ ] **Step 3: Manual browser verification**

Start the app (the preview path works off-robot):
```bash
cd /home/edgeai/openpi-client-test/examples/trossen_ai && python -m uvicorn webapp.server:app --port 8000
```
Open `http://localhost:8000/replay` and verify:
- Browse to a LeRobot dataset dir → episode dropdown fills.
- Pick an episode → click **Preview**: the 3D robot renders and both charts draw (clamped solid, raw dashed).
- **Play** animates the 3D robot and moves the red cursor on both charts in sync; **Pause**/**Stop** work; dragging the slider scrubs the robot pose.
- Camera buttons (Iso/Front/Top/L/R) snap the view; mouse orbit/zoom works.
- On an episode with spikes: the red banner shows, triangles mark spike frames on the joints chart, and clicking **Replay Episode** opens the severe modal requiring the checkbox before the confirm.
- On a clean episode: **Replay Episode** skips the modal (still asks the autonomous confirm in autonomous mode).

- [ ] **Step 4: Run the full backend test suite**

Run: `cd /home/edgeai/openpi-client-test/examples/trossen_ai && python -m pytest webapp/tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /home/edgeai/openpi-client-test
git add examples/trossen_ai/webapp/static/js/replay.js examples/trossen_ai/webapp/static/theme.css
git commit -m "feat(replay): wire 3D preview, transport, and spike gate"
```

---

## Self-Review notes (resolved during planning)

- **Spec coverage:** raw-vs-clamped (Tasks 1/2/7), IK seed best-effort (Task 3),
  `/api/episode_trajectory` (Task 3), URDF+mesh routes (Task 4), trimmed viewer
  (Task 6), charts+transport+cursor (Task 7), preview stack + importmap (Task 8),
  Preview button + WS cursor + spike gate (Task 9). All covered.
- **Type consistency:** `build_trajectory` returns keys `joints_raw`,
  `joints_clamped`, `ee_pos.{left,right}`, `velocity`, `spikes`, `n_frames`,
  `fps`, `control_freq`, `max_joint_speed`; consumed verbatim in `trajectory.js`
  and `replay.js`. Viewer method `setFrameJoints(joints14)` matches its callers.
  Transport `els` keys (`play, stop, slider, readout, speed`) match the markup
  ids wired in `replay.js`.
- **Manual-test rationale:** repo has no JS runner; frontend Tasks 6–9 use
  browser verification, backend Tasks 1–4 are TDD with pytest.
- **Open follow-up (not blocking):** if `RobotController(robot)` requires extra
  constructor args in this codebase version, the Task 3 seed block already
  swallows the exception and falls back to the home seed — confirm the signature
  during implementation and simplify if it differs.
```
