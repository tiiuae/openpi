# Replay Episode Preview — 3D + Charts + Safety Gate

**Date:** 2026-06-26
**Branch:** ibrahim/feat/web-app-eval
**Component:** `examples/trossen_ai/webapp` (Replay page)

## Motivation

Testing the full stack on hardware is risky. While replaying a dataset episode,
the arm made an aggressive movement that nearly destroyed the hardware. The
operator had no way to see what an episode would do before committing it to the
real robot.

This feature makes the Replay page a **look-before-run** tool: when an episode
is selected, the operator can preview the entire trajectory — both as a 3D URDF
animation and as value charts — driven by a media-player transport (play / pause
/ stop / scrub). Dangerous velocity spikes are flagged, and launching the
hardware run is gated behind a severe warning when spikes are present.

The robot does **not** move during preview. Preview playback animates a 3D model
and a chart cursor only. Hardware motion remains exclusively behind the existing
**Replay Episode** button.

## Key facts grounding the design

- Episode data is a `(N, 16)` absolute-EE chunk read by
  `dataset_replay.EpisodeReader.read_episode` — pure data, no robot.
- Joint angles are produced by `EEToJointsConverter.decode_chunk(ee_chunk16,
  seed)` — pure inverse kinematics, **no hardware required**. Output is `(N, 14)`
  (7 joints per arm, including gripper), matching the viewer's `STATE_IDX`
  layout (left 0–6, right 7–13).
- The replay runner does not execute raw decoded joints. Each step is passed
  through `robot_control.limit_joint_velocity(last, target, dt, max_joint_speed)`
  (`runners.py:159`), where `dt = 1/control_freq`. This clamp is what actually
  protects the hardware; a raw IK branch-flip spike is flattened (the arm lags
  and migrates toward the true target).
- A complete, off-robot URDF viewer with frame playback already exists in
  `get_inspired_from_web_wizard/viewer/`. Its joint-driving core is directly
  reusable.
- URDF + meshes exist at
  `examples/trossen_ai/external/joint_to_ee/trossen_arm_description/`
  (`urdf/generated/mobile_ai.urdf`, 25 mesh files). The URDF references meshes as
  `package://trossen_arm_description/meshes/...`.

## Two trajectories: raw vs clamped

Because the runner clamps velocity, two trajectories exist per episode:

- **raw** — `decode_chunk` output. The scary IK branch-flip spike lives here.
- **clamped** — what `execute_action` actually receives after
  `limit_joint_velocity`. This is the true hardware motion.

The preview shows **both** so the operator sees what will happen *and* why it is
flagged:

- The **3D robot animates the clamped path** (truthful to hardware).
- The **charts overlay clamped (solid) and raw (dashed)**, with spike markers at
  the frames where the two diverge / where any joint's raw step velocity exceeds
  `max_joint_speed`.

## IK seed

`decode_chunk` needs an initial guess ("seed") for frame 0; later frames seed
from the previous solution. The real run seeds frame 0 from the robot's live
joints (`current_joints14()`, `runners.py:147`). The preview seeds as follows:

- **If a robot is connected and no session is running**, read `current_joints14`
  once (no motion) and seed with it — matches the real run exactly.
- **Otherwise**, fall back to a fixed home pose seed so preview still works fully
  off-robot.

This is best-effort: when the robot is unavailable or busy, the home-seed
preview may pick a different frame-0 IK branch than the eventual run, but it
remains a faithful inspection of the episode shape.

## Architecture

### Backend

**New pure-compute module: `webapp/episode_preview.py`**

```
build_trajectory(dataset_dir, episode_index, control_freq, max_joint_speed,
                 seed_joints=None) -> dict
```

- Loads the episode via `EpisodeReader`.
- Seeds with `seed_joints` if provided, else a module-level home pose constant.
- `raw = decode_chunk(ee_chunk16, seed)`.
- Computes `clamped` by folding `limit_joint_velocity` over `raw` with
  `dt = 1/control_freq` (identical logic to the runner, so the preview matches
  execution exactly).
- Computes per-step raw joint velocity `|raw[t] - raw[t-1]| / dt` and the spike
  list (frames where any joint exceeds `max_joint_speed`).
- Returns:

```jsonc
{
  "fps": 30,
  "control_freq": 30,
  "n_frames": 412,
  "max_joint_speed": 3.0,
  "joints_raw":     [[14], ...],   // decoded angles
  "joints_clamped": [[14], ...],   // post-clamp angles (drives 3D)
  "ee_pos": { "left":  [[x,y,z], ...],
              "right": [[x,y,z], ...] },
  "velocity": [[14], ...],          // raw per-step joint speed
  "spikes":   [stepIdx, ...]
}
```

No robot is touched by `build_trajectory` itself. Pure, unit-testable.

**New route in `server.py`:**

```
GET /api/episode_trajectory?dataset_dir=...&episode_index=N&control_freq=...&max_joint_speed=...
```

- Lazy-imports `episode_preview` (and its IK deps) like the existing
  `/api/episodes` route, so the REST layer still loads off-robot.
- Best-effort seed: if no session is running and a robot can be read cheaply,
  pass `current_joints14` as `seed_joints`; otherwise omit (home seed). The
  query must pass the **same `control_freq` the run will use** (form value, else
  `episode.fps`) so spike detection is truthful.
- Runs as a sync `def` (FastAPI threadpool) so multi-second IK does not block the
  event loop.

**New asset routes in `server.py`** (mirroring `get_inspired_from_web_wizard/server.py`):

- `GET /robot.urdf` → `FileResponse` of
  `external/joint_to_ee/trossen_arm_description/urdf/generated/mobile_ai.urdf`.
- `StaticFiles` mount at `/pkg/trossen_arm_description` →
  `external/joint_to_ee/trossen_arm_description/`, resolving the URDF's
  `package://trossen_arm_description/...` mesh refs.

### Frontend

**Page layout (`static/replay.html`)** — the "Episode actions" card becomes a
preview stack:

1. Spike banner (hidden unless `spikes.length > 0`):
   `⚠ N velocity spikes — replay may damage the robot`.
2. **3D URDF panel** (primary, on top) with camera-view preset buttons.
3. **Two stacked charts** below: Chart A = 14 joint angles, Chart B = EE position
   x/y/z per arm (6 lines). Each draws clamped (solid) + raw (dashed); Chart A
   marks spikes.
4. **Transport bar** (shared): ▶ Play / ⏸ Pause / ⏹ Stop, advancement slider,
   `frame / total · mm:ss` readout, speed selector (0.25× / 0.5× / 1× / 2×).

Three.js + addons load from the jsDelivr **importmap** (mirrors the wizard);
`URDFLoader.js` is vendored to `/static/vendor/`. URDF + meshes are served by the
backend routes above.

**New `static/js/urdf_view.js`** (trimmed from `get_inspired_from_web_wizard/viewer`):

- Reuses: `makeURDFLoader` (`packages = { trossen_arm_description:
  '/pkg/trossen_arm_description' }`, `ColladaLoader` for `.dae`),
  `JOINT_MAP` / `GRIPPER_MAP` / `STATE_IDX`, `applyRobotJoints`, scene / camera /
  `OrbitControls` / render loop, and the camera-view preset buttons.
- Drops: EE analysis, FK validation, trails, calibration, the second panel, the
  dataset browser, and the Style/Graphs/Values panels.
- Exposes `setFrameJoints(joints14)` — applies one frame's clamped joints to the
  URDF.

**New `static/js/trajectory.js`** — charts + transport:

- `buildTrajectoryCharts(data)`: builds Chart A (joints) and Chart B (EE pos)
  with clamped solid + raw dashed datasets and a spike scatter overlay on A.
- Cursor: a small custom Chart.js plugin (`afterDraw`) draws a vertical line at
  the current frame on both charts. No annotation library needed.
- `Transport` controller owns `frameIdx`. `updateFrame(i)` moves both chart
  cursors, calls `urdf_view.setFrameJoints(joints_clamped[i])`, and updates the
  slider + readout. Play steps at `fps × speed` via `setTimeout`; Stop resets to
  0; the slider scrubs/seeks. Pure client-side over static data — no WebSocket.

**`static/js/replay.js` wiring:**

- Add a **Preview** button. On click: fetch `/api/episode_trajectory` (with a
  loading state), then build charts + 3D + transport + spike banner. Replaces the
  current joint-0 `LiveChart` initialization.
- During an actual hardware Replay run, advance the same transport cursor from
  incoming WebSocket `action` events (`e.step`) so the bar mirrors real progress.

### Safety gate (the core ask)

On **Replay Episode** (hardware) click:

1. Ensure spike data exists. If the episode was already previewed, reuse it;
   otherwise fetch `/api/episode_trajectory` first (so the gate is never skipped
   even though previewing is optional).
2. If `spikes.length > 0`, show a **severe blocking modal** (red): spike count,
   worst observed `rad/s` vs the `max_joint_speed` limit, and an explicit
   "I understand, run anyway" checkbox that must be ticked before the existing
   autonomous-mode confirm dialog.
3. If no spikes, proceed with the normal confirm flow.

Backend belt-and-suspenders: the runner's per-step `limit_joint_velocity` clamp
stays (it is the real protection). Add a logged warning when a run launches with
acknowledged spikes. The clamp protects; the modal informs.

## Testing

- **`webapp/tests/test_episode_preview.py`** (no robot): on a small synthetic
  episode, assert raw vs clamped divergence, per-step velocity math, spike
  detection at a known injected jump, and the returned JSON shape.
- **Manual**: serve the app, select an episode, click Preview → 3D robot and both
  charts render; scrub and play stay in sync; spikes flag on the chart and in the
  banner; clicking Replay Episode on a spiky episode shows the severe gate modal.

## Out of scope (YAGNI)

- EE orientation (quaternion) and gripper value charts (joints chart already
  shows gripper joints; 3D shows gripper open/close).
- Trails, FK overlays, joint calibration, the second analysis panel.
- Disk caching of computed trajectories (add only if decode latency proves
  painful).
- Robot-following 3D during a live hardware run (preview 3D is cursor-only;
  during a real run only the chart cursor advances from WS events).
- Local vendoring of Three.js (CDN importmap chosen; robot machine needs internet
  at page load for the 3D view).

## File touch list

- `webapp/episode_preview.py` — new pure-compute module.
- `webapp/server.py` — `/api/episode_trajectory`, `/robot.urdf`, `/pkg/...` mount.
- `webapp/static/replay.html` — preview stack markup, importmap, script tags.
- `webapp/static/js/urdf_view.js` — new, trimmed URDF viewer.
- `webapp/static/js/trajectory.js` — new, charts + transport + cursor plugin.
- `webapp/static/js/replay.js` — Preview button, fetch + wire, run-gate.
- `webapp/static/vendor/URDFLoader.js` — vendored loader.
- `webapp/tests/test_episode_preview.py` — new unit test.
