#!/usr/bin/env python
"""LeRobot v3.0 (right7_2view_v1) -> Xiaomi-Robotics-1 (XR-1) JSON + video exporter.

Repo-independent (numpy / pyarrow / ffmpeg only). Pure function of the source dataset:
the source is opened read-only and never modified.

XR-1 data format (xr1/docs/data_format.md + xr1/mibot/data/datasets/json_dataset.py, commit 0dd7aef):
  * one JSON per episode; proprios/actions are per-frame arrays; videos referenced by path (+ frame offset `start`)
  * the loader packs a (30, 60) RELATIVE action tensor per frame f from
        right_ee_pos  (dims 8-10) = R_f^T (p_t - p_f)          (p = ee_pos, R = ee_rotm of proprio at f, target at t)
        right_ee_aa   (dims 11-13) = log(R_f^T R_t)            (axis-angle)
        right_gripper (dim 14)     = g_t - g_f
        left block (dims 0-6), waist (16) and base velocity (17-19) analogously; the rest is reserved/zero
  * the (1, 60) state is joint-space: [left_joint(7) | left_gripper | right_joint(7) | right_gripper | zeros(44)]

XR-1 has NO joint-space ACTION slot, and the loader derives the action tensor from ee_pos/ee_rotm.  Our canonical
dataset is joint-space (right_joint_0..5 rad + right_joint_6 gripper carriage in metres, absolute targets).  We keep
the benchmark's joint-space action and write pseudo end-effector fields such that the UNMODIFIED loader produces plain
joint DELTAS (a_t - q_f), i.e. a purely linear re-indexing (default, wrist-encoding "additive"):

  our dim              XR-1 JSON field(s) written                             packed action slot   loader output
  right_joint_0..2  -> proprios/actions.right_ee_pos = q[0:3] / a[0:3], rotm=I  8,9,10               a_t[0:3] - q_f[0:3]  (exact)
  right_joint_3..5  -> proprios/actions.left_ee_pos  = q[3:6] / a[3:6], rotm=I  0,1,2                a_t[3:6] - q_f[3:6]  (exact)  <- DEVIATION: left-arm slots
  right_joint_6     -> proprios/actions.right_gripper_pos = q[6] / a[6]         14                   a_t[6] - q_f[6]      (exact)
  state             -> proprios.right_arm_joint = q[0:6], right_gripper_pos = q[6]  -> state dims 8..13 and 15; dim 14 = 0 (7th joint pad)
  everything else   -> zeros / identity rotations -> action dims 3-7, 11-13, 15-59 = 0; state dims 0-7, 14, 16-59 = 0 (q01 = q99 = 0 -> padding)

Why not the right-arm block only (dims 8-14)?  The only right-arm slots are ee_pos (8-10), ee_aa (11-13) and gripper
(14); the loader computes ee_pos deltas as R_f^T (p_t - p_f) and ee_aa as log(R_f^T R_t) with the SAME R_f, so putting
the wrist joints into a rotation vector ("rotvec" option below) makes both the wrist target (log-map composition) and
the shoulder target (rotated by the wrist configuration) nonlinear, state-dependent functions of the joint deltas.  They
are still exactly invertible (|q[3:6]| < pi holds: max 1.11 rad), but they handicap learning for no benefit, so the
linear mapping is the default.  Option --wrist-encoding rotvec keeps the 7 dims inside dims 8-14 with that nonlinearity.

Cameras: 2 of XR-1's 3 views (ego <- observation.images.cam_high, wrist_right <- observation.images.cam_right_wrist);
the prompt lists only these two <image> placeholders (the loader only requires that the referenced keys exist).

Video modes: "reference" writes absolute paths to the source LeRobot AV1 files with `start` = first frame of the
episode inside the concatenated file; "transcode" writes one H.264 (libx264, crf 18, yuv420p) clip per episode/camera.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ACTION_DIM = 60
STATE_DIM = 60
ROTVEC_NORM_LIMIT = 0.9 * np.pi
CAMERA_MAP = {  # XR-1 observation key -> LeRobot video key
    "ego": "observation.images.cam_high",
    "wrist_right": "observation.images.cam_right_wrist",
}
PROMPT_2VIEW = (
    "The following observations are captured from multiple views.\n"
    "# Ego View\n<image>\n# Right-Wrist View\n<image>\n"
    "Generate robot actions for the task:\n{task}"
)
EYE9 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


# ----------------------------------------------------------------------------- rotations (same formulas as mibot.utils.io)
def aa2rotm(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    angle = float(np.linalg.norm(axis_angle))
    axis = axis_angle / (angle + 1e-10)
    x, y, z = axis.tolist()
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


def rotm2aa(rotm: np.ndarray) -> np.ndarray:
    """Principal log map (theta in [0, pi]); matches mibot.utils.io.rotm2aa_batch away from theta = pi."""
    rotm = np.asarray(rotm, dtype=np.float64)
    vec = np.array([rotm[2, 1] - rotm[1, 2], rotm[0, 2] - rotm[2, 0], rotm[1, 0] - rotm[0, 1]])  # = 2 sin(theta) * axis
    s = np.linalg.norm(vec)
    theta = np.arctan2(s, np.trace(rotm) - 1.0)  # atan2(2 sin, 2 cos): well conditioned near 0 (the loader's arccos is not)
    if theta <= 1e-9:
        return np.zeros(3)
    return vec / s * theta


# ----------------------------------------------------------------------------- source reading
def read_source(src: Path):
    info = json.loads((src / "meta" / "info.json").read_text())
    fps = int(info["fps"])
    tasks_tbl = pq.read_table(src / "meta" / "tasks.parquet").to_pandas()
    # LeRobot v3: tasks.parquet is indexed by the task string with a task_index column
    if "task_index" in tasks_tbl.columns:
        task_by_index = {int(r.task_index): str(idx) for idx, r in tasks_tbl.iterrows()}
    else:
        task_by_index = {i: str(t) for i, t in enumerate(tasks_tbl.index)}
    ep_dir = src / "meta" / "episodes"
    ep_frames = sorted(ep_dir.rglob("*.parquet"))
    episodes = pq.read_table(ep_frames[0]).to_pandas() if len(ep_frames) == 1 else __import__("pandas").concat(
        [pq.read_table(p).to_pandas() for p in ep_frames], ignore_index=True
    )
    data_files = sorted((src / "data").rglob("*.parquet"))
    cols = ["episode_index", "frame_index", "timestamp", "task_index", "observation.state", "action"]
    tbls = [pq.read_table(p, columns=cols).to_pandas() for p in data_files]
    data = tbls[0] if len(tbls) == 1 else __import__("pandas").concat(tbls, ignore_index=True)
    data = data.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    return info, fps, task_by_index, episodes, data


def video_source(src: Path, info: dict, ep_row, lerobot_key: str, fps: int):
    chunk = int(ep_row[f"videos/{lerobot_key}/chunk_index"])
    file_index = int(ep_row[f"videos/{lerobot_key}/file_index"])
    from_ts = float(ep_row[f"videos/{lerobot_key}/from_timestamp"])
    to_ts = float(ep_row[f"videos/{lerobot_key}/to_timestamp"])
    rel = info["video_path"].format(video_key=lerobot_key, chunk_index=chunk, file_index=file_index)
    start = int(round(from_ts * fps))
    n = int(round((to_ts - from_ts) * fps))
    return (src / rel).resolve(), start, n, from_ts


# ----------------------------------------------------------------------------- per-episode records
def episode_record(
    ep_index: int, task: str, state: np.ndarray, action: np.ndarray, videos: dict, wrist_encoding: str, decimals: int
) -> dict:
    n = len(state)
    assert state.shape == (n, 7) and action.shape == (n, 7)
    r = lambda arr: np.round(np.asarray(arr, dtype=np.float64), decimals).tolist()  # noqa: E731
    zeros1 = [[0.0]] * n
    zeros3 = [[0.0, 0.0, 0.0]] * n
    eye = [EYE9] * n

    if wrist_encoding == "rotvec":
        right_ee_pos_p, right_ee_pos_a = r(state[:, 0:3]), r(action[:, 0:3])
        right_rotm_p = r(np.stack([aa2rotm(v).reshape(9) for v in state[:, 3:6]]))
        right_rotm_a = r(np.stack([aa2rotm(v).reshape(9) for v in action[:, 3:6]]))
        left_ee_pos_p, left_ee_pos_a = zeros3, zeros3
    elif wrist_encoding == "additive":
        right_ee_pos_p, right_ee_pos_a = r(state[:, 0:3]), r(action[:, 0:3])
        right_rotm_p = right_rotm_a = eye
        left_ee_pos_p, left_ee_pos_a = r(state[:, 3:6]), r(action[:, 3:6])
    else:
        raise ValueError(wrist_encoding)

    obs = {}
    for xr1_key, v in videos.items():
        obs[xr1_key] = [{"path": str(v["path"]), "start": int(v["start"]), "crop_bbox": None}]

    return {
        "trajectory_type": "success",
        "time": f"episode_{ep_index:06d}",
        "num_frames": n,
        "instruction": {
            "general": [
                {
                    "images": [f"observations.{k}" for k in videos.keys()],
                    "conversations": [
                        {"from": "human", "value": PROMPT_2VIEW.format(task=task)},
                        {"from": "gpt", "value": ""},
                    ],
                }
            ]
        },
        "observations": obs,
        "proprios": {
            "left_ee_pos": left_ee_pos_p,
            "left_ee_rotm": eye,
            "left_arm_joint": [[0.0] * 6] * n,
            "left_gripper_pos": zeros1,
            "right_ee_pos": right_ee_pos_p,
            "right_ee_rotm": right_rotm_p,
            "right_arm_joint": r(state[:, 0:6]),
            "right_gripper_pos": r(state[:, 6:7]),
            "waist_pos": zeros1,
        },
        "actions": {
            "left_ee_pos": left_ee_pos_a,
            "left_ee_rotm": eye,
            "left_gripper_pos": zeros1,
            "right_ee_pos": right_ee_pos_a,
            "right_ee_rotm": right_rotm_a,
            "right_gripper_pos": r(action[:, 6:7]),
            "waist_pos": zeros1,
            "base_vel": zeros3,
        },
        "_source": {"episode_index": ep_index, "task": task, "wrist_encoding": wrist_encoding},
    }


# ----------------------------------------------------------------------------- packed tensors (replicates the loader / compute_normalize.py)
def packed_state(rec: dict, f: int) -> np.ndarray:
    s = np.zeros(STATE_DIM, dtype=np.float64)
    lj = np.asarray(rec["proprios"]["left_arm_joint"][f], dtype=np.float64)
    rj = np.asarray(rec["proprios"]["right_arm_joint"][f], dtype=np.float64)
    s[: len(lj)] = lj
    s[7] = rec["proprios"]["left_gripper_pos"][f][0]
    s[8 : 8 + len(rj)] = rj
    s[15] = rec["proprios"]["right_gripper_pos"][f][0]
    return s


def packed_action_window(rec: dict, f: int, length: int) -> np.ndarray:
    """(length, 60) relative action exactly as json_dataset.py / compute_normalize.py build it (edge-padded)."""
    n = int(rec["num_frames"])
    steps = min(length, n - f)
    P, A = rec["proprios"], rec["actions"]
    out = np.zeros((length, ACTION_DIM), dtype=np.float64)

    def pad(v):
        v = np.asarray(v, dtype=np.float64)
        return v if steps == length else np.concatenate([v, np.repeat(v[-1:], length - steps, axis=0)], axis=0)

    for arm, (sp, sa, sg) in (("left", (slice(0, 3), slice(3, 6), slice(6, 7))), ("right", (slice(8, 11), slice(11, 14), slice(14, 15)))):
        R = np.asarray(P[f"{arm}_ee_rotm"][f], dtype=np.float64).reshape(3, 3)
        p = np.asarray(P[f"{arm}_ee_pos"][f], dtype=np.float64)
        tp = np.asarray(A[f"{arm}_ee_pos"][f : f + steps], dtype=np.float64)
        tR = np.asarray(A[f"{arm}_ee_rotm"][f : f + steps], dtype=np.float64).reshape(-1, 3, 3)
        out[:, sp] = pad((R.T @ (tp - p).T).T)
        out[:, sa] = pad(np.stack([rotm2aa(R.T @ m) for m in tR]))
        g = np.asarray(A[f"{arm}_gripper_pos"][f : f + steps], dtype=np.float64) - np.asarray(P[f"{arm}_gripper_pos"][f], dtype=np.float64)
        out[:, sg] = pad(g)
    out[:, 16:17] = pad(np.asarray(A["waist_pos"][f : f + steps], dtype=np.float64) - np.asarray(P["waist_pos"][f], dtype=np.float64))
    out[:, 17:20] = pad(np.asarray(A["base_vel"][f : f + steps], dtype=np.float64))
    return out


def unpack_action(packed: np.ndarray, state7: np.ndarray, wrist_encoding: str) -> np.ndarray:
    """Inverse of the encoding: (T, 60) relative packed action + current 7-D joint state -> (T, 7) absolute joint targets."""
    packed = np.asarray(packed, dtype=np.float64)
    out = np.zeros((packed.shape[0], 7))
    out[:, 6] = state7[6] + packed[:, 14]
    if wrist_encoding == "rotvec":
        Rf = aa2rotm(state7[3:6])
        out[:, 0:3] = state7[0:3] + packed[:, 8:11] @ Rf.T  # loader stored R_f^T (a - q) -> undo (as recover_action does)
        out[:, 3:6] = np.stack([rotm2aa(Rf @ aa2rotm(d)) for d in packed[:, 11:14]])
    else:
        out[:, 0:3] = state7[0:3] + packed[:, 8:11]
        out[:, 3:6] = state7[3:6] + packed[:, 0:3]
    return out


# ----------------------------------------------------------------------------- video
def transcode_clip(src_mp4: Path, from_ts: float, n_frames: int, fps: int, dst_mp4: Path, crf: int = 18) -> None:
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_mp4.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(from_ts - 0.25 / fps, 0.0):.6f}", "-i", str(src_mp4),  # quarter-frame early -> first output frame = episode frame 0
        "-frames:v", str(n_frames), "-fps_mode", "passthrough",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-g", "15", "-pix_fmt", "yuv420p", "-an",
        "-movflags", "+faststart", str(tmp),
    ]
    subprocess.run(cmd, check=True)
    os.replace(tmp, dst_mp4)  # tmp is ours; never overwrites a pre-existing path except our own previous output


def ffprobe_nb_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
         "-of", "default=nw=1:nk=1", str(path)], check=True, capture_output=True, text=True
    ).stdout.strip()
    return int(out)


def ffmpeg_frame_rgb(path: Path, index: int, fps: int, width: int, height: int) -> np.ndarray:
    """Decode exactly frame `index` of `path` with the ffmpeg CLI (accurate input seek), as HxWx3 uint8 RGB."""
    ts = (index - 0.25) / fps  # quarter-frame early: the first frame with pts >= ts is exactly frame `index`
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{max(ts, 0.0):.6f}", "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True
    ).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(height, width, 3)


# ----------------------------------------------------------------------------- main export
def export(args):
    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    assert src.exists(), src
    info, fps, task_by_index, episodes, data = read_source(src)
    L = args.action_length
    ep_ids = sorted(int(e) for e in episodes["episode_index"].tolist())
    if args.episodes is not None:
        ep_ids = [e for e in ep_ids if e in set(args.episodes)]
    print(f"[export] source {src}: {len(ep_ids)} episodes, fps {fps}, tasks {task_by_index}", flush=True)

    S_all = np.stack(data["observation.state"].values).astype(np.float64)
    A_all = np.stack(data["action"].values).astype(np.float64)
    wrist_norm = max(np.linalg.norm(S_all[:, 3:6], axis=1).max(), np.linalg.norm(A_all[:, 3:6], axis=1).max())
    wrist_encoding = args.wrist_encoding
    print(f"[export] max |q[3:6]| = {wrist_norm:.4f} rad (rotvec invertibility limit {ROTVEC_NORM_LIMIT:.4f}); wrist_encoding = {wrist_encoding}", flush=True)
    if wrist_encoding == "rotvec" and wrist_norm >= np.pi:
        raise SystemExit("rotvec encoding is not invertible for |q[3:6]| >= pi; use --wrist-encoding additive")

    (dst / "json").mkdir(parents=True, exist_ok=True)
    if args.video_mode == "transcode":
        (dst / "videos").mkdir(parents=True, exist_ok=True)

    # statistics accumulators (compute_normalize.py semantics: complete L-step windows for actions, all frames for states)
    a_sum = np.zeros((L, ACTION_DIM)); a_sq = np.zeros((L, ACTION_DIM)); a_cnt = 0
    states = []
    manifest = {"episodes": []}
    transcode_jobs = []

    ep_by_id = {int(r["episode_index"]): r for _, r in episodes.iterrows()}
    for ep in ep_ids:
        row = ep_by_id[ep]
        d = data[data["episode_index"] == ep]
        assert (d["frame_index"].values == np.arange(len(d))).all(), f"episode {ep}: non-contiguous frame_index"
        state = np.stack(d["observation.state"].values).astype(np.float64)
        action = np.stack(d["action"].values).astype(np.float64)
        task_idx = int(d["task_index"].iloc[0])
        assert (d["task_index"].values == task_idx).all(), f"episode {ep}: mixed task_index"
        task = task_by_index[task_idx]
        n = len(d)

        videos = {}
        for xr1_key, lerobot_key in CAMERA_MAP.items():
            src_mp4, start, n_vid, from_ts = video_source(src, info, row, lerobot_key, fps)
            assert n_vid == n, f"episode {ep} {lerobot_key}: video span {n_vid} != parquet {n}"
            if args.video_mode == "reference":
                videos[xr1_key] = {"path": src_mp4, "start": start, "source": str(src_mp4), "source_start": start}
            else:
                clip = dst / "videos" / f"episode_{ep:06d}_{xr1_key}.mp4"
                videos[xr1_key] = {"path": clip, "start": 0, "source": str(src_mp4), "source_start": start}
                transcode_jobs.append((src_mp4, from_ts, n, fps, clip))

        rec = episode_record(ep, task, state, action, videos, wrist_encoding, args.decimals)
        out_json = dst / "json" / f"episode_{ep:06d}.json"
        out_json.write_text(json.dumps(rec, separators=(",", ":")))

        # statistics from the JSON record (i.e. from what the loader will actually see, incl. rounding)
        rec = json.loads(out_json.read_text())
        for f in range(n):
            states.append(packed_state(rec, f))
        for f in range(n - L + 1):
            w = packed_action_window(rec, f, L)
            a_sum += w; a_sq += w * w; a_cnt += 1
        manifest["episodes"].append({
            "episode_index": ep, "json": str(out_json.relative_to(dst)), "num_frames": n, "task_index": task_idx, "task": task,
            "videos": {k: {"path": str(v["path"]), "start": v["start"], "source": v["source"], "source_start": v["source_start"]} for k, v in videos.items()},
        })
        if ep % 20 == 0:
            print(f"[export] episode {ep} ({n} frames) written", flush=True)

    if transcode_jobs:
        print(f"[export] transcoding {len(transcode_jobs)} clips with {args.workers} workers ...", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda j: transcode_clip(*j), transcode_jobs))
        print(f"[export] transcode done in {time.time() - t0:.0f}s", flush=True)
        bad = []
        for _, _, n, _, clip in transcode_jobs:
            got = ffprobe_nb_frames(clip)
            if got != n:
                bad.append((str(clip), n, got))
        if bad:
            raise SystemExit(f"frame-count mismatch after transcode: {bad[:5]}")
        print("[export] all clips have the expected frame count", flush=True)

    mean = a_sum / a_cnt
    std = np.sqrt(np.maximum(a_sq / a_cnt - mean * mean, 0.0))
    states = np.asarray(states)[:, None, :]  # (frames, 1, 60)
    q01 = np.quantile(states, 0.01, axis=0)
    q99 = np.quantile(states, 0.99, axis=0)
    norm = {"mean": mean.tolist(), "std": std.tolist(), "q01": q01.tolist(), "q99": q99.tolist(),
            "_meta": {"action_length": L, "action_windows": int(a_cnt), "state_frames": int(states.shape[0]), "wrist_encoding": wrist_encoding}}
    (dst / "normalize.json").write_text(json.dumps(norm, indent=1))

    manifest.update({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": str(src),
        "source_info_sha256": hashlib.sha256((src / "meta" / "info.json").read_bytes()).hexdigest(),
        "source_data_sha256": {str(p.relative_to(src)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((src / "data").rglob("*.parquet"))},
        "fps": fps, "action_length": L, "wrist_encoding": wrist_encoding, "max_wrist_norm_rad": float(wrist_norm),
        "video_mode": args.video_mode, "cameras": CAMERA_MAP, "prompt_template": PROMPT_2VIEW, "decimals": args.decimals,
        "tasks": task_by_index, "num_episodes": len(ep_ids), "num_frames": int(states.shape[0]),
        "exporter": str(Path(__file__).resolve()), "argv": sys.argv,
    })
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[export] wrote {len(ep_ids)} episodes, {states.shape[0]} frames, normalize.json ({a_cnt} action windows) -> {dst}", flush=True)


# ----------------------------------------------------------------------------- verification
def verify(args):
    """Re-derive 5 random (episode, frame) samples from the SOURCE parquet/videos and compare with the export."""
    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    info, fps, task_by_index, episodes, data = read_source(src)
    manifest = json.loads((dst / "manifest.json").read_text())
    norm = json.loads((dst / "normalize.json").read_text())
    enc = manifest["wrist_encoding"]
    L = manifest["action_length"]
    rng = random.Random(args.seed)
    eps = [e["episode_index"] for e in manifest["episodes"]]
    H, W = info["features"][CAMERA_MAP["ego"]]["shape"][:2]
    try:
        from decord import VideoReader  # what the XR-1 loader uses
    except Exception as e:  # pragma: no cover
        VideoReader = None
        print(f"[verify] decord unavailable ({e}); video check uses ffmpeg only", flush=True)

    ok = True
    report = []
    for k in range(args.samples):
        ep = rng.choice(eps)
        rec = json.loads((dst / "json" / f"episode_{ep:06d}.json").read_text())
        d = data[data["episode_index"] == ep]
        n = len(d)
        f = rng.randrange(n)
        s7 = np.asarray(d["observation.state"].iloc[f], dtype=np.float64)
        a7 = np.stack(d["action"].values[f : min(n, f + L)]).astype(np.float64)
        # 1) state
        st = packed_state(rec, f)
        e_state = max(abs(st[8:14] - s7[0:6]).max(), abs(st[15] - s7[6]), abs(st[[0, 1, 2, 3, 4, 5, 6, 7, 14]]).max(), abs(st[16:]).max())
        # 2) action window -> unpack -> compare to the source absolute joint targets
        w = packed_action_window(rec, f, L)
        steps = len(a7)
        e_act = abs(unpack_action(w[:steps], s7, enc) - a7).max()
        e_zero = abs(np.delete(w, list(range(8, 15)) + ([0, 1, 2] if enc == "additive" else []), axis=1)).max()
        # 3) task string
        task_ok = rec["instruction"]["general"][0]["conversations"][0]["value"].endswith(task_by_index[int(d["task_index"].iloc[0])])
        # 4) video frame: XR-1 path (decord, path+start+f) vs ffmpeg decode of the SOURCE file at source_start+f
        vid = {}
        for xr1_key, lerobot_key in CAMERA_MAP.items():
            entry = rec["observations"][xr1_key][0]
            m = [e for e in manifest["episodes"] if e["episode_index"] == ep][0]["videos"][xr1_key]
            ref = ffmpeg_frame_rgb(Path(m["source"]), m["source_start"] + f, fps, W, H).astype(np.int16)
            if VideoReader is not None:
                vr = VideoReader(entry["path"], num_threads=2)
                got = vr.get_batch([f + int(entry.get("start", 0))]).asnumpy()[0].astype(np.int16)
            else:
                got = ffmpeg_frame_rgb(Path(entry["path"]), f + int(entry.get("start", 0)), fps, W, H).astype(np.int16)
            mad = float(abs(got - ref).mean())
            mse = float(((got - ref) ** 2).mean())
            psnr = 99.0 if mse == 0 else float(10 * np.log10(255.0**2 / mse))
            vid[xr1_key] = {"mad": round(mad, 3), "psnr_db": round(psnr, 2)}
        vid_ok = all(v["psnr_db"] >= args.min_psnr for v in vid.values())
        line = {"episode": ep, "frame": f, "steps": steps, "state_err": float(e_state), "action_err": float(e_act),
                "offblock_max": float(e_zero), "task_ok": bool(task_ok), "video": vid}
        good = e_state < args.tol and e_act < args.tol and e_zero == 0.0 and task_ok and vid_ok
        ok &= good
        report.append({**line, "ok": bool(good)})
        print(f"[verify] {'OK ' if good else 'BAD'} {line}", flush=True)

    # 5) normalisation statistics sanity
    mean, std = np.asarray(norm["mean"]), np.asarray(norm["std"])
    q01, q99 = np.asarray(norm["q01"]), np.asarray(norm["q99"])
    stats_ok = mean.shape == (L, 60) and std.shape == (L, 60) and q01.shape == (1, 60) and q99.shape == (1, 60)
    active = [8, 9, 10, 11, 12, 13, 14] if enc == "rotvec" else [0, 1, 2, 8, 9, 10, 14]
    stats_ok &= bool((std[:, active] > 0).all()) and bool((std[:, [d for d in range(60) if d not in active]] == 0).all())
    stats_ok &= bool((q99[0, [8, 9, 10, 11, 12, 13, 15]] > q01[0, [8, 9, 10, 11, 12, 13, 15]]).all())
    stats_ok &= bool((q01[0, [d for d in range(60) if d not in (8, 9, 10, 11, 12, 13, 15)]] == 0).all())
    print(f"[verify] normalize.json shapes/padding {'OK' if stats_ok else 'BAD'}; std(step 0, active) = {np.round(std[0, active], 5).tolist()}", flush=True)
    ok &= stats_ok
    out = {"ok": bool(ok), "samples": report, "stats_ok": bool(stats_ok), "wrist_encoding": enc, "seed": args.seed}
    (dst / "verification.json").write_text(json.dumps(out, indent=1))
    print(f"[verify] {'PASS' if ok else 'FAIL'} -> {dst / 'verification.json'}", flush=True)
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--src", required=True)
    e.add_argument("--dst", required=True)
    e.add_argument("--action-length", type=int, default=30)
    e.add_argument("--video-mode", choices=["reference", "transcode"], default="transcode")
    e.add_argument("--wrist-encoding", choices=["additive", "rotvec"], default="additive")
    e.add_argument("--decimals", type=int, default=7)
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--episodes", type=int, nargs="*", default=None, help="subset of episode indices (testing)")
    v = sub.add_parser("verify")
    v.add_argument("--src", required=True)
    v.add_argument("--dst", required=True)
    v.add_argument("--samples", type=int, default=5)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--tol", type=float, default=2e-6)
    v.add_argument("--min-psnr", type=float, default=35.0)
    args = p.parse_args()
    if args.cmd == "export":
        export(args)
    else:
        sys.exit(verify(args))


if __name__ == "__main__":
    main()
