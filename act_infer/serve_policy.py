"""Serve a converted ACT HF snapshot to the openpi robot client over websocket.

    python -m inference.serve_policy \
        --hf_ckpt /lustre1/.../act_aloha_14_25_kl10/hf_export_best \
        --host 0.0.0.0 --port 8000 \
        --camera_map primary=cam_high,secondary=cam_low,wrist=cam_right_wrist

Then on the robot, run the stock openpi example pointed at this server:

    python examples/trossen_ai/main.py --policy_host <SERVER_IP> --policy_port 8000 \
        --mode test --task_prompt "pick the cup and place in the basket"

The `--task_prompt` is accepted but ignored: ACT is a visuomotor policy with no
language conditioning (the cup task is disambiguated by the camera view, not text).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from act_policy import ActPolicy  # noqa: E402
from server import WebsocketPolicyServer  # noqa: E402


def load_model(hf_ckpt_dir: str, device: str):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(hf_ckpt_dir, trust_remote_code=True)
    ns_path = os.path.join(hf_ckpt_dir, "norm_stats.json")
    if not os.path.exists(ns_path):
        raise FileNotFoundError(
            f"norm_stats.json not found in {hf_ckpt_dir} — re-run the converter; "
            "without it predictions stay in normalized space."
        )
    with open(ns_path) as f:
        model.load_norm_stats(json.load(f))
    return model.to(device).eval()


def parse_camera_map(s: str) -> dict:
    out = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, _, v = pair.partition("=")
        out[k.strip()] = v.strip()
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="ACT websocket policy server (openpi-compatible)")
    ap.add_argument("--hf_ckpt", required=True, help="converted HF snapshot dir (has norm_stats.json)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--camera_map",
        default="primary=cam_high,secondary=cam_low,wrist=cam_right_wrist",
        help="model_cam=robot_cam pairs. MUST match how the training data cameras "
            "were assigned. Default assumes primary=overhead, secondary=low, wrist=right.",
    )
    ap.add_argument("--image_height", type=int, default=480)
    ap.add_argument("--image_width", type=int, default=640)
    ap.add_argument(
        "--robot_action_dim",
        type=int,
        default=14,
        help="Action dims the robot client uses (default 14). Model outputs are trimmed to this.",
    )
    ap.add_argument(
        "--static_state_threshold",
        type=float,
        default=0.03,
        help="State dims with training qpos_std below this are pinned to the training mean "
        "(single-arm checkpoints: pins the static arm + near-constant extras). Set large "
        "(e.g. 999) to disable pinning.",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.hf_ckpt, device)
    camera_names = list(model.config.camera_names)
    state_dim = int(model.config.state_dim)
    action_dim = int(model.config.action_dim)
    chunk = int(model.config.chunk_size)
    logging.info(
        f"loaded ACT: cameras={camera_names} state_dim={state_dim} action_dim={action_dim} "
        f"chunk={chunk} robot_action_dim={args.robot_action_dim} device={device}"
    )

    camera_map = parse_camera_map(args.camera_map)
    policy = ActPolicy(
        model=model,
        camera_names=camera_names,
        camera_map=camera_map,
        image_height=args.image_height,
        image_width=args.image_width,
        robot_action_dim=args.robot_action_dim,
        static_state_threshold=args.static_state_threshold,
    )
    logging.info(f"camera_map (model->robot): {camera_map}")
    static_dims = [i for i, m in enumerate(policy._static_mask) if m]
    active_dims = [i for i, m in enumerate(policy._static_mask) if not m]
    logging.info(
        "state pinning (threshold=%.4f): static/padded dims → training mean: %s",
        args.static_state_threshold,
        static_dims if static_dims else "none",
    )
    logging.info("state pinning: active dims (live proprio): %s", active_dims)
    if args.robot_action_dim < state_dim and not static_dims:
        logging.warning(
            "robot sends %d-dim state but checkpoint state_dim=%d and NO dims are pinned — "
            "extra state dims will be filled with the training mean.",
            args.robot_action_dim,
            state_dim,
        )

    metadata = {
        "policy": "act",
        "state_dim": state_dim,
        "action_dim": args.robot_action_dim,
        "model_action_dim": action_dim,
        "chunk_size": chunk,
        "camera_names": camera_names,
    }
    server = WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata)
    logging.info(f"serving on ws://{args.host}:{args.port} — Ctrl-C to stop")
    server.serve_forever()


if __name__ == "__main__":
    main()
