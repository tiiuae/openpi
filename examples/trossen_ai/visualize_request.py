#!/usr/bin/env python3
"""
Visualize a captured request msgpack: print its contents and export the images.

Loads a msgpack file produced by capture_request.py, prints every field
(state, prompt, image shapes/dtypes), and writes true-color PNGs — one per
camera plus a side-by-side montage. Standalone: only numpy, msgpack and
opencv-python are required.

The images on the wire are channel-first (3, H, W) uint8 in BGR order (see
CLIENT_SCHEMA.md §2); this script converts them to viewable PNGs. Pass
--assume-rgb if you are inspecting a file that was stored as RGB instead.

Usage:
    python visualize_request.py --input captured_request.msgpack

    # Also open a preview window (requires a display):
    python visualize_request.py --input captured_request.msgpack --show
"""

import argparse
import os

import cv2
import msgpack
import numpy as np


def _unpack_hook(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def load_request(path: str) -> dict:
    with open(path, "rb") as f:
        return msgpack.unpackb(f.read(), object_hook=_unpack_hook)


def to_bgr_hwc(image_chw: np.ndarray, assume_rgb: bool) -> np.ndarray:
    """Convert a wire image (3, H, W) to an (H, W, 3) BGR array for cv2."""
    hwc = np.transpose(image_chw, (1, 2, 0))
    if assume_rgb:
        hwc = hwc[:, :, ::-1]  # RGB -> BGR for cv2
    return np.ascontiguousarray(hwc)


def visualize(input_path: str, output_dir: str, assume_rgb: bool, show: bool) -> None:
    obs = load_request(input_path)

    print(f"File: {input_path} ({os.path.getsize(input_path):,} bytes)")
    print(f"Top-level keys: {list(obs.keys())}")
    print()
    print(f"prompt: {obs['prompt']!r}")
    print()

    state = obs["state"]
    print(f"state: shape={state.shape} dtype={state.dtype}")
    if state.ndim == 1 and state.size == 14:
        labels = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper"]
        for arm, offset in (("left ", 0), ("right", 7)):
            joints = "  ".join(f"{n}={state[offset + i]:+.3f}" for i, n in enumerate(labels))
            print(f"  {arm}: {joints}")
    else:
        print(f"  values: {np.round(state, 4)}")
    print()

    os.makedirs(output_dir, exist_ok=True)
    panels = []
    for cam, chw in obs["images"].items():
        print(f"images[{cam!r}]: shape={chw.shape} dtype={chw.dtype} min={chw.min()} max={chw.max()}")
        bgr = to_bgr_hwc(chw, assume_rgb=assume_rgb)
        out_path = os.path.join(output_dir, f"{cam}.png")
        cv2.imwrite(out_path, bgr)
        print(f"  -> {out_path}")

        labeled = bgr.copy()
        cv2.putText(labeled, cam, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(labeled, cam, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(labeled)

    montage = cv2.hconcat(panels)
    montage_path = os.path.join(output_dir, "montage.png")
    cv2.imwrite(montage_path, montage)
    print(f"\nMontage ({len(panels)} cameras side by side) -> {montage_path}")

    if show:
        cv2.imshow(f"{os.path.basename(input_path)} — {obs['prompt']}", montage)
        print("Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a captured request msgpack")
    parser.add_argument("--input", default="captured_request.msgpack", help="Captured msgpack file")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for exported PNGs (default: viz_<input file stem>/)",
    )
    parser.add_argument(
        "--assume-rgb",
        action="store_true",
        help="Treat stored images as RGB instead of the default wire BGR",
    )
    parser.add_argument("--show", action="store_true", help="Open a preview window (requires a display)")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        output_dir = f"viz_{stem}"

    visualize(input_path=args.input, output_dir=output_dir, assume_rgb=args.assume_rgb, show=args.show)
