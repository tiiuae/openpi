"""Save, per inference call, the images a served policy receives.

Enabled with ``serve_policy.py --debug_dir <dir>``. Each call gets its own
``<dir>/run_<time>/call_NNNNN/`` directory:

``wire_<camera>.png``
    The images exactly as the robot client sent them, before anything on the
    server touches them. Written for every policy type. The array is saved as
    RGB, so if the client sends RGB (the current Trossen client does) this
    shows true colours: the wood table brown, not blue. A blue table here means
    the client is sending BGR.

``model_input_<key>.png``
    For openpi pi0/pi05 policies only: the images after openpi's input
    transforms (repack, AlohaInputs, ResizeImages), which is what the model's
    Observation is built from. They should be 224x224 with the 640x480 frame
    letterboxed to 224x168 and 28 black rows above and below, in true colour.
    Other policy types have no such stage in openpi: FalconVLA passes the wire
    images straight to its Hugging Face processor, and the REST policies
    (CogACT, OpenVLA, OpenVLA-OFT) convert them in their own client code before
    posting to an external server.

``meta.json``
    Per image: shape, dtype, value range and per-channel means, plus the prompt
    and the policy class. With the current rig a wood-table scene has a red mean
    clearly above the blue mean when the channels are in RGB order.

Disk writes run on a background thread, so the request path does not wait on
PNG encoding. Nothing is written without ``--debug_dir``.
"""

from __future__ import annotations

from collections.abc import Callable
import datetime
import json
import logging
import pathlib
import queue
import threading
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy
from PIL import Image

logger = logging.getLogger(__name__)


def _to_hwc_uint8(image: Any) -> np.ndarray | None:
    """Best-effort conversion of a CHW/HWC uint8 or float image for viewing."""
    array = np.asarray(image)
    if array.ndim != 3:
        return None
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))  # CHW -> HWC
    if array.shape[-1] not in (1, 3):
        return None
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if array.min() < 0.0:  # [-1, 1]
            array = (array + 1.0) * 127.5
        elif array.max() <= 1.0:  # [0, 1]
            array = array * 255.0
        array = np.clip(np.round(array), 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return np.ascontiguousarray(array)


def _describe(image: Any) -> dict:
    array = np.asarray(image)
    info = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if array.size:
        info["min"] = float(array.min())
        info["max"] = float(array.max())
    hwc = _to_hwc_uint8(array)
    if hwc is not None and hwc.shape[-1] == 3:
        means = hwc.reshape(-1, 3).mean(axis=0)
        info["channel_mean_as_rgb"] = {"r": round(float(means[0]), 1), "g": round(float(means[1]), 1), "b": round(float(means[2]), 1)}
    return info


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name).rsplit(".", 1)[-1]) or "image"


class DebugCapturePolicy(_base_policy.BasePolicy):
    """Wraps a policy and saves what it receives on every call."""

    def __init__(self, policy: _base_policy.BasePolicy, debug_dir: str) -> None:
        self._policy = policy
        run_name = datetime.datetime.now(datetime.UTC).strftime("run_%Y%m%dT%H%M%S")
        self._root = pathlib.Path(debug_dir).expanduser() / run_name
        self._root.mkdir(parents=True, exist_ok=False)
        self._call_idx = 0
        self._call_dir: pathlib.Path | None = None
        self._meta: dict = {}

        self._queue: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._writer = threading.Thread(target=self._write_loop, daemon=True, name="debug-capture-writer")
        self._writer.start()

        self._captures_model_input = self._install_model_input_hook()
        logger.info(
            "Debug capture: saving wire_*.png%s and meta.json per call under %s",
            " and model_input_*.png" if self._captures_model_input else "",
            self._root,
        )

    @property
    def metadata(self) -> dict:
        return getattr(self._policy, "metadata", {})

    def infer(self, obs: dict, **kwargs) -> dict:
        call_dir = self._root / f"call_{self._call_idx:05d}"
        call_dir.mkdir()
        self._call_idx += 1
        self._meta = {
            "policy": type(self._policy).__name__,
            "prompt": obs.get("prompt"),
            "wire_images": {},
            "model_input_images": {},
        }

        images = obs.get("images")
        if isinstance(images, dict):
            for name, image in images.items():
                self._save(call_dir / f"wire_{_safe_name(name)}.png", image)
                self._meta["wire_images"][name] = _describe(image)

        self._call_dir = call_dir
        try:
            return self._policy.infer(obs, **kwargs)
        finally:
            self._call_dir = None
            meta_text = json.dumps(self._meta, indent=2, default=str)
            self._queue.put(lambda p=call_dir / "meta.json", t=meta_text: p.write_text(t))

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    # -- internals ---------------------------------------------------------

    def _install_model_input_hook(self) -> bool:
        """For openpi's Policy: capture the output of its input transforms."""
        transform = getattr(self._policy, "_input_transform", None)
        if transform is None:
            return False

        def capture_transformed(data):
            out = transform(data)
            call_dir = self._call_dir
            model_images = out.get("image") if isinstance(out, dict) else None
            if call_dir is not None and isinstance(model_images, dict):
                for key, image in model_images.items():
                    self._save(call_dir / f"model_input_{_safe_name(key)}.png", image)
                    self._meta["model_input_images"][key] = _describe(image)
            return out

        self._policy._input_transform = capture_transformed  # noqa: SLF001
        return True

    def _save(self, path: pathlib.Path, image: Any) -> None:
        hwc = _to_hwc_uint8(image)
        if hwc is None:
            logger.warning("Debug capture: cannot save %s with shape %s", path.name, np.shape(image))
            return
        self._queue.put(lambda p=path, a=hwc: Image.fromarray(a).save(p))

    def _write_loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                job()
            except Exception:
                logger.exception("Debug capture: write failed")
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        """Block until every queued write has finished (used by tests)."""
        self._queue.join()
