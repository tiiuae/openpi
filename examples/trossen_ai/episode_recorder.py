"""Interactive LeRobot-format episode recording for the eval client.

Records (observation, executed action) pairs from the control loop into a
LeRobot v3.0 dataset (the format written by the lerobot 0.4.x in this venv).
The operator drives the lifecycle from the terminal:

    record            -> start()   begin buffering frames
    hold / stop       -> end()     close the take, decision pending
    save              -> save()    queue the take for background saving
    reject / discard  -> reject()  drop the buffered take

Frames are buffered through lerobot's own episode buffer: images go to
temporary PNGs via a background image-writer thread pool (cheap per step).
save() detaches the buffer and hands it to a single background saver thread,
so the expensive ffmpeg video encoding never blocks the operator — the next
take can start recording while the previous one is still encoding. The saver
reports per-camera progress through save_cb so the terminal status line can
show it. close() waits for queued saves before finalizing.

Multiple episodes recorded in one run (and across runs — an existing dataset
at the same root is appended to) land in a single dataset.
"""

from __future__ import annotations

from collections.abc import Callable
import io
import logging
import os
from pathlib import Path
import queue
import threading
from typing import Any

# The SVT-AV1 encoder prints a ~20-line config banner per encoded video
# directly to stderr, bypassing the libav log level lerobot already sets.
# It reads this env var at encoder init, so it must be set before encoding.
os.environ.setdefault("SVT_LOG", "1")  # 1 = errors only

import av
import datasets
import lerobot.datasets.lerobot_dataset as _lerobot_dataset_module
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.datasets.video_utils import encode_video_frames
from lerobot.utils.constants import ACTION
from lerobot.utils.constants import OBS_STR
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Same default as lerobot-record: PNG writer threads per camera.
_IMAGE_WRITER_THREADS_PER_CAMERA = 4

IDLE = "idle"
RECORDING = "recording"
PENDING = "pending"

# NVENC options mirroring lerobot's CPU defaults where they exist: g=2 keeps a
# keyframe every 2 frames for fast random access during training, which NVENC
# only accepts with B-frames off. cq 30 lands near libsvtav1 crf=30 file sizes.
_NVENC_OPTIONS = {"g": "2", "bf": "0", "preset": "p4", "tune": "hq", "rc": "vbr", "cq": "30"}


def _nvenc_smoke_test() -> bool:
    """Whether h264_nvenc encoding actually works here — the codec being listed
    is not enough (libnvidia-encode and a working driver are also needed), so
    run one tiny in-memory encode with the exact options used in production.
    NVENC refuses frames below 145x49, so probe just above that."""
    try:
        buffer = io.BytesIO()
        with av.open(buffer, "w", format="mp4") as output:
            stream = output.add_stream("h264_nvenc", 25, options=dict(_NVENC_OPTIONS))
            stream.width, stream.height = 160, 64
            stream.pix_fmt = "yuv420p"
            frame = av.VideoFrame.from_ndarray(np.zeros((64, 160, 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
    except Exception:
        return False
    return True


def _encode_video_frames_nvenc(imgs_dir: Path | str, video_path: Path | str, fps: int) -> None:
    """h264_nvenc counterpart of lerobot's encode_video_frames: same frame
    globbing and muxing, GPU encoder. Leaves the PNGs in place (the caller,
    _encode_temporary_episode_video, deletes them)."""
    input_list = sorted(Path(imgs_dir).glob("frame-" + "[0-9]" * 6 + ".png"))
    if not input_list:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")
    with Image.open(input_list[0]) as first_image:
        width, height = first_image.size

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path), "w") as output:
        stream = output.add_stream("h264_nvenc", fps, options=dict(_NVENC_OPTIONS))
        stream.pix_fmt = "yuv420p"
        stream.width = width
        stream.height = height
        for input_path in input_list:
            with Image.open(input_path) as input_image:
                frame = av.VideoFrame.from_image(input_image.convert("RGB"))
            packet = stream.encode(frame)
            if packet:
                output.mux(packet)
        packet = stream.encode()
        if packet:
            output.mux(packet)

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


class EpisodeRecorder:
    """One LeRobot dataset per run; one take at a time, saves in background.

    State machine: IDLE --start()--> RECORDING --end()--> PENDING, then
    save()/reject() back to IDLE. save()/reject() also accept the RECORDING
    state directly (they close the take first), so the operator can type
    "save" without typing "hold" beforehand.

    Episode indices are assigned by this class (not by lerobot's default of
    meta.total_episodes) because a new take can begin while earlier takes are
    still queued for saving — i.e. before the dataset's episode count has
    caught up. The saver thread processes takes in FIFO order, which is what
    lerobot's save-time validation (buffer index == total episodes) needs.
    """

    def __init__(self, robot, fps: int, root: str | Path, repo_id: str):
        self._joint_names = list(robot.action_features.keys())
        features = {
            **hw_to_dataset_features(robot.observation_features, OBS_STR),
            **hw_to_dataset_features(robot.action_features, ACTION),
        }
        root = Path(root)
        num_cameras = sum(isinstance(shape, tuple) for shape in robot.observation_features.values())
        threads = _IMAGE_WRITER_THREADS_PER_CAMERA * max(num_cameras, 1)

        if (root / "meta" / "info.json").exists():
            self.dataset = LeRobotDataset(repo_id, root=root)
            if self.dataset.fps != fps:
                raise ValueError(
                    f"Existing dataset at {root} was recorded at {self.dataset.fps} Hz, "
                    f"but the control loop runs at {fps} Hz — pick another --record_dir"
                )
            missing = set(features) - set(self.dataset.features)
            if missing:
                raise ValueError(
                    f"Existing dataset at {root} lacks features {sorted(missing)} — pick another --record_dir"
                )
            self.dataset.start_image_writer(num_processes=0, num_threads=threads)
            logger.info(
                "Recording resumes dataset %s at %s (%d episodes so far)",
                repo_id,
                root,
                self.dataset.meta.total_episodes,
            )
        else:
            self.dataset = LeRobotDataset.create(
                repo_id,
                fps,
                features=features,
                root=root,
                robot_type=getattr(robot, "name", None),
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=threads,
            )
            logger.info("Recording to new dataset %s at %s", repo_id, root)

        # save_episode maps the buffer through an HF dataset, whose "Map: 100%|…"
        # progress bars would scribble over the pinned input line. Loading an
        # existing dataset above re-enables them, so disable *after* it.
        datasets.disable_progress_bars()

        # GPU (NVENC) video encoding when possible. lerobot only exposes its
        # CPU codecs, so the encode function its dataset module imported is
        # swapped for _patched_encode_video_frames (restored in close()).
        self._vcodec = self._choose_video_codec()
        if self._vcodec is not None:
            _lerobot_dataset_module.encode_video_frames = self._patched_encode_video_frames

        self._next_episode_index = self.dataset.meta.total_episodes
        self.dataset.episode_buffer = self.dataset.create_episode_buffer(self._next_episode_index)

        # Report per-camera encoding progress: _save_episode_video is called by
        # the saver thread once per camera, so wrapping it is the one hook that
        # sees "which camera, which episode" without reimplementing save_episode.
        self._video_keys = list(self.dataset.meta.video_keys)
        original_save_video = self.dataset._save_episode_video

        def _save_video_with_progress(video_key: str, episode_index: int):
            position = self._video_keys.index(video_key) + 1
            self._set_save_status(f"ep {episode_index}: encoding {video_key} ({position}/{len(self._video_keys)})")
            return original_save_video(video_key, episode_index)

        self.dataset._save_episode_video = _save_video_with_progress

        self._state = IDLE
        self._save_queue: queue.Queue[tuple[dict, int] | None] = queue.Queue()
        self._saver: threading.Thread | None = None
        # Called with (state, steps) on every transition and every added frame,
        # so the terminal UI can show a live "● REC n steps" indicator.
        self.status_cb: Callable[[str, int], None] | None = None
        # Called with a short progress string while a save is running in the
        # background (per-camera encoding progress), and None when done.
        self.save_cb: Callable[[str | None], None] | None = None

    # -- video codec ---------------------------------------------------------

    def _choose_video_codec(self) -> str | None:
        """Pick "h264_nvenc" (GPU), "h264" (CPU), or None for lerobot's default
        (libsvtav1, CPU).

        Episodes are appended into shared video files by concatenation, so the
        codec must stay consistent within a dataset: a resumed dataset keeps
        whatever codec it already uses, and only an h264 one can go to NVENC.
        """
        existing = None
        for key in self.dataset.meta.video_keys:
            info = self.dataset.meta.info["features"][key].get("info") or {}
            if info.get("video.codec"):
                existing = info["video.codec"]
                break

        if existing is not None and existing != "h264":
            logger.info("Video encoding: CPU (%s — keeping the resumed dataset's codec)", existing)
            return None
        if _nvenc_smoke_test():
            logger.info("Video encoding: GPU (h264_nvenc)")
            return "h264_nvenc"
        if existing == "h264":
            logger.info("Video encoding: CPU (h264 — NVENC unavailable, keeping the resumed dataset's codec)")
            return "h264"
        logger.info("Video encoding: CPU (libsvtav1 — NVENC unavailable)")
        return None

    def _patched_encode_video_frames(self, imgs_dir, video_path, fps, **kwargs) -> None:
        """Stands in for encode_video_frames inside lerobot's save path. Tries
        NVENC, and on failure falls back to CPU — staying on h264 so already
        encoded episodes can still be concatenated with."""
        if self._vcodec == "h264_nvenc":
            try:
                _encode_video_frames_nvenc(imgs_dir, video_path, fps)
                return
            except Exception:
                logger.warning("NVENC encoding failed — falling back to CPU h264", exc_info=True)
                self._vcodec = "h264"
                Path(video_path).unlink(missing_ok=True)  # drop any partial output
        kwargs.setdefault("overwrite", True)
        encode_video_frames(imgs_dir, video_path, fps, vcodec="h264", **kwargs)

    # -- state -------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_recording(self) -> bool:
        return self._state == RECORDING

    @property
    def is_pending(self) -> bool:
        return self._state == PENDING

    @property
    def steps(self) -> int:
        buffer = self.dataset.episode_buffer
        return 0 if buffer is None else buffer["size"]

    def _notify(self) -> None:
        if self.status_cb is not None:
            self.status_cb(self._state, self.steps)

    def _set_save_status(self, text: str | None) -> None:
        if text is not None and (queued := self._save_queue.qsize()):
            text += f" · +{queued} queued"
        if self.save_cb is not None:
            self.save_cb(text)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._state != IDLE:
            raise RuntimeError(f"cannot start recording from state {self._state!r}")
        self._state = RECORDING
        logger.info("● Recording — 'hold'/'stop' ends the take, then 'save' or 'reject'")
        self._notify()

    def add(self, observation: dict[str, Any], action: np.ndarray, task: str) -> None:
        """Buffer one control step. No-op unless a take is running, so callers
        can invoke it unconditionally from every point that drives the arm."""
        if self._state != RECORDING:
            return
        frame = {
            **build_dataset_frame(self.dataset.features, observation, prefix=OBS_STR),
            **build_dataset_frame(
                self.dataset.features, dict(zip(self._joint_names, action, strict=True)), prefix=ACTION
            ),
            "task": str(task),
        }
        self.dataset.add_frame(frame)
        self._notify()

    def end(self) -> None:
        """Close the running take; it stays buffered until save()/reject()."""
        if self._state != RECORDING:
            return
        if self.steps == 0:
            self._state = IDLE
            logger.info("Take had no frames — nothing to save")
        else:
            self._state = PENDING
            logger.info("■ Take ended after %d steps — type 'save' or 'reject'", self.steps)
        self._notify()

    def save(self) -> None:
        """Detach the buffered take and queue it for background saving.
        Returns immediately — the next take can start while this one encodes."""
        self.end()
        if self._state != PENDING:
            return
        buffer = self.dataset.episode_buffer
        steps = buffer["size"]
        self._next_episode_index += 1
        self.dataset.episode_buffer = self.dataset.create_episode_buffer(self._next_episode_index)
        self._state = IDLE
        if self._saver is None:
            self._saver = threading.Thread(target=self._save_worker, daemon=True, name="episode-saver")
            self._saver.start()
        self._save_queue.put((buffer, steps))
        logger.info(
            "Episode queued for saving (%d steps) — encoding in the background, 'record' works right away", steps
        )
        self._notify()

    def reject(self) -> None:
        """Drop the buffered take (frames and temporary images)."""
        self.end()
        if self._state != PENDING:
            return
        steps = self.steps
        self.dataset.clear_episode_buffer()
        # clear_episode_buffer recreated the buffer with the dataset's episode
        # count as index, which lags while saves are queued — reissue ours.
        self.dataset.episode_buffer = self.dataset.create_episode_buffer(self._next_episode_index)
        self._state = IDLE
        logger.info("Rejected take (%d steps discarded)", steps)
        self._notify()

    def close(self) -> None:
        """Discard any unsaved take (quitting is a rejection), wait for queued
        saves, and finalize the dataset so its parquet files are readable."""
        if self._state in (RECORDING, PENDING) and self.steps > 0:
            logger.warning("Quitting with an unsaved take — discarding %d steps", self.steps)
        if self._state in (RECORDING, PENDING):
            self._state = PENDING
            self.reject()
        if self._saver is not None:
            if not self._save_queue.empty():
                logger.info("Waiting for background episode saves to finish...")
            self._save_queue.put(None)
            self._saver.join()
            self._saver = None
        self.dataset.stop_image_writer()
        self.dataset.finalize()
        _lerobot_dataset_module.encode_video_frames = encode_video_frames
        total = self.dataset.meta.total_episodes
        if total:
            logger.info("Dataset finalized: %d episode(s) at %s", total, self.dataset.root)

    # -- background saver ----------------------------------------------------

    def _save_worker(self) -> None:
        """Saves queued takes strictly in FIFO order. The main thread only
        touches the *current* episode buffer and the (thread-safe) image
        writer, so save_episode(episode_data=...) here never races it."""
        while True:
            job = self._save_queue.get()
            if job is None:
                return
            buffer, steps = job
            episode_index = buffer["episode_index"]
            try:
                self._set_save_status(f"ep {episode_index}: writing data ({steps} steps)")
                self.dataset.save_episode(episode_data=buffer)
                logger.info("Saved episode %d (%d steps)", episode_index, steps)
            except Exception:
                # Episode indices are consecutive, so a failed save also breaks
                # the takes queued behind it — say so instead of failing quietly.
                logger.exception(
                    "Saving episode %d FAILED — its frames are lost and queued saves may fail too", episode_index
                )
            finally:
                self._set_save_status(None)
                self._save_queue.task_done()
