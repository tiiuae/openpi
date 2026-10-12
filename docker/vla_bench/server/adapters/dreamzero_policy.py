r"""Adapter for DreamZero (dreamzero0/dreamzero @ ab790c19, NVIDIA GEAR, Apache-2.0): `dreamzero`.

A 14 B world-action model: the Wan2.1-I2V-14B-480P DiT made block-causal, with action and state
registers, that jointly denoises the next 2 latent video frames and a 24-action chunk. It was
LoRA-post-trained from `GEAR-Dreams/DreamZero-AgiBot` on 7-D right-arm absolute joints with relative
actions (BLOCKERS D-DZ1 ... D-DZ12). It returns 24 rows, all executed (user decision B-DZ.1).

Everything model-side is the authors' own code, not a re-implementation:

* Loading: `GrootSimPolicy(embodiment_tag=trossen_right7, model_path=<checkpoint>)` with a 1-D
  device mesh "ip", as `socket_test_optimized_AR.main()` builds it; `VLA.load_lora` (patch 0003 =
  upstream PR #83) rebuilds the LoRA checkpoint on the DreamZero-AgiBot base it was trained from,
  the policy merges the LoRA and runs in bf16 with the authors' torch.compile of the encoders. The
  only change is where the base files live: the checkpoint's `config.json` and
  `experiment_cfg/conf.yaml` ship as trained and name training-host paths, so the adapter writes a
  private view of the checkpoint (a temporary directory: the LoRA and `metadata.json` linked, the
  two configs with the base paths replaced by `agibot_dir`, `wan_dir` and `tokenizer_dir`). The
  shipped files are never edited.
* Inference: `policy.lazy_joint_forward_causal`, the authors' settings unchanged: 16 UniPC flow
  steps, classifier-free guidance 5.0, the DiT run on 8 of the 16 steps (`dit_step_mask`, the
  release's default `NUM_DIT_STEPS` 8; `ENABLE_DIT_CACHE` is set as the authors' server sets it but
  no code reads it), fixed noise seed 1140, a KV cache over past blocks. The policy's own `unapply`
  de-normalises the relative chunk (q99 bounds from `experiment_cfg/metadata.json`) and adds the
  observation's state, so the output is ABSOLUTE joint targets.
* Two GPUs (default, `gpus=2`): the server process is rank 0 of a 2-rank torch.distributed group
  on GPU 0 and starts ONE child process, rank 1 on GPU 1, running the authors' worker-loop logic
  (`WebsocketPolicyServer._worker_loop`: wait for a signal, receive the observation, call the same
  forward in lockstep between two barriers). The two guidance passes run in parallel over the
  "ip" mesh and are exchanged by NCCL P2P inside the authors' forward. The signal and the
  observation travel over a local pipe instead of a gloo + NCCL broadcast, so no collective is
  pending while the robot is idle (the authors give theirs a 50,000 s timeout). The child sends
  its own chunk back and the adapter compares the two ranks' outputs on every call.
* One GPU (`gpus=1` or `DREAMZERO_GPUS=1`): a 1-rank group, `ip_size` 1, the two guidance passes
  in sequence (supported by the authors' code); about twice the latency.

Frames per request (`frame_mode` or `DREAMZERO_FRAME_MODE`; BLOCKERS "DreamZero deployment"):

* `single` (variant i): every request is a fresh 1-frame call, the authors' first-chunk condition
  (the model resets its KV cache on a 1-frame input). Nothing is carried between requests.
* `repeat4` (variant ii): the first request after `reset()` is a 1-frame call; every later one
  keeps the authors' autoregressive KV cache and sends the current frame repeated 4 times, in place
  of the 4 frames at offsets [-23, -16, -8, 0] that the authors' client accumulates during the
  executed block (the robot client sends one frame per request). The model itself restarts its
  cache every 4 blocks (local attention window 9 latent frames) and on a prompt change.
  `vla_server.py` calls `reset()` on every new connection, i.e. per episode.

Observation (BLOCKERS D-DZ2): `video.cam_high` <- primary, `video.cam_right_wrist` <- wrist, each
(T, H, W, 3) uint8 RGB at the native 640x480; `state.joint_pos` (1, 6) and `state.gripper_pos`
(1, 1) float64 from the raw 7-D right-arm state; `annotation.task`. The authors' eval transform
then centre-crops 0.95, resizes to 176x320, tiles the 2x2 canvas (cam_high top-left,
cam_right_wrist top-right, bottom row black; patch 0001), q99-normalises the state and builds the
prompt.

Fail-closed checks at load: the base files (sizes, sha256 of the small files), the checkpoint's
q01/q99 bounds against the dataset statistics shipped beside it (`norm_stats_dir`), the horizon,
the inference settings, the embodiment's relative-action keys, `ip_size` equal to the GPU count.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import logging
import os
import socket
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

from .base import PolicyAdapter

log = logging.getLogger(__name__)

DREAMZERO_SHA = "ab790c198fbce33503358efbbd4187ce9a89adf3"  # gitleaks:allow (a git commit id)
TAG = "trossen_right7"
H = 24
FRAME_MODES = ("single", "repeat4")
DEFAULT_FRAME_MODE = "single"
DIT_MASK = [True, True, True, False, False, False, True, False,
            False, False, True, False, False, True, True, True]
# The base repositories training loaded, at the pinned revisions; (bytes, sha256 or None) per file
# read at load: sha256 for the small files, size for the large ones (public Hub metadata).
AGIBOT = ("GEAR-Dreams/DreamZero-AgiBot",
          "a8e108456e537ab30e7e84523d41d12a7e739ca5")  # gitleaks:allow (a git commit id)
WAN = ("Wan-AI/Wan2.1-I2V-14B-480P",
       "6b73f84e66371cdfe870c72acd6826e1d61cf279")  # gitleaks:allow (a git commit id)
UMT5 = ("google/umt5-xxl", "66cb9e7e85526fe440a945569e42c72fb6cbc0ad")  # gitleaks:allow
_AGIBOT_SHARDS = (4935816720, 4983107552, 4937567096, 4987670360, 4987714608, 4950963456,
                  4951004896, 4950963472, 4950984200, 1212823208)
ASSET_FILES: dict[str, dict[str, tuple[int, str | None]]] = {
    "agibot": {
        "model.safetensors.index.json": (
            203009,
            "232b161ae7fd8edbcf5898bfe81dd8bfc9ce1e04b8a757dcd4a7674756b92946",  # gitleaks:allow
        ),
        **{f"model-{i:05d}-of-00010.safetensors": (n, None)
           for i, n in enumerate(_AGIBOT_SHARDS, start=1)},
    },
    "wan": {
        "models_t5_umt5-xxl-enc-bf16.pth": (11361920418, None),
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": (4772359047, None),
        "Wan2.1_VAE.pth": (507609880, None),
    },
    "tokenizer": {
        "spiece.model": (
            4548313,
            "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458",  # gitleaks:allow
        ),
        "tokenizer.json": (
            16853013,
            "af904105ce1071b1202bba0059a841f4a7b85b48b6ec179c4948e3483476e0dd",  # gitleaks:allow
        ),
        "tokenizer_config.json": (
            6841,
            "616015e2b6c26f9a98ae3171634518d440a46260ebc1115c2596792d33827bdc",  # gitleaks:allow
        ),
        "special_tokens_map.json": (
            6623,
            "14ef492a69e59396e918521a5f8394af1acf9f8dddd47273693b2b42570dfd12",  # gitleaks:allow
        ),
        "config.json": (
            799,
            "17a62e1c5e1cdfa62dd15794081a523d1310c77135ad99158a9bd5cfbdef83f6",  # gitleaks:allow
        ),
    },
}
ASSET_SOURCES = {"agibot": AGIBOT, "wan": WAN, "tokenizer": UMT5}
Json = dict | list | str | int | float | bool | None


def _sha256(path: Path) -> str:
    """Hex sha256 of a file, read in 8 MiB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def resolve_asset_dir(spec: str, component: str) -> Path:
    """A base directory from a plain path or 'hf:<repo>@<40-hex revision>'.

    Args:
        spec: a directory, or a pinned Hub snapshot in the local Hugging Face cache.
        component: 'agibot', 'wan' or 'tokenizer' (selects the repo training used).

    Returns:
        The directory.

    Raises:
        ValueError: the spec names another repo or revision than training used.
    """
    if not spec.startswith("hf:"):
        return Path(spec)
    repo, _, rev = spec[3:].partition("@")
    if (repo, rev) != ASSET_SOURCES[component]:
        want = ASSET_SOURCES[component]
        raise ValueError(f"dreamzero: {component} must be {want[0]}@{want[1]}, got {spec}")
    from .hub_pins import pinned_snapshot

    return pinned_snapshot(repo, rev)


def check_asset_dir(path: Path, component: str) -> None:
    """Refuse a base directory whose files differ from the pinned ones (size; small files sha256).

    Raises:
        RuntimeError: a file is missing, has another size or another sha256.
    """
    bad = []
    for name, (size, sha) in ASSET_FILES[component].items():
        f = path / name
        if not f.is_file():
            bad.append(f"{name}: missing")
        elif (got := f.stat().st_size) != size:
            bad.append(f"{name}: {got} bytes, expected {size}")
        elif sha is not None and (h := _sha256(f)) != sha:
            bad.append(f"{name}: sha256 {h} != pinned {sha}")
    if bad:
        repo, rev = ASSET_SOURCES[component]
        raise RuntimeError(f"dreamzero: {path} is not {repo}@{rev}:\n  " + "\n  ".join(bad))


def _replace_paths(node: Json, mapping: dict[str, str]) -> Json:
    """Every string equal to a mapped directory, or below one, re-rooted; the rest unchanged."""
    if isinstance(node, dict):
        return {k: _replace_paths(v, mapping) for k, v in node.items()}
    if isinstance(node, list):
        return [_replace_paths(v, mapping) for v in node]
    if isinstance(node, str):
        for old, new in mapping.items():
            if node == old or node.startswith(old + "/"):
                return new + node[len(old):]
    return node


def make_checkpoint_view(checkpoint: Path, dirs: dict[str, Path], out: Path) -> dict[str, str]:
    """Write a private view of a LoRA checkpoint whose base paths point at `dirs`.

    The LoRA (`model.safetensors`) and `experiment_cfg/metadata.json` are linked; `config.json` and
    `experiment_cfg/conf.yaml` are rewritten: each string equal to (or below) the recorded
    DreamZero-AgiBot, Wan2.1 or umt5-xxl directory is re-rooted. Paths not read at inference (the
    dataset root, the output dir) are left as trained.

    Args:
        checkpoint: the shipped checkpoint directory (read only).
        dirs: {'agibot', 'wan', 'tokenizer'} -> the local base directories.
        out: an empty directory to write the view into.

    Returns:
        {recorded directory: local directory}.
    """
    from omegaconf import OmegaConf

    conf = OmegaConf.to_container(OmegaConf.load(checkpoint / "experiment_cfg/conf.yaml"))
    cfg = json.loads((checkpoint / "config.json").read_text())
    dit = cfg["action_head_cfg"]["config"]["diffusion_model_cfg"]["diffusion_model_pretrained_path"]
    mapping = {str(conf["pretrained_model_path"]): str(dirs["agibot"]),
               str(dit): str(dirs["wan"]), str(conf["tokenizer_path"]): str(dirs["tokenizer"])}
    (out / "experiment_cfg").mkdir(parents=True)
    (out / "model.safetensors").symlink_to((checkpoint / "model.safetensors").resolve())
    (out / "experiment_cfg/metadata.json").symlink_to(
        (checkpoint / "experiment_cfg/metadata.json").resolve())
    (out / "config.json").write_text(json.dumps(_replace_paths(cfg, mapping), indent=2))
    new_conf = OmegaConf.create(_replace_paths(conf, mapping))
    OmegaConf.save(new_conf, out / "experiment_cfg/conf.yaml")
    return mapping


def build_observation(obs: dict, n_frames: int) -> dict:
    """The authors' observation dict from the benchmark's {primary, wrist, state, task}.

    Args:
        obs: primary / wrist (H, W, 3) uint8 RGB, state (7,) right arm (6 rad + 1 m), task str.
        n_frames: 1 (a fresh call) or 4 (the current frame repeated, variant ii).

    Returns:
        video.cam_high / video.cam_right_wrist (n, H, W, 3) uint8, state.joint_pos (1, 6) and
        state.gripper_pos (1, 1) float64, annotation.task.
    """
    out = {}
    for key, cam in (("primary", "cam_high"), ("wrist", "cam_right_wrist")):
        a = np.asarray(obs[key])
        if a.ndim != 3 or a.shape[-1] != 3 or a.dtype != np.uint8:
            raise ValueError(f"dreamzero expects (H, W, 3) uint8 RGB for {key}, got {a.shape}")
        out[f"video.{cam}"] = np.repeat(np.ascontiguousarray(a)[None], n_frames, axis=0)
    state = np.asarray(obs["state"], np.float32).reshape(-1)
    if state.size != 7 or not np.isfinite(state).all():
        raise ValueError(f"dreamzero needs a finite 7-D right-arm state, got {state.shape}")
    out["state.joint_pos"] = state[None, :6].astype(np.float64)
    out["state.gripper_pos"] = state[None, 6:7].astype(np.float64)
    out["annotation.task"] = str(obs.get("task") or "")
    return out


def chunk_from_result(act: dict) -> np.ndarray:
    """(24, 7) float32 from the policy's action batch (the gripper arrives squeezed to (24,))."""
    j = np.asarray(act["action.joint_pos"], dtype=np.float64)
    g = np.asarray(act["action.gripper_pos"], dtype=np.float64)
    chunk = np.concatenate([j.reshape(H, -1), g.reshape(H, -1)], axis=1).astype(np.float32)
    if chunk.shape != (H, 7) or not np.isfinite(chunk).all():
        finite = bool(np.isfinite(chunk).all())
        raise RuntimeError(f"dreamzero returned {chunk.shape}, finite={finite}")
    return chunk


def _free_port() -> int:
    """An unused TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _build_policy(rank: int, world: int, init_method: str, view: str) -> object:
    """Join the group, build the authors' mesh and GrootSimPolicy on GPU `rank`."""
    os.environ.setdefault("ENABLE_DIT_CACHE", "true")
    os.environ.setdefault("ATTENTION_BACKEND", "TE")
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world,
                            timeout=timedelta(minutes=10))
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("ip",))
    torch._dynamo.config.recompile_limit = 800
    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

    return GrootSimPolicy(embodiment_tag=EmbodimentTag(TAG), model_path=view, device="cuda",
                          device_mesh=mesh)


def _forward(policy: object, observation: dict) -> np.ndarray:
    """One lockstep forward between two barriers (the authors' server and worker loop)."""
    import torch
    import torch.distributed as dist
    from tianshou.data import Batch

    dist.barrier()
    with torch.no_grad():
        result, _video = policy.lazy_joint_forward_causal(Batch(obs=observation))
    dist.barrier()
    return chunk_from_result(result.act)


def _worker_main(conn: object, init_method: str, view: str) -> None:
    """Rank 1: the authors' worker loop, with the signal and observation read from a pipe."""
    try:
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG: die with the server
    except (OSError, AttributeError):
        pass
    import torch
    import torch.distributed as dist

    try:
        policy = _build_policy(1, 2, init_method, view)
        conn.send(("ready", {"ip_size": int(policy.trained_model.action_head.ip_size)}))
        while True:
            msg = conn.recv()
            if msg[0] == "stop":
                break
            chunk = _forward(policy, msg[1])
            conn.send(("done", chunk, float(torch.cuda.max_memory_allocated() / 2**30)))
    except EOFError:
        pass
    except Exception as exc:  # noqa: BLE001  (reported to rank 0, which fails the request)
        import traceback

        conn.send(("error", f"{exc!r}\n{traceback.format_exc()}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class DreamZeroAdapter(PolicyAdapter):
    """DreamZero (LoRA on DreamZero-AgiBot) behind the benchmark's adapter interface."""

    name = "dreamzero"
    chunk_len = H
    exec_len = H

    def __init__(
        self,
        checkpoint: str,
        agibot_dir: str,
        wan_dir: str,
        tokenizer_dir: str,
        norm_stats_dir: str,
        *,
        gpus: int | None = None,
        frame_mode: str | None = None,
        verify_assets: bool = True,
        ready_timeout_s: float = 7200.0,
    ) -> None:
        """Load the checkpoint with the authors' GrootSimPolicy on 1 or 2 GPUs.

        Args:
            checkpoint: the LoRA checkpoint directory (`model.safetensors`, `config.json`,
                `experiment_cfg/{conf.yaml,metadata.json}`).
            agibot_dir: DreamZero-AgiBot (a directory or 'hf:<repo>@<rev>').
            wan_dir: Wan2.1-I2V-14B-480P without its DiT shards (a directory or 'hf:...').
            tokenizer_dir: the umt5-xxl tokenizer (a directory or 'hf:...').
            norm_stats_dir: the dataset's `relative_stats_dreamzero.json` and `stats.json`; their
                q01/q99 must equal the checkpoint's `metadata.json`.
            gpus: 2 (guidance passes in parallel) or 1; default `DREAMZERO_GPUS`, else 2.
            frame_mode: 'single' (variant i) or 'repeat4' (variant ii); default
                `DREAMZERO_FRAME_MODE`, else 'single'.
            verify_assets: check the base files before loading.
            ready_timeout_s: how long to wait for rank 1 to load.
        """
        t0 = time.time()
        self.checkpoint = Path(checkpoint)
        self.world = int(gpus or os.environ.get("DREAMZERO_GPUS") or 2)
        mode = frame_mode or os.environ.get("DREAMZERO_FRAME_MODE") or DEFAULT_FRAME_MODE
        self.frame_mode = mode
        if self.world not in (1, 2) or mode not in FRAME_MODES:
            raise ValueError(f"dreamzero: gpus must be 1 or 2 and frame_mode one of "
                             f"{FRAME_MODES}; got {self.world}, {mode!r}")
        import torch

        if torch.cuda.device_count() < self.world:
            raise RuntimeError(f"dreamzero: gpus={self.world} but {torch.cuda.device_count()} "
                               "GPU(s) visible; give the container 2 GPUs, or set DREAMZERO_GPUS=1")
        dirs = {k: resolve_asset_dir(v, k) for k, v in
                (("agibot", agibot_dir), ("wan", wan_dir), ("tokenizer", tokenizer_dir))}
        if verify_assets:
            for k, v in dirs.items():
                check_asset_dir(v, k)
        self.asset_dirs = {k: str(v) for k, v in dirs.items()}
        self.norm_stats_dir = str(norm_stats_dir)
        self._check_stats()
        self.view = Path(tempfile.mkdtemp(prefix="dreamzero_ckpt_view_"))
        self.path_map = make_checkpoint_view(self.checkpoint, dirs, self.view)
        self._fresh, self._proc, self._conn = True, None, None
        self.calls, self.rank_mismatch_calls, self.peak_gib = 0, 0, {}
        init_method = f"tcp://127.0.0.1:{_free_port()}"
        if self.world == 2:
            self._start_worker(init_method)
        self.policy = _build_policy(0, self.world, init_method, str(self.view))
        if self.world == 2:
            self._wait_ready(ready_timeout_s)
        self._check_contract()
        atexit.register(self.close)
        self.load_seconds = time.time() - t0
        log.info("dreamzero loaded in %.0f s on %d GPU(s), frame_mode %s, view %s",
                 self.load_seconds, self.world, self.frame_mode, self.view)

    # ----------------------------------------------------------------------------------- loading
    def _start_worker(self, init_method: str) -> None:
        """Start rank 1 (spawn: a fresh interpreter, CUDA-safe)."""
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        # Not daemonic: torch.compile may start worker processes, which a daemonic process may
        # not. The child exits with this process anyway (PR_SET_PDEATHSIG, and close() at exit).
        self._proc = ctx.Process(target=_worker_main, args=(child, init_method, str(self.view)),
                                 daemon=False, name="dreamzero-rank1")
        self._proc.start()
        child.close()

    def _recv(self, timeout_s: float) -> tuple:
        """The next message from rank 1; fail closed if it died, errored or timed out."""
        deadline = time.time() + timeout_s
        while not self._conn.poll(5.0):
            if not self._proc.is_alive():
                raise RuntimeError(f"dreamzero: rank 1 exited (code {self._proc.exitcode})")
            if time.time() > deadline:
                raise RuntimeError(f"dreamzero: rank 1 silent for {timeout_s:.0f} s")
        msg = self._conn.recv()
        if msg[0] == "error":
            raise RuntimeError(f"dreamzero: rank 1 failed:\n{msg[1]}")
        return msg

    def _wait_ready(self, timeout_s: float) -> None:
        """Rank 1 has built its policy and joined the mesh."""
        msg = self._recv(timeout_s)
        if msg[0] != "ready" or msg[1]["ip_size"] != 2:
            raise RuntimeError(f"dreamzero: rank 1 reported {msg}")

    def _check_stats(self) -> None:
        """The checkpoint's q01/q99 (metadata.json) must equal the dataset statistics."""
        meta = json.loads((self.checkpoint / "experiment_cfg/metadata.json").read_text())[TAG]
        st = meta["statistics"]
        d = Path(self.norm_stats_dir)
        rel = json.loads((d / "relative_stats_dreamzero.json").read_text())
        state = json.loads((d / "stats.json").read_text())["observation.state"]
        for q in ("q01", "q99"):
            got_s = st["state"]["joint_pos"][q] + st["state"]["gripper_pos"][q]
            got_a = [st["action"][k][q] for k in ("joint_pos", "gripper_pos")]
            want_a = [rel[k][q] for k in ("joint_pos", "gripper_pos")]
            if got_s != state[q] or got_a != want_a:
                raise RuntimeError(f"dreamzero: experiment_cfg/metadata.json {q} differs from "
                                   f"{self.norm_stats_dir}")

    def _check_contract(self) -> None:
        """Horizon, inference settings, relative-action keys and the mesh are what this serves."""
        ah = self.policy.trained_model.action_head
        cfg = self.policy.train_cfg
        got = {"action_horizon": int(ah.action_horizon),
               "num_inference_steps": int(ah.num_inference_steps), "cfg_scale": float(ah.cfg_scale),
               "seed": int(ah.seed), "ip_size": int(ah.ip_size),
               "dit_step_mask": [bool(x) for x in ah.dit_step_mask],
               "relative_action": bool(cfg.get("relative_action")),
               "relative_action_keys": list(cfg.get("relative_action_keys") or []),
               "local_attn_size": int(ah.model.local_attn_size)}
        want = {"action_horizon": H, "num_inference_steps": 16, "cfg_scale": 5.0, "seed": 1140,
                "ip_size": self.world, "dit_step_mask": DIT_MASK, "relative_action": True,
                "relative_action_keys": ["joint_pos", "gripper_pos"], "local_attn_size": 9}
        if got != want:
            raise RuntimeError(f"dreamzero: loaded {got}, expected {want}")
        self.inference = got

    # --------------------------------------------------------------------------------- inference
    def predict(self, obs: dict) -> np.ndarray:
        """Native-wire observation -> float32 (24, 7) absolute joint targets."""
        import torch

        n = 1 if (self.frame_mode == "single" or self._fresh) else 4
        observation = build_observation(obs, n)
        if self.world == 2:
            if not self._proc.is_alive():
                raise RuntimeError(f"dreamzero: rank 1 exited (code {self._proc.exitcode})")
            self._conn.send(("infer", observation))
        chunk = _forward(self.policy, observation)
        self.peak_gib["rank0"] = float(torch.cuda.max_memory_allocated() / 2**30)
        if self.world == 2:
            _, other, peak1 = self._recv(600.0)
            self.peak_gib["rank1"] = peak1
            if not np.array_equal(other, chunk):
                self.rank_mismatch_calls += 1
                log.warning("dreamzero: ranks disagree on call %d (max |d| %.3g)", self.calls,
                            float(np.abs(other - chunk).max()))
        self._fresh = False
        self.calls += 1
        if self.calls % 50 == 0:
            log.info("dreamzero: %d calls, peak GPU memory %s GiB, rank mismatches %d",
                     self.calls, {k: round(v, 2) for k, v in self.peak_gib.items()},
                     self.rank_mismatch_calls)
        return chunk

    def warmup(self) -> None:
        """Compile and fill caches: two 1-frame calls, plus two 4-frame calls in repeat4 mode."""
        t0 = time.time()
        yy, xx = np.mgrid[0:480, 0:640]
        frame = np.stack([(xx % 256), (yy % 256), ((xx + yy) % 256)], -1).astype(np.uint8)
        state = np.asarray([0.0, 1.047, 0.524, 0.628, 0.0, 0.0, 0.0], np.float32)
        obs = {"primary": frame, "wrist": frame[:, ::-1].copy(), "state": state,
               "task": "pick up the green cup and place it in the basket"}
        times = []
        for _ in range(2):
            self.reset()
            for _ in range(3 if self.frame_mode == "repeat4" else 1):
                t1 = time.time()
                self.predict(obs)
                times.append(round(time.time() - t1, 1))
        self.reset()
        self.calls = 0
        self.warmup_seconds = {"total": round(time.time() - t0, 1), "calls": times}
        log.info("dreamzero warm-up: %d calls in %.0f s %s, peak GPU memory %s GiB", len(times),
                 time.time() - t0, times, {k: round(v, 2) for k, v in self.peak_gib.items()})

    def reset(self) -> None:
        """New episode: the next request is a 1-frame call (the model rebuilds its KV cache)."""
        self._fresh = True

    def close(self) -> None:
        """Stop rank 1 and leave the process group (idempotent; also run at exit)."""
        if self._proc is not None and self._proc.is_alive():
            with contextlib.suppress(BrokenPipeError, OSError):
                self._conn.send(("stop",))
            self._proc.join(timeout=30)
            if self._proc.is_alive():
                self._proc.terminate()
        self._proc = None
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()

    def info(self) -> dict:
        """Server metadata."""
        d = super().info()
        state = ("none across requests: every request is a fresh 1-frame call"
                 if self.frame_mode == "single" else
                 "KV cache kept within an episode; reset() (each new connection) makes the next "
                 "request a 1-frame call; later requests repeat the current frame 4x")
        d.update({
            "policy_type": "dreamzero (GrootSimPolicy, LoRA on DreamZero-AgiBot, merged)",
            "dreamzero_sha": DREAMZERO_SHA, "gpus": self.world, "frame_mode": self.frame_mode,
            "statefulness": state,
            "cameras": {"primary": "cam_high -> canvas top-left (AgiBot head slot)",
                        "wrist": "cam_right_wrist -> canvas top-right (AgiBot right-hand slot)"},
            "inference": getattr(self, "inference", None), "asset_dirs": self.asset_dirs,
            "norm_stats_dir": self.norm_stats_dir, "peak_gpu_gib": self.peak_gib,
            "rank_mismatch_calls": self.rank_mismatch_calls,
            "horizon_note": "24 predicted, 24 executed (user decision B-DZ.1)"})
        return d
