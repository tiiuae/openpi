"""FLUX 3 Action (Black Forest Labs) -- the authors' own inference path, with its observation history.

Nothing about the model is re-implemented here. The adapter loads the export with the authors'
`FluxActionPolicy.from_pretrained` (manifest checksums verified on every load) and calls the authors'
`predict_action_chunk`, which packs the history, encodes the two visual snapshots with the video VAE,
runs the 4-step Euler sampler with classifier-free guidance and un-normalises the result into dataset
units. Output: (30, 7) float32 absolute targets, dims 0-5 radians, dim 6 gripper carriage metres.

WHAT THE MODEL CONDITIONS ON (exported config: inference_profile "history", n_obs_steps 8,
history_snapshots 2, condition_on_past_actions true, fps 30). For the current control tick t:

  * the measured 7-D state at the 8 consecutive 30 Hz ticks t-7 .. t;
  * the absolute command issued at the tick BEFORE each of those observations, t-8 .. t-1
    (`command_history`; index 0 is zeroed by the model, so t-7 .. t-1 are the seven that count);
  * camera images at only two of those ticks: t-7 (the oldest) and t (the current one) --
    `processing/history.py::snapshot_indices` = [0, 7] for 8 ticks and 2 snapshots.

Training windows are exactly that (`training/data.py:116-118, 310-314`: states [s-7, s], commands
[s-8, s-1], images from s-7), and every training start has a complete history (s >= 8). The authors'
control loop (`select_action`, docs/setup.md "call select_action at EVERY 30 Hz tick") records every
tick; at an episode's first tick it repeats the first observation 8 times with command = measured state.

WHY THIS ADAPTER IS STATELESS. Over the robot client's websocket the server sees an observation only when
the client queries -- every 20 ticks in sync mode, roughly once per inference in async mode -- so a
server-side ring buffer could never hold ticks t-7 .. t-1. The faithful history therefore has to come
from the client (`examples/trossen_ai/main.py --send_history 8`, opt-in), which sends it with each
request; the server hands it to this adapter as `obs["history"]`:

    obs["history"] = {"state":   (8, 7) float32   measured state, ticks t-7 .. t (last row == obs["state"])
                      "command": (8, 7) float32   command sent on the tick before each of those ticks
                      "primary": (H, W, 3) uint8  RGB cam_high frame of tick t-7
                      "wrist":   (H, W, 3) uint8  RGB cam_right_wrist frame of tick t-7
                      "valid":   int              real ticks in the window (the rest repeat the first tick)}

Without it ("history": "auto" and no window in the request), the adapter falls back to the authors' own
stateless path for a single observation: `predict_action_chunk` pads the history with 8 copies of the
current observation and command = state -- what the authors document for "a new episode". That input is
OUT OF DISTRIBUTION for every query after the first: training never shows a padded window. Its cost on the
benchmark replay is measured separately (results/deploy_eval/flux3_action/single_frame/).

mode "history": "auto"    use the request's window when present, else the padded single observation
                "require" refuse a request that carries no window (a deployment that must not degrade)
                "off"     ignore any window (the single-frame evaluation every other model gets)

Seeded and deterministic by construction: the authors' sampler draws its noise from a fresh
`torch.Generator().manual_seed(config.inference_seed)` on every call (`policy.py::_sample`), so the same
inputs give the same chunk on every call. `seed` overrides the export's inference_seed (0) -- leave it None
to reproduce the benchmark numbers.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

from .base import PolicyAdapter

log = logging.getLogger("vla_bench.flux3_action")

SCENE, WRIST = "images.scene", "images.wrist"
MODES = ("auto", "require", "off")


class Flux3ActionAdapter(PolicyAdapter):
    """(30, 7) absolute joint targets from FLUX 3 Action through the authors' predict_action_chunk."""

    def __init__(self, checkpoint: str, device: str = "cuda:0", chunk_len: int = 30, exec_len: int = 30,
                 history: str = "auto", seed: int | None = None,
                 video_vae: str | None = None, text_encoder: str | None = None,
                 own_compile: bool = False):
        if history not in MODES:
            raise ValueError(f"history must be one of {MODES}, got {history!r}")
        self.name = f"flux3_action:{checkpoint}"
        self.checkpoint, self.device_str, self.mode = str(checkpoint), str(device), history
        self.chunk_len, self.exec_len = int(chunk_len), int(exec_len)
        self.seed, self.own_compile = seed, bool(own_compile)
        # Optional overrides for the two frozen encoders. The export's config.json names them by the path
        # (or Hub spec) they had at training time; its manifest checksums config.json, so a deployer who
        # cannot use those paths either rewrites the two keys AND the manifest's config.json sha256, or
        # passes the new locations here (env FLUX_VIDEO_VAE / FLUX_TEXT_ENCODER), which leaves the export
        # byte-for-byte as produced. Either way the loaded weights are the same files.
        self.video_vae_spec = video_vae or os.environ.get("FLUX_VIDEO_VAE") or None
        self.text_encoder_spec = text_encoder or os.environ.get("FLUX_TEXT_ENCODER") or None
        self.policy = None
        self.served = {"history": 0, "single": 0}

    # ------------------------------------------------------------------------------------------ loading
    def warmup(self) -> None:
        if self.policy is not None:
            return
        import torch
        from flux_action.inference.precision import prepare_for_serving
        from flux_action.policy import FluxActionPolicy
        from flux_action.processing import history as fh

        t0 = time.time()
        kw = {}
        if self.video_vae_spec:
            from flux_action.models.video_vae import load_video_vae
            kw["video_vae"] = load_video_vae(self.video_vae_spec)
        if self.text_encoder_spec:
            from flux_action.models.text_encoder import load_text_encoder
            kw["text_encoder"] = load_text_encoder(self.text_encoder_spec)
        # The authors' offline inference (inference/offline.py::run_inference): restore on the CPU, then
        # prepare_for_serving moves it to the device. Their prepared/compiled backend accepts only the
        # released DROID geometry (policy.py::_released_inference_config) and returns prepared=False here,
        # so this is the plain bf16 eager path -- the one the checkpoint was validated in.
        policy = FluxActionPolicy.from_pretrained(self.checkpoint, device="cpu", **kw)
        setup = prepare_for_serving(policy, device=torch.device(self.device_str), compile_dit=False)
        c = policy.config
        if self.seed is not None:
            c.inference_seed = int(self.seed)
        c.validate_inference()

        # Refuse to serve a checkpoint whose contract differs from what this adapter packs.
        want = {"inference_profile": "history", "action_dim": 7, "camera_layout": "side_by_side",
                "camera_keys": (SCENE, WRIST), "action_parameterization": "absolute",
                "condition_on_past_actions": True, "chunk_size": self.chunk_len, "fps": 30.0}
        got = {k: (tuple(getattr(c, k)) if k == "camera_keys" else getattr(c, k)) for k in want}
        if got != want:
            raise RuntimeError(f"flux3_action: export contract {got} differs from the adapter's {want}")
        self.n_obs = int(c.n_obs_steps)
        snaps = fh.snapshot_indices(c)
        if snaps != [0, self.n_obs - 1]:
            raise RuntimeError(f"flux3_action: snapshots at {snaps}; the wire history carries only the "
                               f"oldest tick's images, so only [0, n_obs_steps-1] can be served")
        if self.own_compile:
            # NOT author-supported: nn.Module.compile of the DiT in place (keeps the JointSingleSeq type the
            # eager sampler asserts). Measured separately; off for every benchmark number.
            policy.dit.compile(dynamic=False)
        self.policy, self.torch = policy, torch
        self.device = policy.device
        self.load_seconds = time.time() - t0
        log.info("flux3_action loaded in %.1f s: %s, n_obs_steps %d, snapshots %s, sampler %s x%d shift %s "
                 "guidance %s, seed %d, mode %s, prepared=%s", self.load_seconds, self.checkpoint, self.n_obs,
                 snaps, c.sampler, c.num_inference_steps, c.sampler_shift, c.guidance_scale,
                 c.inference_seed, self.mode, setup.get("prepared"))

    def reset(self) -> None:
        """Nothing to clear: the history arrives complete with each request (or is padded per request by the
        authors' stateless path), and the text-embedding cache is a pure function of the caption."""
        return None

    # ------------------------------------------------------------------------------------------ inference
    def _img(self, a) -> "object":
        a = np.asarray(a)
        if a.ndim != 3 or a.shape[-1] != 3 or a.dtype != np.uint8:
            raise ValueError(f"flux3_action expects (H, W, 3) uint8 RGB frames, got {a.shape} {a.dtype}")
        return self.torch.from_numpy(np.array(a, dtype=np.uint8, copy=True)).permute(2, 0, 1)   # (3, H, W) uint8

    def batch(self, obs: dict) -> tuple[dict, str]:
        """The authors' batch for one request, and which path it took ('history' or 'single')."""
        torch = self.torch
        state = np.asarray(obs["state"], np.float32).reshape(-1)
        if state.size != 7:
            raise ValueError(f"flux3_action needs the 7-D right-arm state, got {state.size}")
        task = str(obs.get("task") or "")
        cur_p, cur_w = self._img(obs["primary"]), self._img(obs["wrist"])
        h = obs.get("history") if self.mode != "off" else None
        if h is None:
            if self.mode == "require":
                raise ValueError("flux3_action is in history='require' mode and the request carries no history "
                                 "window: start the robot client with --send_history 8")
            # authors' stateless single observation -> history.observation_window pads it
            return {SCENE: cur_p[None].to(self.device), WRIST: cur_w[None].to(self.device),
                    "state": torch.from_numpy(state)[None].to(self.device), "task": [task]}, "single"
        n = self.n_obs
        hs = np.asarray(h["state"], np.float32).reshape(-1, 7)
        hc = np.asarray(h["command"], np.float32).reshape(-1, 7)
        if hs.shape != (n, 7) or hc.shape != (n, 7):
            raise ValueError(f"flux3_action history must be ({n}, 7) state and command, got {hs.shape} {hc.shape}")
        if not np.array_equal(hs[-1], state):
            raise ValueError("flux3_action: the history's last state row is not the request's current state")
        old_p, old_w = self._img(h["primary"]), self._img(h["wrist"])
        if old_p.shape != cur_p.shape or old_w.shape != cur_w.shape:
            raise ValueError("flux3_action: the oldest history frames and the current frames differ in size")
        # (1, n, 3, H, W) uint8 on the policy device (the authors require cameras and state on one device; the
        # uint8 -> [0, 1] division itself happens on the CPU inside pack_conditioning, as the authors want).
        # Only indices 0 (tick t-n+1) and n-1 (tick t) are encoded
        # (snapshot_indices, checked at load); the frames in between never reach the model, so they are
        # filled with the oldest frame rather than sent over the wire.
        cams = {k: torch.stack([o] * (n - 1) + [c])[None].to(self.device)
                for k, o, c in ((SCENE, old_p, cur_p), (WRIST, old_w, cur_w))}
        return {**cams, "state": torch.from_numpy(hs)[None].to(self.device),
                "command_history": torch.from_numpy(hc)[None].to(self.device), "task": [task]}, "history"

    def predict(self, obs: dict) -> np.ndarray:
        if self.policy is None:
            self.warmup()
        b, path = self.batch(obs)
        with self.torch.inference_mode():
            chunk = self.policy.predict_action_chunk(b)
        out = chunk[0].float().cpu().numpy().astype(np.float32)
        if out.shape != (self.chunk_len, 7) or not np.isfinite(out).all():
            raise RuntimeError(f"flux3_action returned {out.shape}, finite={np.isfinite(out).all()}")
        self.served[path] += 1
        if self.served[path] == 1:
            log.info("flux3_action: first request served on the %s path", path)
        return out

    def info(self) -> dict:
        d = super().info()
        d["history"] = self.mode
        return d
