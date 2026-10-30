r"""Adapter for OpenWAM (OpenWAM/OpenWAM @ 4b3814a8, Stanford, AGPL-3.0), model id `openwam`.

A world-action model: a Wan2.2-TI2V-5B-shaped causal video expert (OpenWAM-Pretraining) plus a
2 B action expert (dual-expert MoT, program `joint`). It predicts 32 actions per call (4 latent
frames x 8 rows at 30 Hz); this adapter returns the FIRST 30 (BLOCKERS D-OW2), 7-D right-arm
absolute joint targets.

Everything model-side is the authors' own code, not a re-implementation:

* Loading: `load_libero_policy_runtime`, the loader of the authors' closed-loop policy evaluator
  (`scripts/run_libero_policy.py`), on the checkpoint's own `resolved_config.yaml` with
  `frontend_encode_mode=lingbot_streaming_vae` and `startup_model_obs_frames=1` (their defaults).
  The only config overrides are paths: where the base assets (Wan2.2 VAE, UMT5-xxl text encoder,
  tokenizer) live on this machine. Without a `transformer/` export in the checkpoint directory,
  `backbone.load_reference_core_weights=false` builds the pipeline without one and
  `model_state.pt` provides every tensor (the authors' loader refuses a missing key; D-OW16).
* One request = the authors' episode-start transaction (`LiberoPolicyPlanner.start` + `_predict`
  at chunk 0): `runner.reset`, `visual_tower.reset_runtime_state()`, `seed_everywhere(seed + 0)`,
  the streaming Wan-VAE frontend on ONE observed frame per camera
  (`_prepare_policy_visual_outputs`), `runner.infer_prepared_step` (the checkpoint's `inference`
  section: 20 + 20 denoising steps, video CFG 5.0, frame chunk 4, attention window 30, KV cache),
  `pipeline.action_adapter.to_source` (it inverts the target's final `normalization`, `none` here),
  then the authors' `denormalize_joint_positions` with the checkpoint's
  `joint_position_normalization` (quantiles): the absolute-joint target family normalises the joint
  channels with it, and the LIBERO rollout path never needs it (its targets are raw EEF deltas), so
  `to_source` alone would return the normalised values. The parameters run in bf16
  (`param_dtype`), the precision of the training forward pass (FSDP `MixedPrecisionPolicy`
  `param_dtype=bfloat16`); `model_state.pt` holds the fp32 masters, which the authors' loader would
  leave in fp32 (D-OW18). Buffers keep their dtype, as under FSDP. Why not the authors' rolling
  history: their maintained loop commits every executed control AND the observation of every
  executed control through the streaming VAE before it replans (`LiberoPolicyPlanner.observe`;
  docs/rollout_contracts.md: "Do not substitute speculative actions ... or later images for
  missing observations"). The robot client sends one frame per request and never the executed
  actions, so that history cannot be built faithfully; the episode-start transaction is the part
  of their loop this wire feeds exactly. Nothing is carried between requests (each one resets and
  reseeds), so episode boundaries need no detection and `reset()` has nothing to clear (D-OW15).

Observation pipeline, reproducing the training data (BLOCKERS D-OW3, D-OW4, D-OW13, D-OW14):

* cam_high -> the agent-view slot `observation.images.cam_high` (left 128x128 of the 128x256
  canvas), cam_right_wrist -> `observation.images.cam_right_wrist` (right half); checked against
  the checkpoint's view layout at load.
* Each native frame is letterboxed with the authors' `fit_frames(..., 128, 128, "letterbox_pad")`
  on the CPU (bilinear + antialias to 128x96, 16-px black bars) and rounded to uint8, exactly as
  make_openwam_v21.py stored the training videos; the authors' canonicaliser divides by 255.
* Latents: the Wan2.2 VAE runs in **fp32** (the training latents were encoded in fp32, D-OW13;
  the coordinator's binding decision) on `vae_device`, with TF32 off for the encode (cuDNN
  convolutions default to TF32 on Ampere, 0.1 % from fp32; D-OW14). The authors' `_encode_chunk`
  is unchanged (RGB -> [-1, 1] in fp32, posterior mean, the VAE's latent normalisation). The result
  is rounded to fp16 and back (`latent_storage_dtype`), as the training latents were stored, so the
  model sees the values it was trained on.
* State: the raw 7-D right-arm joints as float32 (`state_encoding: identity`), shape [1, 1, 7].
* Text: the task string through the same UMT5-xxl, on the CPU in bf16 as the training config's
  `reference_assets_device_policy: cpu_offload` ran it, plus the blank prompt for video CFG; the
  authors' frontend caches the embedding per prompt. On a CPU without bf16 units that takes
  minutes per prompt, so `text_cache` (repo/scripts/openwam/make_openwam_text_cache.py: the same
  function on the same files) preloads the four training instructions and the blank prompt into
  that cache (D-OW17); any other prompt is encoded online on its first request.

Fail-closed checks at load: the base-asset files (sizes; sha256 of the small files), the
checkpoint's de-normalisation quantiles against `norm_stats` (the dataset's
`meta/openwam_norm_stats.json`), the camera layout, horizon 32 / dim 7, the text cache's encoder
identity, and no missing checkpoint key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .base import PolicyAdapter

log = logging.getLogger(__name__)

OPENWAM_SHA = "4b3814a82268f3523df5fcdadcbeb2c021ac7737"  # gitleaks:allow (a git commit id)
VAE_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
VAE_REV = "b8fff7315c768468a5333511427288870b2e9635"  # gitleaks:allow (a git commit id)
TEXT_REPO = "robbyant/lingbot-va-base"
TEXT_REV = "68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c"  # gitleaks:allow (a git commit id)
CAM_HIGH = "observation.images.cam_high"
CAM_WRIST = "observation.images.cam_right_wrist"
TRAINING_TASKS = (
    "pick up the metal pot and place it in the basket",
    "pick up the green cup and place it in the basket",
    "pick up the blue cup and place it in the basket",
    "pick up the small mug and place it in the basket",
)
# The base assets training loaded, at the pinned revisions: (bytes, sha256 or None). sha256 is
# checked for the small files, the large ones by size (public file hashes, not secrets).
ASSET_FILES = {
    "vae": {
        "config.json": (
            1701,
            "d996c340fe9a7df5d7371f76a7d8d6956f6c98256080074d8434fa5eeac11360",  # gitleaks:allow
        ),
        "diffusion_pytorch_model.safetensors": (2818777808, None),
    },
    "text_encoder": {
        "config.json": (
            855,
            "a2bcb24699f6c009a2427432bdd483ef8b2b42a712abc9503759cdc77d171f07",  # gitleaks:allow
        ),
        "model.safetensors.index.json": (
            22476,
            "31c4c7bcce679eaa0dd4667462394ddb013dc2f748e0bffc893dc9146a320dab",  # gitleaks:allow
        ),
        "model-00001-of-00003.safetensors": (4935812536, None),
        "model-00002-of-00003.safetensors": (4983103192, None),
        "model-00003-of-00003.safetensors": (1442935480, None),
    },
    "tokenizer": {
        "special_tokens_map.json": (
            7079,
            "456b58fd240a06c743a7c2cf8008bec501240d68ebd1fc4018ea569505fea270",  # gitleaks:allow
        ),
        "spiece.model": (
            4548313,
            "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458",  # gitleaks:allow
        ),
        "tokenizer.json": (
            16837459,
            "20a46ac256746594ed7e1e3ef733b83fbc5a6f0922aa7480eda961743de080ef",  # gitleaks:allow
        ),
        "tokenizer_config.json": (
            61758,
            "1d8d2a216bf8e70ac15b7ddcea566c4dd0433c024b39a58ca5e4c66bd78defbd",  # gitleaks:allow
        ),
    },
}
ASSET_SOURCES = {
    "vae": (VAE_REPO, VAE_REV),
    "text_encoder": (TEXT_REPO, TEXT_REV),
    "tokenizer": (TEXT_REPO, TEXT_REV),
}
LAYOUT = {CAM_HIGH: ("image", 0, 0, 128, 128), CAM_WRIST: ("wrist_image", 0, 128, 128, 128)}


def _sha256(path: Path) -> str:
    """Hex sha256 of a file, read in 8 MiB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def resolve_asset_dir(spec: str, component: str) -> Path:
    """A base-asset directory from a plain path or 'hf:<repo>@<40-hex revision>/<subdir>'.

    Args:
        spec: a directory, or a pinned Hub snapshot sub-directory in the local Hugging Face cache.
        component: 'vae', 'text_encoder' or 'tokenizer' (selects the repo training used).

    Returns:
        The directory.

    Raises:
        SystemExit: the pinned snapshot is not in the cache (the message names the command).
        ValueError: the spec names another repo or revision than training used.
    """
    if not spec.startswith("hf:"):
        return Path(spec)
    # 'hf:Wan-AI/Wan2.2-TI2V-5B-Diffusers@<rev>/vae': the repo id holds a '/', so split at '@'.
    repo, _, tail = spec[3:].partition("@")
    rev, _, sub = tail.partition("/")
    want = ASSET_SOURCES[component]
    if (repo, rev) != want:
        raise ValueError(f"openwam: {component} must come from {want[0]}@{want[1]}, got {spec}")
    from .hub_pins import pinned_snapshot

    return pinned_snapshot(repo, rev) / sub


def check_asset_dir(path: Path, component: str) -> dict[str, int]:
    """Refuse a base-asset directory whose files differ from the pinned ones.

    Args:
        path: the directory.
        component: 'vae', 'text_encoder' or 'tokenizer'.

    Returns:
        {file name: bytes} of the checked files.

    Raises:
        RuntimeError: a file is missing, has another size, or (small files) another sha256.
    """
    bad, seen = [], {}
    for name, (size, sha) in ASSET_FILES[component].items():
        f = path / name
        if not f.is_file():
            bad.append(f"{name}: missing")
            continue
        got = f.stat().st_size
        if got != size:
            bad.append(f"{name}: {got} bytes, expected {size}")
            continue
        if sha is not None and (h := _sha256(f)) != sha:
            bad.append(f"{name}: sha256 {h} != pinned {sha}")
        seen[name] = got
    if bad:
        repo, rev = ASSET_SOURCES[component]
        msg = f"openwam: {path} is not {repo}@{rev} {component}:\n  " + "\n  ".join(bad)
        raise RuntimeError(msg)
    return seen


class OpenWAMAdapter(PolicyAdapter):
    """OpenWAM dual-expert joint policy behind the benchmark's adapter interface."""

    name = "openwam"

    def __init__(
        self,
        checkpoint: str,
        vae_dir: str,
        text_encoder_dir: str,
        tokenizer_dir: str,
        norm_stats: str,
        *,
        device: str = "cuda:0",
        vae_device: str | None = None,
        vae_dtype: str = "float32",
        latent_storage_dtype: str | None = "float16",
        param_dtype: str | None = "bfloat16",
        chunk_len: int = 30,
        exec_len: int = 30,
        seed: int = 1000,
        warm_prompts: list[str] | None = None,
        use_checkpoint_transformer: bool = True,
        verify_assets: bool = True,
        text_cache: str | None = None,
    ) -> None:
        """Load the checkpoint with the authors' policy-runtime loader.

        Args:
            checkpoint: the checkpoint directory (`model_state.pt`, `resolved_config.yaml`,
                optionally `transformer/`).
            vae_dir: the Wan2.2 VAE (a directory or 'hf:<repo>@<rev>/vae').
            text_encoder_dir: the UMT5-xxl text encoder (a directory or 'hf:...').
            tokenizer_dir: its tokenizer (a directory or 'hf:...').
            norm_stats: the dataset's `openwam_norm_stats.json`; its action q01/q99 must equal
                the checkpoint's de-normalisation quantiles.
            device: the runtime device (video and action experts).
            vae_device: where the fp32 VAE runs (default: `device`).
            vae_dtype: the VAE's arithmetic, 'float32' as the training latents were encoded.
            latent_storage_dtype: round the latents through this dtype, as the training latents
                were stored ('float16'); None skips it.
            param_dtype: cast every pipeline parameter (not the buffers) to this dtype after the
                load, as the training forward ran them ('bfloat16'); None keeps the fp32 values.
            chunk_len: rows returned (30 of the 32 predicted).
            exec_len: rows executed per call.
            seed: `seed_everywhere(seed + chunk_index)`, chunk_index 0, before every request.
            warm_prompts: prompts looked up or encoded at warm-up (default: the four tasks).
            use_checkpoint_transformer: build from `<checkpoint>/transformer` when it exists (the
                authors' loader); otherwise every tensor comes from `model_state.pt`.
            verify_assets: check the base-asset files before loading.
            text_cache: a make_openwam_text_cache.py file, put into the frontend's prompt cache at
                warm-up (None: every prompt is encoded online).
        """
        import torch

        t0 = time.time()
        self.torch = torch
        self.checkpoint = Path(checkpoint)
        self.chunk_len, self.exec_len, self.seed = int(chunk_len), int(exec_len), int(seed)
        self.warm_prompts = list(TRAINING_TASKS if warm_prompts is None else warm_prompts)
        self.text_cache = text_cache
        self.text_cache_prompts: list[str] = []
        self.vae_dtype = getattr(torch, vae_dtype)
        store = None if latent_storage_dtype in (None, "", "none") else latent_storage_dtype
        self.latent_storage_dtype = None if store is None else getattr(torch, store)
        dirs = {k: resolve_asset_dir(v, k) for k, v in
                (("vae", vae_dir), ("text_encoder", text_encoder_dir),
                 ("tokenizer", tokenizer_dir))}
        self.asset_dirs = {k: str(v) for k, v in dirs.items()}
        if verify_assets:
            for k, v in dirs.items():
                check_asset_dir(v, k)
        cfg_path = self.checkpoint / "resolved_config.yaml"
        if not (self.checkpoint / "model_state.pt").is_file() or not cfg_path.is_file():
            raise FileNotFoundError(f"openwam: {self.checkpoint} needs model_state.pt and "
                                    "resolved_config.yaml")
        export = self.checkpoint / "transformer" / "config.json"
        self.uses_transformer_export = bool(use_checkpoint_transformer) and export.is_file()
        self.overrides = [
            f"backbone.pretrained_model_name_or_path={dirs['text_encoder'].parent}",
            f"backbone.vae_subdir={dirs['vae']}",
            f"backbone.text_encoder_subdir={dirs['text_encoder']}",
            f"backbone.tokenizer_subdir={dirs['tokenizer']}",
        ]
        if not self.uses_transformer_export:
            self.overrides.append("backbone.load_reference_core_weights=false")
        self._load(cfg_path, device, vae_device or device)
        self._cast_params(param_dtype)
        self._check_contract(norm_stats)
        self._pin_vae()
        self.load_seconds = time.time() - t0
        log.info("openwam loaded in %.1f s from %s (transformer export: %s); VAE %s on %s",
                 self.load_seconds, self.checkpoint, self.uses_transformer_export, self.vae_dtype,
                 self.frontend_device)

    # ----------------------------------------------------------------------------------- loading
    def _load(self, cfg_path: Path, device: str, vae_device: str) -> None:
        """The authors' closed-loop policy loader (load_libero_policy_runtime)."""
        from open_wam.evals.libero_policy_runtime import (
            LiberoPolicyLoadOptions,
            load_libero_policy_runtime,
        )

        opts = LiberoPolicyLoadOptions(
            config=str(cfg_path), checkpoint=str(self.checkpoint),
            merge_checkpoint_runtime_config=False, set_overrides=tuple(self.overrides),
            source="vla_bench openwam adapter", checkpoint_error="openwam: no model_state.pt",
            raw_window_frames=None, startup_model_obs_frames=1, startup_env_init_steps=5,
            inference_window_size=None, rollout_frame_chunk_size=None, action_only_rollout=False,
            policy_action_route="native", execute_action_steps=None,
            execute_frame_chunk_size=None, frontend_encode_mode="lingbot_streaming_vae",
            reset_policy_state_each_chunk=False, runtime_device=device, action_device=None,
            frontend_device=vae_device, decode_device=None, allow_deprecated_libero_config=False,
            allow_deprecated_frontend_encode_mode=False)
        rt = load_libero_policy_runtime(opts)
        self.pipeline, self.runner, self.config = rt.pipeline, rt.runner, rt.config
        self.runtime_device, self.frontend_device = rt.runtime_device, rt.frontend_device
        self.assets = self.pipeline.visual_tower.frontend.reference_assets
        self.component_report = {k: v for k, v in rt.component_report.items()
                                 if isinstance(v, (str, int, float, bool))}

    def _cast_params(self, name: str | None) -> None:
        """Every pipeline parameter in `name` (FSDP's forward param_dtype); buffers untouched."""
        torch = self.torch
        if name not in (None, "", "none"):
            dtype = getattr(torch, name)
            with torch.no_grad():
                for p in self.pipeline.parameters():
                    if p.is_floating_point() and p.dtype != dtype:
                        p.data = p.data.to(dtype)
        self.param_dtypes = sorted({str(p.dtype) for p in self.pipeline.parameters()})
        self.n_params = sum(p.numel() for p in self.pipeline.parameters())

    def _check_contract(self, norm_stats: str) -> None:
        """Horizon, dims, camera layout and de-normalisation quantiles must be what this feeds."""
        c = self.config
        schema = c.data.action_schema
        shape = (int(schema.action_horizon), int(schema.action_dim), int(schema.state_dim))
        if shape != (32, 7, 7) or self.chunk_len > shape[0]:
            raise RuntimeError(f"openwam: checkpoint (horizon, action, state) = {shape}")
        layout = {v.source_name: (v.canonical_name, v.top, v.left, v.height, v.width)
                  for v in c.data.view_layout}
        if layout != LAYOUT or (c.data.canonical_height, c.data.canonical_width) != (128, 256):
            raise RuntimeError(f"openwam: view layout {layout} differs from {LAYOUT}")
        target = c.data.action_target
        rep_ = getattr(target.representation, "value", target.representation)
        final = target.normalization
        if rep_ != "absolute_joint_position" or getattr(final.mode, "value", final.mode) != "none":
            raise RuntimeError(f"openwam: target {rep_} with final normalization {final.mode}")
        norm = target.joint_position_normalization
        stats = json.loads(Path(norm_stats).read_text())["action"]
        for key in ("q01", "q99"):
            got = np.asarray(getattr(norm, key), np.float32)
            ref = np.asarray(stats[key], np.float32)
            if got.shape != (7,) or not np.array_equal(got, ref):
                raise RuntimeError(f"openwam: checkpoint {key} {got.tolist()} != {norm_stats}")
        keys = ("video_num_inference_steps", "action_num_inference_steps", "frame_chunk_size",
                "attention_window_size", "use_cache", "guidance_scale")
        self.inference = {k: getattr(c.inference, k) for k in keys}
        self.norm_stats = str(norm_stats)

    def _pin_vae(self) -> None:
        """The VAE on (vae_device, vae_dtype) once; the authors' per-call policy would recast it."""
        a, torch = self.assets, self.torch
        if not a.has_vae:
            raise RuntimeError("openwam: the Wan VAE did not load")
        from open_wam.models.visual_tower.reference_assets import WanVAEStreamingWrapper

        a.vae = a.vae.to(device=self.frontend_device, dtype=self.vae_dtype)
        a.streaming_vae = WanVAEStreamingWrapper(a.vae)
        a.streaming_vae_by_key.clear()
        a._ensure_vae_runtime_device = lambda device: None  # noqa: ARG005 (pinned above)
        encode, store = a.encode_video, self.latent_storage_dtype

        def encode_video(canonical_video: torch.Tensor, *, placements: tuple | None = None,
                         reset_cache: bool = True) -> torch.Tensor:
            """The authors' encode_video in true fp32 (no TF32), then the storage rounding."""
            cudnn, matmul = torch.backends.cudnn, torch.backends.cuda.matmul
            tf32 = matmul.allow_tf32
            matmul.allow_tf32 = False
            try:
                with cudnn.flags(enabled=cudnn.enabled, benchmark=cudnn.benchmark,
                                 deterministic=cudnn.deterministic, allow_tf32=False):
                    z = encode(canonical_video, placements=placements, reset_cache=reset_cache)
            finally:
                matmul.allow_tf32 = tf32
            z = z if store is None else z.to(store).to(torch.float32)
            self.last_latents = z.detach()
            return z

        a.encode_video = encode_video

    def _preload_text_cache(self) -> None:
        """Put the text_cache embeddings into the frontend's prompt cache, under the lookup key."""
        torch, a = self.torch, self.assets
        d = torch.load(self.text_cache, map_location="cpu", weights_only=True)
        want = {"format": "openwam_text_cache_v1", "max_text_tokens": int(a.config.max_text_tokens),
                "text_encoder": {"repo": TEXT_REPO, "revision": TEXT_REV}}
        got = {k: d.get(k) for k in want}
        dtype = str(next(a.text_encoder.parameters()).dtype)
        if got != want or d.get("encoder_dtype") != dtype:
            msg = f"openwam: text_cache is {got}, {d.get('encoder_dtype')}; need {want}"
            raise RuntimeError(msg)
        shape = (1, int(a.config.max_text_tokens), int(a.config.text_dim))
        for prompt, z in d["entries"].items():
            if tuple(z.shape) != shape or not torch.isfinite(z).all():
                raise RuntimeError(f"openwam: text_cache entry {prompt!r} is {tuple(z.shape)}")
            key = ((str(prompt),), str(torch.device(self.frontend_device)), str(torch.float32),
                   shape[1])
            a.text_embedding_cache[key] = z.to(device=self.frontend_device, dtype=torch.float32)
            self.text_cache_prompts.append(str(prompt))

    # --------------------------------------------------------------------------------- inference
    def letterbox(self, frame: np.ndarray) -> np.ndarray:
        """uint8 HxWx3 RGB -> uint8 1x128x128x3 by the authors' fit_frames, rounded as trained."""
        from open_wam.data.preparation.frames import fit_frames

        a = np.asarray(frame)
        if a.ndim != 3 or a.shape[-1] != 3 or a.dtype != np.uint8:
            raise ValueError(f"openwam expects (H, W, 3) uint8 RGB, got {a.shape} {a.dtype}")
        if a.shape[:2] != (480, 640) and not getattr(self, "_warned_size", False):
            log.warning("openwam: frame %s is not the 480x640 the model trained on", a.shape)
            self._warned_size = True
        x = fit_frames(np.ascontiguousarray(a)[None], 128, 128, "letterbox_pad")
        torch = self.torch
        return torch.round(x * 255.0).clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()

    def views(self, obs: dict) -> dict[str, np.ndarray]:
        """The two camera slots as the authors' canonicaliser takes them ([1, 128, 128, 3])."""
        return {CAM_HIGH: self.letterbox(obs["primary"]), CAM_WRIST: self.letterbox(obs["wrist"])}

    def _step(self, obs: dict) -> tuple[np.ndarray, Any, Any]:
        """One episode-start transaction (LiberoPolicyPlanner.start + _predict at chunk 0)."""
        torch = self.torch
        from open_wam.evals.libero_policy_inputs import _prepare_policy_visual_outputs
        from open_wam.models.policy_variants import PolicyInferContext, PolicyTemporalGeometry
        from open_wam.utils import seed_everywhere

        state = np.asarray(obs["state"], np.float32).reshape(-1)
        if state.size != 7 or not np.isfinite(state).all():
            raise ValueError(f"openwam needs a finite 7-D right-arm state, got {state.shape}")
        task = str(obs.get("task") or "")
        dev = self.frontend_device
        views = {k: torch.from_numpy(v).to(dev) for k, v in self.views(obs).items()}
        session = self.runner.reset(task_text=(task,))
        self.pipeline.visual_tower.reset_runtime_state()
        seed_everywhere(self.seed + 0)
        inf = self.config.inference
        with torch.inference_mode():
            visual = _prepare_policy_visual_outputs(
                self.pipeline, views=views, task_text=(task,), frontend_device=self.frontend_device,
                runtime_device=self.runtime_device, use_streaming_frontend=True,
                preserve_stream_cache=False, text_context=session.text_context,
                negative_text_context=session.negative_text_context)
            context = PolicyInferContext(
                state=torch.from_numpy(state).view(1, 1, 7).to(self.runtime_device),
                task_text=(task,),
                temporal_geometry=PolicyTemporalGeometry(
                    frame_chunk_size=int(inf.frame_chunk_size),
                    attention_window_size=int(inf.attention_window_size)))
            infer_session = replace(
                self.runner.reset(task_text=session.task_text, text_context=session.text_context,
                                  negative_text_context=session.negative_text_context),
                policy_state=session.policy_state)
            step = self.runner.infer_prepared_step(session=infer_session, context=context,
                                                   visual_outputs=visual)
            if step.action_plan is None:
                raise RuntimeError("openwam: the action stage published no executable plan")
            from open_wam.data.action_normalization import denormalize_joint_positions

            plan = self.pipeline.action_adapter.to_source(step.action_plan.actions).float()
            joints = denormalize_joint_positions(
                plan, normalization=self.config.data.action_target.joint_position_normalization)
        return joints.cpu().numpy(), visual, step

    def predict(self, obs: dict) -> np.ndarray:
        """Native-wire observation -> float32 (30, 7) absolute targets (first 30 of 32)."""
        actions, _, _ = self._step(obs)
        actions = np.asarray(actions, np.float32).reshape(-1, 7)
        if actions.shape[0] != 32 or not np.isfinite(actions).all():
            ok = bool(np.isfinite(actions).all())
            raise RuntimeError(f"openwam returned {actions.shape}, finite={ok}")
        return actions[: self.chunk_len]

    def warmup(self) -> None:
        """Preload text_cache, look up or encode the warm and blank prompts, infer once."""
        torch, a = self.torch, self.assets
        t0 = time.time()
        if self.text_cache:
            self._preload_text_cache()
        for p in self.warm_prompts:
            a.encode_text((p,), device=self.frontend_device, dtype=torch.float32)
        a.encode_blank_text(batch_size=1, device=self.frontend_device, dtype=torch.float32)
        t1 = time.time()
        yy, xx = np.mgrid[0:480, 0:640]
        frame = np.stack([(xx % 256), (yy % 256), ((xx + yy) % 256)], -1).astype(np.uint8)
        state = np.asarray([0.3, 1.5, 1.2, 0.0, 0.0, 0.1, 0.02], np.float32)
        task = self.warm_prompts[0] if self.warm_prompts else ""
        obs = {"primary": frame, "wrist": frame[:, ::-1].copy(), "state": state, "task": task}
        self.predict(obs)
        self.warmup_seconds = {"prompts": round(t1 - t0, 1),
                               "first_inference": round(time.time() - t1, 1)}
        log.info("openwam warm-up: %d prompts + blank in %.1f s (%d from text_cache), first "
                 "inference %.1f s", len(self.warm_prompts), t1 - t0,
                 len(self.text_cache_prompts), time.time() - t1)

    def reset(self) -> None:
        """Nothing to clear: every request is the authors' episode-start transaction.

        Each request resets the runner session and the streaming-VAE cache and reseeds; the
        text-embedding cache is a pure function of the prompt.
        """
        return

    def info(self) -> dict:
        """Server metadata."""
        d = super().info()
        d.update({
            "policy_type": "openwam (dual_expert, program joint)", "openwam_sha": OPENWAM_SHA,
            "predicted_horizon": 32,
            "cameras": {"primary": f"cam_high -> {CAM_HIGH} (canvas left)",
                        "wrist": f"cam_right_wrist -> {CAM_WRIST} (canvas right)"},
            "letterbox": "fit_frames letterbox_pad 640x480 -> 128x96 in 128x128, uint8",
            "vae": f"{self.vae_dtype} on {self.frontend_device}, TF32 off",
            "latent_storage_dtype": str(self.latent_storage_dtype),
            "param_dtypes": self.param_dtypes, "parameters": self.n_params,
            "text_encoder": "UMT5-xxl bf16 on the CPU (cpu_offload), cached per prompt",
            "text_cache": self.text_cache, "text_cache_prompts": len(self.text_cache_prompts),
            "inference": {k: (float(v) if isinstance(v, float) else v)
                          for k, v in self.inference.items()},
            "statefulness": "none across requests: the authors' episode-start transaction",
            "seed": self.seed, "uses_transformer_export": self.uses_transformer_export,
            "asset_dirs": self.asset_dirs, "norm_stats": self.norm_stats,
            "horizon_note": "32 predicted (4 latent frames x 8); the first 30 returned"})
        return d
