"""Flex-pi (Yan et al., UW / AI2; geyan21/flex-pi @ 20c1b2b7) -- the authors' deployment path, for the benchmark robot.

A 6.83 B multi-stream world-action model (Wan2.2-TI2V-5B video expert + 1.02 B ActionDiT in a Mixture-of-Transformers)
that co-denoises future RGB, DINOv3 features and DA3 pointmaps with the action chunk. Nothing about the model is
re-implemented here: the adapter builds it from the run's own `config.yaml` exactly as the authors' YAM deploy does
(`experiments/yam/flexpi_policy/deploy_policy.py::build_policy_from_checkpoint`: DiT bootstrap loads neutralised,
`load_checkpoint`, the saved processor for normalisation) and calls the authors' `infer_action`. What the authors' YAM
code cannot do for this robot -- 2 physical cameras instead of 3, a 7-D joint state instead of 32-D EEF, no depth
sensor -- is supplied here, following `models/flexpi/sanity_check.py` section D and the training loader
(`RobotVideoDataset`) line for line:

  RGB      each real camera: uint8 HWC RGB -> /255 -> torchvision resize to its layout slot (cam_high 256x320,
           cam_right_wrist 224x224; bilinear + antialias, a SQUASH, no crop) -> (x - 0.5)/0.5.
  BLACK    the layout's third slot (cam_left_wrist) was an all-black synthetic camera in training
           (`synthetic_zero_cams`): RGB -1, depth 0, identity K. The adapter fabricates it itself; whatever the client
           sends for cam_left_wrist never reaches the model (the server only hands over primary + wrist).
  DEPTH    none was recorded; training used Depth Anything 3 METRIC-LARGE on the native 640x480 frames with the
           estimated K (fx = fy = 394.223, `camera_intrinsics.json`; `scripts/da3_depth/label_depth.py`). Here the
           same network runs on the live frames, geometry identical to the labeller: the frame is resized to the
           intrinsics' own 480x640 grid (the labeller's bilinear + antialias kernel), ImageNet-normalised,
           reflect-padded to a multiple of 14, DA3 forward (bf16 autocast, as DA3's API runs it), scaled by
           ((fx + fy) / 2) / 300 with K at that 480x640 grid, metres -> uint16 mm (round, clamp) exactly as the
           labeller stores it, then the loader's linspace-endpoint NEAREST resize to the depth grid of each slot.
  K        per slot, endpoint-rescaled to the DEPTH grid exactly as `_load_layout_intrinsics_for_dir` does
           (s = (dst - 1) / (src - 1)), identity for the synthetic slot, stacked in the layout's slot order.
           Not the authors' YAM deploy's plain ratio w/W (it differs from the loader by 0.1-0.5 %).
  STATE    7-D absolute (6 rad + gripper m) -> the saved processor: ConcatLeftAlign + q01/q99 normaliser.
  TEXT     the four training instructions' precomputed umT5 embeddings (the run's `text_embeds_cache`), padding
           zeroed and an all-ones mask -- what the loader fed the model and what `encode_prompt` returns. The
           11.4 GB umT5 encoder is not loaded; an instruction outside the cache is a hard error.
  OUTPUT   infer_action(action_horizon=32, num_video_frames=9, 4 Euler steps, seed) -> (32, 7) normalised ->
           ConcatLeftAlign^-1 + q01/q99^-1 -> absolute -> the FIRST 30 rows. The horizon must be a multiple of 4
           (BLOCKERS D-FP2, as FastWAM's D-FW1); the robot executes 30, so the adapter returns (30, 7).

Regimes: "full_joint" (default; the authors' deployed mode -- every stream present and generated, their headline
real-robot result and `serve_flexpi_yam.sh` default) or "action_only" (joint_* false, presence left at the trained
default, i.e. still depth-anchored). Seeded: every call draws its noise from `seed` (the run's seed, 1000, unless
given), so the same observation gives the same chunk. `torch_compile` reproduces the authors' serve defaults
(reduce-overhead, scope "loop", attn "auto", encoder CUDA graphs) at a multi-minute first-call compile; off by default.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from .base import PolicyAdapter

log = logging.getLogger("vla_bench.flexpi")

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
REGIMES = {
    "full_joint": dict(joint_video=True, joint_dino=True, joint_pointmap=True,
                       present_video=True, present_dino=True, present_pointmap=True),
    "action_only": dict(joint_video=False, joint_dino=False, joint_pointmap=False),
}
DA3_PATCH = 14
DA3_METRIC_SCALE = 300.0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _hf_dir(spec: str) -> Path:
    """'hf:<repo>@<40-hex revision>' -> that snapshot in the local Hugging Face cache; a plain path is returned as is."""
    if not spec.startswith("hf:"):
        return Path(spec)
    repo, rev = spec[3:].split("@")
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE") or \
        os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    snap = Path(cache) / ("models--" + repo.replace("/", "--")) / "snapshots" / rev
    if not (snap / "model.safetensors").is_file():
        raise FileNotFoundError(f"{repo}@{rev} is not in the Hugging Face cache ({snap}). On the workstation run:  "
                                f"hf download {repo} --revision {rev} --include model.safetensors && "
                                f"hf download {repo} --revision {rev} --include config.json")
    return snap


def endpoint_K(entry: dict, h: int, w: int):
    """(3, 3) K for a (h, w) grid: RobotVideoDataset._load_layout_intrinsics_for_dir / label_depth.K_at_grid."""
    import torch
    sx = (w - 1) / (float(entry["width"]) - 1) if float(entry["width"]) > 1 else 1.0
    sy = (h - 1) / (float(entry["height"]) - 1) if float(entry["height"]) > 1 else 1.0
    fx, fy, cx, cy = (float(entry[k]) for k in ("fx", "fy", "cx", "cy"))
    return torch.tensor([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]], dtype=torch.float32)


def endpoint_nearest(depth, h: int, w: int):
    """RobotVideoDataset._decode_per_cam_depth's linspace-endpoint nearest resize of a [..., H, W] uint16 map."""
    import torch
    if tuple(depth.shape[-2:]) == (h, w):
        return depth
    src_h, src_w = depth.shape[-2:]
    ys = torch.linspace(0, src_h - 1, h, device=depth.device).round().long()
    xs = torch.linspace(0, src_w - 1, w, device=depth.device).round().long()
    return depth.to(torch.int32)[..., ys[:, None], xs[None, :]].to(torch.uint16)


class DA3Metric:
    """scripts/da3_depth/label_depth.py::DA3DepthModule, line for line, with the network built from depth_anything_3's
    own registry config + the snapshot's model.safetensors (strict) instead of `depth_anything_3.api`, whose import
    pulls in export dependencies (moviepy, trimesh, ...) that inference never touches. The forward is what
    `DepthAnything3.forward` does: bf16 autocast (A100) around the network, no_grad."""

    def __init__(self, weights_dir, device):
        import torch
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY
        from safetensors.torch import load_file
        weights_dir = Path(weights_dir)
        name = json.loads((weights_dir / "config.json").read_text())["model_name"]
        if "DA3METRIC" not in name.upper() or "NESTED" in name.upper():
            raise ValueError(f"expected DA3METRIC weights (the labeller's), got model_name={name!r}")
        net = create_object(load_config(MODEL_REGISTRY[name]))
        sd = load_file(str(weights_dir / "model.safetensors"))
        net.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}, strict=True)
        if any(not k.startswith("model.") for k in sd):
            raise ValueError("unexpected non-'model.' keys in the DA3 checkpoint")
        self.net = net.to(device).eval()
        self.device = torch.device(device)
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.autocast = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.name = name

    def __call__(self, images, K):
        """(B, 3, H, W) in [0, 1] + (B, 3, 3) K at (H, W) -> (B, 1, H, W) metric depth in metres."""
        import torch
        with torch.no_grad():
            x = (images - self.mean) / self.std
            _, _, H, W = x.shape
            ph, pw = (DA3_PATCH - H % DA3_PATCH) % DA3_PATCH, (DA3_PATCH - W % DA3_PATCH) % DA3_PATCH
            if ph or pw:
                x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
            with torch.autocast(device_type=x.device.type, dtype=self.autocast):
                raw = self.net(x.unsqueeze(1), None, None, [], False, False, "saddle_balanced")
            depth = raw["depth"][:, :, :H, :W]
            focal = (K[:, 0, 0] + K[:, 1, 1]) / 2.0
            return depth * (focal.view(-1, 1, 1, 1) / DA3_METRIC_SCALE)


class FlexPiAdapter(PolicyAdapter):
    """(30, 7) absolute joint targets from Flex-pi's infer_action, with live DA3 depth and a black third camera."""

    def __init__(self, checkpoint: str, run_dir: str, text_embed_cache: str, intrinsics_json: str,
                 da3_weights: str, da3_src: str | None = None, device: str = "cuda:0",
                 chunk_len: int = 30, exec_len: int = 30, regime: str = "full_joint",
                 num_inference_steps: int = 4, seed: int | None = None, torch_compile: bool = False,
                 dataset_stats: str | None = None):
        if regime not in REGIMES:
            raise ValueError(f"regime must be one of {list(REGIMES)}, got {regime!r}")
        self.name = f"flexpi:{checkpoint}"
        self.checkpoint, self.run_dir = str(checkpoint), Path(run_dir)
        self.text_embed_cache, self.intrinsics_json = Path(text_embed_cache), Path(intrinsics_json)
        self.da3_weights_spec, self.da3_src = str(da3_weights), da3_src
        self.device_str, self.regime = str(device), regime
        self.chunk_len, self.exec_len = int(chunk_len), int(exec_len)
        self.steps, self.seed_arg, self.torch_compile = int(num_inference_steps), seed, bool(torch_compile)
        self.stats_path = Path(dataset_stats) if dataset_stats else self.run_dir / "dataset_stats.json"
        self.pol = None
        self._ctx: dict[str, tuple] = {}
        self.last_timing: dict[str, float] = {}

    # ------------------------------------------------------------------------------------------ loading
    def warmup(self) -> None:
        if self.pol is not None:
            return
        if self.da3_src and self.da3_src not in sys.path:
            sys.path.insert(0, self.da3_src)
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from flexpi.composite_layouts import get_layout
        from flexpi.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        from flexpi.per_cam_compose import compose_from_per_cam

        t0 = time.time()
        self.torch = torch
        self.compose = compose_from_per_cam
        trained = OmegaConf.load(self.run_dir / "config.yaml")
        dcfg = trained.data.train
        # The authors' deploy (build_policy_from_checkpoint): the checkpoint is the source of truth for every DiT
        # parameter, so the Wan2.2 DiT / ActionDiT bootstrap loads are skipped; the frozen VAE and DINOv3 still load.
        mcfg = OmegaConf.create(OmegaConf.to_container(trained.model, resolve=True))
        mcfg.action_dit_pretrained_path = None
        mcfg.skip_dit_load_from_pretrain = True
        mcfg.load_text_encoder = False                        # as trained: the instructions come from the cache
        if int(mcfg.action_dit_config.action_dim) != 7 or int(mcfg.proprio_dim) != 7:
            raise RuntimeError(f"flexpi: expected a 7-D action/proprio model, got {mcfg.action_dit_config.action_dim}"
                               f"/{mcfg.proprio_dim}")
        pol = instantiate(mcfg, model_dtype=torch.bfloat16, device=self.device_str)
        payload = pol.load_checkpoint(self.checkpoint)        # strict_shape=True; mot loads strict=False, so check:
        for key, mod in [("mot", pol.mot), ("proprio_encoder", pol.proprio_encoder)] + \
                [(k, getattr(pol, k)) for k in pol._dino_ckpt_keys() + pol._mode_ckpt_keys()]:
            want, got = set(mod.state_dict()), set(payload.get(key, {}))
            if want != got:
                raise RuntimeError(f"flexpi: checkpoint '{key}' keys differ from the model: missing "
                                   f"{sorted(want - got)[:5]}, unexpected {sorted(got - want)[:5]}")
        self.train_step = int(payload.get("step") or -1)
        del payload
        self.pol = pol.to(self.device_str).eval()
        self.pol.prepare_for_inference(
            torch_compile=self.torch_compile, torch_compile_mode="reduce-overhead", torch_compile_scope="loop",
            attn_backend="auto" if self.torch_compile else "sdpa", encoder_cuda_graph=self.torch_compile)

        proc = instantiate(dcfg.processor).eval()
        proc.set_normalizer_from_stats(load_dataset_stats_from_json(str(self.stats_path)))
        self.proc = proc

        # geometry, all from the trained config: layout slots (RGB), shape_meta.depth (depth grid + K), synthetic cams
        self.layout_kwargs = self.pol._layout_kwargs()
        layout = get_layout(str(dcfg.concat_multi_camera))
        if str(dcfg.concat_multi_camera) != str(mcfg.composite_layout):
            raise RuntimeError(f"flexpi: data layout {dcfg.concat_multi_camera} != model layout {mcfg.composite_layout}")
        kmap = self.layout_kwargs["slot_key_map"]
        self.slot_cams = [kmap[s.key] for s in layout.cam_slots()]                       # slot order
        self.rgb_hw = {kmap[s.key]: tuple(s.src_hw) for s in layout.cam_slots()}
        self.synthetic = list(dcfg.synthetic_zero_cams or [])
        disk = [m.key for m in dcfg.shape_meta.images]
        self.depth_hw = {m.key: (int(m.shape[-2]), int(m.shape[-1])) for m in dcfg.shape_meta.depth}
        if disk != ["cam_high", "cam_right_wrist"] or self.synthetic != ["cam_left_wrist"] or \
                sorted(disk + self.synthetic) != sorted(self.slot_cams):
            raise RuntimeError(f"flexpi: camera contract changed (disk {disk}, synthetic {self.synthetic}, "
                               f"slots {self.slot_cams}); this adapter serves cam_high + cam_right_wrist + black left")
        self.cam_of = {"primary": "cam_high", "wrist": "cam_right_wrist"}
        intr = json.loads(self.intrinsics_json.read_text())
        self.intr = {c: intr[c] for c in disk}
        self.da3_hw = {c: (int(intr[c]["height"]), int(intr[c]["width"])) for c in disk}   # the labeller's grid
        K = [endpoint_K(self.intr[c], *self.depth_hw[c]) if c in disk else torch.eye(3, dtype=torch.float32)
             for c in self.slot_cams]
        self.K = torch.stack(K, 0).to(self.device_str)                                     # [3, 3, 3] slot order
        self.K_da3 = {c: endpoint_K(self.intr[c], *self.da3_hw[c]).to(self.device_str) for c in disk}  # identity rescale
        self.num_frames = int(dcfg.num_frames)
        self.ratio = int(dcfg.action_video_freq_ratio)
        self.horizon = self.num_frames - 1                                                  # 32
        self.n_video = (self.num_frames - 1) // self.ratio + 1                              # 9
        if self.chunk_len > self.horizon:
            raise ValueError(f"chunk_len {self.chunk_len} > the model horizon {self.horizon}")
        self.seed = int(self.seed_arg if self.seed_arg is not None else trained.seed)
        self.da3 = DA3Metric(_hf_dir(self.da3_weights_spec), self.device_str)
        self.load_seconds = time.time() - t0
        log.info("flexpi loaded in %.1f s: step %d, regime %s, %d Euler steps, seed %d, compile %s, horizon %d -> "
                 "first %d, slots %s (synthetic %s), depth grid %s, DA3 %s at %s, K fx %s", self.load_seconds,
                 self.train_step, self.regime, self.steps, self.seed, self.torch_compile, self.horizon,
                 self.chunk_len, self.slot_cams, self.synthetic, self.depth_hw, self.da3.name, self.da3_hw,
                 [round(float(k[0, 0]), 4) for k in self.K])
        if self.torch_compile:   # pay the compile at startup, not on the first robot query (the authors' warm-up)
            t1 = time.time()
            first_task = self._first_cached_task()
            z = np.zeros((480, 640, 3), np.uint8)
            for _ in range(2):
                self.predict({"primary": z, "wrist": z, "state": np.zeros(7, np.float32), "task": first_task})
            log.info("flexpi: torch.compile warm-up took %.1f s", time.time() - t1)

    def _first_cached_task(self) -> str:
        return "pick up the metal pot and place it in the basket"

    # ------------------------------------------------------------------------------------------ inputs
    def _context(self, task: str):
        if task not in self._ctx:
            h = hashlib.sha256(DEFAULT_PROMPT.format(task=task).encode("utf-8")).hexdigest()
            p = self.text_embed_cache / f"{h}.t5_len128.wan22ti2v5b.pt"
            if not p.exists():
                raise FileNotFoundError(f"flexpi: no cached umT5 embedding for task {task!r} ({p}); the model is served "
                                        f"without its text encoder and knows only the four training instructions")
            payload = self.torch.load(p, map_location="cpu")
            ctx, mask = payload["context"].clone(), payload["mask"].bool()
            ctx[~mask] = 0.0                                  # RobotVideoDataset._get: zero the padding ...
            self._ctx[task] = (ctx.unsqueeze(0).to(self.device_str, self.torch.bfloat16),
                               self.torch.ones_like(mask).unsqueeze(0).to(self.device_str))   # ... all-ones mask
        return self._ctx[task]

    def _u8(self, a):
        a = np.asarray(a)
        if a.ndim != 3 or a.shape[-1] != 3 or a.dtype != np.uint8:
            raise ValueError(f"flexpi expects (H, W, 3) uint8 RGB frames, got {a.shape} {a.dtype}")
        return self.torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).to(self.torch.float32) / 255.0

    def depth_mm(self, frames: dict) -> dict:
        """{cam: (3, H, W) float [0, 1]} -> {cam: (1, 1, h, w) uint16 mm at the slot's depth grid}. One DA3 batch."""
        torch = self.torch
        cams = list(frames)
        x = torch.stack([torch.nn.functional.interpolate(frames[c][None].to(self.device_str), size=self.da3_hw[c],
                                                         mode="bilinear", align_corners=False, antialias=True)[0]
                         if tuple(frames[c].shape[-2:]) != self.da3_hw[c] else frames[c].to(self.device_str)
                         for c in cams])
        d = self.da3(x, torch.stack([self.K_da3[c] for c in cams]))[:, 0]                  # (B, 480, 640) metres
        mm = (d * 1000.0).round().clamp(0, 65535).to(torch.int32).to(torch.uint16)
        return {c: endpoint_nearest(mm[i], *self.depth_hw[c])[None, None] for i, c in enumerate(cams)}

    def build(self, obs: dict) -> dict:
        """The exact tensors infer_action receives for one observation (also used by the input-verification job)."""
        torch = self.torch
        import torchvision.transforms.functional as TF
        state = np.asarray(obs["state"], np.float32).reshape(-1)
        if state.size != 7:
            raise ValueError(f"flexpi needs the 7-D right-arm state, got {state.size}")
        frames = {self.cam_of[k]: self._u8(obs[k]) for k in ("primary", "wrist")}
        per_cam = {}
        for c in self.slot_cams:
            h, w = self.rgb_hw[c]
            if c in frames:
                t = TF.resize(frames[c], size=[h, w], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
                per_cam[c] = ((t - 0.5) / 0.5)[None].to(self.device_str, torch.bfloat16)   # Normalize(0.5, 0.5)
            else:                                                                           # synthetic: black = -1
                per_cam[c] = torch.full((1, 3, h, w), -1.0, dtype=torch.float32).to(self.device_str, torch.bfloat16)
        input_image = self.compose({k: v.unsqueeze(2) for k, v in per_cam.items()}, **self.layout_kwargs).squeeze(2)
        t0 = time.perf_counter()
        per_cam_depth = self.depth_mm(frames)
        for c in self.synthetic:                             # zeros shaped like the first real depth camera's
            per_cam_depth[c] = torch.zeros_like(per_cam_depth[self.slot_cams[0]])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_timing["da3_ms"] = (time.perf_counter() - t0) * 1000.0
        p = self.proc
        sb = p.action_state_merger.forward(p.normalizer.forward(p.action_state_transform(
            {"state": {"default": torch.from_numpy(state)[None]}})))
        ctx, mask = self._context(str(obs.get("task") or ""))
        return {"per_cam": per_cam, "input_image": input_image, "per_cam_depth": per_cam_depth,
                "camera_intrinsics": self.K, "proprio": sb["state"], "context": ctx, "context_mask": mask}

    def denorm(self, a_norm) -> np.ndarray:
        x = a_norm.float().cpu()
        x = x[None] if x.ndim == 2 else x
        b = self.proc.action_state_merger.backward({"action": x})
        return self.proc.normalizer.backward({"action": b["action"]})["action"]["default"][0].numpy()

    # ------------------------------------------------------------------------------------------ inference
    def predict(self, obs: dict) -> np.ndarray:
        if self.pol is None:
            self.warmup()
        torch = self.torch
        inp = self.build(obs)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.pol.infer_action(
                prompt=None, context=inp["context"], context_mask=inp["context_mask"],
                input_image=inp["input_image"], action_horizon=self.horizon, num_video_frames=self.n_video,
                proprio=inp["proprio"], negative_prompt="", text_cfg_scale=1.0, num_inference_steps=self.steps,
                sigma_shift=None, seed=self.seed, rand_device="cpu", tiled=False,
                camera_intrinsics=inp["camera_intrinsics"], per_cam=inp["per_cam"],
                per_cam_depth=inp["per_cam_depth"], return_stream_latents=False, **REGIMES[self.regime])
        a = self.denorm(out["action"])                                                     # (32, 7) absolute
        self.last_timing["model_ms"] = (time.perf_counter() - t0) * 1000.0
        if a.shape != (self.horizon, 7) or not np.isfinite(a).all():
            raise RuntimeError(f"flexpi returned {a.shape}, finite={np.isfinite(a).all()}")
        self.calls = getattr(self, "calls", 0) + 1
        if self.calls in (1, 2) or self.calls % 50 == 0:
            peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else float("nan")
            log.info("flexpi call %d: DA3 %.1f ms + model %.1f ms; torch peak allocated %.2f GiB", self.calls,
                     self.last_timing["da3_ms"], self.last_timing["model_ms"], peak)
        return np.ascontiguousarray(a[: self.chunk_len], dtype=np.float32)                 # execute the first 30

    def info(self) -> dict:
        d = super().info()
        d.update({"policy_type": "flexpi", "regime": self.regime, "num_inference_steps": self.steps,
                  "seed": getattr(self, "seed", self.seed_arg), "torch_compile": self.torch_compile,
                  "model_horizon": getattr(self, "horizon", 32),
                  "cameras": {"primary": "cam_high (top slot 256x320)", "wrist": "cam_right_wrist (224x224)",
                              "cam_left_wrist": "synthetic black slot (RGB -1, depth 0, identity K)"},
                  "depth": "DA3METRIC-LARGE on the live frames at 480x640, K fx=fy=394.223 (estimated)"})
        return d
