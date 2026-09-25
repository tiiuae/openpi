# lingbot_v2

LingBot-VLA 2.0 (Qwen3-VL-4B VLM + 32-expert / top-4 MoE + a 36-layer flow-matching action expert,
10 denoising steps) on the right-arm 7-D, 2-camera embodiment. Predicts and executes **30 actions per
call**. Source: `Robbyant/lingbot-vla-v2` cloned at pinned SHA `bc643d74`; the container drives the
authors' own deployment class `deploy/lingbot_vla_v2_policy.py::LingbotVLAv2Server`.

**This is the largest model of the twelve: 6.376 B parameters, 23.75 GiB of float32 safetensors on disk.**
It is built on the CPU and only then cast to bfloat16 and moved to the GPU, so budget ~50 GB of host RAM for
the load. See the gotchas for what that means for an A6000 and an Orin.

## Build

```bash
docker build -f docker/vla_bench/base/Dockerfile.torch28-cu128 -t vla-bench-base:torch28-cu128 .
docker build -f docker/vla_bench/models/lingbot_v2/Dockerfile -t vla-bench-lingbot_v2:latest .
```

## Mount and run

```bash
WEIGHTS_DIR=/opt/vla_weights HF_CACHE_DIR=$HOME/.cache/huggingface \
  docker/vla_bench/run.sh lingbot_v2 --probe
docker/vla_bench/run.sh lingbot_v2 \
  --checkpoint /models/lingbot_v2/eval_ckpt/step16000/checkpoints/global_step_16000/hf_ckpt
```

Two things must exist under the mounts, and the container fails loudly without either:

* **`/models/lingbot_v2/eval_ckpt/step16000/`** — the run's HF export **with the run's directory shape
  kept**: `lingbotvla_cli.yaml` at the root and `checkpoints/global_step_16000/hf_ckpt/` under it. The
  authors' loader reads the training config from `<ckpt>/../../../lingbotvla_cli.yaml` and builds the
  model config from it; a bare `hf_ckpt` directory on its own will not load. The sibling `model/`
  (DCP shards) and `optimizer/` (26 GB) are training state and are not used.
* **`/hf/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec…`** — Qwen3-VL-4B-Instruct's
  **config, tokenizer and processor only; no VLM weights** (12 MB). The VLM weights come from the
  checkpoint. The image passes this snapshot path explicitly because the training config records an
  absolute host path that does not exist in a container.

The depth (MoGe-2, LingBot-Depth) and DINO-Video teacher weights that `lingbotvla_cli.yaml` lists are
**not needed**: `build_depth_model` / `build_video_model` are called only from the trainer, never from
the policy. Those paths sit unused in the config.

## This model's specific gotchas

**1. Camera order is load-bearing, and it is not the dict insertion order.** Both distillation
teachers read camera index 0 only (`vision_models/module_utils.py:364,429`: `pil_images[:, :1]`), and
the canonical order comes from `data.cameras` in the training config. Index 0 must be the fixed scene
view: `camera_top ← cam_high`, then `camera_wrist_right ← cam_right_wrist`. The adapter **asserts**
this against the live `FeatureTransform` and refuses to serve if it does not hold, rather than
trusting whatever config it was handed. Swapping the two would feed the model a wrist image where it
expects the scene.

**2. It is the largest model in the set.** 6.376 B parameters; **23.75 GiB of float32 safetensors** to
read off the mount, built on the CPU and then cast to **bfloat16** on the GPU. The measured resident and peak
VRAM are in `VLA-SOTA/results/deployment/lingbot_v2/deployment_guide.md` — do not guess them from the disk
size. Practical consequences: budget ~50 GB of **host RAM** for the load; a 48 GB RTX A6000 has the VRAM for
it; a **Jetson Orin is ruled out on memory alone** (and there is no torch 2.8 cu128 build for it), before
latency is even considered. Reading 23.75 GiB off a network filesystem is also why `HEALTHCHECK` has a 900 s
start period and why `--probe` takes minutes, not seconds.

**3. The canonical-slot assumption.** Our 6 right-arm joints occupy LingBot's canonical
`arm.position` dims **0-5** (the slot RoboTwin's *left* arm uses) and the gripper occupies
`effector.position` dim 0; the remaining dims are zero and masked out. The 20 pretraining embodiments
are not published, so whether single right arms were pretrained at 0-5 or 7-12 is not knowable from
the release — the joint masks make either choice self-consistent and post-training adapted the
projections. This is recorded as an open item (`BLOCKERS.md` B13.1). If it ever changes it is a
one-line edit to the baked `configs/robot_configs/right7_2view.yaml` **plus a retrain**, not a
re-deploy.

**4. The two baked config files are dataset-specific.** `configs/robot_configs/right7_2view.yaml`
(the 7-D → 55-D mapping, with `subtract_state: False` on both action entries because our actions are
absolute joint targets) and `assets/norm_stats/right7_2view.json` (`meanstd` over all 139,662 training
frames) are the only non-upstream files in the image. `FeatureTransform` resolves both by
**cwd-relative** path, which is why the adapter `chdir()`s into `/opt/src`. A different dataset needs
a different norm-stats file, regenerated with the repo's own `scripts/compute_norm_stats.py`.

**5. bfloat16, eager attention, no `torch.compile` -- and turn compile on for a robot.** The authors'
deploy loader rewrites the action expert's attention to `eager` (FlexAttention is slow uncompiled) and
defaults to bf16; the image does the same, uncompiled, because that is how the benchmark's reference
numbers were recorded and it is the path verified below. `torch.compile` is worth **3x**: measured on
one A100 (SLURM 474923, `models/lingbot_v2/eval_probe/verify_and_latency.json`), bf16 + eager is
**884 ms** median per call and bf16 + eager + `torch.compile` is **293 ms** median, after a **107 s**
one-time graph capture on the first call. Enable it with `"use_compile": true` in
`VLA_BENCH_ADAPTER_KWARGS` and budget the capture into startup (it lands on the first request, so send a
warm-up request before the robot needs an answer). The compiled path was measured in the host venv; it
has **not** been replayed through this image, so treat its accuracy as unverified here. The image has
the C toolchain and Python headers that inductor needs.

**6. It is not bit-reproducible, even seeded -- by design of its MoE kernel, not because of Docker.**
The adapter seeds torch at warmup, so the flow-matching noise is the same on every run, but the
arithmetic is not: at inference every one of the 36 action-expert MoE layers runs
`lingbotvla/ops/robby_moe.py::robby_moe_forward`, which sums each token's four expert outputs into an fp32
buffer with `tl.atomic_add` in whatever order the GPU schedules them. Calling that kernel 50 times on
identical inputs gives a different result on 49 of 49 repeats, in ~35 % of elements, by fp32 rounding
(3e-8). Through 36 layers x 10 denoising steps this surfaces as **bfloat16-rounding flips** of the
normalised action in about a third of the returned values (72 % of them exactly one bf16 step, the rest a
few; never more than 4.96e-3 rad after un-normalisation). Two runs of the *same* path differ by exactly as much
as the container differs from the host, so compare this model on aggregate metrics, never element-wise.

## Verification -- **verified** against the recorded reference (2026-09-24)

`--probe` against the real mounted checkpoint (`eval_ckpt/step16000/.../global_step_16000/hf_ckpt`, with
the run's directory shape so `lingbotvla_cli.yaml` is found three levels up): **ok**, `(30, 7)`, 130.9 s
to load, 4434 ms first call.

The 170-query replay of the five reference episodes (`39,102,120,163,167`, rate 20) through this image,
driven by the same `deploy_eval/client.py` that produced the benchmark's numbers, reproduces the recorded
reference (`VLA-SOTA/results/deploy_eval/lingbot_v2/metrics.json`) to **+0.12 %** on joint MAE:

| aggregate metric | reference | container | delta |
|---|---:|---:|---:|
| `mae_joints_rad` | 0.0111865 | 0.0111999 | +0.12 % |
| `rmse_joints_rad` | 0.0142972 | 0.0143145 | +0.12 % |
| `mae_gripper_m` | 0.000292925 | 0.000292945 | +0.01 % |
| `mae_normalised` | 0.0331887 | 0.0332211 | +0.10 % |
| `traj_mae_joints_rad` | 0.0110063 | 0.0110208 | +0.13 % |

Per episode the deltas are -0.15 % .. +0.44 %. It is **not** bit-exact, and because this adapter seeds
its sampler that was investigated rather than accepted (gotcha 6). Five fresh-load runs of the same 170
queries (the reference; two replays of the validated host venv on the same A100; two of this image)
differ **pairwise** in 11,250-12,665 of the 35,700 predicted values, always by bfloat16 rounding of the
normalised action (never more than 4.96e-3 rad), and on aggregate by at most 0.29 %. Host-vs-host and container-vs-container pairs differ
exactly as much as container-vs-host. The validated host path does not reproduce its own recorded
reference bit-exactly either (+0.014 % and +0.116 % joint MAE on its two replays), so the container is
as close to the reference as any path gets. The package set in the image matches the validated venv
exactly apart from `ninja`/`setuptools`/`wheel`.

Command (benchmark workspace):
`GPU=3 openpi_integration/tools/verify_container.sh lingbot_v2 /models/lingbot_v2/eval_ckpt/step16000/checkpoints/global_step_16000/hf_ckpt 39,102,120,163,167 8853`.
Server-side latency in the replay: 894 ms p50, against 900 ms p50 in the recorded reference (uncompiled).

Image `vla-bench-lingbot_v2:latest`: **10.50 GB** (`docker image inspect`, 10,495,227,172 bytes), on
`vla-bench-base:torch28-cu128` (7.65 GB).
