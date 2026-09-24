# internvla_a15

InternVLA-A1.5 (Qwen3.5-VL backbone + flow-matching action expert) on the right-arm 7-D, 2-camera
embodiment. Predicts and executes **30 actions per call**. Source: `InternRobotics/InternVLA-A1`,
cloned at a pinned SHA during the build.

## Build

```bash
docker build -f docker/vla_bench/models/internvla_a15/Dockerfile -t vla-bench-internvla_a15:latest .
```

## Run

```bash
WEIGHTS_DIR=/opt/vla_weights HF_CACHE_DIR=$HOME/.cache/huggingface \
  docker/vla_bench/run.sh internvla_a15 --probe \
    --checkpoint /models/internvla_a15/<run>/checkpoints/015000/pretrained_model
```

The checkpoint is the run's `pretrained_model/` directory; it also supplies `stats.json` (the state
and action mean/std) and `train_config.json` (which selects the un-normalisation mode).

## Five things specific to this model

**It is a LeRobot *fork*, not a LeRobot policy.** The clone at `/opt/src` **is** the `lerobot`
distribution (version 1.0.0). `adapters/lerobot_policy.py` cannot drive it — its inference contract is
a tokenised Qwen3.5-VL chat batch and its normalisation lives in the dataset transform pipeline — so
the adapter drives the repo's own deployment backend
(`evaluation/LIBERO/policy_server/backends/policy_backend_internvla_a1_5.py`) instead.

**The transformers overlay is a build step, not a patch.** The repo carries its own copies of several
`transformers.models.*` packages under `<policy>/transformers_replace/models/` and its install
instructions say to copy them **over the installed transformers inside site-packages**. There is no
import hook; without that step `transformers.models.qwen3_5` is the stock implementation and the
policy does not load. The Dockerfile reproduces it and then asserts the overlay took, so a silent miss
fails the build rather than the first robot session.

**`causal-conv1d` is published under `v1.6.1.post4`, not `v1.6.1`.** Both prebuilt CUDA-kernel wheels
(`causal_conv1d` 1.6.1 and `flash_attn` 2.8.1, cp312 / torch2.10 / cu12 / cxx11abiTRUE) are pinned by
URL in the lock and both change the **numbers**, not just the speed: the Qwen3.5 gated-delta-net path
computes `is_fast_path_available` from whether those kernels import, and silently falls back to
pure-PyTorch equivalents when they do not. The reference numbers were measured with the kernels
present. The naive `v{version}` release URL 404s for causal-conv1d, which is the only reason this
model ever looked like it needed an nvcc build from source — it does not.

**`HF_HUB_OFFLINE=0`, deliberately.** The chat processor is resolved by repo id (`Qwen/Qwen3.5-2B`)
and transformers makes a metadata call that strict offline mode refuses, so the image matches the eval
job and leaves offline mode off. With `/hf` mounted nothing is actually downloaded. Set
`HF_HUB_OFFLINE=1` to forbid the network outright and accept that the load fails if the cache is
incomplete.

**The Triton build and its autotune decisions are part of the numbers.** Qwen3.5's gated-delta-net
layers run flash-linear-attention (`fla`) Triton kernels, and `fla` autotunes them with
`cache_results=True`: the first process to hit a kernel benchmarks its candidate block configs and
persists the fastest under `TRITON_CACHE_DIR`. Different block configs sum in a different order, so the
choice changes the output. The validated venv (and the SLURM job that recorded the reference) share one
persistent cache holding 38 such decisions, so every host run uses the same configs. This image
therefore (a) replaces the base image's Triton 3.6.0 with the **exact PyPI wheel the venv has**
(same version, different `libtriton.so` build; Triton's cache key hashes that library, so with the base
image's build no persisted decision is ever found), and (b) bakes the 38 decisions into
`/opt/triton_cache` (`assets/triton_autotune/`, 175 KB) with `TRITON_CACHE_DIR` pointing there. The
cache key also includes the GPU target, so they are used on A100 (sm_80) only; any other card autotunes
as usual, and is then reproducible only within one process. Keep `/opt/triton_cache` writable (Triton
adds compiled kernels there). Cost: +0.67 GB, because the base image's Triton stays in a lower layer.

**`action_loss_only=true`.** The checkpoint was *trained* with the video-foresight loss on (a frozen
WAN2.2-TI2V-5B branch). The action path never reads it, so inference skips it; this is a saving, not a
deviation.

## Verification status — **verified** (2026-09-24): bit-exact when seeded, within 2.5 % unseeded

This policy samples its flow-matching noise from an **unseeded** `torch.normal`
(`modeling_internvla_a1_5.py::sample_noise`, and the adapter does not seed), so the recorded reference is
one stochastic draw. It was verified in three steps, on one A100 of the build host:

**1. The validated host path's own spread** (`openpi_integration/tools/model_spread.sh`, six replays of
the five reference episodes, 170 queries each, one model load). The recorded reference is a typical draw
(|z| < 1 on every metric):

| aggregate metric | reference | host, n=6: mean [min, max] | host range | ref vs host mean |
|---|---:|---|---:|---:|
| `mae_joints_rad` | 0.0133017 | 0.0132006 [0.0128919, 0.0135686] | 5.13 % | +0.77 % |
| `rmse_joints_rad` | 0.0202019 | 0.0200591 [0.0192017, 0.0208875] | 8.40 % | +0.71 % |
| `mae_gripper_m` | 0.00120989 | 0.00121285 [0.00120078, 0.00123050] | 2.45 % | -0.24 % |
| `mae_normalised` | 0.0585496 | 0.0578020 [0.0567343, 0.0591150] | 4.12 % | +1.29 % |
| `traj_mae_joints_rad` | 0.0113979 | 0.0113165 [0.0111922, 0.0115429] | 3.10 % | +0.72 % |

**2. The decisive test: seed both paths and compare element by element.** torch is seeded once in the
server process before the policy is built (a `runpy` wrapper around the same server module; no code
changed). Two fresh seeded runs of the **validated venv** agree with each other bit-for-bit, so the
policy is deterministic given its seed. Two fresh seeded runs of **this image** each match them
**bit-exactly: all 35,700 predicted values over the 170 queries, max |delta| = 0.000e+00**.

This test is what found the Triton problem described above. The image as first built passed step 3 on
aggregate (-1.10 % joint MAE; and six unseeded container replays overlapped the host's six on every
metric), but seeded it differed from the host in 89 % of the returned values (up to 6.8e-3 rad) and
differed from *itself* between two runs. Traced to the base image's Triton build: all 110 Triton
Python files and 7 of its 9 shared libraries were identical to the venv's, `libtriton.so`/`libproton.so`
were not, so `triton_key()`
differed and each container process re-autotuned the `fla` kernels with an empty cache
(`TRITON_PRINT_AUTOTUNING=1`: seven kernels autotuned at run time, first call 38.2 s). With the venv's
wheel and its decisions in place: zero run-time autotunes, first call 5.5 s, bit-exact.

**3. `--probe` and the benchmark replay** (`verify_container.sh`, unseeded, like the reference):

`--probe` against the real mounted checkpoint: **ok**, `(30, 7)`, 49.7 s to load, 5477 ms first call.

| aggregate metric | reference | container | delta |
|---|---:|---:|---:|
| `mae_joints_rad` | 0.0133017 | 0.0131557 | -1.10 % |
| `rmse_joints_rad` | 0.0202019 | 0.0198125 | -1.93 % |
| `mae_gripper_m` | 0.00120989 | 0.00123008 | +1.67 % |
| `mae_normalised` | 0.0585496 | 0.0580446 | -0.86 % |
| `traj_mae_joints_rad` | 0.0113979 | 0.0111146 | -2.49 % |

Per episode -7.56 % .. +6.50 %, the usual envelope of one unseeded draw. Quote a mean over several
replays for this model, never one. Server-side latency in that replay: 522 ms p50 (reference: 532 ms).

Command (benchmark workspace):
`HF_HUB_OFFLINE=0 GPU=3 openpi_integration/tools/verify_container.sh internvla_a15 /models/internvla_a15/runs/<run>/checkpoints/015000/pretrained_model 39,102,120,163,167 8854`.
With `HF_HUB_OFFLINE=0` and `/hf` mounted read-only, `huggingface_hub` logs harmless
`Could not cache non-existence of file ... Read-only file system` errors while it resolves
`Qwen/Qwen3.5-2B`; nothing is downloaded.

On CPU (`device=cpu`, `dtype=float32`) `--probe` also passes, at 460 s per inference and without the
`causal_conv1d`/FLA fast path, so that remains a wiring check only.

Image `vla-bench-internvla_a15:latest`: **11.01 GB** (`docker image inspect`, 11,010,518,303 bytes;
10.34 GB before the Triton wheel and autotune cache were added).
