# xr1

XR-1 (Xiaomi-Robotics-1-5B) on the right-arm 7-D, 2-camera embodiment. Predicts and executes **30
actions per call**. Source: `XiaomiRobotics/Xiaomi-Robotics-1`, cloned at a pinned SHA; the installable
package is the repo's `xr1/` sub-directory (import name `mibot`).

## Build

```bash
docker build -f docker/vla_bench/models/xr1/Dockerfile -t vla-bench-xr1:latest .
```

## Run

```bash
WEIGHTS_DIR=/opt/vla_weights HF_CACHE_DIR=$HOME/.cache/huggingface \
  docker/vla_bench/run.sh xr1 --probe --checkpoint /models/xr1/checkpoint/mp_rank_00_model_states.pt
```

In the benchmark's upload set (`VLA-SOTA/repo/scripts/upload_checkpoints.sh`) the blob is
`xr1/checkpoint/mp_rank_00_model_states.pt`. The image default `/models/xr1/mp_rank_00_model_states.pt` does not
exist there, so `--checkpoint` is required.

The checkpoint is a **single `.pt` file**: the DeepSpeed ZeRO-2 model-states blob. Stage 2 keeps every
parameter unsharded on every rank, so that one file already holds the full bf16 state dict under
`"module"` — no `zero_to_fp32.py` conversion is needed or performed.

Beside it, the adapter reads the run's dataset metadata from `/models/xr1/data/`. Only two files are read:
`normalize.json` (mean/std and q01/q99 over the packed 60-D vector, and `action_length`, which must be
30) and `manifest.json` (only its `wrist_encoding`, `additive`). The `json/` and `videos/` subtrees of that
dataset are not touched. `upload_checkpoints.sh` ships both files as `xr1/data/` since 2026-09-25. Before that,
the upload set failed at load with `FileNotFoundError: [Errno 2] No such file or directory:
'/models/xr1/data/normalize.json'`. To point `data_dir` somewhere else, override `VLA_BENCH_ADAPTER_KWARGS`,
which replaces the whole JSON, so copy every key. With both files in place `--probe` passes offline with
nothing but the upload set and the Qwen3-VL-4B config/tokenizer files in `/hf`. Its output is identical, value
for value, to the verified `epoch=0-step=10000` configuration (the shipped `last.ckpt` blob holds byte-identical
tensors).

## Four things specific to this model

**It needs ~11 GB of VRAM just for the weights.** The blob is 11 GB of bf16 and is loaded unshared;
budget a 16 GB card at minimum, and expect a long first load off a network filesystem.

**`export_xr1_dataset.py` is baked in and is not upstream.** It provides `unpack_action`, the
packed-60 → absolute-7 inverse: XR-1 emits a 60-D *relative* packed vector that has to be un-packed
against the current joint state to become the absolute joint targets the robot client expects. The
adapter imports it by module name from `/opt/xr1_tools`. It is benchmark code, so it ships as a file
rather than a patch.

**`xr1/assets/config.py` is provenance, not a dependency.** That path is where
`mibot/utils/cfg_utils.py` dumps the fully resolved Hydra config of a training run; upstream ships the
authors' copy and the image replaces it with ours, so it records the run the checkpoint came from.
Nothing on the inference path reads it — the adapter builds the model from explicit kwargs and takes
the 60-D statistics from the mounted `normalize.json`. Regenerate it if you retrain, or drop the
`COPY` if you would rather ship upstream's.

**The Qwen3-VL-4B weights are not needed, only its processor.** XR-1 builds the VLM `_from_config` and
every weight comes from the checkpoint, so `/hf` only has to hold the config/tokenizer/processor files
of `Qwen/Qwen3-VL-4B-Instruct` (~12 MB), not the multi-GB safetensors:
`hf download Qwen/Qwen3-VL-4B-Instruct --include "*.json" "*.txt"`. It is resolved by repo id, not by revision,
so check that the cache's `refs/main` is `ebb281ec70b05090aa6165b016eac8ec08e71b17` (the verified snapshot). A
newer `main` would silently change the prompt/image processing. Downloading with `--revision <sha>` does not
write `refs/main`, and the offline lookup then fails.

## Verification status — **verified, bit-exact** (2026-09-24)

`--probe` against the real mounted checkpoint (the H30_10k_seed1000 step-10000 ZeRO-2 blob): **ok**,
`(30, 7)`, 184.7 s to load off Lustre, 2849 ms first call (includes lazy CUDA/Triton init).

The 170-query replay of the five reference episodes (`39,102,120,163,167`, rate 20) through this image,
driven by the same `deploy_eval/client.py` that produced the benchmark's numbers, reproduces the recorded
reference **bit-exactly**: every aggregate metric matches to 17 significant digits, and element by element
**all 35,700 predicted values over the 170 queries are identical (max |delta| = 0.000e+00)**. The adapter
seeds torch per call, so this is the expected result and the strongest one available: the containerised
computation *is* the validated venv's computation.

| aggregate metric | reference | container | delta |
|---|---:|---:|---:|
| `mae_joints_rad` | 0.015562274124142775 | 0.015562274124142775 | 0.00 % |
| `rmse_joints_rad` | 0.03316962031435663 | 0.03316962031435663 | 0.00 % |
| `mae_gripper_m` | 0.0011287973737922677 | 0.0011287973737922677 | 0.00 % |
| `mae_normalised` | 0.06254925006435862 | 0.06254925006435862 | 0.00 % |
| `traj_mae_joints_rad` | 0.011623176484654144 | 0.011623176484654144 | 0.00 % |

Command (benchmark workspace; the XR-1 dataset derivative that holds `normalize.json` lives under
`data/`, so `data_dir` is overridden to it):

```bash
GPU=3 openpi_integration/tools/verify_container.sh xr1 \
  "/models/xr1/runs/H30_10k_seed1000/project_xiaomi-robotics-1/H30_10k_seed1000/epoch=0-step=10000.ckpt/checkpoint/mp_rank_00_model_states.pt" \
  39,102,120,163,167 8851 \
  -e 'VLA_BENCH_ADAPTER_KWARGS={"repo_dir":"/opt/src/xr1","model_dir":"/opt/xr1_tools","data_dir":"/data/xr1/right7_2view_v1","device":"cuda:0","chunk_len":30,"exec_len":30,"seed":1000}'
```

**The one thing the first GPU run caught: the image had no C compiler.** XR-1's Qwen3-VL uses
`liger_kernel`'s RMSNorm, a Triton kernel, and Triton JIT-compiles a small C helper
(`triton/backends/nvidia/driver.py::CudaUtils`) the first time it launches anything. The image built,
imported and loaded the 11 GB checkpoint cleanly and then died on the first forward pass with
`RuntimeError: Failed to find C compiler`. The Dockerfile now installs `build-essential` (after the pip
layers; `pip freeze` of the rebuilt image is identical to the one before, 118 packages). Nothing short of
a real inference on a GPU would have found this, which is why "it builds and loads" was never the bar.

Earlier evidence, still true: the whole observation pipeline hashes identically to the validated venv
(`openpi_integration/tools/xr1_pipeline_probe.py`), and `load_state_dict(..., strict=True)` accepts all
1135 tensors. There is no CPU fallback: the Qwen3-VL backbone is built with
`_attn_implementation="flash_attention_2"`.

Server-side latency during the replay was 443 ms p50 (mean 456 ms) with another user's process resident on
the same A100; the recorded reference measured 284 ms p50. Take latency
from the benchmark's deployment guide, not from this verification run.

Image `vla-bench-xr1:latest`: **10.06 GB** (`docker image inspect`, 10,061,596,190 bytes; 9.78 GB before
the compiler was added).
