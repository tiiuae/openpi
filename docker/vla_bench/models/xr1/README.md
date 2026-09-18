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
  docker/vla_bench/run.sh xr1 --probe --checkpoint /models/xr1/mp_rank_00_model_states.pt
```

The checkpoint is a **single `.pt` file**: the DeepSpeed ZeRO-2 model-states blob. Stage 2 keeps every
parameter unsharded on every rank, so that one file already holds the full bf16 state dict under
`"module"` — no `zero_to_fp32.py` conversion is needed or performed.

Beside it, mount the run's dataset metadata at `/models/xr1/data/`. Only two files are read:
`normalize.json` (mean/std and q01/q99 over the packed 60-D vector, and `action_length`, which must be
30) and `manifest.json` (which records `wrist_encoding`). The `json/` and `videos/` subtrees of that
dataset are not touched. Point `data_dir` somewhere else by overriding `VLA_BENCH_ADAPTER_KWARGS`.

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
of `Qwen/Qwen3-VL-4B-Instruct` (~12 MB), not the multi-GB safetensors.

## Verification status — built, pipeline-verified, **not** replay-verified

Honest accounting, because "it builds" is not the bar the rest of these images were held to.

**Verified.** The image builds; `mibot` imports; the 11 GB DeepSpeed blob loads and
`load_state_dict(..., strict=True)` accepts all 1135 tensors. The whole observation pipeline was then
compared against the validated venv, tensor by tensor, on identical inputs
(`openpi_integration/tools/xr1_pipeline_probe.py` in the benchmark workspace, run once in each): the
`resize_image` output, the full Qwen3-VL chat payload (`input_ids`, `attention_mask`, `pixel_values`,
`image_grid_thw`), the 60-D `compose_state` vector, its q01/q99 normalisation, the action mask, the
mean/std/q01/q99 tables and the packed-60 → absolute-7 `unpack_action` inverse all hash **identically**,
on the same torch 2.8.0+cu128 / transformers 4.57.1 / numpy 2.1.3.

**Not verified.** `--probe` and the 170-query replay. XR-1's bf16 weights need **~11-12 GB of VRAM**
and the build host had **7.83 GB free per card** for the whole session (another user's 8-GPU job held
73.3 GB of each A100). The probe fails at exactly that point, in `model.eval().to(device)`, with
`torch.OutOfMemoryError ... this process has 7.62 GiB memory in use`. Nothing about the image is
implicated. Re-run
`openpi_integration/tools/verify_container.sh xr1 <ckpt> 39,102,120,163,167` on a card with 16 GB free
and compare against `results/deploy_eval/xr1/metrics.json`; the adapter seeds torch per call, so unlike
the LeRobot policies here this one should reproduce its reference exactly rather than approximately.
