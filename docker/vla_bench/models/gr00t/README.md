# gr00t

NVIDIA GR00T N1.7 fine-tuned on the right-arm 7-D, 2-camera embodiment. Predicts and executes **30
actions per call**. Source: `NVIDIA/Isaac-GR00T`, cloned at a pinned SHA during the build.

## Build

```bash
docker build -f docker/vla_bench/models/gr00t/Dockerfile -t vla-bench-gr00t:latest .   # openpi root
```

## Run

```bash
# HF_HUB_OFFLINE=0 is required -- see below
HF_HUB_OFFLINE=0 WEIGHTS_DIR=/opt/vla_weights HF_CACHE_DIR=$HOME/.cache/huggingface \
  docker/vla_bench/run.sh gr00t --probe
HF_HUB_OFFLINE=0 docker/vla_bench/run.sh gr00t      # the image default /models/gr00t is the checkpoint
```

The checkpoint is the HF-Trainer directory. Only `config.json`, the two safetensors shards + index and
`processor/{processor_config.json,statistics.json,embodiment_id.json}` are read (~6.9 GB); `experiment_cfg/`
is not, and the DeepSpeed `global_step*/` state is training-only. The benchmark's upload set
(`VLA-SOTA/repo/scripts/upload_checkpoints.sh`) has exactly these directly under `gr00t/`, so the image
default needs no `--checkpoint`.

## Two things specific to this model

**It cannot start offline, however complete the cache -- it needs the network and a token.** The backbone
and its processor are loaded from `nvidia/Cosmos-Reason2-2B` by repo id (full snapshot, 4.9 GB, main =
`9ce19a19…`; the checkpoint overwrites the weights, but they are read). While loading that tokenizer,
transformers 4.57.3 in this image calls `huggingface_hub.model_info()` on the repo (`_patch_mistral_regex`)
with no offline guard, so under `run.sh`'s default `HF_HUB_OFFLINE=1` the load always fails with
`OfflineModeIsEnabled: Cannot reach https://huggingface.co/api/models/nvidia/Cosmos-Reason2-2B`. Online, the
repo is gated, and without a token every file check fails with `401 … Cannot access gated repo`. What works
through `run.sh`, verified 2026-09-25 with nothing but the upload set mounted: request access to the repo,
run `hf auth login` on the workstation (the token lands in `$HF_CACHE_DIR/token`, which `run.sh` mounts as
`/hf/token`), pre-seed `hf download nvidia/Cosmos-Reason2-2B`, and start with `HF_HUB_OFFLINE=0` and network
access. The output was identical, value for value, to the verified configuration. `run.sh` cannot pass
`HF_TOKEN` as an environment variable, which is why the token file matters.

For a machine without network access there is a self-contained variant, verified identical too: copy the
snapshot into the checkpoint as `gr00t/nvidia/Cosmos-Reason2-2B/` and set `model_name` in `config.json` and
`processor/processor_config.json` to `/models/gr00t/nvidia/Cosmos-Reason2-2B`. A local path skips the Hub call,
and it must contain `nvidia/Cosmos-Reason2`, which `get_backbone_cls` requires, so the plain HF-cache snapshot
path cannot be used. `upload_checkpoints.sh` does not do this by default (it would put 4.9 GB of gated
NVIDIA weights into the upload).

**No `dataset_dir` is configured, on purpose.** The adapter can fall back to recomputing the q01/q99
percentiles from a LeRobot dataset, but this checkpoint's `statistics.json` already has them, so the
image runs the fast path and needs no dataset mount. If you point it at a checkpoint that predates that
fix, add `dataset_dir` and `modality_config` to `VLA_BENCH_ADAPTER_KWARGS` and mount the dataset.
