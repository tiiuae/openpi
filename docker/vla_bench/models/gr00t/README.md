# gr00t

NVIDIA GR00T N1.7 fine-tuned on the right-arm 7-D, 2-camera embodiment. Predicts and executes **30
actions per call**. Source: `NVIDIA/Isaac-GR00T`, cloned at a pinned SHA during the build.

## Build

```bash
docker build -f docker/vla_bench/models/gr00t/Dockerfile -t vla-bench-gr00t:latest .   # openpi root
```

## Run

```bash
WEIGHTS_DIR=/opt/vla_weights HF_CACHE_DIR=$HOME/.cache/huggingface \
  docker/vla_bench/run.sh gr00t --probe
docker/vla_bench/run.sh gr00t --checkpoint /models/gr00t/checkpoint-10000
```

The checkpoint is the HF-Trainer directory. Only `config.json`, the two safetensors shards + index,
`processor_config.json`, `statistics.json`, `experiment_cfg/` and `embodiment_id.json` are read
(~6.9 GB); the DeepSpeed `global_step*/` state in the same directory is training-only and can be left
out of the mount.

## Two things specific to this model

**The backbone is fetched by Hub id, and that repo is gated.** `qwen3_backbone.py` instantiates
`nvidia/Cosmos-Reason2-2B` *by repo id*, not from the checkpoint, so `/hf` must contain that snapshot
(~4.6 GB) or the container needs an `HF_TOKEN` with access granted. This is the most common reason the
probe fails for this model.

**No `dataset_dir` is configured, on purpose.** The adapter can fall back to recomputing the q01/q99
percentiles from a LeRobot dataset, but this checkpoint's `statistics.json` already has them, so the
image runs the fast path and needs no dataset mount. If you point it at a checkpoint that predates that
fix, add `dataset_dir` and `modality_config` to `VLA_BENCH_ADAPTER_KWARGS` and mount the dataset.
