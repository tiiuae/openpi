# gr00t

NVIDIA GR00T N1.7 fine-tuned on the right-arm 7-D, 2-camera embodiment. Predicts and executes **30
actions per call**. Source: `NVIDIA/Isaac-GR00T`, cloned at a pinned SHA during the build.

## Build

```bash
docker build -f docker/vla_bench/models/gr00t/Dockerfile -t vla-bench-gr00t:latest .   # openpi root
```

## Run

```bash
# fully offline: no network, no Hub token, and no Hugging Face cache needed
WEIGHTS_DIR=/opt/vla_weights docker/vla_bench/run.sh gr00t --probe
WEIGHTS_DIR=/opt/vla_weights docker/vla_bench/run.sh gr00t      # the image default /models/gr00t is the checkpoint
```

The checkpoint is the HF-Trainer directory. Only `config.json`, the two safetensors shards + index and
`processor/{processor_config.json,statistics.json,embodiment_id.json}` are read (~6.9 GB), plus the backbone in
`cosmos-reason2-2b/` (below); `experiment_cfg/` is not read, and the DeepSpeed `global_step*/` state is
training-only. The benchmark's upload set (`VLA-SOTA/repo/scripts/upload_checkpoints.sh`) has exactly these
directly under `gr00t/`, so the image default needs no `--checkpoint`.

## Two things specific to this model

**The backbone ships in the upload set, so it starts offline (since 2026-09-28).** GR00T rebuilds its VLM
backbone, processor and tokenizer from `nvidia/Cosmos-Reason2-2B` (4.9 GB; the checkpoint overwrites the
weights, but they are read). That repo is gated, and loaded **by name** it could not start offline at all:
transformers 4.57.3 calls `huggingface_hub.model_info()` on it while building the tokenizer
(`_patch_mistral_regex`, no offline guard), so `HF_HUB_OFFLINE=1` failed with `OfflineModeIsEnabled`, and online
it needed a Hub token. Now the upload set carries the snapshot training loaded, revision
`9ce19a195e423419c349abfc86fd07178b230561`, as `gr00t/cosmos-reason2-2b/`, and the image's adapter default is
`"backbone": "/models/gr00t/cosmos-reason2-2b"`. The adapter hands GR00T that **local directory** as `model_name`
at load time, through a `<tmp>/nvidia/Cosmos-Reason2-2B` symlink, because `get_backbone_cls` requires that
substring. A local path takes transformers' `_is_local` branch, which makes no Hub call. `config.json` and
`processor/processor_config.json` ship unmodified and still say `"nvidia/Cosmos-Reason2-2B"`.

Verified 2026-09-28 with `--network none`, no `/hf` mount and only the upload set at `/models/gr00t`: the probe
passes, and the seeded output equals the audit's verified configuration value for value.

To use a Hugging Face cache instead of the shipped copy, set the backbone to null:
`-e 'VLA_BENCH_ADAPTER_KWARGS={"backbone":null}'`. The adapter then takes `nvidia/Cosmos-Reason2-2B` at exactly
revision `9ce19a19…` from `/hf` (`hf download nvidia/Cosmos-Reason2-2B --revision 9ce19a195e423419c349abfc86fd07178b230561`,
gated, needs `hf auth login` once), still as a local directory, so still offline.

**No `dataset_dir` is configured, on purpose.** The adapter can fall back to recomputing the q01/q99
percentiles from a LeRobot dataset, but this checkpoint's `statistics.json` already has them, so the
image runs the fast path and needs no dataset mount. If you point it at a checkpoint that predates that
fix, add `dataset_dir` and `modality_config` to `VLA_BENCH_ADAPTER_KWARGS` and mount the dataset.
