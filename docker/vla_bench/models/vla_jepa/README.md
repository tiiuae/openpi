# vla_jepa

LeRobot-native VLA-JEPA (HF LeRobot port, `policy.type = vla_jepa`) fine-tuned on the right-arm 7-D,
2-camera embodiment. Predicts and executes **30 actions per call**. Source: `huggingface/lerobot` v0.6.1,
cloned at a pinned SHA during the build.

## Build

```bash
docker build -f docker/vla_bench/models/vla_jepa/Dockerfile -t vla-bench-vla_jepa:latest .   # openpi root
```

## Run

```bash
WEIGHTS_DIR=/opt/vla_weights HF_CACHE_DIR=$HOME/.cache/huggingface \
  docker/vla_bench/run.sh vla_jepa --probe
docker/vla_bench/run.sh vla_jepa                   # the image default /models/vla_jepa is the checkpoint
```

The benchmark's upload set (`VLA-SOTA/repo/scripts/upload_checkpoints.sh`) puts the `pretrained_model/`
contents directly under `vla_jepa/`, so the image default needs no `--checkpoint`.

## Two things specific to this model

**Two post-processor steps MUST be dropped, and the image drops them.** The run was configured with
`pre_snap_gripper_action=false` and `binarize_gripper_action=false`, but `lerobot-train` re-loads the
processor pipeline from the base checkpoint and re-saves it verbatim, so `policy_postprocessor.json`
still lists `PreSnapGripperProcessorStep` and `BinarizeGripperProcessorStep`. Left in place they force
dimension 6 to a binary open/closed — this robot's gripper is a *continuous carriage position in
metres*, so the gripper output is destroyed. `VLA_BENCH_ADAPTER_KWARGS.drop_postprocessor_steps` removes
them by class name, which reproduces the from-config pipeline exactly while keeping the baked-in camera
rename and the normalisation statistics. Do not remove that kwarg, and remember that overriding
`VLA_BENCH_ADAPTER_KWARGS` replaces the whole JSON, so an override must carry it too: measured 2026-09-25,
without it every predicted gripper value is exactly **1.0 m** (valid range 0-0.044 m) while the joints are
unchanged -- the server starts and serves normally.

**Hub cache needed: two repos, with their weights.** `Qwen/Qwen3-VL-2B-Instruct` (config `qwen_model_name`,
4.3 GB, main = `89644892…`) and `facebook/vjepa2-vitl-fpc64-256` (`jepa_encoder_name`, main = `b3c1679b…`) are
both loaded with `from_pretrained`, weights included -- the checkpoint overwrites them afterwards, but a missing
weight file is a hard `OSError: … does not appear to have a file named pytorch_model.bin or model.safetensors`.
For V-JEPA2 only `model.safetensors` + the JSON configs are read (1.3 GB); a plain download also pulls the
5.1 GB `original/model.pth`, so use `hf download facebook/vjepa2-vitl-fpc64-256 --include "*.json" "model.safetensors"`.
`lerobot/VLA-JEPA-Pretrain` is **not** needed (it is only a PEFT naming field). Deployment audit 2026-09-25:
`--probe` passes with nothing but the upload set and these two repos mounted, `--network none`.
