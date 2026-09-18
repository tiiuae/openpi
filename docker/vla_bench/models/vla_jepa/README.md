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
docker/vla_bench/run.sh vla_jepa --checkpoint /models/vla_jepa/020000/pretrained_model
```

## Two things specific to this model

**Two post-processor steps MUST be dropped, and the image drops them.** The run was configured with
`pre_snap_gripper_action=false` and `binarize_gripper_action=false`, but `lerobot-train` re-loads the
processor pipeline from the base checkpoint and re-saves it verbatim, so `policy_postprocessor.json`
still lists `PreSnapGripperProcessorStep` and `BinarizeGripperProcessorStep`. Left in place they force
dimension 6 to a binary open/closed — this robot's gripper is a *continuous carriage position in
metres*, so the gripper output is destroyed. `VLA_BENCH_ADAPTER_KWARGS.drop_postprocessor_steps` removes
them by class name, which reproduces the from-config pipeline exactly while keeping the baked-in camera
rename and the normalisation statistics. Do not remove that kwarg.

**Hub cache needed.** The backbone (`facebook/vjepa2-vitl-fpc64-256`, plus `lerobot/VLA-JEPA-Pretrain`)
is resolved by repo id at load time, so `/hf` must be mounted.
