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
rename and the normalisation statistics. Do not remove that kwarg: measured 2026-09-25, without it every
predicted gripper value is exactly **1.0 m** (valid range 0-0.044 m) while the joints are unchanged -- the
server starts and serves normally. Until 2026-09-28 a `VLA_BENCH_ADAPTER_KWARGS` override replaced the image's
whole JSON and so dropped this key silently. The image default now lives in `VLA_BENCH_ADAPTER_KWARGS_DEFAULTS`
and an override is **merged over it** (see the top-level README), so `-e 'VLA_BENCH_ADAPTER_KWARGS={"chunk_len":30}'`
keeps `drop_postprocessor_steps`. Only setting `drop_postprocessor_steps` itself (to `[]` or `null`) removes it.

**Frames reach Qwen at native resolution, as in training.** The checkpoint config carries
`resize_images_to=[224,224]`, which the policy applies only at inference (`predict_action` squashes each view to
224x224); training fed the native 480x640 frames unresized (a 30x40 Qwen grid per view instead of 16x16). The
image sets `overrides.resize_images_to=null` in its kwargs defaults so the served inputs equal training's. An
override of `overrides` replaces that object whole, so keep the key if you set it.

**Hub cache needed: two repos, with their weights.** `Qwen/Qwen3-VL-2B-Instruct` (config `qwen_model_name`,
4.3 GB, main = `89644892…`) and `facebook/vjepa2-vitl-fpc64-256` (`jepa_encoder_name`, main = `b3c1679b…`) are
both loaded with `from_pretrained`, weights included -- the checkpoint overwrites them afterwards, but a missing
weight file is a hard `OSError: … does not appear to have a file named pytorch_model.bin or model.safetensors`.
For V-JEPA2 only `model.safetensors` + the JSON configs are read (1.3 GB); a plain download also pulls the
5.1 GB `original/model.pth`, so run `hf download facebook/vjepa2-vitl-fpc64-256 --include "*.json"` and then
`hf download facebook/vjepa2-vitl-fpc64-256 --include model.safetensors` (one pattern per command: hf ≥ 1.0 takes one
value per `--include`, and a second bare value is read as a file name).
`lerobot/VLA-JEPA-Pretrain` is **not** needed (it is only a PEFT naming field). Deployment audit 2026-09-25:
`--probe` passes with nothing but the upload set and these two repos mounted, `--network none`.
**Pinned (since 2026-09-28).** The image sets `VLA_BENCH_HUB_PINS`, so the server serves both repos at exactly `89644892e4d85e24eaac8bacfd4f463576704203` and `b3c1679b7c34d3255ef3547f27c7b226aefab26f`, whatever the cache's `refs/main` says, and stops at startup with the `hf download --revision` command if that snapshot is missing. Either way of seeding the cache works: the commands above, or the same with `--revision <sha>`.
