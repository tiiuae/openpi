# xvla

LeRobot-native X-VLA fine-tuned on the right-arm 7-D, 2-camera embodiment. Predicts and executes
**30 actions per call**. Source: `huggingface/lerobot` v0.6.1, cloned at a pinned SHA during the build
(the same tag as `smolvla` and `vla_jepa`).

## Build

```bash
docker build -f docker/vla_bench/models/xvla/Dockerfile -t vla-bench-xvla:latest .   # from the openpi root
```

## Run

```bash
WEIGHTS_DIR=/opt/vla_weights HF_CACHE_DIR=$HOME/.cache/huggingface \
  docker/vla_bench/run.sh xvla --probe
docker/vla_bench/run.sh xvla                       # the image default /models/xvla is the checkpoint
```

The checkpoint is the LeRobot `pretrained_model/` directory. The benchmark's upload set
(`VLA-SOTA/repo/scripts/upload_checkpoints.sh`) puts its contents directly under `xvla/`, so with
`WEIGHTS_DIR` pointing at a download of that set the image default needs no `--checkpoint`.

## Three things specific to this model

**Hub cache needed, for one tokenizer only.** The Florence-2 backbone is built from the inline config in the
checkpoint's `config.json`; `lerobot/xvla-base` and the vision backbone are **not** touched at load. The one
thing resolved by repo id is the preprocessor's tokenizer, `facebook/bart-large` (config + tokenizer files,
2.7 MB; main = `cb48c136…`). A plain `hf download facebook/bart-large` pulls 5.5 GB of weights in four formats
that are never read, so use `hf download facebook/bart-large --include "*.json" "*.txt"`. Without it the load
fails with `ValueError: Failed to instantiate processor step 'tokenizer_processor'`. (Deployment audit
2026-09-25: `--probe` passes with nothing but the upload set and that one repo mounted, `--network none`.)

**Camera names are pinned, not guessed.** Training renamed `cam_high -> image` and
`cam_right_wrist -> image2`, and that rename is baked into `policy_preprocessor.json`. The image passes
the *original* dataset key names and lets the checkpoint's own rename step map them — the same
arrangement as smolvla, and the reason `rename_map` looks like an identity in the adapter kwargs.

**This policy does not repeat itself, and that is not the container.** `modeling_xvla.py:256` starts
flow inference from an **unseeded** `torch.randn`, with no `generator` argument, so every call draws
fresh noise and no two replays of the same checkpoint agree. Read the next section before comparing a
replay of this model against any recorded number.

## Run-to-run spread, measured

Replaying the five reference episodes (`--episode-list 39,102,120,163,167 --rate-of-inference 20`,
170 queries) **31 times** — 16 through the validated host path, 15 through this image:

| aggregate metric | host, 16 replays | this image, 15 replays | spread (host / image) |
|---|---|---|---|
| `mae_joints_rad` | 0.032924 `[0.031727, 0.033899]` | 0.032360 `[0.031340, 0.033679]` | 6.6 % / 7.2 % |
| `rmse_joints_rad` | 0.048714 `[0.046041, 0.050293]` | 0.047789 `[0.045894, 0.050752]` | 8.7 % / 10.2 % |
| `mae_gripper_m` | 0.0017674 `[0.0016178, 0.0019039]` | 0.0017283 `[0.0015461, 0.0019227]` | 16.2 % / 21.8 % |
| `mae_normalised` | 0.140691 `[0.136087, 0.144870]` | 0.138203 `[0.132183, 0.143023]` | 6.2 % / 7.8 % |
| `traj_mae_joints_rad` | 0.027924 `[0.026856, 0.028880]` | 0.027357 `[0.026664, 0.028127]` | 7.3 % / 5.4 % |

So a single replay of X-VLA carries roughly **±3 %** on the joint MAE and **±10 %** on the gripper
before anything about the deployment has changed. Quote a mean over several replays, not one number.

**Same seed → bit-identical.** Seeding `torch` once in the server process before the policy is built
(`openpi_integration/tools/xvla_seeded.sh` in the benchmark workspace, which does it through `runpy`
without touching any committed code) makes the two paths agree exactly: **all 35,700 predicted values
across the 170 queries matched to 0.000e+00**, and every aggregate metric matched to 17 significant
digits. The containerised computation is the venv's computation.

**The recorded reference is a high draw.** `results/deploy_eval/xvla/metrics.json` has
`mae_joints_rad = 0.035281`, which is **+7.2 %** above the mean of the 16 host replays (z = +3.4) and
above *all 31* replays, host and container alike. Most of that comes from one episode: on episode 167
the reference is **+24 %** above a replay band that is itself only ±5 % wide, and six of its 34 queries
account for ~92 % of the excess — the same handful of queries that are high-variance in every run. The
reference is one unlucky draw of the sampler, not a different code path, and X-VLA's recorded row in
the benchmark is therefore mildly pessimistic about itself.
