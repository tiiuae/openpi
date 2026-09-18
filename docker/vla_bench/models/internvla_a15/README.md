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

## Four things specific to this model

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

**`action_loss_only=true`.** The checkpoint was *trained* with the video-foresight loss on (a frozen
WAN2.2-TI2V-5B branch). The action path never reads it, so inference skips it; this is a saving, not a
deviation.

## Verification status — built, **not** replay-verified

**Verified.** The image builds; the pinned CUDA kernels install from their release URLs; `lerobot`
1.0.0, `transformers` 5.2.0, `flash_attn` 2.8.1 and `fla` 0.5.0 all import; all seven
`transformers_replace` package overlays are applied and the build asserts that
`transformers.models.qwen3_5` is the repo's implementation and not the stock one.

**Not verified.** `--probe` and the 170-query replay. `from_pretrained` builds the policy on the GPU
and then loads the 5.39 GB state dict onto it, so it needs roughly **11 GB of VRAM**; the build host
had **7.83 GB free per card** for the whole session (another user's 8-GPU job held 73.3 GB of each
A100). It fails inside `safetensors.torch.load_file(..., device='cuda:0')` with
`torch.OutOfMemoryError ... this process has 7.63 GiB memory in use`. Re-run
`openpi_integration/tools/verify_container.sh internvla_a15 <ckpt> 39,102,120,163,167` with
`HF_HUB_OFFLINE=0` on a card with 16 GB free and compare against
`results/deploy_eval/internvla_a15/metrics.json`. Note that this policy samples its action noise
unseeded, like X-VLA, so compare a mean over several replays — see `models/xvla/README.md` for what
that costs if you do not.
