# vla-bench-openwam — OpenWAM (recipe from video pretraining)

A 7.67 B world-action model: a Wan2.2-TI2V-5B-shaped causal video expert (OpenWAM-Pretraining, robot/human video)
plus a 2 B action expert (dual-expert MoT, program `joint`), which denoises the next 4 latent video frames together
with the action chunk. It was trained from the video prior on `right7_2view` alone: no robot-action-pretrained OpenWAM
policy is released. The model predicts 32 steps and the image returns the first 30 as `(30, 7)` absolute joint targets
(6 rad + gripper m), padded to 14 for the Trossen client. Upstream `OpenWAM/OpenWAM` @ `4b3814a8` (AGPL-3.0), cloned
at that commit during the build and unpatched; none of its source is in this repository. Full deployment guide:
`VLA-SOTA/results/deployment/openwam/deployment_guide.md`.

## Build

```bash
docker build -f docker/vla_bench/base/Dockerfile.torch211-cu130 -t vla-bench-base:torch211-cu130 .   # from the openpi root
docker build -f docker/vla_bench/models/openwam/Dockerfile -t vla-bench-openwam:latest .
```

The base carries torch 2.11.0+cu130 (asserted at build), the build the model trained and was evaluated with, so the
host needs an NVIDIA driver of the R580 series or newer. The build needs network access to GitHub and PyPI. The image
holds no weights.

## Mount

| host | container | what |
|---|---|---|
| `$WEIGHTS_DIR/openwam/` | `/models/openwam` | the upload set: `model_state.pt` (30.7 GB, every tensor, fp32; the default checkpoint), `resolved_config.yaml`, `openwam_norm_stats.json`, `text_embeds_cache/right7_2view_umt5_cpu_bf16.pt` (42 MB), `train_state.json` |
| `$HF_CACHE_DIR` | `/hf` | the frozen Wan2.2 VAE and the UMT5-xxl text encoder + tokenizer are not in the upload set. Download them once (14.2 GB, public, no token): |

```bash
# one pattern per command: hf >= 1.0 takes one value per --include
HF_HOME=$HF_CACHE_DIR hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --revision b8fff7315c768468a5333511427288870b2e9635 --include "vae/*"
HF_HOME=$HF_CACHE_DIR hf download robbyant/lingbot-va-base --revision 68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c --include "text_encoder/*"
HF_HOME=$HF_CACHE_DIR hf download robbyant/lingbot-va-base --revision 68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c --include "tokenizer/*"
```

Both revisions are pinned in the image (`VLA_BENCH_HUB_PINS`): a missing snapshot stops the server and prints the
command, and the adapter checks every file's size (and the small files' sha256) before loading.

## Probe, then serve

```bash
docker/vla_bench/run.sh openwam --probe     # ~2 min load on an A100 (warm files), first call ~4 s; "action_shape": [30, 7]
docker/vla_bench/run.sh openwam             # serves ws://<host>:8800
```

**Synchronous only.** One chunk takes about 3.9 s on an A100 (40 denoising passes of a 7.67 B model), almost four
times the 1.00 s a 30-row chunk lasts. Robot client: `--use_right_arm_only --control_freq 30 --rate_of_inference 20`
(or 30), without `--async_inference`. The arm moves for 0.67 s and then waits while the next chunk is computed.

**Memory.** On an A100 the GPU holds about 17 GiB after the load. The load itself peaks near 29 GiB, while the fp32
parameters are on the GPU before the cast to bf16. The host needs about 60 GiB of RAM to load: `model_state.pt` is read
whole, the model is built on the CPU, and the text encoder stays there.

## Gotchas

1. **Every request starts from scratch.** OpenWAM's own closed loop keeps a rolling observed history and a KV cache,
   and commits every executed control and its observation before it replans. The Trossen client sends one frame per
   request and never the executed actions, so the adapter runs the authors' episode-start transaction (one observed
   frame, a fresh session, seed 1000) at every request. Nothing is carried between requests; `reset()` is a no-op.
2. **The VAE runs in fp32, with TF32 off.** The training latents were encoded with the Wan2.2 VAE in fp32 and
   stored in fp16. The adapter encodes live frames the same way and rounds them through fp16. OpenWAM's own rollout
   would cast the VAE to bf16 on a GPU, and cuDNN's default TF32 convolutions would be 0.1 % off; the adapter avoids
   both.
3. **The model's parameters run in bf16.** That is the precision of the training forward pass (FSDP mixed precision).
   `model_state.pt` holds the fp32 masters, which OpenWAM's own loader would leave in fp32, at twice the latency.
4. **The text encoder runs on the CPU, in bf16.** This is how training ran it (`cpu_offload`). The embeddings of the
   four training instructions and the blank prompt ship in the upload set, computed the same way, and the adapter
   preloads them. Any other prompt is encoded on its first request, which took about 150 s per prompt on the
   benchmark's CPUs. Use the training instructions verbatim.
5. **Both views are letterboxed.** Each 640x480 frame is fitted into 128x128 with the authors' `fit_frames` (128x96
   content, 16-px black bars), and the two views sit side by side: cam_high on the left, cam_right_wrist on the right.
6. **De-normalisation.** OpenWAM's action adapter inverts only the target's final normalisation (`none`). The
   adapter then applies OpenWAM's `denormalize_joint_positions` with the checkpoint's quantiles. Without it the
   server would return normalised values, which OpenWAM's LIBERO rollout never needs because its targets are raw.
7. `resolved_config.yaml` ships as trained and still names training-host paths. The adapter overrides the three
   base-asset paths. The others (dataset root, checkpoint dirs) are not read.
