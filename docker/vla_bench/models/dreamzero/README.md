# vla-bench-dreamzero — DreamZero (LoRA on DreamZero-AgiBot)

NVIDIA GEAR's 14 B world-action model: the Wan2.1-I2V-14B-480P DiT made block-causal, with action and state registers,
which denoises the next 2 latent video frames together with a 24-action chunk. It was LoRA-post-trained (rank 4) from
`GEAR-Dreams/DreamZero-AgiBot` on `right7_2view`, with relative actions. The image returns `(24, 7)` absolute joint
targets (6 rad + gripper m), all 24 executed, padded to 14 for the Trossen client. Upstream `dreamzero0/dreamzero` @
`ab790c19` (Apache-2.0), cloned at that commit during the build, plus two patches in `patches/`: `0001` registers the
`trossen_right7` embodiment (ours) and `0003` is upstream PR #83 verbatim (LoRA loading on the post-training base).
Full deployment guide: `VLA-SOTA/results/deployment/dreamzero/deployment_guide.md`.

## Build

```bash
docker build -f docker/vla_bench/base/Dockerfile.torch28-cu129 -t vla-bench-base:torch28-cu129 .   # from the openpi root
docker build -f docker/vla_bench/models/dreamzero/Dockerfile -t vla-bench-dreamzero:latest .
```

The base carries torch 2.8.0+cu129 on Python 3.11 (asserted at build), the build the model trained and was evaluated
with, so the host needs an NVIDIA driver of the R575 series or newer. The build needs network access to GitHub and PyPI.
The image holds no weights.

## Mount

| host | container | what |
|---|---|---|
| `$WEIGHTS_DIR/dreamzero/` | `/models/dreamzero` | the upload set: `model.safetensors` (the LoRA and the action/state heads, 217 MB), `config.json`, `experiment_cfg/{conf.yaml,metadata.json}` (the q99 statistics), `norm_stats/` (the dataset statistics the adapter checks them against), `trainer_state.json` |
| `$HF_CACHE_DIR` | `/hf` | the base model is not in the upload set. Download it once (62.6 GB, public, no token): |

```bash
# one pattern flag per command: hf >= 1.0 takes one value per flag
HF_HOME=$HF_CACHE_DIR hf download GEAR-Dreams/DreamZero-AgiBot --revision a8e108456e537ab30e7e84523d41d12a7e739ca5
HF_HOME=$HF_CACHE_DIR hf download Wan-AI/Wan2.1-I2V-14B-480P --revision 6b73f84e66371cdfe870c72acd6826e1d61cf279 --exclude "diffusion_pytorch_model-*.safetensors"
HF_HOME=$HF_CACHE_DIR hf download google/umt5-xxl --revision 66cb9e7e85526fe440a945569e42c72fb6cbc0ad --exclude "pytorch_model*"
```

All three revisions are pinned in the image (`VLA_BENCH_HUB_PINS`): a missing snapshot stops the server and prints the
command, and the adapter checks the files' sizes (and the small files' sha256) before loading. The Wan2.1 DiT shards
(65 GB) and the umt5-xxl encoder weights (52 GB) are not needed: DreamZero-AgiBot carries every model weight.

## Probe, then serve — on two GPUs

```bash
docker/vla_bench/run.sh dreamzero --probe     # 11-14 min load and warm-up on two A100s; "action_shape": [24, 7]
docker/vla_bench/run.sh dreamzero             # serves ws://<host>:8800
DREAMZERO_GPUS=1 docker/vla_bench/run.sh dreamzero   # the 1-GPU fallback, about twice as slow
```

**Two GPUs by default.** The server process drives GPU 0 and starts a second process on GPU 1; the two
classifier-free-guidance passes of every chunk run in parallel and meet over NCCL. `run.sh` gives the container 2 GPUs
(`--gpus 2`) and 16 GB of `/dev/shm`; pick cards with `GPUS='"device=2,3"'`. Each GPU holds the whole model (about
49 GiB at peak), and the host needs RAM for two copies of the 46 GB base model while both processes load it. The 1-GPU
fallback returns the same chunks bit for bit (the guidance passes run in sequence instead of in parallel).

**Synchronous only.** One 24-action chunk (0.8 s of motion at 30 Hz) takes about 3.4 s on two A100s (6.4 s on one).
Robot client: `--use_right_arm_only --control_freq 30 --rate_of_inference 24`, without `--async_inference`. The arm
moves for 0.8 s and then waits while the next chunk is computed.

## Gotchas

1. **The checkpoint is a LoRA.** `VLA.load_lora` (patch 0003) rebuilds DreamZero-AgiBot from `/hf`, injects the LoRA,
   loads the shipped deltas and merges them. Without patch 0003 upstream would rebuild on the raw Wan2.1 DiT, a
   different model.
2. **The configs ship as trained.** `config.json` and `experiment_cfg/conf.yaml` name training-host paths. The adapter
   writes a private copy of the checkpoint's configs with the three base paths pointed at the pinned snapshots under
   `/hf`; nothing else in them is read, and the shipped files are not edited.
3. **One frame per request.** The authors' client accumulates 4 frames per executed block; the Trossen client sends
   one. By default every request is a fresh 1-frame call (the model's first-chunk condition, no state between
   requests). `DREAMZERO_FRAME_MODE=repeat4` keeps the model's KV cache within an episode instead (reset on each new
   connection) and repeats the current frame 4 times. On the benchmark replay it was 1.9 % better on joints at the
   robot's stride 24, worse on two of five episodes, and 13 % slower at p95; the guide has the numbers.
4. **Cold start.** The first two calls compile the encoders and capture CUDA graphs (~75 s, inside the warm-up before
   `READY`); the first call after `READY` is already at full speed. A server that is still loading answers nothing; wait for `READY`.
5. **Cameras.** `cam_high` goes to the top-left tile of the 2x2 canvas (the AgiBot head slot), `cam_right_wrist` to the
   top-right (right hand); the bottom row is black. The client's `cam_left_wrist` frame is never used.
6. **The adapter refuses a bad folder instead of serving it**: a base file with another size or sha256, statistics in
   `experiment_cfg/metadata.json` that differ from `norm_stats/`, other inference settings, or fewer GPUs than asked.
