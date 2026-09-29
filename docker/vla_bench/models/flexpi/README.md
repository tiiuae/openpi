# vla-bench-flexpi — Flex-π (recipe from Wan2.2)

A 6.83 B multi-stream world-action model (Wan2.2-TI2V-5B video expert + 1.02 B ActionDiT) that generates future RGB,
DINOv3 features and depth pointmaps together with the action chunk. It was trained from the Wan2.2 video prior on
`right7_2view` alone: no robot-pretrained Flex-π base is released. The model predicts 32 steps and the image returns the
first 30 as `(30, 7)` absolute joint targets (6 rad + gripper m), padded to 14 for the Trossen client. Upstream
`geyan21/flex-pi` @ `20c1b2b`, unpatched. Full deployment guide: `VLA-SOTA/results/deployment/flexpi/deployment_guide.md`.

## Build

```bash
docker build -f docker/vla_bench/models/flexpi/Dockerfile -t vla-bench-flexpi:latest .   # from the openpi root
```

Needs `vla-bench-base:torch271-cu128` (torch 2.7.1+cu128, cuDNN 9.7.1, both asserted at build) and network access to
GitHub and PyPI. The image is 8.64 GB and holds no weights.

## Mount

| host | container | what |
|---|---|---|
| `$WEIGHTS_DIR/flexpi/` | `/models/flexpi` | the upload set, which is the run directory: `checkpoints/weights/step_002430.pt` (12.08 GB, default checkpoint), `config.yaml`, `dataset_stats.json`, `text_embeds_cache/right7_2view/` (4 files), `camera_intrinsics.json`, `diffsynth/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors` (1.41 GB) |
| `$HF_CACHE_DIR` | `/hf` | DINOv3 and Depth Anything 3 METRIC-LARGE are not in the upload set. Download them once (1.68 GB, public, no token): |

```bash
# one pattern per command: hf >= 1.0 takes one value per --include
HF_HOME=$HF_CACHE_DIR hf download timm/vit_base_patch16_dinov3.lvd1689m --revision c6a5fb7d12bbd3cf3b0079253141c3332aaed7da --include model.safetensors
HF_HOME=$HF_CACHE_DIR hf download depth-anything/DA3METRIC-LARGE --revision 4010e39f3634a45bc60553321fb49fb760bd594e --include model.safetensors
HF_HOME=$HF_CACHE_DIR hf download depth-anything/DA3METRIC-LARGE --revision 4010e39f3634a45bc60553321fb49fb760bd594e --include config.json
```

Both revisions are pinned in the image (`VLA_BENCH_HUB_PINS`). If a snapshot is missing, the server stops and prints
the command.

## Probe, then serve

```bash
docker/vla_bench/run.sh flexpi --probe     # ~70 s load, first call ~5.6 s; prints "action_shape": [30, 7]
docker/vla_bench/run.sh flexpi             # serves ws://<host>:8800
```

Robot client: `--use_right_arm_only --async_inference --ensemble_type exp --control_freq F`, with F taken from the
guide's rule and your own measured latency. On an A100 that gives F = 17 Hz with this image as shipped (eager,
~567 ms round trip) and 24 Hz with the compiled path in gotcha 4. Keep F ≤ 26 Hz on an A100.

## Gotchas

1. **Depth is estimated on the server.** Training depth came from Depth Anything 3 METRIC-LARGE with an estimated K
   (fx = fy = 394.2 px at 640×480, `camera_intrinsics.json`). The adapter runs the same network on the two frames
   it receives, with the same K. The client sends no depth. Do not change `camera_intrinsics.json`: the model was
   trained with this K, and a different K means retraining.
2. **The third camera slot is black by construction.** The model has a `cam_left_wrist` slot, which was an
   all-black synthetic camera in training. The adapter fills it itself. The left-wrist frame the client sends is never used.
3. **Only the four training instructions work.** There is no text encoder in the image; prompts are looked up in the
   umT5 embedding cache. Any other string is a `FileNotFoundError`. Case and whitespace must match.
4. **Compiled path (the authors' serve default).** Set `-e 'VLA_BENCH_ADAPTER_KWARGS={"torch_compile":true}'`. It
   runs in 414 ms instead of 569 ms of server time on an A100. The compile is paid at start-up (51 s with a warm
   inductor cache, several minutes cold). Its outputs are not bit-identical to eager (joint MAE 0.00594 vs 0.00597 rad
   on the benchmark replay).
5. **Not bit-exact from pass to pass.** Repeating the same request back to back gives the same chunk. A replay
   repeated after other requests differs in single bf16 steps (≤ 5.0e-3 rad) on 4–6 % of the values. cuDNN
   deterministic mode and `CUBLAS_WORKSPACE_CONFIG` do not remove this.
6. **Memory:** 15.5 GiB torch peak and 17.5 GB in nvidia-smi on an A100. Plan for ≥ 24 GB free.
7. `config.yaml` ships as trained and still names training-host paths (`output_dir`, `data.*.dataset_dirs`,
   `pretrained_norm_stats`, `text_embedding_cache_dir`, `model.action_dit_pretrained_path`). None of them is read here.
   The adapter builds only the model and the processor, and it nulls `action_dit_pretrained_path` the way the authors' deploy does.

## Verification (2026-09-29)

This image was run from its enroot export under apptainer on a SLURM A100: every GPU on the docker host was held by
another user's job, and docker must not run inside batch jobs. The run used the same filesystem, ENV, entrypoint and
read-only mounts.

- `--probe` with only the staged upload set at `/models/flexpi` and the three-file HF cache above at `/hf`, with no
  network: ok, 71.9 s load. A docker `--network none` probe on CPU also resolved every file and loaded the model, then
  stopped at the first inference: torch has no CPU bf16 antialiased resize.
- Seeded probe values against the host adapter on the full workspace: **210 / 210 identical** (same sha256).
- 170-query replay of episodes 39,102,120,163,167 (stride 20): 1538 of 35,700 values differ from the host harness run
  (max 5.0e-3 rad). That is the same as two host runs differ from each other (1482–1748). Joint MAE is 0.005968 in the
  container and 0.005972 on the host. Latency in the container: server 566 ms p50, round trip 567 / 569 ms p50 / p95.
