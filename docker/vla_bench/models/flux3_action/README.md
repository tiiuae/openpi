# vla-bench-flux3_action — FLUX 3 Action (Black Forest Labs)

A 6.95 B DiT world-action model + frozen KinoVAE video VAE + frozen Qwen3-VL-4B text encoder, fine-tuned on
`right7_2view` (right arm, `cam_high` + `cam_right_wrist`, 30 Hz). Returns `(30, 7)` absolute joint targets
(6 rad + gripper m), padded to 14 for the Trossen client. Upstream `black-forest-labs/flux-action` @ `e2dd1d8`,
unpatched. Full deployment guide: `VLA-SOTA/results/deployment/flux3_action/deployment_guide.md`.

## Build

```bash
docker build -f docker/vla_bench/models/flux3_action/Dockerfile -t vla-bench-flux3_action:latest .   # from the openpi root
```

Needs `vla-bench-base:torch210-cu128` (torch 2.10.0+cu128, Python 3.12; asserted at build) and network access to
GitHub and PyPI. Image: 9.62 GB. No weights inside.

## Mount

| host | container | what |
|---|---|---|
| `$WEIGHTS_DIR/flux3_action/step1000_bf16/` | `/models/flux3_action/step1000_bf16` (default checkpoint) | the upload set: `config.json`, `manifest.json`, `model.safetensors` (13.9 GB) |
| `$HF_CACHE_DIR` | `/hf` | the two frozen encoders, NOT in the upload set — download once (11.4 GB, public, no token): |

```bash
HF_HOME=$HF_CACHE_DIR hf download black-forest-labs/flux-3-action-base \
    --revision eb267865d35e49e4066bde4936237f8d9f15a68c --include video_vae.safetensors "text_encoder/*"
```

## Probe, then serve

```bash
docker/vla_bench/run.sh flux3_action --probe     # loads (~40 s warm, ~150 s cold) and runs one chunk; prints (30, 7)
docker/vla_bench/run.sh flux3_action             # serves ws://<host>:8800
```

Robot client (all required): `--use_right_arm_only --control_freq 30 --send_history 8`, **synchronous** (never
`--async_inference`), `--rate_of_inference 20` (default) or 30.

## Gotchas

1. **The history does not reach the model yet.** FLUX conditions on 8 ticks (states, the commands sent before
   them, the frames of the oldest and current tick), which the client sends with `--send_history 8`. The
   committed `/app/vla_server.py` drops `request["history"]`, so this image serves the authors' padded
   single-observation path whatever the client sends: measured +13.7 % joint MAE, +60 % gripper MAE on the
   benchmark replay. The fix is `VLA-SOTA/results/deployment/flux3_action/vla_server_history.patch` (verified
   bit-exact in this image, not applied here). The adapter logs which path it served (`first request served on
   the history|single path`).
2. **Never async.** ~1.32 s per chunk on an A100 is longer than the 1.00 s the chunk lasts at 30 Hz: in async mode
   every chunk arrives after its steps have passed and the client sends **zero** joint targets. Synchronous
   mode averages ~10 Hz at stride 20 (the arm pauses ~1.3 s per query). `--control_freq` must be 30: the history
   spacing is 1/30 s.
3. **Encoder paths.** The export's `config.json` names the VAE and the text encoder by the training cluster's paths
   (`video_vae_id`, `text_encoder_id`), shipped as produced. The image ignores them: `FLUX_VIDEO_VAE` and
   `FLUX_TEXT_ENCODER` point the adapter at the snapshot above under `/hf`. To use the config instead, set both
   to empty AND rewrite the two keys AND the `config.json` sha256 in `manifest.json` (the loader re-hashes it).
4. **Memory and load.** ~25 GiB GPU (24.19 GiB torch peak, 25,558 MiB nvidia-smi); every start re-hashes the
   13.9 GB `model.safetensors` against `manifest.json` (health-check start period 900 s).
5. **No compiled path.** The authors' compiled backend accepts only their DROID geometry; an own `torch.compile`
   (1033 ms) changes outputs by up to 0.027 rad and is not enabled or verified.
6. **Deterministic.** The authors re-seed the sampler on every call, so the same request gives the same chunk.

## Verification (2026-09-28)

- 170-query replay of episodes 39,102,120,163,167 (stride 20) through this image vs the host reference:
  **bit-exact, 35,700 / 35,700 values**, single-frame (image as shipped) and with history (image + the patch
  above, mounted over `/app/vla_server.py` for the run).
- Upload set alone (staged tree at `/models/flux3_action`, a 14-file HF cache at `/hf`, `--network none`):
  `--probe` ok; seeded values identical (210 / 210) to the full-workspace configuration with the image defaults,
  with the two keys rewritten to `/hf` paths, and with them rewritten to Hub specs; `config.json` as produced
  without `FLUX_*` fails with `FileNotFoundError` on the `/lustre1` path, as it must.
