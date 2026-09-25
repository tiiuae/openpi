# molmoact2

MolmoAct2 with a merged LoRA, on the right-arm 7-D, 2-camera embodiment. Predicts and executes **30
actions per call**. The policy code is LeRobot's `molmoact2` policy, so the pinned source is
`huggingface/lerobot` at the commit this run installed (0.6.2) — not the `allenai/molmoact2` author
clone, which the validated environment does not install either.

## Build

```bash
docker build -f docker/vla_bench/models/molmoact2/Dockerfile -t vla-bench-molmoact2:latest .
```

## Run

```bash
WEIGHTS_DIR=/opt/vla_weights HF_CACHE_DIR=$HOME/.cache/huggingface \
  docker/vla_bench/run.sh molmoact2 --probe
docker/vla_bench/run.sh molmoact2                  # the image default /models/molmoact2 is the checkpoint
```

The benchmark's upload set (`VLA-SOTA/repo/scripts/upload_checkpoints.sh`) puts the `pretrained_model/`
contents directly under `molmoact2/`, so the image default needs no `--checkpoint`.

Expect a slow first load: ~6-8 minutes to READY. The full 21 GB base snapshot is read first, then the
12 GB checkpoint -- which holds every base tensor with the LoRA merged, not a delta -- is loaded over it.
Inference is ~260 ms.

## Two things specific to this model

**The checkpoint names its base model by path, twice, and the upload set repoints both.** `checkpoint_path`
is in `config.json` (read by the policy) and in `policy_preprocessor.json` (step `molmoact2_pack_inputs`, which
loads the tokenizer and processor from it); the two are read independently. As trained, both held the
absolute path of the `allenai/MolmoAct2` snapshot in the training host's HF cache. Anywhere else
`resolve_checkpoint_location` hands that string to `snapshot_download` as a repo id, and the load dies with
`HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/lustre1/…'`
(deployment audit, 2026-09-25). Since then `upload_checkpoints.sh` ships both files with
`checkpoint_path = /hf/hub/models--allenai--MolmoAct2/snapshots/e432d85f6e039edca44afb93c262f3084ab72a9c`,
the same snapshot under this image's `/hf` mount. So the workstation needs exactly
`hf download allenai/MolmoAct2 --revision e432d85f6e039edca44afb93c262f3084ab72a9c` (21.8 GB, not gated; the
weights are overwritten by the checkpoint but they are read, so the snapshot must be complete) in
`HF_CACHE_DIR`, and nothing else. With that, `--probe` passes offline with nothing but the upload set mounted,
and its output is identical, value for value, to the verified configuration. Two things that do not work:
mounting the cache at the old absolute path through `run.sh` (its extra arguments go to the server, not to
`docker run`), and a repo id plus `checkpoint_revision` (huggingface_hub 1.31's `snapshot_download` lists the
repo tree even with `HF_HUB_OFFLINE=1` and raises `OfflineModeIsEnabled`).

**The camera names are already the dataset's own.** Training did no rename, so `input_features` uses
`observation.images.cam_high` / `cam_right_wrist` directly. The `rename_map` in the adapter kwargs is
the identity and is passed only to pin the camera-to-slot assignment instead of relying on the
adapter's name heuristics.
