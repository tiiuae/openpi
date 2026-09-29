# VLA-bench model servers

One container per model. Each serves a single checkpoint over the websocket protocol that
`examples/trossen_ai/main.py` already speaks, so **the robot client needs no change** and models are
swapped by stopping one container and starting another on the same port.

```
robot client  ──ws──▶  vla-bench-<model> container  ──▶  action chunk
 (unchanged)            (one model, one checkpoint)
```

## Quick start

```bash
docker/vla_bench/run.sh --list                  # what is available
docker/vla_bench/run.sh gr00t --probe           # load the checkpoint and exit; no robot needed
docker/vla_bench/run.sh gr00t                   # serve on :8800
```

Then point the robot client at it as usual:

```bash
# --use_right_arm_only is required (without it the left arm is commanded to zeros);
# take --control_freq and sync/async from the model's deployment guide
python examples/trossen_ai/main.py --policy_host <server-host> --policy_port 8800 --use_right_arm_only \
    --control_freq <F> --task_prompt "..."
```

## Weights are not in the images

Images hold code only. Two host directories are mounted read-only at run time.

**`/models` — the checkpoints** (`WEIGHTS_DIR`, default `/opt/vla_weights`):

```
/opt/vla_weights/            <- WEIGHTS_DIR on the host
  gr00t/checkpoint-10000/
  smolvla/020000/pretrained_model/
  ...
```

Override with `WEIGHTS_DIR=/your/path docker/vla_bench/run.sh <model>`, and select a specific checkpoint
with `--checkpoint /models/<model>/<ckpt>`.

**`/hf` — the Hugging Face cache** (`HF_CACHE_DIR`, default `$HOME/.cache/huggingface`, exported as
`HF_HOME` inside the container). This is not optional for most models: a fine-tuned checkpoint stores the
*trained* weights, but the policy class rebuilds its frozen backbone, its processor and its tokenizer from
a Hub **repo id** every time it loads — SmolVLA wants `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, GR00T
wants `nvidia/Cosmos-Reason2-2B`, and so on. `run.sh` mounts the cache and sets `HF_HUB_OFFLINE=1` when the
directory exists, and falls back to letting the container download from the Hub when it does not. Each
model's `models/<m>/README.md` lists the repo ids it needs.

**Revisions are pinned.** A repo loaded by name would otherwise resolve to whatever `refs/main` the cache
holds (or the Hub's current `main`), so a newer upstream tokenizer or processor would change the model's
input silently. Images whose policy loads a repo by name set `VLA_BENCH_HUB_PINS` (smolvla, xvla, vla_jepa,
internvla_a15, xr1, flexpi). Before the adapter is imported, the server builds an overlay of the `/hf` cache under
`/tmp` in which each pinned repo exposes only its pinned snapshot. `/hf` itself is never written. A missing
snapshot stops the server with the exact `hf download --revision <sha>` command. With no cache and the network
up, the pinned revision is downloaded first. gr00t ships its backbone in the upload set. molmoact2,
gigabrain07, walloss05 and lingbot_v2 name a snapshot directory, which fixes the revision by construction.
fastwam reads Wan2.2 from a plain directory under `/hf/diffsynth`, whose revision its README gives, and
openvla_oft reads nothing from the Hub. `server/adapters/hub_pins.py` has the details.

## Where the model source comes from

The Dockerfiles **clone the upstream repo at a pinned commit SHA** (`SRC_REPO` / `SRC_SHA` build args)
rather than adding ten git submodules to openpi. Local source changes that the benchmark run depended on
are carried as `.patch` files next to each Dockerfile and applied during the build, so they are reviewable.
`openvla_oft` is the exception: it already exists as an openpi submodule and the image uses that.

**One exception.** `openvla_oft` rewrites `config.json`, `modeling_prismatic.py` and
`configuration_prismatic.py` inside whatever checkpoint directory it is given. `run.sh` therefore mounts
its weights **read-write**, and that directory must be a private copy, never a shared snapshot — otherwise
the first run corrupts the weights for every other consumer.

## Layout

```
docker/vla_bench/
  base/          shared base per torch version — built once, reused (4 models share torch 2.11)
  server/        vla_server.py + adapters/ — the wire contract and the model glue, one copy
  models/<m>/    per-model Dockerfile: model source + its pins on top of a base
  compose/       one compose file per model, for anyone who prefers compose
  run.sh         the launcher
```

## What the server guarantees

Getting either of these wrong is silent, so they live in `server/vla_server.py` and nowhere else.

**Colour and layout.** The client sends **channel-first BGR** at 224×224 — its `cv2.cvtColor(BGR2RGB)`
runs on an already-RGB frame, which `examples/trossen_ai/CLIENT_SCHEMA.md` documents as unfixable on the
client. Every server flips it back to RGB before inference.

**Embodiment.** The client sends a **14-D** bimanual state and truncates responses with `[:, :14]`,
crashing on missing columns. These policies are **7-D right arm**: the server reads `state[7:14]` and
writes predictions back into columns `7:14`, zeros elsewhere.

## Adapter kwargs: image defaults, merged overrides

Each image sets its adapter's constructor arguments as a JSON object in `VLA_BENCH_ADAPTER_KWARGS_DEFAULTS`
(see its Dockerfile). To change one, pass only that key:

```bash
docker run ... -e 'VLA_BENCH_ADAPTER_KWARGS={"exec_len":10}' vla-bench-<model>    # or --adapter-kwargs '{...}'
```

The server merges the override **over** the defaults: a key you give wins, a key you do not give keeps the
image's value. The merge is top-level, so a nested value such as `rename_map` is replaced whole, and a key set
to `null` reaches the adapter as `None`. The startup log and the `--probe` output print the effective kwargs.
Until 2026-09-28 an override *replaced* the whole JSON, which silently dropped every key it did not repeat; for
`vla_jepa` that pinned every gripper output to 1.0 m while the server served normally.

## Episode boundaries

Each websocket connection is one episode. On every new connection the server calls the adapter's `reset()`
before it sends the metadata dict. The twelve benchmark adapters keep no state between calls and inherit a
no-op; an adapter that keeps history (frames, previously issued commands) clears it there.

## Verifying a backend before the robot

`--probe` loads the checkpoint, runs one inference on a blank frame and prints the shape, load time and
first-call latency. It exits non-zero if the mount or the environment is wrong. Run it for every model at
the start of a session; it is much cheaper than discovering a broken mount mid-experiment.

A stronger check, when you want it, is to replay recorded episodes through the running server and compare
against the recorded reference numbers in `VLA-SOTA/results/deploy_eval/<model>/metrics.json`. A backend
that does not reproduce its reference error is not correctly integrated, whatever the server reports.

## Jetson Orin

These images are **x86_64 only** and will not run on an AGX Orin, which is arm64 with JetPack-specific
CUDA. Orin needs separate images on NVIDIA's L4T bases using NVIDIA's own torch wheels, and several models
pin torch versions with no Jetson build at all. Measured latency also says only the three fastest models
could approach 30 Hz there, and only with TensorRT/INT8 work. Treat Orin as a separate exercise.
