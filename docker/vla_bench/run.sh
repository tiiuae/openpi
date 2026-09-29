#!/usr/bin/env bash
# One command per model. Evaluate them one at a time on the same port.
#
#   ./run.sh --list
#   ./run.sh smolvla --probe                         # env + checkpoint check, no robot needed
#   ./run.sh smolvla                                 # serve on 8800
#   ./run.sh smolvla --checkpoint /models/smolvla/020000/pretrained_model
#
# WEIGHTS_DIR is the host directory holding every model's checkpoints; it is mounted read-only at
# /models. Override with WEIGHTS_DIR=/somewhere ./run.sh ...
#
# HF_CACHE_DIR is the host Hugging Face cache, mounted read-only at /hf (HF_HOME). Most of these
# policies rebuild a frozen backbone or a tokenizer from a Hub repo id at load time — SmolVLA wants
# HuggingFaceTB/SmolVLM2-500M-Video-Instruct, GR00T wants nvidia/Cosmos-Reason2-2B, and so on — so
# without that cache the container has to reach the Hub. Each model's README.md lists what it needs.
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="${WEIGHTS_DIR:-/opt/vla_weights}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
PORT="${PORT:-8800}"
GPUS="${GPUS:-all}"

if [ "${1:-}" = "--list" ] || [ $# -eq 0 ]; then
  echo "backends:"; ls "$SELF/models" | grep -v TEMPLATE | sed 's/^/  /'; exit 0
fi
MODEL="$1"; shift
[ -d "$SELF/models/$MODEL" ] || { echo "unknown backend: $MODEL (try --list)" >&2; exit 2; }

# OpenVLA-OFT rewrites config.json and two .py files inside whatever checkpoint dir it is given, so it
# must never see a read-only or shared mount. Give it a writable private copy instead.
MOUNT_OPTS=":ro"
[ "$MODEL" = "openvla_oft" ] && MOUNT_OPTS=""

# Hub cache: mount it when it exists, otherwise let the container fetch from the Hub itself.
EXTRA=()
if [ -d "$HF_CACHE_DIR" ]; then
  EXTRA+=(-v "${HF_CACHE_DIR}:/hf:ro" -e HF_HOME=/hf -e "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}")
else
  echo "note: no Hugging Face cache at $HF_CACHE_DIR — the container will download backbones from the Hub." >&2
  EXTRA+=(-e HF_HUB_OFFLINE=0)
fi

exec docker run --rm --gpus "$GPUS" \
  -p "${PORT}:8800" \
  -v "${WEIGHTS_DIR}:/models${MOUNT_OPTS}" \
  "${EXTRA[@]}" \
  -e VLA_BENCH_IMAGE="vla-bench-${MODEL}" \
  "vla-bench-${MODEL}:latest" "$@"
