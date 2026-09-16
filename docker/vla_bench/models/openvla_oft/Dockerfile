# VLA-bench model image — OpenVLA-OFT+ (FiLM, L1 regression head, proprio, LoRA r=32 merged)
#
# Build context is the OPENPI REPO ROOT. Unlike the other models this one does NOT clone upstream:
# openvla-oft is already an openpi submodule, so make sure it is checked out first.
#
#   git submodule update --init openvla-oft
#   docker build -f docker/vla_bench/models/openvla_oft/Dockerfile -t vla-bench-openvla_oft:latest .
#
# This is the only image on torch 2.2 / CUDA 12.1 / Python 3.10, and the only one that needs a *devel*
# base — flash-attn 2.5.5 has a prebuilt wheel for this exact torch/ABI combination, taken from the
# upstream release page rather than compiled here.
ARG BASE=vla-bench-base:torch220-cu121
FROM ${BASE}

# ---------------------------------------------------------------------------------------------
# 1. model source — openpi's own openvla-oft submodule
# ---------------------------------------------------------------------------------------------
COPY openvla-oft /opt/model

# ---------------------------------------------------------------------------------------------
# 2. dependencies, pinned to the validated venv. Three of them are not on PyPI and are pinned by URL:
#    the moojink transformers fork (bidirectional attention / parallel decoding), dlimp, and the
#    flash-attn 2.5.5 release wheel. tensorflow 2.15 is required — openvla_utils uses it for the
#    lanczos resize and the centre crop — and tensorflow-metadata must stay at 1.13.1 (models/
#    openvla_oft/ENV.md); peft must stay at 0.11.1 or the merged LoRA fails to load.
# ---------------------------------------------------------------------------------------------
COPY docker/vla_bench/models/openvla_oft/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements.lock.txt
RUN pip install --no-cache-dir --no-deps -e /opt/model

# ---------------------------------------------------------------------------------------------
# 3. source patches. Applied AFTER the editable install so the expensive dependency layer stays
#    cached when a patch changes; `patch` rather than `git apply` because a copied submodule's .git
#    is a dangling gitdir pointer.
#      0001  adds the RIGHT7 embodiment block (chunk 30, action 7, proprio 7) to prismatic/vla/constants.py
#      0002  removes this fork's second BGR->RGB flip and its per-inference PNG dumps
#    The benchmark's other four source changes (RLDS/oxe registration, finetune.py) are training-side
#    only and are deliberately not carried into a serving image.
# ---------------------------------------------------------------------------------------------
COPY docker/vla_bench/models/openvla_oft/patches /tmp/patches
RUN apt-get update && apt-get install -y --no-install-recommends patch && apt-get clean \
 && for p in /tmp/patches/*.patch; do echo "applying $p"; patch -p1 -d /opt/model < "$p"; done

# Fail the BUILD, not the first robot session, if the RIGHT7 block did not take. constants.py is
# loaded by file path on purpose: importing `prismatic` drags in the whole package, and the argv
# sniffing it does has to happen before the module is first imported anyway.
RUN python - <<'PYCHECK'
import sys, importlib.util
sys.argv.append("--dataset_name=right7_2view_v1")
spec = importlib.util.spec_from_file_location("_c", "/opt/model/prismatic/vla/constants.py")
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
got = (c.NUM_ACTIONS_CHUNK, c.ACTION_DIM, c.PROPRIO_DIM)
assert got == (30, 7, 7), f"RIGHT7 constants did not take: {got}"
print("RIGHT7 constants active:", got, c.ACTION_PROPRIO_NORMALIZATION_TYPE)
PYCHECK

COPY packages/openpi-client /opt/openpi-client
COPY docker/vla_bench/server /app

# checkpoint_dir, not checkpoint: this adapter names the parameter differently from the rest.
ENV VLA_BENCH_MODEL=openvla_oft \
    VLA_BENCH_ADAPTER=openvla_oft:OpenVLAOFTAdapter \
    VLA_BENCH_CHECKPOINT_KWARG=checkpoint_dir \
    VLA_BENCH_ADAPTER_KWARGS='{"repo_dir":"/opt/model","chunk_len":30,"exec_len":30,"center_crop":true,"unnorm_key":"right7_2view_v1"}' \
    VLA_BENCH_PORT=8800 \
    VLA_BENCH_PAD_TO=14 \
    HF_HOME=/hf \
    TOKENIZERS_PARALLELISM=false \
    TF_CPP_MIN_LOG_LEVEL=2

EXPOSE 8800
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
    CMD curl -fsS http://localhost:8800/healthz || exit 1
ENTRYPOINT ["python", "/app/vla_server.py"]
