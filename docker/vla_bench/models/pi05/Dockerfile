# VLA-bench model image — pi0.5 (LeRobot `pi05` port, chunk 30, 7-D right arm)
#
# Build context is the OPENPI REPO ROOT:
#   docker build -f docker/vla_bench/models/pi05/Dockerfile -t vla-bench-pi05:latest .
#
# The model source is NOT a submodule: it is cloned at a pinned commit during the build, so the image
# is reproducible without adding a 400 MB submodule to openpi. Override SRC_SHA to move the pin.
#
# Weights are never baked in — see docker/vla_bench/models/pi05/README.md. Neither is the PaliGemma tokenizer:
# it ships in the model's upload set, beside the checkpoint.
ARG BASE=vla-bench-base:torch211-cu128
FROM ${BASE}

# ---------------------------------------------------------------------------------------------
# 1. model source — LeRobot main @ bad0260a4 (the commit pi05 was fine-tuned with; it needs Python >= 3.12,
#    which this base has: 3.12.3, torch 2.11.0+cu128, the training venv's torch build)
# ---------------------------------------------------------------------------------------------
ARG SRC_REPO=https://github.com/huggingface/lerobot
ARG SRC_SHA=bad0260a461557a09dc3a16a327091dbebd3217d
RUN git init -q /opt/model \
 && git -C /opt/model remote add origin "${SRC_REPO}" \
 && git -C /opt/model fetch -q --depth 1 origin "${SRC_SHA}" \
 && git -C /opt/model checkout -q FETCH_HEAD \
 && echo "lerobot @ $(git -C /opt/model rev-parse HEAD)"

# ---------------------------------------------------------------------------------------------
# 2. dependencies, pinned to the venv the reference numbers were measured in
#    (torch/torchvision/nvidia-* deliberately absent: the base already has that exact build)
# ---------------------------------------------------------------------------------------------
COPY docker/vla_bench/models/pi05/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements.lock.txt
# torchcodec is only reached by LeRobotDataset's video decoding, which a policy server never does.
# --no-deps keeps it from pulling the 3 GB cuda-toolkit wheel; a failure here is not fatal.
RUN pip install --no-cache-dir --no-deps torchcodec==0.11.1 \
 || echo "torchcodec unavailable — inference does not decode video, continuing"
RUN pip install --no-cache-dir --no-deps -e /opt/model \
 && python -c "import sys, lerobot, transformers, torch; print('python', sys.version.split()[0], '| lerobot', lerobot.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__)"

# ---------------------------------------------------------------------------------------------
# 3. openpi's own wire codec, so the msgpack bytes match production exactly
# ---------------------------------------------------------------------------------------------
COPY packages/openpi-client /opt/openpi-client

# ---------------------------------------------------------------------------------------------
# 4. the shared server and the adapters
# ---------------------------------------------------------------------------------------------
COPY docker/vla_bench/server /app

# Adapter kwargs are the ones the reference metrics were produced with (VLA-SOTA/repo/scripts/eval/jobs/pi05.cfg),
# except tokenizer_dir, which points at the copy inside the upload set. pi05_policy:Pi05Adapter is the shared
# LeRobotAdapter's inference path plus load-time checks: the tokenizer's six files against their sha256 at
# google/paligemma-3b-pt-224@35e4f46485b4d07967e7e9935bc3786aad50687c (the revision training used), every weight
# tensor against model.safetensors (PI05Policy.from_pretrained returns RANDOM weights when a load fails), the camera
# slots and the quantile statistics. rename_map feeds the ORIGINAL dataset key names and lets the checkpoint's own
# baked-in rename step map them (cam_high -> base_0_rgb, cam_right_wrist -> left_wrist_0_rgb).
# No VLA_BENCH_HUB_PINS: nothing is loaded from the Hub by name (the tokenizer comes from tokenizer_dir), so the
# server needs no /hf cache, no token and no network.
ENV VLA_BENCH_MODEL=pi05 \
    VLA_BENCH_ADAPTER=pi05_policy:Pi05Adapter \
    VLA_BENCH_ADAPTER_KWARGS_DEFAULTS='{"tokenizer_dir":"/models/pi05/paligemma-3b-pt-224","device":"cuda","dtype":"bfloat16","chunk_len":30,"exec_len":30,"rename_map":{"primary":"observation.images.cam_high","wrist":"observation.images.cam_right_wrist"}}' \
    VLA_BENCH_CHECKPOINT=/models/pi05 \
    VLA_BENCH_PORT=8800 \
    VLA_BENCH_PAD_TO=14 \
    HF_HOME=/hf \
    HF_HUB_OFFLINE=1

EXPOSE 8800
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s \
    CMD curl -fsS http://localhost:8800/healthz || exit 1
ENTRYPOINT ["python", "/app/vla_server.py"]
