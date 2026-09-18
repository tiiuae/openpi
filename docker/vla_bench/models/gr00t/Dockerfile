# VLA-bench model image — NVIDIA GR00T N1.7 (Isaac-GR00T), chunk 30, 7-D right arm
#
# Build context is the OPENPI REPO ROOT:
#   docker build -f docker/vla_bench/models/gr00t/Dockerfile -t vla-bench-gr00t:latest .
ARG BASE=vla-bench-base:torch29-cu128
FROM ${BASE}

# ---------------------------------------------------------------------------------------------
# 1. model source — Isaac-GR00T at the commit the benchmark trained and evaluated with
# ---------------------------------------------------------------------------------------------
ARG SRC_REPO=https://github.com/NVIDIA/Isaac-GR00T
ARG SRC_SHA=51d4c89f72fda44cbf77285c6a8114b52676b8a1
RUN git init -q /opt/model \
 && git -C /opt/model remote add origin "${SRC_REPO}" \
 && git -C /opt/model fetch -q --depth 1 origin "${SRC_SHA}" \
 && git -C /opt/model checkout -q FETCH_HEAD \
 && echo "Isaac-GR00T @ $(git -C /opt/model rev-parse HEAD)"

# ---------------------------------------------------------------------------------------------
# 2. dependencies. flash-attn is pinned to the upstream release wheel for this exact
#    torch 2.9 / cu12 / cxx11abiTRUE / cp312 combination — it is not built here.
# ---------------------------------------------------------------------------------------------
COPY docker/vla_bench/models/gr00t/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements.lock.txt
RUN pip install --no-cache-dir --no-deps torchcodec==0.8.0 \
 || echo "torchcodec unavailable — inference does not decode video, continuing"
RUN pip install --no-cache-dir --no-deps -e /opt/model \
 && python -c "import gr00t, transformers; print('gr00t ok | transformers', transformers.__version__)"

COPY packages/openpi-client /opt/openpi-client
COPY docker/vla_bench/server /app

# dataset_dir/modality_config are deliberately NOT set: they only feed the statistics-injection
# fallback, and this checkpoint's own statistics.json already carries the q01/q99 percentiles the
# N1.7 processor needs. The state slices the dataset would have supplied (single_arm 0:6,
# gripper 6:7) are the adapter's defaults and were checked against meta/modality.json.
ENV VLA_BENCH_MODEL=gr00t \
    VLA_BENCH_ADAPTER=gr00t_policy:Gr00tAdapter \
    VLA_BENCH_ADAPTER_KWARGS='{"repo_dir":"/opt/model","device":"cuda:0","chunk_len":30,"exec_len":30}' \
    VLA_BENCH_CHECKPOINT=/models/gr00t \
    VLA_BENCH_PORT=8800 \
    VLA_BENCH_PAD_TO=14 \
    HF_HOME=/hf \
    HF_HUB_OFFLINE=1

EXPOSE 8800
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
    CMD curl -fsS http://localhost:8800/healthz || exit 1
ENTRYPOINT ["python", "/app/vla_server.py"]
