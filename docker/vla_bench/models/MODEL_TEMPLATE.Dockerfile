# VLA-bench model image — TEMPLATE
#
# Build context is the OPENPI REPO ROOT, so packages/openpi-client and docker/vla_bench/ are reachable:
#   docker build -f docker/vla_bench/models/<model>/Dockerfile -t vla-bench-<model>:latest .
#
# Model source is CLONED AT A PINNED SHA rather than vendored as a submodule: ten new submodules is
# maintenance nobody asked for, and six of these models need source patches that are far easier to
# review as .patch files than as a fork. The one exception is openvla_oft, which is already a submodule
# in openpi and uses it.
#
# Weights are NEVER baked in. Mount them at run time:
#   docker run --gpus all -p 8800:8800 -v /local/weights:/models:ro -v ~/.cache/huggingface:/hf:ro \
#       vla-bench-<model> --checkpoint /models/<model>/<ckpt>
ARG BASE=vla-bench-base:torch211-cu128
FROM ${BASE}

# 1. the model's own source at a pinned commit
ARG SRC_REPO=https://github.com/<org>/<repo>
ARG SRC_SHA=<40-hex-commit>
RUN git init -q /opt/model \
 && git -C /opt/model remote add origin "${SRC_REPO}" \
 && git -C /opt/model fetch -q --depth 1 origin "${SRC_SHA}" \
 && git -C /opt/model checkout -q FETCH_HEAD \
 && echo "<repo> @ $(git -C /opt/model rev-parse HEAD)"

# 2. the local source changes the benchmark run actually used, as reviewable patches.
#    Omit this stanza entirely when the model needs none.
COPY docker/vla_bench/models/<model>/patches /tmp/patches
RUN for p in /tmp/patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; git -C /opt/model apply -v "$p"; done

# 3. the model's pinned dependencies, measured from the venv the reference numbers came from
#    (torch/torchvision/nvidia-* are deliberately absent — the base already carries that exact build)
COPY docker/vla_bench/models/<model>/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements.lock.txt
RUN pip install --no-cache-dir --no-deps -e /opt/model

# 4. openpi's wire codec, so msgpack bytes match production exactly
COPY packages/openpi-client /opt/openpi-client
# 5. the shared server and the adapters
COPY docker/vla_bench/server /app

ENV VLA_BENCH_MODEL=<model> \
    VLA_BENCH_ADAPTER=<adapter_module>:<AdapterClass> \
    VLA_BENCH_ADAPTER_KWARGS='{}' \
    VLA_BENCH_PORT=8800 \
    VLA_BENCH_PAD_TO=14 \
    HF_HOME=/hf

EXPOSE 8800
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s \
    CMD curl -fsS http://localhost:8800/healthz || exit 1
ENTRYPOINT ["python", "/app/vla_server.py"]
