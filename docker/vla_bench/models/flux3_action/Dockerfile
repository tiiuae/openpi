# VLA-bench model image — FLUX 3 Action (Black Forest Labs; 6.95 B DiT world-action model + KinoVAE + Qwen3-VL-4B)
#
# Build context is the OPENPI REPO ROOT:
#   docker build -f docker/vla_bench/models/flux3_action/Dockerfile -t vla-bench-flux3_action:latest .
#
# Code only. The export (config.json + manifest.json + model.safetensors, 13.9 GB) is mounted at /models, and
# the two frozen encoders come from the Hugging Face cache mounted at /hf (see README.md).
ARG BASE=vla-bench-base:torch210-cu128
FROM ${BASE}

# ---------------------------------------------------------------------------------------------
# 0. the base must carry exactly the torch stack the validated venv ran (torch 2.10.0+cu128,
#    torchvision 0.25.0+cu128, triton 3.6.0). The NATTEN wheel below is built against that torch.
# ---------------------------------------------------------------------------------------------
RUN python -c "import sys, torch, torchvision, triton; \
assert sys.version_info[:2] == (3, 12), sys.version; \
assert torch.__version__ == '2.10.0+cu128', torch.__version__; \
assert torchvision.__version__ == '0.25.0+cu128', torchvision.__version__; \
assert triton.__version__ == '3.6.0', triton.__version__; print('base ok', sys.version.split()[0], torch.__version__)"

# ---------------------------------------------------------------------------------------------
# 1. model source, pinned. No patches: the benchmark ran the checkout unmodified (git status clean).
# ---------------------------------------------------------------------------------------------
ARG SRC_REPO=https://github.com/black-forest-labs/flux-action.git
ARG SRC_SHA=e2dd1d8dbc5977b54315d61f7548c63c043d6d4f
RUN git init -q /opt/src \
 && git -C /opt/src remote add origin "${SRC_REPO}" \
 && git -C /opt/src fetch -q --depth 1 origin "${SRC_SHA}" \
 && git -C /opt/src checkout -q FETCH_HEAD \
 && test "$(git -C /opt/src rev-parse HEAD)" = "${SRC_SHA}" \
 && echo "flux-action @ $(git -C /opt/src rev-parse HEAD)"

# ---------------------------------------------------------------------------------------------
# 2. dependencies: a freeze of the validated venv (the repo's uv.lock resolution), --no-deps so pip installs
#    exactly that set and resolves nothing of its own. NATTEN is not in uv.lock by design (docs/setup.md): it
#    is a torch-specific wheel, fetched from the GitHub release whl.natten.org links to, checked by sha256.
# ---------------------------------------------------------------------------------------------
COPY docker/vla_bench/models/flux3_action/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir --no-deps -r /tmp/requirements.lock.txt
ARG NATTEN_URL=https://github.com/SHI-Labs/NATTEN/releases/download/v0.21.6/natten-0.21.6%2Btorch2100cu128-cp312-cp312-linux_x86_64.whl
ARG NATTEN_SHA256=333cc930134481798625a5f9e0c4935c0f06ec0c428620f04f8702ccadeb7b36
RUN echo "natten @ ${NATTEN_URL} --hash=sha256:${NATTEN_SHA256}" > /tmp/natten.txt \
 && pip install --no-cache-dir --no-deps --require-hashes -r /tmp/natten.txt
RUN pip install --no-cache-dir --no-deps -e /opt/src \
 && python -c "import flux_action, natten, transformers, av, numpy, torch; \
from flux_action.policy import FluxActionPolicy; from flux_action.models import video_vae, text_encoder; \
print('flux_action', flux_action.__file__, '| natten', natten.__version__, '| transformers', transformers.__version__, \
      '| av', av.__version__, '| numpy', numpy.__version__, '| torch', torch.__version__)"
RUN pip check || echo "[see requirements.lock.txt: the freeze is installed verbatim]"

COPY packages/openpi-client /opt/openpi-client
COPY docker/vla_bench/server /app

# ---------------------------------------------------------------------------------------------
# 3. runtime. The export's config.json names the two frozen encoders by the CLUSTER path they had in training
#    (video_vae_id, text_encoder_id -> /lustre1/...). It is left exactly as produced, and manifest.json
#    checksums it. The image instead points the adapter at the same files inside the /hf mount, pinned to the
#    Hub revision the benchmark downloaded and verified (eb267865 = main, byte-identical to the docs' 62878e29):
#        hf download black-forest-labs/flux-3-action-base --revision eb267865d35e49e4066bde4936237f8d9f15a68c \
#            --include video_vae.safetensors "text_encoder/*"
#    Set FLUX_VIDEO_VAE= and FLUX_TEXT_ENCODER= (empty) to use the export's config.json paths instead.
#    history "auto": a request that carries the robot client's --send_history 8 window is served with it; one
#    that does not gets the authors' padded single-observation path (README.md, gotcha 1).
#    OMP_NUM_THREADS=8, as the benchmark's host runs used: without it torch's CPU ops (the uint8 -> float step of the
#    8 history frames runs on the CPU, by the authors' design) spread over every core and the history path measured
#    1542 ms p50 in this image instead of 1331 ms with it (host: 1320 ms). Outputs bit-exact either way.
# ---------------------------------------------------------------------------------------------
ENV VLA_BENCH_MODEL=flux3_action \
    VLA_BENCH_ADAPTER=flux3_action:Flux3ActionAdapter \
    VLA_BENCH_ADAPTER_KWARGS_DEFAULTS='{"device":"cuda:0","chunk_len":30,"exec_len":30,"history":"auto"}' \
    VLA_BENCH_CHECKPOINT=/models/flux3_action/step1000_bf16 \
    VLA_BENCH_PROBE_TASK="pick up the metal pot and place it in the basket" \
    FLUX_VIDEO_VAE=/hf/hub/models--black-forest-labs--flux-3-action-base/snapshots/eb267865d35e49e4066bde4936237f8d9f15a68c/video_vae.safetensors \
    FLUX_TEXT_ENCODER=/hf/hub/models--black-forest-labs--flux-3-action-base/snapshots/eb267865d35e49e4066bde4936237f8d9f15a68c/text_encoder \
    VLA_BENCH_PORT=8800 \
    VLA_BENCH_PAD_TO=14 \
    OMP_NUM_THREADS=8 \
    HF_HOME=/hf

EXPOSE 8800
# The loader sha256-hashes the 13.9 GB model.safetensors on every start (manifest check) and loads 25 GB of
# weights: budget minutes, not seconds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s \
    CMD curl -fsS http://localhost:8800/healthz || exit 1
ENTRYPOINT ["python", "/app/vla_server.py"]
