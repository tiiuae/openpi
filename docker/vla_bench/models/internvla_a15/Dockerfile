# VLA-bench model image — InternVLA-A1.5 (Qwen3.5-VL backbone + flow-matching action expert, chunk 30, 7-D right arm)
#
# Build context is the OPENPI REPO ROOT:
#   docker build -f docker/vla_bench/models/internvla_a15/Dockerfile -t vla-bench-internvla_a15:latest .
#
# Weights are never baked in — see docker/vla_bench/models/internvla_a15/README.md.
ARG BASE=vla-bench-base:torch210-cu128
FROM ${BASE}

# ---------------------------------------------------------------------------------------------
# 1. model source. InternVLA-A1.5 ships inside a *fork* of LeRobot (distribution `lerobot`, version
#    1.0.0), so this clone IS the lerobot package as well as the policy. The validated checkout at
#    this SHA has NO local source modifications — the only non-upstream file is the dataset schema
#    added in step 2.
# ---------------------------------------------------------------------------------------------
ARG SRC_REPO=https://github.com/InternRobotics/InternVLA-A1
ARG SRC_SHA=e6fc904f9edbfb14532e97095fc2372202517f76
RUN git init -q /opt/src \
 && git -C /opt/src remote add origin "${SRC_REPO}" \
 && git -C /opt/src fetch -q --depth 1 origin "${SRC_SHA}" \
 && git -C /opt/src checkout -q FETCH_HEAD \
 && echo "InternVLA-A1 @ $(git -C /opt/src rev-parse HEAD)"

# ---------------------------------------------------------------------------------------------
# 2. the embodiment schema. `dataset_schemas/configs/<robot_type>.yaml` is how this repo maps a
#    dataset onto the policy's canonical slots, and it ships one file per robot the authors have.
#    Ours is not upstream: it maps cam_high -> image0, cam_right_wrist -> image1 (slot image2 stays
#    masked white) and declares 7-D absolute joint targets. The backend resolves it by `robot_type`,
#    so the name of this file is load-bearing. Retraining on a different embodiment needs a new one.
# ---------------------------------------------------------------------------------------------
COPY docker/vla_bench/models/internvla_a15/assets/trossen_ai_mobile.yaml \
     /opt/src/src/lerobot/dataset_schemas/configs/trossen_ai_mobile.yaml

# ---------------------------------------------------------------------------------------------
# 3. dependencies, pinned to the venv the reference numbers were measured in.
#
#    TWO PREBUILT CUDA-KERNEL WHEELS are pinned by URL in the lock and both are load-bearing for the
#    NUMBERS, not just for speed. transformers' Qwen3.5 gated-delta-net path sets
#      is_fast_path_available = all((causal_conv1d_fn, causal_conv1d_update,
#                                    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule))
#    and silently falls back to pure-PyTorch equivalents when the kernels are missing. The reference
#    run had them, so the image must too, or it evaluates a different (still working) numeric path.
#    causal-conv1d's 1.6.1 wheels are published under the post-release tag **v1.6.1.post4**, not
#    under "v1.6.1" — the obvious URL 404s, which is what made this model look like it needed an
#    nvcc build from source. It does not: there is a cp312 / torch2.10 / cu12 / cxx11abiTRUE wheel.
# ---------------------------------------------------------------------------------------------
COPY docker/vla_bench/models/internvla_a15/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements.lock.txt

# torchcodec is LeRobot's default video backend. A policy server decodes no video — the client sends
# decoded frames over the wire — but `import lerobot.datasets...` touches it, so it is installed with
# --no-deps (its own deps would drag the multi-GB cuda-toolkit and NPP wheels back in) and it reads
# FFmpeg from the base image's apt packages instead of the host venv's conda link farm.
RUN pip install --no-cache-dir --no-deps torchcodec==0.10.0 \
 || echo "torchcodec unavailable — inference does not decode video, continuing"

RUN pip install --no-cache-dir --no-deps -e /opt/src \
 && python -c "import lerobot, transformers, causal_conv1d, flash_attn, fla; \
print('lerobot', lerobot.__version__, '| transformers', transformers.__version__, \
      '| flash_attn', flash_attn.__version__, '| fla', fla.__version__)"

# ---------------------------------------------------------------------------------------------
# 4. the transformers overlay — step 7 of the repo's own tutorials/installation.md, reproduced from
#    $WS/models/internvla_a15/runs/patch_transformers.sh. The repo carries its own copies of several
#    `transformers.models.*` packages under `<policy>/transformers_replace/models/` and expects them
#    to be COPIED OVER the installed transformers, inside site-packages. It is not a source patch and
#    there is no import hook for it: without this step `transformers.models.qwen3_5` is the stock
#    implementation and the policy does not load. The assert makes a silent miss fail the build.
# ---------------------------------------------------------------------------------------------
RUN set -eu; \
    TD="$(python -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')"; \
    echo "transformers at $TD"; \
    for pol in pi0 pi05 internvla_a1_5; do \
      src="/opt/src/src/lerobot/policies/$pol/transformers_replace/models"; \
      [ -d "$src" ] || { echo "skip $pol (no transformers_replace)"; continue; }; \
      for d in "$src"/*/; do n=$(basename "$d"); cp -r "$d"/. "$TD/models/$n"/; echo "overlaid models/$n from $pol"; done; \
    done
RUN python - <<'PYCHECK'
import inspect
import transformers.models.qwen3_5.modeling_qwen3_5 as m
src = inspect.getsource(m)
assert "torch_chunk_gated_delta_rule" in src and "is_flash_linear_attention_available" in src, \
    "the qwen3_5 transformers overlay did not take"
print("qwen3_5 overlay verified:", m.__file__)
PYCHECK

# ---------------------------------------------------------------------------------------------
# 5. openpi's own wire codec, so the msgpack bytes match production exactly
# ---------------------------------------------------------------------------------------------
COPY packages/openpi-client /opt/openpi-client

# ---------------------------------------------------------------------------------------------
# 6. the shared server and the adapters
# ---------------------------------------------------------------------------------------------
COPY docker/vla_bench/server /app

# Adapter kwargs are the ones the reference metrics were produced with
# (VLA-SOTA/repo/scripts/eval/jobs/internvla_a15.cfg).
#   action_loss_only=true skips the frozen WAN2.2-TI2V-5B video-foresight branch. The checkpoint was
#     TRAINED with that loss on; the action path never reads it, so it is only unused at inference.
#   HF_HUB_OFFLINE is 0, as in the eval job: the chat processor is resolved by repo id
#     (Qwen/Qwen3.5-2B) and transformers makes a metadata call that strict offline mode refuses. With
#     /hf mounted nothing is actually downloaded; set HF_HUB_OFFLINE=1 to forbid the network outright
#     and accept that the load fails if the cache is incomplete.
ENV VLA_BENCH_MODEL=internvla_a15 \
    VLA_BENCH_ADAPTER=internvla_a15:InternVLAA15Adapter \
    VLA_BENCH_ADAPTER_KWARGS='{"repo_dir":"/opt/src","device":"cuda","dtype":"bfloat16","chunk_len":30,"exec_len":30,"robot_type":"trossen_ai_mobile","inference_backend":"standard","action_loss_only":true}' \
    VLA_BENCH_CHECKPOINT=/models/internvla_a15 \
    VLA_BENCH_PORT=8800 \
    VLA_BENCH_PAD_TO=14 \
    HF_HOME=/hf \
    HF_HUB_OFFLINE=0

EXPOSE 8800
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
    CMD curl -fsS http://localhost:8800/healthz || exit 1
ENTRYPOINT ["python", "/app/vla_server.py"]
