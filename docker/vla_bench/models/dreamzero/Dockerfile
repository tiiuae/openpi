# VLA-bench model image — DreamZero (NVIDIA GEAR world-action model: the Wan2.1-I2V-14B-480P DiT made block-causal,
# with action and state registers; LoRA post-trained from GEAR-Dreams/DreamZero-AgiBot). TWO GPUs by default.
#
# Build context is the OPENPI REPO ROOT:
#   docker build -f docker/vla_bench/models/dreamzero/Dockerfile -t vla-bench-dreamzero:latest .
#
# Code only. DreamZero (Apache-2.0) is cloned at the pinned commit during the build, and two patches are applied:
#   0001  ours: registers the trossen_right7 embodiment (modality config, transforms, projector index, prompt, and the
#         2-view canvas: cam_high top-left, cam_right_wrist top-right). Configuration and code only, no data.
#   0003  upstream PR #83, verbatim (Apache-2.0): VLA.load_lora rebuilds a post-trained LoRA checkpoint on the
#         DreamZero-AgiBot base recorded in its experiment_cfg, as training composed it.
# The upload set (the LoRA, config.json, experiment_cfg/, the dataset statistics) is mounted at /models/dreamzero;
# DreamZero-AgiBot, Wan2.1-I2V-14B-480P without its DiT shards and the umt5-xxl tokenizer come from the Hugging Face
# cache mounted at /hf (see README.md). Returns 24 steps, all executed.
ARG BASE=vla-bench-base:torch28-cu129
FROM ${BASE}

# 0. the base must carry exactly the torch build the validated venv ran (torch 2.8.0+cu129, Python 3.11).
RUN python -c "import sys, torch; \
assert torch.__version__ == '2.8.0+cu129', torch.__version__; \
assert sys.version_info[:2] == (3, 11), sys.version; print('base ok', torch.__version__, sys.version.split()[0])"

# 1. model source, pinned, with the two patches the benchmark ran (git apply --check first: a patch that no longer
#    applies fails the build instead of producing a different model).
ARG SRC_REPO=https://github.com/dreamzero0/dreamzero
ARG SRC_SHA=ab790c198fbce33503358efbbd4187ce9a89adf3
COPY docker/vla_bench/models/dreamzero/patches /opt/dreamzero_patches
RUN git init -q /opt/model \
 && git -C /opt/model remote add origin "${SRC_REPO}" \
 && git -C /opt/model fetch -q --depth 1 origin "${SRC_SHA}" \
 && git -C /opt/model checkout -q FETCH_HEAD \
 && test "$(git -C /opt/model rev-parse HEAD)" = "${SRC_SHA}" \
 && git -C /opt/model apply --check /opt/dreamzero_patches/0001-embodiment-trossen_right7.patch \
        /opt/dreamzero_patches/0003-load-lora-on-pretrained-base-upstream-pr83.patch \
 && git -C /opt/model apply /opt/dreamzero_patches/0001-embodiment-trossen_right7.patch \
        /opt/dreamzero_patches/0003-load-lora-on-pretrained-base-upstream-pr83.patch \
 && echo "DreamZero @ $(git -C /opt/model rev-parse HEAD) + 0001 + 0003" && git -C /opt/model status --short

# 2. dependencies, pinned to the venv the reference numbers were measured in (requirements.lock.txt says what is
#    left out and why: deepspeed, the trainer's engine, would make transformers fail at import on a GPU without a
#    CUDA toolkit): every wheel first, then the two source-only pins without build isolation, then the flash-attn
#    release wheel the venv has, then DreamZero itself, editable from the patched clone, --no-deps.
#    Optional local wheel mirror (default: none, i.e. PyPI and GitHub). The benchmark's own build served its prefetched
#    wheelhouse (each file sha256-checked against PyPI when it was fetched) over http on the build host, because that
#    host's link throttles every TCP flow:  docker build --network host --build-arg PIP_FIND_LINKS=http://127.0.0.1:8765/
#    --build-arg PIP_NO_INDEX=1 --build-arg FLASH_ATTN_WHL=http://127.0.0.1:8765/<the same wheel file name> ...
ARG PIP_FIND_LINKS
ARG PIP_NO_INDEX
COPY docker/vla_bench/models/dreamzero/requirements.lock.txt /tmp/requirements.lock.txt
RUN grep -v '# sdist' /tmp/requirements.lock.txt > /tmp/wheels.txt \
 && grep '# sdist' /tmp/requirements.lock.txt | sed 's/ *# sdist//' > /tmp/sdists.txt \
 && pip install --no-cache-dir --no-deps --only-binary=:all: -r /tmp/wheels.txt \
 && pip install --no-cache-dir --no-deps --no-build-isolation -r /tmp/sdists.txt
ARG FLASH_ATTN_WHL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
RUN pip install --no-cache-dir --no-deps "${FLASH_ATTN_WHL}"
RUN env -u PIP_NO_INDEX -u PIP_FIND_LINKS pip install --no-cache-dir --no-deps -e /opt/model \
 && python -c "import importlib.util, torch, flash_attn, transformers; \
assert importlib.util.find_spec('deepspeed') is None, 'deepspeed must not be installed (see the lock)'; \
from groot.vla.data.schema import EmbodimentTag; from groot.vla.model.n1_5.sim_policy import GrootSimPolicy; \
from groot.vla.model.dreamzero.transform.dreamzero_cotrain import trossen_right7_prompt; \
assert EmbodimentTag('trossen_right7').value == 'trossen_right7'; \
print('groot ok | flash_attn', flash_attn.__version__, '| transformers', transformers.__version__)"

# 3. openpi's own wire codec, so the msgpack bytes match production exactly
COPY packages/openpi-client /opt/openpi-client

# 4. the shared server and the adapters
COPY docker/vla_bench/server /app

# 5. runtime.
#  * checkpoint=/models/dreamzero: the LoRA (model.safetensors), config.json and experiment_cfg/ as trained; the
#    adapter writes a private view of it with the base paths pointed at /hf (the shipped files are not edited).
#  * agibot_dir / wan_dir / tokenizer_dir: the revisions training loaded, from /hf through VLA_BENCH_HUB_PINS (the
#    adapter checks file sizes and the small files' sha256 before loading).
#  * norm_stats_dir: the dataset statistics in the upload set; the adapter refuses a checkpoint whose q01/q99 differ.
#  * TWO GPUs: the server is rank 0 on GPU 0 and starts rank 1 on GPU 1 (the two guidance passes in parallel);
#    DREAMZERO_GPUS=1 runs both on one GPU, about twice as slow. DREAMZERO_FRAME_MODE selects the frames per request
#    (default single: every request a fresh 1-frame call; repeat4: the KV cache kept within an episode).
#  * the authors' inference settings: 16 flow steps, guidance 5.0, the DiT on 8 of them, noise seed 1140.
#  * OMP_NUM_THREADS=8, as the benchmark's host runs used. NO_ALBUMENTATIONS_UPDATE: no version check at import.
ENV VLA_BENCH_MODEL=dreamzero \
    VLA_BENCH_ADAPTER=dreamzero_policy:DreamZeroAdapter \
    VLA_BENCH_ADAPTER_KWARGS_DEFAULTS='{"agibot_dir":"hf:GEAR-Dreams/DreamZero-AgiBot@a8e108456e537ab30e7e84523d41d12a7e739ca5","wan_dir":"hf:Wan-AI/Wan2.1-I2V-14B-480P@6b73f84e66371cdfe870c72acd6826e1d61cf279","tokenizer_dir":"hf:google/umt5-xxl@66cb9e7e85526fe440a945569e42c72fb6cbc0ad","norm_stats_dir":"/models/dreamzero/norm_stats"}' \
    VLA_BENCH_CHECKPOINT=/models/dreamzero \
    VLA_BENCH_HUB_PINS='{"GEAR-Dreams/DreamZero-AgiBot":"a8e108456e537ab30e7e84523d41d12a7e739ca5","Wan-AI/Wan2.1-I2V-14B-480P":{"revision":"6b73f84e66371cdfe870c72acd6826e1d61cf279","ignore_patterns":["diffusion_pytorch_model-*.safetensors"]},"google/umt5-xxl":{"revision":"66cb9e7e85526fe440a945569e42c72fb6cbc0ad","ignore_patterns":["pytorch_model*"]}}' \
    VLA_BENCH_PROBE_TASK="pick up the green cup and place it in the basket" \
    VLA_BENCH_PORT=8800 \
    VLA_BENCH_PAD_TO=14 \
    OMP_NUM_THREADS=8 \
    NO_ALBUMENTATIONS_UPDATE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/hf \
    HF_HUB_OFFLINE=1

EXPOSE 8800
HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s \
    CMD curl -fsS http://localhost:8800/healthz || exit 1
ENTRYPOINT ["python", "/app/vla_server.py"]
