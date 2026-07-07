PYTHON := .venv/bin/python
VLA_MODELS := /home/ibrahim/storage/VLA_MODELS

# Generic target: make serve CONFIG=<name> CKPT=<dir> [PORT=8800]
CONFIG ?=
CKPT   ?=
PORT   ?= 8800

# Checkpoint paths for the FalconVLA configs added/fixed in this session, verified to exist
# on disk (see config.py comments for how each was picked / cross-checked).
CKPT_EE_AD16_H25_NP           := $(VLA_MODELS)/june_2026_batch3/FalconVLA-8B-aidrc_geometry_simple_translated_eef-singlearm-3v-np-z7-ep100
CKPT_AD16_H25_P               := $(VLA_MODELS)/june_2026_batch2/FalconVLA-8B-ResNet-aloha-geometry-translated-16-p-film
CKPT_AD16_H25_GEOMETRY_NP     := $(VLA_MODELS)/may_2026_batch1/falcon-openvla/FalconVLA-8B-aidrc_geometry-dualarm-3v-np
CKPT_AD14_H25_GEOMETRY_NP     := $(VLA_MODELS)/may_2026_batch1/falcon-openvla/FalconVLA-8B-ResNet-aidrc_geometry-er-3v-np-15k/FalconVLA-8B-ResNet-aidrc_geometry-er-3v-np-15k
CKPT_AD16_H25_SINGLEARM_NP    := $(VLA_MODELS)/may_2026_batch2/falcon-openvla/FalconVLA-8B-FM-DiT-aidrc_geometry_single_arm_16d
CKPT_AD16_H25_SIMPLESINGLEARM_NP := $(VLA_MODELS)/june_2026_batch1/falcon-openVLA/FalconVLA-8B-aidrc_geometry_simple-singlearm-3v-np-z7-ep75
CKPT_AD7_H25_CUPS_NP          := $(VLA_MODELS)/may_2026_batch1/falcon-openvla/FalconVLA-8B-aidrc_cups-singlearm-7dof-z7
CKPT_AD16_H25_CUPS_NP         := $(VLA_MODELS)/april_2026_batch2/FalconVLA-8B-ResNet-aidrc_cups_manipulation_16er-3v-np-bounds

.PHONY: help serve serve-auto \
        serve-ee-ad16-h25-np serve-ad16-h25-p serve-ad16-h25-geometry-np serve-ad14-h25-geometry-np \
        serve-ad16-h25-singlearm-np serve-ad16-h25-simplesinglearm-np serve-ad7-h25-cups-np serve-ad16-h25-cups-np

help:
	@echo "  make serve CONFIG=<name> CKPT=<dir> [PORT=8800]        serve any FalconVLA config"
	@echo "  make serve-auto CKPT=<dir> [PORT=8800]                 serve any FalconVLA checkpoint,"
	@echo "                                                          auto-detecting its config -- no"
	@echo "                                                          CONFIG= / _CONFIGS entry needed"
	@echo ""
	@echo "  Ready-to-run presets (checkpoint path baked in, PORT=8800 overridable):"
	@echo "  make serve-ee-ad16-h25-np             FalconVLA-EE-AD16-H25-NP  (end-effector actions)"
	@echo "  make serve-ad16-h25-p                 FalconVLA-AD16-H25-P      (proprio, FiLM)"
	@echo "  make serve-ad16-h25-geometry-np        FalconVLA-AD16-H25-Geometry-NP"
	@echo "  make serve-ad14-h25-geometry-np        FalconVLA-AD14-H25-Geometry-NP"
	@echo "  make serve-ad16-h25-singlearm-np       FalconVLA-AD16-H25-SingleArm-NP"
	@echo "  make serve-ad16-h25-simplesinglearm-np FalconVLA-AD16-H25-SimpleSingleArm-NP"
	@echo "  make serve-ad7-h25-cups-np             FalconVLA-AD7-H25-Cups-NP"
	@echo "  make serve-ad16-h25-cups-np            FalconVLA-AD16-H25-Cups-NP"

serve:
	$(PYTHON) scripts/serve_policy.py --port $(PORT) policy:checkpoint --policy.config=$(CONFIG) --policy.dir=$(CKPT)

serve-auto:
	$(PYTHON) scripts/serve_policy.py --port $(PORT) policy:auto-checkpoint --policy.dir=$(CKPT)

serve-ee-ad16-h25-np:
	$(PYTHON) scripts/serve_policy.py --port $(PORT) policy:checkpoint --policy.config=FalconVLA-EE-AD16-H25-NP --policy.dir=$(CKPT_EE_AD16_H25_NP)

serve-ad16-h25-p:
	$(PYTHON) scripts/serve_policy.py --port $(PORT) policy:checkpoint --policy.config=FalconVLA-AD16-H25-P --policy.dir=$(CKPT_AD16_H25_P)

serve-ad16-h25-geometry-np:
	$(PYTHON) scripts/serve_policy.py --port $(PORT) policy:checkpoint --policy.config=FalconVLA-AD16-H25-Geometry-NP --policy.dir=$(CKPT_AD16_H25_GEOMETRY_NP)

serve-ad14-h25-geometry-np:
	$(PYTHON) scripts/serve_policy.py --port $(PORT) policy:checkpoint --policy.config=FalconVLA-AD14-H25-Geometry-NP --policy.dir=$(CKPT_AD14_H25_GEOMETRY_NP)

serve-ad16-h25-singlearm-np:
	$(PYTHON) scripts/serve_policy.py --port $(PORT) policy:checkpoint --policy.config=FalconVLA-AD16-H25-SingleArm-NP --policy.dir=$(CKPT_AD16_H25_SINGLEARM_NP)

serve-ad16-h25-simplesinglearm-np:
	$(PYTHON) scripts/serve_policy.py --port $(PORT) policy:checkpoint --policy.config=FalconVLA-AD16-H25-SimpleSingleArm-NP --policy.dir=$(CKPT_AD16_H25_SIMPLESINGLEARM_NP)

serve-ad7-h25-cups-np:
	$(PYTHON) scripts/serve_policy.py --port $(PORT) policy:checkpoint --policy.config=FalconVLA-AD7-H25-Cups-NP --policy.dir=$(CKPT_AD7_H25_CUPS_NP)

serve-ad16-h25-cups-np:
	$(PYTHON) scripts/serve_policy.py --port $(PORT) policy:checkpoint --policy.config=FalconVLA-AD16-H25-Cups-NP --policy.dir=$(CKPT_AD16_H25_CUPS_NP)
