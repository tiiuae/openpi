PYTHON := .venv/bin/python

CONFIG ?=
CKPT   ?=
PORT   ?= 8800

.PHONY: help serve serve-auto

# ANSI color codes
BOLD   := \033[1m
CYAN   := \033[36m
YELLOW := \033[33m
GREEN  := \033[32m
RESET  := \033[0m

help:
	@echo "$(BOLD)$(CYAN)FalconVLA Policy Server$(RESET)"
	@echo ""
	@echo "$(BOLD)Usage:$(RESET)"
	@echo "  $(YELLOW)make serve$(RESET) CONFIG=<name> CKPT=<dir> [PORT=8800]"
	@echo "    Serve a FalconVLA checkpoint with explicit config"
	@echo ""
	@echo "  $(YELLOW)make serve-auto$(RESET) CKPT=<dir> [PORT=8800]"
	@echo "    $(GREEN)Recommended$(RESET): Auto-detect config from checkpoint directory"
	@echo "    No CONFIG= needed — infers it from checkpoint metadata"
	@echo ""
	@echo "$(BOLD)Examples:$(RESET)"
	@echo "  $(GREEN)make serve-auto CKPT=/path/to/checkpoint$(RESET)"
	@echo "  $(GREEN)make serve-auto CKPT=/path/to/checkpoint PORT=9000$(RESET)"
	@echo "  make serve CONFIG=FalconVLA-AD16-H25-NP CKPT=/path/to/checkpoint"

serve:
	@if [ -z "$(CONFIG)" ] || [ -z "$(CKPT)" ]; then \
		echo "$(BOLD)Error:$(RESET) CONFIG and CKPT are required"; \
		echo "Usage: make serve CONFIG=<name> CKPT=<dir> [PORT=8800]"; \
		exit 1; \
	fi
	$(PYTHON) scripts/serve_policy.py --port $(PORT) policy:checkpoint --policy.config=$(CONFIG) --policy.dir=$(CKPT)

serve-auto:
	@if [ -z "$(CKPT)" ]; then \
		echo "$(BOLD)Error:$(RESET) CKPT is required"; \
		echo "Usage: make serve-auto CKPT=<dir> [PORT=8800]"; \
		exit 1; \
	fi
	$(PYTHON) scripts/serve_policy.py --port $(PORT) policy:auto-checkpoint --policy.dir=$(CKPT)
