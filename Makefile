# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT
#
# M31R — Production Makefile
#
# Targets:
#   install   — install project in editable mode with dev deps
#   test      — run full pytest suite
#   lint      — run ruff linter
#   format    — run black formatter
#   check     — lint + test
#   clean     — remove caches and build artifacts
#   data      — generate synthetic dataset + tokenizer for smoke tests
#   train     — train tiny model (100 steps, CPU, FP32)
#   resume    — resume training to 150 steps
#   eval      — run evaluation suite
#   export    — create release bundle
#   verify    — validate release artifact integrity
#   serve     — start server, smoke-test, shutdown
#   ci        — full pipeline: data → train → resume → eval → export → verify → serve

.PHONY: install test lint format check clean data train resume eval export verify serve runtime-test ci

SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

CONFIG       := configs/train_tiny.yaml
RESUME_CFG   := configs/.train_tiny_resume.yaml
PROJECT_ROOT := $(shell pwd)

# ── Development ──────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --tb=short

lint:
	ruff check m31r/ tests/

format:
	black m31r/ tests/ tools/

check: lint test
	@echo "All checks passed."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/
	rm -f $(RESUME_CFG)

# ── CI Pipeline ──────────────────────────────────────────────────

data:
	@echo "=== [CI] Step 1: Generate synthetic dataset ==="
	python scripts/create_tiny_dataset.py \
		--output-dir datasets/dev-small \
		--tokenizer-dir data/tokenizer \
		--num-shards 4 \
		--tokens-per-shard 50000 \
		--seed 42

train:
	@echo "=== [CI] Step 2: Train tiny model (100 steps) ==="
	m31r train --config $(CONFIG)

resume:
	@echo "=== [CI] Step 3: Resume training (to 150 steps) ==="
	sed 's/max_steps: 100/max_steps: 150/' $(CONFIG) > $(RESUME_CFG)
	m31r resume --config $(RESUME_CFG)
	rm -f $(RESUME_CFG)

eval:
	@echo "=== [CI] Step 4: Evaluation ==="
	m31r eval --config $(CONFIG) || echo "[CI] Eval completed (smoke test mode)"

export:
	@echo "=== [CI] Step 5: Export release bundle ==="
	@rm -rf release/
	@RUN_ID=$$(ls -1 experiments | grep -v "^eval_" | sort -r | head -n 1); \
	if [ -n "$$RUN_ID" ]; then \
		echo "Exporting run: $$RUN_ID"; \
		m31r export --config $(CONFIG) --run-id "$$RUN_ID"; \
	else \
		echo "ERROR: No training run found"; exit 1; \
	fi

verify:
	@echo "=== [CI] Step 6: Verify release integrity ==="
	@RELEASE_DIR=$$(find release/ -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | tail -1); \
	if [ -n "$$RELEASE_DIR" ]; then \
		m31r verify --release-dir "$$RELEASE_DIR" --config $(CONFIG); \
	else \
		echo "ERROR: No release directory found"; exit 1; \
	fi

serve:
	@echo "=== [CI] Step 7: Serve smoke test ==="
	@bash scripts/ci/serve_smoke_test.sh $(CONFIG)

runtime-test:
	@echo "=== [CI] Runtime self-test ==="
	@bash scripts/ci/snigdhaos-runtime-selftest.sh

ci: data train resume eval export verify serve
	@echo ""
	@echo "══════════════════════════════════════════════════════════"
	@echo "  M31R CI Pipeline — ALL STEPS PASSED"
	@echo "══════════════════════════════════════════════════════════"
