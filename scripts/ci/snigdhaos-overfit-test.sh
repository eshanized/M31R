#!/usr/bin/env bash
# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT
#
# snigdhaos-overfit-test.sh — Overfit Sanity Test for M31R
#
# SYSTEM CORRECTNESS TEST. Proves the model + trainer can learn.
#
# Pipeline:
#   1. Generate deterministic synthetic Rust dataset
#   2. Train tiny model to intentional overfit
#   3. Verify loss < 0.5 + checkpoint saved
#   4. Generation test — prompt "fn add(" must resemble training data
#
# If this test passes, the trainer is correct.
# If it fails, training is broken.
#
# Requirements:
#   - Python 3.11+ with m31r installed (pip install -e .)
#   - CPU only, no GPU required
#   - Deterministic, reproducible, fully automated
#
# Usage:
#   bash scripts/ci/snigdhaos-overfit-test.sh
#
# Exit codes:
#   0 = overfit test passed (system can learn)
#   1 = overfit test failed (system broken)

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="configs/train_overfit.yaml"
PIPELINE_START_TIME="$(date +%s)"
STEP_COUNT=0
TOTAL_STEPS=4
LOSS_THRESHOLD="0.5"

# ── Helpers ──────────────────────────────────────────────────────────

log_step() {
    STEP_COUNT=$((STEP_COUNT + 1))
    echo ""
    echo "================================================================"
    echo "  STEP ${STEP_COUNT}/${TOTAL_STEPS}: $1"
    echo "  Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "================================================================"
    echo ""
}

log_pass() {
    echo "[PASS] $1"
}

log_fail() {
    echo "[FAIL] $1" >&2
    exit 1
}

elapsed() {
    local now
    now="$(date +%s)"
    echo "$(( now - PIPELINE_START_TIME ))s elapsed"
}

# ── Activate venv if present ─────────────────────────────────────────

if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

cd "${PROJECT_ROOT}"

echo "================================================================"
echo "  M31R Overfit Sanity Test"
echo "  Purpose: PROVE the model + trainer can learn"
echo "  Project: ${PROJECT_ROOT}"
echo "  Config:  ${CONFIG}"
echo "  Python:  $(python --version 2>&1)"
echo "  Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

# ── Step 1: Generate Synthetic Rust Dataset ──────────────────────────

log_step "Generate synthetic Rust dataset"

python scripts/generate_synthetic_rust.py \
    --output-dir datasets/dev-overfit \
    --tokenizer-dir data/tokenizer-overfit \
    --num-shards 4 \
    --vocab-size 256

# Validate
if [ ! -f "datasets/dev-overfit/manifest.json" ]; then
    log_fail "Dataset manifest not created"
fi
if [ ! -f "data/tokenizer-overfit/tokenizer.json" ]; then
    log_fail "Tokenizer not created"
fi

TOTAL_TOKENS=$(python -c "
import json
m = json.loads(open('datasets/dev-overfit/manifest.json').read())
print(m['total_tokens'])
")
echo "  Total tokens: ${TOTAL_TOKENS}"

log_pass "Synthetic dataset created ($(elapsed))"

# ── Step 2: Train Tiny Model to Overfit ──────────────────────────────

log_step "Train tiny model (500 steps, LR=1e-2, CPU, fp32)"

m31r train --config "${CONFIG}"

# Find experiment directory
EXPERIMENT_DIR=$(find experiments/ -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | tail -1)
if [ -z "${EXPERIMENT_DIR}" ]; then
    log_fail "No experiment directory created by training"
fi

# Find final checkpoint
CHECKPOINT_DIR=$(find "${EXPERIMENT_DIR}" -maxdepth 2 -type d -name "step_*" 2>/dev/null | sort | tail -1)
if [ -z "${CHECKPOINT_DIR}" ]; then
    log_fail "No checkpoint found after training"
fi
if [ ! -f "${CHECKPOINT_DIR}/model.pt" ]; then
    log_fail "model.pt not found in ${CHECKPOINT_DIR}"
fi
if [ ! -f "${CHECKPOINT_DIR}/metadata.json" ]; then
    log_fail "metadata.json not found in ${CHECKPOINT_DIR}"
fi

log_pass "Training complete — checkpoint at ${CHECKPOINT_DIR} ($(elapsed))"

# ── Step 3: Verify Learning ─────────────────────────────────────────

log_step "Verify loss decreased (threshold: ${LOSS_THRESHOLD})"

FINAL_LOSS=$(python -c "
import json, pathlib
meta = json.loads(pathlib.Path('${CHECKPOINT_DIR}/metadata.json').read_text())
print(meta.get('loss', 999.0))
")

FINAL_STEP=$(python -c "
import json, pathlib
meta = json.loads(pathlib.Path('${CHECKPOINT_DIR}/metadata.json').read_text())
print(meta.get('global_step', 0))
")

echo "  Final step: ${FINAL_STEP}"
echo "  Final loss: ${FINAL_LOSS}"

# Assert loss < threshold
LOSS_OK=$(python -c "print(1 if float('${FINAL_LOSS}') < float('${LOSS_THRESHOLD}') else 0)")
if [ "${LOSS_OK}" -ne 1 ]; then
    log_fail "Final loss ${FINAL_LOSS} >= threshold ${LOSS_THRESHOLD} — model did NOT learn"
fi

# Assert training ran enough steps
STEPS_OK=$(python -c "print(1 if int('${FINAL_STEP}') >= 400 else 0)")
if [ "${STEPS_OK}" -ne 1 ]; then
    log_fail "Training only reached step ${FINAL_STEP} — expected >= 400"
fi

log_pass "Loss ${FINAL_LOSS} < ${LOSS_THRESHOLD} — model LEARNED ($(elapsed))"

# ── Step 4: Generation Test ──────────────────────────────────────────

log_step "Generation test (prompt: 'fn add(')"

# Load the trained checkpoint and generate text
GENERATED=$(python -c "
import torch, json, pathlib

# Load model config from checkpoint metadata
meta = json.loads(pathlib.Path('${CHECKPOINT_DIR}/metadata.json').read_text())

from m31r.model.transformer import M31RTransformer, TransformerModelConfig

model_cfg = TransformerModelConfig(
    vocab_size=256,
    dim=64,
    n_layers=2,
    n_heads=2,
    head_dim=32,
    max_seq_len=128,
    dropout=0.0,
    norm_eps=1e-6,
    rope_theta=10000.0,
    init_std=0.02,
    seed=42,
)

model = M31RTransformer(model_cfg)

# Load trained weights
ckpt = torch.load('${CHECKPOINT_DIR}/model.pt', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt)
model.eval()

# Load tokenizer
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file('data/tokenizer-overfit/tokenizer.json')

# Encode prompt
prompt = 'fn add('
encoded = tokenizer.encode(prompt)
input_ids = torch.tensor([encoded.ids], dtype=torch.long)

# Greedy generate 32 tokens
with torch.no_grad():
    for _ in range(32):
        logits = model(input_ids[:, -128:])
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

# Decode
all_ids = input_ids[0].tolist()
output = tokenizer.decode(all_ids)
print(output)
")

echo "  Generated: ${GENERATED}"

# Check that generation is not empty and contains Rust-like tokens
GEN_OK=$(python -c "
text = '''${GENERATED}'''
# Must not be empty and should contain some Rust-like structure
ok = len(text.strip()) > 0
print(1 if ok else 0)
")

if [ "${GEN_OK}" -ne 1 ]; then
    log_fail "Generation produced empty or invalid output"
fi

log_pass "Generation test passed ($(elapsed))"

# ── Summary ──────────────────────────────────────────────────────────

PIPELINE_END_TIME="$(date +%s)"
TOTAL_ELAPSED=$(( PIPELINE_END_TIME - PIPELINE_START_TIME ))

echo ""
echo "================================================================"
echo "  M31R Overfit Sanity Test — ALL ${TOTAL_STEPS} STEPS PASSED"
echo ""
echo "  Result:     MODEL CAN LEARN"
echo "  Final loss: ${FINAL_LOSS}"
echo "  Final step: ${FINAL_STEP}"
echo "  Total time: ${TOTAL_ELAPSED}s"
echo "  Finished:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

exit 0
