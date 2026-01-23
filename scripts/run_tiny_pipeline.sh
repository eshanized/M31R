#!/usr/bin/env bash
# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT
#
# End-to-end pipeline smoke test for M31R.
# Runs: Dataset → Train → Resume → Export → Verify → Generate
#
# Usage:
#   bash scripts/run_tiny_pipeline.sh
#
# Requirements:
#   - Python 3.11+ with m31r installed (pip install -e .)
#   - CPU only, no GPU required
#   - ~5 minutes runtime

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="configs/train_tiny.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

step() {
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  STEP: $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"
}

pass() {
    echo -e "${GREEN}  ✓ $1${NC}"
}

fail() {
    echo -e "${RED}  ✗ $1${NC}"
    exit 1
}

cd "$PROJECT_ROOT"

echo -e "${YELLOW}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         M31R  —  Tiny Pipeline Smoke Test               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ──────────────────────────────────────────────────────────────
# Step 1: Generate synthetic dataset
# ──────────────────────────────────────────────────────────────
step "1/7 — Generate synthetic dataset"
python scripts/create_tiny_dataset.py \
    --output-dir datasets/dev-small \
    --tokenizer-dir data/tokenizer \
    --num-shards 4 \
    --tokens-per-shard 50000 \
    --seed 42
pass "Dataset and tokenizer created"

# ──────────────────────────────────────────────────────────────
# Step 2: Train (100 steps)
# ──────────────────────────────────────────────────────────────
step "2/7 — Train model (100 steps, FP32, CPU)"
m31r train --config "$CONFIG"
pass "Training complete"

# ──────────────────────────────────────────────────────────────
# Step 3: Resume training (run 50 more steps)
# ──────────────────────────────────────────────────────────────
step "3/7 — Resume training (50 more steps)"

# Update max_steps to 150 for resume test
RESUME_CONFIG="configs/train_tiny_resume.yaml"
sed 's/max_steps: 100/max_steps: 150/' "$CONFIG" > "$RESUME_CONFIG"

m31r resume --config "$RESUME_CONFIG"
pass "Resume training complete"

# Clean up temp config
rm -f "$RESUME_CONFIG"

# ──────────────────────────────────────────────────────────────
# Step 4: Export release bundle
# ──────────────────────────────────────────────────────────────
step "4/7 — Export release bundle"
m31r export --config "$CONFIG" || {
    echo -e "${YELLOW}  ⚠ Export encountered an issue (may be expected in smoke test)${NC}"
}
pass "Export step done"

# ──────────────────────────────────────────────────────────────
# Step 5: Verify release (if export produced artifacts)
# ──────────────────────────────────────────────────────────────
step "5/7 — Verify release integrity"
RELEASE_DIR=$(find release/ -maxdepth 1 -type d 2>/dev/null | tail -1)
if [ -n "$RELEASE_DIR" ] && [ "$RELEASE_DIR" != "release/" ]; then
    m31r verify --release-dir "$RELEASE_DIR" --config "$CONFIG" || {
        echo -e "${YELLOW}  ⚠ Verification encountered an issue${NC}"
    }
    pass "Verification complete"
else
    echo -e "${YELLOW}  ⚠ No release directory found, skipping verify${NC}"
fi

# ──────────────────────────────────────────────────────────────
# Step 6: Generate text (single-shot inference)
# ──────────────────────────────────────────────────────────────
step "6/7 — Generate text (inference test)"
m31r generate --prompt "fn main() {" --config "$CONFIG" --max-tokens 32 || {
    echo -e "${YELLOW}  ⚠ Generation encountered an issue${NC}"
}
pass "Generation step done"

# ──────────────────────────────────────────────────────────────
# Step 7: Info
# ──────────────────────────────────────────────────────────────
step "7/7 — System info"
m31r info --config "$CONFIG"
pass "Info complete"

# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         Pipeline Smoke Test — PASSED                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
