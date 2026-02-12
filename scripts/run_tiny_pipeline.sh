#!/usr/bin/env bash
# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT
#
# End-to-end pipeline smoke test for M31R.
#
# This is a convenience wrapper around the production CI pipeline.
# For CI use, prefer: make ci or scripts/ci/snigdhaos-ci-pipeline.sh
#
# Usage:
#   bash scripts/run_tiny_pipeline.sh
#
# Requirements:
#   - Python 3.11+ with m31r installed (pip install -e .)
#   - CPU only, no GPU required

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "M31R — Tiny Pipeline Smoke Test"
echo "Delegating to production CI pipeline..."
echo ""

exec bash scripts/ci/snigdhaos-ci-pipeline.sh
