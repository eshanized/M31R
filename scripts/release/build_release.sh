#!/usr/bin/env bash
# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT
#
# End-to-end release build automation.
# Usage: scripts/release/build_release.sh <version>
#
# Steps:
#   1. Validate environment
#   2. Run full test suite
#   3. Create release bundle (m31r export)
#   4. Verify release bundle (m31r verify)

set -euo pipefail

VERSION="${1:?Usage: $0 <version>}"

echo "=== M31R Release Build v${VERSION} ==="
echo ""

# Step 1: Environment validation
echo "--- Step 1: Environment Validation ---"
python -m m31r info --log-level INFO
echo ""

# Step 2: Full test suite
echo "--- Step 2: Running Test Suite ---"
python -m pytest tests/ -v --tb=short
echo ""

# Step 3: Create release bundle
echo "--- Step 3: Creating Release Bundle ---"
python -m m31r export --version "${VERSION}" --log-level INFO
echo ""

# Step 4: Verify release
RELEASE_DIR="release/${VERSION}"
echo "--- Step 4: Verifying Release ---"
python -m m31r verify --release-dir "${RELEASE_DIR}" --log-level INFO
echo ""

echo "=== Release v${VERSION} built and verified successfully ==="
