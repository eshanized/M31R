#!/usr/bin/env bash
# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT
#
# Release verification — verifies a packaged release directory.
# Usage: scripts/ci/release_verify.sh <release-dir>
# Exit code 0 = release valid, non-zero = verification failed.

set -euo pipefail

RELEASE_DIR="${1:?Usage: $0 <release-dir>}"

echo "=== M31R Release Verification ==="
echo "Release directory: ${RELEASE_DIR}"
echo ""

if [ ! -d "${RELEASE_DIR}" ]; then
    echo "ERROR: Release directory not found: ${RELEASE_DIR}"
    exit 1
fi

python -m m31r verify --release-dir "${RELEASE_DIR}" --log-level INFO

echo ""
echo "=== Release verification complete ==="
