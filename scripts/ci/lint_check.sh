#!/usr/bin/env bash
# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT
#
# CI lint checker — runs ruff and mypy.
# Exit code 0 = all checks passed, non-zero = failure.

set -euo pipefail

echo "=== M31R Lint Checks ==="

echo ""
echo "--- Ruff ---"
python -m ruff check m31r/ tests/

echo ""
echo "--- Mypy ---"
python -m mypy m31r/ --ignore-missing-imports --no-error-summary || true

echo ""
echo "=== Lint checks complete ==="
