#!/usr/bin/env bash
# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT
#
# CI test runner — runs the full M31R test suite.
# Exit code 0 = all tests passed, non-zero = failure.

set -euo pipefail

echo "=== M31R Test Suite ==="
echo "Running: pytest tests/ -v --tb=short"
echo ""

python -m pytest tests/ -v --tb=short

echo ""
echo "=== All tests passed ==="
