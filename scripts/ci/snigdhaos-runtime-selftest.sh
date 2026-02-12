#!/usr/bin/env bash
# Author : Snigdha OS Team
# SPDX-License-Identifier: MIT
#
# M31R Runtime Self-Test — fully automated smoke test for the inference
# server.  Starts the server, runs the Python client and the pytest
# suite, then shuts down cleanly.
#
# Usage:
#   bash scripts/ci/snigdhaos-runtime-selftest.sh [--model-dir DIR] [--port PORT] [--config PATH]
#
# Exit codes:
#   0 = all checks passed
#   1 = one or more checks failed

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────
MODEL_DIR="release/0.1.0"
PORT=18731
HOST="127.0.0.1"
CONFIG="configs/train_tiny.yaml"
MAX_STARTUP_WAIT=60
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SERVE_PID=""
SERVE_LOG=""
EXIT_CODE=0

# ─── Argument parsing ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --port)      PORT="$2";      shift 2 ;;
        --config)    CONFIG="$2";    shift 2 ;;
        *)           echo "[selftest] Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── Cleanup trap ────────────────────────────────────────────────────
cleanup() {
    local rc=$?
    echo "[selftest] Cleanup (exit_code=${rc})"

    # Try graceful shutdown via /shutdown endpoint first
    if [ -n "${SERVE_PID}" ] && kill -0 "${SERVE_PID}" 2>/dev/null; then
        echo "[selftest] Sending POST /shutdown"
        curl -s -X POST "http://${HOST}:${PORT}/shutdown" --max-time 3 >/dev/null 2>&1 || true
        sleep 2

        # Force kill if still alive
        if kill -0 "${SERVE_PID}" 2>/dev/null; then
            echo "[selftest] Force-killing server (PID ${SERVE_PID})"
            kill -9 "${SERVE_PID}" 2>/dev/null || true
        fi
        wait "${SERVE_PID}" 2>/dev/null || true
    fi

    # Print server log on failure
    if [ "${EXIT_CODE}" -ne 0 ] && [ -n "${SERVE_LOG}" ] && [ -f "${SERVE_LOG}" ]; then
        echo "[selftest] ──── Server log (last 50 lines) ────"
        tail -50 "${SERVE_LOG}" 2>/dev/null || true
        echo "[selftest] ──── End server log ────"
    fi

    # Clean up temp log
    if [ -n "${SERVE_LOG}" ]; then
        rm -f "${SERVE_LOG}"
    fi
}

trap cleanup EXIT

# ─── Banner ──────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════"
echo "  M31R Runtime Self-Test"
echo "  model-dir : ${MODEL_DIR}"
echo "  port      : ${PORT}"
echo "  config    : ${CONFIG}"
echo "══════════════════════════════════════════════════════════"

# ─── Validate prerequisites ─────────────────────────────────────────
if ! command -v m31r >/dev/null 2>&1; then
    echo "[selftest] FATAL: 'm31r' CLI not found on PATH"
    EXIT_CODE=1
    exit 1
fi

if ! command -v python >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
    echo "[selftest] FATAL: python not found on PATH"
    EXIT_CODE=1
    exit 1
fi

PYTHON_CMD="python3"
if ! command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python"
fi

# ─── Step 1: Start server ───────────────────────────────────────────
echo "[selftest] Step 1: Starting m31r serve on ${HOST}:${PORT}"
SERVE_LOG="$(mktemp)"

cd "${PROJECT_ROOT}"

m31r serve \
    --config "${CONFIG}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --device cpu \
    > "${SERVE_LOG}" 2>&1 &
SERVE_PID=$!

echo "[selftest] Server started (PID ${SERVE_PID})"

# ─── Step 2: Wait for server readiness ──────────────────────────────
echo "[selftest] Step 2: Waiting for server (max ${MAX_STARTUP_WAIT}s)"
READY=false

for i in $(seq 1 "${MAX_STARTUP_WAIT}"); do
    # Check process is still alive
    if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
        echo "[selftest] FATAL: Server process died during startup"
        EXIT_CODE=1
        exit 1
    fi

    # Check /status endpoint
    STATUS_BODY=$(curl -s --max-time 2 "http://${HOST}:${PORT}/status" 2>/dev/null) || true
    if echo "${STATUS_BODY}" | grep -q '"model_loaded": true' 2>/dev/null; then
        READY=true
        break
    fi
    # Also try without space after colon
    if echo "${STATUS_BODY}" | grep -q '"model_loaded":true' 2>/dev/null; then
        READY=true
        break
    fi

    sleep 1
done

if [ "${READY}" = false ]; then
    echo "[selftest] FATAL: Server did not become ready within ${MAX_STARTUP_WAIT}s"
    EXIT_CODE=1
    exit 1
fi

echo "[selftest] Server ready after ${i}s"

# ─── Step 3: Run Python self-test client ────────────────────────────
echo "[selftest] Step 3: Running runtime_client.py"

if ! "${PYTHON_CMD}" "${PROJECT_ROOT}/scripts/ci/runtime_client.py" \
    --base-url "http://${HOST}:${PORT}"; then
    echo "[selftest] FAIL: runtime_client.py exited non-zero"
    EXIT_CODE=1
    exit 1
fi

echo "[selftest] runtime_client.py PASSED"

# ─── Step 4: Run pytest suite ───────────────────────────────────────
echo "[selftest] Step 4: Running pytest test_runtime_smoke.py"

export M31R_TEST_PORT="${PORT}"

if ! "${PYTHON_CMD}" -m pytest \
    "${PROJECT_ROOT}/tests/runtime/test_runtime_smoke.py" \
    -v --tb=short -x; then
    echo "[selftest] FAIL: pytest exited non-zero"
    EXIT_CODE=1
    exit 1
fi

echo "[selftest] pytest PASSED"

# ─── Step 5: Clean shutdown ─────────────────────────────────────────
echo "[selftest] Step 5: Shutting down server"

curl -s -X POST "http://${HOST}:${PORT}/shutdown" --max-time 5 >/dev/null 2>&1 || true
sleep 2

if kill -0 "${SERVE_PID}" 2>/dev/null; then
    echo "[selftest] Server still alive — force killing"
    kill -9 "${SERVE_PID}" 2>/dev/null || true
fi

wait "${SERVE_PID}" 2>/dev/null || true
SERVE_PID=""

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  M31R Runtime Self-Test — ALL CHECKS PASSED"
echo "══════════════════════════════════════════════════════════"
exit 0
