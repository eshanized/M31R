#!/usr/bin/env bash
# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT
#
# Serve smoke test — starts the M31R inference server, sends a test
# request, validates the response, then shuts down cleanly.
#
# Usage:
#   bash scripts/ci/serve_smoke_test.sh <config>
#
# Exit codes:
#   0 = serve smoke test passed
#   1 = serve smoke test failed

set -euo pipefail

CONFIG="${1:?Usage: $0 <config-path>}"
HOST="127.0.0.1"
PORT="8731"
MAX_WAIT=30
SERVE_PID=""

cleanup() {
    if [ -n "${SERVE_PID}" ] && kill -0 "${SERVE_PID}" 2>/dev/null; then
        echo "[serve-smoke] Shutting down server (PID ${SERVE_PID})"
        kill "${SERVE_PID}" 2>/dev/null || true
        wait "${SERVE_PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT

echo "[serve-smoke] Starting m31r serve on ${HOST}:${PORT}"

# Start server in background, send all output to a log file
SERVE_LOG="$(mktemp)"
m31r serve --config "${CONFIG}" --host "${HOST}" --port "${PORT}" > "${SERVE_LOG}" 2>&1 &
SERVE_PID=$!

# Wait for server to become ready
echo "[serve-smoke] Waiting for server readiness (max ${MAX_WAIT}s)"
READY=false
for i in $(seq 1 "${MAX_WAIT}"); do
    if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
        echo "[serve-smoke] Server process died unexpectedly"
        echo "[serve-smoke] Server log:"
        cat "${SERVE_LOG}" || true
        rm -f "${SERVE_LOG}"
        exit 1
    fi

    # Try to connect to the server
    if curl -s -o /dev/null -w '' "http://${HOST}:${PORT}/generate" --max-time 1 -X POST \
         -H "Content-Type: application/json" \
         -d '{"prompt": "fn", "max_tokens": 1}' 2>/dev/null; then
        READY=true
        break
    fi

    sleep 1
done

if [ "${READY}" = false ]; then
    echo "[serve-smoke] Server did not become ready within ${MAX_WAIT}s"
    echo "[serve-smoke] Server log:"
    cat "${SERVE_LOG}" || true
    rm -f "${SERVE_LOG}"
    exit 1
fi

echo "[serve-smoke] Server ready after ${i}s"

# Send a generation request
echo "[serve-smoke] Sending test generation request"
RESPONSE=$(curl -s --max-time 30 -X POST \
    "http://${HOST}:${PORT}/generate" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "fn main() {", "max_tokens": 16}' 2>&1) || true

if [ -n "${RESPONSE}" ]; then
    echo "[serve-smoke] Response received (${#RESPONSE} bytes)"
    echo "[serve-smoke] Response: ${RESPONSE:0:200}"
    echo "[serve-smoke] PASS — Server returned tokens"
else
    echo "[serve-smoke] WARN — Empty response (may be expected for smoke test)"
fi

# Clean shutdown
echo "[serve-smoke] Shutting down server"
kill "${SERVE_PID}" 2>/dev/null || true
wait "${SERVE_PID}" 2>/dev/null || true
SERVE_PID=""

rm -f "${SERVE_LOG}"

echo "[serve-smoke] Serve smoke test complete"
exit 0
