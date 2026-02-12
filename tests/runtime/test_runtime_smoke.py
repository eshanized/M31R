#!/usr/bin/env python3
# Author : Snigdha OS Team
# SPDX-License-Identifier: MIT

"""
Pytest runtime smoke tests for the M31R inference server.

These tests hit a **running** server and validate every endpoint,
determinism, and latency.  The server must be started externally
(e.g. by snigdhaos-runtime-selftest.sh) before running this suite.

Port is read from the ``M31R_TEST_PORT`` environment variable
(default: 18731).

Usage:
    M31R_TEST_PORT=18731 python -m pytest tests/runtime/test_runtime_smoke.py -v
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# HTTP client — prefer httpx, fall back to requests
# ---------------------------------------------------------------------------
try:
    import httpx  # type: ignore[import-untyped]

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

try:
    import requests as _requests_mod  # type: ignore[import-untyped]

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

logger: logging.Logger = logging.getLogger("m31r.test_runtime_smoke")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DEFAULT_PORT = 18731
_TIMEOUT_S = 30.0
_MAX_LATENCY_S = 5.0


@pytest.fixture(scope="module")
def base_url() -> str:
    """Resolve the server base URL from the environment."""
    port = int(os.environ.get("M31R_TEST_PORT", str(_DEFAULT_PORT)))
    return f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _post(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    if _HAS_HTTPX:
        with httpx.Client(timeout=_TIMEOUT_S) as client:
            resp = client.post(url, json=payload)
            return resp.status_code, resp.json()
    elif _HAS_REQUESTS:
        resp = _requests_mod.post(url, json=payload, timeout=_TIMEOUT_S)
        return resp.status_code, resp.json()
    else:
        pytest.skip("Neither httpx nor requests is installed")
        return 0, {}  # unreachable


def _get(url: str) -> tuple[int, dict[str, Any]]:
    if _HAS_HTTPX:
        with httpx.Client(timeout=_TIMEOUT_S) as client:
            resp = client.get(url)
            return resp.status_code, resp.json()
    elif _HAS_REQUESTS:
        resp = _requests_mod.get(url, timeout=_TIMEOUT_S)
        return resp.status_code, resp.json()
    else:
        pytest.skip("Neither httpx nor requests is installed")
        return 0, {}


# ---------------------------------------------------------------------------
# Prompt payloads
# ---------------------------------------------------------------------------

_GENERATE_PLAIN = {
    "prompt": "write a rust function to print fibonacci numbers",
    "max_tokens": 64,
    "temperature": 0.0,
    "seed": 42,
}

_GENERATE_COMMENT = {
    "prompt": "// print fibonacci numbers\nfn fibonacci(",
    "max_tokens": 64,
    "temperature": 0.0,
    "seed": 42,
}

_FIM_PAYLOAD = {
    "prefix": " fn add(a:i32,b:i32)->i32{ ",
    "suffix": " } ",
    "max_tokens": 64,
    "temperature": 0.0,
    "seed": 42,
}

_COMPLETION_PAYLOAD = {
    "prefix": "fn main(){",
    "max_tokens": 64,
    "temperature": 0.0,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Shared validation helper
# ---------------------------------------------------------------------------


def _assert_valid_generation(body: dict[str, Any]) -> None:
    """Common assertions for any generation response."""
    text = body.get("text", "")
    assert isinstance(text, str), f"text is not a string: {type(text)}"
    assert len(text) > 0, "Response text is empty"
    # UTF-8 validity
    text.encode("utf-8")
    assert body.get("tokens_generated", 0) > 0, f"tokens_generated={body.get('tokens_generated')}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStatusEndpoint:
    """Verify the server is healthy."""

    def test_status_returns_200(self, base_url: str) -> None:
        code, body = _get(f"{base_url}/status")
        assert code == 200, f"Expected 200 got {code}"

    def test_model_loaded(self, base_url: str) -> None:
        _, body = _get(f"{base_url}/status")
        assert body.get("model_loaded") is True, f"model_loaded={body.get('model_loaded')}"


class TestGenerateEndpoint:
    """POST /generate with various prompt styles."""

    def test_plain_english_prompt(self, base_url: str) -> None:
        code, body = _post(f"{base_url}/generate", _GENERATE_PLAIN)
        assert code == 200, f"HTTP {code}: {body}"
        _assert_valid_generation(body)

    def test_comment_style_prompt(self, base_url: str) -> None:
        code, body = _post(f"{base_url}/generate", _GENERATE_COMMENT)
        assert code == 200, f"HTTP {code}: {body}"
        _assert_valid_generation(body)


class TestFIMEndpoint:
    """POST /fim fill-in-the-middle."""

    def test_fim_returns_valid_text(self, base_url: str) -> None:
        code, body = _post(f"{base_url}/fim", _FIM_PAYLOAD)
        assert code == 200, f"HTTP {code}: {body}"
        _assert_valid_generation(body)


class TestCompletionEndpoint:
    """POST /completion code completion."""

    def test_completion_returns_valid_text(self, base_url: str) -> None:
        code, body = _post(f"{base_url}/completion", _COMPLETION_PAYLOAD)
        assert code == 200, f"HTTP {code}: {body}"
        _assert_valid_generation(body)


class TestDeterminism:
    """Same prompt at temperature=0 must produce identical output."""

    def test_deterministic_output(self, base_url: str) -> None:
        _, body_a = _post(f"{base_url}/generate", _GENERATE_PLAIN)
        _, body_b = _post(f"{base_url}/generate", _GENERATE_PLAIN)

        text_a = body_a.get("text", "")
        text_b = body_b.get("text", "")

        assert len(text_a) > 0, "First response is empty"
        assert text_a == text_b, (
            f"Non-deterministic output at temperature=0:\n"
            f"  run1: {text_a[:120]!r}\n"
            f"  run2: {text_b[:120]!r}"
        )


class TestLatency:
    """Total request time must be under the threshold."""

    def test_latency_under_threshold(self, base_url: str) -> None:
        import time

        t0 = time.monotonic()
        code, body = _post(f"{base_url}/generate", _GENERATE_PLAIN)
        elapsed = time.monotonic() - t0

        assert code == 200, f"HTTP {code}"
        assert elapsed < _MAX_LATENCY_S, (
            f"Latency {elapsed:.2f}s exceeds threshold {_MAX_LATENCY_S}s"
        )


class TestUTF8Validity:
    """Generated text must be valid UTF-8."""

    def test_output_encodes_as_utf8(self, base_url: str) -> None:
        _, body = _post(f"{base_url}/generate", _GENERATE_PLAIN)
        text = body.get("text", "")
        assert isinstance(text, str)
        encoded = text.encode("utf-8")
        assert len(encoded) > 0, "Encoded text is empty"
