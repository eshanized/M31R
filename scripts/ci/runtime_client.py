#!/usr/bin/env python3
# Author : Snigdha OS Team
# SPDX-License-Identifier: MIT

"""
Runtime self-test client for M31R inference server.

Exercises every HTTP endpoint, validates responses, checks determinism
at temperature=0, and measures latency.  Exits 0 only if every check
passes — any failure is an immediate exit 1.

This script is designed to run in CI on every commit.  It talks to an
already-running server; the shell wrapper handles lifecycle.

Usage:
    python scripts/ci/runtime_client.py --base-url http://127.0.0.1:18731
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# HTTP client — prefer httpx, fall back to requests
# ---------------------------------------------------------------------------
try:
    import httpx

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

try:
    import requests as _requests_mod

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

if not _HAS_HTTPX and not _HAS_REQUESTS:
    sys.stderr.write("FATAL: neither httpx nor requests is installed\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Logging — structured JSON, no print()
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "module": record.name,
            "msg": record.getMessage(),
        }
        # Standard attributes to exclude from extra fields
        exclude_attrs = {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "module",
            "msecs", "message", "msg", "name", "pathname", "process",
            "processName", "relativeCreated", "stack_info", "thread",
            "threadName", "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in exclude_attrs:
                log_record[key] = value
        return json.dumps(log_record)


def _setup_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger = logging.getLogger("m31r.runtime_selftest")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger: logging.Logger = _setup_logging()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestCase:
    """One prompt + endpoint to exercise."""

    name: str
    endpoint: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class TestResult:
    """Outcome of a single test."""

    name: str
    passed: bool
    elapsed_s: float
    detail: str = ""


@dataclass
class Report:
    """Aggregate report for the entire self-test run."""

    results: list[TestResult] = field(default_factory=list)
    determinism_passed: bool = False
    latency_passed: bool = False

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results) and self.determinism_passed and self.latency_passed


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _post(url: str, payload: dict[str, Any], timeout: float = 30.0) -> tuple[int, dict[str, Any], float]:
    """Send POST, return (status_code, json_body, elapsed_seconds)."""
    t0 = time.monotonic()

    if _HAS_HTTPX:
        client = httpx.Client(timeout=timeout)
        try:
            resp = client.post(url, json=payload)
            elapsed = time.monotonic() - t0
            return resp.status_code, resp.json(), elapsed
        finally:
            client.close()
    else:
        resp = _requests_mod.post(url, json=payload, timeout=timeout)  # type: ignore[union-attr]
        elapsed = time.monotonic() - t0
        return resp.status_code, resp.json(), elapsed


def _get(url: str, timeout: float = 10.0) -> tuple[int, dict[str, Any]]:
    """Send GET, return (status_code, json_body)."""
    if _HAS_HTTPX:
        client = httpx.Client(timeout=timeout)
        try:
            resp = client.get(url)
            return resp.status_code, resp.json()
        finally:
            client.close()
    else:
        resp = _requests_mod.get(url, timeout=timeout)  # type: ignore[union-attr]
        return resp.status_code, resp.json()


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

PROMPT_CASES: list[TestCase] = [
    TestCase(
        name="plain_english",
        endpoint="/generate",
        payload={
            "prompt": "write a rust function to print fibonacci numbers",
            "max_tokens": 64,
            "temperature": 0.0,
            "seed": 42,
        },
    ),
    TestCase(
        name="comment_style",
        endpoint="/generate",
        payload={
            "prompt": "// print fibonacci numbers\nfn fibonacci(",
            "max_tokens": 64,
            "temperature": 0.0,
            "seed": 42,
        },
    ),
    TestCase(
        name="fim_style",
        endpoint="/fim",
        payload={
            "prefix": " fn add(a:i32,b:i32)->i32{ ",
            "suffix": " } ",
            "max_tokens": 64,
            "temperature": 0.0,
            "seed": 42,
        },
    ),
    TestCase(
        name="short_completion",
        endpoint="/completion",
        payload={
            "prefix": "fn main(){",
            "max_tokens": 64,
            "temperature": 0.0,
            "seed": 42,
        },
    ),
]

MAX_LATENCY_S: float = 5.0


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------


def _validate_response(
    name: str,
    status_code: int,
    body: dict[str, Any],
    elapsed: float,
) -> TestResult:
    """Check a single response satisfies all smoke-test invariants."""
    if status_code != 200:
        return TestResult(
            name=name,
            passed=False,
            elapsed_s=elapsed,
            detail=f"HTTP {status_code}: {body.get('error', 'unknown')}",
        )

    text = body.get("text", "")
    if not isinstance(text, str):
        return TestResult(
            name=name,
            passed=False,
            elapsed_s=elapsed,
            detail="Response text is not a string",
        )

    # UTF-8 validity — the JSON decoder already guarantees this, but be explicit
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return TestResult(
            name=name,
            passed=False,
            elapsed_s=elapsed,
            detail="Response text is not valid UTF-8",
        )

    tokens_generated = body.get("tokens_generated", 0)
    if tokens_generated <= 0:
        return TestResult(
            name=name,
            passed=False,
            elapsed_s=elapsed,
            detail=f"tokens_generated={tokens_generated} (expected > 0)",
        )

    return TestResult(name=name, passed=True, elapsed_s=elapsed, detail=f"OK ({tokens_generated} tokens)")


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------


def run_selftest(base_url: str) -> Report:
    """Execute the full self-test suite against a running server."""
    report = Report()

    # ------------------------------------------------------------------
    # 0. Status endpoint sanity check
    # ------------------------------------------------------------------
    logger.info("Checking /status endpoint", extra={"base_url": base_url})
    try:
        code, status_body = _get(f"{base_url}/status")
        if code != 200 or not status_body.get("model_loaded", False):
            report.results.append(
                TestResult(
                    name="status_check",
                    passed=False,
                    elapsed_s=0.0,
                    detail=f"Server not ready: HTTP {code}, body={status_body}",
                )
            )
            return report
        report.results.append(TestResult(name="status_check", passed=True, elapsed_s=0.0, detail="Server healthy"))
        logger.info("Server healthy", extra={"status": status_body})
    except Exception as exc:
        report.results.append(
            TestResult(name="status_check", passed=False, elapsed_s=0.0, detail=f"Connection error: {exc}")
        )
        return report

    # ------------------------------------------------------------------
    # 1. Prompt tests
    # ------------------------------------------------------------------
    latency_ok = True
    for tc in PROMPT_CASES:
        url = f"{base_url}{tc.endpoint}"
        logger.info("Running test case", extra={"test_name": tc.name, "endpoint": tc.endpoint})

        try:
            code, body, elapsed = _post(url, tc.payload)
        except Exception as exc:
            result = TestResult(name=tc.name, passed=False, elapsed_s=0.0, detail=f"Request failed: {exc}")
            report.results.append(result)
            logger.error("Test FAILED", extra={"test_name": tc.name, "detail": result.detail})
            continue

        result = _validate_response(tc.name, code, body, elapsed)
        report.results.append(result)

        if result.passed:
            logger.info(
                "Test PASSED",
                extra={"test_name": tc.name, "elapsed_s": round(elapsed, 3), "detail": result.detail},
            )
        else:
            logger.error("Test FAILED", extra={"test_name": tc.name, "detail": result.detail})

        if elapsed > MAX_LATENCY_S:
            latency_ok = False
            logger.error(
                "Latency exceeded",
                extra={"test_name": tc.name, "elapsed_s": round(elapsed, 3), "max_s": MAX_LATENCY_S},
            )

    report.latency_passed = latency_ok

    # ------------------------------------------------------------------
    # 2. Determinism test — same prompt, temperature=0, must match
    # ------------------------------------------------------------------
    logger.info("Running determinism test")
    determinism_prompt = PROMPT_CASES[0]  # plain_english
    det_url = f"{base_url}{determinism_prompt.endpoint}"

    try:
        _, body_a, _ = _post(det_url, determinism_prompt.payload)
        _, body_b, _ = _post(det_url, determinism_prompt.payload)

        text_a = body_a.get("text", "")
        text_b = body_b.get("text", "")

        if text_a == text_b:
            report.determinism_passed = True
            report.results.append(
                TestResult(name="determinism", passed=True, elapsed_s=0.0, detail="Outputs match")
            )
            logger.info("Determinism PASSED", extra={"text_length": len(text_a)})
        else:
            report.determinism_passed = False
            report.results.append(
                TestResult(
                    name="determinism",
                    passed=False,
                    elapsed_s=0.0,
                    detail=f"Outputs differ: len_a={len(text_a)}, len_b={len(text_b)}",
                )
            )
            logger.error(
                "Determinism FAILED",
                extra={"text_a_prefix": text_a[:80], "text_b_prefix": text_b[:80]},
            )
    except Exception as exc:
        report.determinism_passed = False
        report.results.append(
            TestResult(name="determinism", passed=False, elapsed_s=0.0, detail=f"Error: {exc}")
        )
        logger.error("Determinism test error", extra={"error": str(exc)})

    return report


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M31R runtime self-test client",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:18731",
        dest="base_url",
        help="Base URL of the running M31R server.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logger.info("Starting runtime self-test", extra={"base_url": args.base_url})

    report = run_selftest(args.base_url)

    # Emit final summary
    passed = sum(1 for r in report.results if r.passed)
    failed = sum(1 for r in report.results if not r.passed)

    summary: dict[str, Any] = {
        "total": len(report.results),
        "passed": passed,
        "failed": failed,
        "determinism": report.determinism_passed,
        "latency": report.latency_passed,
        "all_passed": report.all_passed,
    }
    logger.info("Self-test complete", extra=summary)

    if not report.all_passed:
        logger.error("RUNTIME SELF-TEST FAILED", extra={"failed_tests": [asdict(r) for r in report.results if not r.passed]})
        sys.exit(1)

    logger.info("RUNTIME SELF-TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
