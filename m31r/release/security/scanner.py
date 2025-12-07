# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Security scanner for M31R releases and source code.

Per 19_SECURITY_AND_SAFETY.md:
- Configs must not contain credentials, secrets, tokens, API keys (§19)
- Default: no telemetry, no automatic data collection (§27)
- Default: no network access for training/serving (§14)
- File permissions must be safe (§18)

This scanner performs static analysis — no code execution on scanned files.
"""

import logging
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from m31r.logging.logger import get_logger

_logger: logging.Logger = get_logger(__name__)


@dataclass(frozen=True)
class SecurityFinding:
    """A single security finding from a scan."""

    severity: str  # "high", "medium", "low"
    category: str
    file: str
    line: int
    message: str


# Patterns that suggest secrets or credentials in files
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("api_key", re.compile(r"""(?:api[_-]?key|apikey)\s*[=:]\s*['"][^'"]{8,}['"]""", re.IGNORECASE)),
    ("secret", re.compile(r"""(?:secret|password|passwd|pwd)\s*[=:]\s*['"][^'"]{8,}['"]""", re.IGNORECASE)),
    ("token", re.compile(r"""(?:token|auth_token|access_token)\s*[=:]\s*['"][^'"]{8,}['"]""", re.IGNORECASE)),
    ("aws_key", re.compile(r"""AKIA[0-9A-Z]{16}""")),
    ("private_key", re.compile(r"""-----BEGIN (?:RSA |EC )?PRIVATE KEY-----""")),
    ("github_token", re.compile(r"""gh[ps]_[A-Za-z0-9_]{36,}""")),
]

# Imports that indicate telemetry or analytics
_TELEMETRY_IMPORTS: frozenset[str] = frozenset({
    "sentry_sdk",
    "bugsnag",
    "rollbar",
    "newrelic",
    "datadog",
    "mixpanel",
    "segment",
    "amplitude",
    "posthog",
    "plausible",
})

# Imports that indicate network calls (in non-test code)
_NETWORK_IMPORTS: frozenset[str] = frozenset({
    "requests",
    "httpx",
    "aiohttp",
    "urllib.request",
    "http.client",
    "urllib3",
    "grpc",
})

# File extensions to scan
_SCANNABLE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".env",
})


def _scan_file_for_patterns(
    file_path: Path,
    patterns: list[tuple[str, re.Pattern[str]]],
) -> list[SecurityFinding]:
    """Scan a single file against a list of regex patterns."""
    findings: list[SecurityFinding] = []
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return findings

    for line_num, line in enumerate(content.splitlines(), start=1):
        for pattern_name, pattern in patterns:
            if pattern.search(line):
                findings.append(SecurityFinding(
                    severity="high",
                    category="secret",
                    file=str(file_path),
                    line=line_num,
                    message=f"Potential {pattern_name} detected",
                ))
    return findings


def scan_for_secrets(directory: Path) -> list[SecurityFinding]:
    """
    Scan a directory tree for embedded secrets and credentials.

    Checks config files, Python files, and environment files for patterns
    that look like API keys, passwords, tokens, or private keys.

    Args:
        directory: Root directory to scan recursively.

    Returns:
        List of SecurityFinding objects. Empty list means no issues found.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    findings: list[SecurityFinding] = []

    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix not in _SCANNABLE_EXTENSIONS:
            continue
        # Skip test files — they may have mock secrets
        if "/tests/" in str(file_path) or "test_" in file_path.name:
            continue

        file_findings = _scan_file_for_patterns(file_path, _SECRET_PATTERNS)
        findings.extend(file_findings)

    _logger.info(
        "Secret scan complete",
        extra={"directory": str(directory), "findings": len(findings)},
    )
    return findings


def check_no_telemetry(directory: Path) -> tuple[bool, list[SecurityFinding]]:
    """
    Verify no telemetry or analytics libraries are imported.

    Args:
        directory: Root directory to scan.

    Returns:
        (is_clean, findings) — True if no telemetry found.
    """
    findings: list[SecurityFinding] = []

    for file_path in sorted(directory.rglob("*.py")):
        if not file_path.is_file():
            continue
        if "/tests/" in str(file_path):
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for line_num, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            for lib in _TELEMETRY_IMPORTS:
                if lib in stripped:
                    findings.append(SecurityFinding(
                        severity="medium",
                        category="telemetry",
                        file=str(file_path),
                        line=line_num,
                        message=f"Telemetry import detected: {lib}",
                    ))

    _logger.info(
        "Telemetry scan complete",
        extra={"directory": str(directory), "findings": len(findings)},
    )
    return len(findings) == 0, findings


def check_no_network_calls(directory: Path) -> tuple[bool, list[SecurityFinding]]:
    """
    Verify no network libraries are imported in production code.

    Skips test files since tests may use network mocks.

    Args:
        directory: Root directory to scan.

    Returns:
        (is_clean, findings) — True if no network imports found.
    """
    findings: list[SecurityFinding] = []

    for file_path in sorted(directory.rglob("*.py")):
        if not file_path.is_file():
            continue
        # Skip test files
        if "/tests/" in str(file_path) or "test_" in file_path.name:
            continue
        # Skip scripts — they may legitimately use network
        if "/scripts/" in str(file_path):
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for line_num, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            for lib in _NETWORK_IMPORTS:
                if lib in stripped:
                    findings.append(SecurityFinding(
                        severity="medium",
                        category="network",
                        file=str(file_path),
                        line=line_num,
                        message=f"Network library import detected: {lib}",
                    ))

    _logger.info(
        "Network scan complete",
        extra={"directory": str(directory), "findings": len(findings)},
    )
    return len(findings) == 0, findings


def check_file_permissions(directory: Path) -> list[SecurityFinding]:
    """
    Check for insecure file permissions (world-writable files).

    Per spec: use least privileges, avoid system directories.

    Args:
        directory: Root directory to scan.

    Returns:
        List of findings for files with dangerous permissions.
    """
    findings: list[SecurityFinding] = []

    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file():
            continue
        # Skip .git internals
        if "/.git/" in str(file_path):
            continue

        try:
            file_stat = file_path.stat()
            mode = file_stat.st_mode

            # Check for world-writable
            if mode & stat.S_IWOTH:
                findings.append(SecurityFinding(
                    severity="medium",
                    category="permissions",
                    file=str(file_path),
                    line=0,
                    message=f"World-writable file: {oct(mode)}",
                ))

            # Check for setuid/setgid bits
            if mode & (stat.S_ISUID | stat.S_ISGID):
                findings.append(SecurityFinding(
                    severity="high",
                    category="permissions",
                    file=str(file_path),
                    line=0,
                    message=f"SetUID/SetGID bit set: {oct(mode)}",
                ))
        except OSError:
            continue

    _logger.info(
        "Permission scan complete",
        extra={"directory": str(directory), "findings": len(findings)},
    )
    return findings
