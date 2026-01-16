# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for security scanner.
"""

import shutil
from pathlib import Path

from m31r.release.security.scanner import (
    scan_for_secrets,
    check_no_telemetry,
    check_no_network_calls,
)


def test_secret_detection(tmp_path: Path):
    """Test detection of API keys and secrets."""
    # Create a file with a fake secret
    bad_file = tmp_path / "config.py"
    bad_file.write_text('API_KEY = "sk-1234567890abcdef12345678"')

    findings = scan_for_secrets(tmp_path)
    assert len(findings) == 1
    assert findings[0].file == str(bad_file)
    assert "api_key" in findings[0].message.lower()


def test_telemetry_detection(tmp_path: Path):
    """Test detection of telemetry imports."""
    bad_file = tmp_path / "main.py"
    bad_file.write_text("import sentry_sdk\nsentry_sdk.init()")

    clean, findings = check_no_telemetry(tmp_path)
    assert not clean
    assert len(findings) == 1
    assert "sentry_sdk" in findings[0].message


def test_network_import_detection(tmp_path: Path):
    """Test detection of network libraries in non-test code."""
    bad_file = tmp_path / "utils.py"
    bad_file.write_text("import requests\nresp = requests.get('...')")

    clean, findings = check_no_network_calls(tmp_path)
    assert not clean
    assert len(findings) == 1
    assert "requests" in findings[0].message


def test_clean_directory_passes(tmp_path: Path):
    """Test that a clean directory passes all scans."""
    (tmp_path / "clean.py").write_text("x = 1\nprint(x)")

    assert not scan_for_secrets(tmp_path)
    assert check_no_telemetry(tmp_path)[0]
    assert check_no_network_calls(tmp_path)[0]
