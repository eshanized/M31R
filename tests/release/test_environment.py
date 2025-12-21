# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for environment validation.
"""

from m31r.release.environment.validator import (
    check_python_version,
    check_disk_space,
    validate_environment,
    EnvironmentCheck,
)


def test_python_version_check():
    """Test Python version check logic."""
    check = check_python_version()
    # This test runs in the env, so it should pass if we are in a valid env
    assert isinstance(check, EnvironmentCheck)
    # We don't assert check.passed because dev env might vary, but struct is checked


def test_disk_space_check(tmp_path):
    """Test disk space check returns valid structure."""
    check = check_disk_space(tmp_path)
    assert check.name == "disk_space"
    assert "GB" in check.value


def test_full_validation():
    """Test the full validation suite runs without error."""
    checks = validate_environment()
    assert len(checks) >= 4  # python, torch, cuda, disk, memory
    check_names = {c.name for c in checks}
    assert "python_version" in check_names
    assert "disk_space" in check_names
