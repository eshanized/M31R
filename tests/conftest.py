# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Shared pytest fixtures for M31R tests.

Fixtures here are available to every test file automatically.
We keep them minimal — just the stuff that multiple test modules need.
"""

import textwrap
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_config_file(tmp_path: Path) -> Path:
    """
    Create a minimal valid config YAML file in a temp directory.

    This is the smallest config that passes schema validation.
    Tests that need specific config values should write their own files.
    """
    config_content = textwrap.dedent("""\
        global:
          config_version: "1.0.0"
          project_name: "m31r-test"
          seed: 42
          log_level: "DEBUG"
    """)
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content, encoding="utf-8")
    return config_file


@pytest.fixture()
def invalid_config_file(tmp_path: Path) -> Path:
    """A config file that's valid YAML but fails schema validation (missing required field)."""
    config_content = textwrap.dedent("""\
        global:
          project_name: "m31r-test"
          seed: 42
    """)
    config_file = tmp_path / "invalid_config.yaml"
    config_file.write_text(config_content, encoding="utf-8")
    return config_file


@pytest.fixture()
def broken_yaml_file(tmp_path: Path) -> Path:
    """A file that isn't valid YAML at all."""
    config_file = tmp_path / "broken.yaml"
    config_file.write_text("{{not: yaml: at: all:::", encoding="utf-8")
    return config_file
