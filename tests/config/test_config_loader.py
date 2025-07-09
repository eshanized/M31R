# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for config loader — the entry point for all config loading in M31R.

We test four things:
  1. Valid YAML loads into a frozen, correct config object
  2. Missing required fields raise ConfigValidationError
  3. Unknown fields raise ConfigValidationError (extra="forbid")
  4. Broken YAML raises ConfigLoadError
  5. Loaded config is truly immutable
"""

import textwrap
from pathlib import Path

import pytest

from m31r.config.exceptions import ConfigLoadError, ConfigValidationError
from m31r.config.loader import load_config


class TestLoadValidConfig:
    def test_loads_minimal_valid_config(self, tmp_config_file: Path) -> None:
        config = load_config(tmp_config_file)
        assert config.global_config.project_name == "m31r-test"
        assert config.global_config.seed == 42
        assert config.global_config.config_version == "1.0.0"

    def test_default_directories_are_populated(self, tmp_config_file: Path) -> None:
        config = load_config(tmp_config_file)
        dirs = config.global_config.directories
        assert dirs.data == "data"
        assert dirs.checkpoints == "checkpoints"
        assert dirs.logs == "logs"
        assert dirs.experiments == "experiments"

    def test_optional_sections_default_to_none(self, tmp_config_file: Path) -> None:
        config = load_config(tmp_config_file)
        assert config.dataset is None
        assert config.tokenizer is None
        assert config.model is None
        assert config.train is None
        assert config.eval is None
        assert config.runtime is None

    def test_loads_full_config_with_all_sections(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            global:
              config_version: "1.0.0"
              project_name: "full-test"
              seed: 99
              log_level: "DEBUG"
            dataset:
              config_version: "1.0.0"
            tokenizer:
              config_version: "1.0.0"
            model:
              config_version: "1.0.0"
            train:
              config_version: "1.0.0"
            eval:
              config_version: "1.0.0"
            runtime:
              config_version: "1.0.0"
        """)
        config_file = tmp_path / "full.yaml"
        config_file.write_text(content, encoding="utf-8")

        config = load_config(config_file)
        assert config.global_config.seed == 99
        assert config.dataset is not None
        assert config.runtime is not None


class TestLoadInvalidConfig:
    def test_missing_required_field_raises_validation_error(
        self, invalid_config_file: Path
    ) -> None:
        with pytest.raises(ConfigValidationError):
            load_config(invalid_config_file)

    def test_unknown_field_raises_validation_error(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            global:
              config_version: "1.0.0"
              project_name: "test"
              seed: 42
              some_nonsense_field: true
        """)
        config_file = tmp_path / "unknown_field.yaml"
        config_file.write_text(content, encoding="utf-8")

        with pytest.raises(ConfigValidationError):
            load_config(config_file)

    def test_wrong_type_raises_validation_error(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            global:
              config_version: "1.0.0"
              project_name: "test"
              seed: "not_a_number"
        """)
        config_file = tmp_path / "wrong_type.yaml"
        config_file.write_text(content, encoding="utf-8")

        with pytest.raises(ConfigValidationError):
            load_config(config_file)

    def test_broken_yaml_raises_load_error(self, broken_yaml_file: Path) -> None:
        with pytest.raises(ConfigLoadError):
            load_config(broken_yaml_file)

    def test_nonexistent_file_raises_load_error(self, tmp_path: Path) -> None:
        fake_path = tmp_path / "does_not_exist.yaml"
        with pytest.raises(ConfigLoadError):
            load_config(fake_path)

    def test_directory_path_raises_load_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigLoadError):
            load_config(tmp_path)


class TestConfigImmutability:
    def test_cannot_mutate_frozen_config(self, tmp_config_file: Path) -> None:
        config = load_config(tmp_config_file)
        with pytest.raises(Exception):
            config.global_config.seed = 999  # type: ignore[misc]

    def test_cannot_mutate_nested_directories(self, tmp_config_file: Path) -> None:
        config = load_config(tmp_config_file)
        with pytest.raises(Exception):
            config.global_config.directories.data = "/hacked"  # type: ignore[misc]
