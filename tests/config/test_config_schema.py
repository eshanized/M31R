# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Schema-level validation tests.

These focus on the pydantic models themselves — boundary values, constraint
enforcement, and structural correctness.
"""

import pytest
from pydantic import ValidationError

from m31r.config.schema import DirectoryConfig, GlobalConfig, M31RConfig


class TestGlobalConfigSchema:
    def test_seed_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            GlobalConfig(config_version="1.0.0", seed=-1)

    def test_seed_zero_is_valid(self) -> None:
        config = GlobalConfig(config_version="1.0.0", seed=0)
        assert config.seed == 0

    def test_large_seed_is_valid(self) -> None:
        config = GlobalConfig(config_version="1.0.0", seed=2**31)
        assert config.seed == 2**31

    def test_default_log_level_is_info(self) -> None:
        config = GlobalConfig(config_version="1.0.0")
        assert config.log_level == "INFO"

    def test_default_project_name(self) -> None:
        config = GlobalConfig(config_version="1.0.0")
        assert config.project_name == "m31r"

    def test_config_version_is_required(self) -> None:
        with pytest.raises(ValidationError):
            GlobalConfig()  # type: ignore[call-arg]


class TestDirectoryConfigSchema:
    def test_defaults_are_populated(self) -> None:
        dirs = DirectoryConfig()
        assert dirs.data == "data"
        assert dirs.checkpoints == "checkpoints"
        assert dirs.logs == "logs"
        assert dirs.experiments == "experiments"
        assert dirs.configs == "configs"

    def test_custom_paths_are_accepted(self) -> None:
        dirs = DirectoryConfig(data="my_data", logs="my_logs")
        assert dirs.data == "my_data"
        assert dirs.logs == "my_logs"

    def test_unknown_field_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DirectoryConfig(data="data", secret_dir="hidden")  # type: ignore[call-arg]


class TestM31RConfigSchema:
    def test_requires_global_section(self) -> None:
        with pytest.raises(ValidationError):
            M31RConfig()  # type: ignore[call-arg]

    def test_rejects_top_level_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            M31RConfig.model_validate({
                "global": {"config_version": "1.0.0"},
                "unknown_section": {"something": True},
            })
