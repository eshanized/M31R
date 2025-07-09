# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Determinism tests for the runtime bootstrap.

Per 15_TESTING_STRATEGY.md section 16:
  - run twice with the same seed
  - compare outputs
  - mismatch is failure

We also test environment validation and the bootstrap function itself.
"""

import random

import pytest

from m31r.config.schema import GlobalConfig
from m31r.runtime.bootstrap import bootstrap, set_deterministic_seed
from m31r.runtime.environment import check_minimum_python, get_python_version, get_system_info


class TestDeterministicSeed:
    def test_same_seed_produces_same_random_sequence(self) -> None:
        """
        This is the core determinism guarantee: if you set the same seed,
        you get the same sequence of random numbers. We generate 10 values
        twice with the same seed and compare them.
        """
        set_deterministic_seed(42)
        sequence_a = [random.random() for _ in range(10)]

        set_deterministic_seed(42)
        sequence_b = [random.random() for _ in range(10)]

        assert sequence_a == sequence_b

    def test_different_seeds_produce_different_sequences(self) -> None:
        set_deterministic_seed(42)
        sequence_a = [random.random() for _ in range(10)]

        set_deterministic_seed(99)
        sequence_b = [random.random() for _ in range(10)]

        assert sequence_a != sequence_b

    def test_seed_zero_is_valid(self) -> None:
        set_deterministic_seed(0)
        value = random.random()
        set_deterministic_seed(0)
        assert random.random() == value


class TestEnvironmentValidation:
    def test_python_version_returns_tuple(self) -> None:
        version = get_python_version()
        assert isinstance(version, tuple)
        assert len(version) == 3
        assert all(isinstance(v, int) for v in version)

    def test_minimum_python_check_passes(self) -> None:
        """We're running this test, so Python must be >= 3.11."""
        check_minimum_python()

    def test_system_info_has_all_fields(self) -> None:
        info = get_system_info()
        assert info.python_version
        assert info.platform
        assert info.architecture
        assert info.hostname is not None


class TestBootstrap:
    def test_bootstrap_completes_without_error(self) -> None:
        config = GlobalConfig(config_version="1.0.0", seed=42, log_level="WARNING")
        bootstrap(config)

    def test_bootstrap_sets_seed(self) -> None:
        config = GlobalConfig(config_version="1.0.0", seed=12345, log_level="WARNING")
        bootstrap(config)

        value_a = random.random()

        set_deterministic_seed(12345)
        value_b = random.random()

        assert value_a == value_b
