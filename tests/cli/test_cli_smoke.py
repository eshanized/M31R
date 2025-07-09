# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
CLI smoke tests.

Per 15_TESTING_STRATEGY.md section 22, CLI tests must verify:
  - commands execute
  - exit codes are correct
  - help text exists

We use subprocess to test the actual CLI entrypoint the way a user would.
This catches issues that unit tests miss, like broken imports or entrypoint
registration.
"""

import subprocess
import sys

import pytest


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run `m31r` with the given arguments and capture output."""
    return subprocess.run(
        [sys.executable, "-m", "m31r.cli.main", *args],
        capture_output=True,
        text=True,
        timeout=10,
    )


class TestHelpTexts:
    """Every subcommand must have working --help output."""

    @pytest.mark.parametrize(
        "subcommand",
        [
            "crawl", "filter", "dataset", "tokenizer", "train",
            "eval", "serve", "generate", "export", "verify", "info",
        ],
    )
    def test_subcommand_help_exits_zero(self, subcommand: str) -> None:
        result = _run_cli(subcommand, "--help")
        assert result.returncode == 0
        assert subcommand in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_root_help_exits_with_user_error(self) -> None:
        """Running m31r with no args should show help and exit with USER_ERROR (1)."""
        result = _run_cli()
        assert result.returncode == 1


class TestSubcommandExecution:
    """Subcommands should execute cleanly without a config file."""

    def test_info_runs_without_config(self) -> None:
        result = _run_cli("info")
        assert result.returncode == 0

    def test_crawl_runs_without_config(self) -> None:
        result = _run_cli("crawl")
        assert result.returncode == 0

    def test_verify_runs_without_config(self) -> None:
        result = _run_cli("verify")
        assert result.returncode == 0


class TestConfigLoading:
    """Subcommands should handle config loading failures gracefully."""

    def test_nonexistent_config_returns_config_error(self) -> None:
        result = _run_cli("crawl", "--config", "/nonexistent/path.yaml")
        assert result.returncode == 2  # CONFIG_ERROR

    def test_valid_config_is_accepted(self, tmp_config_file) -> None:  # type: ignore[no-untyped-def]
        result = _run_cli("crawl", "--config", str(tmp_config_file))
        assert result.returncode == 0


class TestGlobalOptions:
    def test_log_level_option_is_accepted(self) -> None:
        result = _run_cli("info", "--log-level", "DEBUG")
        assert result.returncode == 0

    def test_seed_option_is_accepted(self) -> None:
        result = _run_cli("info", "--seed", "123")
        assert result.returncode == 0

    def test_dry_run_option_is_accepted(self) -> None:
        result = _run_cli("info", "--dry-run")
        assert result.returncode == 0
