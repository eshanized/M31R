# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
CLI smoke tests.

Per 15_TESTING_STRATEGY.md section 22, CLI tests must verify:
  - commands execute
  - exit codes are correct
  - help text exists

We test the CLI in-process by patching sys.argv and catching SystemExit.
This is more reliable than subprocess testing and catches import/wiring
issues just as well.
"""

import sys
from unittest import mock

import pytest

from m31r.cli.commands import (
    handle_crawl,
    handle_eval,
    handle_info,
    handle_verify,
)
from m31r.cli.exit_codes import CONFIG_ERROR, SUCCESS, USER_ERROR
from m31r.cli.main import main


def _run_main(*argv: str) -> int:
    """
    Run the CLI main function with the given arguments and return exit code.

    Patches sys.argv and catches the SystemExit that main() raises via sys.exit().
    Returns the exit code as an int. If argparse's --help triggers a SystemExit(0),
    that also gets caught cleanly.
    """
    with mock.patch.object(sys, "argv", ["m31r", *argv]):
        try:
            main()
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 0
    return 0


class TestHelpTexts:
    """Every subcommand must have working --help output."""

    @pytest.mark.parametrize(
        "subcommand",
        [
            "crawl",
            "filter",
            "dataset",
            "tokenizer",
            "train",
            "eval",
            "serve",
            "generate",
            "export",
            "verify",
            "info",
        ],
    )
    def test_subcommand_help_exits_zero(self, subcommand: str) -> None:
        exit_code = _run_main(subcommand, "--help")
        assert exit_code == 0

    def test_root_help_exits_zero(self) -> None:
        exit_code = _run_main("--help")
        assert exit_code == 0

    def test_no_args_exits_with_user_error(self) -> None:
        exit_code = _run_main()
        assert exit_code == USER_ERROR


class TestSubcommandExecution:
    """Subcommands should execute cleanly without a config file."""

    def test_info_runs_without_config(self) -> None:
        exit_code = _run_main("info")
        assert exit_code == SUCCESS

    def test_crawl_runs_without_config(self) -> None:
        exit_code = _run_main("crawl")
        assert exit_code == SUCCESS

    def test_verify_runs_without_config(self) -> None:
        exit_code = _run_main("verify")
        assert exit_code == USER_ERROR


class TestConfigLoading:
    """Subcommands should handle config loading failures gracefully."""

    def test_nonexistent_config_returns_config_error(self) -> None:
        exit_code = _run_main("crawl", "--config", "/nonexistent/path.yaml")
        assert exit_code == CONFIG_ERROR

    def test_valid_config_is_accepted(self, tmp_config_file) -> None:  # type: ignore[no-untyped-def]
        exit_code = _run_main("crawl", "--config", str(tmp_config_file))
        assert exit_code == SUCCESS


class TestGlobalOptions:
    def test_log_level_option_is_accepted(self) -> None:
        exit_code = _run_main("info", "--log-level", "DEBUG")
        assert exit_code == SUCCESS

    def test_seed_option_is_accepted(self) -> None:
        exit_code = _run_main("info", "--seed", "123")
        assert exit_code == SUCCESS

    def test_dry_run_option_is_accepted(self) -> None:
        exit_code = _run_main("info", "--dry-run")
        assert exit_code == SUCCESS
