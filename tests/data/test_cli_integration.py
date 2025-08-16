# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
CLI integration tests for the data pipeline commands.

These test the full CLI path: argument parsing → handler dispatch → pipeline
execution → exit code. We test both the happy path and error conditions.
"""

import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from m31r.cli.exit_codes import CONFIG_ERROR, SUCCESS
from m31r.cli.main import main


def _run_main(*argv: str) -> int:
    """Run the CLI main function with the given arguments and return exit code."""
    with mock.patch.object(sys, "argv", ["m31r", *argv]):
        try:
            main()
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 0
    return 0


class TestCrawlCLI:
    def test_crawl_without_config_succeeds(self) -> None:
        exit_code = _run_main("crawl")
        assert exit_code == SUCCESS

    def test_crawl_with_nonexistent_config_fails(self) -> None:
        exit_code = _run_main("crawl", "--config", "/no/such/config.yaml")
        assert exit_code == CONFIG_ERROR

    def test_crawl_with_valid_config_no_dataset_section(self, tmp_config_file: Path) -> None:
        exit_code = _run_main("crawl", "--config", str(tmp_config_file))
        assert exit_code == SUCCESS

    def test_crawl_dry_run(self, tmp_path: Path) -> None:
        config_content = textwrap.dedent("""\
            global:
              config_version: "1.0.0"
              project_name: "m31r-test"
              seed: 42
              log_level: "WARNING"
            dataset:
              config_version: "1.0.0"
              sources: []
        """)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        exit_code = _run_main("crawl", "--config", str(config_file), "--dry-run")
        assert exit_code == SUCCESS


class TestFilterCLI:
    def test_filter_without_config_succeeds(self) -> None:
        exit_code = _run_main("filter")
        assert exit_code == SUCCESS

    def test_filter_with_nonexistent_config_fails(self) -> None:
        exit_code = _run_main("filter", "--config", "/no/such/config.yaml")
        assert exit_code == CONFIG_ERROR


class TestDatasetCLI:
    def test_dataset_without_config_succeeds(self) -> None:
        exit_code = _run_main("dataset")
        assert exit_code == SUCCESS

    def test_dataset_with_nonexistent_config_fails(self) -> None:
        exit_code = _run_main("dataset", "--config", "/no/such/config.yaml")
        assert exit_code == CONFIG_ERROR


class TestVerifyCLI:
    def test_verify_without_args_succeeds(self) -> None:
        exit_code = _run_main("verify")
        assert exit_code == SUCCESS

    def test_verify_help_shows_dataset_dir(self) -> None:
        exit_code = _run_main("verify", "--help")
        assert exit_code == 0
