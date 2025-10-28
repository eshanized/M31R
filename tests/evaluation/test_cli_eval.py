# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
CLI smoke tests for the eval and benchmark commands.

These verify that the CLI argument parser correctly recognizes the eval
and benchmark commands and their arguments. We test the parser directly
instead of launching subprocesses to avoid environment-specific issues
with output capture.
"""

import argparse
import io
import sys

import pytest

from m31r.cli.commands import handle_benchmark, handle_eval


class TestEvalCLI:
    def test_eval_handler_exists(self) -> None:
        """The eval command should have a callable handler."""
        assert callable(handle_eval)

    def test_eval_command_registered(self) -> None:
        """The 'eval' subcommand should be in the parser's choices."""
        from m31r.cli.main import _build_global_parser, _register_subcommands

        parent = _build_global_parser()
        root = argparse.ArgumentParser(prog="m31r", parents=[parent])
        subs = root.add_subparsers(dest="command")
        _register_subcommands(subs, parent)

        assert "eval" in subs.choices

    def test_eval_has_checkpoint_arg(self) -> None:
        """The eval parser should accept --checkpoint."""
        from m31r.cli.main import _build_global_parser, _register_subcommands

        parent = _build_global_parser()
        root = argparse.ArgumentParser(prog="m31r", parents=[parent])
        subs = root.add_subparsers(dest="command")
        _register_subcommands(subs, parent)

        args = root.parse_args(["eval", "--checkpoint", "/some/path"])
        assert args.checkpoint == "/some/path"

    def test_eval_has_benchmark_dir_arg(self) -> None:
        """The eval parser should accept --benchmark-dir."""
        from m31r.cli.main import _build_global_parser, _register_subcommands

        parent = _build_global_parser()
        root = argparse.ArgumentParser(prog="m31r", parents=[parent])
        subs = root.add_subparsers(dest="command")
        _register_subcommands(subs, parent)

        args = root.parse_args(["eval", "--benchmark-dir", "/bench/dir"])
        assert args.benchmark_dir == "/bench/dir"

    def test_eval_default_args(self) -> None:
        from m31r.cli.main import _build_global_parser, _register_subcommands

        parent = _build_global_parser()
        root = argparse.ArgumentParser(prog="m31r", parents=[parent])
        subs = root.add_subparsers(dest="command")
        _register_subcommands(subs, parent)

        args = root.parse_args(["eval"])
        assert args.checkpoint is None
        assert args.benchmark_dir is None


class TestBenchmarkCLI:
    def test_benchmark_handler_exists(self) -> None:
        assert callable(handle_benchmark)

    def test_benchmark_command_registered(self) -> None:
        from m31r.cli.main import _build_global_parser, _register_subcommands

        parent = _build_global_parser()
        root = argparse.ArgumentParser(prog="m31r", parents=[parent])
        subs = root.add_subparsers(dest="command")
        _register_subcommands(subs, parent)

        assert "benchmark" in subs.choices

    def test_benchmark_has_benchmark_dir_arg(self) -> None:
        from m31r.cli.main import _build_global_parser, _register_subcommands

        parent = _build_global_parser()
        root = argparse.ArgumentParser(prog="m31r", parents=[parent])
        subs = root.add_subparsers(dest="command")
        _register_subcommands(subs, parent)

        args = root.parse_args(["benchmark", "--benchmark-dir", "/bench"])
        assert args.benchmark_dir == "/bench"

    def test_benchmark_handler_wired_up(self) -> None:
        from m31r.cli.main import _build_global_parser, _register_subcommands

        parent = _build_global_parser()
        root = argparse.ArgumentParser(prog="m31r", parents=[parent])
        subs = root.add_subparsers(dest="command")
        _register_subcommands(subs, parent)

        args = root.parse_args(["benchmark"])
        assert args.func == handle_benchmark
