# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
End-to-end determinism tests for the data pipeline.

The core guarantee: run the filter + dataset stages twice with identical input,
get identical output. If this test fails, something in the pipeline is
non-deterministic and we have a real bug.
"""

import json
import textwrap
from pathlib import Path

import pytest

from m31r.config.schema import DatasetConfig, FilterConfig, ShardConfig, SourceConfig
from m31r.data.dataset.builder import build_dataset
from m31r.data.filter.pipeline import run_filter_pipeline


def _create_test_repo(base_dir: Path, source_name: str, commit: str) -> None:
    """
    Set up a fake crawled repo directory with some Rust files and a license.

    This mimics what the crawler would produce, but without actually cloning
    anything. We create the directory structure, drop in some .rs files,
    and add a LICENSE file so the pipeline can detect it.
    """
    repo_dir = base_dir / "data" / "raw" / source_name / commit
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    (repo_dir / "LICENSE").write_text(
        "MIT License\n\nPermission is hereby granted, free of charge",
        encoding="utf-8",
    )
    (repo_dir / ".m31r_crawl_marker").write_text(commit, encoding="utf-8")

    (src_dir / "lib.rs").write_text(
        textwrap.dedent("""\
            pub fn add(a: i32, b: i32) -> i32 {
                a + b
            }
        """),
        encoding="utf-8",
    )
    (src_dir / "main.rs").write_text(
        textwrap.dedent("""\
            use crate::lib::add;

            fn main() {
                let result = add(1, 2);
            }
        """),
        encoding="utf-8",
    )

    # A file that should get filtered out (non-Rust)
    (src_dir / "notes.txt").write_text("just some notes", encoding="utf-8")


def _make_test_config() -> DatasetConfig:
    return DatasetConfig(
        config_version="1.0.0",
        sources=[
            SourceConfig(
                name="test_repo",
                url="https://example.com/test.git",
                commit="abc123",
                license="MIT",
            ),
        ],
        filter=FilterConfig(
            max_file_size_bytes=1_000_000,
            max_lines=5000,
            allowed_extensions=[".rs"],
            excluded_directories=["target", ".git"],
            allowed_licenses=["MIT", "Apache-2.0"],
            enable_deduplication=True,
        ),
        shard=ShardConfig(shard_size_bytes=10_000),
    )


class TestFilterDeterminism:
    def test_two_runs_produce_identical_stats(self, tmp_path: Path) -> None:
        """Run the filter pipeline twice on the same data — stats must match."""
        config = _make_test_config()

        base_a = tmp_path / "run_a"
        base_b = tmp_path / "run_b"

        for base in [base_a, base_b]:
            _create_test_repo(base, "test_repo", "abc123")

        config_a = DatasetConfig(
            **{**config.model_dump(), "filtered_directory": "data/filtered"},
        )
        config_b = DatasetConfig(
            **{**config.model_dump(), "filtered_directory": "data/filtered"},
        )

        stats_a = run_filter_pipeline(config_a, base_a)
        stats_b = run_filter_pipeline(config_b, base_b)

        assert stats_a.kept == stats_b.kept
        assert stats_a.total_files_scanned == stats_b.total_files_scanned
        assert stats_a.rejected_extension == stats_b.rejected_extension


class TestFullPipelineDeterminism:
    def test_two_runs_produce_identical_version_hash(self, tmp_path: Path) -> None:
        """
        The ultimate determinism test: filter + build, twice, same hash.

        This is the property the spec demands — if the inputs and config
        are the same, the output dataset must be byte-for-byte identical.
        """
        config = _make_test_config()

        base_a = tmp_path / "run_a"
        base_b = tmp_path / "run_b"

        for base in [base_a, base_b]:
            _create_test_repo(base, "test_repo", "abc123")

        run_filter_pipeline(config, base_a)
        run_filter_pipeline(config, base_b)

        result_a = build_dataset(config, base_a)
        result_b = build_dataset(config, base_b)

        assert result_a.version_hash == result_b.version_hash
        assert result_a.total_files == result_b.total_files
        assert result_a.total_shards == result_b.total_shards
