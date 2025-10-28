# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for the report writer.

Verifies that metrics.json is valid JSON, report.txt has the expected
sections, and the output directory structure is correct.
"""

import json
from pathlib import Path

from m31r.evaluation.benchmarks.models import EvalMetrics
from m31r.evaluation.reporting.writer import format_report_text, write_report


def _sample_metrics() -> EvalMetrics:
    return EvalMetrics(
        compile_success_rate=0.75,
        pass_at_k={1: 0.5, 5: 0.8, 10: 0.9},
        category_compile_rates={"completion": 1.0, "bugs": 0.5},
        category_pass_rates={"completion": 0.8, "bugs": 0.2},
        total_tasks=10,
        total_attempts=100,
        total_compiled=75,
        total_passed=50,
        avg_generation_time_seconds=0.5,
        avg_compile_time_seconds=0.1,
        avg_test_time_seconds=0.05,
        seed=42,
    )


class TestWriteReport:
    def test_creates_output_directory(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "results" / "eval"
        write_report(_sample_metrics(), output_dir)
        assert output_dir.is_dir()

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "eval"
        write_report(_sample_metrics(), output_dir)

        metrics_path = output_dir / "metrics.json"
        assert metrics_path.is_file()

        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert data["compile_success_rate"] == 0.75
        assert data["total_tasks"] == 10
        assert "1" in data["pass_at_k"] or 1 in data["pass_at_k"]

    def test_writes_report_txt(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "eval"
        write_report(_sample_metrics(), output_dir)

        report = (output_dir / "report.txt").read_text(encoding="utf-8")
        assert "EVALUATION REPORT" in report
        assert "Compile Success Rate" in report
        assert "Pass@1" in report

    def test_writes_config_snapshot(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "eval"
        write_report(_sample_metrics(), output_dir, config_snapshot={"seed": 42})

        config_path = output_dir / "config_snapshot.yaml"
        assert config_path.is_file()
        assert "seed" in config_path.read_text(encoding="utf-8")

    def test_no_config_snapshot_when_none(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "eval"
        write_report(_sample_metrics(), output_dir, config_snapshot=None)
        assert not (output_dir / "config_snapshot.yaml").exists()


class TestFormatReportText:
    def test_contains_all_sections(self) -> None:
        text = format_report_text(_sample_metrics())

        assert "PRIMARY METRICS" in text
        assert "SUMMARY" in text
        assert "TIMING" in text
        assert "CATEGORY COMPILE RATES" in text
        assert "CATEGORY PASS RATES" in text

    def test_shows_compile_rate(self) -> None:
        text = format_report_text(_sample_metrics())
        assert "75.00%" in text

    def test_shows_pass_at_k(self) -> None:
        text = format_report_text(_sample_metrics())
        assert "Pass@1" in text
        assert "Pass@5" in text
        assert "Pass@10" in text
