"""Tests for the reporting layer.

These build :class:`MonthResult` / :class:`PipelineRun` objects directly so the
summary, narrative, and file-writing logic can be checked without running any
stage. File writes are redirected to a tmp dir via the ``settings`` singleton.
"""

from __future__ import annotations

import json

import pytest

from taxi_etl import reporting
from taxi_etl.pipeline import MonthResult, PipelineRun
from taxi_etl.reporting import (
    _top_reasons,
    build_data_quality_md,
    build_summary,
    format_console_summary,
    write_reports,
)
from taxi_etl.stages.transform import TransformReport
from taxi_etl.stages.validate import ValidationReport


def _validation(month: str, total: int, valid: int, failures: dict) -> ValidationReport:
    q = total - valid
    return ValidationReport(
        month=month,
        taxi_type="yellow",
        total_rows=total,
        valid_rows=valid,
        quarantined_rows=q,
        valid_pct=round(100.0 * valid / total, 2),
        quarantined_pct=round(100.0 * q / total, 2),
        failure_counts=failures,
        valid_path="/tmp/valid.parquet",
        quarantine_path="/tmp/quarantine.parquet",
    )


def _transform(month: str, in_rows: int, out_rows: int, reasons: dict) -> TransformReport:
    return TransformReport(
        month=month,
        taxi_type="yellow",
        input_rows=in_rows,
        output_rows=out_rows,
        dropped_rows=in_rows - out_rows,
        output_pct=round(100.0 * out_rows / in_rows, 2),
        dropped_by_reason=reasons,
        feature_columns=["trip_duration_min", "avg_speed_mph"],
        n_partitions=3,
        output_path="/tmp/features",
    )


def _month(month: str) -> MonthResult:
    return MonthResult(
        month=month,
        taxi_type="yellow",
        stages_run=["ingest", "validate", "transform"],
        timings_s={"ingest": 1.0, "validate": 0.5, "transform": 0.5},
        validation=_validation(month, 100, 90, {"fare_amount": 7, "trip_distance": 3}),
        transform=_transform(month, 90, 80, {"speed": 6, "duration": 4}),
        raw_rows=100,
        valid_rows=90,
        feature_rows=80,
    )


@pytest.fixture
def run() -> PipelineRun:
    return PipelineRun(months=[_month("2024-01"), _month("2024-02")])


# ---------------------------------------------------------------------------
# _top_reasons
# ---------------------------------------------------------------------------

class TestTopReasons:
    def test_sorted_by_count_desc(self):
        assert _top_reasons({"a": 1, "b": 5, "c": 3}) == [("b", 5), ("c", 3), ("a", 1)]

    def test_ties_broken_by_name(self):
        assert _top_reasons({"z": 2, "a": 2}) == [("a", 2), ("z", 2)]

    def test_zero_counts_dropped(self):
        assert _top_reasons({"a": 0, "b": 4}) == [("b", 4)]

    def test_limit_applied(self):
        counts = {f"r{i}": i for i in range(1, 11)}
        assert len(_top_reasons(counts, limit=3)) == 3


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------

class TestBuildSummary:
    def test_totals_and_retention(self, run):
        summary = build_summary(run)
        assert summary["months_processed"] == 2
        assert summary["totals"]["raw_rows"] == 200
        assert summary["totals"]["feature_rows"] == 160
        assert summary["totals"]["overall_retention_pct"] == 80.0

    def test_is_json_serializable(self, run):
        # asdict() over nested dataclasses must round-trip through json.
        text = json.dumps(build_summary(run))
        assert "2024-01" in text

    def test_per_month_payload_complete(self, run):
        summary = build_summary(run)
        m0 = summary["months"][0]
        assert m0["validation"]["valid_rows"] == 90
        assert m0["transform"]["output_rows"] == 80
        assert m0["end_to_end_retention_pct"] == 80.0

    def test_empty_run_retention_none(self):
        summary = build_summary(PipelineRun())
        assert summary["totals"]["overall_retention_pct"] is None
        assert summary["months_processed"] == 0


# ---------------------------------------------------------------------------
# Console + markdown rendering
# ---------------------------------------------------------------------------

class TestConsoleSummary:
    def test_empty(self):
        assert format_console_summary(PipelineRun()) == "No months were processed."

    def test_lists_months(self, run):
        out = format_console_summary(run)
        assert "2024-01" in out and "2024-02" in out
        assert "2 month(s)" in out


class TestDataQualityMd:
    def test_contains_sections_and_reasons(self, run):
        md = build_data_quality_md(run)
        assert "# Data Quality Report" in md
        assert "## yellow — 2024-01" in md
        assert "Top quarantine reasons" in md
        assert "`fare_amount`" in md
        assert "Top outlier reasons" in md
        assert "End-to-end retention" in md

    def test_handles_validate_only_month(self):
        m = MonthResult(
            month="2024-03",
            taxi_type="yellow",
            stages_run=["ingest", "validate"],
            timings_s={"ingest": 1.0, "validate": 0.5},
            validation=_validation("2024-03", 50, 50, {}),
            raw_rows=50,
            valid_rows=50,
        )
        md = build_data_quality_md(PipelineRun(months=[m]))
        assert "## yellow — 2024-03" in md
        # No transform → no outlier section, no retention line.
        assert "Top outlier reasons" not in md


# ---------------------------------------------------------------------------
# File writing
# ---------------------------------------------------------------------------

class TestWriteReports:
    def test_writes_both_artifacts(self, run, tmp_path, monkeypatch):
        monkeypatch.setattr(reporting.settings, "reports_dir", tmp_path)
        # ensure_dirs touches several dirs; point them all at tmp_path too.
        monkeypatch.setattr(reporting.settings, "data_dir", tmp_path)

        summary_path, dq_path = write_reports(run)

        assert summary_path.exists() and summary_path.suffix == ".json"
        assert dq_path.exists() and dq_path.suffix == ".md"
        payload = json.loads(summary_path.read_text())
        assert payload["months_processed"] == 2
