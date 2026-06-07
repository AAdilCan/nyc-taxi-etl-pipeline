"""Tests for the pipeline orchestrator.

The stage functions themselves are exercised in their own modules; here I stub
them out and focus on the wiring: stage selection/ordering, error propagation,
row accounting, and batch resilience.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taxi_etl import pipeline
from taxi_etl.pipeline import (
    IngestSummary,
    MonthResult,
    PipelineRun,
    run_month,
    run_months,
)
from taxi_etl.stages.ingest import IngestResult
from taxi_etl.stages.transform import TransformReport, TransformResult
from taxi_etl.stages.validate import ValidationReport, ValidationResult


# ---------------------------------------------------------------------------
# Stub stage results
# ---------------------------------------------------------------------------

def _ingest_result(month: str, rows: int = 100) -> IngestResult:
    return IngestResult(
        month=month,
        taxi_type="yellow",
        dest_path=Path(f"/tmp/{month}.parquet"),
        row_count=rows,
        columns=["fare_amount"],
        size_bytes=2048,
        skipped=False,
    )


def _validate_result(month: str, total: int = 100, valid: int = 90) -> ValidationResult:
    report = ValidationReport(
        month=month,
        taxi_type="yellow",
        total_rows=total,
        valid_rows=valid,
        quarantined_rows=total - valid,
        valid_pct=round(100.0 * valid / total, 2),
        quarantined_pct=round(100.0 * (total - valid) / total, 2),
        failure_counts={"fare_amount": total - valid},
        valid_path="/tmp/valid.parquet",
        quarantine_path="/tmp/quarantine.parquet",
    )
    return ValidationResult(
        month=month,
        taxi_type="yellow",
        total_rows=total,
        valid_rows=valid,
        quarantined_rows=total - valid,
        valid_path=Path("/tmp/valid.parquet"),
        quarantine_path=Path("/tmp/quarantine.parquet"),
        report=report,
    )


def _transform_result(month: str, in_rows: int = 90, out_rows: int = 80) -> TransformResult:
    import pandas as pd

    report = TransformReport(
        month=month,
        taxi_type="yellow",
        input_rows=in_rows,
        output_rows=out_rows,
        dropped_rows=in_rows - out_rows,
        output_pct=round(100.0 * out_rows / in_rows, 2),
        dropped_by_reason={"speed": in_rows - out_rows},
        feature_columns=["trip_duration_min", "avg_speed_mph"],
        n_partitions=2,
        output_path="/tmp/features",
    )
    return TransformResult(
        month=month,
        taxi_type="yellow",
        input_rows=in_rows,
        output_rows=out_rows,
        dropped_rows=in_rows - out_rows,
        output_path=Path("/tmp/features"),
        report=report,
        frame=pd.DataFrame(),
    )


@pytest.fixture
def stub_stages(monkeypatch):
    """Replace the real stage ``run`` functions with deterministic stubs."""
    monkeypatch.setattr(
        pipeline.ingest, "run", lambda month, force=False: _ingest_result(month)
    )
    monkeypatch.setattr(pipeline.validate, "run", lambda month: _validate_result(month))
    monkeypatch.setattr(pipeline.transform, "run", lambda month: _transform_result(month))


# ---------------------------------------------------------------------------
# Stage selection / ordering
# ---------------------------------------------------------------------------

class TestStageSelection:
    def test_unknown_stage_raises(self):
        with pytest.raises(ValueError, match="unknown stage"):
            run_month("2024-01", stages=("ingest", "bogus"))

    def test_stages_normalised_to_canonical_order(self, stub_stages):
        # Passed reversed; should still run ingest → validate → transform.
        result = run_month("2024-01", stages=("transform", "ingest", "validate"))
        assert result.stages_run == ["ingest", "validate", "transform"]

    def test_subset_runs_only_requested_stages(self, stub_stages):
        result = run_month("2024-01", stages=("ingest",))
        assert result.stages_run == ["ingest"]
        assert result.validation is None
        assert result.transform is None
        assert result.feature_rows is None


# ---------------------------------------------------------------------------
# Row accounting
# ---------------------------------------------------------------------------

class TestRowAccounting:
    def test_full_run_collects_all_reports(self, stub_stages):
        result = run_month("2024-01")
        assert result.taxi_type == "yellow"
        assert result.raw_rows == 100
        assert result.valid_rows == 90
        assert result.feature_rows == 80
        assert set(result.timings_s) == {"ingest", "validate", "transform"}

    def test_end_to_end_retention(self, stub_stages):
        result = run_month("2024-01")
        # 80 features from 100 raw → 80%.
        assert result.end_to_end_retention_pct == 80.0

    def test_retention_none_without_feature_rows(self):
        m = MonthResult(month="2024-01", taxi_type="yellow", stages_run=[], timings_s={})
        assert m.end_to_end_retention_pct is None

    def test_total_seconds_sums_timings(self):
        m = MonthResult(
            month="2024-01",
            taxi_type="yellow",
            stages_run=["ingest", "validate"],
            timings_s={"ingest": 1.5, "validate": 0.5},
        )
        assert m.total_seconds == 2.0


# ---------------------------------------------------------------------------
# IngestSummary
# ---------------------------------------------------------------------------

class TestIngestSummary:
    def test_from_result_drops_path(self):
        summary = IngestSummary.from_result(_ingest_result("2024-01", rows=42))
        assert summary == IngestSummary(
            month="2024-01",
            taxi_type="yellow",
            row_count=42,
            size_bytes=2048,
            skipped=False,
        )
        assert not hasattr(summary, "dest_path")


# ---------------------------------------------------------------------------
# Batch behaviour
# ---------------------------------------------------------------------------

class TestRunMonths:
    def test_processes_each_month(self, stub_stages):
        run = run_months(["2024-01", "2024-02"])
        assert [m.month for m in run.months] == ["2024-01", "2024-02"]
        assert run.total_seconds == sum(m.total_seconds for m in run.months)

    def test_continue_on_error_skips_failing_month(self, monkeypatch, stub_stages):
        def flaky_ingest(month, force=False):
            if month == "2024-02":
                raise RuntimeError("download failed")
            return _ingest_result(month)

        monkeypatch.setattr(pipeline.ingest, "run", flaky_ingest)
        run = run_months(["2024-01", "2024-02", "2024-03"], continue_on_error=True)
        assert [m.month for m in run.months] == ["2024-01", "2024-03"]

    def test_fail_fast_propagates(self, monkeypatch, stub_stages):
        def flaky_ingest(month, force=False):
            raise RuntimeError("boom")

        monkeypatch.setattr(pipeline.ingest, "run", flaky_ingest)
        with pytest.raises(RuntimeError, match="boom"):
            run_months(["2024-01"], continue_on_error=False)

    def test_empty_run(self):
        run = PipelineRun()
        assert run.months == []
        assert run.total_seconds == 0.0
