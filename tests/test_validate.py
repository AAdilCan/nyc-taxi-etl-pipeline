"""Tests for the validation stage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from taxi_etl.stages.validate import (
    RAW_SCHEMA,
    ValidationResult,
    _merge_failures,
    _pandera_failures,
    _temporal_failures,
    run,
)


# ---------------------------------------------------------------------------
# Minimal valid DataFrame factory
# ---------------------------------------------------------------------------

def _valid_row(**overrides) -> dict:
    base = {
        "VendorID": np.int32(2),
        "tpep_pickup_datetime": pd.Timestamp("2024-01-15 08:30:00"),
        "tpep_dropoff_datetime": pd.Timestamp("2024-01-15 08:55:00"),
        "passenger_count": 1.0,
        "trip_distance": 3.2,
        "RatecodeID": 1.0,
        "store_and_fwd_flag": "N",
        "PULocationID": np.int32(161),
        "DOLocationID": np.int32(236),
        "payment_type": np.int64(1),
        "fare_amount": 14.5,
        "extra": 0.5,
        "mta_tax": 0.5,
        "tip_amount": 3.0,
        "tolls_amount": 0.0,
        "improvement_surcharge": 0.3,
        "total_amount": 19.3,
        "congestion_surcharge": 2.5,
        "Airport_fee": 0.0,
    }
    base.update(overrides)
    return base


def _make_df(*row_dicts: dict) -> pd.DataFrame:
    return pd.DataFrame(list(row_dicts))


# ---------------------------------------------------------------------------
# _pandera_failures
# ---------------------------------------------------------------------------

class TestPanderaFailures:
    def test_clean_row_no_failures(self):
        df = _make_df(_valid_row())
        assert _pandera_failures(df) == {}

    def test_invalid_vendor_id(self):
        df = _make_df(_valid_row(VendorID=np.int32(99)))
        failures = _pandera_failures(df)
        assert 0 in failures
        assert any("VendorID" in r for r in failures[0])

    def test_negative_trip_distance(self):
        df = _make_df(_valid_row(trip_distance=-1.0))
        failures = _pandera_failures(df)
        assert 0 in failures
        assert any("trip_distance" in r for r in failures[0])

    def test_extreme_trip_distance(self):
        df = _make_df(_valid_row(trip_distance=312_722.0))
        failures = _pandera_failures(df)
        assert 0 in failures
        assert any("trip_distance" in r for r in failures[0])

    def test_location_id_out_of_range(self):
        df = _make_df(_valid_row(PULocationID=np.int32(300)))
        failures = _pandera_failures(df)
        assert 0 in failures
        assert any("PULocationID" in r for r in failures[0])

    def test_nullable_column_allows_none(self):
        """Null passenger_count should NOT generate a failure."""
        df = _make_df(_valid_row(passenger_count=None))
        # Pandera nullable=True columns skip null checks.
        failures = _pandera_failures(df)
        assert not any("passenger_count" in r for rs in failures.values() for r in rs)

    def test_rate_code_99_allowed(self):
        """RatecodeID=99 is explicitly documented as valid (unknown trips)."""
        df = _make_df(_valid_row(RatecodeID=99.0))
        failures = _pandera_failures(df)
        assert not any("RatecodeID" in r for rs in failures.values() for r in rs)

    def test_multiple_bad_columns_same_row(self):
        df = _make_df(_valid_row(VendorID=np.int32(99), trip_distance=-5.0))
        failures = _pandera_failures(df)
        assert 0 in failures
        reasons = failures[0]
        assert any("VendorID" in r for r in reasons)
        assert any("trip_distance" in r for r in reasons)

    def test_only_bad_rows_indexed(self):
        """Row 1 is good, row 0 is bad — only row 0 should appear."""
        df = _make_df(
            _valid_row(trip_distance=-1.0),  # row 0 — bad
            _valid_row(),                    # row 1 — good
        )
        failures = _pandera_failures(df)
        assert 0 in failures
        assert 1 not in failures

    def test_extreme_fare_flagged(self):
        df = _make_df(_valid_row(fare_amount=-999_999.0))
        failures = _pandera_failures(df)
        assert 0 in failures
        assert any("fare_amount" in r for r in failures[0])

    def test_invalid_payment_type(self):
        df = _make_df(_valid_row(payment_type=np.int64(99)))
        failures = _pandera_failures(df)
        assert 0 in failures
        assert any("payment_type" in r for r in failures[0])


# ---------------------------------------------------------------------------
# _temporal_failures
# ---------------------------------------------------------------------------

class TestTemporalFailures:
    def test_dropoff_before_pickup(self):
        df = _make_df(_valid_row(
            tpep_pickup_datetime=pd.Timestamp("2024-01-15 09:00:00"),
            tpep_dropoff_datetime=pd.Timestamp("2024-01-15 08:00:00"),
        ))
        failures = _temporal_failures(df)
        assert 0 in failures
        assert failures[0] == ["tpep_dropoff_datetime:lt_pickup"]

    def test_equal_timestamps_ok(self):
        ts = pd.Timestamp("2024-01-15 08:30:00")
        df = _make_df(_valid_row(
            tpep_pickup_datetime=ts,
            tpep_dropoff_datetime=ts,
        ))
        # Equal timestamps are not < pickup, so should pass.
        failures = _temporal_failures(df)
        assert failures == {}

    def test_normal_trip_no_failure(self):
        df = _make_df(_valid_row())
        assert _temporal_failures(df) == {}

    def test_mixed_good_and_bad_rows(self):
        df = _make_df(
            _valid_row(),  # row 0 — good
            _valid_row(   # row 1 — bad
                tpep_pickup_datetime=pd.Timestamp("2024-01-15 10:00:00"),
                tpep_dropoff_datetime=pd.Timestamp("2024-01-15 09:00:00"),
            ),
        )
        failures = _temporal_failures(df)
        assert 0 not in failures
        assert 1 in failures


# ---------------------------------------------------------------------------
# _merge_failures
# ---------------------------------------------------------------------------

class TestMergeFailures:
    def test_non_overlapping_sources(self):
        a = {0: ["VendorID:isin='99'"]}
        b = {1: ["trip_distance:ge=-1.0"]}
        merged = _merge_failures(a, b)
        assert merged == {0: ["VendorID:isin='99'"], 1: ["trip_distance:ge=-1.0"]}

    def test_overlapping_rows_merged(self):
        a = {0: ["VendorID:isin='99'"]}
        b = {0: ["tpep_dropoff_datetime:lt_pickup"]}
        merged = _merge_failures(a, b)
        assert len(merged[0]) == 2

    def test_empty_sources(self):
        assert _merge_failures({}, {}) == {}


# ---------------------------------------------------------------------------
# run() integration test (uses tmp_path, no real downloads)
# ---------------------------------------------------------------------------

class TestRunValidation:
    @pytest.fixture()
    def data_dirs(self, tmp_path, monkeypatch):
        """Redirect settings paths to a temp directory."""
        monkeypatch.setattr("taxi_etl.stages.validate.settings.data_dir", tmp_path)
        monkeypatch.setattr("taxi_etl.stages.validate.settings.reports_dir", tmp_path / "reports")
        monkeypatch.setattr("taxi_etl.stages.validate.settings.taxi_type", "yellow")
        return tmp_path

    def _write_raw(self, data_dirs: Path, df: pd.DataFrame, month: str = "2024-01") -> Path:
        raw_dir = data_dirs / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"yellow_tripdata_{month}.parquet"
        df.to_parquet(path, index=False)
        return path

    def test_all_clean_rows_no_quarantine(self, data_dirs):
        df = _make_df(_valid_row(), _valid_row(), _valid_row())
        self._write_raw(data_dirs, df)

        result = run("2024-01")

        assert isinstance(result, ValidationResult)
        assert result.total_rows == 3
        assert result.valid_rows == 3
        assert result.quarantined_rows == 0
        assert result.quarantine_path is None
        assert result.valid_path.exists()

    def test_bad_rows_quarantined(self, data_dirs):
        df = _make_df(
            _valid_row(),                          # row 0 — good
            _valid_row(VendorID=np.int32(99)),     # row 1 — bad
            _valid_row(trip_distance=-5.0),        # row 2 — bad
        )
        self._write_raw(data_dirs, df)

        result = run("2024-01")

        assert result.valid_rows == 1
        assert result.quarantined_rows == 2
        assert result.quarantine_path is not None
        assert result.quarantine_path.exists()

        # Quarantine file must have the annotation column.
        qdf = pd.read_parquet(result.quarantine_path)
        assert "_quarantine_reasons" in qdf.columns
        assert len(qdf) == 2

    def test_temporal_failure_quarantined(self, data_dirs):
        bad = _valid_row(
            tpep_pickup_datetime=pd.Timestamp("2024-01-15 10:00:00"),
            tpep_dropoff_datetime=pd.Timestamp("2024-01-15 09:00:00"),
        )
        df = _make_df(_valid_row(), bad)
        self._write_raw(data_dirs, df)

        result = run("2024-01")
        assert result.quarantined_rows == 1
        assert result.valid_rows == 1

    def test_report_json_written(self, data_dirs):
        df = _make_df(_valid_row(), _valid_row(VendorID=np.int32(99)))
        self._write_raw(data_dirs, df)

        run("2024-01")

        report_path = data_dirs / "reports" / "validation_yellow_2024-01.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["total_rows"] == 2
        assert report["quarantined_rows"] == 1
        assert report["valid_rows"] == 1
        assert "failure_counts" in report

    def test_missing_file_raises(self, data_dirs):
        with pytest.raises(FileNotFoundError):
            run("2024-99")

    def test_valid_output_excludes_quarantine_column(self, data_dirs):
        """The _quarantine_reasons column must NOT appear in valid output."""
        df = _make_df(_valid_row(), _valid_row(trip_distance=-1.0))
        self._write_raw(data_dirs, df)

        result = run("2024-01")
        valid_df = pd.read_parquet(result.valid_path)
        assert "_quarantine_reasons" not in valid_df.columns
