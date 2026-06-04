"""Tests for the transformation stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from taxi_etl.stages.transform import (
    TransformResult,
    add_features,
    coerce_types,
    filter_outliers,
    run,
)


# ---------------------------------------------------------------------------
# Validated-row factory (mirrors the validation stage's clean output)
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


def _make_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------------------
# coerce_types
# ---------------------------------------------------------------------------

class TestCoerceTypes:
    def test_datetimes_parsed_from_strings(self):
        df = _make_df(_valid_row(tpep_pickup_datetime="2024-01-15 08:30:00"))
        out = coerce_types(df)
        assert pd.api.types.is_datetime64_any_dtype(out["tpep_pickup_datetime"])

    def test_ids_downcast(self):
        out = coerce_types(_make_df(_valid_row()))
        assert out["VendorID"].dtype == np.int8
        assert out["PULocationID"].dtype == "Int16"

    def test_passenger_count_nullable_int(self):
        out = coerce_types(_make_df(_valid_row(passenger_count=None)))
        assert out["passenger_count"].dtype == "Int8"
        assert pd.isna(out["passenger_count"].iloc[0])

    def test_ratecode_99_sentinel_preserved(self):
        out = coerce_types(_make_df(_valid_row(RatecodeID=99.0)))
        assert out["RatecodeID"].iloc[0] == 99


# ---------------------------------------------------------------------------
# add_features
# ---------------------------------------------------------------------------

class TestAddFeatures:
    def _featured(self, **overrides) -> pd.Series:
        df = coerce_types(_make_df(_valid_row(**overrides)))
        return add_features(df).iloc[0]

    def test_trip_duration_minutes(self):
        row = self._featured()
        assert row["trip_duration_min"] == pytest.approx(25.0)

    def test_speed_mph(self):
        # 3.2 miles in 25 min -> 7.68 mph.
        row = self._featured()
        assert row["trip_speed_mph"] == pytest.approx(3.2 / (25 / 60))

    def test_speed_nan_when_zero_duration(self):
        ts = pd.Timestamp("2024-01-15 08:30:00")
        row = self._featured(tpep_pickup_datetime=ts, tpep_dropoff_datetime=ts)
        assert pd.isna(row["trip_speed_mph"])

    def test_time_of_day_buckets(self):
        assert self._featured(
            tpep_pickup_datetime=pd.Timestamp("2024-01-15 03:00:00"),
            tpep_dropoff_datetime=pd.Timestamp("2024-01-15 03:20:00"),
        )["time_of_day"] == "night"
        assert self._featured(
            tpep_pickup_datetime=pd.Timestamp("2024-01-15 14:00:00"),
            tpep_dropoff_datetime=pd.Timestamp("2024-01-15 14:20:00"),
        )["time_of_day"] == "afternoon"

    def test_weekend_flag(self):
        # 2024-01-13 is a Saturday.
        row = self._featured(
            tpep_pickup_datetime=pd.Timestamp("2024-01-13 12:00:00"),
            tpep_dropoff_datetime=pd.Timestamp("2024-01-13 12:20:00"),
        )
        assert bool(row["is_weekend"]) is True

    def test_rush_hour_weekday_morning(self):
        # 2024-01-15 is a Monday; 08:30 pickup is within 7–9 morning rush.
        assert bool(self._featured()["is_rush_hour"]) is True

    def test_rush_hour_excludes_weekend(self):
        # Saturday 08:30 is not rush hour even though the hour matches.
        row = self._featured(
            tpep_pickup_datetime=pd.Timestamp("2024-01-13 08:30:00"),
            tpep_dropoff_datetime=pd.Timestamp("2024-01-13 08:55:00"),
        )
        assert bool(row["is_rush_hour"]) is False

    def test_tip_pct(self):
        # tip 3.0 on fare 14.5 -> ~20.69%.
        row = self._featured()
        assert row["tip_pct"] == pytest.approx(100 * 3.0 / 14.5)

    def test_fare_per_mile_nan_when_zero_distance(self):
        row = self._featured(trip_distance=0.0)
        assert pd.isna(row["fare_per_mile"])

    def test_airport_trip_flag(self):
        assert bool(self._featured(Airport_fee=1.75)["is_airport_trip"]) is True
        assert bool(self._featured(Airport_fee=0.0)["is_airport_trip"]) is False

    def test_pickup_date_partition_key(self):
        assert self._featured()["pickup_date"] == "2024-01-15"


# ---------------------------------------------------------------------------
# filter_outliers
# ---------------------------------------------------------------------------

class TestFilterOutliers:
    def _featured(self, *rows: dict) -> pd.DataFrame:
        return add_features(coerce_types(_make_df(*rows)))

    def test_clean_row_kept(self):
        clean, reasons = filter_outliers(self._featured(_valid_row()))
        assert len(clean) == 1
        assert sum(reasons.values()) == 0

    def test_short_trip_dropped(self):
        df = self._featured(
            _valid_row(),
            _valid_row(tpep_dropoff_datetime=pd.Timestamp("2024-01-15 08:30:05")),
        )
        clean, reasons = filter_outliers(df)
        assert len(clean) == 1
        assert reasons["duration_too_short"] == 1

    def test_long_trip_dropped(self):
        df = self._featured(
            _valid_row(tpep_dropoff_datetime=pd.Timestamp("2024-01-15 12:30:00")),
        )
        clean, reasons = filter_outliers(df)
        assert len(clean) == 0
        assert reasons["duration_too_long"] == 1

    def test_zero_distance_dropped(self):
        clean, reasons = filter_outliers(self._featured(_valid_row(trip_distance=0.0)))
        assert len(clean) == 0
        assert reasons["zero_distance"] == 1

    def test_non_positive_fare_dropped(self):
        clean, reasons = filter_outliers(self._featured(_valid_row(fare_amount=0.0)))
        assert len(clean) == 0
        assert reasons["non_positive_fare"] == 1

    def test_zero_passengers_dropped(self):
        clean, reasons = filter_outliers(self._featured(_valid_row(passenger_count=0.0)))
        assert len(clean) == 0
        assert reasons["too_few_passengers"] == 1

    def test_null_passengers_kept(self):
        """Null passenger_count must not be treated as < min and dropped."""
        clean, reasons = filter_outliers(self._featured(_valid_row(passenger_count=None)))
        assert len(clean) == 1
        assert reasons["too_few_passengers"] == 0

    def test_row_violating_multiple_rules_dropped_once(self):
        # 5-second trip over zero distance: two reasons, one dropped row.
        df = self._featured(
            _valid_row(
                trip_distance=0.0,
                tpep_dropoff_datetime=pd.Timestamp("2024-01-15 08:30:05"),
            )
        )
        clean, reasons = filter_outliers(df)
        assert len(clean) == 0
        assert reasons["zero_distance"] == 1
        assert reasons["duration_too_short"] == 1


# ---------------------------------------------------------------------------
# run() integration
# ---------------------------------------------------------------------------

class TestRunTransform:
    @pytest.fixture()
    def data_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr("taxi_etl.stages.transform.settings.data_dir", tmp_path)
        monkeypatch.setattr(
            "taxi_etl.stages.transform.settings.reports_dir", tmp_path / "reports"
        )
        monkeypatch.setattr("taxi_etl.stages.transform.settings.taxi_type", "yellow")
        return tmp_path

    def _write_validated(self, data_dirs: Path, df: pd.DataFrame, month="2024-01") -> Path:
        proc = data_dirs / "processed"
        proc.mkdir(parents=True, exist_ok=True)
        path = proc / f"validated_yellow_{month}.parquet"
        df.to_parquet(path, index=False)
        return path

    def test_end_to_end(self, data_dirs):
        df = _make_df(
            _valid_row(),                                   # good
            _valid_row(trip_distance=0.0),                  # dropped
            _valid_row(fare_amount=-1.0),                   # dropped
        )
        self._write_validated(data_dirs, df)

        result = run("2024-01")

        assert isinstance(result, TransformResult)
        assert result.input_rows == 3
        assert result.output_rows == 1
        assert result.dropped_rows == 2
        assert result.output_path.exists()

    def test_partitioned_by_date(self, data_dirs):
        df = _make_df(
            _valid_row(
                tpep_pickup_datetime=pd.Timestamp("2024-01-15 08:30:00"),
                tpep_dropoff_datetime=pd.Timestamp("2024-01-15 08:55:00"),
            ),
            _valid_row(
                tpep_pickup_datetime=pd.Timestamp("2024-01-16 08:30:00"),
                tpep_dropoff_datetime=pd.Timestamp("2024-01-16 08:55:00"),
            ),
        )
        self._write_validated(data_dirs, df)

        result = run("2024-01")
        part_dirs = sorted(p.name for p in result.output_path.glob("pickup_date=*"))
        assert part_dirs == ["pickup_date=2024-01-15", "pickup_date=2024-01-16"]
        assert result.report.n_partitions == 2

    def test_output_roundtrips(self, data_dirs):
        self._write_validated(data_dirs, _make_df(_valid_row()))
        result = run("2024-01")
        back = pd.read_parquet(result.output_path)
        assert "trip_duration_min" in back.columns
        assert "pickup_date" in back.columns
        assert len(back) == 1

    def test_rerun_is_idempotent(self, data_dirs):
        """Rerunning must overwrite, not append duplicate partitions."""
        self._write_validated(data_dirs, _make_df(_valid_row()))
        run("2024-01")
        result = run("2024-01")
        back = pd.read_parquet(result.output_path)
        assert len(back) == 1

    def test_report_written(self, data_dirs):
        import json

        self._write_validated(data_dirs, _make_df(_valid_row(), _valid_row(trip_distance=0.0)))
        run("2024-01")
        report_path = data_dirs / "reports" / "transform_yellow_2024-01.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["input_rows"] == 2
        assert report["output_rows"] == 1
        assert report["dropped_by_reason"]["zero_distance"] == 1

    def test_missing_input_raises(self, data_dirs):
        with pytest.raises(FileNotFoundError):
            run("2099-01")
