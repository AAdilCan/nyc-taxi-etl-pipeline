"""Transformation stage: turn validated trips into an ML-ready feature table.

The validation stage guarantees rows are *schema-valid* (correct dtypes, values
inside documented ranges).  This stage goes further and produces a feature
table suitable for modelling:

  1. Type coercion / downcasting — datetimes parsed, categorical columns made
     ``category``, integer ids downcast to save memory.
  2. Derived features — trip duration, average speed, time-of-day buckets, and
     fare ratios that a model would actually use.
  3. Business-rule outlier filtering — drop rows that pass the schema but are
     physically implausible (4-second trips, 250 mph speeds, zero fares). Every
     drop is attributed to a reason and counted in the report.
  4. Partitioned output — the result is written as a parquet dataset
     partitioned by ``pickup_date`` so downstream consumers can read a single
     day without scanning the whole month.

The clean input lives in ``data/processed/validated_*.parquet`` (produced by the
validation stage); the output dataset lives under ``data/features/`` and a JSON
summary is written to ``reports/``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from taxi_etl.config import settings
from taxi_etl.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Column groups used during coercion
# ---------------------------------------------------------------------------

_DATETIME_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]

# Downcast targets: keeping ids small shaves a lot of memory on large months.
_INT16_COLS = ["PULocationID", "DOLocationID", "RatecodeID"]
_INT8_COLS = ["VendorID", "payment_type"]
_CATEGORY_COLS = ["store_and_fwd_flag", "time_of_day", "pickup_day_name"]

# Time-of-day buckets keyed by the half-open hour ranges [start, end).
_TIME_OF_DAY_BUCKETS: list[tuple[int, int, str]] = [
    (0, 6, "night"),
    (6, 12, "morning"),
    (12, 17, "afternoon"),
    (17, 21, "evening"),
    (21, 24, "night"),
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TransformReport:
    month: str
    taxi_type: str
    input_rows: int
    output_rows: int
    dropped_rows: int
    output_pct: float
    dropped_by_reason: dict[str, int]
    feature_columns: list[str]
    n_partitions: int
    output_path: str


@dataclass
class TransformResult:
    month: str
    taxi_type: str
    input_rows: int
    output_rows: int
    dropped_rows: int
    output_path: Path
    report: TransformReport
    frame: pd.DataFrame = field(repr=False)


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parse datetimes and downcast numeric/categorical columns in place-safe copy."""
    out = df.copy()

    for col in _DATETIME_COLS:
        out[col] = pd.to_datetime(out[col], errors="coerce")

    for col in _INT8_COLS:
        if col in out.columns:
            out[col] = out[col].astype("int8", errors="ignore")

    for col in _INT16_COLS:
        if col in out.columns:
            # RatecodeID is float (carries 99.0 sentinel); round before cast.
            out[col] = out[col].round().astype("Int16")

    if "passenger_count" in out.columns:
        out["passenger_count"] = out["passenger_count"].round().astype("Int8")

    return out


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _time_of_day(hour: pd.Series) -> pd.Series:
    """Map an hour-of-day series to coarse time-of-day buckets."""
    result = pd.Series(np.empty(len(hour), dtype=object), index=hour.index)
    for start, end, label in _TIME_OF_DAY_BUCKETS:
        result[(hour >= start) & (hour < end)] = label
    return result


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive duration, speed, temporal, and fare-ratio features.

    Ratios guard against division by zero by substituting ``NaN`` where the
    denominator is non-positive; rows with implausible derived values are
    removed later by :func:`filter_outliers`, not here.
    """
    out = df.copy()

    pickup = out["tpep_pickup_datetime"]
    dropoff = out["tpep_dropoff_datetime"]

    # Duration in minutes (float); negative/zero handled downstream.
    duration_min = (dropoff - pickup).dt.total_seconds() / 60.0
    out["trip_duration_min"] = duration_min

    # Average speed in mph; NaN when duration is non-positive.
    duration_h = duration_min / 60.0
    out["trip_speed_mph"] = np.where(
        duration_h > 0, out["trip_distance"] / duration_h, np.nan
    )

    # Temporal features from pickup time.
    out["pickup_hour"] = pickup.dt.hour.astype("int8")
    out["pickup_dayofweek"] = pickup.dt.dayofweek.astype("int8")  # Mon=0
    out["pickup_day_name"] = pickup.dt.day_name()
    out["is_weekend"] = pickup.dt.dayofweek >= 5
    out["time_of_day"] = _time_of_day(pickup.dt.hour)

    # Rush hour: weekday mornings 7–10 and evenings 16–19.
    weekday = pickup.dt.dayofweek < 5
    morning_rush = pickup.dt.hour.between(7, 9)
    evening_rush = pickup.dt.hour.between(16, 18)
    out["is_rush_hour"] = weekday & (morning_rush | evening_rush)

    # Fare ratios — common, informative model inputs.
    out["fare_per_mile"] = np.where(
        out["trip_distance"] > 0, out["fare_amount"] / out["trip_distance"], np.nan
    )
    out["tip_pct"] = np.where(
        out["fare_amount"] > 0, 100.0 * out["tip_amount"] / out["fare_amount"], np.nan
    )
    out["cost_per_min"] = np.where(
        duration_min > 0, out["total_amount"] / duration_min, np.nan
    )

    # Airport_fee is nullable in raw data; treat missing as no airport fee.
    if "Airport_fee" in out.columns:
        out["is_airport_trip"] = out["Airport_fee"].fillna(0) > 0
    else:
        out["is_airport_trip"] = False

    # Partition key: one parquet directory per calendar day.
    out["pickup_date"] = pickup.dt.date.astype("string")

    return out


# ---------------------------------------------------------------------------
# Outlier filtering
# ---------------------------------------------------------------------------

def filter_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop physically implausible rows; return (clean_df, dropped_by_reason).

    A single row may violate several rules; each violated rule is counted so the
    report shows *why* rows were removed, but the row is dropped only once.
    """
    reasons: dict[str, pd.Series] = {
        "duration_too_short": df["trip_duration_min"] < settings.min_trip_duration_min,
        "duration_too_long": df["trip_duration_min"] > settings.max_trip_duration_min,
        "speed_implausible": df["trip_speed_mph"] > settings.max_trip_speed_mph,
        "non_positive_fare": df["fare_amount"] < settings.min_fare_amount,
        "zero_distance": df["trip_distance"] <= 0,
        "too_few_passengers": (
            df["passenger_count"] < settings.min_passenger_count
        ).fillna(False),
    }

    dropped_by_reason = {name: int(mask.sum()) for name, mask in reasons.items()}

    drop_mask = pd.Series(False, index=df.index)
    for mask in reasons.values():
        drop_mask |= mask.fillna(False)

    clean = df[~drop_mask].copy()
    return clean, dropped_by_reason


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_partitioned(df: pd.DataFrame, dest: Path) -> int:
    """Write *df* as a parquet dataset partitioned by ``pickup_date``.

    Returns the number of partitions written.  An existing destination is
    removed first so reruns are idempotent rather than appending duplicates.
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    df.to_parquet(dest, partition_cols=["pickup_date"], index=False)
    return df["pickup_date"].nunique()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run(month: str, *, input_path: Path | None = None) -> TransformResult:
    """Transform one month of validated data into the ML-ready feature dataset.

    Args:
        month:       Target month in ``YYYY-MM`` format.
        input_path:  Override the default validated-parquet path (for tests).

    Returns:
        ``TransformResult`` with row counts, output path, and the feature frame.
    """
    settings.ensure_dirs()
    taxi_type = settings.taxi_type

    if input_path is None:
        input_path = (
            settings.processed_dir / f"validated_{taxi_type}_{month}.parquet"
        )
    if not input_path.exists():
        raise FileNotFoundError(f"Validated parquet not found: {input_path}")

    log.info("loading %s", input_path.name)
    df = pd.read_parquet(input_path)
    input_rows = len(df)
    log.info("loaded %d validated rows", input_rows)

    df = coerce_types(df)
    df = add_features(df)
    log.info("engineered %d feature columns", len(df.columns))

    clean, dropped_by_reason = filter_outliers(df)
    output_rows = len(clean)
    dropped_rows = input_rows - output_rows
    log.info(
        "kept %d (%.2f%%), dropped %d outlier rows",
        output_rows,
        100.0 * output_rows / input_rows if input_rows else 0.0,
        dropped_rows,
    )
    for reason, count in dropped_by_reason.items():
        if count:
            log.info("  dropped by %s: %d", reason, count)

    # Final category coercion for compact, ML-friendly storage.
    for col in _CATEGORY_COLS:
        if col in clean.columns:
            clean[col] = clean[col].astype("category")

    dest = settings.features_dir / f"{taxi_type}_{month}"
    n_partitions = _write_partitioned(clean, dest)
    log.info("wrote %d partitions → %s", n_partitions, dest)

    report = TransformReport(
        month=month,
        taxi_type=taxi_type,
        input_rows=input_rows,
        output_rows=output_rows,
        dropped_rows=dropped_rows,
        output_pct=round(100.0 * output_rows / input_rows, 4) if input_rows else 0.0,
        dropped_by_reason=dropped_by_reason,
        feature_columns=list(clean.columns),
        n_partitions=n_partitions,
        output_path=str(dest),
    )
    report_path = settings.reports_dir / f"transform_{taxi_type}_{month}.json"
    report_path.write_text(json.dumps(asdict(report), indent=2))
    log.info("wrote transform report → %s", report_path.name)

    return TransformResult(
        month=month,
        taxi_type=taxi_type,
        input_rows=input_rows,
        output_rows=output_rows,
        dropped_rows=dropped_rows,
        output_path=dest,
        report=report,
        frame=clean,
    )
