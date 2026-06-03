"""Validation stage: apply Pandera schema to raw NYC TLC data, quarantine bad rows.

Runs two categories of checks:
  1. Column-level schema checks via Pandera (dtype coercion, value ranges, allowed
     sets).  Pandera is run in lazy mode so ALL failures are collected before any
     rows are dropped.
  2. Cross-column temporal check: dropoff must not precede pickup.

Rows that fail any check are written to ``data/quarantine/`` with an added
``_quarantine_reasons`` column that lists every violated rule.  Clean rows go
to ``data/processed/`` as a validated parquet.  A JSON summary report is
written to ``reports/``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from taxi_etl.config import settings
from taxi_etl.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

# Allowed vendor IDs as seen in TLC data dictionary (1=CMT, 2=VeriFone,
# 6 and 7 are newer providers added in later data years).
_VALID_VENDOR_IDS: list[int] = [1, 2, 6, 7]

# RatecodeID 99 = "unknown" — officially documented in the TLC data dictionary.
_VALID_RATE_CODES: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0]

# Payment codes from TLC data dictionary (0 used for some automated reports).
_VALID_PAYMENT_TYPES: list[int] = [0, 1, 2, 3, 4, 5, 6]

# TLC zones are numbered 1–265; values outside this range are mapping errors.
_LOCATION_ID_MIN = 1
_LOCATION_ID_MAX = 265

# Fare bounds: negative values represent refunds (observed min ~-900);
# positives above $10k indicate data-entry errors rather than real fares.
_FARE_BOUND_LOW = -1_000.0
_FARE_BOUND_HIGH = 10_000.0

# Trip distance: 500 miles is a conservative upper cap; observed max ~313k miles
# is clearly erroneous. Negative values are impossible.
_DISTANCE_MAX = 500.0

RAW_SCHEMA = DataFrameSchema(
    columns={
        "VendorID": Column(
            pa.Int32,
            checks=Check.isin(_VALID_VENDOR_IDS),
            nullable=False,
        ),
        "tpep_pickup_datetime": Column(pa.DateTime, nullable=False),
        "tpep_dropoff_datetime": Column(pa.DateTime, nullable=False),
        "passenger_count": Column(
            pa.Float64,
            checks=[Check.ge(0), Check.le(9)],
            nullable=True,
        ),
        "trip_distance": Column(
            pa.Float64,
            checks=[Check.ge(0), Check.le(_DISTANCE_MAX)],
            nullable=False,
        ),
        "RatecodeID": Column(
            pa.Float64,
            checks=Check.isin(_VALID_RATE_CODES),
            nullable=True,
        ),
        "store_and_fwd_flag": Column(
            pa.Object,
            checks=Check.isin(["Y", "N"]),
            nullable=True,
        ),
        "PULocationID": Column(
            pa.Int32,
            checks=[Check.ge(_LOCATION_ID_MIN), Check.le(_LOCATION_ID_MAX)],
            nullable=False,
        ),
        "DOLocationID": Column(
            pa.Int32,
            checks=[Check.ge(_LOCATION_ID_MIN), Check.le(_LOCATION_ID_MAX)],
            nullable=False,
        ),
        "payment_type": Column(
            pa.Int64,
            checks=Check.isin(_VALID_PAYMENT_TYPES),
            nullable=False,
        ),
        "fare_amount": Column(
            pa.Float64,
            checks=[Check.ge(_FARE_BOUND_LOW), Check.le(_FARE_BOUND_HIGH)],
            nullable=False,
        ),
        "extra": Column(pa.Float64, nullable=False),
        "mta_tax": Column(pa.Float64, nullable=False),
        "tip_amount": Column(pa.Float64, nullable=False),
        "tolls_amount": Column(pa.Float64, nullable=False),
        "improvement_surcharge": Column(pa.Float64, nullable=False),
        "total_amount": Column(
            pa.Float64,
            checks=[Check.ge(_FARE_BOUND_LOW), Check.le(_FARE_BOUND_HIGH)],
            nullable=False,
        ),
        "congestion_surcharge": Column(pa.Float64, nullable=True),
        "Airport_fee": Column(pa.Float64, nullable=True),
    },
    # strict=False: tolerate extra columns added by future TLC schema revisions.
    strict=False,
    coerce=False,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    month: str
    taxi_type: str
    total_rows: int
    valid_rows: int
    quarantined_rows: int
    valid_pct: float
    quarantined_pct: float
    failure_counts: dict[str, int]
    valid_path: str
    quarantine_path: str | None


@dataclass
class ValidationResult:
    month: str
    taxi_type: str
    total_rows: int
    valid_rows: int
    quarantined_rows: int
    valid_path: Path
    quarantine_path: Path | None
    report: ValidationReport


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pandera_failures(df: pd.DataFrame) -> dict[int, list[str]]:
    """Run Pandera in lazy mode; return {row_index: [violated_rule, ...]}."""
    try:
        RAW_SCHEMA.validate(df, lazy=True)
        return {}
    except pa.errors.SchemaErrors as exc:
        failures: dict[int, list[str]] = {}
        fc: pd.DataFrame = exc.failure_cases

        for _, row in fc.iterrows():
            raw_idx = row.get("index")
            if raw_idx is None or (isinstance(raw_idx, float) and pd.isna(raw_idx)):
                # Schema-level error (e.g. missing column) — not row-specific.
                continue

            idx = int(raw_idx)
            column = row.get("column", "unknown")
            check = row.get("check", "unknown")
            case = row.get("failure_case", "")
            failures.setdefault(idx, []).append(f"{column}:{check}={case!r}")

        return failures


def _temporal_failures(df: pd.DataFrame) -> dict[int, list[str]]:
    """Identify rows where dropoff precedes pickup."""
    mask = df["tpep_dropoff_datetime"] < df["tpep_pickup_datetime"]
    result: dict[int, list[str]] = {}
    for idx in df.index[mask]:
        result[int(idx)] = ["tpep_dropoff_datetime:lt_pickup"]
    return result


def _merge_failures(
    *sources: dict[int, list[str]],
) -> dict[int, list[str]]:
    merged: dict[int, list[str]] = {}
    for src in sources:
        for idx, reasons in src.items():
            merged.setdefault(idx, []).extend(reasons)
    return merged


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run(month: str, *, input_path: Path | None = None) -> ValidationResult:
    """Validate one month of raw NYC TLC data and quarantine bad rows.

    Reads the raw parquet for *month*, applies the Pandera schema and temporal
    checks, writes clean rows to ``data/processed/`` and bad rows to
    ``data/quarantine/``, then saves a JSON report to ``reports/``.

    Args:
        month:       Target month in ``YYYY-MM`` format.
        input_path:  Override the default raw file path (useful in tests).

    Returns:
        ``ValidationResult`` with counts and output paths.
    """
    settings.ensure_dirs()
    taxi_type = settings.taxi_type

    if input_path is None:
        input_path = settings.raw_dir / f"{taxi_type}_tripdata_{month}.parquet"

    if not input_path.exists():
        raise FileNotFoundError(f"Raw parquet not found: {input_path}")

    log.info("loading %s", input_path.name)
    df = pd.read_parquet(input_path)
    total_rows = len(df)
    log.info("loaded %d rows, %d columns", total_rows, len(df.columns))

    # Collect all failures before dropping any rows.
    log.info("running Pandera schema checks (lazy mode)…")
    schema_failures = _pandera_failures(df)
    log.info("running temporal checks…")
    temporal_fails = _temporal_failures(df)

    all_failures = _merge_failures(schema_failures, temporal_fails)
    bad_indices: set[int] = set(all_failures.keys())

    valid_df = df[~df.index.isin(bad_indices)].copy()
    quarantine_df = df[df.index.isin(bad_indices)].copy()

    if not quarantine_df.empty:
        quarantine_df["_quarantine_reasons"] = [
            "; ".join(all_failures[int(i)]) for i in quarantine_df.index
        ]

    valid_rows = len(valid_df)
    quarantined_rows = len(quarantine_df)
    log.info(
        "valid: %d (%.2f%%), quarantined: %d (%.2f%%)",
        valid_rows, 100.0 * valid_rows / total_rows,
        quarantined_rows, 100.0 * quarantined_rows / total_rows,
    )

    # Write outputs.
    valid_path = settings.processed_dir / f"validated_{taxi_type}_{month}.parquet"
    valid_df.to_parquet(valid_path, index=False)
    log.info("wrote valid parquet → %s", valid_path.name)

    quarantine_path: Path | None = None
    if quarantined_rows > 0:
        quarantine_path = (
            settings.quarantine_dir / f"quarantine_{taxi_type}_{month}.parquet"
        )
        quarantine_df.to_parquet(quarantine_path, index=False)
        log.info(
            "wrote quarantine parquet → %s (%d rows)",
            quarantine_path.name,
            quarantined_rows,
        )

    # Aggregate failure counts per check (column:check, not per case value).
    failure_counts: dict[str, int] = {}
    for reasons in all_failures.values():
        for reason in reasons:
            check_key = reason.rsplit("=", 1)[0]  # drop the actual value
            failure_counts[check_key] = failure_counts.get(check_key, 0) + 1

    report = ValidationReport(
        month=month,
        taxi_type=taxi_type,
        total_rows=total_rows,
        valid_rows=valid_rows,
        quarantined_rows=quarantined_rows,
        valid_pct=round(100.0 * valid_rows / total_rows, 4),
        quarantined_pct=round(100.0 * quarantined_rows / total_rows, 4),
        failure_counts=failure_counts,
        valid_path=str(valid_path),
        quarantine_path=str(quarantine_path) if quarantine_path else None,
    )

    report_path = settings.reports_dir / f"validation_{taxi_type}_{month}.json"
    report_path.write_text(json.dumps(asdict(report), indent=2))
    log.info("wrote validation report → %s", report_path.name)

    return ValidationResult(
        month=month,
        taxi_type=taxi_type,
        total_rows=total_rows,
        valid_rows=valid_rows,
        quarantined_rows=quarantined_rows,
        valid_path=valid_path,
        quarantine_path=quarantine_path,
        report=report,
    )
