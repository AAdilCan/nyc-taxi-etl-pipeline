"""One-off data exploration for a single month of NYC TLC trip data.

Downloads (or reuses a cached copy of) one monthly parquet file, then prints a
profile: schema, row count, null fractions, and summary statistics for the
numeric columns. The profile is also written to ``reports/profile_<month>.json``
so the schema decisions made later in the pipeline are traceable.

Usage:
    python scripts/explore_data.py --month 2024-01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

from taxi_etl.config import settings
from taxi_etl.logging_setup import get_logger

log = get_logger("explore")


def _file_name(month: str) -> str:
    return f"{settings.taxi_type}_tripdata_{month}.parquet"


def download_month(month: str) -> Path:
    """Download a month's parquet into data/raw if not already cached."""
    settings.ensure_dirs()
    dest = settings.raw_dir / _file_name(month)
    if dest.exists():
        log.info("using cached file %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return dest

    url = f"{settings.base_url}/{_file_name(month)}"
    log.info("downloading %s", url)
    resp = requests.get(url, timeout=settings.download_timeout_s)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    log.info("saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def profile(df: pd.DataFrame) -> dict:
    """Build a JSON-serializable profile of the dataframe."""
    null_fraction = (df.isna().mean()).round(4).to_dict()
    dtypes = {col: str(dt) for col, dt in df.dtypes.items()}

    numeric = df.select_dtypes("number")
    stats = {
        col: {
            "min": float(numeric[col].min()),
            "max": float(numeric[col].max()),
            "mean": round(float(numeric[col].mean()), 4),
            "p50": round(float(numeric[col].quantile(0.5)), 4),
        }
        for col in numeric.columns
    }

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "null_fraction": null_fraction,
        "numeric_stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile one month of NYC TLC data.")
    parser.add_argument("--month", default="2024-01", help="Month as YYYY-MM.")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=0,
        help="If >0, profile only the first N rows (faster, for quick looks).",
    )
    args = parser.parse_args()

    path = download_month(args.month)
    df = pd.read_parquet(path)
    if args.sample_rows > 0:
        df = df.head(args.sample_rows)

    prof = profile(df)

    log.info("rows=%d  cols=%d", prof["rows"], len(prof["columns"]))
    log.info("columns: %s", ", ".join(prof["columns"]))
    high_null = {c: f for c, f in prof["null_fraction"].items() if f > 0}
    if high_null:
        log.info("columns with nulls: %s", high_null)

    out = settings.reports_dir / f"profile_{args.month}.json"
    out.write_text(json.dumps(prof, indent=2))
    log.info("wrote profile to %s", out)


if __name__ == "__main__":
    main()
