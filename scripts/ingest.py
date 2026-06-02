"""CLI entry point for the ingestion stage.

Downloads one or more months of NYC TLC trip data into data/raw/, recording
each in the manifest so reruns skip already-ingested partitions.

Usage:
    # ingest January 2024
    python scripts/ingest.py --month 2024-01

    # ingest a range (inclusive on both ends)
    python scripts/ingest.py --start 2024-01 --end 2024-03

    # force re-download even if already in manifest
    python scripts/ingest.py --month 2024-01 --force
"""

from __future__ import annotations

import argparse
import calendar
import sys
from datetime import datetime

from taxi_etl.logging_setup import get_logger
from taxi_etl.stages.ingest import run, run_range

log = get_logger("cli.ingest")


def _months_between(start: str, end: str) -> list[str]:
    """Return list of YYYY-MM strings from *start* to *end*, inclusive."""
    s = datetime.strptime(start, "%Y-%m")
    e = datetime.strptime(end, "%Y-%m")
    if s > e:
        raise ValueError(f"--start {start} is after --end {end}")

    months = []
    cur = s
    while cur <= e:
        months.append(cur.strftime("%Y-%m"))
        # Advance by one month
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return months


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NYC TLC monthly parquet files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--month", help="Single month to ingest (YYYY-MM).")
    group.add_argument("--start", help="Start of month range (YYYY-MM). Requires --end.")
    parser.add_argument("--end", help="End of month range, inclusive (YYYY-MM). Requires --start.")
    parser.add_argument("--force", action="store_true", help="Re-download even if already manifested.")
    args = parser.parse_args()

    if args.start and not args.end:
        parser.error("--end is required when --start is given")

    if args.month:
        result = run(args.month, force=args.force)
        action = "skipped (cached)" if result.skipped else "ingested"
        log.info(
            "%s %s/%s — %d rows, %.1f MB",
            action, result.taxi_type, result.month,
            result.row_count, result.size_bytes / 1e6,
        )
    else:
        months = _months_between(args.start, args.end)
        log.info("ingesting %d month(s): %s → %s", len(months), months[0], months[-1])
        results = run_range(months, force=args.force)
        skipped = sum(1 for r in results if r.skipped)
        downloaded = len(results) - skipped
        total_rows = sum(r.row_count for r in results)
        total_mb = sum(r.size_bytes for r in results) / 1e6
        log.info(
            "done — downloaded=%d  skipped=%d  total_rows=%d  total_size=%.1f MB",
            downloaded, skipped, total_rows, total_mb,
        )


if __name__ == "__main__":
    main()
