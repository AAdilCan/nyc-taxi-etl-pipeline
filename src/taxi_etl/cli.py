"""Unified command-line interface for the NYC TLC ETL pipeline.

Exposes the whole pipeline and each stage individually under one entry point::

    taxi-etl run        --month 2024-01            # full pipeline, one month
    taxi-etl run        --start 2024-01 --end 2024-03   # full pipeline, range
    taxi-etl ingest     --month 2024-01            # download only
    taxi-etl validate   --month 2024-01            # validate cached raw file
    taxi-etl transform  --month 2024-01            # transform validated file

The ``run`` command orchestrates ingest → validate → transform, then writes a
JSON run summary and a Markdown data-quality report to ``reports/`` and prints
an overview table.  Per-stage commands are thin wrappers that run a single
stage — handy for re-running just the part that changed.

Installed as the ``taxi-etl`` console script (see ``pyproject.toml``); also
runnable as ``python -m taxi_etl``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from taxi_etl import pipeline, reporting
from taxi_etl.logging_setup import get_logger
from taxi_etl.stages import ingest, transform, validate

log = get_logger("cli")


# ---------------------------------------------------------------------------
# Month/range argument handling
# ---------------------------------------------------------------------------

def months_between(start: str, end: str) -> list[str]:
    """Return ``YYYY-MM`` strings from *start* to *end*, inclusive."""
    s = datetime.strptime(start, "%Y-%m")
    e = datetime.strptime(end, "%Y-%m")
    if s > e:
        raise ValueError(f"--start {start} is after --end {end}")

    months: list[str] = []
    cur = s
    while cur <= e:
        months.append(cur.strftime("%Y-%m"))
        cur = cur.replace(year=cur.year + 1, month=1) if cur.month == 12 else cur.replace(month=cur.month + 1)
    return months


def _resolve_months(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    """Turn ``--month`` or ``--start/--end`` into a concrete list of months."""
    if args.month:
        return [args.month]
    if args.start and not args.end:
        parser.error("--end is required when --start is given")
    try:
        return months_between(args.start, args.end)
    except ValueError as exc:
        parser.error(str(exc))
        raise  # unreachable; parser.error exits


def _add_month_args(p: argparse.ArgumentParser) -> None:
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--month", help="Single month (YYYY-MM).")
    group.add_argument("--start", help="Start of an inclusive month range (YYYY-MM). Needs --end.")
    p.add_argument("--end", help="End of the month range, inclusive (YYYY-MM).")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    months = _resolve_months(args, parser)
    log.info("running full pipeline for %d month(s): %s", len(months), ", ".join(months))

    run = pipeline.run_months(
        months,
        force=args.force,
        continue_on_error=not args.fail_fast,
    )

    if not run.months:
        log.error("no months processed successfully")
        return 1

    if not args.no_report:
        summary_path, dq_path = reporting.write_reports(run)
        log.info("reports: %s , %s", summary_path.name, dq_path.name)

    print(reporting.format_console_summary(run))
    return 0


def _cmd_ingest(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    months = _resolve_months(args, parser)
    results = ingest.run_range(months, force=args.force)
    skipped = sum(1 for r in results if r.skipped)
    total_rows = sum(r.row_count for r in results)
    total_mb = sum(r.size_bytes for r in results) / 1e6
    log.info(
        "ingest done — downloaded=%d skipped=%d rows=%d size=%.1f MB",
        len(results) - skipped, skipped, total_rows, total_mb,
    )
    return 0


def _cmd_validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    months = _resolve_months(args, parser)
    for month in months:
        r = validate.run(month)
        log.info(
            "validate %s — valid=%d (%.2f%%) quarantined=%d",
            month, r.valid_rows, r.report.valid_pct, r.quarantined_rows,
        )
    return 0


def _cmd_transform(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    months = _resolve_months(args, parser)
    for month in months:
        r = transform.run(month)
        log.info(
            "transform %s — kept=%d (%.2f%%) dropped=%d partitions=%d",
            month, r.output_rows, r.report.output_pct, r.dropped_rows, r.report.n_partitions,
        )
    return 0


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="taxi-etl",
        description="NYC TLC trip-data ETL pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run (full pipeline)
    p_run = sub.add_parser("run", help="Run the full pipeline (ingest → validate → transform).")
    _add_month_args(p_run)
    p_run.add_argument("--force", action="store_true", help="Re-download even if already manifested.")
    p_run.add_argument("--fail-fast", action="store_true", help="Stop on the first month that errors.")
    p_run.add_argument("--no-report", action="store_true", help="Skip writing the JSON/Markdown reports.")
    p_run.set_defaults(handler=_cmd_run)

    # ingest
    p_ing = sub.add_parser("ingest", help="Download monthly parquet into data/raw/.")
    _add_month_args(p_ing)
    p_ing.add_argument("--force", action="store_true", help="Re-download even if already manifested.")
    p_ing.set_defaults(handler=_cmd_ingest)

    # validate
    p_val = sub.add_parser("validate", help="Validate cached raw file(s), quarantine bad rows.")
    _add_month_args(p_val)
    p_val.set_defaults(handler=_cmd_validate)

    # transform
    p_tr = sub.add_parser("transform", help="Transform validated file(s) into the feature dataset.")
    _add_month_args(p_tr)
    p_tr.set_defaults(handler=_cmd_transform)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args, parser)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        log.error("hint: run an earlier stage first (e.g. `taxi-etl ingest`).")
        return 1
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
