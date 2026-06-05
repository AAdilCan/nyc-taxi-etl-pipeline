"""Reporting: turn a :class:`~taxi_etl.pipeline.PipelineRun` into summaries.

Two artefacts are produced from a completed run:

* a **run summary** (JSON) — machine-readable record of every month processed,
  per-stage timings, and end-to-end row accounting, written to
  ``reports/run_summary_<timestamp>.json``;
* a **data-quality report** (Markdown) — a human-readable digest showing, per
  month, how many rows were quarantined and why, how many were dropped as
  outliers and why, and the overall retention from raw to feature table,
  written to ``reports/data_quality_<timestamp>.md``.

The same tables are rendered to the console via :func:`format_console_summary`
so a CLI run ends with an at-a-glance result without opening a file.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from tabulate import tabulate

from taxi_etl.config import settings
from taxi_etl.logging_setup import get_logger
from taxi_etl.pipeline import MonthResult, PipelineRun

log = get_logger(__name__)

# How many distinct failure / drop reasons to surface in the narrative report
# before collapsing the rest into an "other" bucket.
_TOP_REASONS = 5


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _top_reasons(counts: dict[str, int], limit: int = _TOP_REASONS) -> list[tuple[str, int]]:
    """Return the *limit* highest-count reasons, descending, ties broken by name."""
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(k, v) for k, v in ranked if v > 0][:limit]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _month_row(m: MonthResult) -> dict[str, object]:
    """Flatten a month result into the columns shown in the overview table."""
    return {
        "month": m.month,
        "raw_rows": m.raw_rows or 0,
        "valid_rows": m.valid_rows if m.valid_rows is not None else "—",
        "feature_rows": m.feature_rows if m.feature_rows is not None else "—",
        "retention_%": m.end_to_end_retention_pct
        if m.end_to_end_retention_pct is not None
        else "—",
        "time_s": round(m.total_seconds, 2),
    }


def build_summary(run: PipelineRun) -> dict[str, object]:
    """Build the JSON-serializable run-summary payload."""
    months_payload = []
    for m in run.months:
        months_payload.append(
            {
                "month": m.month,
                "taxi_type": m.taxi_type,
                "stages_run": m.stages_run,
                "timings_s": m.timings_s,
                "raw_rows": m.raw_rows,
                "valid_rows": m.valid_rows,
                "feature_rows": m.feature_rows,
                "end_to_end_retention_pct": m.end_to_end_retention_pct,
                "ingest": asdict(m.ingest) if m.ingest else None,
                "validation": asdict(m.validation) if m.validation else None,
                "transform": asdict(m.transform) if m.transform else None,
            }
        )

    raw_total = sum(m.raw_rows or 0 for m in run.months)
    feat_total = sum(m.feature_rows or 0 for m in run.months if m.feature_rows is not None)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "months_processed": len(run.months),
        "total_seconds": round(run.total_seconds, 3),
        "totals": {
            "raw_rows": raw_total,
            "feature_rows": feat_total,
            "overall_retention_pct": round(100.0 * feat_total / raw_total, 4)
            if raw_total
            else None,
        },
        "months": months_payload,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _overview_table(run: PipelineRun, fmt: str) -> str:
    rows = [_month_row(m) for m in run.months]
    return tabulate(rows, headers="keys", tablefmt=fmt)


def format_console_summary(run: PipelineRun) -> str:
    """Render a compact, fixed-width overview table for stdout."""
    if not run.months:
        return "No months were processed."
    table = _overview_table(run, "github")
    header = f"Pipeline run summary ({len(run.months)} month(s), {run.total_seconds:.1f}s total)"
    return f"\n{header}\n{table}"


def build_data_quality_md(run: PipelineRun) -> str:
    """Render the per-month data-quality narrative as Markdown."""
    lines: list[str] = []
    lines.append("# Data Quality Report")
    lines.append("")
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(_overview_table(run, "github"))
    lines.append("")

    for m in run.months:
        lines.append(f"## {m.taxi_type or 'yellow'} — {m.month}")
        lines.append("")

        if m.validation is not None:
            v = m.validation
            lines.append(
                f"- **Validation:** {v.valid_rows:,} of {v.total_rows:,} rows valid "
                f"({v.valid_pct:.2f}%); {v.quarantined_rows:,} quarantined "
                f"({v.quarantined_pct:.2f}%)."
            )
            top = _top_reasons(v.failure_counts)
            if top:
                lines.append("  - Top quarantine reasons:")
                for reason, count in top:
                    lines.append(f"    - `{reason}` — {count:,}")

        if m.transform is not None:
            t = m.transform
            lines.append(
                f"- **Transform:** {t.output_rows:,} of {t.input_rows:,} validated rows "
                f"kept ({t.output_pct:.2f}%); {t.dropped_rows:,} dropped as outliers "
                f"across {t.n_partitions} day-partition(s)."
            )
            top = _top_reasons(t.dropped_by_reason)
            if top:
                lines.append("  - Top outlier reasons:")
                for reason, count in top:
                    lines.append(f"    - `{reason}` — {count:,}")

        if m.end_to_end_retention_pct is not None:
            lines.append(
                f"- **End-to-end retention:** {m.feature_rows:,} feature rows "
                f"from {m.raw_rows:,} raw ({m.end_to_end_retention_pct:.2f}%)."
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def write_reports(run: PipelineRun) -> tuple[Path, Path]:
    """Write both the JSON run summary and the Markdown data-quality report.

    Returns:
        ``(summary_json_path, data_quality_md_path)``.
    """
    settings.ensure_dirs()
    ts = _timestamp()

    summary_path = settings.reports_dir / f"run_summary_{ts}.json"
    summary_path.write_text(json.dumps(build_summary(run), indent=2))
    log.info("wrote run summary → %s", summary_path.name)

    dq_path = settings.reports_dir / f"data_quality_{ts}.md"
    dq_path.write_text(build_data_quality_md(run))
    log.info("wrote data-quality report → %s", dq_path.name)

    return summary_path, dq_path
