"""Pipeline orchestration: chain ingest → validate → transform for a month.

The individual stages each expose a ``run(month, ...)`` function and are happy
to be called on their own.  This module wires them together so a single call
processes a month end-to-end, times each stage, and collects the per-stage
reports into one :class:`MonthResult`.  Multiple months can be processed with
:func:`run_months`.  Ingestion is incremental by virtue of the manifest, so a
re-run of an already-downloaded month re-reads the cached raw file rather than
hitting the network; validation and transform always re-run on request.

Nothing here writes the aggregate summary; that is the reporting module's job
(:mod:`taxi_etl.reporting`), keeping orchestration and presentation separate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from taxi_etl.logging_setup import get_logger
from taxi_etl.stages import ingest, transform, validate
from taxi_etl.stages.ingest import IngestResult
from taxi_etl.stages.transform import TransformReport
from taxi_etl.stages.validate import ValidationReport

log = get_logger(__name__)

# Canonical stage order. A caller may pass a subset (e.g. skip ingest when the
# raw file is already present) but the relative order is always preserved.
ALL_STAGES: tuple[str, ...] = ("ingest", "validate", "transform")


@dataclass
class IngestSummary:
    """Serializable slice of :class:`IngestResult` (drops the ``Path``)."""

    month: str
    taxi_type: str
    row_count: int
    size_bytes: int
    skipped: bool

    @classmethod
    def from_result(cls, r: IngestResult) -> "IngestSummary":
        return cls(
            month=r.month,
            taxi_type=r.taxi_type,
            row_count=r.row_count,
            size_bytes=r.size_bytes,
            skipped=r.skipped,
        )


@dataclass
class MonthResult:
    """Aggregated outcome of running the pipeline for a single month."""

    month: str
    taxi_type: str
    stages_run: list[str]
    timings_s: dict[str, float]
    ingest: IngestSummary | None = None
    validation: ValidationReport | None = None
    transform: TransformReport | None = None

    # End-to-end row accounting, filled once the relevant stages have run.
    raw_rows: int | None = None
    valid_rows: int | None = None
    feature_rows: int | None = None

    @property
    def total_seconds(self) -> float:
        return sum(self.timings_s.values())

    @property
    def end_to_end_retention_pct(self) -> float | None:
        """Percentage of raw rows that survive to the feature table."""
        if not self.raw_rows or self.feature_rows is None:
            return None
        return round(100.0 * self.feature_rows / self.raw_rows, 4)


@dataclass
class PipelineRun:
    """Outcome of running the pipeline over one or more months."""

    months: list[MonthResult] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return sum(m.total_seconds for m in self.months)


def run_month(
    month: str,
    *,
    force: bool = False,
    stages: tuple[str, ...] = ALL_STAGES,
) -> MonthResult:
    """Run the requested stages for *month* in canonical order.

    Args:
        month:  Target month in ``YYYY-MM`` format.
        force:  Forwarded to the ingest stage (re-download even if manifested).
        stages: Subset of :data:`ALL_STAGES` to execute.  Order is normalised to
            the canonical pipeline order regardless of how it is passed.

    Returns:
        A :class:`MonthResult` with timings and per-stage reports.

    Raises:
        ValueError: if *stages* contains an unknown stage name.
        FileNotFoundError: if a stage's input is missing (e.g. running
            ``validate`` without the raw file present).
    """
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        raise ValueError(f"unknown stage(s): {sorted(unknown)}")
    ordered = [s for s in ALL_STAGES if s in stages]

    result = MonthResult(
        month=month,
        taxi_type="",  # filled by the first stage that runs
        stages_run=ordered,
        timings_s={},
    )

    log.info("=== month %s | stages: %s ===", month, ", ".join(ordered))

    for stage in ordered:
        start = time.perf_counter()

        if stage == "ingest":
            r = ingest.run(month, force=force)
            result.ingest = IngestSummary.from_result(r)
            result.taxi_type = r.taxi_type
            result.raw_rows = r.row_count

        elif stage == "validate":
            v = validate.run(month)
            result.validation = v.report
            result.taxi_type = v.taxi_type
            result.raw_rows = v.total_rows
            result.valid_rows = v.valid_rows

        elif stage == "transform":
            t = transform.run(month)
            result.transform = t.report
            result.taxi_type = t.taxi_type
            result.feature_rows = t.output_rows

        elapsed = time.perf_counter() - start
        result.timings_s[stage] = round(elapsed, 3)
        log.info("stage %-9s done in %.2fs", stage, elapsed)

    if result.end_to_end_retention_pct is not None:
        log.info(
            "month %s complete — %d raw → %d features (%.2f%% retained) in %.2fs",
            month,
            result.raw_rows,
            result.feature_rows,
            result.end_to_end_retention_pct,
            result.total_seconds,
        )

    return result


def run_months(
    months: list[str],
    *,
    force: bool = False,
    stages: tuple[str, ...] = ALL_STAGES,
    continue_on_error: bool = True,
) -> PipelineRun:
    """Run the pipeline for several months.

    Args:
        months:            ``YYYY-MM`` strings, processed in the given order.
        force:             Forwarded to the ingest stage.
        stages:            Subset of stages to run for every month.
        continue_on_error: When ``True`` (default), a failing month is logged
            and skipped so the rest of the batch still runs; when ``False`` the
            exception propagates immediately.

    Returns:
        A :class:`PipelineRun` holding one :class:`MonthResult` per month that
        completed successfully.
    """
    run = PipelineRun()
    for month in months:
        try:
            run.months.append(run_month(month, force=force, stages=stages))
        except Exception as exc:  # noqa: BLE001 — batch resilience is intentional
            if not continue_on_error:
                raise
            log.error("month %s failed: %s — skipping", month, exc)
    return run
