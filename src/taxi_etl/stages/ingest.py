"""Ingestion stage: download NYC TLC monthly parquet files with retry + caching.

Each call to ``run()`` downloads one (taxi_type, month) file from the TLC CDN
into ``data/raw/``.  Already-ingested months are tracked in
``data/manifest.json`` and skipped automatically, making the stage idempotent
and safe to rerun after failures.

Download failures are retried up to ``settings.download_retries`` times using
exponential backoff.  Partial downloads are cleaned up before retrying.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from taxi_etl.config import settings
from taxi_etl.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class ManifestEntry:
    month: str
    taxi_type: str
    file: str
    size_bytes: int
    row_count: int
    columns: list[str]
    ingested_at: str


@dataclass
class Manifest:
    entries: dict[str, ManifestEntry] = field(default_factory=dict)

    @staticmethod
    def _key(taxi_type: str, month: str) -> str:
        return f"{taxi_type}/{month}"

    @classmethod
    def load(cls) -> "Manifest":
        path = settings.manifest_path
        if not path.exists():
            return cls()
        raw: dict[str, Any] = json.loads(path.read_text())
        entries = {k: ManifestEntry(**v) for k, v in raw.get("entries", {}).items()}
        return cls(entries=entries)

    def save(self) -> None:
        settings.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": {k: asdict(v) for k, v in self.entries.items()}}
        settings.manifest_path.write_text(json.dumps(payload, indent=2))

    def contains(self, taxi_type: str, month: str) -> bool:
        return self._key(taxi_type, month) in self.entries

    def record(self, entry: ManifestEntry) -> None:
        self.entries[self._key(entry.taxi_type, entry.month)] = entry
        self.save()


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _file_name(taxi_type: str, month: str) -> str:
    return f"{taxi_type}_tripdata_{month}.parquet"


def _download_with_retry(url: str, dest: Path) -> None:
    """Stream-download *url* to *dest* with retry + exponential backoff.

    On each failed attempt, any partial file is removed before sleeping.
    Raises ``requests.HTTPError`` or ``requests.ConnectionError`` if all
    retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1, settings.download_retries + 2):  # +1 for the first try
        try:
            log.info("download attempt %d/%d: %s", attempt, settings.download_retries + 1, url)
            with requests.get(url, stream=True, timeout=settings.download_timeout_s) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                with dest.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = 100 * downloaded / total
                            log.debug("  %.0f%% (%d MB / %d MB)", pct, downloaded >> 20, total >> 20)
            log.info("saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
            return
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if dest.exists():
                dest.unlink()
            remaining = settings.download_retries + 1 - attempt
            if remaining <= 0:
                break
            wait = settings.download_backoff_s * (2 ** (attempt - 1))
            log.warning("download failed (%s), retrying in %.1fs (%d left)", exc, wait, remaining)
            time.sleep(wait)

    raise RuntimeError(f"all download attempts failed for {url}") from last_exc


# ---------------------------------------------------------------------------
# Schema inference
# ---------------------------------------------------------------------------

def _infer_schema(df: pd.DataFrame) -> list[str]:
    """Return column names as a simple schema snapshot."""
    return list(df.columns)


def _read_parquet_metadata(path: Path) -> tuple[int, list[str]]:
    """Return (row_count, columns) without loading the full frame into memory."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    meta = pf.metadata
    row_count = meta.num_rows
    schema = pf.schema_arrow
    columns = [schema.field(i).name for i in range(len(schema))]
    return row_count, columns


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    month: str
    taxi_type: str
    dest_path: Path
    row_count: int
    columns: list[str]
    size_bytes: int
    skipped: bool  # True when already in manifest


def run(month: str, *, force: bool = False) -> IngestResult:
    """Ingest one month of NYC TLC trip data.

    Downloads the parquet from the TLC CDN into ``data/raw/`` and records the
    entry in the manifest.  If the month is already present in the manifest,
    the download is skipped unless *force* is ``True``.

    Args:
        month:  Target month in ``YYYY-MM`` format (e.g. ``"2024-01"``).
        force:  Re-download and overwrite even if already manifested.

    Returns:
        ``IngestResult`` describing what was done.
    """
    import datetime

    settings.ensure_dirs()
    taxi_type = settings.taxi_type
    manifest = Manifest.load()

    dest = settings.raw_dir / _file_name(taxi_type, month)

    if not force and manifest.contains(taxi_type, month) and dest.exists():
        log.info("skipping %s/%s — already in manifest", taxi_type, month)
        row_count, columns = _read_parquet_metadata(dest)
        return IngestResult(
            month=month,
            taxi_type=taxi_type,
            dest_path=dest,
            row_count=row_count,
            columns=columns,
            size_bytes=dest.stat().st_size,
            skipped=True,
        )

    url = f"{settings.base_url}/{_file_name(taxi_type, month)}"

    if not dest.exists() or force:
        _download_with_retry(url, dest)

    row_count, columns = _read_parquet_metadata(dest)
    log.info("ingested %s — %d rows, %d columns", dest.name, row_count, len(columns))

    entry = ManifestEntry(
        month=month,
        taxi_type=taxi_type,
        file=str(dest.relative_to(settings.data_dir)),
        size_bytes=dest.stat().st_size,
        row_count=row_count,
        columns=columns,
        ingested_at=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    manifest.record(entry)

    return IngestResult(
        month=month,
        taxi_type=taxi_type,
        dest_path=dest,
        row_count=row_count,
        columns=columns,
        size_bytes=dest.stat().st_size,
        skipped=False,
    )


def run_range(months: list[str], *, force: bool = False) -> list[IngestResult]:
    """Ingest multiple months, skipping already-manifested ones.

    Args:
        months: List of ``YYYY-MM`` strings in any order.
        force:  Passed through to :func:`run`.

    Returns:
        List of ``IngestResult``, one per month, in the same order as *months*.
    """
    results = []
    for month in months:
        result = run(month, force=force)
        results.append(result)
    return results
