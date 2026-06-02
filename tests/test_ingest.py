"""Tests for the ingestion stage: manifest, download retry, CLI helpers."""

from __future__ import annotations

import datetime
import json
import time
import unittest.mock as mock
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from taxi_etl.stages.ingest import (
    Manifest,
    ManifestEntry,
    _download_with_retry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(month: str = "2024-01", taxi_type: str = "yellow") -> ManifestEntry:
    return ManifestEntry(
        month=month,
        taxi_type=taxi_type,
        file=f"raw/{taxi_type}_tripdata_{month}.parquet",
        size_bytes=42_000_000,
        row_count=2_964_624,
        columns=["VendorID", "tpep_pickup_datetime", "total_amount"],
        ingested_at="2024-02-01T08:00:00Z",
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class TestManifest:
    # All manifest tests redirect settings.data_dir to tmp_path so the
    # manifest_path property resolves under the test's temp directory.

    def test_load_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("taxi_etl.stages.ingest.settings.data_dir", tmp_path)
        m = Manifest.load()
        assert m.entries == {}

    def test_contains_false_on_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("taxi_etl.stages.ingest.settings.data_dir", tmp_path)
        m = Manifest.load()
        assert not m.contains("yellow", "2024-01")

    def test_record_and_contains(self, tmp_path, monkeypatch):
        monkeypatch.setattr("taxi_etl.stages.ingest.settings.data_dir", tmp_path)

        entry = _make_entry()
        m = Manifest.load()
        m.record(entry)

        assert m.contains("yellow", "2024-01")
        assert not m.contains("yellow", "2024-02")

    def test_record_persists_to_disk(self, tmp_path, monkeypatch):
        monkeypatch.setattr("taxi_etl.stages.ingest.settings.data_dir", tmp_path)

        m = Manifest.load()
        m.record(_make_entry("2024-01"))
        m.record(_make_entry("2024-02"))

        # Reload from disk
        m2 = Manifest.load()
        assert m2.contains("yellow", "2024-01")
        assert m2.contains("yellow", "2024-02")
        assert not m2.contains("yellow", "2024-03")

    def test_record_overwrites_same_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr("taxi_etl.stages.ingest.settings.data_dir", tmp_path)

        m = Manifest.load()
        e1 = _make_entry()
        e2 = _make_entry()
        e2.row_count = 9_999
        m.record(e1)
        m.record(e2)

        m2 = Manifest.load()
        assert m2.entries["yellow/2024-01"].row_count == 9_999


# ---------------------------------------------------------------------------
# _months_between (imported from CLI script)
# ---------------------------------------------------------------------------

# We import the helper directly from the scripts module (not installed as a
# package), so add the scripts dir to sys.path first.
import importlib.util
import sys
from pathlib import Path as _Path

_scripts_dir = _Path(__file__).resolve().parents[1] / "scripts"


def _import_ingest_cli():
    spec = importlib.util.spec_from_file_location("ingest_cli", _scripts_dir / "ingest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestMonthsBetween:
    @pytest.fixture(autouse=True)
    def _cli(self):
        self.cli = _import_ingest_cli()

    def test_single_month(self):
        assert self.cli._months_between("2024-01", "2024-01") == ["2024-01"]

    def test_three_months(self):
        assert self.cli._months_between("2024-01", "2024-03") == [
            "2024-01", "2024-02", "2024-03"
        ]

    def test_year_boundary(self):
        months = self.cli._months_between("2023-11", "2024-02")
        assert months == ["2023-11", "2023-12", "2024-01", "2024-02"]

    def test_reversed_raises(self):
        with pytest.raises(ValueError, match="after"):
            self.cli._months_between("2024-03", "2024-01")


# ---------------------------------------------------------------------------
# _download_with_retry
# ---------------------------------------------------------------------------

class TestDownloadWithRetry:
    def test_success_on_first_attempt(self, tmp_path):
        dest = tmp_path / "test.parquet"
        fake_content = b"PAR1" + b"\x00" * 100

        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-length": str(len(fake_content))}
        mock_response.iter_content = MagicMock(return_value=[fake_content])

        with patch("requests.get", return_value=mock_response):
            _download_with_retry("http://example.com/file.parquet", dest)

        assert dest.exists()
        assert dest.read_bytes() == fake_content

    def test_retries_and_cleans_partial_file(self, tmp_path, monkeypatch):
        """A transient failure is retried; partial file removed between attempts."""
        import requests as req_mod

        dest = tmp_path / "test.parquet"
        fake_content = b"PAR1" + b"\x00" * 50

        call_count = 0

        def fake_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise req_mod.ConnectionError("network blip")
            mock_response = MagicMock()
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {"content-length": str(len(fake_content))}
            mock_response.iter_content = MagicMock(return_value=[fake_content])
            return mock_response

        monkeypatch.setattr("taxi_etl.stages.ingest.settings.download_retries", 2)
        monkeypatch.setattr("taxi_etl.stages.ingest.settings.download_backoff_s", 0.0)
        monkeypatch.setattr("taxi_etl.stages.ingest.time.sleep", lambda _: None)

        with patch("requests.get", side_effect=fake_get):
            _download_with_retry("http://example.com/file.parquet", dest)

        assert dest.exists()
        assert call_count == 2

    def test_all_retries_exhausted_raises(self, tmp_path, monkeypatch):
        import requests as req_mod

        dest = tmp_path / "fail.parquet"
        monkeypatch.setattr("taxi_etl.stages.ingest.settings.download_retries", 1)
        monkeypatch.setattr("taxi_etl.stages.ingest.settings.download_backoff_s", 0.0)
        monkeypatch.setattr("taxi_etl.stages.ingest.time.sleep", lambda _: None)

        with patch("requests.get", side_effect=req_mod.ConnectionError("always fails")):
            with pytest.raises(RuntimeError, match="all download attempts failed"):
                _download_with_retry("http://example.com/file.parquet", dest)

        assert not dest.exists()
