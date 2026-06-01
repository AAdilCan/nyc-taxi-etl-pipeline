"""Centralized configuration for the ETL pipeline.

All paths and tunable thresholds live here so the stages stay free of magic
numbers. Values can be overridden with environment variables prefixed with
``TAXI_`` (e.g. ``TAXI_BASE_URL``) or via a local ``.env`` file.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = three levels up from this file (src/taxi_etl/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings, overridable via env vars prefixed with ``TAXI_``."""

    model_config = SettingsConfigDict(
        env_prefix="TAXI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Source ---------------------------------------------------------
    # NYC TLC publishes one parquet file per taxi type per month at this CDN.
    base_url: str = Field(
        default="https://d37ci6vzurychx.cloudfront.net/trip-data",
        description="Base URL for NYC TLC monthly trip-data parquet files.",
    )
    taxi_type: str = Field(
        default="yellow",
        description="TLC service type (yellow, green, fhv, fhvhv).",
    )

    # --- Local paths ----------------------------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    reports_dir: Path = Field(default=PROJECT_ROOT / "reports")

    # --- Ingestion tunables --------------------------------------------
    download_timeout_s: int = Field(default=120, ge=1)
    download_retries: int = Field(default=3, ge=0)
    download_backoff_s: float = Field(default=2.0, ge=0)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def quarantine_dir(self) -> Path:
        return self.data_dir / "quarantine"

    @property
    def manifest_path(self) -> Path:
        """Tracks which (taxi_type, month) partitions have been ingested."""
        return self.data_dir / "manifest.json"

    def ensure_dirs(self) -> None:
        """Create all data/report directories if they do not exist."""
        for path in (
            self.raw_dir,
            self.processed_dir,
            self.quarantine_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
