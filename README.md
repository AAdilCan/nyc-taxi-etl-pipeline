# NYC Taxi ETL Pipeline

An incremental ETL pipeline for the public [NYC TLC trip-record dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
It ingests monthly parquet files, validates them against a strict schema,
transforms them into a clean ML-ready feature table, and writes partitioned
parquet output — with a per-stage CLI and data-quality reporting.

> Work in progress — being built out over the week. See `data/manifest.json`
> for ingested partitions and `reports/` for run summaries.

## Why

The TLC data is a realistic, messy, large public dataset (~3M rows/month). It
has negative fares, impossible trip distances, and unknown rate codes — exactly
the kind of input a production pipeline has to defend against. I'm using it to
build a pipeline that is incremental (never reprocesses a month twice),
validated (bad rows are quarantined, not silently dropped), and reproducible.

## Pipeline stages

```
download → ingest → validate → transform → partitioned parquet
              │          │           │
          raw/      quarantine/   processed/
```

1. **Ingest** — download a month's parquet with retry + local caching; record
   the partition in a manifest so re-runs skip completed months.
2. **Validate** — check every row against a [Pandera](https://pandera.readthedocs.io/)
   schema; rows that fail are written to `data/quarantine/` with the failing
   checks, never dropped silently.
3. **Transform** — coerce types, derive features (trip duration, average speed,
   tip ratio, time-of-day buckets), filter outliers, and write partitioned
   parquet to `data/processed/`.

## Quick start

```bash
pip install -r requirements.txt

# Profile a month of raw data (downloads ~50 MB once, cached in data/raw/)
PYTHONPATH=src python scripts/explore_data.py --month 2024-01
```

Full CLI usage and benchmark results land at the end of the week — see
[DOCUMENTATION.md](DOCUMENTATION.md) (coming) for the deep dive.

## Project layout

```
src/taxi_etl/
  config.py          # pydantic-settings config (paths, thresholds, env overrides)
  logging_setup.py   # shared logger
  stages/            # ingest / validate / transform (built out over the week)
scripts/
  explore_data.py    # one-off data profiler
tests/               # pytest suite
docs/
  raw_schema.md      # NYC TLC yellow-taxi raw column reference
```

## Data

NYC TLC **yellow** taxi monthly parquet, served from the official CloudFront
CDN. No API key required. Raw and processed data are gitignored — the pipeline
downloads what it needs on demand.
