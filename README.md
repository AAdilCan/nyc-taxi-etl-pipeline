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
pip install -e .            # installs the `taxi-etl` command

# Run the whole pipeline for one month (downloads ~50 MB once, cached in data/raw/)
taxi-etl run --month 2024-01

# Or a range, inclusive on both ends
taxi-etl run --start 2024-01 --end 2024-03
```

`run` chains ingest → validate → transform, then writes a JSON run summary and
a Markdown data-quality report to `reports/` and prints an overview table:

```
Pipeline run summary (1 month(s), 6.2s total)
| month   |   raw_rows |   valid_rows |   feature_rows |   retention_% |   time_s |
|---------|------------|--------------|----------------|---------------|----------|
| 2024-01 |    2964624 |      2964543 |        2831703 |       95.5164 |     6.23 |
```

Each stage is also runnable on its own, handy for re-running just the part that
changed:

```bash
taxi-etl ingest    --month 2024-01     # download only
taxi-etl validate  --month 2024-01     # validate the cached raw file
taxi-etl transform --month 2024-01     # build the feature dataset
```

Without installing, swap `taxi-etl` for `python -m taxi_etl`. Benchmark results
and the deep dive land at the end of the week in
[DOCUMENTATION.md](DOCUMENTATION.md) (coming).

## Project layout

```
src/taxi_etl/
  cli.py             # `taxi-etl` entry point: run / ingest / validate / transform
  pipeline.py        # orchestrates the stages and times them per month
  reporting.py       # run summary (JSON) + data-quality report (Markdown)
  config.py          # pydantic-settings config (paths, thresholds, env overrides)
  logging_setup.py   # shared logger
  stages/            # ingest / validate / transform
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
