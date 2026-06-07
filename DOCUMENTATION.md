# Documentation — NYC Taxi ETL Pipeline

A deep dive into how this pipeline is put together, the decisions behind it, and
what the numbers actually look like on a real month of data. The
[README](README.md) is the quick pitch; this is the part I'd want an interviewer
to read.

## 1. Overview

The pipeline turns the public NYC TLC yellow-taxi trip records into a clean,
ML-ready feature table. It does three things that a throwaway script usually
skips:

- **Incremental ingestion** — a month is downloaded once and recorded in a
  manifest; re-runs read the cached file instead of hitting the network again.
- **Validation with quarantine** — every row is checked against an explicit
  schema. Rows that fail are written to a quarantine parquet *with the reason
  they failed*, never silently dropped.
- **Auditable transformation** — outlier rows are removed by named business
  rules, and every drop is attributed to a reason and counted in a report.

The end result is a parquet dataset partitioned by pickup day, plus a JSON run
summary and a Markdown data-quality report for every run.

## 2. Architecture

```
                         taxi-etl CLI (cli.py)
                                 │
                                 ▼
                   pipeline.run_months / run_month
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
   stages/ingest.py      stages/validate.py     stages/transform.py
   download + manifest   Pandera + temporal     features + outliers
          │                      │                      │
          ▼                      ▼                      ▼
      data/raw/         data/processed/  +       data/features/
                        data/quarantine/        (partitioned parquet)
                                 │
                                 ▼
                         reporting.py
              run_summary_*.json + data_quality_*.md
```

| Module | Responsibility |
|---|---|
| `src/taxi_etl/config.py` | All paths and tunable thresholds as a `pydantic-settings` `Settings` object; overridable via `TAXI_*` env vars. |
| `src/taxi_etl/logging_setup.py` | One shared logger factory so every module logs consistently. |
| `src/taxi_etl/stages/ingest.py` | Download a month's parquet with retry + backoff, cache it, record the partition in `data/manifest.json`. |
| `src/taxi_etl/stages/validate.py` | Apply the Pandera schema (`RAW_SCHEMA`) in lazy mode + a cross-column temporal check; split rows into valid / quarantined. |
| `src/taxi_etl/stages/transform.py` | Coerce types, engineer features, filter outliers, write partitioned parquet. |
| `src/taxi_etl/pipeline.py` | Chain the stages for a month, time each one, collect per-stage reports into a `MonthResult`; batch over months. |
| `src/taxi_etl/reporting.py` | Turn a `PipelineRun` into a JSON run summary and a Markdown data-quality narrative. |
| `src/taxi_etl/cli.py` | `taxi-etl` entry point: `run` (full pipeline) and per-stage subcommands. |

Stages are intentionally decoupled: each exposes a `run(month, ...)` that reads
its input from disk and writes its output to disk, so any stage can run on its
own. The pipeline module only wires them; it never formats output. Reporting
only reads results; it never runs a stage. This separation is what makes
`taxi-etl validate --month 2024-01` work without re-downloading.

## 3. Data

- **Source:** NYC Taxi & Limousine Commission yellow-taxi monthly trip records,
  served as parquet from the official CloudFront CDN
  (`https://d37ci6vzurychx.cloudfront.net/trip-data`). No API key.
- **Size:** ~2.96M rows / ~50 MB per month (January 2024 = 2,964,624 rows).
- **Schema:** 19 columns — see [`docs/raw_schema.md`](docs/raw_schema.md) for the
  full reference with per-column notes that motivate each validation check.

The data is genuinely messy, which is the point of using it. Profiling January
2024 (via [`scripts/explore_data.py`](scripts/explore_data.py)) surfaced:
`trip_distance` up to **312,722 miles**, `fare_amount` down to **−$899**,
`RatecodeID` with a documented **99 = "unknown"** sentinel, and ~4.7% nulls in
`passenger_count` / `RatecodeID` / `store_and_fwd_flag`. Those observations are
exactly what the validation and transform thresholds defend against.

Raw and processed data are gitignored — the pipeline regenerates everything on
demand, so the repo stays small and reproducible.

## 4. Methodology

### Validation (schema-valid)

`RAW_SCHEMA` in [`validate.py`](src/taxi_etl/stages/validate.py) is a Pandera
`DataFrameSchema` run in **lazy mode**, so *all* failures across all columns are
collected before any row is dropped — you get the complete failure picture in
one pass rather than failing on the first bad cell. Checks encode the TLC data
dictionary: allowed vendor IDs `{1,2,6,7}`, rate codes including the `99`
sentinel, zone IDs in `1–265`, fares within `[−1000, 10000]`, distance
`[0, 500]`. A separate vectorised temporal check flags rows where dropoff
precedes pickup (Pandera wide checks don't express cross-column rules cleanly).
`strict=False` tolerates new columns from future TLC schema revisions instead of
hard-failing.

Failing rows go to `data/quarantine/` with a `_quarantine_reasons` column
listing every rule they broke. **Quarantine over drop** is deliberate: silent
dropping hides data problems; quarantining makes them inspectable and lets me
prove the pipeline didn't lose good rows by accident.

### Transformation (ML-ready)

Schema-valid is not the same as model-ready. A 4-second trip or a 250 mph
average speed passes the schema but is physically implausible.
[`transform.py`](src/taxi_etl/stages/transform.py) therefore:

1. **Coerces/downcasts types** — ids to `Int8`/`Int16`, categoricals to
   `category`. On a ~3M-row month this is a real memory saving.
2. **Engineers features** a model would use — `trip_duration_min`,
   `trip_speed_mph`, `pickup_hour`/`dayofweek`/`day_name`, `is_weekend`,
   `time_of_day` buckets, `is_rush_hour`, and fare ratios (`fare_per_mile`,
   `tip_pct`, `cost_per_min`). Ratios substitute `NaN` on non-positive
   denominators rather than throwing.
3. **Filters outliers** by named rules (`filter_outliers`): duration bounds,
   speed cap, non-positive fare, zero distance, too-few-passengers. A row can
   violate several rules; each is counted, the row is dropped once.

Thresholds live in `config.py` so tuning them never touches stage code.

### Alternatives considered

- **Great Expectations instead of Pandera** — heavier, more ceremony (data docs,
  checkpoints, context). Pandera expresses the same column contracts as plain
  Python with far less setup, which fits a single-repo pipeline.
- **Drop bad rows inline** — rejected for the quarantine approach above.
- **Spark/Dask** — overkill at one month / ~3M rows. Pandas + pyarrow processes
  a month end-to-end in ~6 s on a laptop (see §5). Partitioned parquet output
  keeps the door open for a distributed reader later.

## 5. Results

Full pipeline on **January 2024** (`taxi-etl run --month 2024-01`):

| Stage | Rows in → out | Time |
|---|---|---|
| Ingest | — → 2,964,624 raw | 0.05 s (cached) |
| Validate | 2,964,624 → 2,964,543 valid | 1.95 s |
| Transform | 2,964,543 → 2,831,703 features | 4.23 s |
| **End-to-end** | **2,964,624 raw → 2,831,703 features** | **6.23 s** |

**Overall retention: 95.52%.**

Validation quarantined **81 rows** (0.003%): 56 with dropoff-before-pickup, 25
with `trip_distance > 500`. Transform dropped **132,840 rows** (4.48%) as
outliers, broken down by reason:

| Reason | Rows dropped |
|---|---|
| `zero_distance` | 60,371 |
| `non_positive_fare` | 38,341 |
| `too_few_passengers` | 31,465 |
| `duration_too_short` | 27,565 |
| `duration_too_long` | 1,983 |
| `speed_implausible` | 999 |

(Reasons sum to more than 132,840 because a row can fail several rules; it is
still removed only once.) These numbers come straight from
`reports/validation_yellow_2024-01.json` and
`reports/transform_yellow_2024-01.json`; the same digest is rendered to
`reports/data_quality_*.md` on every run.

## 6. Tradeoffs & Decisions

- **Quarantine vs. silent drop.** Quarantine costs an extra parquet write and a
  reason-tagging pass, but it makes data loss auditable. Worth it.
- **Lazy Pandera validation.** Collecting all failure cases for ~3M rows uses
  more memory than fail-fast, but a pipeline that reports *every* problem in one
  run is far more useful than one that stops at the first. The validate stage is
  still the cheaper half of the run (1.95 s vs 4.23 s for transform).
- **Partition by pickup day.** Lets a consumer read one day without scanning the
  month. The honest wrinkle: a handful of rows carry pickup timestamps that fall
  just outside the target month (data-entry artifacts), so January produces **35**
  day-partitions rather than 31. I left those rows in rather than clip the
  partition set, because they're schema-valid and the extra partitions are tiny —
  but it's a real edge a downstream reader should know about.
- **Reasons can double-count.** `dropped_by_reason` sums above the actual drop
  count by design — I'd rather see that 60k rows have zero distance *and* that
  38k have non-positive fare than collapse them into one number and lose the
  diagnostic signal.
- **Config in one module.** Every threshold and path is in `config.py`. The cost
  is one more import everywhere; the benefit is no magic numbers in stage logic
  and env-var overrides for free via `pydantic-settings`.

### Limitations

- Yellow taxi only; green/fhv would need schema variants (the config has a
  `taxi_type` knob, but `RAW_SCHEMA` is yellow-specific).
- Single-machine pandas; a multi-year backfill would want chunking or Dask.
- No automated freshness/scheduling — `run --start/--end` is manual.

## 7. How to Run

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                      # installs the `taxi-etl` command

# full pipeline, one month (downloads ~50 MB once, cached in data/raw/)
taxi-etl run --month 2024-01

# a range, inclusive on both ends
taxi-etl run --start 2024-01 --end 2024-03

# re-run a single stage on the cached file
taxi-etl ingest    --month 2024-01    # download only
taxi-etl validate  --month 2024-01    # validate cached raw file
taxi-etl transform --month 2024-01    # build the feature dataset

# useful flags on `run`
taxi-etl run --month 2024-01 --force        # ignore manifest, re-download
taxi-etl run --start 2024-01 --end 2024-03 --fail-fast   # stop on first error
taxi-etl run --month 2024-01 --no-report    # skip writing JSON/MD reports

# tests
pytest                                # 90 tests
pytest --cov                          # with coverage (needs pytest-cov)
```

Without installing, swap `taxi-etl` for `python -m taxi_etl`.

## 8. How to Extend

- **Add a taxi type.** Add a `GREEN_SCHEMA` in `validate.py` and select it off
  `settings.taxi_type`; the green files use `lpep_*` datetime columns, so the
  temporal check and feature engineering need the column names parameterised.
- **New feature.** Add it in `transform.add_features`; it flows into the output
  and `feature_columns` automatically. Add a case to `tests/test_transform.py`.
- **New validation rule.** Extend `RAW_SCHEMA` (column check) or add a vectorised
  function alongside `_temporal_failures` and include it in `_merge_failures`.
- **Different storage.** `_write_partitioned` is the only place that touches the
  output layout — point it at a different sink (e.g. S3 via `s3fs`) there.
- **Scheduling.** Wrap `taxi-etl run --start <last> --end <this>` in cron; the
  manifest already makes re-runs cheap.

## 9. References

- [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
  and the [yellow-taxi data dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf).
- [Pandera](https://pandera.readthedocs.io/) — dataframe schema validation.
- [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) — typed config from env.
- [Apache Parquet](https://parquet.apache.org/) / [pyarrow](https://arrow.apache.org/docs/python/) — columnar storage and partitioning.
