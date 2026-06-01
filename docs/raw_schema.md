# Raw schema — NYC TLC yellow taxi

Columns as published in the monthly `yellow_tripdata_YYYY-MM.parquet` files.
Profiled from `2024-01` (2,964,624 rows). Stats below motivate the validation
checks and outlier filters built in later stages.

| column | dtype | notes |
|---|---|---|
| `VendorID` | int32 | 1, 2, 6, 7 — provider code |
| `tpep_pickup_datetime` | datetime64[us] | trip start |
| `tpep_dropoff_datetime` | datetime64[us] | trip end; must be ≥ pickup |
| `passenger_count` | float64 | ~4.7% null; observed 0–9 |
| `trip_distance` | float64 | miles; **max 312,722** → needs outlier cap |
| `RatecodeID` | float64 | 1–6 valid; **99 = unknown**; ~4.7% null |
| `store_and_fwd_flag` | object | 'Y'/'N'; ~4.7% null |
| `PULocationID` | int32 | TLC zone 1–265 |
| `DOLocationID` | int32 | TLC zone 1–265 |
| `payment_type` | int64 | 0–6; 1=card, 2=cash |
| `fare_amount` | float64 | **min −899** (refunds/errors) |
| `extra` | float64 | misc surcharges |
| `mta_tax` | float64 | usually 0.5 |
| `tip_amount` | float64 | **min −80** |
| `tolls_amount` | float64 | **min −80** |
| `improvement_surcharge` | float64 | −1 to 1 |
| `total_amount` | float64 | **min −900, max 5000** |
| `congestion_surcharge` | float64 | ~4.7% null |
| `Airport_fee` | float64 | ~4.7% null |

## Known data-quality issues

- **Negative money columns** — refunds and data-entry errors appear as negative
  `fare_amount`, `total_amount`, `tip_amount`, `tolls_amount`.
- **Impossible trip distances** — values up to 312,722 miles; the 50th
  percentile is 1.68 miles.
- **Unknown rate codes** — `RatecodeID = 99` is used for unclassified trips.
- **~4.7% nulls** concentrated in `passenger_count`, `RatecodeID`,
  `store_and_fwd_flag`, `congestion_surcharge`, `Airport_fee` — these rows tend
  to come from a single vendor's feed.

The validate stage encodes these as explicit checks; the transform stage caps
or filters outliers and derives clean features.
