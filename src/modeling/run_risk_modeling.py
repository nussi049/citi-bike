#!/usr/bin/env python
"""
run_risk_modeling.py - NYC Bike Crash Risk Modeling Pipeline

This script fits a Poisson GLM to predict bike crashes
using CitiBike exposure as a feature. It implements:

1. TEMPORAL SEPARATION: Train on 2021-2024 (excludes COVID 2020), test on 2025
2. GRID-LEVEL MODELING: Spatial cells × daily resolution
3. UNCERTAINTY QUANTIFICATION: Monte Carlo with parameter + weather bootstrap
4. EXPOSURE SCENARIOS: Sensitivity analysis with historical exposure patterns

INPUTS:
    - data/interim/tripdata_2013_2025_clean.parquet  (from clean_data.py)
    - data/interim/crashes_bike_clean.parquet        (from clean_data.py)
    - data/raw/weather_hourly_openmeteo/**/*.parquet

OUTPUTS (used by dashboard):
    - crash_cell_hour.parquet              Crashes binned to grid cells
    - exposure_cell_hour_2020_2024.parquet Training exposure
    - exposure_cell_hour_2025.parquet      Test exposure
    - grid_train_cell_hour_2020_2024.parquet Training mart with features
    - grid_cells_keep_2020_2024.parquet    Cells included in model
    - grid_2025_cell_hour_bike_all.parquet 2025 test mart
    - model_meta_bike_all.json             Normalization stats, formula
    - model_comparison_bike_all.parquet    AIC, dispersion, quantiles
    - risk_eval_2025_monthly_bike_all.parquet Monthly observed vs predicted
    - risk_mc_2025_totals_bike_all_scenarios.parquet Monte Carlo results
    - risk_exposure_scenarios_summary.parquet Scenario aggregates

MODEL SPECIFICATION:
    y_bike ~ C(hour) + C(dow) + C(month)
           + grid_lat_norm + grid_lng_norm + grid_lat_norm² + grid_lng_norm² + grid_lat_norm×grid_lng_norm
           + temp + prcp + snow + wspd
           + trend
           + log1p_exposure

    NOTE: Exposure is modeled as a FEATURE (not offset), allowing:
    - Training on ALL crashes (including hours without CitiBike exposure)
    - Estimation of exposure coefficient β (not fixed at 1)
    - Testing whether exposure is a significant predictor

    log1p_exposure = log(exposure_min + 1)  # handles exposure=0
    trend = (hour_ts - TMIN).days / 365.25 (normalized to years since start of training)

    SPATIAL BINNING (Grid System):
        Raw crash/trip coordinates are binned into 0.025° × 0.025° grid cells
        (~2.5 km × 2.5 km). This is intentional - we model risk at the cell level,
        not at exact point locations.

        Example:
            Crash at (40.7589, -73.9851) → Cell (40.75, -74.00)
            Crash at (40.7612, -73.9823) → Cell (40.75, -74.00)  # same cell!

        Multiple crashes in the same cell/hour are aggregated (y_bike = count).

    COORDINATE NORMALIZATION:
        Grid cell coordinates are z-score normalized using training statistics:
            grid_lat_norm = (grid_lat - lat_mean) / lat_std
            grid_lng_norm = (grid_lng - lng_mean) / lng_std

        This prevents numerical issues in GLM fitting due to small variance
        across the active grid cells.

    Family: Poisson (dispersion ~1, no overdispersion)
    Training: 2021-2024 (4 years, excludes COVID 2020)
    Testing: 2025 (true out-of-sample)

Usage:
    python src/modeling/run_risk_modeling.py
    # or via Makefile:
    make modeling
"""

import os
import json
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# -----------------------------
# INPUTS
# -----------------------------
TRIPS = PROJECT_ROOT / "data" / "interim" / "tripdata_2013_2025_clean.parquet"
CRASH_BIKE = PROJECT_ROOT / "data" / "interim" / "crashes_bike_clean.parquet"
WEATHER_HOURLY_DIR = PROJECT_ROOT / "data" / "raw" / "weather_hourly_openmeteo"

# -----------------------------
# OUTPUT DIR
# -----------------------------
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "risk_hourly_mc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TMP_DIR = PROJECT_ROOT / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# GRID + WINDOW
# -----------------------------
GRID_DEG = 0.025
TMIN = "2021-01-01"  # Excludes COVID year 2020 (had highest summer crash rate)
TMAX_TRAIN = "2025-01-01"  # Training: 2020-2024 only
TMAX = "2026-01-01"        # For exposure data collection

DB_PATH = OUT_DIR / f"analysis_bike_all_grid{int(GRID_DEG*10000):04d}.duckdb"
con = duckdb.connect(DB_PATH.as_posix())

# Make DuckDB stable on laptop RAM
con.execute("PRAGMA threads=4;")
con.execute("PRAGMA preserve_insertion_order=false;")
con.execute("PRAGMA memory_limit='8GB';")
con.execute(f"PRAGMA temp_directory='{TMP_DIR.as_posix()}';")

# Forcing rebuild prevents "same fit" due to cached files.
FORCE_REBUILD = True

# Cell selection strategy: keep cells covering X% of total CRASHES (not exposure!)
# Exposure-based selection was problematic: 80% exposure = only 37% crashes
# Crash-based selection ensures we cover most crashes:
#   80% crash coverage = ~50 cells (manageable for GLM)
#   90% crash coverage = ~77 cells (may cause OOM)
CRASH_COVERAGE = 1.00   # Target ALL cells with CitiBike exposure (~140 cells, ~94% crashes)

print("DuckDB:", DB_PATH)
print("GRID_DEG:", GRID_DEG, "| FORCE_REBUILD:", FORCE_REBUILD, "| CRASH_COVERAGE:", CRASH_COVERAGE)
print("Training:", TMIN, "→", TMAX_TRAIN, "(4 years, excludes COVID 2020)")
print("Testing: 2025-01-01 → 2026-01-01")
print("Exposure collection:", TMIN, "→", TMAX)


# In[3]:


crash_cell_hour_path = OUT_DIR / "crash_cell_hour.parquet"

if FORCE_REBUILD and crash_cell_hour_path.exists():
    crash_cell_hour_path.unlink()
    print("Deleted:", crash_cell_hour_path)

if crash_cell_hour_path.exists():
    print("Exists, skipping:", crash_cell_hour_path)
else:
    con.execute(f"""
    COPY (
      WITH base AS (
        SELECT
          -- Preferred: DATE + TIME (works if crash_date is DATE/TIMESTAMP and crash_time is TIME-like)
          (try_cast(crash_date AS DATE) + try_cast(crash_time AS TIME))::TIMESTAMP AS crash_ts_dt,

          -- Fallback: parse crash_time as "H:MM" / "HH:MM" strings
          try_strptime(
            strftime(try_cast(crash_date AS DATE), '%Y-%m-%d') || ' ' ||
            printf(
              '%02d:%02d',
              try_cast(regexp_extract(CAST(crash_time AS VARCHAR), '^(\\d{{1,2}})', 1) AS INTEGER),
              try_cast(regexp_extract(CAST(crash_time AS VARCHAR), ':(\\d{{2}})', 1) AS INTEGER)
            ),
            '%Y-%m-%d %H:%M'
          ) AS crash_ts_str,

          latitude,
          longitude
        FROM read_parquet('{CRASH_BIKE.as_posix()}')
      ),
      ts AS (
        SELECT
          COALESCE(crash_ts_dt, crash_ts_str) AS crash_ts,
          latitude,
          longitude
        FROM base
      ),
      binned AS (
        SELECT
          date_trunc('hour', crash_ts) AS hour_ts,
          floor(latitude  / {GRID_DEG}) * {GRID_DEG} AS grid_lat,
          floor(longitude / {GRID_DEG}) * {GRID_DEG} AS grid_lng,
          1 AS y_bike
        FROM ts
        WHERE crash_ts IS NOT NULL
          AND latitude IS NOT NULL AND longitude IS NOT NULL
          AND crash_ts >= TIMESTAMP '{TMIN}'
          AND crash_ts <  TIMESTAMP '{TMAX}'  -- Full range for initial collection
      )
      SELECT
        hour_ts,
        CAST(grid_lat AS DOUBLE) AS grid_lat,
        CAST(grid_lng AS DOUBLE) AS grid_lng,
        CAST(grid_lat AS VARCHAR) || '_' || CAST(grid_lng AS VARCHAR) AS cell_id,
        SUM(y_bike) AS y_bike
      FROM binned
      GROUP BY 1,2,3,4
    )
    TO '{crash_cell_hour_path.as_posix()}'
    (FORMAT PARQUET);
    """)

    print("Wrote:", crash_cell_hour_path)

# sanity
crash_check = con.execute(f"""
SELECT COUNT(*) n, MIN(hour_ts) min_ts, MAX(hour_ts) max_ts
FROM read_parquet('{crash_cell_hour_path.as_posix()}');
""").fetch_df()
print(crash_check)

# Check if we have any crashes
if crash_check['n'].iloc[0] == 0:
    print("\n" + "="*70)
    print("⚠️  CRITICAL ERROR: NO BIKE CRASHES FOUND!")
    print("="*70)
    print("\nPossible causes:")
    print("  1. crash_time field is NULL or incorrectly formatted in crashes.parquet")
    print("  2. crash_date + crash_time parsing is failing")
    print("  3. No crashes fall within the time window:", TMIN, "to", TMAX)
    print("\nDebugging steps:")
    print("  1. Check raw crash data:")
    print(f"     duckdb -c \"SELECT crash_date, crash_time FROM read_parquet('{CRASH_BIKE.as_posix()}') LIMIT 10\"")
    print("  2. Delete crash_cell_hour.parquet and rerun with FORCE_REBUILD=True")
    print("  3. Check if crashes.parquet needs to be re-downloaded")
    print("="*70)
    raise RuntimeError("No bike crashes found - cannot build risk model")


# In[ ]:


# Separate exposure paths for train and test to prevent leakage
exposure_train_path = OUT_DIR / "exposure_cell_hour_2020_2024.parquet"
exposure_test_path = OUT_DIR / "exposure_cell_hour_2025.parquet"

if FORCE_REBUILD:
    if exposure_train_path.exists():
        exposure_train_path.unlink()
        print("Deleted:", exposure_train_path)
    if exposure_test_path.exists():
        exposure_test_path.unlink()
        print("Deleted:", exposure_test_path)

# ============================================================================
# LINE INTERPOLATION FOR EXPOSURE
# ============================================================================
# Instead of assigning all exposure to the start station cell, we distribute
# it across all cells along the straight line between start and end station.
# This captures mid-route exposure and expands coverage to more cells.
#
# Approach:
#   1. Interpolate N points along the line from start to end
#   2. Bin each point to a grid cell
#   3. Get unique cells traversed
#   4. Distribute trip duration equally across all cells
# ============================================================================
N_INTERP_POINTS = 5  # Number of interpolation points (including start/end)

# Build training exposure (2020-2024 only)
if exposure_train_path.exists():
    print("Exists, skipping:", exposure_train_path)
else:
    # Process quarter-by-quarter to manage memory (interpolation uses more RAM)
    print(f"Creating TRAINING exposure with LINE INTERPOLATION for {TMIN} to {TMAX_TRAIN}...")
    print(f"Using {N_INTERP_POINTS} interpolation points per trip")

    # Extract year range for TRAINING data only
    import datetime
    year_min = datetime.datetime.fromisoformat(TMIN).year
    year_max = datetime.datetime.fromisoformat(TMAX_TRAIN).year - 1  # 2024

    temp_files = []

    for year in range(year_min, year_max + 1):
        for quarter in range(1, 5):
            q_start_month = (quarter - 1) * 3 + 1
            q_end_month = quarter * 3 + 1

            if quarter == 4:
                if year == year_max:
                    q_start = f"{year}-10-01"
                    q_end = TMAX_TRAIN
                else:
                    q_start = f"{year}-10-01"
                    q_end = f"{year+1}-01-01"
            else:
                q_start = f"{year}-{q_start_month:02d}-01"
                q_end = f"{year}-{q_end_month:02d}-01"

            # Skip quarters before TMIN
            if q_end <= TMIN:
                continue
            q_start = max(q_start, TMIN)

            temp_path = OUT_DIR / f"_temp_exp_{year}_Q{quarter}.parquet"
            temp_files.append(temp_path)

            if temp_path.exists():
                print(f"  {year} Q{quarter}: already exists, skipping")
                continue

            print(f"  Processing {year} Q{quarter} ({q_start} to {q_end})...")

            # Line interpolation: distribute exposure across all cells along the route
            con.execute(f"""
            COPY (
              WITH trips AS (
                SELECT
                  try_cast(started_at AS TIMESTAMP) AS started_at,
                  try_cast(ended_at   AS TIMESTAMP) AS ended_at,
                  start_lat, start_lng,
                  end_lat, end_lng,
                  duration_sec
                FROM read_parquet('{TRIPS.as_posix()}')
                WHERE try_cast(started_at AS TIMESTAMP) >= TIMESTAMP '{q_start}'
                  AND try_cast(started_at AS TIMESTAMP) <  TIMESTAMP '{q_end}'
                  AND start_lat IS NOT NULL AND start_lng IS NOT NULL
                  AND end_lat IS NOT NULL AND end_lng IS NOT NULL
                  AND ended_at IS NOT NULL
                  AND duration_sec IS NOT NULL
                  AND duration_sec > 0
                  AND duration_sec < 4*60*60
              ),
              -- Generate interpolation points along the line
              interp_points AS (
                SELECT
                  started_at, ended_at, duration_sec,
                  -- Interpolate: start + t * (end - start) for t in [0, 1]
                  start_lat + (gs.t / {N_INTERP_POINTS - 1}.0) * (end_lat - start_lat) AS interp_lat,
                  start_lng + (gs.t / {N_INTERP_POINTS - 1}.0) * (end_lng - start_lng) AS interp_lng
                FROM trips
                CROSS JOIN generate_series(0, {N_INTERP_POINTS - 1}) AS gs(t)
              ),
              -- Bin each interpolated point to grid cell
              binned_points AS (
                SELECT
                  started_at, ended_at, duration_sec,
                  floor(interp_lat / {GRID_DEG}) * {GRID_DEG} AS grid_lat,
                  floor(interp_lng / {GRID_DEG}) * {GRID_DEG} AS grid_lng
                FROM interp_points
              ),
              -- Get unique cells per trip and count them
              unique_cells_per_trip AS (
                SELECT
                  started_at, ended_at, duration_sec,
                  grid_lat, grid_lng,
                  COUNT(*) OVER (PARTITION BY started_at, ended_at, duration_sec) AS n_cells
                FROM (
                  SELECT DISTINCT started_at, ended_at, duration_sec, grid_lat, grid_lng
                  FROM binned_points
                ) sub
              ),
              -- Distribute duration equally across cells
              exposure_per_cell AS (
                SELECT
                  started_at, ended_at,
                  grid_lat, grid_lng,
                  CAST(floor(grid_lat / {GRID_DEG}) * {GRID_DEG} AS VARCHAR) || '_' ||
                  CAST(floor(grid_lng / {GRID_DEG}) * {GRID_DEG} AS VARCHAR) AS cell_id,
                  duration_sec / n_cells AS cell_duration_sec
                FROM unique_cells_per_trip
              ),
              -- Expand to hourly buckets
              expanded AS (
                SELECT
                  cell_id, grid_lat, grid_lng,
                  gs AS hour_ts,
                  started_at, ended_at, cell_duration_sec
                FROM exposure_per_cell
                CROSS JOIN generate_series(
                  date_trunc('hour', started_at),
                  date_trunc('hour', ended_at),
                  INTERVAL '1 hour'
                ) AS t(gs)
              ),
              -- Calculate overlap with each hour
              overlap AS (
                SELECT
                  cell_id, grid_lat, grid_lng, hour_ts,
                  GREATEST(
                    0,
                    (cell_duration_sec / NULLIF(EXTRACT(EPOCH FROM ended_at - started_at), 0)) *
                    EXTRACT(EPOCH FROM LEAST(ended_at, hour_ts + INTERVAL '1 hour')
                                  - GREATEST(started_at, hour_ts))
                  ) AS overlap_sec
                FROM expanded
              )
              SELECT
                hour_ts,
                CAST(grid_lat AS DOUBLE) AS grid_lat,
                CAST(grid_lng AS DOUBLE) AS grid_lng,
                cell_id,
                SUM(overlap_sec)/60.0 AS exposure_min
              FROM overlap
              WHERE overlap_sec > 0
              GROUP BY 1,2,3,4
            )
            TO '{temp_path.as_posix()}'
            (FORMAT PARQUET);
            """)

    # Combine all quarters and filter to strictly < 2025-01-01
    print("  Combining training quarters (filtering to < 2025-01-01)...")
    existing_files = [f for f in temp_files if f.exists()]
    if existing_files:
        all_files = "', '".join([f.as_posix() for f in existing_files])
        con.execute(f"""
        COPY (
          SELECT * FROM read_parquet(['{all_files}'])
          WHERE hour_ts < TIMESTAMP '2025-01-01'
        )
        TO '{exposure_train_path.as_posix()}'
        (FORMAT PARQUET);
        """)

        # Cleanup
        for f in temp_files:
            if f.exists():
                f.unlink()

    print("Wrote:", exposure_train_path)

# Build test exposure (2025 only) - with same line interpolation
if exposure_test_path.exists():
    print("Exists, skipping:", exposure_test_path)
else:
    print(f"Creating TEST exposure for 2025 with LINE INTERPOLATION...")

    # Line interpolation: distribute exposure across all cells along the route
    con.execute(f"""
    COPY (
      WITH trips AS (
        SELECT
          try_cast(started_at AS TIMESTAMP) AS started_at,
          try_cast(ended_at   AS TIMESTAMP) AS ended_at,
          start_lat, start_lng,
          end_lat, end_lng,
          duration_sec
        FROM read_parquet('{TRIPS.as_posix()}')
        WHERE try_cast(started_at AS TIMESTAMP) >= TIMESTAMP '2025-01-01'
          AND try_cast(started_at AS TIMESTAMP) <  TIMESTAMP '2026-01-01'
          AND start_lat IS NOT NULL AND start_lng IS NOT NULL
          AND end_lat IS NOT NULL AND end_lng IS NOT NULL
          AND ended_at IS NOT NULL
          AND duration_sec IS NOT NULL
          AND duration_sec > 0
          AND duration_sec < 4*60*60
      ),
      -- Generate interpolation points along the line
      interp_points AS (
        SELECT
          started_at, ended_at, duration_sec,
          -- Interpolate: start + t * (end - start) for t in [0, 1]
          start_lat + (gs.t / {N_INTERP_POINTS - 1}.0) * (end_lat - start_lat) AS interp_lat,
          start_lng + (gs.t / {N_INTERP_POINTS - 1}.0) * (end_lng - start_lng) AS interp_lng
        FROM trips
        CROSS JOIN generate_series(0, {N_INTERP_POINTS - 1}) AS gs(t)
      ),
      -- Bin each interpolated point to grid cell
      binned_points AS (
        SELECT
          started_at, ended_at, duration_sec,
          floor(interp_lat / {GRID_DEG}) * {GRID_DEG} AS grid_lat,
          floor(interp_lng / {GRID_DEG}) * {GRID_DEG} AS grid_lng
        FROM interp_points
      ),
      -- Get unique cells per trip and count them
      unique_cells_per_trip AS (
        SELECT
          started_at, ended_at, duration_sec,
          grid_lat, grid_lng,
          COUNT(*) OVER (PARTITION BY started_at, ended_at, duration_sec) AS n_cells
        FROM (
          SELECT DISTINCT started_at, ended_at, duration_sec, grid_lat, grid_lng
          FROM binned_points
        ) sub
      ),
      -- Distribute duration equally across cells
      exposure_per_cell AS (
        SELECT
          started_at, ended_at,
          grid_lat, grid_lng,
          CAST(floor(grid_lat / {GRID_DEG}) * {GRID_DEG} AS VARCHAR) || '_' ||
          CAST(floor(grid_lng / {GRID_DEG}) * {GRID_DEG} AS VARCHAR) AS cell_id,
          duration_sec / n_cells AS cell_duration_sec
        FROM unique_cells_per_trip
      ),
      -- Expand to hourly buckets
      expanded AS (
        SELECT
          cell_id, grid_lat, grid_lng,
          gs AS hour_ts,
          started_at, ended_at, cell_duration_sec
        FROM exposure_per_cell
        CROSS JOIN generate_series(
          date_trunc('hour', started_at),
          date_trunc('hour', ended_at),
          INTERVAL '1 hour'
        ) AS t(gs)
      ),
      -- Calculate overlap with each hour
      overlap AS (
        SELECT
          cell_id, grid_lat, grid_lng, hour_ts,
          GREATEST(
            0,
            (cell_duration_sec / NULLIF(EXTRACT(EPOCH FROM ended_at - started_at), 0)) *
            EXTRACT(EPOCH FROM LEAST(ended_at, hour_ts + INTERVAL '1 hour')
                          - GREATEST(started_at, hour_ts))
          ) AS overlap_sec
        FROM expanded
      )
      SELECT
        hour_ts,
        CAST(grid_lat AS DOUBLE) AS grid_lat,
        CAST(grid_lng AS DOUBLE) AS grid_lng,
        cell_id,
        SUM(overlap_sec)/60.0 AS exposure_min
      FROM overlap
      WHERE overlap_sec > 0
      GROUP BY 1,2,3,4
    )
    TO '{exposure_test_path.as_posix()}'
    (FORMAT PARQUET);
    """)
    print("Wrote:", exposure_test_path)


# Sanity checks
print("\nTraining exposure:")
print(con.execute(f"""
SELECT COUNT(*) n, MIN(hour_ts) min_ts, MAX(hour_ts) max_ts, SUM(exposure_min) total_exp_min
FROM read_parquet('{exposure_train_path.as_posix()}');
""").fetch_df())

print("\nTest exposure:")
print(con.execute(f"""
SELECT COUNT(*) n, MIN(hour_ts) min_ts, MAX(hour_ts) max_ts, SUM(exposure_min) total_exp_min
FROM read_parquet('{exposure_test_path.as_posix()}');
""").fetch_df())


# In[5]:


con.execute(f"""
CREATE OR REPLACE VIEW weather_hourly AS
SELECT
  CAST(timestamp AS TIMESTAMP) AS hour_ts,
  temp, prcp, snow, wspd
FROM read_parquet('{(WEATHER_HOURLY_DIR.as_posix() + "/**/*.parquet")}')
WHERE hour_ts >= TIMESTAMP '{TMIN}'
  AND hour_ts <  TIMESTAMP '{TMAX}';  -- Full range OK, filtered at join time
""")

print("weather_hourly rows:", con.execute("SELECT COUNT(*) FROM weather_hourly").fetchone()[0])
print("weather range:", con.execute("SELECT MIN(hour_ts), MAX(hour_ts) FROM weather_hourly").fetchone())


# In[7]:


# Two separate grids:
# 1. Daily grid for GLM training (memory-efficient: ~117K rows)
# 2. Hourly grid for dashboard analysis (kept for hour-of-day patterns)
grid_train_day_path = OUT_DIR / "grid_train_cell_day_2020_2024.parquet"  # For GLM
grid_train_hour_path = OUT_DIR / "grid_train_cell_hour_2020_2024.parquet"  # For Dashboard
cells_keep_path = OUT_DIR / "grid_cells_keep_2020_2024.parquet"

if FORCE_REBUILD:
    if grid_train_day_path.exists(): grid_train_day_path.unlink()
    if grid_train_hour_path.exists(): grid_train_hour_path.unlink()
    if cells_keep_path.exists(): cells_keep_path.unlink()

# 1) Determine cells covering CRASH_COVERAGE of total CRASHES (TRAINING DATA ONLY)
# NOTE: We select by CRASH count, not exposure! This ensures we cover most crashes.
# Exposure-based selection was problematic: 80% exposure = only 37% crashes
con.execute(f"""
COPY (
  WITH cell_crashes AS (
    -- Count crashes per cell in training period
    SELECT
      cell_id,
      SUM(y_bike) AS crash_sum
    FROM read_parquet('{crash_cell_hour_path.as_posix()}')
    WHERE hour_ts >= TIMESTAMP '{TMIN}'
      AND hour_ts <  TIMESTAMP '{TMAX_TRAIN}'
    GROUP BY 1
  ),
  cell_exp AS (
    -- Also get exposure for reference
    SELECT
      cell_id,
      SUM(exposure_min) AS exp_sum
    FROM read_parquet('{exposure_train_path.as_posix()}')
    WHERE hour_ts >= TIMESTAMP '{TMIN}'
      AND hour_ts <  TIMESTAMP '{TMAX_TRAIN}'
    GROUP BY 1
  ),
  merged AS (
    -- Join crashes and exposure, keep only cells with exposure > 0
    SELECT
      COALESCE(c.cell_id, e.cell_id) AS cell_id,
      COALESCE(c.crash_sum, 0) AS crash_sum,
      COALESCE(e.exp_sum, 0) AS exp_sum
    FROM cell_crashes c
    FULL OUTER JOIN cell_exp e USING(cell_id)
    WHERE e.exp_sum > 0  -- Must have some exposure to be in model
  ),
  ranked AS (
    SELECT
      cell_id,
      crash_sum,
      exp_sum,
      SUM(crash_sum) OVER (ORDER BY crash_sum DESC) AS cum_crash,
      SUM(crash_sum) OVER () AS total_crash
    FROM merged
  )
  SELECT
    cell_id,
    crash_sum,
    exp_sum,
    cum_crash / NULLIF(total_crash, 0) AS cum_share
  FROM ranked
  WHERE cum_crash / NULLIF(total_crash, 0) <= {CRASH_COVERAGE}
)
TO '{cells_keep_path.as_posix()}'
(FORMAT PARQUET);
""")

print("Wrote:", cells_keep_path)

# ============================================================================
# 2) Build DAILY grid for GLM training (memory-efficient: ~117K rows)
# ============================================================================
# Aggregates hourly data to daily level to reduce RAM usage from ~2.8M to ~117K rows
# This is sufficient for GLM training since hour-of-day is not a critical feature
print("\nBuilding DAILY training grid (cells × days CROSS JOIN)...")
print("This creates ~117K rows (64 cells × 1826 days)...")

con.execute(f"""
COPY (
  WITH cell_coords AS (
    -- Get grid coordinates from exposure file (one row per cell)
    SELECT DISTINCT cell_id, grid_lat, grid_lng
    FROM read_parquet('{exposure_train_path.as_posix()}')
  ),
  keep_cells AS (
    -- List of cells to keep (top X% crashes)
    SELECT cell_id
    FROM read_parquet('{cells_keep_path.as_posix()}')
  ),
  all_days AS (
    -- Generate all days in training period
    SELECT DISTINCT date_trunc('day', hour_ts) AS day_ts
    FROM weather_hourly
    WHERE hour_ts >= TIMESTAMP '{TMIN}'
      AND hour_ts <  TIMESTAMP '{TMAX_TRAIN}'
  ),
  all_grid AS (
    -- CROSS JOIN: every cell × every day in training period
    SELECT
      c.cell_id, c.grid_lat, c.grid_lng,
      d.day_ts
    FROM cell_coords c
    INNER JOIN keep_cells k ON c.cell_id = k.cell_id
    CROSS JOIN all_days d
  ),
  e_daily AS (
    -- Aggregate exposure to daily level
    SELECT
      date_trunc('day', hour_ts) AS day_ts,
      cell_id,
      SUM(exposure_min) AS exposure_min
    FROM read_parquet('{exposure_train_path.as_posix()}')
    WHERE hour_ts >= TIMESTAMP '{TMIN}'
      AND hour_ts <  TIMESTAMP '{TMAX_TRAIN}'
    GROUP BY 1, 2
  ),
  c_daily AS (
    -- Aggregate crashes to daily level
    SELECT
      date_trunc('day', hour_ts) AS day_ts,
      cell_id,
      SUM(y_bike) AS y_bike
    FROM read_parquet('{crash_cell_hour_path.as_posix()}')
    WHERE hour_ts >= TIMESTAMP '{TMIN}'
      AND hour_ts <  TIMESTAMP '{TMAX_TRAIN}'
    GROUP BY 1, 2
  ),
  w_daily AS (
    -- Aggregate weather to daily level (AVG for temp/wspd, SUM for prcp/snow)
    SELECT
      date_trunc('day', hour_ts) AS day_ts,
      AVG(temp) AS temp,
      SUM(prcp) AS prcp,
      SUM(snow) AS snow,
      AVG(wspd) AS wspd
    FROM weather_hourly
    WHERE hour_ts >= TIMESTAMP '{TMIN}'
      AND hour_ts <  TIMESTAMP '{TMAX_TRAIN}'
    GROUP BY 1
  )
  SELECT
    g.day_ts,
    g.cell_id,
    g.grid_lat,
    g.grid_lng,
    COALESCE(e.exposure_min, 0) AS exposure_min,  -- 0 if no CitiBike trips
    COALESCE(c.y_bike, 0) AS y_bike,              -- 0 if no crashes
    EXTRACT(DOW   FROM g.day_ts) AS dow,
    EXTRACT(MONTH FROM g.day_ts) AS month,
    w.temp, w.prcp, w.snow, w.wspd
  FROM all_grid g
  LEFT JOIN e_daily e USING(day_ts, cell_id)
  LEFT JOIN c_daily c USING(day_ts, cell_id)
  LEFT JOIN w_daily w USING(day_ts)
)
TO '{grid_train_day_path.as_posix()}'
(FORMAT PARQUET);
""")

print("Wrote:", grid_train_day_path)
print(con.execute(f"""
SELECT COUNT(*) n, AVG(y_bike) y_mean, AVG(exposure_min) exp_mean
FROM read_parquet('{grid_train_day_path.as_posix()}');
""").fetch_df())

# ============================================================================
# 3) Build HOURLY grid for Dashboard analysis (hour-of-day patterns)
# ============================================================================
# This file is NOT loaded into pandas during GLM training - only used by dashboard
# Dashboard uses Streamlit's @st.cache_data which is separate from this script's RAM
print("\nBuilding HOURLY training grid for Dashboard (cells × hours)...")
print("This file will only be used by the dashboard, not loaded here...")

con.execute(f"""
COPY (
  WITH cell_coords AS (
    SELECT DISTINCT cell_id, grid_lat, grid_lng
    FROM read_parquet('{exposure_train_path.as_posix()}')
  ),
  keep_cells AS (
    SELECT cell_id
    FROM read_parquet('{cells_keep_path.as_posix()}')
  ),
  all_grid AS (
    SELECT
      c.cell_id, c.grid_lat, c.grid_lng,
      h.hour_ts
    FROM cell_coords c
    INNER JOIN keep_cells k ON c.cell_id = k.cell_id
    CROSS JOIN (
      SELECT DISTINCT hour_ts
      FROM weather_hourly
      WHERE hour_ts >= TIMESTAMP '{TMIN}'
        AND hour_ts <  TIMESTAMP '{TMAX_TRAIN}'
    ) h
  ),
  e AS (
    SELECT hour_ts, cell_id, exposure_min
    FROM read_parquet('{exposure_train_path.as_posix()}')
    WHERE hour_ts >= TIMESTAMP '{TMIN}'
      AND hour_ts <  TIMESTAMP '{TMAX_TRAIN}'
  ),
  c AS (
    SELECT hour_ts, cell_id, y_bike
    FROM read_parquet('{crash_cell_hour_path.as_posix()}')
    WHERE hour_ts >= TIMESTAMP '{TMIN}'
      AND hour_ts <  TIMESTAMP '{TMAX_TRAIN}'
  )
  SELECT
    g.hour_ts,
    g.cell_id,
    g.grid_lat,
    g.grid_lng,
    COALESCE(e.exposure_min, 0) AS exposure_min,
    COALESCE(c.y_bike, 0) AS y_bike,
    EXTRACT(HOUR  FROM g.hour_ts) AS hour_of_day,
    EXTRACT(DOW   FROM g.hour_ts) AS dow,
    EXTRACT(MONTH FROM g.hour_ts) AS month,
    w.temp, w.prcp, w.snow, w.wspd
  FROM all_grid g
  LEFT JOIN e USING(hour_ts, cell_id)
  LEFT JOIN c USING(hour_ts, cell_id)
  LEFT JOIN weather_hourly w USING(hour_ts)
)
TO '{grid_train_hour_path.as_posix()}'
(FORMAT PARQUET);
""")

print("Wrote:", grid_train_hour_path)
# NOTE: We do NOT load this into pandas to save RAM - dashboard will load it separately


# In[16]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

TARGET = "y_bike"
# NOTE: No MIN_EXPOSURE_MIN filter - we include ALL observations
# Exposure=0 is valid and handled via log1p transformation
weather_cols = ["temp","prcp","snow","wspd"]

# ============================================================================
# Load DAILY grid for GLM training (memory-efficient: ~117K rows instead of ~2.8M)
# ============================================================================
print("\nLoading DAILY training grid for GLM...")
train_df = con.execute(f"SELECT * FROM read_parquet('{grid_train_day_path.as_posix()}')").fetch_df()
print(f"Loaded {len(train_df):,} rows (daily aggregation)")

# clean - only drop rows with missing required features
# NOTE: No hour_of_day in daily data - removed from formula
need = ["exposure_min", TARGET, "dow", "month", "grid_lat", "grid_lng"] + weather_cols
train_df = train_df.dropna(subset=need).copy()

# EXPOSURE AS FEATURE (not offset!)
# log1p handles exposure=0 gracefully: log(0+1) = 0
train_df["log1p_exposure"] = np.log1p(train_df["exposure_min"].values)

# categories (no hour_of_day in daily data)
train_df["dow"] = train_df["dow"].astype(int).astype("category")
train_df["month"] = train_df["month"].astype(int).astype("category")

# CRITICAL: Compute normalization statistics from TRAINING DATA ONLY (2020-2024)
# These statistics will be applied to test data (2025) to prevent data leakage
weather_stats = {}
for col in weather_cols:
    m = float(train_df[col].mean())
    s = float(train_df[col].std())
    if s == 0 or np.isnan(s):
        train_df.drop(columns=[col], inplace=True)
    else:
        weather_stats[col] = (m, s)
        train_df[col] = (train_df[col] - m) / s

active_weather_cols = [c for c in weather_cols if c in train_df.columns]

# Normalize spatial features (FROM TRAINING DATA ONLY)
# NOTE: We standardize to put lat/lng on similar scale as other features,
# preventing numerical issues in the GLM fitting algorithm. With 64 unique
# grid cells, the raw coordinates have small variance which could cause
# optimization problems. Standardization ensures stable coefficient estimation.
lat_mean, lat_std = train_df["grid_lat"].mean(), train_df["grid_lat"].std()
lng_mean, lng_std = train_df["grid_lng"].mean(), train_df["grid_lng"].std()

train_df["grid_lat_norm"] = (train_df["grid_lat"] - lat_mean) / lat_std
train_df["grid_lng_norm"] = (train_df["grid_lng"] - lng_mean) / lng_std

# spatial features (normalized versions)
train_df["lat2"] = train_df["grid_lat_norm"]**2
train_df["lng2"] = train_df["grid_lng_norm"]**2
train_df["lat_lng"] = train_df["grid_lat_norm"] * train_df["grid_lng_norm"]

# TREND VARIABLE: Captures temporal changes in crash rates
# Normalized to years since start of training period
# NOTE: Using day_ts instead of hour_ts for daily data
TREND_BASE_DATE = pd.Timestamp(TMIN)  # Uses TMIN constant (2021-01-01)
train_df["trend"] = (train_df["day_ts"] - TREND_BASE_DATE).dt.days / 365.25

# ============================================================================
# DAILY MODEL FORMULA (no hour_of_day - aggregated to daily level)
# ============================================================================
# Removed C(hour_of_day) because we aggregate to daily level for memory efficiency
# This reduces training rows from ~2.8M to ~117K
rhs = [
    "C(dow)", "C(month)",  # NOTE: No hour_of_day in daily model
    "grid_lat_norm", "grid_lng_norm", "lat2", "lng2", "lat_lng",
    "trend",  # Captures temporal changes - improves predictions significantly
    "log1p_exposure",  # FEATURE, not offset!
] + active_weather_cols

formula = f"{TARGET} ~ " + " + ".join(rhs)
print("Formula:", formula)
print("\nNOTE: Daily aggregation - no hour_of_day feature (saves ~24x RAM)")
print("NOTE: Exposure is a FEATURE with estimated coefficient (not offset with β=1)")

# ---- Poisson GLM ----
# NOTE: No offset - exposure is a feature in the formula
poisson_model = smf.glm(
    formula=formula,
    data=train_df,
    family=sm.families.Poisson()
)
poisson_res = poisson_model.fit(cov_type="HC0")
print("\n=== POISSON (grid model) ===")
print(poisson_res.summary())

pearson_chi2 = float(np.sum(poisson_res.resid_pearson**2))
disp_poiss = pearson_chi2 / float(poisson_res.df_resid)
print("\nPoisson overdispersion χ²/df:", disp_poiss)

# NOTE: Negative Binomial removed - Poisson dispersion ~1 indicates no overdispersion

meta = {
    "GRID_DEG": GRID_DEG,
    "CRASH_COVERAGE": CRASH_COVERAGE,
    "weather_stats": {k: {"mean": v[0], "std": v[1]} for k,v in weather_stats.items()},
    "spatial_stats": {
        "lat_mean": float(lat_mean),
        "lat_std": float(lat_std),
        "lng_mean": float(lng_mean),
        "lng_std": float(lng_std)
    },
    "trend_included": True,
    "trend_base_date": TMIN,  # Uses TMIN constant
    "exposure_as_feature": True,  # Exposure is a feature, not offset
    "exposure_coefficient": float(poisson_res.params.get("log1p_exposure", 0)),
    "formula": formula,
    "active_weather_cols": active_weather_cols
}
meta_path = OUT_DIR / "model_meta_bike_all.json"
meta_path.write_text(json.dumps(meta, indent=2))
print("\nSaved:", meta_path)


# In[18]:


# ============================================================================
# Build DAILY 2025 grid for Monte Carlo predictions (memory-efficient)
# ============================================================================
# IMPORTANT: Only include cells that had ACTUAL 2025 exposure!
# This ensures prediction and observed use the same cell set.
# Previously we used training cells (cells_keep), but some training cells
# may not have 2025 exposure, leading to inflated predictions.
grid_2025_day_path = OUT_DIR / "grid_2025_cell_day_bike_all.parquet"
cells_2025_path = OUT_DIR / "grid_cells_2025.parquet"  # Cells with 2025 exposure

if FORCE_REBUILD and grid_2025_day_path.exists():
    grid_2025_day_path.unlink()
if FORCE_REBUILD and cells_2025_path.exists():
    cells_2025_path.unlink()

# First: Identify cells that had ANY exposure in 2025
# These must ALSO be in cells_keep (training cells) to have model coefficients
print("\nIdentifying cells with 2025 exposure (intersection with training cells)...")
con.execute(f"""
COPY (
  SELECT DISTINCT e.cell_id
  FROM read_parquet('{exposure_test_path.as_posix()}') e
  INNER JOIN read_parquet('{cells_keep_path.as_posix()}') k USING(cell_id)
  WHERE e.hour_ts >= TIMESTAMP '2025-01-01'
    AND e.hour_ts <  TIMESTAMP '2026-01-01'
    AND e.exposure_min > 0
)
TO '{cells_2025_path.as_posix()}'
(FORMAT PARQUET);
""")

n_cells_2025 = con.execute(f"SELECT COUNT(*) FROM read_parquet('{cells_2025_path.as_posix()}')").fetchone()[0]
n_cells_train = con.execute(f"SELECT COUNT(*) FROM read_parquet('{cells_keep_path.as_posix()}')").fetchone()[0]
print(f"Training cells (2020-2024): {n_cells_train}")
print(f"Cells with 2025 exposure:   {n_cells_2025}")
print(f"Coverage: {n_cells_2025/n_cells_train*100:.1f}% of training cells active in 2025")

print("\nBuilding DAILY 2025 grid (only cells with 2025 exposure)...")

con.execute(f"""
COPY (
  WITH cell_coords AS (
    -- Get coordinates from training exposure (has grid_lat, grid_lng)
    SELECT DISTINCT cell_id, grid_lat, grid_lng
    FROM read_parquet('{exposure_train_path.as_posix()}')
  ),
  cells_2025 AS (
    -- Only cells that had 2025 exposure AND are in training set
    SELECT cell_id
    FROM read_parquet('{cells_2025_path.as_posix()}')
  ),
  all_days AS (
    -- Generate all days in 2025
    SELECT DISTINCT date_trunc('day', hour_ts) AS day_ts
    FROM weather_hourly
    WHERE hour_ts >= TIMESTAMP '2025-01-01'
      AND hour_ts <  TIMESTAMP '2026-01-01'
  ),
  all_grid AS (
    -- CROSS JOIN: only 2025-active cells × every day in 2025
    SELECT
      c.cell_id, c.grid_lat, c.grid_lng,
      d.day_ts
    FROM cell_coords c
    INNER JOIN cells_2025 k ON c.cell_id = k.cell_id
    CROSS JOIN all_days d
  ),
  e_daily AS (
    -- Aggregate exposure to daily level
    SELECT
      date_trunc('day', hour_ts) AS day_ts,
      cell_id,
      SUM(exposure_min) AS exposure_min
    FROM read_parquet('{exposure_test_path.as_posix()}')
    WHERE hour_ts >= TIMESTAMP '2025-01-01'
      AND hour_ts <  TIMESTAMP '2026-01-01'
    GROUP BY 1, 2
  ),
  w_daily AS (
    -- Aggregate weather to daily level (same as training)
    SELECT
      date_trunc('day', hour_ts) AS day_ts,
      AVG(temp) AS temp,
      SUM(prcp) AS prcp,
      SUM(snow) AS snow,
      AVG(wspd) AS wspd
    FROM weather_hourly
    WHERE hour_ts >= TIMESTAMP '2025-01-01'
      AND hour_ts <  TIMESTAMP '2026-01-01'
    GROUP BY 1
  )
  SELECT
    g.day_ts,
    g.cell_id,
    g.grid_lat,
    g.grid_lng,
    COALESCE(e.exposure_min, 0) AS exposure_min,
    EXTRACT(DOW   FROM g.day_ts) AS dow,
    EXTRACT(MONTH FROM g.day_ts) AS month,
    w.temp, w.prcp, w.snow, w.wspd
  FROM all_grid g
  LEFT JOIN e_daily e USING(day_ts, cell_id)
  LEFT JOIN w_daily w USING(day_ts)
  ORDER BY g.day_ts, g.cell_id
)
TO '{grid_2025_day_path.as_posix()}'
(FORMAT PARQUET);
""")

print("Wrote:", grid_2025_day_path)
print(con.execute(f"SELECT COUNT(*) n FROM read_parquet('{grid_2025_day_path.as_posix()}')").fetch_df())


# In[ ]:


import patsy
import gc  # For explicit garbage collection to save RAM
from numpy.random import default_rng

rng = default_rng(42)
S = 30  # Monte Carlo simulations (reduced from 50 for RAM; still sufficient for CI)

# ============================================================================
# STEP 1: Load Weather for ALL Years (by day-of-year) - DAILY AGGREGATION
# ============================================================================
print("="*70)
print("PREPARING YEAR-LEVEL WEATHER BOOTSTRAP (day-of-year mapping - DAILY)")
print("="*70)

# Load training weather from DAILY grid (already aggregated)
train_weather = con.execute(f"SELECT * FROM read_parquet('{grid_train_day_path.as_posix()}')").fetch_df()
train_weather["day_ts"] = pd.to_datetime(train_weather["day_ts"])
train_weather["year"] = train_weather["day_ts"].dt.year
train_weather["dayofyear"] = train_weather["day_ts"].dt.dayofyear

weather_cols = ["temp", "prcp", "snow", "wspd"]
available_weather_years = [2021, 2022, 2023, 2024]  # Training years for weather (excludes COVID 2020)

# Build lookup: year → dayofyear → weather (DAILY, no hour dimension)
year_weather_dict = {}

for year in available_weather_years:
    year_data = train_weather[train_weather['year'] == year].copy()

    # Aggregate to dayofyear - take median if multiple observations (across cells)
    year_agg = year_data.groupby(['dayofyear'])[weather_cols].median().reset_index()

    # Convert to dict (key is just dayofyear, not (dayofyear, hour))
    year_dict = {}
    for _, row in year_agg.iterrows():
        key = int(row['dayofyear'])
        year_dict[key] = {col: row[col] for col in weather_cols}

    year_weather_dict[year] = year_dict
    print(f"  Year {year}: {len(year_dict)} unique days")

# Also add 2025 weather from the 2025 grid
print("\nAdding 2025 weather...")
weather_2025 = con.execute(f"""
SELECT
    EXTRACT(dayofyear FROM day_ts)::INT AS dayofyear,
    AVG(temp) AS temp,
    AVG(prcp) AS prcp,
    AVG(snow) AS snow,
    AVG(wspd) AS wspd
FROM read_parquet('{grid_2025_day_path.as_posix()}')
GROUP BY EXTRACT(dayofyear FROM day_ts)
""").fetch_df()

year_dict_2025 = {}
for _, row in weather_2025.iterrows():
    key = int(row['dayofyear'])
    year_dict_2025[key] = {col: row[col] for col in weather_cols}
year_weather_dict[2025] = year_dict_2025
print(f"  Year 2025: {len(year_dict_2025)} unique days")

# Update available years to include 2025 (still excludes 2020)
available_weather_years = [2021, 2022, 2023, 2024, 2025]

# Check coverage
test_key = 15  # 15th day of year
print(f"\nDay 15 across years:")
for year in available_weather_years:
    if test_key in year_weather_dict[year]:
        w = year_weather_dict[year][test_key]
        print(f"  {year}: temp={w['temp']:+.1f}°C, prcp={w['prcp']:.1f}mm")

# Free memory from train_weather
del train_weather
gc.collect()

# ============================================================================
# STEP 1b: Extract Daily Exposure for ALL Years (2020-2025)
# ============================================================================
# This enables comprehensive uncertainty simulation with different exposure scenarios
print("\n" + "="*70)
print("EXTRACTING EXPOSURE FOR ALL YEARS (2020-2025)")
print("="*70)

# Extract 2020-2024 from training data
exposure_train_years = con.execute(f"""
SELECT
    cell_id,
    EXTRACT(year FROM hour_ts)::INT AS year,
    EXTRACT(dayofyear FROM hour_ts)::INT AS dayofyear,
    SUM(exposure_min) AS exposure_min
FROM read_parquet('{exposure_train_path.as_posix()}')
GROUP BY cell_id, EXTRACT(year FROM hour_ts), EXTRACT(dayofyear FROM hour_ts)
""").fetch_df()

# Extract 2025 from test data
exposure_2025_df = con.execute(f"""
SELECT
    cell_id,
    2025 AS year,
    EXTRACT(dayofyear FROM hour_ts)::INT AS dayofyear,
    SUM(exposure_min) AS exposure_min
FROM read_parquet('{exposure_test_path.as_posix()}')
WHERE hour_ts >= TIMESTAMP '2025-01-01'
  AND hour_ts <  TIMESTAMP '2026-01-01'
GROUP BY cell_id, EXTRACT(dayofyear FROM hour_ts)
""").fetch_df()

# Build nested lookup dict: year -> (cell_id, dayofyear) -> exposure
exposure_by_year = {}
available_exposure_years = [2021, 2022, 2023, 2024, 2025]  # Excludes COVID 2020

# Add 2021-2024 (excludes 2020)
for year in [2021, 2022, 2023, 2024]:
    year_data = exposure_train_years[exposure_train_years['year'] == year]
    year_lookup = {}
    for _, row in year_data.iterrows():
        key = (row['cell_id'], int(row['dayofyear']))
        year_lookup[key] = row['exposure_min']
    exposure_by_year[year] = year_lookup
    print(f"  {year}: {len(year_lookup):,} cell-day combinations")

# Add 2025
year_lookup_2025 = {}
for _, row in exposure_2025_df.iterrows():
    key = (row['cell_id'], int(row['dayofyear']))
    year_lookup_2025[key] = row['exposure_min']
exposure_by_year[2025] = year_lookup_2025
print(f"  2025: {len(year_lookup_2025):,} cell-day combinations")

# Free memory
del exposure_train_years, exposure_2025_df
gc.collect()

# ============================================================================
# STEP 2: Prepare 2025 DAILY Data
# ============================================================================
print("\nLoading DAILY 2025 grid for Monte Carlo...")
df2025 = con.execute(f"SELECT * FROM read_parquet('{grid_2025_day_path.as_posix()}')").fetch_df()
df2025 = df2025.dropna(subset=["exposure_min", "dow", "month", "grid_lat", "grid_lng"]).copy()
# NOTE: No exposure_min > 0 filter - we include ALL observations
# Exposure=0 is valid and handled via log1p transformation

df2025["day_ts"] = pd.to_datetime(df2025["day_ts"])
df2025["dayofyear"] = df2025["day_ts"].dt.dayofyear.astype(int)

# Categories (no hour_of_day in daily data)
df2025["dow"] = df2025["dow"].astype(int).astype("category")
df2025["month"] = df2025["month"].astype(int).astype("category")

# Add trend variable for 2025 (same base date as training)
# Note: TREND_BASE_DATE already defined earlier using TMIN
df2025["trend"] = (df2025["day_ts"] - TREND_BASE_DATE).dt.days / 365.25

# EXPOSURE AS FEATURE (not offset!) - same as training
df2025["log1p_exposure"] = np.log1p(df2025["exposure_min"].values)

print(f"2025 data prepared: {len(df2025):,} rows (DAILY aggregation)")
print(f"Trend range: {df2025['trend'].min():.2f} to {df2025['trend'].max():.2f} years")
print(f"log1p_exposure range: {df2025['log1p_exposure'].min():.2f} to {df2025['log1p_exposure'].max():.2f}")

if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    wstats = meta["weather_stats"]
    sstats = meta["spatial_stats"]
else:
    raise ValueError("meta_path not found!")

print(f"\nPrepared 2025 data: {len(df2025):,} rows")

# Pre-extract keys for fast lookup: (cell_id, dayofyear) pairs
df2025_keys = df2025['dayofyear'].values
df2025_cell_ids = df2025['cell_id'].values
df2025_cell_day_keys = list(zip(df2025_cell_ids, df2025_keys))

# ============================================================================
# STEP 3: FULLY RANDOM Monte Carlo Simulation
# ============================================================================
# For EACH simulation, randomly sample:
# - Weather year (2021-2025) - excludes COVID 2020
# - Exposure year (2021-2025) - excludes COVID 2020
# - Growth factor: Uniform(0.85, 1.15) - continuous sampling
# - Parameters (from covariance matrix)
# ============================================================================

# ============================================================================
# EXPOSURE SCENARIO INTERPRETATION:
# When exposure_year != 2025, we apply historical usage patterns to the 2025 network.
# This answers: "What if the 2025 network had been used like in year X?"
# This is NOT a reconstruction of actual historical crash totals.
# Cells that didn't exist in year X will have exposure=0 (contributes minimally).
# ============================================================================

print("\n" + "="*70)
print("RUNNING MONTE CARLO SIMULATION (POISSON)")
print("="*70)
print("Each simulation randomly samples:")
print("  - Weather year:  2021-2025 (excludes COVID 2020)")
print("  - Exposure year: 2021-2025 (excludes COVID 2020)")
print("  - Growth factor: Uniform(0.80, 1.20)")
print("  - Parameters:    from covariance matrix")
print("="*70)

# Configuration
S = 1000  # Many simulations for smooth distribution
GROWTH_MIN = 0.80  # -20% (historically observed)
GROWTH_MAX = 1.20  # +20% (historically observed)

def run_random_simulation(res, use_nb: bool, model_name: str):
    """
    Run S simulations with ALL dimensions randomly sampled per simulation.
    """
    design_info = res.model.data.design_info

    if use_nb:
        alpha_nb = float(res.scale)
        print(f"\n  {model_name} (alpha={alpha_nb:.6f})")
    else:
        alpha_nb = 0.0
        print(f"\n  {model_name}")

    results = []

    for s in range(S):
        if s % 50 == 0:
            print(f"    Simulation {s}/{S}...")

        # ================================================================
        # A) RANDOM: Sample weather year (2021-2025)
        # ================================================================
        weather_year = int(rng.choice(available_weather_years))
        year_weather = year_weather_dict[weather_year]

        # ================================================================
        # B) RANDOM: Sample exposure year (2021-2025)
        # ================================================================
        exposure_year = int(rng.choice(available_exposure_years))

        # ================================================================
        # C) RANDOM: Sample growth factor from continuous uniform distribution
        # ================================================================
        growth_factor = float(rng.uniform(GROWTH_MIN, GROWTH_MAX))

        # ================================================================
        # D) Apply weather from sampled year
        # ================================================================
        weather_matrix = np.zeros((len(df2025), 4))
        for i, key in enumerate(df2025_keys):
            weather_key = min(key, 365)  # Handle leap year
            if weather_key in year_weather:
                w = year_weather[weather_key]
                weather_matrix[i] = [w['temp'], w['prcp'], w['snow'], w['wspd']]

        df_sim = df2025.copy()
        df_sim['temp'] = weather_matrix[:, 0]
        df_sim['prcp'] = weather_matrix[:, 1]
        df_sim['snow'] = weather_matrix[:, 2]
        df_sim['wspd'] = weather_matrix[:, 3]

        # Scale weather
        for col in weather_cols:
            if col in wstats:
                df_sim[col] = (df_sim[col] - wstats[col]["mean"]) / wstats[col]["std"]

        # Scale spatial
        df_sim["grid_lat_norm"] = (df_sim["grid_lat"] - sstats["lat_mean"]) / sstats["lat_std"]
        df_sim["grid_lng_norm"] = (df_sim["grid_lng"] - sstats["lng_mean"]) / sstats["lng_std"]
        df_sim["lat2"] = df_sim["grid_lat_norm"]**2
        df_sim["lng2"] = df_sim["grid_lng_norm"]**2
        df_sim["lat_lng"] = df_sim["grid_lat_norm"] * df_sim["grid_lng_norm"]

        # ================================================================
        # E) Apply exposure from sampled year with growth factor
        # ================================================================
        if exposure_year == 2025:
            exposure_values = df_sim["exposure_min"].values.copy()
        else:
            exposure_values = np.zeros(len(df2025))
            year_exposure_lookup = exposure_by_year.get(exposure_year, {})
            for i, (cell_id, doy) in enumerate(df2025_cell_day_keys):
                lookup_doy = min(doy, 365)
                exposure_values[i] = year_exposure_lookup.get((cell_id, lookup_doy), 0.0)

        # Apply growth factor
        scenario_exposure = exposure_values * growth_factor
        df_sim["log1p_exposure"] = np.log1p(scenario_exposure)

        # ================================================================
        # F) Sample β from parameter distribution
        # ================================================================
        X = patsy.build_design_matrices([design_info], df_sim, return_type="dataframe")[0]
        xcols = X.columns.tolist()

        beta_hat = res.params.loc[xcols].values
        cov_hat = res.cov_params().loc[xcols, xcols].values
        beta_s = rng.multivariate_normal(mean=beta_hat, cov=cov_hat)

        # ================================================================
        # G) Calculate μ and sample crashes
        # ================================================================
        eta = X.values @ beta_s
        eta = np.clip(eta, -20, 20)
        mu = np.exp(eta)
        mu = np.clip(mu, 0, 1e6)

        if use_nb and alpha_nb > 0:
            shape = 1.0 / alpha_nb
            lam = rng.gamma(shape=shape, scale=alpha_nb * mu)
            lam = np.clip(lam, 0, 1e6)
            y = rng.poisson(lam)
        else:
            y = rng.poisson(mu)

        total = int(y.sum())

        # Store result
        results.append({
            "simulation": s,
            "weather_year": weather_year,
            "exposure_year": exposure_year,
            "growth_factor": growth_factor,
            "model": model_name,
            "total_2025": total
        })

        # Cleanup
        del df_sim, X, weather_matrix, exposure_values
        if s % 50 == 0:
            gc.collect()

    return results

# Run simulations (Poisson only - NB removed due to dispersion ~1)
print("\nRunning Poisson simulations...")
poisson_results = run_random_simulation(poisson_res, use_nb=False, model_name="poisson")

# Combine results (Poisson only)
all_results = poisson_results
mc_df = pd.DataFrame(all_results)

# Save results
mc_path = OUT_DIR / "risk_mc_2025_totals_bike_all_scenarios.parquet"
mc_df.to_parquet(mc_path, index=False)

print("\n" + "="*70)
print("SAVED MONTE CARLO RESULTS (POISSON ONLY)")
print("="*70)
print("Saved:", mc_path)
print(f"Total simulations: {len(mc_df)} rows ({S} simulations)")

# Summary statistics
print("\n" + "="*70)
print("COMPREHENSIVE UNCERTAINTY SUMMARY")
print("="*70)

model_data = mc_df['total_2025'].values
q05, q50, q95 = np.quantile(model_data, [0.05, 0.5, 0.95])
print(f"\nPoisson:")
print(f"  Median: {q50:,.0f}")
print(f"  90% CI: [{q05:,.0f} – {q95:,.0f}]")
print(f"  CI width: {q95 - q05:,.0f} crashes ({(q95 - q05) / q50 * 100:.1f}%)")

# Distribution of sampled dimensions
print("\n" + "="*70)
print("SAMPLED DIMENSION DISTRIBUTIONS")
print("="*70)
print("\nWeather years sampled:")
print(mc_df['weather_year'].value_counts().sort_index())
print("\nExposure years sampled:")
print(mc_df['exposure_year'].value_counts().sort_index())
print("\nGrowth factors (summary):")
print(f"  Min: {mc_df['growth_factor'].min():.3f}")
print(f"  Max: {mc_df['growth_factor'].max():.3f}")
print(f"  Mean: {mc_df['growth_factor'].mean():.3f}")

# Model comparison (Poisson only)
p05, p50, p95 = np.quantile(model_data, [0.05, 0.5, 0.95])

cmp = pd.DataFrame([
    {"model":"poisson", "aic": float(poisson_res.aic),
     "dispersion": float(np.sum(poisson_res.resid_pearson**2)/poisson_res.df_resid),
     "q50_total_2025": float(p50), "q05_total_2025": float(p05), "q95_total_2025": float(p95)},
])
cmp_path = OUT_DIR / "model_comparison_bike_all.parquet"
cmp.to_parquet(cmp_path, index=False)
print("\nSaved:", cmp_path)

# Create summary for dashboard (aggregated view)
# Group by sampled dimensions to show effect of each
print("\n" + "="*70)
print("CREATING DASHBOARD SUMMARY")
print("="*70)

# Summary by exposure year (for dashboard box plot)
scenario_summary = mc_df.groupby(['exposure_year'])['total_2025'].agg([
    ('q05', lambda x: np.quantile(x, 0.05)),
    ('q50', lambda x: np.quantile(x, 0.50)),
    ('q95', lambda x: np.quantile(x, 0.95)),
    ('mean', 'mean'),
    ('std', 'std')
]).reset_index()

# Add columns for dashboard compatibility
scenario_summary['model'] = 'poisson'
scenario_summary['exposure_scenario'] = scenario_summary['exposure_year'].astype(str) + "_random"
scenario_summary['exposure_multiplier'] = 1.0  # Mixed, but we keep for compatibility

scenario_summary_path = OUT_DIR / "risk_exposure_scenarios_summary.parquet"
scenario_summary.to_parquet(scenario_summary_path, index=False)
print("Saved:", scenario_summary_path)

print("\nSummary by Exposure Year (Poisson):")
print(scenario_summary[['exposure_year', 'q05', 'q50', 'q95']])


# In[ ]:





# In[22]:


# ============================================================================
# EVALUATION: Compare predictions against crashes in 2025-ACTIVE cells
# ============================================================================
# CRITICAL: Both prediction and observed must use the SAME cell set!
# We use cells_2025 (cells with 2025 exposure) for both:
# - Prediction: df2025 only contains cells_2025 (set above)
# - Observed: Query filters to cells_2025
#
# This is the correct approach for insurance use case:
# "Given the areas where CitiBike operates in 2025, how many crashes do we expect?"
# ============================================================================

# Total crashes for context (all NYC)
obs_all = con.execute(f"""
SELECT
  SUM(y_bike) AS y_obs_all
FROM read_parquet('{crash_cell_hour_path.as_posix()}')
WHERE hour_ts >= TIMESTAMP '2025-01-01'
  AND hour_ts <  TIMESTAMP '2026-01-01';
""").fetch_df()['y_obs_all'].iloc[0]

# Crashes in training cells (for reference)
obs_train_cells = con.execute(f"""
SELECT SUM(c.y_bike) AS y_obs
FROM read_parquet('{crash_cell_hour_path.as_posix()}') c
INNER JOIN read_parquet('{cells_keep_path.as_posix()}') k USING(cell_id)
WHERE c.hour_ts >= TIMESTAMP '2025-01-01'
  AND c.hour_ts <  TIMESTAMP '2026-01-01';
""").fetch_df()['y_obs'].iloc[0]

# Crashes in 2025-ACTIVE cells - THIS IS WHAT WE EVALUATE AGAINST
obs = con.execute(f"""
SELECT
  date_trunc('month', c.hour_ts) AS month_ts,
  SUM(c.y_bike) AS y_obs
FROM read_parquet('{crash_cell_hour_path.as_posix()}') c
INNER JOIN read_parquet('{cells_2025_path.as_posix()}') k USING(cell_id)
WHERE c.hour_ts >= TIMESTAMP '2025-01-01'
  AND c.hour_ts <  TIMESTAMP '2026-01-01'
GROUP BY 1
ORDER BY 1;
""").fetch_df()

obs_in_cells = obs['y_obs'].sum()

print("\n" + "="*70)
print("OBSERVED CRASHES BREAKDOWN (2025)")
print("="*70)
print(f"Total 2025 crashes (all NYC):           {obs_all:,.0f}")
print(f"Crashes in training cells:              {obs_train_cells:,.0f} ({obs_train_cells/obs_all*100:.1f}%)")
print(f"Crashes in 2025-ACTIVE cells (eval):    {obs_in_cells:,.0f} ({obs_in_cells/obs_all*100:.1f}%)")
print("="*70)
print("✓ Prediction and observed use SAME cell set (2025-active cells)")
print("  This ensures apples-to-apples comparison for insurance use case.")
print("="*70)

# ============================================================================
# Prepare df2025 with ALL features for expected_monthly
# ============================================================================
# df2025 already loaded in previous cell, but needs all derived features

# Apply weather scaling
for col in weather_cols:
    if col in wstats:
        df2025[col] = (df2025[col] - wstats[col]["mean"]) / wstats[col]["std"]

# Apply spatial scaling
df2025["grid_lat_norm"] = (df2025["grid_lat"] - sstats["lat_mean"]) / sstats["lat_std"]
df2025["grid_lng_norm"] = (df2025["grid_lng"] - sstats["lng_mean"]) / sstats["lng_std"]

# Spatial features
df2025["lat2"] = df2025["grid_lat_norm"]**2
df2025["lng2"] = df2025["grid_lng_norm"]**2
df2025["lat_lng"] = df2025["grid_lat_norm"] * df2025["grid_lng_norm"]

print("df2025 prepared with all features")

# ============================================================================
# Expected from grid-model: sum over cells of mu_cellday (DAILY aggregation)
# ============================================================================
def expected_monthly(res, label: str):
    import patsy
    design_info = res.model.data.design_info
    X = patsy.build_design_matrices([design_info], df2025, return_type="dataframe")[0]

    # Align to X's index
    df_aligned = df2025.loc[X.index]

    beta = res.params.loc[X.columns].values
    eta = X.values @ beta

    # Clip for safety
    eta = np.clip(eta, -20, 20)

    # With exposure as FEATURE, μ = exp(Xβ) - no separate exposure term!
    # The exposure effect is INSIDE the design matrix via log1p_exposure
    mu = np.exp(eta)

    # NOTE: Using day_ts instead of hour_ts (daily aggregation)
    tmp = pd.DataFrame({"day_ts": df_aligned["day_ts"].values, "mu": mu})
    tmp["month_ts"] = pd.to_datetime(tmp["day_ts"]).dt.to_period('M').dt.to_timestamp()
    out = tmp.groupby("month_ts", as_index=False)["mu"].sum()
    out = out.rename(columns={"mu": f"y_pred_{label}"})
    return out

pred_p = expected_monthly(poisson_res, "poisson")

# Merge predictions (Poisson only)
eval_wide = obs.merge(pred_p, on="month_ts", how="left")

# Convert to LONG format for dashboard (Poisson only)
eval_df = eval_wide[["month_ts", "y_obs", "y_pred_poisson"]].copy()
eval_df["model"] = "poisson"
eval_df = eval_df.rename(columns={"y_obs": "observed", "y_pred_poisson": "pred_mean"})

# Save
eval_path = OUT_DIR / "risk_eval_2025_monthly_bike_all.parquet"
eval_df.to_parquet(eval_path, index=False)

print("Saved:", eval_path)
print("\nPreview:")
print(eval_df.head(10))

# Show wide format too for inspection
print("\nWide format (for reference):")
print(eval_wide)


# In[ ]:




