#!/usr/bin/env python
"""
run_risk_modeling.py - NYC Bike Crash Risk Modeling Pipeline

This script fits Poisson and Negative Binomial GLMs to predict bike crashes
using CitiBike exposure as an offset. It implements:

1. TEMPORAL SEPARATION: Train on 2020-2024, test on 2025
2. GRID-LEVEL MODELING: 64 spatial cells × hourly resolution
3. UNCERTAINTY QUANTIFICATION: Monte Carlo with parameter + weather bootstrap
4. EXPOSURE SCENARIOS: -10%, actual, +10% sensitivity analysis

INPUTS:
    - data/interim/tripdata_2013_2025_clean.parquet  (from clean_data.py)
    - data/processed/crashes_bike_clean.parquet       (from clean_data.py)
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
           + trend
           + temp + prcp + snow + wspd
           + offset(log(exposure_min))

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
        across the ~64 active grid cells.

    Family: Negative Binomial (accounts for overdispersion)
    Training: 2020-2024 (temporal separation)
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
CRASH_BIKE = PROJECT_ROOT / "data" / "processed" / "crashes_bike_clean.parquet"
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
TMIN = "2020-01-01"
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

# To make grid-model feasible: keep cells covering X% of total exposure (train window).
CELL_COVERAGE = 1.0    # 0.95..0.995 (higher => more cells => slower)

print("DuckDB:", DB_PATH)
print("GRID_DEG:", GRID_DEG, "| FORCE_REBUILD:", FORCE_REBUILD, "| CELL_COVERAGE:", CELL_COVERAGE)
print("Training:", TMIN, "→", TMAX_TRAIN)
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

# Build training exposure (2020-2024 only)
if exposure_train_path.exists():
    print("Exists, skipping:", exposure_train_path)
else:
    # Process year-by-year to manage memory
    print(f"Creating TRAINING exposure (year-by-year) for {TMIN} to {TMAX_TRAIN}...")

    # Extract year range for TRAINING data only
    import datetime
    year_min = datetime.datetime.fromisoformat(TMIN).year
    year_max = datetime.datetime.fromisoformat(TMAX_TRAIN).year - 1  # 2024
    years = range(year_min, year_max + 1)

    temp_files = []

    for year in years:
        year_start = max(f"{year}-01-01", TMIN)  # Don't go before TMIN
        if year == year_max:
            year_end = TMAX_TRAIN  # Use TMAX_TRAIN for last year (2024 → 2025-01-01)
        else:
            year_end = f"{year+1}-01-01"

        temp_path = OUT_DIR / f"_temp_exp_{year}.parquet"
        temp_files.append(temp_path)

        if temp_path.exists():
            print(f"  Year {year}: already exists, skipping")
            continue

        print(f"  Processing year {year} ({year_start} to {year_end})...")

        # Assign ALL exposure to start station cell only
        con.execute(f"""
        COPY (
          WITH trips AS (
            SELECT
              try_cast(started_at AS TIMESTAMP) AS started_at,
              try_cast(ended_at   AS TIMESTAMP) AS ended_at,
              start_lat, start_lng,
              duration_sec
            FROM read_parquet('{TRIPS.as_posix()}')
            WHERE try_cast(started_at AS TIMESTAMP) >= TIMESTAMP '{year_start}'
              AND try_cast(started_at AS TIMESTAMP) <  TIMESTAMP '{year_end}'
              AND start_lat IS NOT NULL AND start_lng IS NOT NULL
              AND ended_at IS NOT NULL
              AND duration_sec IS NOT NULL
              AND duration_sec > 0
              AND duration_sec < 4*60*60
          ),
          binned AS (
            SELECT
              floor(start_lat / {GRID_DEG}) * {GRID_DEG} AS grid_lat,
              floor(start_lng / {GRID_DEG}) * {GRID_DEG} AS grid_lng,
              CAST(floor(start_lat / {GRID_DEG}) * {GRID_DEG} AS VARCHAR) || '_' ||
              CAST(floor(start_lng / {GRID_DEG}) * {GRID_DEG} AS VARCHAR) AS cell_id,
              started_at,
              ended_at
            FROM trips
          ),
          expanded AS (
            SELECT
              cell_id, grid_lat, grid_lng,
              gs AS hour_ts,
              started_at, ended_at
            FROM binned
            CROSS JOIN generate_series(
              date_trunc('hour', started_at),
              date_trunc('hour', ended_at),
              INTERVAL '1 hour'
            ) AS t(gs)
          ),
          overlap AS (
            SELECT
              cell_id, grid_lat, grid_lng, hour_ts,
              GREATEST(
                0,
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

    # Combine all years and filter to strictly < 2025-01-01
    # (trips starting on 2024-12-31 23:xx can generate exposure in 2025-01-01 via generate_series)
    print("  Combining training years (filtering to < 2025-01-01)...")
    all_files = "', '".join([f.as_posix() for f in temp_files if f.exists()])
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

# Build test exposure (2025 only)
if exposure_test_path.exists():
    print("Exists, skipping:", exposure_test_path)
else:
    print(f"Creating TEST exposure for 2025...")

    # Assign ALL exposure to start station cell only (same as training)
    con.execute(f"""
    COPY (
      WITH trips AS (
        SELECT
          try_cast(started_at AS TIMESTAMP) AS started_at,
          try_cast(ended_at   AS TIMESTAMP) AS ended_at,
          start_lat, start_lng,
          duration_sec
        FROM read_parquet('{TRIPS.as_posix()}')
        WHERE try_cast(started_at AS TIMESTAMP) >= TIMESTAMP '2025-01-01'
          AND try_cast(started_at AS TIMESTAMP) <  TIMESTAMP '2026-01-01'
          AND start_lat IS NOT NULL AND start_lng IS NOT NULL
          AND ended_at IS NOT NULL
          AND duration_sec IS NOT NULL
          AND duration_sec > 0
          AND duration_sec < 4*60*60
      ),
      binned AS (
        SELECT
          floor(start_lat / {GRID_DEG}) * {GRID_DEG} AS grid_lat,
          floor(start_lng / {GRID_DEG}) * {GRID_DEG} AS grid_lng,
          CAST(floor(start_lat / {GRID_DEG}) * {GRID_DEG} AS VARCHAR) || '_' ||
          CAST(floor(start_lng / {GRID_DEG}) * {GRID_DEG} AS VARCHAR) AS cell_id,
          started_at,
          ended_at
        FROM trips
      ),
      expanded AS (
        SELECT
          cell_id, grid_lat, grid_lng,
          gs AS hour_ts,
          started_at, ended_at
        FROM binned
        CROSS JOIN generate_series(
          date_trunc('hour', started_at),
          date_trunc('hour', ended_at),
          INTERVAL '1 hour'
        ) AS t(gs)
      ),
      overlap AS (
        SELECT
          cell_id, grid_lat, grid_lng, hour_ts,
          GREATEST(
            0,
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


grid_train_path = OUT_DIR / "grid_train_cell_hour_2020_2024.parquet"
cells_keep_path = OUT_DIR / "grid_cells_keep_2020_2024.parquet"

if FORCE_REBUILD:
    if grid_train_path.exists(): grid_train_path.unlink()
    if cells_keep_path.exists(): cells_keep_path.unlink()

# 1) Determine top exposure cells covering CELL_COVERAGE of total exposure (TRAINING DATA ONLY)
con.execute(f"""
COPY (
  WITH cell_exp AS (
    SELECT
      cell_id,
      SUM(exposure_min) AS exp_sum
    FROM read_parquet('{exposure_train_path.as_posix()}')
    WHERE hour_ts >= TIMESTAMP '2020-01-01'
      AND hour_ts <  TIMESTAMP '2025-01-01'
    GROUP BY 1
  ),
  ranked AS (
    SELECT
      cell_id,
      exp_sum,
      SUM(exp_sum) OVER (ORDER BY exp_sum DESC) AS cum_exp,
      SUM(exp_sum) OVER () AS total_exp
    FROM cell_exp
  )
  SELECT
    cell_id,
    exp_sum,
    cum_exp / NULLIF(total_exp,0) AS cum_share
  FROM ranked
  WHERE cum_exp / NULLIF(total_exp,0) <= {CELL_COVERAGE}
)
TO '{cells_keep_path.as_posix()}'
(FORMAT PARQUET);
""")

print("Wrote:", cells_keep_path)

# 2) Build grid training mart: one row per (cell_id, hour_ts) - TRAINING DATA ONLY
con.execute(f"""
COPY (
  WITH e AS (
    SELECT
      hour_ts, cell_id, grid_lat, grid_lng, exposure_min
    FROM read_parquet('{exposure_train_path.as_posix()}')
    WHERE hour_ts >= TIMESTAMP '2020-01-01'
      AND hour_ts <  TIMESTAMP '2025-01-01'
      AND exposure_min > 0
  ),
  e_keep AS (
    SELECT e.*
    FROM e
    INNER JOIN read_parquet('{cells_keep_path.as_posix()}') k USING(cell_id)
  ),
  c AS (
    SELECT hour_ts, cell_id, y_bike
    FROM read_parquet('{crash_cell_hour_path.as_posix()}')
    WHERE hour_ts >= TIMESTAMP '2020-01-01'
      AND hour_ts <  TIMESTAMP '2025-01-01'
  )
  SELECT
    e_keep.hour_ts,
    e_keep.cell_id,
    e_keep.grid_lat,
    e_keep.grid_lng,
    e_keep.exposure_min,
    COALESCE(c.y_bike, 0) AS y_bike,
    EXTRACT(HOUR  FROM e_keep.hour_ts) AS hour_of_day,
    EXTRACT(DOW   FROM e_keep.hour_ts) AS dow,
    EXTRACT(MONTH FROM e_keep.hour_ts) AS month,
    w.temp, w.prcp, w.snow, w.wspd
  FROM e_keep
  LEFT JOIN c USING(hour_ts, cell_id)
  LEFT JOIN weather_hourly w USING(hour_ts)
)
TO '{grid_train_path.as_posix()}'
(FORMAT PARQUET);
""")

print("Wrote:", grid_train_path)
print(con.execute(f"""
SELECT COUNT(*) n, AVG(y_bike) y_mean, AVG(exposure_min) exp_mean
FROM read_parquet('{grid_train_path.as_posix()}');
""").fetch_df())


# In[16]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

TARGET = "y_bike"
MIN_EXPOSURE_MIN = 50.0
EPS = 1e-6
weather_cols = ["temp","prcp","snow","wspd"]

# Load (grid train) to pandas
train_df = con.execute(f"SELECT * FROM read_parquet('{grid_train_path.as_posix()}')").fetch_df()

# clean
need = ["exposure_min", TARGET, "hour_of_day","dow","month","grid_lat","grid_lng"] + weather_cols
train_df = train_df.dropna(subset=need).copy()
train_df = train_df[train_df["exposure_min"] >= MIN_EXPOSURE_MIN].copy()

# offset
train_df["log_exposure"] = np.log(train_df["exposure_min"].values + EPS)

# categories
train_df["hour_of_day"] = train_df["hour_of_day"].astype(int).astype("category")
train_df["dow"] = train_df["dow"].astype(int).astype("category")
train_df["month"] = train_df["month"].astype(int).astype("category")

# ============================================================================
# TREND FEATURE: Capture long-term changes in crash rate
# ============================================================================
# The crash rate has been declining over time (1.58 in 2020 → 1.01 in 2024).
# Without a trend term, the model learns the average rate and over-predicts.
# We use a linear trend: trend = year - 2020 (so 2020=0, 2021=1, ..., 2025=5)
# This allows extrapolation to 2025 with a single parameter.
# ============================================================================
train_df["hour_ts"] = pd.to_datetime(train_df["hour_ts"])
train_df["year"] = train_df["hour_ts"].dt.year
TREND_BASE_YEAR = 2020
train_df["trend"] = train_df["year"] - TREND_BASE_YEAR

print(f"\nTrend feature added: trend = year - {TREND_BASE_YEAR}")
print(f"Training years: {sorted(train_df['year'].unique())}")
print(f"Trend values: {sorted(train_df['trend'].unique())}")

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

# model formula (GRID MODEL) - use normalized versions
# NOTE: "trend" is included to capture long-term decline in crash rate
rhs = [
    "C(hour_of_day)", "C(dow)", "C(month)",
    "grid_lat_norm", "grid_lng_norm", "lat2", "lng2", "lat_lng",
    "trend"  # Linear trend term: captures year-over-year rate changes
] + active_weather_cols

formula = f"{TARGET} ~ " + " + ".join(rhs)
print("Formula:", formula)

# ---- Poisson GLM ----
poisson_model = smf.glm(
    formula=formula,
    data=train_df,
    family=sm.families.Poisson(),
    offset=train_df["log_exposure"]
)
poisson_res = poisson_model.fit(cov_type="HC0")
print("\n=== POISSON (grid model) ===")
print(poisson_res.summary())

pearson_chi2 = float(np.sum(poisson_res.resid_pearson**2))
disp_poiss = pearson_chi2 / float(poisson_res.df_resid)
print("\nPoisson overdispersion χ²/df:", disp_poiss)

# ---- Negative Binomial GLM (PROPER MLE ESTIMATION) ----
print("\n" + "="*70)
print("FITTING NEGATIVE BINOMIAL WITH MLE ALPHA ESTIMATION")
print("="*70)

# Use NegativeBinomial family WITHOUT specifying alpha
# statsmodels will estimate it via MLE during fitting
nb_model = smf.glm(
    formula=formula,
    data=train_df,
    family=sm.families.NegativeBinomial(),  # No alpha specified!
    offset=train_df["log_exposure"]
)

# Fit with scale='X2' to get proper dispersion parameter
nb_res = nb_model.fit(cov_type="HC0", scale='X2')

print("\n=== NEG BIN (grid model, MLE alpha) ===")
print(nb_res.summary())
print("\nNB alpha (MLE): {:.6f}".format(float(nb_res.scale)))
print("NB dispersion (χ²/df): {:.6f}".format(
    float(np.sum(nb_res.resid_pearson**2) / nb_res.df_resid)
))

# Store the estimated alpha for later use
alpha_mle = float(nb_res.scale)

meta = {
    "GRID_DEG": GRID_DEG,
    "CELL_COVERAGE": CELL_COVERAGE,
    "weather_stats": {k: {"mean": v[0], "std": v[1]} for k,v in weather_stats.items()},
    "spatial_stats": {
        "lat_mean": float(lat_mean),
        "lat_std": float(lat_std),
        "lng_mean": float(lng_mean),
        "lng_std": float(lng_std)
    },
    "trend_base_year": TREND_BASE_YEAR,  # For computing trend = year - base_year
    "formula": formula,
    "active_weather_cols": active_weather_cols,
    "alpha_mle": alpha_mle  # Save MLE alpha for reproducibility
}
meta_path = OUT_DIR / "model_meta_bike_all.json"
meta_path.write_text(json.dumps(meta, indent=2))
print("\nSaved:", meta_path)


# In[18]:


grid_2025_path = OUT_DIR / "grid_2025_cell_hour_bike_all.parquet"

if FORCE_REBUILD and grid_2025_path.exists():
    grid_2025_path.unlink()

con.execute(f"""
COPY (
  WITH e AS (
    SELECT
      hour_ts, cell_id, grid_lat, grid_lng, exposure_min
    FROM read_parquet('{exposure_test_path.as_posix()}')
    WHERE hour_ts >= TIMESTAMP '2025-01-01'
      AND hour_ts <  TIMESTAMP '2026-01-01'
      AND exposure_min > 0
  ),
  e_keep AS (
    SELECT e.*
    FROM e
    INNER JOIN read_parquet('{cells_keep_path.as_posix()}') k USING(cell_id)
  )
  SELECT
    e_keep.hour_ts,
    e_keep.cell_id,
    e_keep.grid_lat,
    e_keep.grid_lng,
    e_keep.exposure_min,
    EXTRACT(HOUR  FROM e_keep.hour_ts) AS hour_of_day,
    EXTRACT(DOW   FROM e_keep.hour_ts) AS dow,
    EXTRACT(MONTH FROM e_keep.hour_ts) AS month,
    w.temp, w.prcp, w.snow, w.wspd
  FROM e_keep
  LEFT JOIN weather_hourly w USING(hour_ts)
  ORDER BY hour_ts, cell_id
)
TO '{grid_2025_path.as_posix()}'
(FORMAT PARQUET);
""")

print("Wrote:", grid_2025_path)
print(con.execute(f"SELECT COUNT(*) n FROM read_parquet('{grid_2025_path.as_posix()}')").fetch_df())


# In[ ]:


import patsy
from numpy.random import default_rng

rng = default_rng(42)
S = 50  # Monte Carlo simulations (reduced for speed; increase for production)

# ============================================================================
# STEP 1: Load Weather for ALL Years (by day-of-year + hour)
# ============================================================================
print("="*70)
print("PREPARING YEAR-LEVEL WEATHER BOOTSTRAP (day-of-year mapping)")
print("="*70)

# Load training weather
train_weather = con.execute(f"SELECT * FROM read_parquet('{grid_train_path.as_posix()}')").fetch_df()
train_weather["hour_ts"] = pd.to_datetime(train_weather["hour_ts"])
train_weather["year"] = train_weather["hour_ts"].dt.year
train_weather["dayofyear"] = train_weather["hour_ts"].dt.dayofyear
train_weather["hour"] = train_weather["hour_ts"].dt.hour

weather_cols = ["temp", "prcp", "snow", "wspd"]
available_years = [2020, 2021, 2022, 2023, 2024]

# Build lookup: year → (dayofyear, hour) → weather
year_weather_dict = {}

for year in available_years:
    year_data = train_weather[train_weather['year'] == year].copy()

    # Aggregate to (dayofyear, hour) - take median if multiple observations
    year_agg = year_data.groupby(['dayofyear', 'hour'])[weather_cols].median().reset_index()

    # Convert to dict
    year_dict = {}
    for _, row in year_agg.iterrows():
        key = (int(row['dayofyear']), int(row['hour']))
        year_dict[key] = {col: row[col] for col in weather_cols}

    year_weather_dict[year] = year_dict
    print(f"  Year {year}: {len(year_dict)} unique (dayofyear, hour) combinations")

# Check coverage
test_key = (15, 14)  # 15th day of year, 14:00
print(f"\nDay 15, hour 14 across years:")
for year in available_years:
    if test_key in year_weather_dict[year]:
        w = year_weather_dict[year][test_key]
        print(f"  {year}: temp={w['temp']:+.1f}°C, prcp={w['prcp']:.1f}mm")

# ============================================================================
# STEP 2: Prepare 2025 Data
# ============================================================================
df2025 = con.execute(f"SELECT * FROM read_parquet('{grid_2025_path.as_posix()}')").fetch_df()
df2025 = df2025.dropna(subset=["exposure_min","hour_of_day","dow","month","grid_lat","grid_lng"]).copy()
df2025 = df2025[df2025["exposure_min"] > 0].copy()

df2025["hour_ts"] = pd.to_datetime(df2025["hour_ts"])
df2025["dayofyear"] = df2025["hour_ts"].dt.dayofyear.astype(int)
df2025["hour"] = df2025["hour_ts"].dt.hour.astype(int)

df2025["hour_of_day"] = df2025["hour_of_day"].astype(int).astype("category")
df2025["dow"] = df2025["dow"].astype(int).astype("category")
df2025["month"] = df2025["month"].astype(int).astype("category")

# Add trend feature for 2025 (year=2025, so trend = 2025 - 2020 = 5)
df2025["year"] = df2025["hour_ts"].dt.year
df2025["trend"] = df2025["year"] - TREND_BASE_YEAR
print(f"2025 trend value: {df2025['trend'].iloc[0]} (year {df2025['year'].iloc[0]} - {TREND_BASE_YEAR})")

if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    wstats = meta["weather_stats"]
    sstats = meta["spatial_stats"]
else:
    raise ValueError("meta_path not found!")

print(f"\nPrepared 2025 data: {len(df2025)} rows")

# Pre-extract keys
df2025_keys = list(zip(df2025['dayofyear'], df2025['hour']))

# ============================================================================
# STEP 3: Simulation with Day-of-Year Weather Mapping + Exposure Scenarios
# ============================================================================
def simulate_totals_year_weather(res, use_nb: bool, exposure_multiplier: float = 1.0):
    """
    For each simulation: Pick ONE year and map 2025 dates to that year's weather

    Args:
        res: Model fit result (poisson or negbin)
        use_nb: If True, use Negative Binomial; else Poisson
        exposure_multiplier: Multiplier for 2025 exposure (e.g., 0.9, 1.0, 1.1)

    NOTE: This weather bootstrapping approach preserves temporal correlation within
    years but samples across years to capture inter-annual variability. While this
    produces narrower confidence intervals than fully independent sampling, it is
    more realistic as weather patterns are temporally autocorrelated.
    """
    design_info = res.model.data.design_info

    # CRITICAL FIX: Use MLE alpha from model fit, not heuristic
    if use_nb:
        # Get alpha from the fitted model's scale parameter
        alpha_nb = float(res.scale)
        print(f"  Using NegBin alpha (MLE): {alpha_nb:.6f}, Exposure multiplier: {exposure_multiplier:.2f}")
    else:
        alpha_nb = 0.0
        print(f"  Exposure multiplier: {exposure_multiplier:.2f}")

    totals = np.zeros(S, dtype=np.int64)

    for s in range(S):
        if s % 50 == 0:
            print(f"  Simulation {s}/{S}...")

        # ----------------------------------------------------------------
        # A) Pick ONE year's weather
        # ----------------------------------------------------------------
        sampled_year = rng.choice(available_years)
        year_weather = year_weather_dict[sampled_year]

        # ----------------------------------------------------------------
        # B) Map 2025 (dayofyear, hour) → sampled year weather
        # ----------------------------------------------------------------
        weather_matrix = np.zeros((len(df2025), 4))

        for i, key in enumerate(df2025_keys):
            if key in year_weather:
                w = year_weather[key]
                weather_matrix[i] = [w['temp'], w['prcp'], w['snow'], w['wspd']]
            # else: keep 0 (missing weather)

        # Apply to df_sim
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

        # ----------------------------------------------------------------
        # C) Sample β
        # ----------------------------------------------------------------
        X = patsy.build_design_matrices([design_info], df_sim, return_type="dataframe")[0]
        xcols = X.columns.tolist()

        beta_hat = res.params.loc[xcols].values
        cov_hat = res.cov_params().loc[xcols, xcols].values
        beta_s = rng.multivariate_normal(mean=beta_hat, cov=cov_hat)

        # ----------------------------------------------------------------
        # D) Calculate μ and Sample Crashes (with exposure scenario)
        # ----------------------------------------------------------------
        df_aligned = df_sim.loc[X.index]
        E = df_aligned["exposure_min"].values * exposure_multiplier  # Apply exposure scenario

        eta = X.values @ beta_s
        eta = np.clip(eta, -20, 20)
        mu = E * np.exp(eta)
        mu = np.clip(mu, 0, 1e6)

        if use_nb and alpha_nb > 0:
            shape = 1.0 / alpha_nb
            lam = rng.gamma(shape=shape, scale=alpha_nb * mu)
            lam = np.clip(lam, 0, 1e6)
            y = rng.poisson(lam)
        else:
            y = rng.poisson(mu)

        totals[s] = int(y.sum())

    return totals

# ============================================================================
# STEP 4: Run Simulations with Exposure Scenarios
# ============================================================================
print("\n" + "="*70)
print("RUNNING YEAR-LEVEL WEATHER BOOTSTRAP WITH EXPOSURE SCENARIOS")
print("="*70)

# Exposure scenarios: -10%, actual (0%), +10%
exposure_scenarios = [0.9, 1.0, 1.1]
scenario_labels = ["-10%", "actual", "+10%"]

# Store all results
all_results = []

for exp_mult, exp_label in zip(exposure_scenarios, scenario_labels):
    print(f"\n{'='*70}")
    print(f"EXPOSURE SCENARIO: {exp_label} (multiplier = {exp_mult:.2f})")
    print(f"{'='*70}")

    print("\nModel: Poisson")
    tot_p = simulate_totals_year_weather(poisson_res, use_nb=False, exposure_multiplier=exp_mult)

    print("\nModel: Negative Binomial (MLE alpha)")
    tot_n = simulate_totals_year_weather(nb_res, use_nb=True, exposure_multiplier=exp_mult)

    def summarize(arr):
        q05, q50, q95 = np.quantile(arr, [0.05, 0.5, 0.95])
        return float(q05), float(q50), float(q95)

    p05, p50, p95 = summarize(tot_p)
    n05, n50, n95 = summarize(tot_n)

    print("\n" + "="*70)
    print(f"RESULTS FOR EXPOSURE {exp_label}")
    print("="*70)
    print(f"Poisson: 5th = {p05:.0f}, Median = {p50:.0f}, 95th = {p95:.0f}")
    print(f"NegBin:  5th = {n05:.0f}, Median = {n50:.0f}, 95th = {n95:.0f}")
    print(f"\nPoisson 90% CI: [{p05:.0f}, {p95:.0f}], width = {p95-p05:.0f} ({(p95-p05)/p50*100:.1f}%)")
    print(f"NegBin  90% CI: [{n05:.0f}, {n95:.0f}], width = {n95-n05:.0f} ({(n95-n05)/n50*100:.1f}%)")

    # Store results for dashboard output
    for s_idx in range(S):
        all_results.append({
            "exposure_scenario": exp_label,
            "exposure_multiplier": exp_mult,
            "model": "poisson",
            "simulation": s_idx,
            "total_2025": int(tot_p[s_idx])
        })
        all_results.append({
            "exposure_scenario": exp_label,
            "exposure_multiplier": exp_mult,
            "model": "neg_bin",
            "simulation": s_idx,
            "total_2025": int(tot_n[s_idx])
        })


# ---- Save with Exposure Scenarios ----
mc_df = pd.DataFrame(all_results)
mc_path = OUT_DIR / "risk_mc_2025_totals_bike_all_scenarios.parquet"
mc_df.to_parquet(mc_path, index=False)
print("\n" + "="*70)
print("SAVED MONTE CARLO RESULTS WITH EXPOSURE SCENARIOS")
print("="*70)
print("Saved:", mc_path)
print(f"Total simulations: {len(mc_df)} rows ({S} per model × 2 models × {len(exposure_scenarios)} scenarios)")
print("\nSample:")
print(mc_df.groupby(['exposure_scenario', 'model'])['total_2025'].describe())

# Model comparison with baseline scenario (actual exposure)
baseline_df = mc_df[mc_df['exposure_scenario'] == 'actual'].copy()
poisson_baseline = baseline_df[baseline_df['model'] == 'poisson']['total_2025'].values
negbin_baseline = baseline_df[baseline_df['model'] == 'neg_bin']['total_2025'].values

p05_base, p50_base, p95_base = np.quantile(poisson_baseline, [0.05, 0.5, 0.95])
n05_base, n50_base, n95_base = np.quantile(negbin_baseline, [0.05, 0.5, 0.95])

cmp = pd.DataFrame([
    {"model":"poisson", "aic": float(poisson_res.aic), "dispersion": float(np.sum(poisson_res.resid_pearson**2)/poisson_res.df_resid),
     "q50_total_2025": float(p50_base), "q05_total_2025": float(p05_base), "q95_total_2025": float(p95_base)},
    {"model":"neg_bin", "aic": float(nb_res.aic), "dispersion": float(np.sum(nb_res.resid_pearson**2)/nb_res.df_resid),
     "alpha_mle": float(nb_res.scale),
     "q50_total_2025": float(n50_base), "q05_total_2025": float(n05_base), "q95_total_2025": float(n95_base)},
])
cmp_path = OUT_DIR / "model_comparison_bike_all.parquet"
cmp.to_parquet(cmp_path, index=False)
print("Saved:", cmp_path)

cmp


# ============================================================================
# STEP 5: Create Exposure Scenario Summary for Dashboard
# ============================================================================
print("\n" + "="*70)
print("CREATING EXPOSURE SCENARIO SUMMARY FOR DASHBOARD")
print("="*70)

# Aggregate by exposure scenario and model
scenario_summary = mc_df.groupby(['exposure_scenario', 'exposure_multiplier', 'model'])['total_2025'].agg([
    ('q05', lambda x: np.quantile(x, 0.05)),
    ('q50', lambda x: np.quantile(x, 0.50)),
    ('q95', lambda x: np.quantile(x, 0.95)),
    ('mean', 'mean'),
    ('std', 'std')
]).reset_index()

scenario_summary_path = OUT_DIR / "risk_exposure_scenarios_summary.parquet"
scenario_summary.to_parquet(scenario_summary_path, index=False)
print("Saved:", scenario_summary_path)
print("\nScenario Summary:")
print(scenario_summary)


# In[ ]:





# In[22]:


# ============================================================================
# CRITICAL FIX: Observed crashes 2025 - ONLY IN MODEL CELLS
# ============================================================================
# The model can only predict for cells with CitiBike exposure.
# Therefore, we must compare predictions against observed crashes
# ONLY in those same cells. Otherwise we're comparing apples to oranges.
#
# Staten Island and other areas without CitiBike have crashes but
# the model has E=0 there, so it would predict 0 crashes.
# Including those in "observed" would inflate the observed count unfairly.
# ============================================================================

# First, get total crashes for context
obs_all = con.execute(f"""
SELECT
  SUM(y_bike) AS y_obs_all
FROM read_parquet('{crash_cell_hour_path.as_posix()}')
WHERE hour_ts >= TIMESTAMP '2025-01-01'
  AND hour_ts <  TIMESTAMP '2026-01-01';
""").fetch_df()['y_obs_all'].iloc[0]

# Now get crashes ONLY in model cells (cells with exposure)
obs = con.execute(f"""
SELECT
  date_trunc('month', c.hour_ts) AS month_ts,
  SUM(c.y_bike) AS y_obs
FROM read_parquet('{crash_cell_hour_path.as_posix()}') c
INNER JOIN read_parquet('{cells_keep_path.as_posix()}') k USING(cell_id)
WHERE c.hour_ts >= TIMESTAMP '2025-01-01'
  AND c.hour_ts <  TIMESTAMP '2026-01-01'
GROUP BY 1
ORDER BY 1;
""").fetch_df()

obs_in_cells = obs['y_obs'].sum()

print("\n" + "="*70)
print("OBSERVED CRASHES BREAKDOWN (2025)")
print("="*70)
print(f"Total 2025 crashes (all areas):     {obs_all:,.0f}")
print(f"Crashes IN model cells (with E):    {obs_in_cells:,.0f} ({obs_in_cells/obs_all*100:.1f}%)")
print(f"Crashes OUTSIDE model cells (E=0):  {obs_all - obs_in_cells:,.0f} ({(obs_all - obs_in_cells)/obs_all*100:.1f}%)")
print("="*70)
print("⚠️  Model predictions are compared against IN-CELL crashes only!")
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
# Expected from grid-model: sum over cells of mu_cellhour
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
    
    mu = df_aligned["exposure_min"].values * np.exp(eta)
    
    tmp = pd.DataFrame({"hour_ts": df_aligned["hour_ts"].values, "mu": mu})
    tmp["month_ts"] = pd.to_datetime(tmp["hour_ts"]).dt.to_period('M').dt.to_timestamp()
    out = tmp.groupby("month_ts", as_index=False)["mu"].sum()
    out = out.rename(columns={"mu": f"y_pred_{label}"})
    return out

pred_p = expected_monthly(poisson_res, "poisson")
pred_n = expected_monthly(nb_res, "negbin")

# Merge predictions (wide format first)
eval_wide = obs.merge(pred_p, on="month_ts", how="left").merge(pred_n, on="month_ts", how="left")

# Convert to LONG format for dashboard
eval_long_poisson = eval_wide[["month_ts", "y_obs", "y_pred_poisson"]].copy()
eval_long_poisson["model"] = "poisson"
eval_long_poisson = eval_long_poisson.rename(columns={"y_obs": "observed", "y_pred_poisson": "pred_mean"})

eval_long_nb = eval_wide[["month_ts", "y_obs", "y_pred_negbin"]].copy()
eval_long_nb["model"] = "neg_bin"
eval_long_nb = eval_long_nb.rename(columns={"y_obs": "observed", "y_pred_negbin": "pred_mean"})

# Combine
eval_df = pd.concat([eval_long_poisson, eval_long_nb], ignore_index=True)

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




