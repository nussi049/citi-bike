#!/usr/bin/env python
"""
proxy_validation.py - Validate CitiBike as Proxy for Total Cycling Exposure

This script tests whether CitiBike trip data is a valid proxy for total cycling
activity in NYC by comparing it against official bike counter measurements.

METHODOLOGY:
    The key assumption in our risk model is that CitiBike exposure correlates
    with total cycling activity. We validate this at the Borough × Month level
    by computing:

        r = corr(log(CitiBike_exposure), log(Counter_total))

    A high correlation (r > 0.8) suggests CitiBike captures temporal variation
    in cycling activity, supporting its use as an exposure proxy.

    NOTE: This validates TEMPORAL correlation, not spatial distribution.
    We cannot validate whether CitiBike accurately represents the spatial
    distribution of cyclists within a borough.

INPUTS:
    - data/interim/tripdata_2013_2025_clean.parquet     (cleaned CitiBike trips)
    - data/raw/borough_boundaries.geojson               (NYC borough polygons)
    - data/processed/nyc_bike_counters/bike_counts_monthly_by_borough.parquet

OUTPUTS:
    - data/processed/proxy_test/proxy_test_borough_month.parquet
      Contains: month_ts, borough, citi_trips, citi_exposure_min,
                counter_bike_count, log_citi, log_cnt, share_idx

Usage:
    python -m src.processing.proxy_validation
    # or via Makefile:
    make proxy-test
"""

from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import scipy.stats as st

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Input files
TRIPS_PATH = PROJECT_ROOT / "data" / "interim" / "tripdata_2013_2025_clean.parquet"
BOROUGH_GEOJSON = RAW_DIR / "borough_boundaries.geojson"
COUNTER_MONTHLY_PATH = PROCESSED_DIR / "nyc_bike_counters" / "bike_counts_monthly_by_borough.parquet"

# Output files
OUT_DIR = PROCESSED_DIR / "proxy_test"
PROXY_DATASET_PATH = OUT_DIR / "proxy_test_borough_month.parquet"

# Validation window (should match training period)
START = "2020-01-01"
END_EXCL = "2024-12-31"

# Grid size for spatial aggregation (degrees, ~2.5km)
GRID_DEG = 0.025


def setup_database() -> duckdb.DuckDBPyConnection:
    """
    Initialize DuckDB connection with spatial extension.

    Returns:
        Configured DuckDB connection
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=str(OUT_DIR / "proxy_test.duckdb"))
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='2GB'")
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")

    return con


def load_borough_polygons(con: duckdb.DuckDBPyConnection) -> None:
    """
    Load NYC borough boundaries from GeoJSON into DuckDB.

    The borough name column varies across GeoJSON sources, so we try
    multiple common column names.

    Args:
        con: DuckDB connection
    """
    print("Loading borough boundaries...")

    con.execute("DROP TABLE IF EXISTS boroughs")
    con.execute(f"""
    CREATE TABLE boroughs AS
    WITH root AS (
      SELECT * FROM read_json('{BOROUGH_GEOJSON.as_posix()}')
    ),
    feat AS (
      SELECT unnest(features) AS feature FROM root
    )
    SELECT
      -- Try multiple common borough column names
      COALESCE(
        json_extract_string(feature, '$.properties.boro_name'),
        json_extract_string(feature, '$.properties.BoroName'),
        json_extract_string(feature, '$.properties.borough'),
        json_extract_string(feature, '$.properties.boroname'),
        json_extract_string(feature, '$.properties.name')
      ) AS borough,
      ST_GeomFromGeoJSON(json_extract(feature, '$.geometry')) AS geom
    FROM feat
    WHERE json_extract(feature, '$.geometry') IS NOT NULL;
    """)

    boroughs = con.execute("SELECT borough FROM boroughs").fetch_df()
    print(f"  Loaded {len(boroughs)} boroughs: {', '.join(boroughs['borough'])}")


def aggregate_citibike_by_borough(con: duckdb.DuckDBPyConnection) -> None:
    """
    Aggregate CitiBike trips to Borough × Month level.

    Process:
        1. Bin trips into grid cells
        2. Spatial join cells to boroughs
        3. Aggregate to borough-month level

    Args:
        con: DuckDB connection
    """
    print(f"\nAggregating CitiBike trips ({START} to {END_EXCL})...")

    con.execute("DROP TABLE IF EXISTS citi_monthly")
    con.execute(f"""
    CREATE TABLE citi_monthly AS
    WITH trips AS (
      SELECT
        date_trunc('month', try_cast(started_at AS TIMESTAMP)) AS month_ts,
        floor(try_cast(start_lat AS DOUBLE) / {GRID_DEG}) * {GRID_DEG} AS glat,
        floor(try_cast(start_lng AS DOUBLE) / {GRID_DEG}) * {GRID_DEG} AS glon,
        try_cast(duration_sec AS DOUBLE)/60.0 AS exposure_min
      FROM read_parquet('{TRIPS_PATH.as_posix()}')
      WHERE try_cast(started_at AS TIMESTAMP) >= TIMESTAMP '{START}'
        AND try_cast(started_at AS TIMESTAMP) <  TIMESTAMP '{END_EXCL}'
        AND start_lat IS NOT NULL AND start_lng IS NOT NULL
        AND duration_sec IS NOT NULL
        AND duration_sec > 0 AND duration_sec < 4*60*60
    ),
    -- Aggregate to grid cells first (reduces spatial join operations)
    cell_month AS (
      SELECT
        month_ts, glat, glon,
        COUNT(*) AS citi_trips,
        SUM(exposure_min) AS citi_exposure_min
      FROM trips
      GROUP BY 1,2,3
    ),
    -- Spatial join: assign each cell to a borough
    cell_boro AS (
      SELECT
        cm.*,
        COALESCE(b.borough, 'UNKNOWN') AS borough
      FROM cell_month cm
      LEFT JOIN boroughs b
        ON ST_Contains(b.geom, ST_Point(cm.glon, cm.glat))
    )
    -- Final aggregation to borough-month
    SELECT
      month_ts,
      borough,
      SUM(citi_trips) AS citi_trips,
      SUM(citi_exposure_min) AS citi_exposure_min
    FROM cell_boro
    WHERE borough <> 'UNKNOWN'
    GROUP BY 1,2
    ORDER BY 1,2;
    """)

    stats = con.execute("""
        SELECT COUNT(*) as n_rows, COUNT(DISTINCT borough) as n_boroughs,
               MIN(month_ts) as min_month, MAX(month_ts) as max_month
        FROM citi_monthly
    """).fetch_df()

    print(f"  Created {int(stats['n_rows'].iloc[0])} borough-month records")
    print(f"  Boroughs: {int(stats['n_boroughs'].iloc[0])}")
    print(f"  Period: {stats['min_month'].iloc[0]} to {stats['max_month'].iloc[0]}")


def join_with_counter_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Join CitiBike aggregates with official bike counter data.

    Args:
        con: DuckDB connection

    Returns:
        DataFrame with both CitiBike and counter data by borough-month
    """
    print("\nJoining with bike counter data...")

    # Load counter data
    con.execute("DROP TABLE IF EXISTS counter_monthly")
    con.execute(f"""
    CREATE TABLE counter_monthly AS
    SELECT
      try_cast(month_ts AS TIMESTAMP) AS month_ts,
      upper(trim(borough)) AS borough,
      try_cast(bike_count AS DOUBLE) AS counter_bike_count
    FROM read_parquet('{COUNTER_MONTHLY_PATH.as_posix()}')
    WHERE month_ts IS NOT NULL;
    """)

    # Normalize CitiBike borough names
    con.execute("DROP TABLE IF EXISTS citi_monthly_norm")
    con.execute("""
    CREATE TABLE citi_monthly_norm AS
    SELECT
      month_ts,
      upper(trim(borough)) AS borough,
      citi_trips,
      citi_exposure_min
    FROM citi_monthly;
    """)

    # Join datasets
    con.execute("DROP TABLE IF EXISTS proxy_bm")
    con.execute("""
    CREATE TABLE proxy_bm AS
    SELECT
      c.month_ts,
      c.borough,
      c.citi_trips,
      c.citi_exposure_min,
      m.counter_bike_count
    FROM citi_monthly_norm c
    LEFT JOIN counter_monthly m
      USING(month_ts, borough)
    WHERE c.borough <> 'UNKNOWN';
    """)

    df = con.execute("SELECT * FROM proxy_bm ORDER BY month_ts, borough").fetch_df()

    n_with_counter = df['counter_bike_count'].notna().sum()
    print(f"  Total records: {len(df)}")
    print(f"  With counter data: {n_with_counter} ({n_with_counter/len(df)*100:.1f}%)")

    return df


def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log-space correlation between CitiBike and counter data.

    The correlation is computed in log-space because:
        - Both variables span multiple orders of magnitude
        - Log-transformation makes the relationship approximately linear
        - Proportional changes matter more than absolute changes

    Args:
        df: DataFrame with citi_exposure_min and counter_bike_count

    Returns:
        DataFrame with additional log columns and share_idx
    """
    print("\n" + "=" * 70)
    print("PROXY VALIDATION RESULTS")
    print("=" * 70)

    # Filter to records with both data sources
    tmp = df.dropna(subset=["counter_bike_count"]).copy()
    tmp = tmp[(tmp["citi_exposure_min"] > 0) & (tmp["counter_bike_count"] > 0)].copy()

    print(f"\nData availability:")
    print(f"  Total borough-months: {len(df)}")
    print(f"  With counter data: {len(tmp)}")

    if len(tmp) < 2:
        print("\nWARNING: Insufficient data for correlation analysis!")
        print("Need at least 2 data points.")
        return pd.DataFrame()

    # Compute log values
    tmp["log_citi"] = np.log(tmp["citi_exposure_min"])
    tmp["log_cnt"] = np.log(tmp["counter_bike_count"])

    # Correlation analysis
    pearson = st.pearsonr(tmp["log_citi"], tmp["log_cnt"])
    spearman = st.spearmanr(tmp["log_citi"], tmp["log_cnt"])

    print(f"\nCorrelation (log-space):")
    print(f"  Pearson r  = {pearson.statistic:.4f} (p = {pearson.pvalue:.2e})")
    print(f"  Spearman ρ = {spearman.statistic:.4f} (p = {spearman.pvalue:.2e})")

    # Share index: ratio of CitiBike to counter (proxy market share)
    tmp["share_idx"] = tmp["citi_exposure_min"] / tmp["counter_bike_count"]

    # Interpretation
    print(f"\nInterpretation:")
    if pearson.statistic > 0.8:
        print(f"  ✓ Strong correlation (r = {pearson.statistic:.2f})")
        print(f"    CitiBike is a VALID temporal proxy for cycling activity.")
    elif pearson.statistic > 0.6:
        print(f"  ~ Moderate correlation (r = {pearson.statistic:.2f})")
        print(f"    CitiBike provides REASONABLE temporal proxy with some noise.")
    else:
        print(f"  ✗ Weak correlation (r = {pearson.statistic:.2f})")
        print(f"    CitiBike may NOT be a reliable proxy for total cycling.")

    # Stability by borough
    print(f"\nShare index stability by borough:")
    stability = (
        tmp.groupby("borough")["share_idx"]
           .agg(["count", "mean", "std"])
           .assign(cv=lambda x: x["std"] / x["mean"])
           .sort_values("cv")
    )
    print(stability.to_string())

    return tmp


def save_output(df: pd.DataFrame) -> None:
    """
    Save proxy validation dataset for dashboard.

    Args:
        df: Validated DataFrame with correlation results
    """
    if df.empty:
        # Create empty output to prevent downstream failures
        out = pd.DataFrame(columns=[
            "month_ts", "borough", "citi_trips", "citi_exposure_min",
            "counter_bike_count", "log_citi", "log_cnt", "share_idx"
        ])
        out.to_parquet(PROXY_DATASET_PATH, index=False)
        print(f"\nWrote empty dataset: {PROXY_DATASET_PATH}")
        return

    out = df[[
        "month_ts", "borough", "citi_trips", "citi_exposure_min",
        "counter_bike_count", "log_citi", "log_cnt", "share_idx"
    ]].copy()

    out.to_parquet(PROXY_DATASET_PATH, index=False)

    print(f"\nSaved: {PROXY_DATASET_PATH}")
    print(f"  Records: {len(out)}")
    print(f"  Size: {PROXY_DATASET_PATH.stat().st_size / 1024:.1f} KB")


def main():
    """Main entry point for proxy validation pipeline."""

    print("=" * 70)
    print("CITIBIKE PROXY VALIDATION")
    print("=" * 70)
    print(f"Testing whether CitiBike correlates with total cycling activity")
    print(f"Validation period: {START} to {END_EXCL}")
    print(f"Aggregation level: Borough × Month")
    print("=" * 70)

    # Initialize
    con = setup_database()

    # Load spatial data
    load_borough_polygons(con)

    # Aggregate CitiBike data
    aggregate_citibike_by_borough(con)

    # Join with counter data
    df = join_with_counter_data(con)

    # Compute correlation
    result = compute_correlation(df)

    # Save output
    save_output(result)

    print("\n" + "=" * 70)
    print("PROXY VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
