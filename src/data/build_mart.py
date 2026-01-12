#!/usr/bin/env python
"""
build_mart.py - Build Dashboard Data Marts

This script creates aggregated data marts optimized for dashboard visualization.
It pre-computes expensive aggregations so the dashboard loads quickly.

INPUTS:
    - data/interim/tripdata_2013_2025_clean.parquet  (cleaned CitiBike trips)
    - data/raw/borough_boundaries.geojson            (NYC borough polygons)

OUTPUTS:
    - data/processed/dashboard_marts/trips_borough_hour.parquet
      Contains: hour_ts, borough, member_casual, rideable_type, n_trips, exposure_min

METHODOLOGY:
    Instead of doing expensive spatial joins on every trip (300M+ records),
    we first extract unique stations (~5000), do spatial join only on stations,
    then aggregate trips by joining on station_id. This is ~60,000x faster.

    Process:
        1. Extract unique stations with their coordinates
        2. Spatial join: station → borough (only ~5000 operations)
        3. Regular join: trips → station_borough
        4. Aggregate to hour × borough × member_type × bike_type

Usage:
    python -m src.data.build_mart
    # or via Makefile:
    make mart
"""

from __future__ import annotations

import duckdb
from pathlib import Path
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input files
TRIPS = PROJECT_ROOT / "data" / "interim" / "tripdata_2013_2025_clean.parquet"
GEOJSON = PROJECT_ROOT / "data" / "raw" / "borough_boundaries.geojson"

# Output files
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "dashboard_marts"
OUT = OUT_DIR / "trips_borough_hour.parquet"

# Time window (all available data)
START = "2013-01-01"
END = "2026-01-01"


def find_borough_column(con: duckdb.DuckDBPyConnection) -> str:
    """
    Find the borough name column in the GeoJSON file.

    Different GeoJSON sources use different column names for borough.
    This function tries common variants.

    Args:
        con: DuckDB connection

    Returns:
        Column name containing borough names

    Raises:
        RuntimeError: If no borough column is found
    """
    geo_cols = con.execute(
        f"DESCRIBE SELECT * FROM st_read('{GEOJSON.as_posix()}')"
    ).fetch_df()["column_name"].tolist()

    candidates = ["boro_name", "boroname", "borough", "name", "boro_nm"]
    for col in candidates:
        if col in geo_cols:
            return col

    raise RuntimeError(f"No borough column found. Available: {geo_cols}")


def main():
    """Build the trips_borough_hour data mart."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize DuckDB with spatial extension
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")

    print("=" * 70)
    print("BUILDING DASHBOARD MART: trips_borough_hour.parquet")
    print("=" * 70)
    print(f"Strategy: Station-based spatial join (fast)")
    print(f"Period: {START} to {END}")

    start_time = time.time()

    # Step 1: Find borough column name
    print("\n[1/6] Inspecting GeoJSON...")
    bcol = find_borough_column(con)
    print(f"  Borough column: '{bcol}'")

    # Step 2: Load trips into view
    print("\n[2/6] Loading trips...")
    step_time = time.time()

    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW trips AS
    SELECT
      try_cast(started_at AS TIMESTAMP) AS started_ts,
      start_station_id,
      start_lat, start_lng,
      rideable_type,
      member_casual,
      duration_sec
    FROM read_parquet('{TRIPS.as_posix()}')
    WHERE try_cast(started_at AS TIMESTAMP) >= TIMESTAMP '{START}'
      AND try_cast(started_at AS TIMESTAMP) <  TIMESTAMP '{END}'
      AND start_station_id IS NOT NULL
      AND start_lat IS NOT NULL
      AND start_lng IS NOT NULL
      AND duration_sec IS NOT NULL
      AND duration_sec > 0
      AND duration_sec < 4*60*60
    """)

    trip_count = con.execute("SELECT COUNT(*) as n FROM trips").fetchdf()['n'].iloc[0]
    print(f"  Loaded {int(trip_count):,} trips ({time.time() - step_time:.1f}s)")

    # Step 3: Extract unique stations
    # KEY OPTIMIZATION: Only ~5000 stations vs 300M trips
    print("\n[3/6] Extracting unique stations...")
    step_time = time.time()

    con.execute("""
    CREATE OR REPLACE TEMP TABLE stations AS
    SELECT
      start_station_id AS station_id,
      any_value(start_lat) AS lat,
      any_value(start_lng) AS lng
    FROM trips
    GROUP BY 1
    """)

    station_count = con.execute("SELECT COUNT(*) as n FROM stations").fetchdf()['n'].iloc[0]
    print(f"  Found {int(station_count):,} unique stations ({time.time() - step_time:.1f}s)")

    # Step 4: Load borough polygons
    print("\n[4/6] Loading borough boundaries...")
    step_time = time.time()

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE borough_poly AS
    SELECT
      upper({bcol}) AS borough,
      geom
    FROM st_read('{GEOJSON.as_posix()}')
    WHERE geom IS NOT NULL
    """)
    print(f"  Loaded borough polygons ({time.time() - step_time:.1f}s)")

    # Step 5: Spatial join stations → boroughs
    # This is the key optimization: only ~5000 spatial operations instead of 300M
    print("\n[5/6] Spatial join: stations → boroughs...")
    step_time = time.time()

    con.execute("""
    CREATE OR REPLACE TEMP TABLE station_borough AS
    SELECT
      s.station_id,
      COALESCE(p.borough, 'UNKNOWN') AS borough
    FROM stations s
    LEFT JOIN borough_poly p
      ON st_contains(p.geom, st_point(s.lng, s.lat))
    """)

    total_stations = con.execute("SELECT COUNT(*) as n FROM station_borough").fetchdf()['n'].iloc[0]
    unknown_stations = con.execute("SELECT COUNT(*) as n FROM station_borough WHERE borough = 'UNKNOWN'").fetchdf()['n'].iloc[0]
    print(f"  Assigned {int(total_stations):,} stations to boroughs ({int(unknown_stations):,} as UNKNOWN) ({time.time() - step_time:.1f}s)")

    # Step 6: Aggregate trips by hour × borough
    # Regular join on station_id is fast (indexed)
    print("\n[6/6] Aggregating trips by hour × borough...")
    step_time = time.time()

    con.execute("""
    CREATE OR REPLACE TABLE trips_borough_hour AS
    SELECT
      date_trunc('hour', t.started_ts) AS hour_ts,
      COALESCE(sb.borough, 'UNKNOWN') AS borough,
      t.member_casual,
      t.rideable_type,
      COUNT(*)::DOUBLE AS n_trips,
      (SUM(t.duration_sec) / 60.0)::DOUBLE AS exposure_min
    FROM trips t
    LEFT JOIN station_borough sb
      ON t.start_station_id = sb.station_id
    GROUP BY 1,2,3,4
    """)

    final_stats = con.execute("""
    SELECT COUNT(*) as rows, SUM(n_trips) as total_trips
    FROM trips_borough_hour
    """).fetchdf()

    print(f"  Aggregation complete ({time.time() - step_time:.1f}s)")
    print(f"  Output rows: {int(final_stats['rows'].iloc[0]):,}")
    print(f"  Total trips: {int(final_stats['total_trips'].iloc[0]):,}")

    # Export to parquet
    print("\nExporting to parquet...")
    con.execute(f"COPY trips_borough_hour TO '{OUT.as_posix()}' (FORMAT PARQUET);")

    file_size = OUT.stat().st_size / (1024**2)
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print(f"Output: {OUT}")
    print(f"Size: {file_size:.1f} MB")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("=" * 70)


if __name__ == "__main__":
    main()
