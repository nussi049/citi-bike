#!/usr/bin/env python
"""
clean_data.py - Data Cleaning Pipeline for NYC Bike Crash Risk Modeling

This script performs two main tasks:
1. Clean CitiBike trip data (filter invalid durations/coordinates)
2. Clean and impute coordinates for bike-related crashes

INPUTS:
    - data/raw/crashes/crashes.parquet        (raw NYPD crash data)
    - data/interim/tripdata_2013_2025.parquet (raw CitiBike trips)

OUTPUTS:
    - data/interim/tripdata_2013_2025_clean.parquet  (cleaned trips)
    - data/interim/crashes_bike_clean.parquet        (bike crashes with imputed coords)

METHODOLOGY:
    Trip Cleaning:
        - Remove trips < 60s (likely false starts) or > 6h (likely errors)
        - Remove trips outside NYC bounding box (excludes Jersey City, keeps NYC proper)

    Crash Cleaning:
        - Filter to bicycle-related crashes only (vehicle type or cyclist injury)
        - Impute missing coordinates using:
            1. Borough centroid + random jitter (if borough known)
            2. Random sampling from valid crash distribution (if no borough)

Usage:
    python -m src.processing.clean_data
    # or via Makefile:
    make clean-data
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
# Fixed random seed for reproducibility of coordinate imputation
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Input files
CRASH_RAW = DATA_DIR / "raw" / "crashes" / "crashes.parquet"
TRIPS_RAW = DATA_DIR / "interim" / "tripdata_2013_2025.parquet"

# Output files
TRIPS_CLEAN = DATA_DIR / "interim" / "tripdata_2013_2025_clean.parquet"
CRASH_CLEAN = DATA_DIR / "interim" / "crashes_bike_clean.parquet"

# NYC bounding boxes - separate for trips vs crashes
# Trips: Exclude Jersey City (CitiBike expanded there, but no NYPD crash data)
# Crashes: Include all NYC boroughs including Staten Island
NYC_BBOX_TRIPS = {
    "lat_min": 40.50,
    "lat_max": 40.92,   # Northern Bronx boundary
    "lng_min": -74.05,  # Hudson River (excludes Jersey City)
    "lng_max": -73.70
}

# Crashes: Full NYC including Staten Island (which extends to ~-74.25)
NYC_BBOX_CRASHES = {
    "lat_min": 40.50,
    "lat_max": 40.92,   # Northern Bronx boundary
    "lng_min": -74.30,  # Include Staten Island
    "lng_max": -73.70
}

# Trip duration bounds (seconds)
MIN_DURATION_SEC = 60      # Minimum 1 minute
MAX_DURATION_SEC = 6 * 3600  # Maximum 6 hours

# Borough centroids for coordinate imputation
# Format: (center_lat, center_lng, lat_radius, lng_radius)
BOROUGH_CENTROIDS = {
    'MANHATTAN':     (40.7831, -73.9712, 0.08, 0.05),
    'BROOKLYN':      (40.6782, -73.9442, 0.10, 0.08),
    'QUEENS':        (40.7282, -73.7949, 0.12, 0.10),
    'BRONX':         (40.8448, -73.8648, 0.08, 0.06),
    'STATEN ISLAND': (40.5795, -74.1502, 0.10, 0.08)
}


def clean_trips(con: duckdb.DuckDBPyConnection) -> None:
    """
    Clean CitiBike trip data by filtering invalid records.

    Filtering criteria:
        - Duration must be between 60 seconds and 6 hours
        - Start and end coordinates must be within NYC bounding box
        - All required fields must be non-null

    Args:
        con: DuckDB connection
    """
    print("=" * 70)
    print("CLEANING CITIBIKE TRIP DATA")
    print("=" * 70)

    # Count raw trips
    raw_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{TRIPS_RAW}')"
    ).fetchone()[0]
    print(f"Raw trips: {raw_count:,}")

    # Apply filters and save
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{TRIPS_RAW}')
            WHERE duration_sec IS NOT NULL
              AND duration_sec >= {MIN_DURATION_SEC}
              AND duration_sec <= {MAX_DURATION_SEC}
              AND start_lat BETWEEN {NYC_BBOX_TRIPS['lat_min']} AND {NYC_BBOX_TRIPS['lat_max']}
              AND start_lng BETWEEN {NYC_BBOX_TRIPS['lng_min']} AND {NYC_BBOX_TRIPS['lng_max']}
              AND end_lat BETWEEN {NYC_BBOX_TRIPS['lat_min']} AND {NYC_BBOX_TRIPS['lat_max']}
              AND end_lng BETWEEN {NYC_BBOX_TRIPS['lng_min']} AND {NYC_BBOX_TRIPS['lng_max']}
        ) TO '{TRIPS_CLEAN}' (FORMAT PARQUET)
    """)

    # Count cleaned trips
    clean_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{TRIPS_CLEAN}')"
    ).fetchone()[0]

    removed = raw_count - clean_count
    print(f"Clean trips: {clean_count:,}")
    print(f"Removed: {removed:,} ({removed/raw_count*100:.2f}%)")
    print(f"Output: {TRIPS_CLEAN}")


def classify_coordinates(row: pd.Series) -> tuple:
    """
    Classify crash coordinate quality for imputation strategy.

    Returns:
        tuple: (classification, latitude, longitude)
            - 'valid': coordinates are usable as-is (within NYC bounds)
            - 'impute_borough': coordinates missing/invalid but borough known → impute to borough
            - 'impute_citywide': coordinates missing/invalid, no borough → impute citywide
    """
    lat, lng = row['latitude'], row['longitude']
    borough = str(row['borough']).strip().upper()

    # Check if coordinates are valid and within NYC
    if (pd.notna(lat) and pd.notna(lng) and
        lat != 0.0 and lng != 0.0 and
        NYC_BBOX_CRASHES['lat_min'] <= lat <= NYC_BBOX_CRASHES['lat_max'] and
        NYC_BBOX_CRASHES['lng_min'] <= lng <= NYC_BBOX_CRASHES['lng_max']):
        return 'valid', lat, lng

    # If borough is known → impute within that borough
    if borough in BOROUGH_CENTROIDS:
        return 'impute_borough', None, None

    # No valid coordinates AND no NYC borough → impute citywide
    # (NYPD data, so still NYC crashes - just missing location info)
    return 'impute_citywide', None, None


def impute_borough_coordinates(borough: str) -> tuple:
    """
    Generate random coordinates within a borough's approximate bounds.

    Uses uniform distribution around borough centroid with borough-specific
    radius to approximate realistic crash locations.

    Args:
        borough: Borough name (uppercase)

    Returns:
        tuple: (latitude, longitude) or (None, None) if borough unknown
    """
    if borough not in BOROUGH_CENTROIDS:
        return None, None

    lat_c, lng_c, lat_r, lng_r = BOROUGH_CENTROIDS[borough]

    # Uniform random within borough bounds
    lat = np.random.uniform(lat_c - lat_r, lat_c + lat_r)
    lng = np.random.uniform(lng_c - lng_r, lng_c + lng_r)

    # Clip to NYC bounds (use crash bounding box for imputation)
    lat = np.clip(lat, NYC_BBOX_CRASHES['lat_min'], NYC_BBOX_CRASHES['lat_max'])
    lng = np.clip(lng, NYC_BBOX_CRASHES['lng_min'], NYC_BBOX_CRASHES['lng_max'])

    return lat, lng


def clean_crashes(con: duckdb.DuckDBPyConnection) -> None:
    """
    Clean bike-related crash data with coordinate imputation.

    Steps:
        1. Filter to bicycle-related crashes (by vehicle type or cyclist injury)
        2. Classify coordinate quality
        3. Impute missing coordinates:
           - Borough-based: random within borough bounds
           - Citywide: sample from valid crash distribution
        4. Save cleaned dataset

    Args:
        con: DuckDB connection
    """
    print("\n" + "=" * 70)
    print("CLEANING BIKE CRASH DATA")
    print("=" * 70)

    # Step 1: Load bike-related crashes
    # Include: BICYCLE, BIKE, E-BIKE (exclude scooters and motorbikes)
    crashes = con.execute(f"""
        WITH base AS (
            SELECT
                *,
                UPPER(TRIM(COALESCE(vehicle_type_code1, ''))) AS vtype1,
                UPPER(TRIM(COALESCE(vehicle_type_code2, ''))) AS vtype2
            FROM read_parquet('{CRASH_RAW}')
        )
        SELECT *
        FROM base
        WHERE
            -- Vehicle type contains bike but not motor/scooter
            (
                vtype1 IN ('BICYCLE', 'BIKE', 'E-BIKE', 'EBIKE', 'E-BICYCLE')
                OR (vtype1 LIKE '%BIKE%' AND vtype1 NOT LIKE '%MOTOR%' AND vtype1 NOT LIKE '%SCOOT%')
            )
            OR (
                vtype2 IN ('BICYCLE', 'BIKE', 'E-BIKE', 'EBIKE', 'E-BICYCLE')
                OR (vtype2 LIKE '%BIKE%' AND vtype2 NOT LIKE '%MOTOR%' AND vtype2 NOT LIKE '%SCOOT%')
            )
            -- OR cyclist was injured/killed (regardless of vehicle type)
            OR COALESCE(number_of_cyclist_injured, 0) > 0
            OR COALESCE(number_of_cyclist_killed, 0) > 0
    """).fetchdf()

    print(f"Total bike crashes: {len(crashes):,}")

    # Step 2: Classify coordinates
    crashes[['coord_type', 'lat_clean', 'lng_clean']] = crashes.apply(
        classify_coordinates, axis=1, result_type='expand'
    )

    n_valid = (crashes['coord_type'] == 'valid').sum()
    n_borough = (crashes['coord_type'] == 'impute_borough').sum()
    n_citywide = (crashes['coord_type'] == 'impute_citywide').sum()

    print(f"\nCoordinate classification:")
    print(f"  Valid:           {n_valid:,} ({n_valid/len(crashes)*100:.1f}%)")
    print(f"  Borough impute:  {n_borough:,} ({n_borough/len(crashes)*100:.1f}%)")
    print(f"  Citywide impute: {n_citywide:,} ({n_citywide/len(crashes)*100:.1f}%)")

    # Step 3: Get valid crashes for citywide sampling
    valid_crashes = crashes[crashes['coord_type'] == 'valid'].copy()

    # Step 4: Impute missing coordinates
    print("\nImputing coordinates...")

    for idx, row in crashes.iterrows():
        if row['coord_type'] == 'valid':
            continue

        if row['coord_type'] == 'impute_borough':
            # Impute within known borough
            borough = str(row['borough']).strip().upper()
            lat, lng = impute_borough_coordinates(borough)
            if lat is not None:
                crashes.at[idx, 'lat_clean'] = lat
                crashes.at[idx, 'lng_clean'] = lng

        elif row['coord_type'] == 'impute_citywide':
            # Sample from valid crash distribution (citywide)
            sample = valid_crashes.sample(n=1, random_state=RANDOM_SEED)
            crashes.at[idx, 'lat_clean'] = sample['lat_clean'].iloc[0]
            crashes.at[idx, 'lng_clean'] = sample['lng_clean'].iloc[0]

    # Verify no missing coordinates
    missing = crashes[['lat_clean', 'lng_clean']].isna().any(axis=1).sum()
    assert missing == 0, f"Still have {missing} missing coordinates!"

    # Step 5: Select and save final columns
    final_columns = [
        'crash_date', 'crash_time', 'borough', 'zip_code',
        'lat_clean', 'lng_clean',
        'on_street_name', 'cross_street_name',
        'number_of_persons_injured', 'number_of_persons_killed',
        'number_of_pedestrians_injured', 'number_of_pedestrians_killed',
        'number_of_cyclist_injured', 'number_of_cyclist_killed',
        'number_of_motorist_injured', 'number_of_motorist_killed',
        'contributing_factor_vehicle_1', 'contributing_factor_vehicle_2',
        'vehicle_type_code1', 'vehicle_type_code2', 'collision_id',
        'coord_type'
    ]

    final_crashes = crashes[final_columns].rename(columns={
        'lat_clean': 'latitude',
        'lng_clean': 'longitude'
    })

    # Ensure output directory exists
    CRASH_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    final_crashes.to_parquet(CRASH_CLEAN, index=False)

    print(f"\nSaved: {CRASH_CLEAN}")
    print(f"Total crashes: {len(final_crashes):,}")
    print(f"Lat range: [{final_crashes['latitude'].min():.4f}, {final_crashes['latitude'].max():.4f}]")
    print(f"Lng range: [{final_crashes['longitude'].min():.4f}, {final_crashes['longitude'].max():.4f}]")


def main():
    """Main entry point for data cleaning pipeline."""

    # Check if outputs already exist
    if TRIPS_CLEAN.exists() and CRASH_CLEAN.exists():
        print("=" * 70)
        print("CLEANED DATA ALREADY EXISTS - SKIPPING")
        print("=" * 70)
        print(f"Trip data:  {TRIPS_CLEAN}")
        print(f"Crash data: {CRASH_CLEAN}")
        print("\nTo regenerate, delete these files first.")
        return

    # Initialize DuckDB
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")

    # Clean trips
    if not TRIPS_CLEAN.exists():
        clean_trips(con)
    else:
        print(f"Trip data already exists: {TRIPS_CLEAN}")

    # Clean crashes
    if not CRASH_CLEAN.exists():
        clean_crashes(con)
    else:
        print(f"Crash data already exists: {CRASH_CLEAN}")

    # Summary
    print("\n" + "=" * 70)
    print("DATA CLEANING COMPLETE")
    print("=" * 70)
    print(f"Trip data:  {TRIPS_CLEAN} ({TRIPS_CLEAN.stat().st_size / 1e9:.2f} GB)")
    print(f"Crash data: {CRASH_CLEAN} ({CRASH_CLEAN.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
