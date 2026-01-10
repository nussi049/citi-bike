"""
settings.py - Central Configuration for Dashboard Data Paths

All file paths used by the dashboard are defined here. This provides:
    - Single source of truth for data locations
    - Easy updates when paths change
    - Clear documentation of data dependencies

DATA FLOW:
    Raw Data (data/raw/)
        ↓
    Processing Scripts (src/processing/)
        ↓
    Processed Data (data/processed/)
        ↓
    Dashboard (src/dashboard/)

PIPELINE DEPENDENCIES:
    clean_data.py        → CRASHES_BIKE, (trips to interim)
    build_mart.py        → TRIPS_BOROUGH_HOUR
    proxy_validation.py  → PROXY_TEST_BM
    run_risk_modeling.py → GRID_TRAIN, EVAL_2025, MC_2025, COMP, etc.
"""

from pathlib import Path

# =============================================================================
# PROJECT STRUCTURE
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# =============================================================================
# CORE CLEANED DATASETS (from clean_data.py)
# =============================================================================
# Bike-related crashes with imputed coordinates
CRASHES_BIKE = PROCESSED_DIR / "crashes_bike_clean.parquet"

# NYC borough boundaries for spatial visualizations
BOROUGH_GEOJSON = RAW_DIR / "borough_boundaries.geojson"

# Hourly weather data (partitioned by year)
WEATHER_GLOB = RAW_DIR / "weather_hourly_openmeteo" / "year=*/weather.parquet"

# =============================================================================
# DASHBOARD MARTS (from build_mart.py)
# =============================================================================
MART_DIR = PROCESSED_DIR / "dashboard_marts"

# Pre-aggregated trips by hour × borough for fast dashboard loading
# Used by: 1_tripdata.py
TRIPS_BOROUGH_HOUR = MART_DIR / "trips_borough_hour.parquet"

# =============================================================================
# RISK MODELING OUTPUTS (from run_risk_modeling.py)
# =============================================================================
RISK_DIR = PROCESSED_DIR / "risk_hourly_mc"

# Training data: grid cell × hour with features
GRID_TRAIN = RISK_DIR / "grid_train_cell_hour_2020_2024.parquet"

# Model evaluation: monthly observed vs predicted for 2025
EVAL_2025 = RISK_DIR / "risk_eval_2025_monthly_bike_all.parquet"

# Monte Carlo simulation results (crash totals by simulation)
# NOTE: MC_2025_SCENARIOS replaces the old MC_2025 file

# Model comparison metrics (AIC, dispersion, quantiles)
COMP = RISK_DIR / "model_comparison_bike_all.parquet"

# Crash and exposure data at cell × hour level
CRASH_CELL_HOUR = RISK_DIR / "crash_cell_hour.parquet"
EXPOSURE_CELL_HOUR_TRAIN = RISK_DIR / "exposure_cell_hour_2020_2024.parquet"
EXPOSURE_CELL_HOUR_TEST = RISK_DIR / "exposure_cell_hour_2025.parquet"

# Grid cells included in model (cells with sufficient exposure)
CELLS_KEEP = RISK_DIR / "grid_cells_keep_2020_2024.parquet"

# Exposure scenario analysis (-10%, actual, +10%)
MC_2025_SCENARIOS = RISK_DIR / "risk_mc_2025_totals_bike_all_scenarios.parquet"
EXPOSURE_SCENARIOS_SUMMARY = RISK_DIR / "risk_exposure_scenarios_summary.parquet"

# =============================================================================
# PROXY VALIDATION (from proxy_validation.py)
# =============================================================================
# Borough × Month correlation between CitiBike and bike counters
# Used by: 3_risk_exposure.py (Section F)
PROXY_TEST_BM = PROCESSED_DIR / "proxy_test" / "proxy_test_borough_month.parquet"

# =============================================================================
# BIKE COUNTERS (from download_bike_counts.py)
# =============================================================================
COUNTERS_DIR = PROCESSED_DIR / "nyc_bike_counters"

# Counter locations and metadata
BIKE_COUNTERS = COUNTERS_DIR / "bicycle_counters.parquet"

# Hourly counts by counter with borough assignment
BIKE_COUNTS_HOURLY = COUNTERS_DIR / "bike_counts_hourly_by_counter_enriched.parquet"
