# src/dashboard/lib/settings.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ============================================================================
# CORE CLEANED DATASETS
# ============================================================================
CRASHES_BIKE = PROCESSED_DIR / "crashes_bike_liability.parquet"
BOROUGH_GEOJSON = RAW_DIR / "borough_boundaries.geojson"
WEATHER_GLOB = RAW_DIR / "weather_hourly_openmeteo" / "year=*/weather.parquet"

# ============================================================================
# DASHBOARD MARTS
# ============================================================================
MART_DIR = PROCESSED_DIR / "dashboard_marts"
TRIPS_BOROUGH_HOUR = MART_DIR / "trips_borough_hour.parquet"

# ============================================================================
# RISK MODELING OUTPUTS
# ============================================================================
RISK_DIR = PROCESSED_DIR / "risk_hourly_mc"
GRID_TRAIN = RISK_DIR / "grid_train_cell_hour_2020_2024.parquet"
EVAL_2025 = RISK_DIR / "risk_eval_2025_monthly_bike_all.parquet"
MC_2025 = RISK_DIR / "risk_mc_2025_totals_bike_all.parquet"
COMP = RISK_DIR / "model_comparison_bike_all.parquet"
CRASH_CELL_HOUR = RISK_DIR / "crash_cell_hour.parquet"
EXPOSURE_CELL_HOUR = RISK_DIR / "exposure_cell_hour.parquet"

# PROXY
PROXY_TEST_BM = PROCESSED_DIR / "proxy_test" / "proxy_test_borough_month.parquet"