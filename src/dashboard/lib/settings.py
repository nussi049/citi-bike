from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ============================================================================
# CORE CLEANED DATASETS (direkt nutzen!)
# ============================================================================
TRIPS_CLEAN = INTERIM_DIR / "tripdata_2013_2025_clean.parquet"
CRASHES_ALL = INTERIM_DIR / "crashes.parquet"
CRASHES_BIKE = PROCESSED_DIR / "crashes_bike_liability.parquet"
BOROUGH_GEOJSON = RAW_DIR / "borough_boundaries.geojson"
WEATHER_GLOB = RAW_DIR / "weather_hourly_openmeteo" / "year=*/weather.parquet"
# ============================================================================
# DASHBOARD MARTS (nur trips - crashes on-the-fly)
# ============================================================================
MART_DIR = PROCESSED_DIR / "dashboard_marts"
TRIPS_BOROUGH_HOUR = MART_DIR / "trips_borough_hour.parquet"

# ============================================================================
# RISK MODELING OUTPUTS
# ============================================================================
RISK_DIR = PROCESSED_DIR / "risk_hourly_mc"
CITY_TRAIN = RISK_DIR / "city_train_bike_all_2020_2024.parquet"
EVAL_2025 = RISK_DIR / "risk_eval_2025_monthly_bike_all.parquet"
MC_2025 = RISK_DIR / "risk_mc_2025_totals_bike_all.parquet"
COMP = RISK_DIR / "model_comparison_bike_all.parquet"
CRASH_CELL_HOUR = RISK_DIR / "crash_cell_hour.parquet"
EXPOSURE_CELL_HOUR = RISK_DIR / "exposure_cell_hour.parquet"
GRID_TRAIN = RISK_DIR / "grid_train_cell_hour_2020_2024.parquet"
GRID_2025 = RISK_DIR / "grid_2025_cell_hour_bike_all.parquet"

# PROXY
PROXY_TEST_BM = PROCESSED_DIR / "proxy_test" / "proxy_test_borough_month.parquet"