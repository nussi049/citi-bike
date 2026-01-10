.PHONY: help data geo weather trips crashes bike_counts clean-data proxy-test mart modeling dashboard process all clean

# Python command - use virtual environment if available
PYTHON := $(shell if [ -f .venv/bin/python ]; then echo ".venv/bin/python"; else echo "python"; fi)

# Default target - show help
help:
	@echo "City-Bike Project - Available targets:"
	@echo ""
	@echo "DATA DOWNLOAD:"
	@echo "  make data        - Download all raw data (geo, weather, trips, crashes, bike_counts)"
	@echo ""
	@echo "DATA PROCESSING:"
	@echo "  make clean-data  - Clean trip/crash data (Notebook 01)"
	@echo "  make proxy-test  - Validate CitiBike proxy (Notebook 02)"
	@echo "  make mart        - Build dashboard mart"
	@echo ""
	@echo "MODELING:"
	@echo "  make modeling    - Run risk modeling (Notebook 03, ~10-15 min)"
	@echo ""
	@echo "DASHBOARD:"
	@echo "  make dashboard   - Start Streamlit dashboard"
	@echo ""
	@echo "PIPELINES:"
	@echo "  make all         - Complete pipeline: data → clean → mart → modeling"
	@echo "  make process     - Processing only: clean-data → proxy-test → mart"
	@echo ""
	@echo "UTILITIES:"
	@echo "  make clean       - Remove generated files"
	@echo ""

# ============================================================================
# DATA DOWNLOAD
# ============================================================================
data: geo weather trips crashes bike_counts
	@echo "✅ All datasets downloaded/built into raw/interim."

geo:
	@echo "Downloading NYC borough boundaries..."
	$(PYTHON) -m src.data.download_borough_geojson

bike_counts:
	@echo "Downloading NYC bike counter data..."
	$(PYTHON) -m src.data.download_bike_counts

weather:
	@echo "Downloading weather data..."
	$(PYTHON) -m src.data.download_weather

trips:
	@echo "Downloading CitiBike trip data..."
	$(PYTHON) -m src.data.download_tripdata

crashes:
	@echo "Downloading crash data..."
	$(PYTHON) -m src.data.download_crashdata

# ============================================================================
# DATA PROCESSING
# ============================================================================

# Step 1: Clean and prepare raw data (Notebook 01)
clean-data: trips crashes
	@echo "Cleaning trip and crash data (Notebook 01)..."
	$(PYTHON) src/processing/clean_data.py

# Step 2: Validate CitiBike as proxy (Notebook 02)
proxy-test: clean-data bike_counts
	@echo "Validating CitiBike exposure as proxy (Notebook 02)..."
	$(PYTHON) src/processing/proxy_validation.py

# Step 3: Build dashboard mart
mart: clean-data
	@echo "Building dashboard mart..."
	$(PYTHON) -m src.data.build_mart

# ============================================================================
# MODELING
# ============================================================================

# Step 4: Risk modeling with GLM (Notebook 03)
modeling: mart clean-data
	@echo "Running risk modeling pipeline (Notebook 03)..."
	@echo "This will take ~10-15 minutes..."
	$(PYTHON) src/modeling/run_risk_modeling.py

# ============================================================================
# DASHBOARD
# ============================================================================
dashboard:
	@echo "Starting Streamlit dashboard..."
	streamlit run src/dashboard/app.py

# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

# Processing pipeline (without data download)
process: clean-data proxy-test mart
	@echo "=========================================="
	@echo "✅ PROCESSING PIPELINE FINISHED!"
	@echo "=========================================="

# Full pipeline (data download + processing + modeling)
all: data clean-data proxy-test mart modeling
	@echo "=========================================="
	@echo "✅ COMPLETE PIPELINE FINISHED!"
	@echo "=========================================="
	@echo ""
	@echo "Pipeline executed:"
	@echo "  1. ✅ Downloaded data"
	@echo "  2. ✅ Cleaned data (Notebook 01)"
	@echo "  3. ✅ Validated proxy (Notebook 02)"
	@echo "  4. ✅ Built dashboard mart"
	@echo "  5. ✅ Run risk modeling (Notebook 03)"
	@echo ""
	@echo "Next steps:"
	@echo "  make dashboard    - Start the dashboard"
	@echo ""

# ============================================================================
# CLEANUP
# ============================================================================
clean:
	@echo "Removing generated files..."
	rm -rf data/processed/risk_hourly_mc/*.parquet
	rm -rf data/processed/dashboard_marts/*.parquet
	rm -rf duckdb_tmp/*
	@echo "✅ Cleanup complete"
