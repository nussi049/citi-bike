# NYC Bike Crash Risk Analysis

An end-to-end data analysis project exploring bicycle crash patterns and risk factors in New York City. The project combines multiple data sources, builds a predictive model, and delivers insights through an interactive dashboard.

## Project Goals

This project delivers two main outputs:

### 1. Interactive Dashboard

A Streamlit dashboard with three pages for exploring NYC bike data:

- **Trip Data**: Analyze CitiBike usage patterns, seasonality, weather impact
- **Crash Data**: Explore historical crash patterns by time, location, weather
- **Risk & Exposure**: View spatial heatmaps, model predictions, and scenario analysis

### 2. Risk Prediction Model

A Poisson GLM that predicts bike crash risk:
- Uses CitiBike trip minutes as proxy for cycling exposure
- Accounts for time (day of week, month), location (grid cells), and weather
- Trained on daily data (2021-2024, excludes COVID year 2020), tested on 2025

## Key Features

**Dashboard:**
- Interactive filters (date range, borough, bike type, weather)
- 20+ visualizations across three pages
- Spatial heatmaps with Folium (crashes, exposure, crash rate)
- Weather impact analysis (temperature, precipitation, snow)

**Model:**
- Spatial grid model at daily resolution (~2.5km cells)
- Exposure as feature (not offset): crash rate with estimated elasticity
- Monte Carlo uncertainty quantification (S=1000) with 4 uncertainty dimensions:
  - Weather bootstrap (2021-2025)
  - Exposure year scenarios (2021-2025)
  - Growth factor uncertainty (±20%)
  - Parameter sampling from coefficient distribution
  

## Project Structure

```
city-bike/
├── data/
│   ├── raw/                  # Downloaded data (crashes, trips, weather)
│   ├── interim/              # Cleaned intermediate files
│   └── processed/            # Model outputs, dashboard marts
├── src/
│   ├── data/                 # Data download scripts
│   ├── processing/           # Data cleaning and validation
│   ├── modeling/             # Risk modeling pipeline
│   └── dashboard/            # Streamlit dashboard (3 pages)
├── Makefile                  # Pipeline automation
└── requirements.txt          # Python dependencies
```

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Download data, clean, model, and prepare dashboard
make all
```

Or run individual steps:

```bash
make data        # Download all raw data
make clean-data  # Clean trip/crash data
make modeling    # Run risk modeling (~2-3 min)
make dashboard   # Start Streamlit dashboard
```

### 3. View Dashboard

```bash
make dashboard
# Opens at http://localhost:8501
```

---

## Dashboard

The interactive Streamlit dashboard is the main deliverable for exploring the data and model results. It consists of three pages:

### Page 1: CitiBike Trips (Usage & Seasonality)

Explore CitiBike usage patterns with interactive filters and visualizations.

**Filters (Sidebar):**
- Date range (start/end)
- Bike type (classic/electric/all)
- Customer type (member/casual/all)
- Borough selection (multi-select)
- Time granularity (day/week/month)
- Metric display (trips/exposure minutes)

**Visualizations:**

| Chart | Description |
|-------|-------------|
| **KPI Cards** | Total trips, total exposure minutes, average trip duration |
| **Usage over time** | Dual-axis line chart: trips (blue) and exposure minutes (orange) over time |
| **Time-of-day pattern** | Hourly distribution (0-23h) showing peak hours |
| **Day-of-week pattern** | Weekly pattern with weekday breakdown |
| **Hourly by weekday** | Multi-line chart comparing hourly patterns across weekdays |
| **Bike type over time** | Classic vs electric bike usage trends |
| **Borough distribution** | Pie chart showing trip distribution by borough |
| **Weather impact** | Three charts showing usage deviation by temperature, precipitation, and snow |

---

### Page 2: Bike Crashes

Analyze historical bike crash patterns across NYC.

**Filters (Sidebar):**
- Date range (start/end)
- Borough selection (multi-select)
- Time granularity (day/week/month)

**Visualizations:**

| Chart | Description |
|-------|-------------|
| **KPI Cards** | Total crashes, crashes per day, percentage of selected boroughs |
| **Crashes over time** | Time series of crash counts |
| **Time-of-day pattern** | Hourly crash distribution (peak hours visible) |
| **Day-of-week pattern** | Weekly crash pattern |
| **Hourly by weekday** | Multi-line chart comparing crash timing across weekdays |
| **Monthly pattern** | Seasonal crash distribution (summer vs winter) |
| **Borough distribution** | Pie chart showing crash distribution by borough |
| **Weather impact** | Three charts showing crash deviation by temperature, precipitation, and snow |

---

### Page 3: Risk & Exposure

Model results, spatial analysis, and predictions.

**Filters (Sidebar):**
- Year selection for heatmaps (2021-2025)

**Visualizations:**

| Section | Charts | Description |
|---------|--------|-------------|
| **Spatial Heatmaps** | 3 interactive Folium maps | Grid cells colored by: (1) total crashes, (2) total exposure minutes, (3) model coverage + bike counters |
| **Poisson Diagnostics** | Table + metrics | AIC, dispersion, coefficient estimates |
| **2025 Predictions** | Distribution chart | Monte Carlo simulation results with median and 90% confidence interval |
| **Observed vs Predicted** | Monthly comparison | Actual 2025 crashes vs model predictions by month |
| **Exposure Scenarios** | Comparison chart | Predicted crashes under -10%, actual, +10% exposure scenarios |
| **Proxy Validation** | Correlation chart | CitiBike trips vs NYC bike counter data to validate exposure proxy |

**Heatmap Details:**

The spatial heatmaps use a 0.025° x 0.025° grid (~2.5 km cells) with:
- Color scale from green (low) to red (high)
- Hover tooltips showing exact values
- NYC borough boundaries overlay
- Interactive zoom and pan

---

## Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| Crashes | NYC Open Data | Motor vehicle collisions with bike involvement |
| Trips | CitiBike | Trip records with start/end locations and times |
| Weather | Open-Meteo | Hourly temperature, precipitation, wind |
| Boundaries | NYC Open Data | Borough boundaries (GeoJSON) |
| Bike Counts | NYC DOT | Automated bike counter data (for proxy validation) |

## Model Overview

The model predicts crash counts using a **Poisson GLM** trained on **daily** data:

```
y_bike ~ C(dow) + C(month)
       + grid_lat_norm + grid_lng_norm + lat2 + lng2 + lat_lng
       + temp + prcp + snow + wspd
       + trend
       + log1p_exposure

where:
  grid_lat_norm, grid_lng_norm = normalized grid cell coordinates
  trend = (day_ts - 2021-01-01).days / 365.25  (years since training start)
  log1p_exposure = log(exposure_min + 1)  (handles exposure=0)
```

**Key Design Decisions:**

1. **Daily Aggregation**: Model trained on daily data (~117K rows) instead of hourly (~2.8M rows) for memory efficiency. Hour-of-day patterns are shown in dashboard from raw data.

2. **Poisson Model**: The data shows dispersion ~1 (practically no overdispersion), so a simple Poisson GLM is sufficient. No Negative Binomial needed.

3. **Exposure as Feature**: The `log1p_exposure` term is a feature with estimated coefficient (β ≈ 0.4-0.7), not a fixed offset. This allows predictions for hours with zero exposure and captures diminishing returns.

4. **Evaluation Consistency**: Predictions and observed crashes use the **same cell set** (cells with 2025 exposure). This ensures apples-to-apples comparison.

5. **Training Period**: 2021-2024 (excludes COVID year 2020 due to anomalous patterns).

See [src/modeling/README.md](src/modeling/README.md) for detailed model documentation.

## Pipeline Steps

| Step | Command | Description |
|------|---------|-------------|
| 1 | `make data` | Download crashes, trips, weather, boundaries |
| 2 | `make clean-data` | Clean and filter data (2021-2025) |
| 3 | `make proxy-test` | Validate CitiBike as cycling proxy |
| 4 | `make mart` | Build dashboard data marts |
| 5 | `make modeling` | Fit GLM, run Monte Carlo simulations |
| 6 | `make dashboard` | Launch interactive dashboard |

## Results

- **Training Period**: 2021-2024 (4 years, daily aggregation, excludes COVID 2020)
- **Backtest Period**: 2025 (using known exposure data)
- **Model**: Poisson GLM (dispersion ~1, no overdispersion)
- **Prediction**: ~6,100 crashes (Predicted) vs ~5,550 crashes (Observed in model cells)

**Note on 2025 Observed Data:**

The observed crash count for 2025 (~5,550 in model cells) appears lower than model predictions (~6,100). This may be partly explained by:

1. **Reporting Delays (Nachmeldelücken)**: Crash data from late 2025 (Nov/Dec) may not yet be fully reported as of early 2026.

2. **2025 was an unusual year**: Total NYC bike crashes in 2025 (6,912) were below the 2021-2024 average (7,732):

| Year | Total Crashes |
|------|---------------|
| 2021 | 7,854 |
| 2022 | 7,883 |
| 2023 | 7,959 |
| 2024 | 7,231 |
| **2025** | **6,912** |
| Avg 2021-24 | 7,732 |

The model, trained on 2021-2024 data, naturally predicts closer to historical averages.

## Requirements

- Python 3.10+
- ~20 GB disk space for data
- ~8 GB RAM (optimized for memory efficiency)
- ~5 minutes for full pipeline
