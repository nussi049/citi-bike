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

A Negative Binomial GLM that predicts bike crash risk:
- Uses CitiBike trip minutes as proxy for cycling exposure
- Accounts for time (hour, day, month), location (grid cells), and weather
- Achieves ~3% prediction error on out-of-sample 2025 data

## Key Features

**Dashboard:**
- Interactive filters (date range, borough, bike type, weather)
- 20+ visualizations across three pages
- Spatial heatmaps with Folium (crashes, exposure, crash rate)
- Weather impact analysis (temperature, precipitation, snow)

**Model:**
- Spatial grid model: ~64 cells × hourly resolution
- Exposure-based: crash *rate* per cycling minute, not just counts
- Monte Carlo uncertainty quantification with weather bootstrap
- Exposure scenarios: -10%, actual, +10% sensitivity analysis

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
make modeling    # Run risk modeling (~15-20 min)
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
- Year selection for heatmaps (2020-2025)

**Visualizations:**

| Section | Charts | Description |
|---------|--------|-------------|
| **Spatial Heatmaps** | 3 interactive Folium maps | Grid cells colored by: (1) total crashes, (2) total exposure minutes, (3) crash rate per 100k minutes |
| **Model Comparison** | Table + metrics | Poisson vs Negative Binomial: AIC, dispersion, alpha parameter |
| **2025 Predictions** | Distribution chart | Monte Carlo simulation results with median and 90% confidence interval |
| **Observed vs Predicted** | Monthly comparison | Actual 2025 crashes vs model predictions by month |
| **Exposure Scenarios** | Comparison chart | Predicted crashes under -10%, actual, +10% exposure scenarios |
| **Proxy Validation** | Correlation chart | CitiBike trips vs NYC bike counter data to validate exposure proxy |

**Heatmap Details:**

The spatial heatmaps use a 0.025° × 0.025° grid (~2.5 km cells) with:
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

The model predicts crash counts using a Negative Binomial GLM:

```
y_bike ~ C(hour) + C(dow) + C(month)
       + lat_n + lng_n + lat_n² + lng_n² + lat_n×lng_n
       + trend
       + temp + prcp + snow + wspd
       + offset(log(exposure_min))

where:
  lat_n, lng_n = normalized grid cell coordinates
  exposure_min = CitiBike minutes in cell/hour
```

**Key insight:** The offset term means we model crash *rate* per minute of cycling, not just crash counts. This accounts for varying cycling activity across locations and times.

See [src/modeling/README.md](src/modeling/README.md) for detailed model documentation.

## Pipeline Steps

| Step | Command | Description |
|------|---------|-------------|
| 1 | `make data` | Download crashes, trips, weather, boundaries |
| 2 | `make clean-data` | Clean and filter data (2020-2025) |
| 3 | `make proxy-test` | Validate CitiBike as cycling proxy |
| 4 | `make mart` | Build dashboard data marts |
| 5 | `make modeling` | Fit GLM, run Monte Carlo simulations |
| 6 | `make dashboard` | Launch interactive dashboard |

## Results

- **Training Period**: 2020-2024 (5 years)
- **Test Period**: 2025 (out-of-sample)
- **Model**: Negative Binomial (handles overdispersion)
- **Accuracy**: ~3% error on 2025 total crash prediction

## Requirements

- Python 3.10+
- ~20 GB disk space for data
- ~20 minutes for full pipeline
