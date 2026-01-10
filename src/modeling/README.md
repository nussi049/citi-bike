# Risk Modeling Pipeline

This directory contains the modeling logic for NYC Bike Crash Risk Prediction.

## Overview

The model predicts bicycle crashes in NYC based on:
- **CitiBike exposure** (ride minutes per cell/hour) as a proxy for cycling activity
- **Spatial features** (grid cells)
- **Temporal features** (hour, day of week, month)
- **Weather** (temperature, precipitation, snow, wind speed)

---

## Spatial Binning (Grid System)

### Why a Grid?

We model risk **not** at exact coordinate level, but at **cell level**. This is intentional:

1. **Aggregation**: Crashes are counted per cell → more stable estimates
2. **Exposure matching**: CitiBike trips are also aggregated to cells
3. **Interpretability**: "Risk in Midtown" instead of "Risk at 40.7589°N"

### Grid Specification

| Parameter | Value |
|-----------|-------|
| Cell size | 0.025° × 0.025° |
| Real size | ~2.5 km × 2.5 km |
| Active cells | ~64 (with CitiBike exposure) |

### The Binning Process

```
Step 1: Raw Coordinates
────────────────────────────
Crash #1: (40.7589, -73.9851)
Crash #2: (40.7612, -73.9823)
Crash #3: (40.7801, -73.9654)

Step 2: Grid Quantization
────────────────────────────
Formula: floor(coord / 0.025) * 0.025

Crash #1: floor(40.7589 / 0.025) * 0.025 = 40.75
          floor(-73.9851 / 0.025) * 0.025 = -74.00
          → Cell: (40.75, -74.00)

Crash #2: floor(40.7612 / 0.025) * 0.025 = 40.75
          floor(-73.9823 / 0.025) * 0.025 = -74.00
          → Cell: (40.75, -74.00)  ← SAME CELL!

Crash #3: floor(40.7801 / 0.025) * 0.025 = 40.775
          floor(-73.9654 / 0.025) * 0.025 = -73.975
          → Cell: (40.775, -73.975)

Step 3: Aggregation per Cell/Hour
────────────────────────────
| cell_id         | hour_ts          | grid_lat | grid_lng | y_bike |
|-----------------|------------------|----------|----------|--------|
| 40.75_-74.00    | 2024-01-15 14:00 | 40.75    | -74.00   | 2      |
| 40.775_-73.975  | 2024-01-15 14:00 | 40.775   | -73.975  | 1      |
```

### Visualization

```
NYC Grid (simplified):
┌───────┬───────┬───────┬───────┐
│   0   │   0   │   0   │   0   │  ← Crash count per cell
├───────┼───────┼───────┼───────┤
│   0   │   1   │   0   │   0   │  ← Crash #3
├───────┼───────┼───────┼───────┤
│   0   │   2   │   0   │   0   │  ← Crash #1 & #2 (aggregated!)
├───────┼───────┼───────┼───────┤
│   0   │   0   │   0   │   0   │
└───────┴───────┴───────┴───────┘
        ↑
    Each box = 2.5 km × 2.5 km
```

**Important:** Exact coordinates are lost. The model only knows which cell a crash occurred in, not where exactly within the cell.

---

## Coordinate Normalization

### Problem

Grid coordinates have very small variance:
- `grid_lat` range: ~40.70 to ~40.85 (only 0.15°)
- `grid_lng` range: ~-74.05 to ~-73.90 (only 0.15°)

This can cause numerical issues in GLM fitting.

### Solution: Z-Score Normalization

```python
# Training statistics (computed once from 2020-2024 data)
lat_mean = 40.77
lat_std  = 0.04
lng_mean = -73.96
lng_std  = 0.03

# Normalization
grid_lat_norm = (grid_lat - lat_mean) / lat_std
grid_lng_norm = (grid_lng - lng_mean) / lng_std

# Example:
# Cell (40.75, -74.00):
# grid_lat_norm = (40.75 - 40.77) / 0.04 = -0.5
# grid_lng_norm = (-74.00 - (-73.96)) / 0.03 = -1.33
```

Normalization statistics are stored in `model_meta_bike_all.json` and reused for predictions on 2025 data.

---

## Model Formula

```
y_bike ~ C(hour_of_day) + C(dow) + C(month)
       + lat_n + lng_n + lat_n² + lng_n² + lat_n×lng_n
       + trend
       + temp + prcp + snow + wspd
       + offset(log(exposure_min))

where:
  lat_n = (grid_lat - mean) / std   (normalized grid cell latitude)
  lng_n = (grid_lng - mean) / std   (normalized grid cell longitude)
```

**Note:** All spatial terms use **normalized grid cell coordinates**, not raw crash coordinates.

### Components

| Term | Description |
|------|-------------|
| `C(hour_of_day)` | 24 categorical effects for time of day |
| `C(dow)` | 7 categorical effects for day of week |
| `C(month)` | 12 categorical effects for month |
| `lat_n`, `lng_n` | Linear spatial trends (normalized grid coordinates) |
| `lat_n²`, `lng_n²` | Quadratic terms allowing curvature |
| `lat_n×lng_n` | Interaction term allowing diagonal patterns |
| `trend` | Linear time trend (days since 2020-01-01) |
| `temp`, `prcp`, `snow`, `wspd` | Weather effects (normalized) |
| `offset(log(exposure_min))` | CitiBike exposure as offset |

### Model Type

- **Family**: Negative Binomial (accounts for overdispersion)
- **Link**: Log-link
- **Interpretation**: Rate model → λ = exposure × exp(Xβ)

---

## Data Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUTS                                   │
├─────────────────────────────────────────────────────────────────┤
│  crashes_bike_clean.parquet     (crashes with coordinates)      │
│  tripdata_2013_2025_clean.parquet (CitiBike trips)              │
│  weather_hourly_openmeteo/      (hourly weather)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SPATIAL BINNING                               │
├─────────────────────────────────────────────────────────────────┤
│  1. Crashes → Grid cells (0.025°)                               │
│  2. Trips → Grid cells (0.025°)                                 │
│  3. Aggregation per cell/hour                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                           │
├─────────────────────────────────────────────────────────────────┤
│  - Time features (hour, dow, month, trend)                      │
│  - Spatial features (grid_lat_norm, grid_lng_norm, ...)         │
│  - Weather features (temp, prcp, snow, wspd - normalized)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL FITTING                                 │
├─────────────────────────────────────────────────────────────────┤
│  Training: 2020-2024 (temporal separation)                      │
│  Test: 2025 (true out-of-sample)                                │
│  Models: Poisson + Negative Binomial                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUTS                                       │
├─────────────────────────────────────────────────────────────────┤
│  model_meta_bike_all.json           (normalization, coeffs)     │
│  model_comparison_bike_all.parquet  (AIC, dispersion)           │
│  risk_mc_2025_totals_*.parquet      (Monte Carlo predictions)   │
│  risk_eval_2025_monthly_*.parquet   (observed vs predicted)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Temporal Separation

| Period | Usage |
|--------|-------|
| 2020-2024 | Training (5 years) |
| 2025 | Test (out-of-sample) |

The model is trained **only** on 2020-2024 data. The 2025 predictions are true out-of-sample forecasts.

---

## Monte Carlo Simulation

For uncertainty quantification:

1. **50 simulations** per model/scenario
2. **Weather bootstrap**: Random training year as weather proxy for 2025
3. **Parameter uncertainty**: Sampling from estimated coefficient distribution
4. **Exposure scenarios**: -10%, actual, +10%

---

## Files

| File | Description |
|------|-------------|
| `run_risk_modeling.py` | Main pipeline (data prep → modeling → evaluation) |

## Usage

```bash
# Direct
python src/modeling/run_risk_modeling.py

# Via Makefile
make modeling
```
