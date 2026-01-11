# Risk Modeling Pipeline

This directory contains the modeling logic for NYC Bike Crash Risk Prediction.

## Overview

The model predicts bicycle crashes in NYC based on:
- **CitiBike exposure** (ride minutes per cell/day) as a proxy for cycling activity
- **Spatial features** (grid cells)
- **Temporal features** (day of week, month, trend)
- **Weather** (temperature, precipitation, snow, wind speed)

---

## Model Architecture (Daily Aggregation)

### Why Daily Instead of Hourly?

The model uses **daily** aggregation for memory efficiency. This ~24x reduction in data size allows the model to run on 8GB RAM machines without memory issues.

**Important:** Hour-of-day patterns are still shown in the dashboard from raw hourly data. The model simply doesn't use hour-of-day as a predictor.

---

## Spatial Binning (Grid System)

### Why a Grid?

We model risk **not** at exact coordinate level, but at **cell level**. This is intentional:

1. **Aggregation**: Crashes are counted per cell -> more stable estimates
2. **Exposure matching**: CitiBike trips are also aggregated to cells
3. **Interpretability**: "Risk in Midtown" instead of "Risk at 40.7589N"

### Grid Specification

| Parameter | Value |
|-----------|-------|
| Cell size | 0.025 x 0.025 deg |
| Real size | ~2.5 km x 2.5 km |

### The Binning Process

```
Step 1: Raw Coordinates
------------------------
Crash #1: (40.7589, -73.9851)
Crash #2: (40.7612, -73.9823)
Crash #3: (40.7801, -73.9654)

Step 2: Grid Quantization
------------------------
Formula: floor(coord / 0.025) * 0.025

Crash #1: floor(40.7589 / 0.025) * 0.025 = 40.75
          floor(-73.9851 / 0.025) * 0.025 = -74.00
          -> Cell: (40.75, -74.00)

Crash #2: floor(40.7612 / 0.025) * 0.025 = 40.75
          floor(-73.9823 / 0.025) * 0.025 = -74.00
          -> Cell: (40.75, -74.00)  <- SAME CELL!

Crash #3: floor(40.7801 / 0.025) * 0.025 = 40.775
          floor(-73.9654 / 0.025) * 0.025 = -73.975
          -> Cell: (40.775, -73.975)

Step 3: Aggregation per Cell/Day
------------------------
| cell_id         | day_ts     | grid_lat | grid_lng | y_bike |
|-----------------|------------|----------|----------|--------|
| 40.75_-74.00    | 2024-01-15 | 40.75    | -74.00   | 2      |
| 40.775_-73.975  | 2024-01-15 | 40.775   | -73.975  | 1      |
```

**Important:** Exact coordinates are lost. The model only knows which cell a crash occurred in, not where exactly within the cell.

---

## Coordinate Normalization

### Problem

Grid coordinates have very small variance:
- `grid_lat` range: ~40.70 to ~40.85 (only 0.15 deg)
- `grid_lng` range: ~-74.05 to ~-73.90 (only 0.15 deg)

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

## Model Formula (Daily)

```
y_bike ~ C(dow) + C(month)
       + grid_lat_norm + grid_lng_norm + lat2 + lng2 + lat_lng
       + temp + prcp + snow + wspd
       + trend
       + log1p_exposure

where:
  grid_lat_norm = (grid_lat - mean) / std   (normalized grid cell latitude)
  grid_lng_norm = (grid_lng - mean) / std   (normalized grid cell longitude)
  trend = (day_ts - 2021-01-01).days / 365.25   (years since training start)
  log1p_exposure = log(exposure_min + 1)   (handles exposure=0)
```

**Note:** The formula does NOT include `C(hour_of_day)` because we aggregate to daily level. Hour-of-day patterns are shown in the dashboard from raw hourly data.

### Exposure as Feature (Not Offset)

The exposure term is modeled as a **feature** rather than an offset. This has important advantages:

| Aspect | Offset Approach | Feature Approach |
|--------|-----------------|------------------|
| Formula | `offset(log(exposure))` | `log1p_exposure` as feature |
| Exposure=0 | Excluded (log(0) = -inf) | Included (log1p(0) = 0) |
| Coefficient | Fixed at beta=1 | Estimated from data |
| Training | Only crashes with exposure | ALL crashes |
| Evaluation | Only hours with exposure | ALL hours |

**Why log1p?** The transformation `log(exposure + 1)` handles zero exposure gracefully:
- log1p(0) = 0 -> baseline crash rate without cycling exposure
- log1p(100) = 4.6 -> effect of 100 minutes exposure

**Interpretation:** The estimated beta coefficient determines the exposure elasticity:
- beta < 1: Diminishing returns (doubling exposure less than doubles crashes)
- beta = 1: Proportional relationship (same as offset)
- beta > 1: Superproportional (doubling exposure more than doubles crashes)

### Components

| Term | Description |
|------|-------------|
| `C(dow)` | 7 categorical effects for day of week |
| `C(month)` | 12 categorical effects for month |
| `grid_lat_norm`, `grid_lng_norm` | Linear spatial trends (normalized grid coordinates) |
| `lat2`, `lng2` | Quadratic terms allowing curvature |
| `lat_lng` | Interaction term allowing diagonal patterns |
| `temp`, `prcp`, `snow`, `wspd` | Weather effects (normalized) |
| `trend` | Temporal trend (years since 2021-01-01) |
| `log1p_exposure` | CitiBike exposure as feature (estimated coefficient) |

### Model Type

- **Family**: Poisson (dispersion ~1, no overdispersion)
- **Link**: Log-link
- **Interpretation**: lambda = exp(X*beta) where X includes log1p(exposure)

**Why Poisson (not Negative Binomial)?**
The data shows a dispersion of ~1.002, indicating practically no overdispersion. A Poisson GLM is therefore sufficient and simpler to interpret than Negative Binomial.

---

## Evaluation Methodology

### Cell Consistency

**Critical:** Both predictions and observed crashes use the **same cell set**.

The model uses `cells_2025` (cells that had CitiBike exposure in 2025) for:
1. **Prediction**: Only predict for cells with 2025 exposure
2. **Observed**: Only count crashes in cells with 2025 exposure

This ensures an apples-to-apples comparison. Without this, the model would predict crashes for cells that had no 2025 exposure, inflating predictions.

```
cells_keep (training)     cells_2025 (evaluation)
     N cells          --->   subset with 2025 exposure
```

### Why This Matters for Insurance

For insurance pricing, we want to answer:
> "Given the areas where CitiBike operates in 2025, how many crashes do we expect?"

This requires predicting only for areas with actual exposure, not historical areas that may no longer have CitiBike service.

---

## Data Pipeline

```
+----------------------------------------------------------+
|                         INPUTS                            |
+----------------------------------------------------------+
|  crashes_bike_clean.parquet     (crashes with coordinates)|
|  tripdata_2013_2025_clean.parquet (CitiBike trips)        |
|  weather_hourly_openmeteo/      (hourly weather)          |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                    SPATIAL BINNING                        |
+----------------------------------------------------------+
|  1. Crashes -> Grid cells (0.025 deg)                     |
|  2. Trips -> Grid cells (0.025 deg)                       |
|  3. Aggregation per cell/DAY                              |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                    FEATURE ENGINEERING                    |
+----------------------------------------------------------+
|  - Time features (dow, month, trend)                      |
|  - Spatial features (grid_lat_norm, grid_lng_norm, ...)   |
|  - Weather features (temp, prcp, snow, wspd - normalized) |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                    MODEL FITTING                          |
+----------------------------------------------------------+
|  Training: 2021-2024 (excludes COVID 2020)                |
|  Test: 2025 (true out-of-sample)                          |
|  Model: Poisson GLM                                       |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                    OUTPUTS                                |
+----------------------------------------------------------+
|  model_meta_bike_all.json           (normalization stats) |
|  model_comparison_bike_all.parquet  (AIC, dispersion)     |
|  risk_mc_2025_totals_*.parquet      (Monte Carlo draws)   |
|  risk_eval_2025_monthly_*.parquet   (observed vs pred)    |
+----------------------------------------------------------+
```

---

## Temporal Separation

| Period | Usage |
|--------|-------|
| 2021-2024 | Training (4 years, excludes COVID 2020) |
| 2025 | Test (out-of-sample) |

The model is trained **only** on 2021-2024 data. The year 2020 is excluded due to anomalous patterns during COVID. The 2025 predictions are true out-of-sample forecasts.

---

## Monte Carlo Simulation

For uncertainty quantification:

1. **S=1000 simulations** for robust uncertainty estimates
2. **4 uncertainty dimensions**:
   - **Weather bootstrap**: Random year (2021-2025) as weather scenario
   - **Exposure year**: Historical exposure patterns (2021-2025) applied to 2025 grid
   - **Growth factor**: Random factor in ±20% range (historically observed)
   - **Parameter uncertainty**: Sampling from estimated coefficient distribution
3. **Exposure scenarios**: -10%, actual, +10%

**Exposure Scenario Interpretation:**
When exposure_year != 2025, we apply historical usage patterns to the 2025 network.
This answers: "What if the 2025 network had been used like in year X?"
Cells that didn't exist in year X will have exposure=0.

### Exposure Scenarios with Feature Approach

With exposure as a feature (not offset), scenarios work non-linearly:

```
Scenario: +10% exposure
------------------------
1. Modify raw exposure: exposure_scenario = exposure * 1.10
2. Recompute feature: log1p_exposure = log(exposure_scenario + 1)
3. Predict: mu = exp(X*beta)

Effect depends on:
- Baseline exposure level
- Estimated beta coefficient
- Non-linear log1p transformation
```

**Example:** With beta=0.8 (diminishing returns):
- Low exposure (10 min -> 11 min): +10% exposure -> ~+8% crashes
- High exposure (100 min -> 110 min): +10% exposure -> ~+7% crashes

This is more realistic than the offset approach where +10% exposure always means +10% crashes.

---

## Results Interpretation

### 2025 Evaluation

The model predicts crash counts for 2025, which can be compared against observed data. Note that:

1. **Reporting Delays**: Late 2025 crashes may not be fully reported yet.
2. **Year-to-year variation**: Crash totals vary year to year based on many factors.
3. **Trend extrapolation**: The trend variable captures historical patterns but may not perfectly predict future changes.

---

## Files

| File | Description |
|------|-------------|
| `run_risk_modeling.py` | Main pipeline (data prep -> modeling -> evaluation) |

## Usage

```bash
# Direct
python src/modeling/run_risk_modeling.py

# Via Makefile
make modeling
```
