import streamlit as st

st.set_page_config(page_title="Citi Bike Risk Dashboard", layout="wide")

st.title("Citi Bike — Trips, Crashes, Risk & Exposure")

st.markdown("""
### NYC Bike Crash Risk Modeling

This dashboard presents a probabilistic model for predicting bike crashes in New York City,
using CitiBike trip data as a proxy for cycling exposure.

**Use the pages in the sidebar:**

1. **Tripdata** — Explore CitiBike trip patterns
2. **Crashdata** — Analyze historical crash data
3. **Risk & Exposure** — Model predictions, validation, and methodology
""")

st.markdown("---")

# Project Overview
st.subheader("Project Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Objective:**
    Predict bike crash counts for NYC grid cells using a Negative Binomial GLM
    with CitiBike exposure as offset.

    **Key Features:**
    - Temporal effects (hour, day-of-week, month)
    - Spatial effects (lat/lng with quadratic terms)
    - Weather covariates (temperature, precipitation, wind)
    - Trend term for declining crash rate
    """)

with col2:
    st.markdown("""
    **Validation:**
    - Training: 2020-2024 (temporal separation)
    - Testing: 2025 (true out-of-sample)
    - Proxy validation: CitiBike vs. bike counters (r ≈ 0.85)

    **Uncertainty:**
    - Monte Carlo simulation (S=50)
    - Parameter + weather uncertainty
    - Exposure scenarios (±10%)
    """)

st.markdown("---")

# Data Pipeline
with st.expander("Data Pipeline Overview", expanded=False):
    st.markdown("""
    ### End-to-End Pipeline

    ```
    Raw Data Sources
    ├── CitiBike trips (2013-2025) ─────────────────┐
    ├── NYC crash data (NYPD) ──────────────────────┤
    ├── Weather (Open-Meteo hourly) ────────────────┤
    ├── Borough boundaries (NYC Open Data) ─────────┤
    └── Bike counters (NYC DOT) ────────────────────┘
                                                    │
                        ┌───────────────────────────┘
                        ▼
    Processing
    ├── Clean trips → exposure_cell_hour.parquet
    ├── Clean crashes → crash_cell_hour.parquet
    ├── Proxy validation (Borough × Month correlation)
    └── Grid training data (2020-2024)
                        │
                        ▼
    Modeling
    ├── Poisson GLM (baseline)
    ├── Negative Binomial GLM (main model)
    ├── Monte Carlo simulation (S=50)
    └── Exposure scenarios (±10%)
                        │
                        ▼
    Dashboard
    ├── Heatmaps (crashes, exposure, coverage)
    ├── Model comparison (Poisson vs. NegBin)
    ├── 2025 forecast vs. observed
    ├── Uncertainty quantification
    └── Proxy quality analysis
    ```

    **Reproducibility:** Run `make all` to execute the complete pipeline.
    """)

# Quick Stats
st.markdown("---")
st.subheader("Quick Navigation")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    **📊 Tripdata**

    Explore CitiBike usage patterns:
    - Trip volumes over time
    - Station activity
    - Seasonal trends
    """)

with col2:
    st.info("""
    **🚨 Crashdata**

    Analyze NYC bike crashes:
    - Temporal patterns
    - Geographic distribution
    - Severity breakdown
    """)

with col3:
    st.info("""
    **📈 Risk & Exposure**

    Model predictions & validation:
    - 2025 forecast
    - Uncertainty bounds
    - Proxy quality
    """)
