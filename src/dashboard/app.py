import os
import streamlit as st

st.set_page_config(page_title="Citi Bike Risk Dashboard", layout="wide")

# =============================================================================
# PASSWORD PROTECTION (optional)
# Set DASHBOARD_PASSWORD environment variable to enable
#
# NOTE: This is a simple protection for the coding challenge demo.
# NOT brute-force resistant - no rate limiting, no account lockout.
# For production use: implement proper auth (OAuth, SSO, etc.)
# =============================================================================
def check_password() -> bool:
    """Returns True if no password set or user authenticated."""
    password = os.environ.get("DASHBOARD_PASSWORD")
    if not password:
        return True

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("NYC Bike Crash Dashboard")
    st.markdown("---")

    with st.form("login"):
        pwd = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if pwd == password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
    return False

if not check_password():
    st.stop()

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
    Predict bike crash counts for NYC grid cells using a Poisson GLM
    with CitiBike exposure as feature.

    **Key Features:**
    - Temporal effects (day-of-week, month)
    - Spatial effects (lat/lng with quadratic terms)
    - Weather covariates (temperature, precipitation, wind)
    - Trend term for declining crash rate
    - Exposure as feature (log1p) with estimated elasticity
    """)

with col2:
    st.markdown("""
    **Validation:**
    - Training: 2021-2024 (excludes COVID 2020)
    - Backtest: 2025 (using known exposure)
    - Proxy validation: CitiBike vs. bike counters (r ≈ 0.85)

    **Uncertainty:**
    - Monte Carlo simulation (S=1000)
    - 4 dimensions: weather, exposure year, growth (±20%), parameters
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
    └── Grid training data (2021-2024)
                        │
                        ▼
    Modeling
    ├── Poisson GLM (dispersion ~1)
    ├── Monte Carlo simulation (S=1000)
    ├── 4 uncertainty dimensions
                        │
                        ▼
    Dashboard
    ├── Heatmaps (crashes, exposure, coverage)
    ├── Poisson model diagnostics
    ├── 2025 forecast vs. observed
    ├── Uncertainty quantification
    └── Proxy quality analysis
    ```

    **Reproducibility:** Run `make all` to execute the complete pipeline.
    """)

