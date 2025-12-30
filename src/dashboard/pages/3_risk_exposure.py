# src/dashboard/pages/3_risk_exposure.py
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import folium
from streamlit_folium import st_folium


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Risk & Exposure (Bike crashes)", layout="wide")
st.title("Risk & Exposure — Bike crashes (Poisson vs Negative Binomial)")

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[3]
RISK_DIR = ROOT / "data" / "processed" / "risk_hourly_mc"
PROXY_DIR = ROOT / "data" / "processed" / "proxy_test"
RAW_DIR = ROOT / "data" / "raw"

CITY_TRAIN = RISK_DIR / "city_train_bike_all_2020_2024.parquet"
GRID_TRAIN = RISK_DIR / "grid_train_cell_hour_2020_2024.parquet"
EVAL_2025 = RISK_DIR / "risk_eval_2025_monthly_bike_all.parquet"
MC_2025 = RISK_DIR / "risk_mc_2025_totals_bike_all.parquet"
COMP = RISK_DIR / "model_comparison_bike_all.parquet"
PROXY_BM = PROXY_DIR / "proxy_test_borough_month.parquet"
BOROUGH_GEOJSON_PATH = RAW_DIR / "borough_boundaries.geojson"

# NEW: RAW GRID DATA (UNFILTERED - ALL CRASHES!)
CRASH_CELL_HOUR = RISK_DIR / "crash_cell_hour.parquet"
EXPOSURE_CELL_HOUR = RISK_DIR / "exposure_cell_hour.parquet"
GRID_2025 = RISK_DIR / "grid_2025_cell_hour_bike_all.parquet"


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def read_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def fmt_int(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{int(round(x)):,}"


def fmt_float(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def rate_per_100k(y: float, exposure_min: float) -> float:
    if exposure_min <= 0:
        return float("nan")
    return (y / exposure_min) * 100000.0


# -----------------------------
# Load COMPLETE grid data (2020-2025)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_complete_grid_data():
    """Load and combine ALL grid data from 2020-2025 (unfiltered)"""
    
    # Try to load raw crash and exposure data (2020-2024)
    if CRASH_CELL_HOUR.exists() and EXPOSURE_CELL_HOUR.exists():
        crashes = pd.read_parquet(CRASH_CELL_HOUR)
        exposure = pd.read_parquet(EXPOSURE_CELL_HOUR)
        
        # Merge crashes and exposure
        grid_2020_2024 = crashes.merge(
            exposure,
            on=['cell_id', 'grid_lat', 'grid_lng', 'hour_ts'],
            how='outer'
        )
        grid_2020_2024['y_bike'] = grid_2020_2024['y_bike'].fillna(0)
        grid_2020_2024['exposure_min'] = grid_2020_2024['exposure_min'].fillna(0)
    else:
        # Fallback to filtered training data
        grid_2020_2024 = read_parquet_safe(GRID_TRAIN)
    
    # Try to load 2025 data
    if GRID_2025.exists():
        grid_2025 = pd.read_parquet(GRID_2025)
        # Combine 2020-2024 with 2025
        grid_complete = pd.concat([grid_2020_2024, grid_2025], ignore_index=True)
    else:
        grid_complete = grid_2020_2024
    
    # Ensure hour_ts is datetime
    if 'hour_ts' in grid_complete.columns:
        grid_complete['hour_ts'] = pd.to_datetime(grid_complete['hour_ts'])
    
    return grid_complete


# ============================================================================
# SIDEBAR - FILTERS (Load grid data EARLY to detect years)
# ============================================================================
st.sidebar.header("🔍 Filters")

# Load COMPLETE grid data early
grid_complete = load_complete_grid_data()

# Year filter for heatmaps - DYNAMICALLY DETECT AVAILABLE YEARS
st.sidebar.subheader("Heatmap Time Range")
st.sidebar.caption("Select years to include in spatial heatmaps")

# Detect available years from complete grid data
@st.cache_data
def get_available_years(grid_df):
    """Extract available years from grid data"""
    if grid_df.empty or 'hour_ts' not in grid_df.columns:
        return [2020, 2021, 2022, 2023, 2024]
    
    years = pd.to_datetime(grid_df['hour_ts']).dt.year.unique()
    return sorted(years.tolist())

available_years = get_available_years(grid_complete)

st.sidebar.info(f"📊 Available years: **{min(available_years)}-{max(available_years)}**")

selected_years = st.sidebar.multiselect(
    "Select Years",
    options=available_years,
    default=available_years,
    help="Choose which years to include in the heatmap analysis"
)

if len(selected_years) == 0:
    st.sidebar.warning("⚠️ Please select at least one year")
    selected_years = available_years

st.sidebar.success(f"✅ Selected: **{min(selected_years)}-{max(selected_years)}**")

year_range_str = f"{min(selected_years)}-{max(selected_years)}" if len(selected_years) > 1 else str(selected_years[0])


# -----------------------------
# Load data (for sections A-E)
# -----------------------------
comp = read_parquet_safe(COMP)
eval_2025 = read_parquet_safe(EVAL_2025)
mc_2025 = read_parquet_safe(MC_2025)
train = read_parquet_safe(CITY_TRAIN)
proxy_bm = read_parquet_safe(PROXY_BM)

missing = []
for p in [COMP, EVAL_2025, MC_2025, CITY_TRAIN]:
    if not p.exists():
        missing.append(str(p))

if missing:
    st.error(
        "Missing required datasets. Please ensure your notebook wrote these files:\n\n"
        + "\n".join([f"- {m}" for m in missing])
    )
    st.stop()

# -----------------------------
# Clean / types
# -----------------------------
if "hour_ts" in train.columns:
    train["hour_ts"] = pd.to_datetime(train["hour_ts"])
if "month_ts" in eval_2025.columns:
    eval_2025["month_ts"] = pd.to_datetime(eval_2025["month_ts"])

train = train.dropna(subset=["exposure_min", "y_bike", "hour_of_day", "dow", "month"]).copy()
train = train[train["exposure_min"] > 0].copy()


# =============================
# A) High-level KPIs
# =============================
st.subheader("A) High-level summary (training window 2020–2024, forecast year 2025)")

obs_train_total = float(train["y_bike"].sum())
exp_train_total = float(train["exposure_min"].sum())
train_rate = rate_per_100k(obs_train_total, exp_train_total)

obs_2025_total = float(eval_2025.drop_duplicates(subset=["month_ts"])[["observed"]].fillna(0)["observed"].sum())

pred_2025 = (
    eval_2025.groupby("model", as_index=False)["pred_mean"]
    .sum()
    .rename(columns={"pred_mean": "pred_total_2025"})
)
pred_poisson = float(pred_2025.loc[pred_2025["model"] == "poisson", "pred_total_2025"].iloc[0]) if (pred_2025["model"] == "poisson").any() else float("nan")
pred_nb = float(pred_2025.loc[pred_2025["model"] == "neg_bin", "pred_total_2025"].iloc[0]) if (pred_2025["model"] == "neg_bin").any() else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observed bike crashes (2020–2024)", fmt_int(obs_train_total))
c2.metric("Exposure minutes (2020–2024)", fmt_int(exp_train_total))
c3.metric("Rate per 100k exposure-min (train)", fmt_float(train_rate, 3))
c4.metric("Observed bike crashes (2025)", fmt_int(obs_2025_total))

c5, c6 = st.columns(2)
c5.metric("Predicted 2025 total (Poisson mean)", fmt_int(pred_poisson))
c6.metric("Predicted 2025 total (NegBin mean)", fmt_int(pred_nb))

st.caption(
    "Interpretation: The model predicts the expected number of bike crashes per hour as "
    "μ = exposure_minutes × exp(linear_predictor). "
    "Exposure is a proxy based on Citi Bike trip time overlapped into hourly bins."
)


# =============================
# B) Model comparison
# =============================
st.subheader("B) Model comparison (Poisson vs Negative Binomial)")

comp_show = comp.copy()
for col in ["aic", "llf", "dispersion", "alpha"]:
    if col in comp_show.columns:
        comp_show[col] = pd.to_numeric(comp_show[col], errors="coerce")

if "dispersion" in comp_show.columns:
    comp_show["dispersion"] = comp_show["dispersion"].map(lambda x: float(x) if pd.notna(x) else np.nan)

st.dataframe(comp_show, width="stretch")

st.markdown(
    """
**How to choose:**
- If Poisson **dispersion (Pearson χ²/df)** is far above 1, the data are more variable than Poisson allows (overdispersion).
- Negative Binomial often fits better under overdispersion. Compare **AIC** (lower is better).
"""
)


# =============================
# C) Forecast 2025 vs observed
# =============================
st.subheader("C) Forecast 2025 vs observed (monthly totals)")

plot_df = eval_2025.copy()
plot_df["month_ts"] = pd.to_datetime(plot_df["month_ts"])

obs_line = eval_2025.drop_duplicates(subset=["month_ts"])[["month_ts", "observed"]].copy()
obs_line["series"] = "Observed"
obs_line.rename(columns={"observed": "value"}, inplace=True)

pred_line = plot_df[["month_ts", "model", "pred_mean"]].copy()
pred_line["series"] = pred_line["model"].map({"poisson": "Poisson (mean)", "neg_bin": "NegBin (mean)"}).fillna(pred_line["model"])
pred_line.rename(columns={"pred_mean": "value"}, inplace=True)

lines = pd.concat([obs_line[["month_ts", "series", "value"]], pred_line[["month_ts", "series", "value"]]], ignore_index=True)

chart = (
    alt.Chart(lines)
    .mark_line()
    .encode(
        x=alt.X("month_ts:T", title="Month (2025)"),
        y=alt.Y("value:Q", title="Bike crashes (monthly total)"),
        color=alt.Color("series:N", title="Series"),
        tooltip=["month_ts:T", "series:N", alt.Tooltip("value:Q", format=",.0f")],
    )
    .properties(height=360)
)
st.altair_chart(chart, width="stretch")


# =============================
# D) Uncertainty (Monte Carlo)
# =============================
st.subheader("D) Uncertainty of total 2025 crashes (Monte Carlo)")

mc = mc_2025.copy()
if mc.empty or "total_2025" not in mc.columns:
    st.warning("Monte Carlo dataset is missing or invalid.")
else:
    mc["total_2025"] = pd.to_numeric(mc["total_2025"], errors="coerce")
    mc = mc.dropna(subset=["total_2025"]).copy()

    q = (
        mc.groupby("model")["total_2025"]
        .quantile([0.05, 0.5, 0.95])
        .unstack()
        .reset_index()
        .rename(columns={0.05: "q05", 0.5: "q50", 0.95: "q95"})
    )

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Simulation quantiles (total 2025):**")
        st.dataframe(q, width="stretch")

    with right:
        st.markdown("**Distribution (histogram):**")
        hist = (
            alt.Chart(mc)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("total_2025:Q", bin=alt.Bin(maxbins=40), title="Total crashes in 2025"),
                y=alt.Y("count():Q", title="Simulation count"),
                color=alt.Color("model:N", title="Model"),
                tooltip=["model:N", alt.Tooltip("count():Q", title="count")],
            )
            .properties(height=320)
        )
        st.altair_chart(hist, width="stretch")


# =============================
# E) Training patterns
# =============================
st.subheader("E) What drives the risk rate? (empirical patterns in 2020–2024)")

def make_rate_chart(group_col: str, title: str, x_title: str):
    g = (
        train.groupby(group_col, as_index=False)
        .agg(y=("y_bike", "sum"), exposure_min=("exposure_min", "sum"))
    )
    g["rate_100k"] = g.apply(lambda r: rate_per_100k(r["y"], r["exposure_min"]), axis=1)
    g[group_col] = g[group_col].astype(int)

    ch = (
        alt.Chart(g)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{group_col}:O", title=x_title),
            y=alt.Y("rate_100k:Q", title="Crashes per 100k exposure-min"),
            tooltip=[group_col, alt.Tooltip("y:Q", format=",.0f"), alt.Tooltip("exposure_min:Q", format=",.0f"), alt.Tooltip("rate_100k:Q", format=",.3f")],
        )
        .properties(height=280, title=title)
    )
    return ch

col1, col2 = st.columns(2)
with col1:
    st.altair_chart(make_rate_chart("hour_of_day", "Crash rate by hour of day", "Hour of day"), width="stretch")
with col2:
    st.altair_chart(make_rate_chart("dow", "Crash rate by day of week (0=Sun)", "Day of week"), width="stretch")

col3, col4 = st.columns(2)
with col3:
    st.altair_chart(make_rate_chart("month", "Crash rate by month", "Month"), width="stretch")

with col4:
    wdf = train[["prcp", "y_bike", "exposure_min"]].copy()
    wdf["prcp"] = pd.to_numeric(wdf["prcp"], errors="coerce")
    wdf = wdf.dropna(subset=["prcp"]).copy()
    wdf["prcp_bin"] = pd.qcut(wdf["prcp"], q=10, duplicates="drop")
    gb = wdf.groupby("prcp_bin", as_index=False).agg(y=("y_bike","sum"), exposure_min=("exposure_min","sum"))
    gb["rate_100k"] = gb.apply(lambda r: rate_per_100k(r["y"], r["exposure_min"]), axis=1)
    gb["prcp_bin_label"] = gb["prcp_bin"].astype(str)

    ch = (
        alt.Chart(gb)
        .mark_bar()
        .encode(
            x=alt.X("prcp_bin_label:N", sort=None, title="Precipitation bin (train quantiles)"),
            y=alt.Y("rate_100k:Q", title="Crashes per 100k exposure-min"),
            tooltip=["prcp_bin_label:N", alt.Tooltip("y:Q", format=",.0f"), alt.Tooltip("exposure_min:Q", format=",.0f"), alt.Tooltip("rate_100k:Q", format=",.3f")],
        )
        .properties(height=280, title="Crash rate vs precipitation (binned)")
    )
    st.altair_chart(ch, width="stretch")

st.caption(
    "These plots show *empirical* rates in the training period. "
    "The GLM learns a smooth version of these patterns (plus weather effects) to forecast future periods."
)


# =============================
# F) SPATIAL HEATMAPS - COMPLETE DATA 2020-2025
# =============================
st.subheader(f"F) Spatial Distribution: NYC Heatmaps ({year_range_str})")

# Load and extract borough boundaries
if BOROUGH_GEOJSON_PATH.exists():
    with open(BOROUGH_GEOJSON_PATH, 'r') as f:
        borough_geojson = json.load(f)
    st.success(f"✅ Loaded {len(borough_geojson.get('features', []))} boroughs")
    
    # Extract boundary coordinates as simple lists for PolyLine rendering
    @st.cache_data
    def extract_borough_boundaries(geojson):
        """Extract coordinates from GeoJSON for simple polyline rendering"""
        boundaries = []
        for feature in geojson.get('features', []):
            geom = feature.get('geometry', {})
            geom_type = geom.get('type')
            coords = geom.get('coordinates', [])
            
            if geom_type == 'Polygon':
                for ring in coords:
                    boundary = [[lat, lng] for lng, lat in ring]
                    boundaries.append(boundary)
            
            elif geom_type == 'MultiPolygon':
                for polygon in coords:
                    for ring in polygon:
                        boundary = [[lat, lng] for lng, lat in ring]
                        boundaries.append(boundary)
        
        return boundaries
    
    borough_boundaries = extract_borough_boundaries(borough_geojson)
    st.info(f"📍 Extracted {len(borough_boundaries)} boundary lines")
else:
    st.error(f"❌ GeoJSON not found at: {BOROUGH_GEOJSON_PATH}")
    borough_boundaries = []

if grid_complete.empty:
    st.info("Grid data not found. Skipping spatial heatmaps.")
else:
    st.markdown(f"""
    **2D Maps of New York City showing:**
    - Grid cells colored by intensity (darker = higher values)
    - **Real NYC Borough boundaries** overlaid
    - **Time period: {year_range_str}**
    - **Complete dataset (all crashes, unfiltered)**
    """)
    
    # ============================================================================
    # FILTER GRID DATA BY SELECTED YEARS
    # ============================================================================
    grid_filtered = grid_complete.copy()
    
    if 'hour_ts' in grid_filtered.columns:
        grid_filtered['year'] = pd.to_datetime(grid_filtered['hour_ts']).dt.year
        grid_filtered = grid_filtered[grid_filtered['year'].isin(selected_years)]
        
        total_crashes = grid_filtered['y_bike'].sum()
        st.success(f"🎯 Filtered to **{total_crashes:,.0f} total crashes** from {len(grid_filtered):,} cell-hours in years {selected_years}")
    else:
        st.warning("⚠️ No timestamp column found - showing all data")
    
    # Prepare heatmap data
    @st.cache_data
    def prepare_heatmap_data(grid_df, years_tuple):
        """Aggregate grid data to cell level - KEEP ALL CELLS!"""
        heatmap = grid_df.groupby(['grid_lat', 'grid_lng', 'cell_id'], as_index=False).agg({
            'y_bike': 'sum',
            'exposure_min': 'sum'
        })
        
        heatmap['rate_per_M_min'] = (
            heatmap['y_bike'] / (heatmap['exposure_min'] / 1e6)
        ).fillna(0)
        
        # KEEP ALL CELLS - only filter out cells with ZERO exposure
        heatmap = heatmap[heatmap['exposure_min'] > 0].copy()
        
        heatmap['exposure_M_min'] = heatmap['exposure_min'] / 1e6
        
        return heatmap
    
    heatmap_data = prepare_heatmap_data(grid_filtered, tuple(selected_years))
    
    if len(heatmap_data) == 0:
        st.warning("No grid cells with data for visualization.")
    else:
        st.info(f"📍 Showing **{len(heatmap_data):,} grid cells** with **{heatmap_data['y_bike'].sum():,.0f} total crashes**")
        
        # Helper function to create colored map
        def create_nyc_map(data, value_col, title, color_scale='YlOrRd'):
            """Create a folium map with colored grid cells"""
            import matplotlib.cm
            import matplotlib.colors
            
            nyc_center = [40.7580, -73.9855]
            
            m = folium.Map(
                location=nyc_center,
                zoom_start=11,
                tiles='OpenStreetMap',
                control_scale=True
            )
            
            # Add borough boundaries as SIMPLE POLYLINES
            for boundary in borough_boundaries:
                folium.PolyLine(
                    locations=boundary,
                    color='black',
                    weight=2,
                    opacity=0.8
                ).add_to(m)
            
            # Normalize values for coloring
            vmin = data[value_col].min()
            vmax = data[value_col].max()
            
            cmap = matplotlib.cm.get_cmap(color_scale)
            
            # Add grid cells as rectangles
            GRID_SIZE = 0.025
            
            for idx, row in data.iterrows():
                lat = row['grid_lat']
                lng = row['grid_lng']
                value = row[value_col]
                
                normalized = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                rgba = cmap(normalized)
                color = matplotlib.colors.rgb2hex(rgba[:3])
                
                bounds = [
                    [lat, lng],
                    [lat + GRID_SIZE, lng],
                    [lat + GRID_SIZE, lng + GRID_SIZE],
                    [lat, lng + GRID_SIZE]
                ]
                
                folium.Polygon(
                    locations=bounds,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6,
                    weight=1,
                    popup=folium.Popup(
                        f"<b>Grid Cell</b><br>"
                        f"Crashes: {row['y_bike']:,.0f}<br>"
                        f"Exposure: {row['exposure_M_min']:.2f}M min<br>"
                        f"Rate: {row['rate_per_M_min']:.2f} per M min",
                        max_width=200
                    )
                ).add_to(m)
            
            # Add title
            title_html = f'''
            <div style="position: fixed; 
                        top: 10px; left: 50px; width: 400px; height: 50px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:16px; padding: 10px">
                <b>{title}</b>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            
            return m
        
        # Create tabs for 3 maps
        tab1, tab2, tab3 = st.tabs(["🚨 Crashes", "🚴 Exposure", "⚠️ Risk Rate"])
        
        # TAB 1: CRASHES MAP
        with tab1:
            st.markdown(f"**Total Crashes by Grid Cell ({year_range_str})**")
            st.caption("Darker red = more bike crashes")
            
            crashes_map = create_nyc_map(
                heatmap_data,
                value_col='y_bike',
                title=f'Bike Crashes ({year_range_str})',
                color_scale='YlOrRd'
            )
            
            st_folium(crashes_map, width=1200, height=600, key="crashes", returned_objects=[])
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Crashes", f"{heatmap_data['y_bike'].sum():,.0f}")
            col_b.metric("Max (single cell)", f"{heatmap_data['y_bike'].max():,.0f}")
            col_c.metric("Median (per cell)", f"{heatmap_data['y_bike'].median():,.0f}")
        
        # TAB 2: EXPOSURE MAP
        with tab2:
            st.markdown(f"**CitiBike Exposure by Grid Cell ({year_range_str})**")
            st.caption("Darker blue = more CitiBike usage (trip minutes)")
            
            exposure_map = create_nyc_map(
                heatmap_data,
                value_col='exposure_M_min',
                title=f'CitiBike Exposure ({year_range_str})',
                color_scale='YlGnBu'
            )
            
            st_folium(exposure_map, width=1200, height=600, key="exposure", returned_objects=[])
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Exposure", f"{heatmap_data['exposure_M_min'].sum():,.0f}M min")
            col_b.metric("Max (single cell)", f"{heatmap_data['exposure_M_min'].max():,.0f}M min")
            col_c.metric("Median (per cell)", f"{heatmap_data['exposure_M_min'].median():.1f}M min")
        
        # TAB 3: RISK RATE MAP
        with tab3:
            st.markdown(f"**Risk Rate: Crashes per Million Minutes ({year_range_str})**")
            st.caption("Darker red/orange = higher crash rate (more dangerous)")
            
            rate_cap = heatmap_data['rate_per_M_min'].quantile(0.99)
            heatmap_data_capped = heatmap_data.copy()
            heatmap_data_capped['rate_capped'] = heatmap_data_capped['rate_per_M_min'].clip(upper=rate_cap)
            
            rate_map = create_nyc_map(
                heatmap_data_capped,
                value_col='rate_capped',
                title=f'Risk Rate ({year_range_str})',
                color_scale='YlOrRd'
            )
            
            st_folium(rate_map, width=1200, height=600, key="risk", returned_objects=[])
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Median Rate", f"{heatmap_data['rate_per_M_min'].median():.2f}")
            col_b.metric("90th Percentile", f"{heatmap_data['rate_per_M_min'].quantile(0.90):.2f}")
            col_c.metric("Max Rate", f"{heatmap_data['rate_per_M_min'].max():.2f}")
        
        st.markdown("---")
        st.caption(f"""
        **Map Details:**
        - Grid cells: 2.5km × 2.5km (GRID_DEG=0.025)
        - Black lines: **Real NYC Borough boundaries**
        - **Time period: {year_range_str}**
        - **Complete dataset: All crashes (no MIN_EXPOSURE filtering)**
        - Click on grid cells for details
        """)


# =============================
# G) Borough allocation (optional)
# =============================
st.subheader("G) Borough allocation (optional, using proxy shares)")

if proxy_bm.empty:
    st.info("Proxy borough-month dataset not found. Skipping borough allocation.")
else:
    pbm = proxy_bm.copy()
    if "month_ts" in pbm.columns:
        pbm["month_ts"] = pd.to_datetime(pbm["month_ts"])
    elif "month" in pbm.columns:
        pbm["month_ts"] = pd.to_datetime(pbm["month"])
    else:
        st.info("Proxy dataset has no recognizable month column. Skipping.")
        pbm = pd.DataFrame()

    if not pbm.empty and "borough" in pbm.columns and "share_idx" in pbm.columns:
        pbm["borough"] = pbm["borough"].astype(str).str.upper()
        pbm["share_idx"] = pd.to_numeric(pbm["share_idx"], errors="coerce")

        pbm_2024 = pbm[(pbm["month_ts"] >= "2024-01-01") & (pbm["month_ts"] < "2025-01-01")].copy()
        if pbm_2024.empty:
            st.warning("No 2024 proxy data found; using all available months as weights.")
            weights = pbm.groupby("borough", as_index=False)["share_idx"].mean()
        else:
            weights = pbm_2024.groupby("borough", as_index=False)["share_idx"].mean()

        weights = weights.dropna(subset=["share_idx"]).copy()
        weights["w"] = weights["share_idx"] / weights["share_idx"].sum()

        model_choice = st.radio("Allocate using model:", ["neg_bin", "poisson"], horizontal=True, index=0)

        annual_pred = float(pred_2025.loc[pred_2025["model"] == model_choice, "pred_total_2025"].iloc[0]) if (pred_2025["model"] == model_choice).any() else float("nan")
        if math.isnan(annual_pred) or annual_pred <= 0:
            st.warning("Could not compute annual prediction for the selected model.")
        else:
            alloc = weights.copy()
            alloc["pred_2025_alloc"] = alloc["w"] * annual_pred
            alloc = alloc.sort_values("pred_2025_alloc", ascending=False)

            st.markdown("**Allocated annual 2025 prediction (using proxy weights):**")
            ch = (
                alt.Chart(alloc)
                .mark_bar()
                .encode(
                    x=alt.X("borough:N", title="Borough", sort="-y"),
                    y=alt.Y("pred_2025_alloc:Q", title="Allocated crashes (2025 total)"),
                    tooltip=["borough:N", alt.Tooltip("w:Q", format=".3f", title="weight"), alt.Tooltip("pred_2025_alloc:Q", format=",.0f")],
                )
                .properties(height=320)
            )
            st.altair_chart(ch, width="stretch")

            st.caption(
                "Allocation uses borough-level proxy weights (share index). "
                "This is not an attribution of specific crashes to Citi Bike—it's a distribution of the *citywide* forecast according to exposure shares."
            )
    else:
        st.info("Proxy dataset missing required columns (borough, share_idx). Skipping.")