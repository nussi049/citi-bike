# src/dashboard/pages/3_risk_exposure.py
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import matplotlib.cm
import matplotlib.colors
import streamlit as st
import altair as alt
import folium
from streamlit_folium import st_folium

from src.dashboard.lib.settings import (
    BOROUGH_GEOJSON,
    CRASH_CELL_HOUR,
    EXPOSURE_CELL_HOUR,
    GRID_TRAIN,
    EVAL_2025,
    MC_2025,
    COMP,
)


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Risk & Exposure (Bike crashes)", layout="wide")
st.title("Risk & Exposure")


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def read_parquet_safe(path) -> pd.DataFrame:
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
    """Load and combine crash + exposure data (2020-2025 complete)"""
    
    if not CRASH_CELL_HOUR.exists() or not EXPOSURE_CELL_HOUR.exists():
        st.error(f"Missing required files:\n- {CRASH_CELL_HOUR}\n- {EXPOSURE_CELL_HOUR}")
        return pd.DataFrame()
    
    # Load crashes and exposure separately
    crashes = pd.read_parquet(CRASH_CELL_HOUR)
    exposure = pd.read_parquet(EXPOSURE_CELL_HOUR)
    
    # CRITICAL: Normalize coordinates to same precision BEFORE merge!
    # Round to 6 decimal places (~0.1 meter precision)
    crashes['grid_lat'] = crashes['grid_lat'].round(6)
    crashes['grid_lng'] = crashes['grid_lng'].round(6)
    exposure['grid_lat'] = exposure['grid_lat'].round(6)
    exposure['grid_lng'] = exposure['grid_lng'].round(6)
    
    # Regenerate cell_id from normalized coordinates
    crashes['cell_id'] = (
        crashes['grid_lat'].astype(str) + '_' + crashes['grid_lng'].astype(str)
    )
    exposure['cell_id'] = (
        exposure['grid_lat'].astype(str) + '_' + exposure['grid_lng'].astype(str)
    )
    
    # Ensure hour_ts is datetime BEFORE aggregation
    crashes['hour_ts'] = pd.to_datetime(crashes['hour_ts'])
    exposure['hour_ts'] = pd.to_datetime(exposure['hour_ts'])
    
    # AGGREGATE to handle duplicates created by rounding!
    crashes_agg = crashes.groupby(
        ['cell_id', 'grid_lat', 'grid_lng', 'hour_ts'], 
        as_index=False
    ).agg({'y_bike': 'sum'})
    
    exposure_agg = exposure.groupby(
        ['cell_id', 'grid_lat', 'grid_lng', 'hour_ts'], 
        as_index=False
    ).agg({'exposure_min': 'sum'})
    
    # NOW merge (should be unique after aggregation!)
    grid_complete = crashes_agg.merge(
        exposure_agg,
        on=['cell_id', 'grid_lat', 'grid_lng', 'hour_ts'],
        how='outer'
    )
    
    # Fill missing values
    grid_complete['y_bike'] = grid_complete['y_bike'].fillna(0)
    grid_complete['exposure_min'] = grid_complete['exposure_min'].fillna(0)
    
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
        return [2020, 2021, 2022, 2023, 2024, 2025]
    
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


# =============================
# F) SPATIAL HEATMAPS - COMPLETE DATA 2020-2025
# =============================
st.subheader(f"Spatial Distribution: NYC Heatmaps ({year_range_str})")

# Load and extract borough boundaries
if BOROUGH_GEOJSON.exists():
    with open(BOROUGH_GEOJSON, 'r') as f:
        borough_geojson = json.load(f)
    
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
else:
    st.error(f"❌ GeoJSON not found at: {BOROUGH_GEOJSON}")
    borough_boundaries = []

if grid_complete.empty:
    st.info("Grid data not found. Skipping spatial heatmaps.")
else:
    st.markdown(f"""
    **2D Maps of New York City showing:**
    - Grid cells colored by intensity (darker = higher values)
    - **Real NYC Borough boundaries** overlaid
    - **Time period: {year_range_str}**
    - **Note:** CitiBike exposure is only a proxy for total bike activity (see Model Limitations below)
    """)
    
    # ============================================================================
    # FILTER GRID DATA BY SELECTED YEARS
    # ============================================================================
    grid_filtered = grid_complete.copy()
    
    if 'hour_ts' in grid_filtered.columns:
        grid_filtered['year'] = pd.to_datetime(grid_filtered['hour_ts']).dt.year
        grid_filtered = grid_filtered[grid_filtered['year'].isin(selected_years)]
        
        total_crashes = grid_filtered['y_bike'].sum()
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
        
        heatmap['exposure_M_min'] = heatmap['exposure_min'] / 1e6
        
        return heatmap
    
    heatmap_data = prepare_heatmap_data(grid_filtered, tuple(selected_years))
    
    if len(heatmap_data) == 0:
        st.warning("No grid cells with data for visualization.")
    else:        
        # Helper function to create colored map
        def create_nyc_map(data, value_col, title, color_scale='YlOrRd'):
            """Create a folium map with colored grid cells"""

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
                        f"Exposure: {row['exposure_M_min']:.2f}M min",
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
        
        # Create tabs for 2 maps
        tab1, tab2 = st.tabs(["🚨 Crashes", "🚴 Exposure"])
        
        # TAB 1: CRASHES MAP
        with tab1:
            st.markdown(f"**Total Crashes by Grid Cell ({year_range_str})**")
            st.caption("Darker red = more bike crashes | Shows ALL crashes (incl. areas without CitiBike)")
            
            # Filter: Show all cells with crashes
            crashes_data = heatmap_data[heatmap_data['y_bike'] > 0].copy()
            
            if len(crashes_data) == 0:
                st.warning("No crashes in selected time period")
            else:
                crashes_map = create_nyc_map(
                    crashes_data,
                    value_col='y_bike',
                    title=f'Bike Crashes ({year_range_str})',
                    color_scale='YlOrRd'
                )
                
                st_folium(crashes_map, width=1200, height=600, key="crashes", returned_objects=[])
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total Crashes", f"{crashes_data['y_bike'].sum():,.0f}")
                col_b.metric("Max (single cell)", f"{crashes_data['y_bike'].max():,.0f}")
                col_c.metric("Median (per cell)", f"{crashes_data['y_bike'].median():,.0f}")
                
                # Show stats about crashes outside CitiBike coverage
                crashes_no_exposure = crashes_data[crashes_data['exposure_min'] == 0]
        
        # TAB 2: EXPOSURE MAP
        with tab2:
            st.markdown(f"**CitiBike Exposure by Grid Cell ({year_range_str})**")
            st.caption("Darker blue = more CitiBike usage (trip minutes) | Only CitiBike coverage areas")
            
            # Filter: Only cells with CitiBike exposure
            exposure_data = heatmap_data[heatmap_data['exposure_min'] > 0].copy()
            
            if len(exposure_data) == 0:
                st.warning("No exposure data in selected time period")
            else:
                exposure_map = create_nyc_map(
                    exposure_data,
                    value_col='exposure_M_min',
                    title=f'CitiBike Exposure ({year_range_str})',
                    color_scale='YlGnBu'
                )
                
                st_folium(exposure_map, width=1200, height=600, key="exposure", returned_objects=[])
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total Exposure", f"{exposure_data['exposure_M_min'].sum():,.0f}M min")
                col_b.metric("Max (single cell)", f"{exposure_data['exposure_M_min'].max():,.0f}M min")
                col_c.metric("Median (per cell)", f"{exposure_data['exposure_M_min'].median():.1f}M min")
        
        st.markdown("---")
        st.caption(f"""
        **Map Details:**
        - Grid cells: 2.5km × 2.5km (GRID_DEG=0.025)
        - Black lines: **Real NYC Borough boundaries**
        - **Time period: {year_range_str}**
        - **Crashes Tab:** All crashes (incl. areas without CitiBike)
        - **Exposure Tab:** Only CitiBike coverage areas
        - **Important:** CitiBike exposure is a proxy for total bike activity—not all bike trips!
        - Click on grid cells for details
        """)


# =============================
# NOW THE MODEL SECTIONS (A-E)
# =============================
st.markdown("---")
st.header("Model Analysis & Forecast")

# -----------------------------
# Load data (for sections A-E)
# -----------------------------
comp = read_parquet_safe(COMP)
eval_2025 = read_parquet_safe(EVAL_2025)
mc_2025 = read_parquet_safe(MC_2025)
train = read_parquet_safe(GRID_TRAIN)

missing = []
for p in [COMP, EVAL_2025, MC_2025, GRID_TRAIN]:
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

# FIXED: Use grid_complete for ACTUAL crash/exposure totals (consistent with maps)
if not grid_complete.empty:
    grid_2020_2024 = grid_complete[
        (grid_complete['hour_ts'] >= '2020-01-01') & 
        (grid_complete['hour_ts'] < '2025-01-01')
    ].copy()
    
    # Total observed crashes (ALL areas, including non-CitiBike)
    obs_train_total = float(grid_2020_2024['y_bike'].sum())
    
    # Total exposure (only CitiBike areas)
    exp_train_total = float(grid_2020_2024[grid_2020_2024['exposure_min'] > 0]['exposure_min'].sum())
    
    # Rate calculated on CitiBike areas only
    crashes_with_exposure = grid_2020_2024[grid_2020_2024['exposure_min'] > 0]
    train_rate = rate_per_100k(
        float(crashes_with_exposure['y_bike'].sum()),
        float(crashes_with_exposure['exposure_min'].sum())
    )
else:
    obs_train_total = exp_train_total = train_rate = float('nan')

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
    "**Model:** μ = exposure_minutes × exp(linear_predictor). "
    "CitiBike exposure is used as a proxy for total bike activity in areas with CitiBike coverage. "
    "**Note:** The model only predicts crashes in CitiBike coverage areas (~95% of all NYC bike crashes)."
)

# Add model limitation explanation
with st.expander("ℹ️ Model Limitations & Why It Works"):
    st.markdown("""
    **Important Model Limitation:**
    
    The model only predicts crashes in areas with CitiBike coverage (Manhattan, Brooklyn, Queens core). 
    However, validation shows that ~95% of all NYC bike crashes occur in these areas, so citywide 
    predictions remain accurate.
    
    **Why this works:**
    - CitiBike operates in the densest, highest-traffic areas of NYC
    - These areas have the most cyclists (both CitiBike and non-CitiBike)
    - Areas without CitiBike (Staten Island, outer Bronx) have significantly fewer cyclists and crashes
    - The model predicts μ = 0 for cells without CitiBike exposure (automatically excluding those areas)
    
    **Result:** The model's citywide predictions closely match actual citywide totals, even though 
    it only models CitiBike coverage areas. The ~5% of crashes outside coverage contribute a small, 
    roughly constant error term.
    """)


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

st.markdown("""
**How to choose:**
- If Poisson **dispersion (Pearson χ²/df)** is far above 1, the data are more variable than Poisson allows (overdispersion).
- Negative Binomial often fits better under overdispersion. Compare **AIC** (lower is better).
""")


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
        .properties(height=320, title=title)
    )
    return ch

col1, col2 = st.columns(2)
with col1:
    st.altair_chart(make_rate_chart("hour_of_day", "Crash rate by hour of day", "Hour of day"), use_container_width=True)
with col2:
    st.altair_chart(make_rate_chart("dow", "Crash rate by day of week (0=Sun)", "Day of week"), use_container_width=True)

# Add vertical spacing between chart rows
st.markdown("")
st.markdown("")

col3, col4 = st.columns(2)
with col3:
    st.altair_chart(make_rate_chart("month", "Crash rate by month", "Month"), use_container_width=True)

with col4:
    if 'prcp' in train.columns:
        wdf = train[["prcp", "y_bike", "exposure_min"]].copy()
        wdf["prcp"] = pd.to_numeric(wdf["prcp"], errors="coerce")
        wdf = wdf.dropna(subset=["prcp"]).copy()
        
        if len(wdf) > 100:
            # Simple categories: No rain vs Rain
            wdf["rain_category"] = pd.cut(
                wdf["prcp"],
                bins=[-0.01, 0.01, 1.0, 5.0, float('inf')],
                labels=["No rain (0 mm)", "Light rain (0-1 mm)", "Moderate rain (1-5 mm)", "Heavy rain (5+ mm)"]
            )
            
            # Aggregate by rain category
            gb = wdf.groupby("rain_category", as_index=False, observed=True).agg(
                n_hours=("prcp", "count"),
                y=("y_bike", "sum"),
                exposure_min=("exposure_min", "sum")
            )
            
            # Calculate crash rate
            gb["rate_100k"] = gb.apply(lambda r: rate_per_100k(r["y"], r["exposure_min"]), axis=1)

            # EXPLICIT TITLE BEFORE CHART
            st.markdown("**Crash rate vs precipitation**")
            
            ch = (
                alt.Chart(gb)
                .mark_line(point=True)
                .encode(
                    x=alt.X("rain_category:N", title="Precipitation level", sort=None),
                    y=alt.Y("rate_100k:Q", title="Crashes per 100k exposure-min"),
                    tooltip=[
                        alt.Tooltip("rain_category:N", title="Category"),
                        alt.Tooltip("n_hours:Q", format=",", title="Hours in period"),
                        alt.Tooltip("y:Q", format=",.0f", title="Total crashes"),
                        alt.Tooltip("rate_100k:Q", format=".2f", title="Crash rate")
                    ],
                )
                .properties(height=280)  # Removed title from properties
            )
            st.altair_chart(ch, use_container_width=True)
            
        else:
            st.info("Insufficient data for precipitation analysis")
    else:
        st.info("Weather data not available in training dataset")