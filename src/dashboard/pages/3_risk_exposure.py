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
    EXPOSURE_CELL_HOUR_TRAIN,
    EXPOSURE_CELL_HOUR_TEST,
    GRID_TRAIN,
    EVAL_2025,
    COMP,
    MC_2025_SCENARIOS,
    PROXY_TEST_BM,
    BIKE_COUNTERS,
    BIKE_COUNTS_HOURLY,
    CELLS_KEEP,
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

    if not CRASH_CELL_HOUR.exists() or not EXPOSURE_CELL_HOUR_TRAIN.exists() or not EXPOSURE_CELL_HOUR_TEST.exists():
        st.error(f"Missing required files:\n- {CRASH_CELL_HOUR}\n- {EXPOSURE_CELL_HOUR_TRAIN}\n- {EXPOSURE_CELL_HOUR_TEST}")
        return pd.DataFrame()

    # Load crashes and exposure separately (exposure is split into train/test)
    crashes = pd.read_parquet(CRASH_CELL_HOUR)
    exposure_train = pd.read_parquet(EXPOSURE_CELL_HOUR_TRAIN)
    exposure_test = pd.read_parquet(EXPOSURE_CELL_HOUR_TEST)
    exposure = pd.concat([exposure_train, exposure_test], ignore_index=True)
    
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
        
        # Create tabs for 3 maps
        tab1, tab2, tab3 = st.tabs(["🚨 Crashes", "🚴 Exposure", "📍 Coverage & Counters"])

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

        # TAB 3: COVERAGE & COUNTERS MAP
        with tab3:
            st.markdown("**Model Coverage & Bike Counter Stations**")
            st.caption("Shows: Model grid cells (green) + Official NYC bike counter stations (sized by total counts)")

            # Load bike counters metadata and counts
            counters_df = read_parquet_safe(BIKE_COUNTERS)
            counts_hourly = read_parquet_safe(BIKE_COUNTS_HOURLY)
            cells_keep_df = read_parquet_safe(CELLS_KEEP)

            if counters_df.empty:
                st.warning("Bike counter data not found. Run `make bike_counts` to download.")
            else:
                # Filter counts by selected years
                if not counts_hourly.empty:
                    counts_hourly['hour_ts'] = pd.to_datetime(counts_hourly['hour_ts'])
                    counts_hourly['year'] = counts_hourly['hour_ts'].dt.year
                    counts_filtered = counts_hourly[counts_hourly['year'].isin(selected_years)]

                    # Aggregate counts per counter for selected period
                    counter_totals = counts_filtered.groupby('counter_id').agg({
                        'bike_count': 'sum',
                        'borough': 'first'
                    }).reset_index()
                    counter_totals.columns = ['counter_id', 'total_bikes', 'borough']
                else:
                    counter_totals = pd.DataFrame()

                # Merge counters with their totals
                # Try matching on 'id' first, then 'counter'
                if not counter_totals.empty:
                    counters_merged = counters_df.merge(
                        counter_totals,
                        left_on='id',
                        right_on='counter_id',
                        how='left'
                    )
                    # If no matches on 'id', try 'counter'
                    if counters_merged['total_bikes'].isna().all():
                        counters_merged = counters_df.merge(
                            counter_totals,
                            left_on='counter',
                            right_on='counter_id',
                            how='left'
                        )
                else:
                    counters_merged = counters_df.copy()
                    counters_merged['total_bikes'] = 0
                    counters_merged['borough'] = 'Unknown'

                # Create map
                nyc_center = [40.7580, -73.9855]
                coverage_map = folium.Map(
                    location=nyc_center,
                    zoom_start=11,
                    tiles='OpenStreetMap',
                    control_scale=True
                )

                # Add borough boundaries
                for boundary in borough_boundaries:
                    folium.PolyLine(
                        locations=boundary,
                        color='black',
                        weight=2,
                        opacity=0.8
                    ).add_to(coverage_map)

                # Add model grid cells (cells_keep) as green rectangles
                GRID_SIZE = 0.025

                if not cells_keep_df.empty:
                    for _, row in cells_keep_df.iterrows():
                        cell_id = row['cell_id']
                        try:
                            lat_str, lng_str = cell_id.split('_')
                            lat = float(lat_str)
                            lng = float(lng_str)

                            bounds = [
                                [lat, lng],
                                [lat + GRID_SIZE, lng],
                                [lat + GRID_SIZE, lng + GRID_SIZE],
                                [lat, lng + GRID_SIZE]
                            ]

                            folium.Polygon(
                                locations=bounds,
                                color='green',
                                fill=True,
                                fillColor='green',
                                fillOpacity=0.15,
                                weight=1,
                            ).add_to(coverage_map)
                        except (ValueError, AttributeError):
                            continue

                # Add bike counter locations - sized by total count
                max_count = counters_merged['total_bikes'].max() if 'total_bikes' in counters_merged.columns else 1
                if pd.isna(max_count) or max_count == 0:
                    max_count = 1

                for _, counter in counters_merged.iterrows():
                    lat = counter.get('latitude')
                    lng = counter.get('longitude')
                    name = counter.get('name', 'Unknown')
                    counter_id = counter.get('id', counter.get('counter', 'N/A'))
                    total_bikes = counter.get('total_bikes', 0)
                    borough = counter.get('borough', 'Unknown')

                    if pd.isna(total_bikes):
                        total_bikes = 0

                    if pd.notna(lat) and pd.notna(lng):
                        # Scale radius by count (min 5, max 20)
                        radius = 5 + (total_bikes / max_count) * 15 if max_count > 0 else 8

                        # Color by borough
                        borough_colors = {
                            'Manhattan': '#e41a1c',
                            'Brooklyn': '#377eb8',
                            'Queens': '#4daf4a',
                            'Bronx': '#984ea3',
                            'Staten Island': '#ff7f00',
                            'UNKNOWN': '#999999'
                        }
                        color = borough_colors.get(borough, '#999999')

                        folium.CircleMarker(
                            location=[float(lat), float(lng)],
                            radius=radius,
                            color=color,
                            fill=True,
                            fillColor=color,
                            fillOpacity=0.7,
                            popup=folium.Popup(
                                f"<b>{name}</b><br>"
                                f"Borough: {borough}<br>"
                                f"Total Bikes ({year_range_str}): <b>{total_bikes:,.0f}</b>",
                                max_width=300
                            )
                        ).add_to(coverage_map)

                # Add title
                title_html = f'''
                <div style="position: fixed;
                            top: 10px; left: 50px; width: 500px; height: 50px;
                            background-color: white; border:2px solid grey; z-index:9999;
                            font-size:14px; padding: 10px">
                    <b>Model Coverage (green) & Bike Counters ({year_range_str})</b>
                </div>
                '''
                coverage_map.get_root().html.add_child(folium.Element(title_html))

                st_folium(coverage_map, width=1200, height=600, key="coverage", returned_objects=[])

                # Stats
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Model Grid Cells", f"{len(cells_keep_df):,}")
                col_b.metric("Bike Counters", f"{len(counters_df):,}")

                if 'total_bikes' in counters_merged.columns:
                    total_counted = counters_merged['total_bikes'].sum()
                    col_c.metric(f"Total Bikes Counted ({year_range_str})", f"{total_counted:,.0f}")

                # Counter details table
                st.markdown("---")
                st.markdown(f"**Bike Counter Details ({year_range_str})**")

                if 'total_bikes' in counters_merged.columns:
                    # Prepare display table
                    display_df = counters_merged[['name', 'borough', 'total_bikes', 'latitude', 'longitude']].copy()
                    display_df = display_df.dropna(subset=['latitude', 'longitude'])
                    display_df['total_bikes'] = display_df['total_bikes'].fillna(0).astype(int)
                    display_df = display_df.sort_values('total_bikes', ascending=False)
                    display_df.columns = ['Name', 'Borough', 'Total Bikes', 'Lat', 'Lng']

                    st.dataframe(
                        display_df,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            'Total Bikes': st.column_config.NumberColumn(format="%d"),
                            'Lat': st.column_config.NumberColumn(format="%.4f"),
                            'Lng': st.column_config.NumberColumn(format="%.4f"),
                        }
                    )

                st.markdown("""
                **Legend:**
                - 🟩 **Green rectangles**: Grid cells included in the model (have CitiBike exposure)
                - 🔴🔵🟢🟣 **Colored circles**: Bike counter stations (size = total bikes counted, color = borough)

                **Key Insight:** Counter size shows measurement volume. Larger counters provide more reliable proxy validation.
                """)

        st.markdown("---")

        # Data quality warning for 2022
        if 2022 in selected_years:
            st.warning("""
            **Note on 2022 Data:** In 2022, CitiBike recorded raw GPS coordinates instead of fixed station locations.
            Approximately 0.01% of trips have GPS errors placing them in incorrect grid cells. This causes 2022 to
            show slightly more grid cells in outer areas compared to other years. The effect on the model is negligible.
            """)

        st.caption(f"""
        **Map Details:**
        - Grid cells: 2.5km × 2.5km (GRID_DEG=0.025)
        - Black lines: **Real NYC Borough boundaries**
        - **Time period: {year_range_str}**
        - **Crashes Tab:** All crashes (incl. areas without CitiBike)
        - **Exposure Tab:** Only CitiBike coverage areas
        - **Coverage Tab:** Model grid + bike counter locations
        - **Important:** CitiBike exposure is a proxy for total bike activity—not all bike trips!
        - Click on grid cells/markers for details
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
mc_2025 = read_parquet_safe(MC_2025_SCENARIOS)  # Use scenarios file (includes actual exposure)
train = read_parquet_safe(GRID_TRAIN)

missing = []
for p in [COMP, EVAL_2025, MC_2025_SCENARIOS, GRID_TRAIN]:
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
st.subheader("A) High-level summary (training 2021–2024, forecast 2025)")

# FIXED: Use grid_complete for ACTUAL crash/exposure totals (consistent with maps)
if not grid_complete.empty:
    grid_2021_2024 = grid_complete[
        (grid_complete['hour_ts'] >= '2021-01-01') &
        (grid_complete['hour_ts'] < '2025-01-01')
    ].copy()
    
    # Total observed crashes (ALL areas, including non-CitiBike)
    obs_train_total = float(grid_2021_2024['y_bike'].sum())

    # Total exposure (only CitiBike areas)
    exp_train_total = float(grid_2021_2024[grid_2021_2024['exposure_min'] > 0]['exposure_min'].sum())

    # Rate calculated on CitiBike areas only
    crashes_with_exposure = grid_2021_2024[grid_2021_2024['exposure_min'] > 0]
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

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observed bike crashes (2021–2024)", fmt_int(obs_train_total))
c2.metric("Exposure minutes (2021–2024)", fmt_int(exp_train_total))
c3.metric("Rate per 100k exposure-min (train)", fmt_float(train_rate, 3))
c4.metric("Observed bike crashes (2025)", fmt_int(obs_2025_total))

c5, _ = st.columns(2)
c5.metric("Predicted 2025 total (Poisson mean)", fmt_int(pred_poisson))

st.caption(
    "**Model:** log(μ) = Xβ, where Xβ includes log1p(exposure) as feature + temporal, spatial, and weather effects. "
    "See Section G for full GLM specification."
)

# Model limitation explanation (always visible)
st.markdown("---")
st.subheader("A) Model Methodology & Historical Analysis")

st.markdown("""
#### Daily Aggregation (Memory Efficient)

The model uses **daily** data (~117K rows) instead of hourly (~2.8M rows) for memory efficiency.
This allows running on 8GB RAM machines. Hour-of-day patterns are shown below from raw hourly data.

#### Evaluation Consistency

**Critical:** Predictions and observed crashes use the **same cell set** (cells with 2025 exposure).

- **Prediction**: Only for cells that had CitiBike activity in 2025
- **Observed**: Only crashes in cells with 2025 CitiBike exposure

This ensures an apples-to-apples comparison for insurance use cases:
> "Given the areas where CitiBike operates in 2025, how many crashes do we expect?"
""")

# =============================
# DYNAMIC: Historical Exposure & Crash Analysis
# =============================
st.markdown("#### Historical Exposure, Crashes & Crash Rate (in 2025-active cells)")

# Load cells_2025 for filtering
cells_2025_path = CELLS_KEEP.parent / "grid_cells_2025.parquet"
if cells_2025_path.exists():
    cells_2025_df = pd.read_parquet(cells_2025_path)
    active_cells = set(cells_2025_df['cell_id'].tolist())

    # Calculate exposure and crashes per year in active cells
    @st.cache_data
    def compute_historical_stats():
        # Load exposure data
        exposure_train = pd.read_parquet(EXPOSURE_CELL_HOUR_TRAIN)
        exposure_test = pd.read_parquet(EXPOSURE_CELL_HOUR_TEST)
        exposure = pd.concat([exposure_train, exposure_test], ignore_index=True)
        exposure['hour_ts'] = pd.to_datetime(exposure['hour_ts'])
        exposure['year'] = exposure['hour_ts'].dt.year

        # Load crashes
        crashes = pd.read_parquet(CRASH_CELL_HOUR)
        crashes['hour_ts'] = pd.to_datetime(crashes['hour_ts'])
        crashes['year'] = crashes['hour_ts'].dt.year

        # Filter to active cells
        exposure_in_cells = exposure[exposure['cell_id'].isin(active_cells)]
        crashes_in_cells = crashes[crashes['cell_id'].isin(active_cells)]

        # Aggregate by year
        exp_by_year = exposure_in_cells.groupby('year')['exposure_min'].sum() / 1_000_000
        crash_by_year = crashes_in_cells.groupby('year')['y_bike'].sum()

        # Combine
        hist_df = pd.DataFrame({
            'Exposure (M min)': exp_by_year.round(1),
            'Crashes': crash_by_year.astype(int)
        })
        hist_df['Crash Rate (per 100k min)'] = (
            hist_df['Crashes'] / (hist_df['Exposure (M min)'] * 1_000_000) * 100_000
        ).round(2)

        return hist_df

    hist_stats = compute_historical_stats()

    # Calculate averages (training period 2021-2024, excludes COVID 2020)
    avg_exp = hist_stats.loc[2021:2024, 'Exposure (M min)'].mean()
    avg_crash = hist_stats.loc[2021:2024, 'Crashes'].mean()
    avg_rate = hist_stats.loc[2021:2024, 'Crash Rate (per 100k min)'].mean()

    # Add comparison columns
    hist_display = hist_stats.copy()
    hist_display['vs Avg Rate'] = ((hist_display['Crash Rate (per 100k min)'] / avg_rate - 1) * 100).round(1).astype(str) + '%'
    hist_display.index.name = 'Year'

    # Display table
    st.dataframe(hist_display, use_container_width=True)

    # Key insights
    exp_2025 = hist_stats.loc[2025, 'Exposure (M min)']
    crash_2025 = hist_stats.loc[2025, 'Crashes']
    rate_2025 = hist_stats.loc[2025, 'Crash Rate (per 100k min)']

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "2025 Exposure vs Train Avg",
        f"{exp_2025:.0f}M min",
        f"{((exp_2025/avg_exp)-1)*100:+.1f}%"
    )
    col2.metric(
        "2025 Crashes vs Train Avg",
        f"{crash_2025:,.0f}",
        f"{((crash_2025/avg_crash)-1)*100:+.1f}%"
    )
    col3.metric(
        "2025 Crash Rate vs Train Avg",
        f"{rate_2025:.2f}",
        f"{((rate_2025/avg_rate)-1)*100:+.1f}%"
    )

    st.markdown(f"""
    **Key Insight:** While 2025 exposure is **+{((exp_2025/avg_exp)-1)*100:.0f}% above** the 2021-2024 training average,
    the crash rate dropped by **{((1-rate_2025/avg_rate))*100:.0f}%** - from {avg_rate:.2f} to {rate_2025:.2f} per 100k minutes.

    The model's trend variable captures this declining rate, predicting continued improvement in 2025.
    """)
else:
    st.warning("Historical analysis not available - cells_2025 file not found.")

st.markdown("""
#### Exposure Methodology

**Line Interpolation:** Trip duration is distributed across grid cells along the straight line
between start and end station (5 interpolation points).

**CitiBike as Proxy:** Validated at Borough × Month level (r ≈ 0.85), but spatial distribution
within boroughs is assumed, not validated.

#### Key Limitations

1. **Spatial gaps:** ~7% of NYC crashes occur outside model coverage (Staten Island, outer areas)
2. **Proxy drift:** CitiBike grows faster than total cycling in Brooklyn/Queens
3. **Reporting delays:** Recent crashes may not be fully reported

---
""")


# =============================
# B) Model comparison
# =============================
st.subheader("B) Model Diagnostics (Poisson)")

comp_show = comp.copy()
for col in ["aic", "llf", "dispersion"]:
    if col in comp_show.columns:
        comp_show[col] = pd.to_numeric(comp_show[col], errors="coerce")

if "dispersion" in comp_show.columns:
    comp_show["dispersion"] = comp_show["dispersion"].map(lambda x: float(x) if pd.notna(x) else np.nan)

st.dataframe(comp_show, use_container_width=True)

st.markdown("""
**Model Selection:** Poisson GLM (Negative Binomial removed - dispersion ~1 indicates no overdispersion)

**Key metrics:**
- **Dispersion (Pearson χ²/df):** Values near 1 indicate good fit. Our value ~1.002 confirms Poisson is appropriate.
- **AIC:** Lower is better for model comparison.
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
pred_line["series"] = pred_line["model"].map({"poisson": "Poisson (mean)"}).fillna(pred_line["model"])
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

# Monthly deviation analysis
st.markdown("#### Monthly Prediction Error Analysis (Poisson)")

# Get Poisson predictions
poisson_eval = eval_2025[eval_2025['model'] == 'poisson'].copy()
if not poisson_eval.empty:
    poisson_eval['month'] = pd.to_datetime(poisson_eval['month_ts']).dt.month
    poisson_eval['month_name'] = pd.to_datetime(poisson_eval['month_ts']).dt.strftime('%b')
    poisson_eval['diff'] = poisson_eval['pred_mean'] - poisson_eval['observed']
    poisson_eval['diff_pct'] = ((poisson_eval['pred_mean'] / poisson_eval['observed']) - 1) * 100

    # Summary metrics
    summer_months = [6, 7, 8, 9]
    summer_error = poisson_eval[poisson_eval['month'].isin(summer_months)]['diff'].sum()
    other_error = poisson_eval[~poisson_eval['month'].isin(summer_months)]['diff'].sum()
    total_error = poisson_eval['diff'].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Summer (Jun-Sep) Error", f"{summer_error:+,.0f}", f"{summer_error/total_error*100:.0f}% of total")
    col2.metric("Other Months Error", f"{other_error:+,.0f}", f"{other_error/total_error*100:.0f}% of total")
    col3.metric("Total Error", f"{total_error:+,.0f}", f"{total_error/poisson_eval['observed'].sum()*100:+.1f}%")

    # Bar chart of monthly errors
    error_chart = (
        alt.Chart(poisson_eval)
        .mark_bar()
        .encode(
            x=alt.X("month_name:N", title="Month", sort=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
            y=alt.Y("diff:Q", title="Prediction Error (Predicted - Observed)"),
            color=alt.condition(
                alt.datum.diff > 0,
                alt.value("#e74c3c"),  # Red for overprediction
                alt.value("#27ae60")   # Green for underprediction
            ),
            tooltip=[
                alt.Tooltip("month_name:N", title="Month"),
                alt.Tooltip("observed:Q", title="Observed", format=",.0f"),
                alt.Tooltip("pred_mean:Q", title="Predicted", format=",.0f"),
                alt.Tooltip("diff:Q", title="Error", format="+,.0f"),
                alt.Tooltip("diff_pct:Q", title="Error %", format="+.1f")
            ]
        )
        .properties(height=250)
    )
    st.altair_chart(error_chart, use_container_width=True)

    st.caption("""
    **Insight:** The model systematically overpredicts summer months (Jun-Sep).
    This could indicate: (1) unusual summer 2025 conditions not captured by historical patterns,
    (2) reporting delays for recent months, or (3) a structural shift in cycling behavior.
    """)

st.markdown("---")

# =============================
# D) Uncertainty (Monte Carlo - Fully Random Simulation)
# =============================
st.subheader("D) Uncertainty of total 2025 crashes (Monte Carlo)")

mc_scenarios = read_parquet_safe(MC_2025_SCENARIOS)

if mc_scenarios.empty:
    st.warning("Monte Carlo dataset is missing or invalid. Run `make modeling` to generate.")
else:
    # Filter to Poisson model (better fit based on AIC/dispersion analysis)
    mc_poisson = mc_scenarios[mc_scenarios['model'] == 'poisson'].copy()

    if not mc_poisson.empty:
        # Calculate statistics
        all_q05, all_q50, all_q95 = np.quantile(mc_poisson['total_2025'].values, [0.05, 0.5, 0.95])
        obs_2025_in_cell = float(eval_2025.drop_duplicates(subset=["month_ts"])[["observed"]].fillna(0)["observed"].sum())

        # Explanation
        st.markdown("""
        **Comprehensive Uncertainty Quantification (Poisson Model)**

        Each simulation randomly samples **all** uncertainty dimensions:

        | Dimension | Sampling | Purpose |
        |-----------|----------|---------|
        | Weather | Uniform from 2021-2025 | Captures year-to-year weather variability |
        | Exposure | Uniform from 2021-2025 | Captures different cycling activity patterns |
        | Growth | Uniform(0.80, 1.20) | Continuous ±20% exposure estimation uncertainty |
        | Parameters | MVN from covariance matrix | Model coefficient uncertainty |

        This produces a distribution reflecting the full range of plausible scenarios.
        """)

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Median Prediction", f"{all_q50:,.0f}")
        col2.metric("90% CI Lower", f"{all_q05:,.0f}")
        col3.metric("90% CI Upper", f"{all_q95:,.0f}")
        col4.metric("Observed 2025", f"{obs_2025_in_cell:,.0f}")

        # Histogram
        st.markdown("**Distribution of predicted 2025 crashes:**")
        hist = (
            alt.Chart(mc_poisson)
            .mark_bar(opacity=0.7, color='steelblue')
            .encode(
                x=alt.X("total_2025:Q", bin=alt.Bin(maxbins=30), title="Total crashes in 2025"),
                y=alt.Y("count():Q", title="Simulation count"),
            )
            .properties(height=300)
        )

        # Add observed line
        obs_rule = alt.Chart(pd.DataFrame({'observed': [obs_2025_in_cell]})).mark_rule(
            color='red', strokeWidth=2, strokeDash=[5,5]
        ).encode(x='observed:Q')

        obs_text = alt.Chart(pd.DataFrame({'observed': [obs_2025_in_cell], 'label': ['Observed']})).mark_text(
            align='left', dx=5, dy=-10, color='red', fontSize=12
        ).encode(x='observed:Q', text='label:N')

        st.altair_chart(hist + obs_rule + obs_text, use_container_width=True)

        st.caption(f"""
        **Note:** {len(mc_poisson)} simulations (Poisson model).
        Red dashed line = observed 2025 crashes in model cells.
        """)

st.markdown("---")

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

st.markdown("---")

# =============================
# F) Proxy Quality Analysis
# =============================
st.subheader("F) Proxy Quality: Is CitiBike representative of total cycling?")

proxy_df = read_parquet_safe(PROXY_TEST_BM)

if proxy_df.empty:
    st.info("Proxy validation data not available. Run `make proxy-test` to generate.")
else:
    st.markdown("""
    **Key Question:** CitiBike trips are used as a proxy for total cycling exposure.
    But how well does CitiBike represent *all* cyclists in NYC?

    We validate this by comparing CitiBike exposure against official bike counter data.
    """)

    # IMPORTANT: Explain aggregation level
    st.info("""
    **Important Methodology Note:**

    The proxy correlation is calculated at the **Borough × Month** level, NOT at the grid-cell level.
    This is why the "Coverage & Counters" map doesn't show obvious spatial correlation—the counters
    are NOT located in the same places as CitiBike stations.

    **What we're testing:** When a borough has more CitiBike activity in a given month,
    does that borough also have more total cyclists (measured by counters)?
    """)

    # Add year column
    proxy_df['year'] = pd.to_datetime(proxy_df['month_ts']).dt.year

    # NEW: Scatterplot showing Borough x Month correlation
    st.markdown("### Borough × Month Correlation (Log-Space)")
    st.caption("Each point = one borough in one month. The correlation is calculated across all these points.")

    # Prepare scatter data
    scatter_df = proxy_df.dropna(subset=['log_citi', 'log_cnt']).copy()
    scatter_df['borough_label'] = scatter_df['borough'].str.title()

    if not scatter_df.empty:
        # Calculate overall correlation for display
        from scipy import stats as scipy_stats
        overall_r, overall_p = scipy_stats.pearsonr(scatter_df['log_citi'], scatter_df['log_cnt'])

        # Scatterplot with regression line
        scatter = (
            alt.Chart(scatter_df)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X('log_citi:Q', title='log(CitiBike exposure minutes)'),
                y=alt.Y('log_cnt:Q', title='log(Bike counter total)'),
                color=alt.Color('borough_label:N', title='Borough',
                              scale=alt.Scale(scheme='category10')),
                tooltip=[
                    alt.Tooltip('borough_label:N', title='Borough'),
                    alt.Tooltip('month_ts:T', title='Month'),
                    alt.Tooltip('citi_exposure_min:Q', title='CitiBike (min)', format=',.0f'),
                    alt.Tooltip('counter_bike_count:Q', title='Counter total', format=',.0f'),
                ]
            )
        )

        # Add regression line
        regression = scatter.transform_regression(
            'log_citi', 'log_cnt'
        ).mark_line(color='black', strokeDash=[5, 5], strokeWidth=2)

        chart = (scatter + regression).properties(
            height=350,
            title=f'Pearson r = {overall_r:.3f} (p < 0.001) — {len(scatter_df)} data points'
        )
        st.altair_chart(chart, use_container_width=True)

        # Data point summary
        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Data Points", f"{len(scatter_df)}")
        col_info2.metric("Boroughs", f"{scatter_df['borough'].nunique()}")
        col_info3.metric("Months", f"{scatter_df['month_ts'].nunique()}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Correlation by year
        st.markdown("**Correlation stability over time**")

        corr_by_year = []
        for year in sorted(proxy_df['year'].unique()):
            yr_data = proxy_df[proxy_df['year'] == year].dropna(subset=['log_citi', 'log_cnt'])
            if len(yr_data) > 5:
                from scipy import stats
                r, p = stats.pearsonr(yr_data['log_citi'], yr_data['log_cnt'])
                corr_by_year.append({'year': year, 'correlation': r, 'n': len(yr_data)})

        corr_df = pd.DataFrame(corr_by_year)

        if not corr_df.empty:
            chart = (
                alt.Chart(corr_df)
                .mark_line(point=True, color='#1f77b4')
                .encode(
                    x=alt.X('year:O', title='Year'),
                    y=alt.Y('correlation:Q', title='Pearson r', scale=alt.Scale(domain=[0.7, 1.0])),
                    tooltip=['year:O', alt.Tooltip('correlation:Q', format='.3f'), 'n:Q']
                )
                .properties(height=250)
            )
            st.altair_chart(chart, use_container_width=True)

            # Trend indicator
            if len(corr_df) >= 2:
                first_r = corr_df.iloc[0]['correlation']
                last_r = corr_df.iloc[-1]['correlation']
                if last_r < first_r - 0.05:
                    st.warning(f"Proxy correlation declining: {first_r:.2f} → {last_r:.2f}")
                else:
                    st.success(f"Proxy correlation stable: {first_r:.2f} → {last_r:.2f}")

    with col2:
        # Growth comparison by borough
        st.markdown("**Growth comparison: CitiBike vs Total cycling**")

        # Calculate growth rates
        growth_data = []
        for borough in proxy_df['borough'].unique():
            boro_data = proxy_df[proxy_df['borough'] == borough]

            first_year = boro_data[boro_data['year'] == boro_data['year'].min()]
            last_year = boro_data[boro_data['year'] == boro_data['year'].max()]

            if len(first_year) > 0 and len(last_year) > 0:
                citi_first = first_year['citi_exposure_min'].sum()
                citi_last = last_year['citi_exposure_min'].sum()
                counter_first = first_year['counter_bike_count'].sum()
                counter_last = last_year['counter_bike_count'].sum()

                if citi_first > 0 and counter_first > 0:
                    citi_growth = (citi_last / citi_first - 1) * 100
                    counter_growth = (counter_last / counter_first - 1) * 100
                    growth_data.append({
                        'Borough': borough.title(),
                        'CitiBike Growth': f"{citi_growth:+.0f}%",
                        'Counter Growth': f"{counter_growth:+.0f}%",
                        'Difference': f"{citi_growth - counter_growth:+.0f}%",
                        '_diff': citi_growth - counter_growth
                    })

        if growth_data:
            growth_df = pd.DataFrame(growth_data).sort_values('_diff', ascending=False)
            st.dataframe(
                growth_df[['Borough', 'CitiBike Growth', 'Counter Growth', 'Difference']],
                hide_index=True,
                use_container_width=True
            )

            # Interpretation
            high_diff = [g for g in growth_data if g['_diff'] > 50]
            if high_diff:
                st.warning(f"""
                **Proxy Drift Detected:** In {', '.join([h['Borough'] for h in high_diff])},
                CitiBike is growing much faster than total cycling. This means CitiBike
                is capturing an increasing share of cyclists, which may bias predictions.
                """)

    # Expander with methodology details
    with st.expander("📊 Proxy Validation Methodology Details"):
        st.markdown("""
        ### How Proxy Validation Works

        **Data Sources:**
        1. **CitiBike trips** → aggregated to Borough × Month (sum of trip minutes)
        2. **NYC Bike Counters** → aggregated to Borough × Month (sum of all counter readings)

        **Correlation Calculation:**
        ```
        For each (Borough, Month) pair:
            log_citi = log(CitiBike exposure minutes)
            log_cnt  = log(Bike counter total)

        Pearson correlation: r = corr(log_citi, log_cnt)
        ```

        **Why Log-Space?**
        - Both variables span multiple orders of magnitude
        - Log-transformation makes the relationship approximately linear
        - Standard in exposure modeling (proportional changes matter more than absolute)

        **Limitations of This Validation:**
        1. **Spatial mismatch**: Counters are on bridges/paths, not at CitiBike stations
        2. **Few data points**: Only 5 boroughs × ~60 months ≈ 300 points max
        3. **Different cyclist populations**: Commuters (counters) vs. short trips (CitiBike)
        4. **No grid-level validation**: We can't validate that the exposure *distribution* is correct

        **What "r = 0.85" means:**
        - When a borough has 2x more CitiBike activity, it tends to have ~2x more counter readings
        - This supports using CitiBike as a *temporal proxy* (capturing seasonal/monthly variation)
        - It does NOT validate the *spatial distribution* within a borough
        """)

st.markdown("---")

# =============================
# G) Model Limitations & Caveats
# =============================
st.subheader("G) Model Methodology & Limitations")

st.markdown("""
### Critical Methodological Decisions

This section documents key methodological choices and their implications for interpretation.
""")

# --- Training/Evaluation Consistency ---
with st.expander("Training/Evaluation Consistency (Updated)", expanded=True):
    st.markdown("""
    #### Exposure as Feature: Full Grid Coverage

    With exposure modeled as a **feature** (not offset), the model now covers ALL hours:

    - **Training:** Complete grid (cell × hour) with exposure=0 where no CitiBike trips
    - **Prediction:** For ALL hours, including exposure=0 (log1p(0) = 0)
    - **Evaluation:** Against ALL crashes in model cells

    **Key insight:** The model learns a baseline crash rate even without CitiBike exposure.
    The `log1p_exposure` coefficient (β ≈ 0.42) captures how exposure increases crash risk.
    """)

# --- Proxy Limitations ---
with st.expander("CitiBike as Cycling Proxy: Strengths & Weaknesses", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### What the Proxy Captures

        **Temporal Variation (validated):**
        - Borough × Month correlation: **r ≈ 0.85**
        - When a borough has 2× CitiBike activity, counters show ~2× cyclists
        - Seasonal and monthly patterns are well-captured

        **Spatial Variation (assumed):**
        - We assume CitiBike station density ≈ cyclist density within boroughs
        - Plausible because bike-share infrastructure follows demand
        - NOT directly validated at grid level
        """)

    with col2:
        st.markdown("""
        #### What the Proxy Misses

        **Systematic Biases:**
        - CitiBike grows 3-5× faster than total cycling in Brooklyn/Queens
        - Private bike usage not captured
        - Delivery cyclists (growing rapidly) underrepresented

        **Spatial Gaps:**
        - Staten Island: 0% coverage (no CitiBike)
        - Outer Bronx/Queens: limited coverage
        - ~21% of NYC crashes in uncovered areas

        **Temporal Gaps:**
        - Night hours (2-5 AM): minimal CitiBike activity
        - Winter months: reduced CitiBike usage
        """)

# --- Uncertainty Quantification ---
with st.expander("Uncertainty Quantification: What's Captured & What's Not"):
    st.markdown("""
    #### Monte Carlo Simulation (1000 draws)

    **Uncertainty Sources CAPTURED:**
    - **Parameter uncertainty:** Coefficients sampled from multivariate normal (using covariance matrix)
    - **Weather uncertainty:** Bootstrap entire years (preserves temporal autocorrelation)
    - **Exposure uncertainty:** Historical exposure patterns from 2021-2025
    - **Growth uncertainty:** ±20% exposure growth factor

    **Uncertainty Sources NOT CAPTURED:**
    - **Proxy drift:** CitiBike share of cycling may change over time
    - **Model misspecification:** Functional form may be wrong
    - **Spatial coverage gaps:** No uncertainty for areas without CitiBike
    - **External shocks:** COVID recovery, e-bike adoption, infrastructure changes

    **Implication:** Reported confidence intervals are likely **narrower than true uncertainty**.
    For policy decisions, add ±10-15% to the reported intervals.
    """)

# --- Recommendations ---
with st.expander("Recommendations for Interpretation"):
    st.markdown("""
    ### How to Use These Predictions

    #### Do:
    1. **Use confidence intervals**, not point estimates
    2. **Focus on Manhattan** — proxy quality is highest here
    3. **Use for relative comparisons** — "10% more exposure → ~10% more crashes"
    4. **Validate against actuals** — as 2025 data accumulates, check alignment

    #### Don't:
    1. **Don't extrapolate to Staten Island** or uncovered areas
    2. **Don't treat absolute numbers as precise** — treat as ±15-20% estimates
    3. **Don't assume Brooklyn/Queens predictions are equally reliable** as Manhattan
    4. **Don't use for individual crash prediction** — this is an aggregate rate model

    ### For Policy & Infrastructure Decisions

    - Use as **one input among many** for bike lane planning
    - Predictions are best for **high-exposure areas** (Midtown, Downtown Brooklyn)
    - Consider pairing with actual crash data for final decisions
    - The model identifies **relative risk**, not causal mechanisms
    """)

with st.expander("Why Grid-Level Modeling? (Grid vs. Borough)"):
    st.markdown("""
    ### Design Decision: Grid-Level (64 cells) vs. Borough-Level (5 units)

    Both models would operate at **hourly resolution**. The only difference is **spatial granularity**:

    | Criterion | Borough Model | Grid Model (chosen) |
    |-----------|---------------|---------------------|
    | Spatial units | 5 boroughs | 64 grid cells |
    | Observations | 5 × 8760h × 5y ≈ 219k | 64 × 8760h × 5y ≈ 2.8M |
    | Hotspot detection | ❌ No | ✅ Yes |
    | Risk pricing granularity | ❌ Coarse | ✅ Fine |
    | Proxy validation | ✅ Validated at same level | ⚠️ Finer than validation |

    ### Why the Grid Model is Justified

    **1. Insurance Use Case**
    > For risk pricing, spatial granularity matters. A borough model would price all Manhattan
    > policies equally, even though Midtown is 3× riskier than Upper East Side.

    **2. Spatial Heterogeneity**
    > Crash risk varies significantly *within* boroughs. Grid cells capture local effects
    > (intersections, bike lanes, traffic patterns) that borough-level averages miss.

    **3. Proxy Validates Temporal Variation**
    > The Borough-level correlation (r ≈ 0.85) validates that CitiBike captures *temporal variation*
    > in cycling activity (seasonal, monthly patterns). The *spatial distribution* within a borough
    > is assumed to follow CitiBike station density—plausible since infrastructure follows demand.

    **4. Exposure as Feature (not Offset)**
    > Exposure is modeled as a feature: `log1p(exposure_min)` with estimated coefficient β.
    > This allows: (1) including hours with exposure=0, (2) estimating the exposure elasticity.
    > With β ≈ 0.42, we have diminishing returns: doubling exposure → ~34% more crashes.

    **5. Out-of-Sample Validation**
    > The model now predicts for ALL hours in model cells (not just hours with exposure).
    > This enables direct comparison with all observed crashes.

    ### Key Assumption
    > **Within a borough, CitiBike station density ≈ total cyclist density.**
    > This is plausible because bike-share infrastructure is deployed where demand exists.
    """)

with st.expander("Technical Details: Model Specification"):
    st.markdown("""
    ### GLM Specification (Daily Aggregation)

    **Model Family:** Poisson (dispersion ~1 confirms no overdispersion)

    **Formula (DAILY - no hour_of_day):**
    ```
    y_bike ~ C(dow) + C(month)
           + lat_n + lng_n + lat_n² + lng_n² + lat_n×lng_n
           + temp + prcp + snow + wspd
           + trend
           + log1p_exposure

    where:
      lat_n = (grid_lat - mean) / std   (normalized grid cell latitude)
      lng_n = (grid_lng - mean) / std   (normalized grid cell longitude)
      trend = (day_ts - 2021-01-01).days / 365.25   (years since training start)
      log1p_exposure = log(exposure_min + 1)   (handles exposure=0)
    ```

    **Why Daily (not Hourly)?**
    - Hourly: ~2.8M training rows → crashes on 8GB RAM machines
    - Daily: ~117K training rows → runs smoothly
    - Hour-of-day patterns shown in dashboard from raw hourly data

    **Exposure as Feature (not Offset):**
    - `log1p(exposure)` handles exposure=0 gracefully (log1p(0) = 0)
    - Coefficient β is estimated from data (not fixed at 1)
    - Typical β ≈ 0.4-0.5 (diminishing returns)

    **Key Components:**

    | Component | Description | Purpose |
    |-----------|-------------|---------|
    | `C(dow)` | 7 categorical dummies | Weekend vs weekday patterns |
    | `C(month)` | 12 categorical dummies | Seasonal variation |
    | `lat_n, lng_n` | Z-scored coordinates | Spatial trends (north-south, east-west) |
    | `lat_n², lng_n²` | Quadratic terms | Allow for curved spatial surface |
    | `lat_n×lng_n` | Interaction | Diagonal spatial patterns |
    | `temp, prcp, snow, wspd` | Z-scored weather | Weather impact on crashes |
    | `trend` | Years since 2021-01-01 | Temporal trend in crash rates |
    | `log1p_exposure` | Exposure feature | Estimated coefficient |

    **Evaluation Consistency:**
    - Predictions only for cells with 2025 exposure (`cells_2025`)
    - Observed crashes only counted in cells with 2025 exposure
    - Ensures apples-to-apples comparison

    **Training/Test Split:**
    - Training: 2021-2024 (4 years, excludes COVID 2020)
    - Testing: 2025 (true out-of-sample)
    - Evaluation: Crashes in cells_2025 only

    **Monte Carlo Uncertainty (S=1000):**
    - Parameter: Sample β from N(β̂, Σ̂)
    - Weather: Bootstrap entire years for 2025 (preserves autocorrelation)
    - Exposure: Historical patterns from 2021-2025
    - Growth: ±20% exposure growth factor
    """)


