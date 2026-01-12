import streamlit as st
import pandas as pd
import altair as alt

from src.dashboard.lib.db import get_con
from src.dashboard.lib.settings import TRIPS_BOROUGH_HOUR, WEATHER_GLOB

st.title("Citi Bike Trips (usage & seasonality)")

con = get_con()

# ============================================================================
# HELPERS
# ============================================================================
def warn_if_missing(path, label):
    try:
        con.execute(f"SELECT 1 FROM read_parquet('{path.as_posix()}') LIMIT 1")
        return True
    except Exception as e:
        st.error(f"Missing/invalid {label}: {path}\n\n{e}")
        return False

if not warn_if_missing(TRIPS_BOROUGH_HOUR, "trips borough-hour mart"):
    st.stop()

# ============================================================================
# SIDEBAR FILTERS
# ============================================================================
with st.sidebar:
    st.header("Filters")
    start = st.date_input("Start date", value=pd.to_datetime("2024-01-01").date())
    end = st.date_input("End date (exclusive)", value=pd.to_datetime("2026-01-01").date())

    rideable = st.selectbox("Bike type", ["ALL", "classic_bike", "electric_bike"])
    rider = st.selectbox("Customer", ["ALL", "member", "casual"])

    boroughs = con.execute(
        f"SELECT DISTINCT borough FROM read_parquet('{TRIPS_BOROUGH_HOUR.as_posix()}') ORDER BY borough"
    ).fetch_df()["borough"].dropna().tolist()
    
    if len(boroughs) == 0:
        st.warning("No borough values found in trips mart.")
    
    borough_sel = st.multiselect("Borough", boroughs, default=boroughs[:])

    gran = st.selectbox("Timeseries", ["day", "week", "month"])
    
    # Metric display selection
    st.subheader("Display Options")
    metric_display = st.multiselect(
        "Show metrics", 
        ["Trips", "Exposure minutes"],
        default=["Trips", "Exposure minutes"],
        help="Select which metrics to display in all charts"
    )

# Build WHERE clause
where = ["1=1"]
where.append(f"hour_ts >= TIMESTAMP '{start.isoformat()} 00:00:00'")
where.append(f"hour_ts <  TIMESTAMP '{end.isoformat()} 00:00:00'")
if rideable != "ALL":
    where.append(f"rideable_type = '{rideable}'")
if rider != "ALL":
    where.append(f"member_casual = '{rider}'")
if borough_sel:
    b_list = ",".join([f"'{b}'" for b in borough_sel])
    where.append(f"borough IN ({b_list})")
where_sql = " AND ".join(where)

trunc = {"day": "day", "week": "week", "month": "month"}[gran]

# ============================================================================
# KPIs
# ============================================================================
kpi = con.execute(f"""
SELECT
  COALESCE(SUM(n_trips),0) AS n_trips,
  COALESCE(SUM(exposure_min),0) AS exposure_min,
  CASE WHEN COALESCE(SUM(n_trips),0)>0 
    THEN COALESCE(SUM(exposure_min),0)/SUM(n_trips) 
    ELSE 0 
  END AS avg_min_per_trip
FROM read_parquet('{TRIPS_BOROUGH_HOUR.as_posix()}')
WHERE {where_sql};
""").fetch_df()

n_trips = float(kpi.loc[0, "n_trips"])
exposure_min = float(kpi.loc[0, "exposure_min"])
avg_min = float(kpi.loc[0, "avg_min_per_trip"])

c1, c2, c3 = st.columns(3)
c1.metric("Trips", f"{int(n_trips):,}")
c2.metric("Exposure (minutes)", f"{exposure_min:,.0f}")
c3.metric("Avg minutes per trip", f"{avg_min:.2f}")

if n_trips == 0:
    st.warning("No trips in the selected filter window. Expand the date range or relax filters.")
    st.stop()

if len(metric_display) == 0:
    st.warning("Please select at least one metric to display in the sidebar.")
    st.stop()

show_trips = "Trips" in metric_display
show_exposure = "Exposure minutes" in metric_display

# ============================================================================
# 1) TIMESERIES: trips + exposure
# ============================================================================
st.subheader("1) Usage over time")

ts = con.execute(f"""
SELECT
  date_trunc('{trunc}', hour_ts) AS t,
  COALESCE(SUM(n_trips),0) AS n_trips,
  COALESCE(SUM(exposure_min),0) AS exposure_min
FROM read_parquet('{TRIPS_BOROUGH_HOUR.as_posix()}')
WHERE {where_sql}
GROUP BY 1
ORDER BY 1;
""").fetch_df()

if show_trips and show_exposure:
    # Dual Y-axis chart
    ts_trips = ts[['t', 'n_trips']].copy()
    ts_exposure = ts[['t', 'exposure_min']].copy()
    
    # Left Y-axis: Trips (blue)
    trips_chart = alt.Chart(ts_trips).mark_line(color='#1f77b4', point=False).encode(
        x=alt.X("t:T", title="Date"),
        y=alt.Y("n_trips:Q", title="Number of Trips"),
        tooltip=[
            alt.Tooltip("t:T", title="Date"),
            alt.Tooltip("n_trips:Q", format=",", title="Trips")
        ]
    )
    
    # Right Y-axis: Exposure minutes (orange)
    exposure_chart = alt.Chart(ts_exposure).mark_line(color='#ff7f0e', point=False).encode(
        x=alt.X("t:T", title="Date"),
        y=alt.Y("exposure_min:Q", title="Exposure Minutes"),
        tooltip=[
            alt.Tooltip("t:T", title="Date"),
            alt.Tooltip("exposure_min:Q", format=",", title="Exposure minutes")
        ]
    )
    
    # Combine with independent Y-scales
    chart_ts = alt.layer(trips_chart, exposure_chart).resolve_scale(
        y='independent'
    ).properties(height=320)
    
    st.altair_chart(chart_ts, use_container_width=True)
    st.caption("Note: **Blue (left axis):** Trips | **Orange (right axis):** Exposure minutes")

elif show_trips:
    # Single Y-axis: Trips only
    chart_ts = (
        alt.Chart(ts)
        .mark_line(color='#1f77b4', point=False)
        .encode(
            x=alt.X("t:T", title="Date"),
            y=alt.Y("n_trips:Q", title="Number of Trips"),
            tooltip=[
                alt.Tooltip("t:T", title="Date"),
                alt.Tooltip("n_trips:Q", format=",", title="Trips")
            ]
        )
        .properties(height=320)
    )
    st.altair_chart(chart_ts, use_container_width=True)

elif show_exposure:
    # Single Y-axis: Exposure only
    chart_ts = (
        alt.Chart(ts)
        .mark_line(color='#ff7f0e', point=False)
        .encode(
            x=alt.X("t:T", title="Date"),
            y=alt.Y("exposure_min:Q", title="Exposure Minutes"),
            tooltip=[
                alt.Tooltip("t:T", title="Date"),
                alt.Tooltip("exposure_min:Q", format=",", title="Exposure minutes")
            ]
        )
        .properties(height=320)
    )
    st.altair_chart(chart_ts, use_container_width=True)


# ============================================================================
# 2) SEASONALITY: hour of day
# ============================================================================
st.subheader("2) Time-of-day pattern (hourly)")

hod = con.execute(f"""
SELECT
  EXTRACT(HOUR FROM hour_ts) AS hour,
  COALESCE(SUM(n_trips),0) AS n_trips,
  COALESCE(SUM(exposure_min),0) AS exposure_min
FROM read_parquet('{TRIPS_BOROUGH_HOUR.as_posix()}')
WHERE {where_sql}
GROUP BY 1
ORDER BY 1;
""").fetch_df()

if show_trips and show_exposure:
    # Dual Y-axis chart
    hod_trips = hod[['hour', 'n_trips']].copy()
    hod_exposure = hod[['hour', 'exposure_min']].copy()
    
    # Left Y-axis: Trips (blue)
    trips_chart_hod = alt.Chart(hod_trips).mark_line(color='#1f77b4', point=True).encode(
        x=alt.X("hour:O", title="Hour of day"),
        y=alt.Y("n_trips:Q", title="Number of Trips"),
        tooltip=[
            alt.Tooltip("hour:O", title="Hour"),
            alt.Tooltip("n_trips:Q", format=",", title="Trips")
        ]
    )
    
    # Right Y-axis: Exposure minutes (orange)
    exposure_chart_hod = alt.Chart(hod_exposure).mark_line(color='#ff7f0e', point=True).encode(
        x=alt.X("hour:O", title="Hour of day"),
        y=alt.Y("exposure_min:Q", title="Exposure Minutes"),
        tooltip=[
            alt.Tooltip("hour:O", title="Hour"),
            alt.Tooltip("exposure_min:Q", format=",", title="Exposure minutes")
        ]
    )
    
    # Combine with independent Y-scales
    chart_hod = alt.layer(trips_chart_hod, exposure_chart_hod).resolve_scale(
        y='independent'
    ).properties(height=320)
    
    st.altair_chart(chart_hod, use_container_width=True)
    st.caption("Note: **Blue (left axis):** Trips | **Orange (right axis):** Exposure minutes")

elif show_trips:
    # Single Y-axis: Trips only
    chart_hod = (
        alt.Chart(hod)
        .mark_line(color='#1f77b4', point=True)
        .encode(
            x=alt.X("hour:O", title="Hour of day"),
            y=alt.Y("n_trips:Q", title="Number of Trips"),
            tooltip=[
                alt.Tooltip("hour:O", title="Hour"),
                alt.Tooltip("n_trips:Q", format=",", title="Trips")
            ]
        )
        .properties(height=320)
    )
    st.altair_chart(chart_hod, use_container_width=True)

elif show_exposure:
    # Single Y-axis: Exposure only
    chart_hod = (
        alt.Chart(hod)
        .mark_line(color='#ff7f0e', point=True)
        .encode(
            x=alt.X("hour:O", title="Hour of day"),
            y=alt.Y("exposure_min:Q", title="Exposure Minutes"),
            tooltip=[
                alt.Tooltip("hour:O", title="Hour"),
                alt.Tooltip("exposure_min:Q", format=",", title="Exposure minutes")
            ]
        )
        .properties(height=320)
    )
    st.altair_chart(chart_hod, use_container_width=True)


# ============================================================================
# 3) DAY-OF-WEEK PATTERN + HOURLY PATTERN BY WEEKDAY
# ============================================================================
st.subheader("3) Day-of-week pattern & Hourly usage by weekday")

# Day-of-week data
dow = con.execute(f"""
SELECT
  EXTRACT(DOW FROM hour_ts) AS dow,
  COALESCE(SUM(n_trips),0) AS n_trips,
  COALESCE(SUM(exposure_min),0) AS exposure_min
FROM read_parquet('{TRIPS_BOROUGH_HOUR.as_posix()}')
WHERE {where_sql}
GROUP BY 1
ORDER BY 1;
""").fetch_df()

# Hourly pattern by weekday data
hod_dow = con.execute(f"""
SELECT
  EXTRACT(HOUR FROM hour_ts) AS hour,
  EXTRACT(DOW FROM hour_ts) AS dow,
  COALESCE(SUM(n_trips),0) AS n_trips,
  COALESCE(SUM(exposure_min),0) AS exposure_min
FROM read_parquet('{TRIPS_BOROUGH_HOUR.as_posix()}')
WHERE {where_sql}
GROUP BY 1, 2
ORDER BY 1, 2;
""").fetch_df()

# Create day of week labels
dow_labels = {
    0: 'Sunday',
    1: 'Monday', 
    2: 'Tuesday',
    3: 'Wednesday',
    4: 'Thursday',
    5: 'Friday',
    6: 'Saturday'
}
hod_dow['day_name'] = hod_dow['dow'].map(dow_labels)

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("**Day-of-week pattern**")
    
    if show_trips and show_exposure:
        # Dual Y-axis chart
        dow_trips = dow[['dow', 'n_trips']].copy()
        dow_exposure = dow[['dow', 'exposure_min']].copy()
        
        # Left Y-axis: Trips (blue)
        trips_chart_dow = alt.Chart(dow_trips).mark_line(color='#1f77b4', point=True).encode(
            x=alt.X("dow:O", title="Day of week (0=Sun … 6=Sat)"),
            y=alt.Y("n_trips:Q", title="Number of Trips", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("dow:O", title="Day"),
                alt.Tooltip("n_trips:Q", format=",", title="Trips")
            ]
        )
        
        # Right Y-axis: Exposure minutes (orange)
        exposure_chart_dow = alt.Chart(dow_exposure).mark_line(color='#ff7f0e', point=True).encode(
            x=alt.X("dow:O", title="Day of week (0=Sun … 6=Sat)"),
            y=alt.Y("exposure_min:Q", title="Exposure Minutes", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("dow:O", title="Day"),
                alt.Tooltip("exposure_min:Q", format=",", title="Exposure minutes")
            ]
        )
        
        # Combine with independent Y-scales
        chart_dow = alt.layer(trips_chart_dow, exposure_chart_dow).resolve_scale(
            y='independent'
        ).properties(height=280)
        
        st.altair_chart(chart_dow, use_container_width=True)
        st.caption("Note: **Blue (left):** Trips | **Orange (right):** Exposure")

    elif show_trips:
        # Single Y-axis: Trips only
        chart_dow = (
            alt.Chart(dow)
            .mark_line(point=True, color='#1f77b4')
            .encode(
                x=alt.X("dow:O", title="Day of week (0=Sun … 6=Sat)"),
                y=alt.Y("n_trips:Q", title="Trips", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("dow:O", title="Day"),
                    alt.Tooltip("n_trips:Q", format=",", title="Trips")
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(chart_dow, use_container_width=True)

    elif show_exposure:
        # Single Y-axis: Exposure only
        chart_dow = (
            alt.Chart(dow)
            .mark_line(point=True, color='#ff7f0e')
            .encode(
                x=alt.X("dow:O", title="Day of week (0=Sun … 6=Sat)"),
                y=alt.Y("exposure_min:Q", title="Exposure Minutes", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("dow:O", title="Day"),
                    alt.Tooltip("exposure_min:Q", format=",", title="Exposure minutes")
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(chart_dow, use_container_width=True)

with col_right:
    st.markdown("**Hourly usage by weekday**")
    
    if show_trips and show_exposure:
        # Dual Y-axis chart
        hod_dow_trips = hod_dow[['hour', 'day_name', 'n_trips']].copy()
        hod_dow_exposure = hod_dow[['hour', 'day_name', 'exposure_min']].copy()
        
        # Left Y-axis: Trips
        trips_chart_hod_dow = alt.Chart(hod_dow_trips).mark_line(point=False).encode(
            x=alt.X("hour:O", title="Hour of day"),
            y=alt.Y("n_trips:Q", title="Trips"),
            color=alt.Color("day_name:N", title="Weekday", sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
            tooltip=[
                alt.Tooltip("hour:O", title="Hour"),
                alt.Tooltip("day_name:N", title="Day"),
                alt.Tooltip("n_trips:Q", format=",", title="Trips")
            ]
        )
        
        # Right Y-axis: Exposure
        exposure_chart_hod_dow = alt.Chart(hod_dow_exposure).mark_line(point=False, strokeDash=[5, 5]).encode(
            x=alt.X("hour:O", title="Hour of day"),
            y=alt.Y("exposure_min:Q", title="Exposure"),
            color=alt.Color("day_name:N", title="Weekday", sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
            tooltip=[
                alt.Tooltip("hour:O", title="Hour"),
                alt.Tooltip("day_name:N", title="Day"),
                alt.Tooltip("exposure_min:Q", format=",", title="Exposure minutes")
            ]
        )
        
        chart_hod_dow = alt.layer(trips_chart_hod_dow, exposure_chart_hod_dow).resolve_scale(
            y='independent'
        ).properties(height=280)
        
        st.altair_chart(chart_hod_dow, use_container_width=True)
        st.caption("Note: **Solid:** Trips (left) | **Dashed:** Exposure (right)")
    
    elif show_trips:
        chart_hod_dow = (
            alt.Chart(hod_dow)
            .mark_line(point=False)
            .encode(
                x=alt.X("hour:O", title="Hour of day"),
                y=alt.Y("n_trips:Q", title="Trips"),
                color=alt.Color("day_name:N", title="Weekday", sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
                tooltip=[
                    alt.Tooltip("hour:O", title="Hour"),
                    alt.Tooltip("day_name:N", title="Day"),
                    alt.Tooltip("n_trips:Q", format=",", title="Trips")
                ]
            )
            .properties(height=280)
        )
        st.altair_chart(chart_hod_dow, use_container_width=True)
    
    elif show_exposure:
        chart_hod_dow = (
            alt.Chart(hod_dow)
            .mark_line(point=False)
            .encode(
                x=alt.X("hour:O", title="Hour of day"),
                y=alt.Y("exposure_min:Q", title="Exposure Minutes"),
                color=alt.Color("day_name:N", title="Weekday", sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
                tooltip=[
                    alt.Tooltip("hour:O", title="Hour"),
                    alt.Tooltip("day_name:N", title="Day"),
                    alt.Tooltip("exposure_min:Q", format=",", title="Exposure minutes")
                ]
            )
            .properties(height=280)
        )
        st.altair_chart(chart_hod_dow, use_container_width=True)


# ============================================================================
# 4) BIKE TYPE OVER TIME + BOROUGH DISTRIBUTION
# ============================================================================
st.subheader("4) Bike type usage over time & Borough distribution")

# Bike type over time data
mix_time = con.execute(f"""
SELECT
  date_trunc('{trunc}', hour_ts) AS t,
  rideable_type,
  COALESCE(SUM(n_trips),0) AS n_trips,
  COALESCE(SUM(exposure_min),0) AS exposure_min
FROM read_parquet('{TRIPS_BOROUGH_HOUR.as_posix()}')
WHERE {where_sql}
GROUP BY 1, 2
ORDER BY 1, 2;
""").fetch_df()

# Borough distribution data
borough_dist = con.execute(f"""
SELECT
  borough,
  COALESCE(SUM(n_trips),0) AS n_trips,
  COALESCE(SUM(exposure_min),0) AS exposure_min
FROM read_parquet('{TRIPS_BOROUGH_HOUR.as_posix()}')
WHERE {where_sql}
GROUP BY 1
ORDER BY 2 DESC;
""").fetch_df()

# Calculate percentages for borough distribution
if not borough_dist.empty:
    borough_dist['trips_pct'] = 100 * borough_dist['n_trips'] / borough_dist['n_trips'].sum()
    borough_dist['exposure_pct'] = 100 * borough_dist['exposure_min'] / borough_dist['exposure_min'].sum()

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("**Bike type usage over time**")
    
    if show_trips and show_exposure:
        # Dual Y-axis chart
        mix_trips = mix_time[['t', 'rideable_type', 'n_trips']].copy()
        mix_exposure = mix_time[['t', 'rideable_type', 'exposure_min']].copy()
        
        # Left Y-axis: Trips
        trips_chart_mix = alt.Chart(mix_trips).mark_line(point=False).encode(
            x=alt.X("t:T", title="Date"),
            y=alt.Y("n_trips:Q", title="Trips"),
            color=alt.Color("rideable_type:N", title="Bike type"),
            tooltip=[
                alt.Tooltip("t:T", title="Date"),
                alt.Tooltip("rideable_type:N", title="Bike type"),
                alt.Tooltip("n_trips:Q", format=",", title="Trips")
            ]
        )
        
        # Right Y-axis: Exposure (dashed)
        exposure_chart_mix = alt.Chart(mix_exposure).mark_line(point=False, strokeDash=[5, 5]).encode(
            x=alt.X("t:T", title="Date"),
            y=alt.Y("exposure_min:Q", title="Exposure"),
            color=alt.Color("rideable_type:N", title="Bike type"),
            tooltip=[
                alt.Tooltip("t:T", title="Date"),
                alt.Tooltip("rideable_type:N", title="Bike type"),
                alt.Tooltip("exposure_min:Q", format=",", title="Exposure minutes")
            ]
        )
        
        chart_mix = alt.layer(trips_chart_mix, exposure_chart_mix).resolve_scale(
            y='independent'
        ).properties(height=280)
        
        st.altair_chart(chart_mix, use_container_width=True)
        st.caption("Note: **Solid:** Trips (left) | **Dashed:** Exposure (right)")
    
    elif show_trips:
        chart_mix = (
            alt.Chart(mix_time)
            .mark_line(point=False)
            .encode(
                x=alt.X("t:T", title="Date"),
                y=alt.Y("n_trips:Q", title="Trips"),
                color=alt.Color("rideable_type:N", title="Bike type"),
                tooltip=[
                    alt.Tooltip("t:T", title="Date"),
                    alt.Tooltip("rideable_type:N", title="Bike type"),
                    alt.Tooltip("n_trips:Q", format=",", title="Trips")
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(chart_mix, use_container_width=True)

    elif show_exposure:
        chart_mix = (
            alt.Chart(mix_time)
            .mark_line(point=False)
            .encode(
                x=alt.X("t:T", title="Date"),
                y=alt.Y("exposure_min:Q", title="Exposure Minutes"),
                color=alt.Color("rideable_type:N", title="Bike type"),
                tooltip=[
                    alt.Tooltip("t:T", title="Date"),
                    alt.Tooltip("rideable_type:N", title="Bike type"),
                    alt.Tooltip("exposure_min:Q", format=",", title="Exposure minutes")
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(chart_mix, use_container_width=True)

with col_right:
    st.markdown("**Borough distribution**")
    
    if borough_dist.empty:
        st.info("No borough data available for the selected filters.")
    else:
        if show_trips and show_exposure:
            # Show trips by default when both selected
            chart_borough = (
                alt.Chart(borough_dist)
                .mark_arc()
                .encode(
                    theta=alt.Theta("n_trips:Q"),
                    color=alt.Color("borough:N", title="Borough"),
                    tooltip=[
                        alt.Tooltip("borough:N", title="Borough"),
                        alt.Tooltip("n_trips:Q", format=",", title="Trips"),
                        alt.Tooltip("trips_pct:Q", format=".1f", title="Percentage (%)")
                    ]
                )
                .properties(height=280)
            )
            st.altair_chart(chart_borough, use_container_width=True)
            st.caption("Showing **Trips** distribution")
        
        elif show_trips:
            chart_borough = (
                alt.Chart(borough_dist)
                .mark_arc()
                .encode(
                    theta=alt.Theta("n_trips:Q"),
                    color=alt.Color("borough:N", title="Borough"),
                    tooltip=[
                        alt.Tooltip("borough:N", title="Borough"),
                        alt.Tooltip("n_trips:Q", format=",", title="Trips"),
                        alt.Tooltip("trips_pct:Q", format=".1f", title="Percentage (%)")
                    ]
                )
                .properties(height=280)
            )
            st.altair_chart(chart_borough, use_container_width=True)
        
        elif show_exposure:
            chart_borough = (
                alt.Chart(borough_dist)
                .mark_arc()
                .encode(
                    theta=alt.Theta("exposure_min:Q"),
                    color=alt.Color("borough:N", title="Borough"),
                    tooltip=[
                        alt.Tooltip("borough:N", title="Borough"),
                        alt.Tooltip("exposure_min:Q", format=",", title="Exposure minutes"),
                        alt.Tooltip("exposure_pct:Q", format=".1f", title="Percentage (%)")
                    ]
                )
                .properties(height=280)
            )
            st.altair_chart(chart_borough, use_container_width=True)


# ============================================================================
# 5) WEATHER IMPACT ON USAGE
# ============================================================================
st.subheader("5) Weather impact on usage patterns")

# Check if weather data is available
try:
    con.execute(f"SELECT 1 FROM read_parquet('{WEATHER_GLOB.as_posix()}', hive_partitioning=1) LIMIT 1")
    has_weather = True
except:
    has_weather = False

if not has_weather:
    st.info("Weather data not available for this analysis.")
else:
    # Create weather view
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW weather_hourly AS
    SELECT
      date_trunc('hour', timezone('America/New_York', timestamp))::TIMESTAMP AS hour_ts,
      temp, prcp, snow, wspd
    FROM read_parquet('{WEATHER_GLOB.as_posix()}', hive_partitioning=1);
    """)
    
    # Join trips with weather
    weather_joined = con.execute(f"""
    WITH trips_agg AS (
      SELECT 
        hour_ts,
        COALESCE(SUM(n_trips),0) AS n_trips,
        COALESCE(SUM(exposure_min),0) AS exposure_min
      FROM read_parquet('{TRIPS_BOROUGH_HOUR.as_posix()}')
      WHERE {where_sql}
      GROUP BY 1
    ),
    overall_avg AS (
      SELECT
        AVG(n_trips) AS avg_trips,
        AVG(exposure_min) AS avg_exposure
      FROM trips_agg
    )
    SELECT 
      t.hour_ts,
      t.n_trips,
      t.exposure_min,
      w.temp,
      w.prcp,
      w.snow,
      w.wspd,
      o.avg_trips,
      o.avg_exposure
    FROM trips_agg t
    LEFT JOIN weather_hourly w USING(hour_ts)
    CROSS JOIN overall_avg o
    WHERE w.temp IS NOT NULL;
    """).fetch_df()
    
    if len(weather_joined) == 0:
        st.info("No weather data available for the selected time range.")
    else:
        # Calculate percentage deviations
        weather_joined['trips_dev_pct'] = 100 * (weather_joined['n_trips'] - weather_joined['avg_trips']) / weather_joined['avg_trips']
        weather_joined['exposure_dev_pct'] = 100 * (weather_joined['exposure_min'] - weather_joined['avg_exposure']) / weather_joined['avg_exposure']
        
        # Create weather categories
        weather_joined['temp_cat'] = pd.cut(
            weather_joined['temp'],
            bins=[-float('inf'), 0, 10, 20, 25, float('inf')],
            labels=['Very Cold (<0°C)', 'Cold (0-10°C)', 'Mild (10-20°C)', 'Warm (20-25°C)', 'Hot (>25°C)']
        )
        
        weather_joined['rain_cat'] = pd.cut(
            weather_joined['prcp'],
            bins=[-0.01, 0.01, 1.0, 5.0, float('inf')],
            labels=['No Rain', 'Light Rain (0-1mm)', 'Moderate Rain (1-5mm)', 'Heavy Rain (>5mm)']
        )
        
        weather_joined['snow_cat'] = (weather_joined['snow'] > 0.1).map({True: 'Snow', False: 'No Snow'})
        
        # Create three columns for different weather aspects
        col_a, col_b, col_c = st.columns(3)
        
        # TEMPERATURE IMPACT
        # TEMPERATURE IMPACT
        with col_a:
            st.markdown("**Temperature Impact**")
            
            # Create temperature bins (2°C intervals for smoother curve)
            weather_temp = weather_joined.dropna(subset=['temp']).copy()
            weather_temp['temp_bin'] = pd.cut(
                weather_temp['temp'],
                bins=range(int(weather_temp['temp'].min()) - 2, int(weather_temp['temp'].max()) + 4, 2),
                include_lowest=True
            )
            
            # Calculate mean temperature and deviations per bin
            temp_agg = weather_temp.groupby('temp_bin', observed=True).agg({
                'temp': 'mean',
                'trips_dev_pct': 'mean',
                'exposure_dev_pct': 'mean',
                'n_trips': 'count'  # To filter out bins with too few observations
            }).reset_index()
            
            # Filter out bins with very few observations (< 10)
            temp_agg = temp_agg[temp_agg['n_trips'] >= 10].copy()
            temp_agg = temp_agg.sort_values('temp')
            
            if len(temp_agg) == 0:
                st.info("Insufficient temperature data for analysis")
            else:
                if show_trips and show_exposure:
                    # Create dual Y-axis chart
                    temp_trips = temp_agg[['temp', 'trips_dev_pct']].copy()
                    temp_exposure = temp_agg[['temp', 'exposure_dev_pct']].copy()
                    
                    # Left Y-axis: Trips
                    trips_chart_temp = alt.Chart(temp_trips).mark_line(color='#1f77b4', point=True, strokeWidth=2).encode(
                        x=alt.X('temp:Q', title='Temperature (°C)', scale=alt.Scale(zero=False)),
                        y=alt.Y('trips_dev_pct:Q', title='Trips deviation (%)'),
                        tooltip=[
                            alt.Tooltip('temp:Q', format='.1f', title='Temperature (°C)'),
                            alt.Tooltip('trips_dev_pct:Q', format='.1f', title='Trips deviation (%)')
                        ]
                    )
                    
                    # Right Y-axis: Exposure
                    exposure_chart_temp = alt.Chart(temp_exposure).mark_line(color='#ff7f0e', point=True, strokeWidth=2).encode(
                        x=alt.X('temp:Q', title='Temperature (°C)', scale=alt.Scale(zero=False)),
                        y=alt.Y('exposure_dev_pct:Q', title='Exposure deviation (%)'),
                        tooltip=[
                            alt.Tooltip('temp:Q', format='.1f', title='Temperature (°C)'),
                            alt.Tooltip('exposure_dev_pct:Q', format='.1f', title='Exposure deviation (%)')
                        ]
                    )
                    
                    chart_temp = alt.layer(trips_chart_temp, exposure_chart_temp).resolve_scale(
                        y='independent'
                    ).properties(height=280)
                    
                    st.altair_chart(chart_temp, use_container_width=True)
                    st.caption("Note: **Blue (left):** Trips | **Orange (right):** Exposure")
                    
                elif show_trips:
                    chart_temp = (
                        alt.Chart(temp_agg)
                        .mark_line(color='#1f77b4', point=True, strokeWidth=2)
                        .encode(
                            x=alt.X('temp:Q', title='Temperature (°C)', scale=alt.Scale(zero=False)),
                            y=alt.Y('trips_dev_pct:Q', title='Trips deviation (%)'),
                            tooltip=[
                                alt.Tooltip('temp:Q', format='.1f', title='Temperature (°C)'),
                                alt.Tooltip('trips_dev_pct:Q', format='.1f', title='Trips deviation (%)')
                            ]
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(chart_temp, use_container_width=True)
                    
                elif show_exposure:
                    chart_temp = (
                        alt.Chart(temp_agg)
                        .mark_line(color='#ff7f0e', point=True, strokeWidth=2)
                        .encode(
                            x=alt.X('temp:Q', title='Temperature (°C)', scale=alt.Scale(zero=False)),
                            y=alt.Y('exposure_dev_pct:Q', title='Exposure deviation (%)'),
                            tooltip=[
                                alt.Tooltip('temp:Q', format='.1f', title='Temperature (°C)'),
                                alt.Tooltip('exposure_dev_pct:Q', format='.1f', title='Exposure deviation (%)')
                            ]
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(chart_temp, use_container_width=True)
        
        # PRECIPITATION IMPACT
        with col_b:
            st.markdown("**Precipitation Impact**")
            
            if show_trips and show_exposure:
                rain_agg = weather_joined.groupby('rain_cat', observed=True).agg({
                    'trips_dev_pct': 'mean',
                    'exposure_dev_pct': 'mean'
                }).reset_index()
                
                rain_melt = rain_agg.melt(
                    id_vars='rain_cat',
                    value_vars=['trips_dev_pct', 'exposure_dev_pct'],
                    var_name='metric',
                    value_name='deviation_pct'
                )
                rain_melt['metric'] = rain_melt['metric'].map({
                    'trips_dev_pct': 'Trips',
                    'exposure_dev_pct': 'Exposure'
                })
                
                chart_rain = (
                    alt.Chart(rain_melt)
                    .mark_bar()
                    .encode(
                        x=alt.X('rain_cat:N', title='Precipitation', axis=alt.Axis(labelAngle=-45)),
                        y=alt.Y('deviation_pct:Q', title='Deviation from average (%)'),
                        color=alt.Color('metric:N', title='Metric', scale=alt.Scale(domain=['Trips', 'Exposure'], range=['#1f77b4', '#ff7f0e'])),
                        xOffset='metric:N',
                        tooltip=[
                            alt.Tooltip('rain_cat:N', title='Precipitation'),
                            alt.Tooltip('metric:N', title='Metric'),
                            alt.Tooltip('deviation_pct:Q', format='.1f', title='Deviation (%)')
                        ]
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart_rain, use_container_width=True)
                
            elif show_trips:
                rain_agg = weather_joined.groupby('rain_cat', observed=True)['trips_dev_pct'].mean().reset_index()
                
                chart_rain = (
                    alt.Chart(rain_agg)
                    .mark_bar(color='#1f77b4')
                    .encode(
                        x=alt.X('rain_cat:N', title='Precipitation', axis=alt.Axis(labelAngle=-45)),
                        y=alt.Y('trips_dev_pct:Q', title='Trips deviation (%)'),
                        tooltip=[
                            alt.Tooltip('rain_cat:N', title='Precipitation'),
                            alt.Tooltip('trips_dev_pct:Q', format='.1f', title='Deviation (%)')
                        ]
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart_rain, use_container_width=True)
                
            elif show_exposure:
                rain_agg = weather_joined.groupby('rain_cat', observed=True)['exposure_dev_pct'].mean().reset_index()
                
                chart_rain = (
                    alt.Chart(rain_agg)
                    .mark_bar(color='#ff7f0e')
                    .encode(
                        x=alt.X('rain_cat:N', title='Precipitation', axis=alt.Axis(labelAngle=-45)),
                        y=alt.Y('exposure_dev_pct:Q', title='Exposure deviation (%)'),
                        tooltip=[
                            alt.Tooltip('rain_cat:N', title='Precipitation'),
                            alt.Tooltip('exposure_dev_pct:Q', format='.1f', title='Deviation (%)')
                        ]
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart_rain, use_container_width=True)
        
        # SNOW IMPACT
        with col_c:
            st.markdown("**Snow Impact**")
            
            if show_trips and show_exposure:
                snow_agg = weather_joined.groupby('snow_cat').agg({
                    'trips_dev_pct': 'mean',
                    'exposure_dev_pct': 'mean'
                }).reset_index()
                
                snow_melt = snow_agg.melt(
                    id_vars='snow_cat',
                    value_vars=['trips_dev_pct', 'exposure_dev_pct'],
                    var_name='metric',
                    value_name='deviation_pct'
                )
                snow_melt['metric'] = snow_melt['metric'].map({
                    'trips_dev_pct': 'Trips',
                    'exposure_dev_pct': 'Exposure'
                })
                
                chart_snow = (
                    alt.Chart(snow_melt)
                    .mark_bar()
                    .encode(
                        x=alt.X('snow_cat:N', title='Snow condition'),
                        y=alt.Y('deviation_pct:Q', title='Deviation from average (%)'),
                        color=alt.Color('metric:N', title='Metric', scale=alt.Scale(domain=['Trips', 'Exposure'], range=['#1f77b4', '#ff7f0e'])),
                        xOffset='metric:N',
                        tooltip=[
                            alt.Tooltip('snow_cat:N', title='Snow'),
                            alt.Tooltip('metric:N', title='Metric'),
                            alt.Tooltip('deviation_pct:Q', format='.1f', title='Deviation (%)')
                        ]
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart_snow, use_container_width=True)
                
            elif show_trips:
                snow_agg = weather_joined.groupby('snow_cat')['trips_dev_pct'].mean().reset_index()
                
                chart_snow = (
                    alt.Chart(snow_agg)
                    .mark_bar(color='#1f77b4')
                    .encode(
                        x=alt.X('snow_cat:N', title='Snow condition'),
                        y=alt.Y('trips_dev_pct:Q', title='Trips deviation (%)'),
                        tooltip=[
                            alt.Tooltip('snow_cat:N', title='Snow'),
                            alt.Tooltip('trips_dev_pct:Q', format='.1f', title='Deviation (%)')
                        ]
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart_snow, use_container_width=True)
                
            elif show_exposure:
                snow_agg = weather_joined.groupby('snow_cat')['exposure_dev_pct'].mean().reset_index()
                
                chart_snow = (
                    alt.Chart(snow_agg)
                    .mark_bar(color='#ff7f0e')
                    .encode(
                        x=alt.X('snow_cat:N', title='Snow condition'),
                        y=alt.Y('exposure_dev_pct:Q', title='Exposure deviation (%)'),
                        tooltip=[
                            alt.Tooltip('snow_cat:N', title='Snow'),
                            alt.Tooltip('exposure_dev_pct:Q', format='.1f', title='Deviation (%)')
                        ]
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart_snow, use_container_width=True)
        
        st.caption("""
        **How to read:** Positive values = higher usage than average, negative values = lower usage than average.
        For example, -30% means usage is 30% below the overall average during those weather conditions.
        """)