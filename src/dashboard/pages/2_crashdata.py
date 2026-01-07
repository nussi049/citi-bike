import streamlit as st
import pandas as pd
import altair as alt

from src.dashboard.lib.db import get_con
from src.dashboard.lib.settings import CRASHES_BIKE, WEATHER_GLOB

st.title("Bike Crashes")

con = get_con()

def warn_if_missing(path, label):
    try:
        con.execute(f"SELECT 1 FROM read_parquet('{path.as_posix()}') LIMIT 1")
        return True
    except Exception as e:
        st.error(f"Missing/invalid {label}: {path}\n\n{e}")
        return False

if not warn_if_missing(CRASHES_BIKE, "bike crashes dataset"):
    st.stop()

# ============================================================================
# PARSE TIMESTAMPS ON-THE-FLY (crash_date + crash_time → hour_ts)
# ============================================================================
con.execute(f"""
CREATE OR REPLACE TEMP VIEW crashes_hourly AS
WITH parsed AS (
  SELECT
    crash_date::TIMESTAMP AS crash_date,
    crash_time,
    upper(trim(coalesce(borough,''))) AS borough,
    trim(coalesce(zip_code,'')) AS zip_code,
    1 AS y_bike,
    -- Parse crash_time (format: 'H:MM' or 'HH:MM')
    (
      try_cast(split_part(crash_time, ':', 1) AS INTEGER) * 60
      + try_cast(split_part(crash_time, ':', 2) AS INTEGER)
    ) AS minutes_since_midnight
  FROM read_parquet('{CRASHES_BIKE.as_posix()}')
  WHERE crash_time IS NOT NULL
)
SELECT
  date_trunc('hour', crash_date + (minutes_since_midnight || ' minutes')::INTERVAL)::TIMESTAMP AS hour_ts,
  NULLIF(borough, '') AS borough,
  NULLIF(zip_code, '') AS zip_code,
  y_bike
FROM parsed
WHERE minutes_since_midnight IS NOT NULL;
""")

# ============================================================================
# SIDEBAR FILTERS
# ============================================================================
with st.sidebar:
    st.header("Filters")
    start = st.date_input("Start date", value=pd.to_datetime("2020-01-01").date())
    end = st.date_input("End date (exclusive)", value=pd.to_datetime("2026-01-01").date())
    
    # Get available boroughs (excluding NULL)
    boroughs = con.execute("""
        SELECT DISTINCT borough
        FROM crashes_hourly
        WHERE borough IS NOT NULL
        ORDER BY borough
    """).fetch_df()["borough"].tolist()
    
    borough_sel = st.multiselect("Borough", boroughs, default=boroughs[:])
    
    gran = st.selectbox("Timeseries", ["day", "week", "month"])

trunc = {"day": "day", "week": "week", "month": "month"}[gran]

# Build WHERE clause
where = ["1=1"]
where.append(f"hour_ts >= TIMESTAMP '{start.isoformat()} 00:00:00'")
where.append(f"hour_ts < TIMESTAMP '{end.isoformat()} 00:00:00'")

# CRITICAL: Only add borough filter if NOT all boroughs are selected
# This ensures crashes with NULL borough are included when showing "all"
if borough_sel and len(borough_sel) < len(boroughs):
    b_list = ",".join([f"'{b}'" for b in borough_sel])
    where.append(f"borough IN ({b_list})")

where_sql = " AND ".join(where)

# ============================================================================
# KPIs
# ============================================================================
kpi = con.execute(f"""
SELECT COALESCE(SUM(y_bike), 0) AS total
FROM crashes_hourly
WHERE {where_sql};
""").fetch_df()

total_crashes = int(float(kpi.loc[0, 'total']))

c1, c2, c3 = st.columns(3)
c1.metric("Total bike crashes", f"{total_crashes:,}")

# Calculate crashes per day
days_in_range = (pd.to_datetime(end) - pd.to_datetime(start)).days
if days_in_range > 0:
    crashes_per_day = total_crashes / days_in_range
    c2.metric("Crashes per day", f"{crashes_per_day:.2f}")

# Calculate percentage by borough (if filtered)
if len(borough_sel) < len(boroughs) and len(borough_sel) > 0:
    all_crashes = con.execute(f"""
        SELECT COALESCE(SUM(y_bike), 0) AS total
        FROM crashes_hourly
        WHERE hour_ts >= TIMESTAMP '{start.isoformat()} 00:00:00'
        AND hour_ts < TIMESTAMP '{end.isoformat()} 00:00:00'
    """).fetch_df()
    all_total = float(all_crashes.loc[0, 'total'])
    if all_total > 0:
        pct = 100 * total_crashes / all_total
        c3.metric("% of all crashes", f"{pct:.1f}%")

if total_crashes == 0:
    st.warning("No crashes in the selected filter window. Expand the date range or relax filters.")
    st.stop()

# ============================================================================
# 1) TREND OVER TIME
# ============================================================================
st.subheader("1) Crashes over time")

ts = con.execute(f"""
SELECT 
  date_trunc('{trunc}', hour_ts) AS t, 
  COALESCE(SUM(y_bike), 0) AS crashes
FROM crashes_hourly
WHERE {where_sql}
GROUP BY 1
ORDER BY 1;
""").fetch_df()

chart_ts = (
    alt.Chart(ts)
    .mark_line(color='#d62728', point=False, strokeWidth=2)
    .encode(
        x=alt.X("t:T", title="Date"),
        y=alt.Y("crashes:Q", title="Bike Crashes"),
        tooltip=[
            alt.Tooltip("t:T", title="Date"),
            alt.Tooltip("crashes:Q", format=",", title="Crashes")
        ]
    )
    .properties(height=320)
)
st.altair_chart(chart_ts, use_container_width=True)

# ============================================================================
# 2) TIME-OF-DAY PATTERN
# ============================================================================
st.subheader("2) Time-of-day pattern (hourly)")

hod = con.execute(f"""
SELECT 
  EXTRACT(HOUR FROM hour_ts) AS hour, 
  COALESCE(SUM(y_bike), 0) AS crashes
FROM crashes_hourly
WHERE {where_sql}
GROUP BY 1
ORDER BY 1;
""").fetch_df()

chart_hod = (
    alt.Chart(hod)
    .mark_line(color='#d62728', point=True, strokeWidth=2)
    .encode(
        x=alt.X("hour:O", title="Hour of day"),
        y=alt.Y("crashes:Q", title="Bike Crashes"),
        tooltip=[
            alt.Tooltip("hour:O", title="Hour"),
            alt.Tooltip("crashes:Q", format=",", title="Crashes")
        ]
    )
    .properties(height=320)
)
st.altair_chart(chart_hod, use_container_width=True)

# ============================================================================
# 3) DAY-OF-WEEK PATTERN + HOURLY PATTERN BY WEEKDAY
# ============================================================================
st.subheader("3) Day-of-week pattern & Hourly crashes by weekday")

# Day-of-week data
dow = con.execute(f"""
SELECT 
  EXTRACT(DOW FROM hour_ts) AS dow, 
  COALESCE(SUM(y_bike), 0) AS crashes
FROM crashes_hourly
WHERE {where_sql}
GROUP BY 1
ORDER BY 1;
""").fetch_df()

# Hourly pattern by weekday data
hod_dow = con.execute(f"""
SELECT 
  EXTRACT(HOUR FROM hour_ts) AS hour,
  EXTRACT(DOW FROM hour_ts) AS dow,
  COALESCE(SUM(y_bike), 0) AS crashes
FROM crashes_hourly
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
    
    chart_dow = (
        alt.Chart(dow)
        .mark_line(point=True, color='#d62728', strokeWidth=2)
        .encode(
            x=alt.X("dow:O", title="Day of week (0=Sun … 6=Sat)"),
            y=alt.Y("crashes:Q", title="Crashes", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("dow:O", title="Day"),
                alt.Tooltip("crashes:Q", format=",", title="Crashes")
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart_dow, use_container_width=True)

with col_right:
    st.markdown("**Hourly crashes by weekday**")
    
    chart_hod_dow = (
        alt.Chart(hod_dow)
        .mark_line(point=False)
        .encode(
            x=alt.X("hour:O", title="Hour of day"),
            y=alt.Y("crashes:Q", title="Crashes"),
            color=alt.Color("day_name:N", title="Weekday", sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
            tooltip=[
                alt.Tooltip("hour:O", title="Hour"),
                alt.Tooltip("day_name:N", title="Day"),
                alt.Tooltip("crashes:Q", format=",", title="Crashes")
            ]
        )
        .properties(height=280)
    )
    st.altair_chart(chart_hod_dow, use_container_width=True)

# ============================================================================
# 4) MONTHLY PATTERN + BOROUGH DISTRIBUTION
# ============================================================================
st.subheader("4) Monthly pattern & Borough distribution")

# Monthly pattern data
month_pattern = con.execute(f"""
SELECT 
  EXTRACT(MONTH FROM hour_ts) AS month,
  COALESCE(SUM(y_bike), 0) AS crashes
FROM crashes_hourly
WHERE {where_sql}
GROUP BY 1
ORDER BY 1;
""").fetch_df()

# Borough distribution data (always show all boroughs for the pie chart)
borough_dist = con.execute(f"""
SELECT 
  COALESCE(borough, 'Unknown') AS borough,
  COALESCE(SUM(y_bike), 0) AS crashes
FROM crashes_hourly
WHERE {where_sql}
GROUP BY 1
ORDER BY 2 DESC;
""").fetch_df()

# Calculate percentages for borough distribution
if not borough_dist.empty:
    borough_dist['crashes_pct'] = 100 * borough_dist['crashes'] / borough_dist['crashes'].sum()

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("**Monthly crash pattern**")
    
    chart_month = (
        alt.Chart(month_pattern)
        .mark_line(point=True, color='#d62728', strokeWidth=2)
        .encode(
            x=alt.X("month:O", title="Month of year"),
            y=alt.Y("crashes:Q", title="Crashes", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("month:O", title="Month"),
                alt.Tooltip("crashes:Q", format=",", title="Crashes")
            ]
        )
        .properties(height=280)
    )
    st.altair_chart(chart_month, use_container_width=True)
    st.caption("Seasonal patterns in bike crashes throughout the year")

with col_right:
    st.markdown("**Borough distribution**")
    
    if borough_dist.empty:
        st.info("No borough data available.")
    else:
        chart_borough = (
            alt.Chart(borough_dist)
            .mark_arc()
            .encode(
                theta=alt.Theta("crashes:Q"),
                color=alt.Color("borough:N", title="Borough"),
                tooltip=[
                    alt.Tooltip("borough:N", title="Borough"),
                    alt.Tooltip("crashes:Q", format=",", title="Crashes"),
                    alt.Tooltip("crashes_pct:Q", format=".1f", title="Percentage (%)")
                ]
            )
            .properties(height=280)
        )
        st.altair_chart(chart_borough, use_container_width=True)

# ============================================================================
# 5) WEATHER IMPACT ON CRASHES
# ============================================================================
st.subheader("5) Weather impact on crash patterns")

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
    
    # Join crashes with weather
    weather_joined = con.execute(f"""
    WITH crashes_agg AS (
      SELECT 
        hour_ts,
        COALESCE(SUM(y_bike), 0) AS crashes
      FROM crashes_hourly
      WHERE {where_sql}
      GROUP BY 1
    ),
    overall_avg AS (
      SELECT AVG(crashes) AS avg_crashes
      FROM crashes_agg
    )
    SELECT 
      c.hour_ts,
      c.crashes,
      w.temp,
      w.prcp,
      w.snow,
      w.wspd,
      o.avg_crashes
    FROM crashes_agg c
    LEFT JOIN weather_hourly w USING(hour_ts)
    CROSS JOIN overall_avg o
    WHERE w.temp IS NOT NULL;
    """).fetch_df()
    
    if len(weather_joined) == 0:
        st.info("No weather data available for the selected time range.")
    else:
        # Calculate percentage deviations
        weather_joined['crashes_dev_pct'] = 100 * (weather_joined['crashes'] - weather_joined['avg_crashes']) / weather_joined['avg_crashes']
        
        # Create weather categories
        weather_joined['rain_cat'] = pd.cut(
            weather_joined['prcp'],
            bins=[-0.01, 0.01, 1.0, 5.0, float('inf')],
            labels=['No Rain', 'Light Rain (0-1mm)', 'Moderate Rain (1-5mm)', 'Heavy Rain (>5mm)']
        )
        
        weather_joined['snow_cat'] = (weather_joined['snow'] > 0.1).map({True: 'Snow', False: 'No Snow'})
        
        # Create three columns for different weather aspects
        col_a, col_b, col_c = st.columns(3)
        
        # TEMPERATURE IMPACT
        with col_a:
            st.markdown("**Temperature Impact**")
            
            # Create temperature bins (2°C intervals)
            weather_temp = weather_joined.dropna(subset=['temp']).copy()
            weather_temp['temp_bin'] = pd.cut(
                weather_temp['temp'],
                bins=range(int(weather_temp['temp'].min()) - 2, int(weather_temp['temp'].max()) + 4, 2),
                include_lowest=True
            )
            
            # Calculate mean temperature and deviations per bin
            temp_agg = weather_temp.groupby('temp_bin', observed=True).agg({
                'temp': 'mean',
                'crashes_dev_pct': 'mean',
                'crashes': 'count'
            }).reset_index()
            
            # Filter out bins with very few observations (< 10)
            temp_agg = temp_agg[temp_agg['crashes'] >= 10].copy()
            temp_agg = temp_agg.sort_values('temp')
            
            if len(temp_agg) == 0:
                st.info("Insufficient temperature data")
            else:
                chart_temp = (
                    alt.Chart(temp_agg)
                    .mark_line(color='#d62728', point=True, strokeWidth=2)
                    .encode(
                        x=alt.X('temp:Q', title='Temperature (°C)', scale=alt.Scale(zero=False)),
                        y=alt.Y('crashes_dev_pct:Q', title='Crashes deviation (%)'),
                        tooltip=[
                            alt.Tooltip('temp:Q', format='.1f', title='Temperature (°C)'),
                            alt.Tooltip('crashes_dev_pct:Q', format='.1f', title='Crashes deviation (%)')
                        ]
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart_temp, use_container_width=True)
        
        # PRECIPITATION IMPACT
        with col_b:
            st.markdown("**Precipitation Impact**")
            
            rain_agg = weather_joined.groupby('rain_cat', observed=True)['crashes_dev_pct'].mean().reset_index()
            
            chart_rain = (
                alt.Chart(rain_agg)
                .mark_bar(color='#d62728')
                .encode(
                    x=alt.X('rain_cat:N', title='Precipitation', axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('crashes_dev_pct:Q', title='Crashes deviation (%)'),
                    tooltip=[
                        alt.Tooltip('rain_cat:N', title='Precipitation'),
                        alt.Tooltip('crashes_dev_pct:Q', format='.1f', title='Deviation (%)')
                    ]
                )
                .properties(height=280)
            )
            st.altair_chart(chart_rain, use_container_width=True)
        
        # SNOW IMPACT
        with col_c:
            st.markdown("**Snow Impact**")
            
            snow_agg = weather_joined.groupby('snow_cat')['crashes_dev_pct'].mean().reset_index()
            
            chart_snow = (
                alt.Chart(snow_agg)
                .mark_bar(color='#d62728')
                .encode(
                    x=alt.X('snow_cat:N', title='Snow condition'),
                    y=alt.Y('crashes_dev_pct:Q', title='Crashes deviation (%)'),
                    tooltip=[
                        alt.Tooltip('snow_cat:N', title='Snow'),
                        alt.Tooltip('crashes_dev_pct:Q', format='.1f', title='Deviation (%)')
                    ]
                )
                .properties(height=280)
            )
            st.altair_chart(chart_snow, use_container_width=True)
        
        st.caption("""
        **How to read:** Positive values = more crashes than average, negative values = fewer crashes than average.
        For example, -20% means crashes are 20% below average during those weather conditions.
        """)