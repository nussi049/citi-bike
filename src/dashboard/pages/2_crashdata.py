import streamlit as st
import pandas as pd
import altair as alt

from src.dashboard.lib.db import get_con
from src.dashboard.lib.settings import CRASHES_BIKE, WEATHER_GLOB

st.title("Panel 2 — Bike Crashes")

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
# SIDEBAR FILTERS
# ============================================================================
with st.sidebar:
    st.header("Filters")
    start = st.date_input("Start date", value=pd.to_datetime("2020-01-01").date())
    end = st.date_input("End date (exclusive)", value=pd.to_datetime("2026-01-01").date())
    gran = st.selectbox("Timeseries", ["day", "week", "month"])
    topn = st.slider("Top ZIPs", 5, 25, 10)

trunc = {"day": "day", "week": "week", "month": "month"}[gran]
where = f"hour_ts >= TIMESTAMP '{start.isoformat()} 00:00:00' AND hour_ts < TIMESTAMP '{end.isoformat()} 00:00:00'"

# ============================================================================
# PARSE TIMESTAMPS ON-THE-FLY (crash_date + crash_time → hour_ts)
# ============================================================================
con.execute("""
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
  FROM read_parquet('""" + CRASHES_BIKE.as_posix() + """')
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
# KPIs
# ============================================================================
kpi = con.execute(f"""
SELECT COALESCE(SUM(y_bike), 0) AS total
FROM crashes_hourly
WHERE {where};
""").fetch_df()

st.metric("Total bike crashes", f"{int(float(kpi.loc[0,'total'])):,}")

# ============================================================================
# 1) TREND OVER TIME
# ============================================================================
st.subheader("1) Trend over time")

ts = con.execute(f"""
SELECT 
  date_trunc('{trunc}', hour_ts) AS t, 
  COALESCE(SUM(y_bike), 0) AS value
FROM crashes_hourly
WHERE {where}
GROUP BY 1
ORDER BY 1;
""").fetch_df()

chart = (
    alt.Chart(ts).mark_line()
    .encode(
        x=alt.X("t:T", title="Date"),
        y=alt.Y("value:Q", title="Bike crashes"),
        tooltip=["t:T", "value:Q"]
    )
    .properties(height=320)
)
st.altair_chart(chart, use_container_width=True)

# ============================================================================
# 2) TIME-OF-DAY PATTERN
# ============================================================================
st.subheader("2) Time-of-day pattern (hourly)")

hod = con.execute(f"""
SELECT 
  EXTRACT(HOUR FROM hour_ts) AS hour, 
  COALESCE(SUM(y_bike), 0) AS value
FROM crashes_hourly
WHERE {where}
GROUP BY 1
ORDER BY 1;
""").fetch_df()

chart = (
    alt.Chart(hod).mark_line(point=True)
    .encode(
        x=alt.X("hour:O", title="Hour of day"),
        y=alt.Y("value:Q", title="Bike crashes (sum)"),
        tooltip=["hour:O", "value:Q"]
    )
    .properties(height=260)
)
st.altair_chart(chart, use_container_width=True)

# ============================================================================
# 3) BOROUGH COMPARISON
# ============================================================================
st.subheader("3) Borough comparison")

bb = con.execute(f"""
SELECT 
  borough, 
  COALESCE(SUM(y_bike), 0) AS value
FROM crashes_hourly
WHERE {where}
GROUP BY 1
ORDER BY value DESC;
""").fetch_df()

chart = (
    alt.Chart(bb).mark_bar()
    .encode(
        x=alt.X("borough:N", title="Borough", sort="-y"),
        y=alt.Y("value:Q", title="Bike crashes (sum)"),
        tooltip=["borough:N", "value:Q"]
    )
    .properties(height=280)
)
st.altair_chart(chart, use_container_width=True)

# ============================================================================
# 4) TOP ZIP CODES
# ============================================================================
st.subheader("4) Top ZIP codes")

zz = con.execute(f"""
SELECT 
  zip_code, 
  COALESCE(SUM(y_bike), 0) AS value
FROM crashes_hourly
WHERE {where} AND zip_code IS NOT NULL
GROUP BY 1
ORDER BY value DESC
LIMIT {topn};
""").fetch_df()

chart = (
    alt.Chart(zz).mark_bar()
    .encode(
        x=alt.X("zip_code:N", title="ZIP", sort="-y"),
        y=alt.Y("value:Q", title="Bike crashes"),
        tooltip=["zip_code:N", "value:Q"]
    )
    .properties(height=280)
)
st.altair_chart(chart, use_container_width=True)

# ============================================================================
# 5) WEATHER EFFECTS
# ============================================================================
st.subheader("5) Weather effects (simple)")

# Check if weather data exists
try:
    con.execute(f"SELECT 1 FROM read_parquet('{WEATHER_GLOB.as_posix()}', hive_partitioning=1) LIMIT 1")
    has_weather = True
except:
    has_weather = False

if has_weather:
    # Create weather hourly view (NYC timezone)
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW weather_hourly AS
    SELECT
      date_trunc('hour', timezone('America/New_York', timestamp))::TIMESTAMP AS hour_ts,
      temp, prcp, snow, wspd
    FROM read_parquet('{WEATHER_GLOB.as_posix()}', hive_partitioning=1);
    """)
    
    joined = con.execute(f"""
    WITH c AS (
      SELECT hour_ts, COALESCE(SUM(y_bike), 0) AS y
      FROM crashes_hourly
      WHERE {where}
      GROUP BY 1
    ),
    w AS (
      SELECT hour_ts, temp, prcp, snow, wspd 
      FROM weather_hourly
    )
    SELECT c.hour_ts, c.y, w.temp, w.prcp, w.snow, w.wspd
    FROM c
    LEFT JOIN w USING(hour_ts)
    WHERE w.temp IS NOT NULL;
    """).fetch_df()

    if len(joined) == 0:
        st.info("No weather-joined data in this window.")
    else:
        x = st.selectbox("Weather variable", ["temp", "prcp", "snow", "wspd"])
        
        if x == "snow":
            joined["snowing"] = (joined["snow"].fillna(0.0) > 0).astype(int)
            agg = joined.groupby("snowing", as_index=False)["y"].sum()
            agg["snowing"] = agg["snowing"].map({0: "No snow", 1: "Snow"})
            
            chart = (
                alt.Chart(agg).mark_bar()
                .encode(
                    x=alt.X("snowing:N", title="Snow indicator"),
                    y=alt.Y("y:Q", title="Bike crashes (sum)"),
                    tooltip=["snowing:N", "y:Q"]
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("Snow is rare, so we show a simple comparison: snow vs no snow.")
        else:
            bins = st.slider("Bins", 10, 50, 25)
            df = joined.dropna(subset=[x]).copy()
            df["bin"] = pd.qcut(df[x], q=bins, duplicates="drop")
            agg = df.groupby("bin", observed=True).agg(
                x_mean=(x, "mean"), 
                y=("y", "sum")
            ).reset_index(drop=True).sort_values("x_mean")
            
            chart = (
                alt.Chart(agg).mark_line(point=True)
                .encode(
                    x=alt.X("x_mean:Q", title=f"{x} (binned mean)"),
                    y=alt.Y("y:Q", title="Bike crashes (sum in bin)"),
                    tooltip=["x_mean:Q", "y:Q"]
                )
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("This is a descriptive relationship (binned totals), not a causal estimate.")
else:
    st.info("Weather data not available.")
