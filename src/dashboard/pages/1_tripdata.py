import streamlit as st
import pandas as pd
import altair as alt

from src.dashboard.lib.db import get_con
from src.dashboard.lib.settings import TRIPS_BOROUGH_HOUR, WEATHER_GLOB

st.title("Panel 1 — Citi Bike Trips (usage & seasonality)")

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

ts_long = ts.melt(id_vars="t", value_vars=["n_trips", "exposure_min"], var_name="metric", value_name="value")
ts_long["metric"] = ts_long["metric"].map({"n_trips": "Trips", "exposure_min": "Exposure minutes"})

chart_ts = (
    alt.Chart(ts_long)
    .mark_line(point=False)
    .encode(
        x=alt.X("t:T", title="Date"),
        y=alt.Y("value:Q", title="Value"),
        color=alt.Color("metric:N", title=""),
        tooltip=["t:T", "metric:N", "value:Q"],
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

hod_long = hod.melt(id_vars="hour", value_vars=["n_trips", "exposure_min"], var_name="metric", value_name="value")
hod_long["metric"] = hod_long["metric"].map({"n_trips": "Trips", "exposure_min": "Exposure minutes"})

chart_hod = (
    alt.Chart(hod_long)
    .mark_line(point=True)
    .encode(
        x=alt.X("hour:O", title="Hour of day"),
        y=alt.Y("value:Q", title="Total (selected range)"),
        color=alt.Color("metric:N", title=""),
        tooltip=["hour:O", "metric:N", "value:Q"],
    )
    .properties(height=280)
)
st.altair_chart(chart_hod, use_container_width=True)

# ============================================================================
# 3) DAY-OF-WEEK PATTERN
# ============================================================================
st.subheader("3) Day-of-week pattern")

dow = con.execute(f"""
SELECT
  EXTRACT(DOW FROM hour_ts) AS dow,
  COALESCE(SUM(n_trips),0) AS n_trips
FROM read_parquet('{TRIPS_BOROUGH_HOUR.as_posix()}')
WHERE {where_sql}
GROUP BY 1
ORDER BY 1;
""").fetch_df()

chart_dow = (
    alt.Chart(dow)
    .mark_bar()
    .encode(
        x=alt.X("dow:O", title="Day of week (0=Sun … 6=Sat)"),
        y=alt.Y("n_trips:Q", title="Trips"),
        tooltip=["dow:O", "n_trips:Q"],
    )
    .properties(height=260)
)
st.altair_chart(chart_dow, use_container_width=True)

# ============================================================================
# 4) BIKE TYPE MIX
# ============================================================================
st.subheader("4) Bike type mix")

mix = con.execute(f"""
SELECT
  rideable_type,
  COALESCE(SUM(n_trips),0) AS n_trips
FROM read_parquet('{TRIPS_BOROUGH_HOUR.as_posix()}')
WHERE {where_sql}
GROUP BY 1;
""").fetch_df()

chart_mix = (
    alt.Chart(mix)
    .mark_bar()
    .encode(
        x=alt.X("rideable_type:N", title="Bike type"),
        y=alt.Y("n_trips:Q", title="Trips"),
        tooltip=["rideable_type:N", "n_trips:Q"],
    )
    .properties(height=260)
)
st.altair_chart(chart_mix, use_container_width=True)

# ============================================================================
# 5) WEATHER EFFECTS
# ============================================================================
st.subheader("5) Weather effects (simple & explainable)")

try:
    con.execute(f"SELECT 1 FROM read_parquet('{WEATHER_GLOB.as_posix()}', hive_partitioning=1) LIMIT 1")
    has_weather = True
except:
    has_weather = False

if has_weather:
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW weather_hourly AS
    SELECT
      date_trunc('hour', timezone('America/New_York', timestamp))::TIMESTAMP AS hour_ts,
      temp, prcp, snow, wspd
    FROM read_parquet('{WEATHER_GLOB.as_posix()}', hive_partitioning=1);
    """)
    
    joined = con.execute(f"""
    WITH t AS (
      SELECT hour_ts, COALESCE(SUM(n_trips),0) AS n_trips
      FROM read_parquet('{TRIPS_BOROUGH_HOUR.as_posix()}')
      WHERE {where_sql}
      GROUP BY 1
    ),
    w AS (
      SELECT hour_ts, temp, prcp, snow, wspd
      FROM weather_hourly
    )
    SELECT t.hour_ts, t.n_trips, w.temp, w.prcp, w.snow, w.wspd
    FROM t
    LEFT JOIN w USING(hour_ts)
    WHERE w.temp IS NOT NULL;
    """).fetch_df()

    if len(joined) == 0:
        st.info("No matching weather rows for the selected time range.")
    else:
        x = st.selectbox("Weather variable", ["temp", "prcp", "snow", "wspd"])
        
        if x == "snow":
            joined["snowing"] = (joined["snow"].fillna(0.0) > 0).astype(int)
            snow_agg = joined.groupby("snowing", as_index=False)["n_trips"].sum()
            snow_agg["snowing"] = snow_agg["snowing"].map({0: "No snow", 1: "Snow"})
            
            chart = (
                alt.Chart(snow_agg)
                .mark_bar()
                .encode(
                    x=alt.X("snowing:N", title="Snow indicator"),
                    y=alt.Y("n_trips:Q", title="Trips (sum)"),
                    tooltip=["snowing:N", "n_trips:Q"],
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
                trips=("n_trips", "sum"),
            ).reset_index(drop=True).sort_values("x_mean")

            chart = (
                alt.Chart(agg)
                .mark_line(point=True)
                .encode(
                    x=alt.X("x_mean:Q", title=f"{x} (binned mean)"),
                    y=alt.Y("trips:Q", title="Trips (sum in bin)"),
                    tooltip=["x_mean:Q", "trips:Q"],
                )
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("This is a descriptive relationship (binned totals), not a causal estimate.")
else:
    st.info("Weather data not available.")