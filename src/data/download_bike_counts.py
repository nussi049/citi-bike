# src/data/download_bike_counts.py
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import duckdb
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

BOROUGH_GEOJSON_PATH = RAW_DIR / "borough_boundaries.geojson"

OUT_DIR = PROCESSED_DIR / "nyc_bike_counters"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# NYC Open Data (Socrata)
BASE = "https://data.cityofnewyork.us/resource"
COUNTS_ID = "uczf-rk3c"     # Bicycle Counts
COUNTERS_ID = "smn3-rzf9"   # Bicycle Counters

# Verified keys:
# ucsf-rk3c: date, id, counts
COUNTS_TIME_COL = "date"
COUNTS_COUNTER_COL = "id"
COUNTS_VALUE_COL = "counts"

# smn3-rzf9: id, latitude, longitude, counter, name, timezone, ...
COUNTER_ID_COL = "id"
COUNTER_CODE_COL = "counter"
COUNTER_LAT_COL = "latitude"
COUNTER_LON_COL = "longitude"

SOCRATA_APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN", "")


def soda_get_json(dataset_id: str, soql: str, limit: int, offset: int) -> list[dict]:
    url = f"{BASE}/{dataset_id}.json"
    params = {"$query": f"{soql} LIMIT {limit} OFFSET {offset}"}
    headers = {}
    if SOCRATA_APP_TOKEN:
        headers["X-App-Token"] = SOCRATA_APP_TOKEN
    r = requests.get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def download_to_parts(dataset_id: str, soql: str, parts_dir: Path, chunk_size: int = 50000) -> int:
    parts_dir.mkdir(parents=True, exist_ok=True)
    offset = 0
    part = 0
    total = 0

    while True:
        rows = soda_get_json(dataset_id, soql, limit=chunk_size, offset=offset)
        if not rows:
            break

        df = pd.DataFrame(rows)
        # Ensure stable column names
        df.columns = [str(c) for c in df.columns]

        part_path = parts_dir / f"part_{part:05d}.parquet"
        df.to_parquet(part_path, index=False)

        n = len(df)
        total += n
        offset += n
        part += 1
        log.info(f"{dataset_id}: downloaded +{n:,} (total {total:,})")

    return total


def parquet_has_numeric_columns(parquet_path: Path) -> bool:
    """
    Detect corrupted parquet where columns are numeric (0..n) instead of names.
    Works without loading full data into memory.
    """
    if not parquet_path.exists():
        return False

    con = duckdb.connect(database=":memory:")
    try:
        # Read schema only
        con.execute(f"CREATE VIEW _v AS SELECT * FROM read_parquet('{parquet_path.as_posix()}') LIMIT 0;")
        cols = [r[0] for r in con.execute("PRAGMA table_info('_v')").fetchall()]
        # If all columns are ints or strings like "0","1","2", treat as corrupted
        if not cols:
            return True
        all_numericish = True
        for c in cols:
            s = str(c)
            if not s.isdigit():
                all_numericish = False
                break
        return all_numericish
    finally:
        con.close()


def remove_if_corrupt(parquet_path: Path, parts_dir: Path) -> None:
    if parquet_has_numeric_columns(parquet_path):
        log.warning(f"Detected corrupt parquet (numeric columns). Deleting: {parquet_path}")
        parquet_path.unlink(missing_ok=True)
        if parts_dir.exists():
            shutil.rmtree(parts_dir, ignore_errors=True)


class BikeCounterDataLoader:
    def __init__(
        self,
        start_date: str = "2020-01-01",
        end_date_exclusive: str = "2026-01-01",
        chunk_size: int = 50000,
    ):
        self.start_date = start_date
        self.end_date_exclusive = end_date_exclusive
        self.chunk_size = chunk_size

        self.duckdb_path = OUT_DIR / "bike_counters.duckdb"

        self.counters_parquet = OUT_DIR / "bicycle_counters.parquet"
        self.counts_raw_parquet = OUT_DIR / f"bicycle_counts_raw_{self.start_date}_{self.end_date_exclusive}.parquet"

        self.parts_counters = OUT_DIR / "parts_counters"
        self.parts_counts = OUT_DIR / "parts_counts"

        self.hourly_by_counter = OUT_DIR / "bike_counts_hourly_by_counter.parquet"
        self.hourly_by_counter_enriched = OUT_DIR / "bike_counts_hourly_by_counter_enriched.parquet"
        self.hourly_by_borough = OUT_DIR / "bike_counts_hourly_by_borough.parquet"
        self.daily_by_borough = OUT_DIR / "bike_counts_daily_by_borough.parquet"
        self.monthly_by_borough = OUT_DIR / "bike_counts_monthly_by_borough.parquet"

    def _connect(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(database=str(self.duckdb_path))
        con.execute("PRAGMA threads=4")
        con.execute("PRAGMA memory_limit='2GB'")
        return con

    def download(self) -> None:
        if not BOROUGH_GEOJSON_PATH.exists():
            raise FileNotFoundError(
                f"Missing {BOROUGH_GEOJSON_PATH}. Run: python -m src.data.geo"
            )

        # Check if all final outputs already exist
        all_outputs_exist = all([
            self.hourly_by_counter.exists(),
            self.hourly_by_counter_enriched.exists(),
            self.hourly_by_borough.exists(),
            self.daily_by_borough.exists(),
            self.monthly_by_borough.exists()
        ])

        if all_outputs_exist:
            log.info("All bike counter outputs already exist, skipping processing:")
            log.info(f"  - {self.hourly_by_counter}")
            log.info(f"  - {self.hourly_by_counter_enriched}")
            log.info(f"  - {self.hourly_by_borough}")
            log.info(f"  - {self.daily_by_borough}")
            log.info(f"  - {self.monthly_by_borough}")
            return

        # --- If previous runs produced corrupted parquets, remove them automatically ---
        remove_if_corrupt(self.counters_parquet, self.parts_counters)
        remove_if_corrupt(self.counts_raw_parquet, self.parts_counts)

        con = self._connect()

        # ---- 1) Download counters metadata (id, lat, lon, counter, name, timezone) ----
        if not self.counters_parquet.exists():
            soql = f"SELECT {COUNTER_ID_COL}, {COUNTER_CODE_COL}, {COUNTER_LAT_COL}, {COUNTER_LON_COL}, name, timezone"
            log.info(f"Downloading counters metadata ({COUNTERS_ID})")
            total = download_to_parts(COUNTERS_ID, soql, self.parts_counters, chunk_size=self.chunk_size)

            con.execute("DROP TABLE IF EXISTS counters_raw")
            con.execute(f"""
                CREATE TABLE counters_raw AS
                SELECT * FROM read_parquet('{(self.parts_counters / "*.parquet").as_posix()}', union_by_name=True);
            """)
            con.execute(f"COPY counters_raw TO '{self.counters_parquet.as_posix()}' (FORMAT PARQUET);")
            log.info(f"Saved: {self.counters_parquet} rows={total:,}")
        else:
            log.info(f"Exists, skipping: {self.counters_parquet}")

        # ---- 2) Download counts (ONLY date,id,counts) ----
        if not self.counts_raw_parquet.exists():
            soql = (
                f"SELECT {COUNTS_TIME_COL}, {COUNTS_COUNTER_COL}, {COUNTS_VALUE_COL} "
                f"WHERE {COUNTS_TIME_COL} >= '{self.start_date}T00:00:00.000' "
                f"AND {COUNTS_TIME_COL} < '{self.end_date_exclusive}T00:00:00.000'"
            )
            log.info(f"Downloading counts ({COUNTS_ID}) {self.start_date} → {self.end_date_exclusive}")
            total = download_to_parts(COUNTS_ID, soql, self.parts_counts, chunk_size=self.chunk_size)

            con.execute("DROP TABLE IF EXISTS counts_raw")
            con.execute(f"""
                CREATE TABLE counts_raw AS
                SELECT * FROM read_parquet('{(self.parts_counts / "*.parquet").as_posix()}', union_by_name=True);
            """)
            con.execute(f"COPY counts_raw TO '{self.counts_raw_parquet.as_posix()}' (FORMAT PARQUET);")
            log.info(f"Saved: {self.counts_raw_parquet} rows={total:,}")
        else:
            log.info(f"Exists, skipping: {self.counts_raw_parquet}")

        # ---- 3) Load clean tables ----
        con.execute("DROP TABLE IF EXISTS counts_raw")
        con.execute(f"CREATE TABLE counts_raw AS SELECT * FROM read_parquet('{self.counts_raw_parquet.as_posix()}');")

        con.execute("DROP TABLE IF EXISTS counters_raw")
        con.execute(f"CREATE TABLE counters_raw AS SELECT * FROM read_parquet('{self.counters_parquet.as_posix()}');")

        # ---- 4) Aggregate counts (daily dataset -> hour_ts at midnight) ----
        con.execute("DROP TABLE IF EXISTS bike_counts_hourly_by_counter")
        con.execute(f"""
            CREATE TABLE bike_counts_hourly_by_counter AS
            SELECT
              date_trunc('hour', try_cast({COUNTS_TIME_COL} AS TIMESTAMP)) AS hour_ts,
              {COUNTS_COUNTER_COL}::VARCHAR AS counter_id,
              SUM(try_cast({COUNTS_VALUE_COL} AS DOUBLE)) AS bike_count
            FROM counts_raw
            WHERE try_cast({COUNTS_TIME_COL} AS TIMESTAMP) IS NOT NULL
            GROUP BY 1, 2;
        """)
        con.execute(f"COPY bike_counts_hourly_by_counter TO '{self.hourly_by_counter.as_posix()}' (FORMAT PARQUET);")
        log.info(f"Wrote: {self.hourly_by_counter}")

        # ---- 5) Spatial borough mapping ----
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")

        con.execute("DROP TABLE IF EXISTS boroughs")
        con.execute(f"""
        CREATE TABLE boroughs AS
        WITH root AS (
        SELECT * FROM read_json('{BOROUGH_GEOJSON_PATH.as_posix()}')
        ),
        feat AS (
        -- GeoJSON: root.features is an array; unnest it into rows
        SELECT unnest(features) AS feature
        FROM root
        )
        SELECT
        COALESCE(
            json_extract_string(feature, '$.properties.boro_name'),
            json_extract_string(feature, '$.properties.BoroName'),
            json_extract_string(feature, '$.properties.borough'),
            json_extract_string(feature, '$.properties.boroname'),
            json_extract_string(feature, '$.properties.name')
        ) AS borough,
        ST_GeomFromGeoJSON(json_extract(feature, '$.geometry')) AS geom
        FROM feat
        WHERE json_extract(feature, '$.geometry') IS NOT NULL;
        """)

        # Choose best join key: counts.counter_id might match counters.id OR counters.counter.
        join_stats = con.execute(f"""
            WITH a AS (
              SELECT COUNT(*) AS n
              FROM bike_counts_hourly_by_counter h
              JOIN counters_raw c ON h.counter_id = c.{COUNTER_ID_COL}::VARCHAR
            ),
            b AS (
              SELECT COUNT(*) AS n
              FROM bike_counts_hourly_by_counter h
              JOIN counters_raw c ON h.counter_id = c.{COUNTER_CODE_COL}::VARCHAR
            )
            SELECT a.n AS match_on_id, b.n AS match_on_counter
            FROM a, b;
        """).fetchone()

        match_on_id, match_on_counter = int(join_stats[0]), int(join_stats[1])
        if match_on_counter > match_on_id:
            join_key = COUNTER_CODE_COL
        else:
            join_key = COUNTER_ID_COL

        log.info(f"Join matches: on counters.id={match_on_id:,}, on counters.counter={match_on_counter:,} -> using {join_key}")

        con.execute("DROP TABLE IF EXISTS counter_borough")
        con.execute(f"""
            CREATE TABLE counter_borough AS
            SELECT
              c.{join_key}::VARCHAR AS counter_id,
              COALESCE(b.borough, 'UNKNOWN') AS borough
            FROM counters_raw c
            LEFT JOIN boroughs b
              ON ST_Contains(
                   b.geom,
                   ST_Point(try_cast(c.{COUNTER_LON_COL} AS DOUBLE), try_cast(c.{COUNTER_LAT_COL} AS DOUBLE))
                 );
        """)

        con.execute("DROP TABLE IF EXISTS bike_counts_hourly_by_counter_enriched")
        con.execute("""
            CREATE TABLE bike_counts_hourly_by_counter_enriched AS
            SELECT
              h.hour_ts,
              h.counter_id,
              h.bike_count,
              COALESCE(cb.borough, 'UNKNOWN') AS borough
            FROM bike_counts_hourly_by_counter h
            LEFT JOIN counter_borough cb USING(counter_id);
        """)
        con.execute(f"COPY bike_counts_hourly_by_counter_enriched TO '{self.hourly_by_counter_enriched.as_posix()}' (FORMAT PARQUET);")
        log.info(f"Wrote: {self.hourly_by_counter_enriched}")

        # ---- 6) Aggregate to borough (hour/day/month) ----
        con.execute("DROP TABLE IF EXISTS bike_counts_hourly_by_borough")
        con.execute("""
            CREATE TABLE bike_counts_hourly_by_borough AS
            SELECT
              hour_ts,
              borough,
              SUM(bike_count) AS bike_count
            FROM bike_counts_hourly_by_counter_enriched
            GROUP BY 1,2;
        """)
        con.execute(f"COPY bike_counts_hourly_by_borough TO '{self.hourly_by_borough.as_posix()}' (FORMAT PARQUET);")
        log.info(f"Wrote: {self.hourly_by_borough}")

        con.execute("DROP TABLE IF EXISTS bike_counts_daily_by_borough")
        con.execute("""
            CREATE TABLE bike_counts_daily_by_borough AS
            SELECT
              date_trunc('day', hour_ts) AS day_ts,
              borough,
              SUM(bike_count) AS bike_count
            FROM bike_counts_hourly_by_borough
            GROUP BY 1,2;
        """)
        con.execute(f"COPY bike_counts_daily_by_borough TO '{self.daily_by_borough.as_posix()}' (FORMAT PARQUET);")
        log.info(f"Wrote: {self.daily_by_borough}")

        con.execute("DROP TABLE IF EXISTS bike_counts_monthly_by_borough")
        con.execute("""
            CREATE TABLE bike_counts_monthly_by_borough AS
            SELECT
              date_trunc('month', hour_ts) AS month_ts,
              borough,
              SUM(bike_count) AS bike_count
            FROM bike_counts_hourly_by_borough
            GROUP BY 1,2;
        """)
        con.execute(f"COPY bike_counts_monthly_by_borough TO '{self.monthly_by_borough.as_posix()}' (FORMAT PARQUET);")
        log.info(f"Wrote: {self.monthly_by_borough}")

        summary = con.execute("""
            SELECT borough, COUNT(*) AS n_rows, SUM(bike_count) AS total_count
            FROM bike_counts_hourly_by_borough
            GROUP BY 1
            ORDER BY total_count DESC;
        """).fetch_df()

        log.info("\nBorough summary:\n%s", summary.to_string(index=False))
        con.close()


def main():
    BikeCounterDataLoader().download()


if __name__ == "__main__":
    main()
