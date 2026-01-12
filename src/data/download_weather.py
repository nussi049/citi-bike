# src/data/weather.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

URL = "https://archive-api.open-meteo.com/v1/archive"

NYC_LAT = 40.7128
NYC_LON = -74.0060


@dataclass
class WeatherHourlyConfig:
    out_dir: Path = RAW_DIR / "weather_hourly_openmeteo"  # dataset directory
    latitude: float = NYC_LAT
    longitude: float = NYC_LON
    start_date: str = "2013-01-01"
    end_date: str | None = None  # defaults to today
    timezone: str = "UTC"

    # Keep the variable list modest to avoid unnecessary size
    hourly_vars: tuple[str, ...] = (
        "temperature_2m",
        "precipitation",
        "snowfall",
        "wind_speed_10m",
        # Optional adds (uncomment if you want them):
        # "relative_humidity_2m",
        # "cloud_cover",
        # "visibility",
    )


class WeatherDataLoaderHourly:
    def __init__(self, cfg: WeatherHourlyConfig = WeatherHourlyConfig()):
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_range(self, start: str, end: str) -> pd.DataFrame:
        params = {
            "latitude": self.cfg.latitude,
            "longitude": self.cfg.longitude,
            "start_date": start,
            "end_date": end,
            "hourly": list(self.cfg.hourly_vars),
            "timezone": self.cfg.timezone,
        }

        log.info(f"Request Open-Meteo hourly: {start} → {end}")
        r = requests.get(URL, params=params, timeout=120)
        r.raise_for_status()
        js = r.json()

        hourly = js.get("hourly")
        if not hourly:
            raise RuntimeError(f"No 'hourly' in response. Keys: {list(js.keys())}")

        df = pd.DataFrame(hourly)

        # Open-Meteo uses "time" key for timestamps
        if "time" not in df.columns:
            raise RuntimeError(f"No 'time' column in hourly payload. Columns: {df.columns.tolist()}")

        df = df.rename(columns={"time": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        # Rename vars to short names (optional)
        rename_map = {
            "temperature_2m": "temp",
            "precipitation": "prcp",
            "snowfall": "snow",
            "wind_speed_10m": "wspd",
            "relative_humidity_2m": "rhum",
            "cloud_cover": "cloud",
            "visibility": "vis",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        return df

    def download(self) -> Path:
        start = datetime.fromisoformat(self.cfg.start_date).date()
        end = date.today() if self.cfg.end_date is None else datetime.fromisoformat(self.cfg.end_date).date()

        if end < start:
            raise ValueError("end_date must be >= start_date")

        # Chunk by year to keep memory safe
        for year in range(start.year, end.year + 1):
            year_start = date(year, 1, 1)
            year_end = date(year, 12, 31)

            chunk_start = max(start, year_start)
            chunk_end = min(end, year_end)

            # output file per year (easy to join later)
            out_file = self.cfg.out_dir / f"year={year}" / "weather.parquet"
            if out_file.exists():
                log.info(f"Exists, skipping: {out_file}")
                continue

            out_file.parent.mkdir(parents=True, exist_ok=True)

            df = self._fetch_range(chunk_start.isoformat(), chunk_end.isoformat())

            # Extra safety: ensure hourly granularity
            # (not strictly necessary, but helpful)
            df["year"] = df["timestamp"].dt.year.astype("int16")
            df["month"] = df["timestamp"].dt.month.astype("int8")
            df["day"] = df["timestamp"].dt.day.astype("int8")
            df["hour"] = df["timestamp"].dt.hour.astype("int8")

            # Write as parquet partition file (one per year)
            df.to_parquet(out_file, index=False)
            log.info(f"Saved: {out_file} rows={len(df):,}")

        return self.cfg.out_dir


def main():
    WeatherDataLoaderHourly().download()


if __name__ == "__main__":
    main()
