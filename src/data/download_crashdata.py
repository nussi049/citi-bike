"""
NYPD Motor Vehicle Collision Data Loader
========================================

Usage:
    from src.data.crashdata import CrashDataLoader
    
    loader = CrashDataLoader()
    loader.download()
"""

import logging
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "crashes"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

# SODA2 API
API_ENDPOINT = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"
BATCH_SIZE = 50000  # Max rows per request


class CrashDataLoader:
    
    def __init__(self):
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    
    def download(self) -> Path:
        """Download all crash data from SODA2 API."""

        output_path = RAW_DIR / "crashes.parquet"

        # Check if parquet file already exists
        if output_path.exists():
            log.info(f"Crash data already exists: {output_path}")
            log.info(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")
            return output_path

        log.info("Downloading crash data from NYC Open Data...")
        
        # Fixed schema
        schema = pa.schema([
            ('crash_date', pa.timestamp('us')),
            ('crash_time', pa.string()),
            ('borough', pa.string()),
            ('zip_code', pa.string()),
            ('latitude', pa.float64()),
            ('longitude', pa.float64()),
            ('on_street_name', pa.string()),
            ('cross_street_name', pa.string()),
            ('number_of_persons_injured', pa.int32()),
            ('number_of_persons_killed', pa.int32()),
            ('number_of_pedestrians_injured', pa.int32()),
            ('number_of_pedestrians_killed', pa.int32()),
            ('number_of_cyclist_injured', pa.int32()),
            ('number_of_cyclist_killed', pa.int32()),
            ('number_of_motorist_injured', pa.int32()),
            ('number_of_motorist_killed', pa.int32()),
            ('contributing_factor_vehicle_1', pa.string()),
            ('contributing_factor_vehicle_2', pa.string()),
            ('vehicle_type_code1', pa.string()),
            ('vehicle_type_code2', pa.string()),
            ('collision_id', pa.int64()),
        ])
        
        writer = pq.ParquetWriter(output_path, schema)
        total_rows = 0

        # NYC Open Data has a 250k row limit even with offset, so we fetch by year
        # CitiBike started June 2013, download through current year
        start_year = 2012  # Start a bit earlier to catch all data
        end_year = 2026    # Go into the future to get all available data

        log.info(f"Downloading crashes by year ({start_year} to {end_year})...")

        with tqdm(desc="Downloading", unit=" rows") as pbar:
            for year in range(start_year, end_year + 1):
                year_start = f"{year}-01-01"
                year_end = f"{year + 1}-01-01"

                offset = 0
                year_rows = 0

                while True:
                    params = {
                        "$limit": BATCH_SIZE,
                        "$offset": offset,
                        "$order": "crash_date ASC",
                        "$where": f"crash_date >= '{year_start}' AND crash_date < '{year_end}'"
                    }

                    response = requests.get(API_ENDPOINT, params=params)
                    response.raise_for_status()

                    data = response.json()

                    if not data:
                        break

                    df = pd.DataFrame(data)
                    df = self._normalize_columns(df)

                    for col in schema.names:
                        if col not in df.columns:
                            df[col] = None

                    df = df[schema.names]

                    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
                    writer.write_table(table)

                    year_rows += len(df)
                    total_rows += len(df)
                    pbar.update(len(df))

                    if len(data) < BATCH_SIZE:
                        break

                    offset += BATCH_SIZE

                    del df, table

                if year_rows > 0:
                    log.info(f"  Year {year}: {year_rows:,} crashes")

        writer.close()

        log.info(f"Done! Saved {total_rows:,} rows to {output_path}")
        log.info(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")
        
        return output_path
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Parse date
        if 'crash_date' in df.columns:
            df['crash_date'] = pd.to_datetime(df['crash_date'], errors='coerce')
        
        # Convert numeric columns
        int_cols = [
            'number_of_persons_injured', 'number_of_persons_killed',
            'number_of_pedestrians_injured', 'number_of_pedestrians_killed',
            'number_of_cyclist_injured', 'number_of_cyclist_killed',
            'number_of_motorist_injured', 'number_of_motorist_killed',
            'collision_id'
        ]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        float_cols = ['latitude', 'longitude']
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # String columns
        string_cols = ['crash_time', 'borough', 'zip_code', 'on_street_name', 
                      'cross_street_name', 'contributing_factor_vehicle_1',
                      'contributing_factor_vehicle_2', 'vehicle_type_code1', 
                      'vehicle_type_code2']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
        
        return df
    
    def check_info(self):
        """Show dataset info."""
        params = {"$select": "count(*) as total"}
        response = requests.get(API_ENDPOINT, params=params)
        total = response.json()[0]['total']
        log.info(f"Total crashes available: {int(total):,}")


def main():
    loader = CrashDataLoader()
    loader.download()
    

if __name__ == "__main__":
    main()