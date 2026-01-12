"""
CitiBike Trip Data Loader
=========================

Usage:
    from src.data.tripdata import TripDataLoader
    
    loader = TripDataLoader()
    loader.download()       # Downloads and extracts all ZIPs
    loader.merge()          # Merges all CSVs to single parquet
"""

import io
import zipfile
import logging
from pathlib import Path
from typing import List
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Paths
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "tripdata"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

# S3 Base URL
S3_BASE = "https://s3.amazonaws.com/tripdata"

# Years with yearly archives
YEAR_ARCHIVES = list(range(2013, 2024))  # 2013-2023

# Monthly files for 2024-2025
MONTHLY_FILES = [(2024, m) for m in range(1, 13)] + [(2025, m) for m in range(1, 13)]


class TripDataLoader:
    
    def __init__(self):
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    
    def download(self):
        """Download all ZIPs, extract, and delete ZIPs."""
        
        log.info("Downloading yearly archives (2015-2023)...")
        for year in YEAR_ARCHIVES:
            url = f"{S3_BASE}/{year}-citibike-tripdata.zip"
            zip_path = RAW_DIR / f"{year}-citibike-tripdata.zip"
            
            if self._download_file(url, zip_path):
                self._extract_and_delete(zip_path)
        
        log.info("Downloading monthly files (2024-2025)...")
        for year, month in MONTHLY_FILES:
            url = f"{S3_BASE}/{year}{month:02d}-citibike-tripdata.zip"
            zip_path = RAW_DIR / f"{year}{month:02d}-citibike-tripdata.zip"
            
            if self._download_file(url, zip_path):
                self._extract_and_delete(zip_path)
        
        log.info("Download complete!")
    
    def _download_file(self, url: str, path: Path) -> bool:
        """Download single file with progress bar. Returns True if downloaded."""
        if self._already_extracted(path):
            log.info(f"Already extracted: {path.stem}")
            return False
        
        if path.exists():
            log.info(f"ZIP exists: {path.name}")
            return True
        
        log.info(f"Downloading: {path.name}")
        
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 404:
                log.warning(f"Not found: {path.name}")
                return False
            response.raise_for_status()
        except requests.RequestException as e:
            log.error(f"Failed: {path.name} - {e}")
            return False
        
        total = int(response.headers.get('content-length', 0))
        
        with open(path, 'wb') as f:
            with tqdm(total=total, unit='B', unit_scale=True, desc=path.name) as pbar:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    
    def _already_extracted(self, zip_path: Path) -> bool:
        """Check if CSVs from this ZIP already exist."""
        stem = zip_path.stem  # e.g. "2023-citibike-tripdata" or "202401-citibike-tripdata"
        
        # Check for any CSV containing this identifier
        for csv in RAW_DIR.glob("*.csv"):
            if stem.split("-")[0] in csv.name:
                return True
        
        # Check in subdirectories too
        for csv in RAW_DIR.rglob("*.csv"):
            if stem.split("-")[0] in csv.name:
                return True
        
        return False
    
    def _extract_and_delete(self, zip_path: Path):
        """Extract ZIP (handles nested ZIPs) and delete after."""
        log.info(f"Extracting: {zip_path.name}")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for item in zf.namelist():
                # Skip Mac OS junk
                if '__MACOSX' in item or item.startswith('.'):
                    continue
                
                # Nested ZIP (yearly archives have these)
                if item.endswith('.zip'):
                    with zf.open(item) as nested:
                        with zipfile.ZipFile(io.BytesIO(nested.read()), 'r') as nzf:
                            for csv in nzf.namelist():
                                if csv.endswith('.csv') and '__MACOSX' not in csv:
                                    csv_name = Path(csv).name
                                    csv_path = RAW_DIR / csv_name
                                    with nzf.open(csv) as src, open(csv_path, 'wb') as dst:
                                        dst.write(src.read())
                
                # Direct CSV
                elif item.endswith('.csv'):
                    csv_name = Path(item).name
                    csv_path = RAW_DIR / csv_name
                    with zf.open(item) as src, open(csv_path, 'wb') as dst:
                        dst.write(src.read())
        
        # Delete ZIP after extraction
        zip_path.unlink()
        log.info(f"Deleted: {zip_path.name}")
    
    def merge(self) -> Path:
        """Merge all CSVs into single parquet file."""

        output_path = INTERIM_DIR / "tripdata_2013_2025.parquet"

        # Check if parquet file already exists
        if output_path.exists():
            log.info(f"Parquet file already exists: {output_path}")
            log.info(f"File size: {output_path.stat().st_size / 1e9:.2f} GB")
            return output_path

        csv_files = sorted(RAW_DIR.glob("*.csv"))

        if not csv_files:
            raise RuntimeError("No CSV files found. Run download() first.")

        log.info(f"Found {len(csv_files)} CSV files")
        
        # Fixed schema for all files
        schema = pa.schema([
            ('ride_id', pa.string()),
            ('rideable_type', pa.string()),
            ('started_at', pa.timestamp('us')),
            ('ended_at', pa.timestamp('us')),
            ('duration_sec', pa.float64()),
            ('start_station_id', pa.string()),
            ('start_station_name', pa.string()),
            ('start_lat', pa.float64()),
            ('start_lng', pa.float64()),
            ('end_station_id', pa.string()),
            ('end_station_name', pa.string()),
            ('end_lat', pa.float64()),
            ('end_lng', pa.float64()),
            ('member_casual', pa.string()),
        ])
        
        writer = pq.ParquetWriter(output_path, schema)
        total_rows = 0
        
        for csv_path in tqdm(csv_files, desc="Processing"):
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                df = self._normalize_columns(df)
                
                # Ensure all columns exist
                for col in schema.names:
                    if col not in df.columns:
                        df[col] = None
                
                # Reorder columns to match schema
                df = df[schema.names]
                
                # Convert to arrow table with fixed schema
                table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
                
                writer.write_table(table)
                total_rows += len(df)
                
                del df, table
                
            except Exception as e:
                log.error(f"Error processing {csv_path.name}: {e}")
                writer.close()
                raise
        
        writer.close()
        
        log.info(f"Done! Saved {total_rows:,} rows to {output_path}")
        log.info(f"File size: {output_path.stat().st_size / 1e9:.2f} GB")
        
        return output_path
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.strip().str.lower()
        
        rename_map = {
            'tripduration': 'duration_sec',
            'trip duration': 'duration_sec',
            'starttime': 'started_at',
            'start time': 'started_at',
            'stoptime': 'ended_at',
            'stop time': 'ended_at',
            'start station id': 'start_station_id',
            'start station name': 'start_station_name',
            'start station latitude': 'start_lat',
            'start station longitude': 'start_lng',
            'end station id': 'end_station_id',
            'end station name': 'end_station_name',
            'end station latitude': 'end_lat',
            'end station longitude': 'end_lng',
            'bikeid': 'bike_id',
            'bike id': 'bike_id',
            'usertype': 'member_casual',
            'user type': 'member_casual'
        }
        
        df = df.rename(columns=rename_map)
        
        is_old_format = 'bike_id' in df.columns and 'ride_id' not in df.columns
        
        if is_old_format:
            df['ride_id'] = [f"legacy_{i}" for i in range(len(df))]
            
            if 'member_casual' in df.columns:
                df['member_casual'] = df['member_casual'].map({
                    'Subscriber': 'member',
                    'Customer': 'casual'
                }).fillna(df['member_casual'])
            
            df['rideable_type'] = 'classic_bike'
        
        if 'started_at' not in df.columns:
            raise ValueError(f"Column 'started_at' not found. Columns: {df.columns.tolist()}")
        
        # Parse dates
        df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
        df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')
        
        # Calculate duration if not present
        if 'duration_sec' not in df.columns:
            df['duration_sec'] = (df['ended_at'] - df['started_at']).dt.total_seconds()
        
        # Convert string columns (handle NaN properly)
        string_cols = ['ride_id', 'rideable_type', 'start_station_id', 'start_station_name',
                      'end_station_id', 'end_station_name', 'member_casual']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
        
        # Convert numeric columns
        numeric_cols = ['duration_sec', 'start_lat', 'start_lng', 'end_lat', 'end_lng']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def list_files(self) -> List[Path]:
        return sorted(RAW_DIR.glob("*.csv"))
    
    def check_completeness(self):
        """Check which months are present in downloaded data."""
        csv_files = self.list_files()

        # Find all year-month combinations in filenames
        found = set()
        for f in csv_files:
            name = f.stem
            for year in range(2013, 2026):
                for month in range(1, 13):
                    if f"{year}{month:02d}" in name:
                        found.add((year, month))

        log.info("Completeness check:")

        # Expected ranges per year
        expected_ranges = {
            2013: range(6, 13),  # CitiBike started June 2013
            2025: range(1, 13),  # Full year 2025 including December
        }

        for year in range(2013, 2026):
            expected_months = expected_ranges.get(year, range(1, 13))
            found_months = [m for m in expected_months if (year, m) in found]
            missing_months = [m for m in expected_months if (year, m) not in found]

            total_expected = len(list(expected_months))
            status = "OK" if not missing_months else f"MISSING {missing_months}"
            log.info(f"{year}: {len(found_months)}/{total_expected} months - {status}")


def main():
    loader = TripDataLoader()
    loader.download()

    # Prüft die Vollständigkeit
    loader.check_completeness()

    # Alles mergen zu einem Parquet
    loader.merge()

if __name__ == "__main__":
    main()