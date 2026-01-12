# src/data/geo.py
from __future__ import annotations

import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = RAW_DIR / "borough_boundaries.geojson"

# NYC Open Data "Borough Boundaries" dataset id: gthc-hcne
URL = "https://data.cityofnewyork.us/api/views/gthc-hcne/rows.geojson"


class GeoDataLoader:
    def __init__(self, out_path: Path = OUT_PATH):
        self.out_path = out_path
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def download(self) -> Path:
        if self.out_path.exists():
            log.info(f"GeoJSON exists, skipping: {self.out_path}")
            return self.out_path

        log.info(f"Downloading borough boundaries from {URL}")
        r = requests.get(URL, headers={"User-Agent": "city-bike-downloader/1.0"}, timeout=60)
        r.raise_for_status()
        self.out_path.write_bytes(r.content)

        log.info(f"Saved: {self.out_path} ({self.out_path.stat().st_size/1e6:.2f} MB)")
        return self.out_path


def main():
    GeoDataLoader().download()


if __name__ == "__main__":
    main()
