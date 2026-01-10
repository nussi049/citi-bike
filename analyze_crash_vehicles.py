#!/usr/bin/env python
"""Quick analysis of vehicle types in crash data"""

import duckdb
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
CRASH_FILE = PROJECT_ROOT / "data" / "interim" / "crashes.parquet"

con = duckdb.connect()

print("="*70)
print("VEHICLE TYPE CODE 1 - All bike/bicycle/scooter related")
print("="*70)
result = con.execute(f"""
    SELECT
        vehicle_type_code1,
        COUNT(*) as count
    FROM read_parquet('{CRASH_FILE.as_posix()}')
    WHERE vehicle_type_code1 IS NOT NULL
      AND vehicle_type_code1 != ''
      AND (
          LOWER(vehicle_type_code1) LIKE '%bike%'
          OR LOWER(vehicle_type_code1) LIKE '%bicy%'
          OR LOWER(vehicle_type_code1) LIKE '%scoot%'
      )
    GROUP BY vehicle_type_code1
    ORDER BY count DESC
    LIMIT 50
""").fetchdf()
print(result.to_string(index=False))

print("\n" + "="*70)
print("VEHICLE TYPE CODE 2 - All bike/bicycle/scooter related")
print("="*70)
result = con.execute(f"""
    SELECT
        vehicle_type_code2,
        COUNT(*) as count
    FROM read_parquet('{CRASH_FILE.as_posix()}')
    WHERE vehicle_type_code2 IS NOT NULL
      AND vehicle_type_code2 != ''
      AND (
          LOWER(vehicle_type_code2) LIKE '%bike%'
          OR LOWER(vehicle_type_code2) LIKE '%bicy%'
          OR LOWER(vehicle_type_code2) LIKE '%scoot%'
      )
    GROUP BY vehicle_type_code2
    ORDER BY count DESC
    LIMIT 50
""").fetchdf()
print(result.to_string(index=False))

print("\n" + "="*70)
print("CYCLIST INJURY/DEATH STATISTICS")
print("="*70)
result = con.execute(f"""
    SELECT
        COUNT(*) as total_cyclist_crashes,
        SUM(CASE WHEN number_of_cyclist_injured > 0 THEN 1 ELSE 0 END) as with_injuries,
        SUM(CASE WHEN number_of_cyclist_killed > 0 THEN 1 ELSE 0 END) as with_deaths,
        SUM(number_of_cyclist_injured) as total_injured,
        SUM(number_of_cyclist_killed) as total_killed
    FROM read_parquet('{CRASH_FILE.as_posix()}')
    WHERE COALESCE(number_of_cyclist_injured, 0) > 0
       OR COALESCE(number_of_cyclist_killed, 0) > 0
""").fetchdf()
print(result.to_string(index=False))

print("\n" + "="*70)
print("TOTAL BIKE CRASHES - Different filter combinations")
print("="*70)

# Method 1: vehicle_type_code = 'BICYCLE' exact
count1 = con.execute(f"""
    SELECT COUNT(*) as count
    FROM read_parquet('{CRASH_FILE.as_posix()}')
    WHERE vehicle_type_code1 = 'BICYCLE' OR vehicle_type_code2 = 'BICYCLE'
""").fetchone()[0]
print(f"Method 1 (exact 'BICYCLE'):                      {count1:,}")

# Method 2: cyclist injured/killed
count2 = con.execute(f"""
    SELECT COUNT(*) as count
    FROM read_parquet('{CRASH_FILE.as_posix()}')
    WHERE COALESCE(number_of_cyclist_injured, 0) > 0
       OR COALESCE(number_of_cyclist_killed, 0) > 0
""").fetchone()[0]
print(f"Method 2 (cyclist injured/killed):               {count2:,}")

# Method 3: LIKE '%bike%' but NOT scooter
count3 = con.execute(f"""
    SELECT COUNT(*) as count
    FROM read_parquet('{CRASH_FILE.as_posix()}')
    WHERE (
        (LOWER(COALESCE(vehicle_type_code1, '')) LIKE '%bike%'
         OR LOWER(COALESCE(vehicle_type_code1, '')) LIKE '%bicy%')
        AND LOWER(COALESCE(vehicle_type_code1, '')) NOT LIKE '%scoot%'
    ) OR (
        (LOWER(COALESCE(vehicle_type_code2, '')) LIKE '%bike%'
         OR LOWER(COALESCE(vehicle_type_code2, '')) LIKE '%bicy%')
        AND LOWER(COALESCE(vehicle_type_code2, '')) NOT LIKE '%scoot%'
    )
""").fetchone()[0]
print(f"Method 3 (LIKE '%bike%' NOT scooter):            {count3:,}")

# Method 4: Combined (vehicle OR cyclist injury)
count4 = con.execute(f"""
    SELECT COUNT(*) as count
    FROM read_parquet('{CRASH_FILE.as_posix()}')
    WHERE (
        (LOWER(COALESCE(vehicle_type_code1, '')) LIKE '%bike%'
         OR LOWER(COALESCE(vehicle_type_code1, '')) LIKE '%bicy%')
        AND LOWER(COALESCE(vehicle_type_code1, '')) NOT LIKE '%scoot%'
    ) OR (
        (LOWER(COALESCE(vehicle_type_code2, '')) LIKE '%bike%'
         OR LOWER(COALESCE(vehicle_type_code2, '')) LIKE '%bicy%')
        AND LOWER(COALESCE(vehicle_type_code2, '')) NOT LIKE '%scoot%'
    )
    OR COALESCE(number_of_cyclist_injured, 0) > 0
    OR COALESCE(number_of_cyclist_killed, 0) > 0
""").fetchone()[0]
print(f"Method 4 (vehicle OR cyclist injury - COMBINED): {count4:,}")

print("\n" + "="*70)
