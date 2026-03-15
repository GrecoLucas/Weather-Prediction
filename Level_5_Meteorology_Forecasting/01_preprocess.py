"""
01_preprocess.py
----------------
Load the raw meteorology CSV, clean it, encode cyclical time and wind features,
and save the result to data/processed.parquet.

Run: python 01_preprocess.py
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "data")
OUT_FILE = os.path.join(OUT_DIR, "processed.parquet")

# Try the most likely dataset locations/names used in this repository.
CSV_CANDIDATES = [
	os.path.join(BASE_DIR, "..", "data", "meteorology_dataset.csv"),
	os.path.join(BASE_DIR, "..", "meteorology_dataset.csv"),
	os.path.join(BASE_DIR, "..", "metherology_dataset.csv"),
]


def resolve_raw_csv() -> str:
	for path in CSV_CANDIDATES:
		if os.path.exists(path):
			return path
	checked = "\n  - ".join(CSV_CANDIDATES)
	raise FileNotFoundError(f"No meteorology CSV found. Checked:\n  - {checked}")


RAW_CSV = resolve_raw_csv()

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
print("Loading CSV …")
df = pd.read_csv(RAW_CSV)

# Strip whitespace from column names and string values
df.columns = df.columns.str.strip()
df["location"] = df["location"].str.strip()

print(f"  Rows: {len(df):,}   Columns: {df.columns.tolist()}")

# ---------------------------------------------------------------------------
# 2. Parse and sort by datetime
# ---------------------------------------------------------------------------
df["time"] = pd.to_datetime(df["time"].str.strip())
df = df.sort_values(["location", "time"]).reset_index(drop=True)

print(f"  Date range: {df['time'].min()} → {df['time'].max()}")
print(f"  Locations : {df['location'].unique().tolist()}")

# ---------------------------------------------------------------------------
# 3. Cyclical Time Encoding
#    sin/cos ensures 23h → 0h is a small step, not a scale jump.
#    hour_sin/cos retained: useful for humidity, cloud cover and rain
#    which exhibit strong within-day cycles.
#    day_of_year: linear position in year for seasonality.
# ---------------------------------------------------------------------------
hour      = df["time"].dt.hour
month     = df["time"].dt.month
day_of_yr = df["time"].dt.day_of_year

df["hour_sin"]    = np.sin(2 * np.pi * hour  / 24)
df["hour_cos"]    = np.cos(2 * np.pi * hour  / 24)
df["month_sin"]   = np.sin(2 * np.pi * month / 12)
df["month_cos"]   = np.cos(2 * np.pi * month / 12)
df["day_of_year"] = day_of_yr

# ---------------------------------------------------------------------------
# 4. Cyclical Wind Direction Encoding
#    359° and 1° are nearly the same direction — sin/cos resolves this.
# ---------------------------------------------------------------------------
df["wind_dir_10m_sin"]  = np.sin(np.deg2rad(df["wind_direction_10m"]))
df["wind_dir_10m_cos"]  = np.cos(np.deg2rad(df["wind_direction_10m"]))
df["wind_dir_100m_sin"] = np.sin(np.deg2rad(df["wind_direction_100m"]))
df["wind_dir_100m_cos"] = np.cos(np.deg2rad(df["wind_direction_100m"]))

# Drop raw direction columns (already encoded cyclically)
df = df.drop(columns=["wind_direction_10m", "wind_direction_100m"])

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
df.to_parquet(OUT_FILE, index=False)
print(f"\nSaved processed data → {OUT_FILE}")
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")