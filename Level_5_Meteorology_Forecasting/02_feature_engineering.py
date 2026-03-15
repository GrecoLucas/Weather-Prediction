"""
02_feature_engineering.py
--------------------------
Reads data/processed.parquet and builds:
  - Target columns (shift -24h per location group)
  - Lag features (shift 0, 1, 24, 48, 72 per location group)
  - Rolling mean features (no leakage: computed over data ≤ t)
  - Interaction features
  - Encodes location as integer

Saves the result to data/features.parquet.

Run: python 02_feature_engineering.py
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
IN_FILE   = os.path.join(BASE_DIR, "data", "processed.parquet")
OUT_FILE  = os.path.join(BASE_DIR, "data", "features.parquet")

# ---------------------------------------------------------------------------
# Target raw column names → engineered target column names
# Fast mode: predict only 2 targets to reduce training cost.
# ---------------------------------------------------------------------------
RAW_TO_TARGET = {
    "temperature_2m"  : "target_temperature_2m",
    "rain"            : "target_rain",
}

# Columns to create lags for
LAG_COLS = [
    "temperature_2m",
    "rain",
    "relative_humidity_2m",
    "cloud_cover",
    "wind_speed_10m",
    "pressure_msl",
]

# Shifts to apply (positive = looking into the past)
# shift(0)  → current value at t  (24 h before the target at t+24)
# shift(1)  → 1 h ago             (25 h before the target)
# shift(24) → 24 h ago            (48 h before the target = yesterday same hour)
# shift(48) → 48 h ago
# shift(72) → 72 h ago
LAG_SHIFTS = [0, 1, 24, 72]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    All transformations are applied per location group to prevent
    lags from bleeding across city boundaries.
    """
    groups = []

    for loc, grp in df.groupby("location", sort=False):
        grp = grp.copy().sort_values("time").reset_index(drop=True)

        # ------------------------------------------------------------------
        # 1. Target columns (shift -24: value 24 h into the future)
        # ------------------------------------------------------------------
        for raw_col, tgt_col in RAW_TO_TARGET.items():
            grp[tgt_col] = grp[raw_col].shift(-24)

        # ------------------------------------------------------------------
        # 2. Lag features (positive shift = past data, no leakage)
        # ------------------------------------------------------------------
        for col in LAG_COLS:
            if col not in grp.columns:
                continue
            for s in LAG_SHIFTS:
                suffix = f"lag_{s}h"
                grp[f"{col}_{suffix}"] = grp[col].shift(s)

        # ------------------------------------------------------------------
        # 3. Rolling mean features (computed over data ≤ t — no leakage)
        #    .rolling(n).mean() at position i uses rows [i-n+1 … i], so
        #    it only touches past + current values, never the future.
        # ------------------------------------------------------------------

        # 3h window: fast-changing variables
        for col in ["cloud_cover", "relative_humidity_2m"]:
            grp[f"{col}_roll3h"] = grp[col].rolling(3, min_periods=1).mean()

        # 6h window: temperature
        for col in ["temperature_2m"]:
            grp[f"{col}_roll6h"] = grp[col].rolling(6, min_periods=1).mean()

        # 24h window: pressure
        for col in ["pressure_msl"]:
            grp[f"{col}_roll24h"] = grp[col].rolling(24, min_periods=1).mean()

        # 72h window: rain (smooths the heavy-zero distribution)
        grp["rain_roll72h"] = grp["rain"].rolling(72, min_periods=1).mean()

        # ------------------------------------------------------------------
        # 4. Interaction features
        # ------------------------------------------------------------------
        grp["temp_dew_spread"]   = grp["temperature_2m"] - grp["dew_point_2m"]
        grp["pressure_diff"]     = grp["pressure_msl"] - grp["surface_pressure"]
        grp["wind_speed_diff"]   = grp["wind_speed_100m"] - grp["wind_speed_10m"]

        groups.append(grp)

    result = pd.concat(groups, ignore_index=True)

    # ------------------------------------------------------------------
    # 5. Encode location as integer category
    # ------------------------------------------------------------------
    result["location_id"] = result["location"].astype("category").cat.codes

    # ------------------------------------------------------------------
    # 6. Drop rows where ANY target is NaN
    #    (first rows lose lag data; last 24 rows lose target from shift(-24))
    # ------------------------------------------------------------------
    target_cols = list(RAW_TO_TARGET.values())
    before = len(result)
    result = result.dropna(subset=target_cols).reset_index(drop=True)
    after  = len(result)
    print(f"  Dropped {before - after:,} rows with NaN targets/lags → {after:,} rows remain")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading processed data …")
    df = pd.read_parquet(IN_FILE)
    print(f"  Shape: {df.shape}")

    print("Building features …")
    features_df = build_features(df)

    print(f"\nFinal feature set shape: {features_df.shape}")
    print(f"Columns ({len(features_df.columns)}):")
    for c in features_df.columns:
        print(f"  {c}")

    features_df.to_parquet(OUT_FILE, index=False)
    print(f"\nSaved → {OUT_FILE}")