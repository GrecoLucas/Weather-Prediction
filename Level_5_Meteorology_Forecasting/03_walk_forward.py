"""
03_walk_forward.py
------------------
Expanding Window Walk-Forward Validation.

Strategy
--------
- Initial training window : first 3 months (~2,160 h per location)
- Step size               : 1 week (168 h)
- Each step trains one model per target and evaluates on the next week.
- Saves per-row predictions + per-step MAE to results/walk_forward_results.csv

Run: python 03_walk_forward.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_04 import get_model, TARGETS, TARGET_SHORT

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
IN_FILE     = os.path.join(BASE_DIR, "data", "features.parquet")
OUT_DIR     = os.path.join(BASE_DIR, "results")
OUT_RESULTS = os.path.join(OUT_DIR, "walk_forward_results.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# Hours -> rows conversion happens inside walk_forward() via n_locs multiplier
INITIAL_TRAIN_HOURS = 3 * 30 * 24   # ~2,160 h per location (3 months)
STEP_SIZE_HOURS     = 7 * 24        # 168 h (1 week)

NON_FEATURE_COLS = set(TARGETS) | {"time", "location"}


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def walk_forward(df: pd.DataFrame):
    """
    Expanding-window walk-forward across all locations.

    The dataframe is sorted by [time, location], so each timestamp produces
    N rows (one per location). Hour constants are multiplied by n_locs to
    get the correct row counts.
    """
    feature_cols = get_feature_cols(df)
    all_n_rows   = len(df)

    n_locs             = df["location"].nunique()
    initial_train_rows = INITIAL_TRAIN_HOURS * n_locs
    step_size_rows     = STEP_SIZE_HOURS     * n_locs

    print(f"  Locations     : {n_locs}")
    print(f"  Initial train : {initial_train_rows:,} rows  ({INITIAL_TRAIN_HOURS}h x {n_locs})")
    print(f"  Step size     : {step_size_rows:,} rows  ({STEP_SIZE_HOURS}h x {n_locs})")

    step_records = []
    pred_records = []

    train_end = initial_train_rows
    step_idx  = 0

    while train_end < all_n_rows:
        t0 = time.time()

        train_df = df.iloc[:train_end].dropna(subset=TARGETS)
        test_df  = df.iloc[train_end: train_end + step_size_rows].dropna(subset=TARGETS)

        if len(test_df) == 0:
            train_end += step_size_rows
            step_idx  += 1
            continue

        X_train = train_df[feature_cols]
        X_test  = test_df[feature_cols]

        step_mae  = {}
        step_preds = {}

        for target in TARGETS:
            y_train = train_df[target]
            y_test  = test_df[target]

            model = get_model(target)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Rain: clip negatives (Tweedie is always >= 0, but safety net)
            if target == "target_rain":
                preds = np.clip(preds, 0, None)

            mae = mean_absolute_error(y_test, preds)
            step_mae[target]   = mae
            step_preds[target] = preds

            step_records.append({
                "step"      : step_idx,
                "train_size": len(train_df),
                "test_start": test_df["time"].iloc[0] if "time" in test_df.columns else None,
                "target"    : target,
                "mae"       : mae,
            })

        for i in range(len(test_df)):
            row = {
                "step"    : step_idx,
                "time"    : test_df["time"].iloc[i] if "time" in test_df.columns else None,
                "location": test_df["location"].iloc[i] if "location" in test_df.columns else None,
            }
            for target in TARGETS:
                row[f"actual_{target}"] = test_df[target].iloc[i]
                row[f"pred_{target}"]   = step_preds[target][i]
            pred_records.append(row)

        elapsed = time.time() - t0
        mae_str = " | ".join(f"{TARGET_SHORT[t][:4]}={step_mae[t]:.3f}" for t in TARGETS)
        print(f"Step {step_idx:>3}  train={len(train_df):>6,}  test={len(test_df):>4,}  [{elapsed:.1f}s]  {mae_str}")

        train_end += step_size_rows
        step_idx  += 1

    return pd.DataFrame(step_records), pd.DataFrame(pred_records)


def compute_score(step_df: pd.DataFrame) -> tuple:
    """Competition formula: Score = 2.5/(1+MAE) x (N/17) x 100"""
    avg_mae_per_target = step_df.groupby("target")["mae"].mean()
    mae_global         = avg_mae_per_target.mean()
    n_targets          = len(avg_mae_per_target)
    score              = (2.5 / (1 + mae_global)) * (n_targets / 17) * 100
    return mae_global, score, n_targets


if __name__ == "__main__":
    print(f"Loading features from {IN_FILE} ...")
    df = pd.read_parquet(IN_FILE)
    print(f"  Shape     : {df.shape}")
    print(f"  Date range: {df['time'].min()} -> {df['time'].max()}")

    df = df.sort_values(["time", "location"]).reset_index(drop=True)

    print(f"\nWalk-Forward Config:")
    print(f"  Initial window : {INITIAL_TRAIN_HOURS:,} h (~3 months x locations)")
    print(f"  Step           : {STEP_SIZE_HOURS} h (1 week x locations)")
    print()

    step_df, pred_df = walk_forward(df)

    pred_df.to_csv(OUT_RESULTS, index=False)
    print(f"\nPredictions saved -> {OUT_RESULTS}")

    print("\n=== Per-Target Average MAE ===")
    avg_mae = step_df.groupby("target")["mae"].mean()
    for target, mae in avg_mae.items():
        print(f"  {TARGET_SHORT.get(target, target):<20}  MAE = {mae:.4f}")

    mae_global, score, n_targets = compute_score(step_df)
    print(f"\n  Targets : {n_targets} / 17")
    print(f"  MAE     : {mae_global:.4f}")
    print(f"  Score   : {score:.4f}")
    print(f"  Formula : 2.5 / (1 + {mae_global:.4f}) x ({n_targets}/17) x 100 = {score:.4f}")