"""
meteorology_forecast.py
-----------------------
Importable prediction helpers for the Level 5 web interface.

Exposes:
  - get_meteorology_options(dataset_path)
  - get_cached_metrics()
  - predict_meteorology_for_location(dataset_path, location, target_date)

Model caching:
  Trained models are saved to  Level_5_Meteorology_Forecasting/models_cache/
  as pickle files so repeated predictions are near-instant.
"""

import os
import sys
import pickle
import time
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_DIR   = os.path.join(BASE_DIR, "models_cache")
REPORT_TXT  = os.path.join(RESULTS_DIR, "main_metrics_report.txt")
RESULTS_CSV = os.path.join(RESULTS_DIR, "walk_forward_results.csv")
FEATURES_PARQUET = os.path.join(DATA_DIR, "features.parquet")

os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Target metadata (must match models_04.py)
# ---------------------------------------------------------------------------
TARGETS = [
    "target_temperature_2m",
    "target_rain",
]

TARGET_LABELS = {
    "target_temperature_2m": "Temperature 2m",
    "target_rain":           "Rain",
}

TARGET_UNITS = {
    "target_temperature_2m": "°C",
    "target_rain":           "mm",
}

CACHE_FILES = {
    "target_temperature_2m": os.path.join(CACHE_DIR, "lgbm_temperature_2m.pkl"),
    "target_rain":           os.path.join(CACHE_DIR, "xgb_rain.pkl"),
}

# Non-feature columns (must be excluded from X)
NON_FEATURE_COLS = set(TARGETS) | {"time", "location"}


def _get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


# ---------------------------------------------------------------------------
# option helpers
# ---------------------------------------------------------------------------

def get_meteorology_options(dataset_path: str) -> dict:
    """
    Read the raw CSV and return metadata for the UI dropdown / date picker.
    """
    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip()
    df["time"] = pd.to_datetime(df["time"].str.strip())
    df["location"] = df["location"].str.strip()

    locations = sorted(df["location"].unique().tolist())
    min_date  = df["time"].min().strftime("%Y-%m-%d")
    max_date  = df["time"].max().strftime("%Y-%m-%d")

    cached = {t: os.path.exists(CACHE_FILES[t]) for t in TARGETS}

    return {
        "locations":    locations,
        "minDate":      min_date,
        "maxDate":      max_date,
        "targets":      [
            {
                "key":   t,
                "label": TARGET_LABELS[t],
                "unit":  TARGET_UNITS[t],
                "cached": cached[t],
            }
            for t in TARGETS
        ],
        "allCached": all(cached.values()),
    }


# ---------------------------------------------------------------------------
# Cached metrics (from walk-forward evaluation report)
# ---------------------------------------------------------------------------

def get_cached_metrics() -> dict:
    """
    Load the pre-computed evaluation results from the pipeline run.
    Returns the raw report text + structured MAE values if available.
    """
    result = {"reportAvailable": False, "perTarget": {}, "globalMAE": None, "score": None}

    if os.path.exists(REPORT_TXT):
        with open(REPORT_TXT, "r", encoding="utf-8") as f:
            result["reportText"] = f.read()
        result["reportAvailable"] = True

    if os.path.exists(RESULTS_CSV):
        try:
            df = pd.read_csv(RESULTS_CSV)
            from sklearn.metrics import mean_absolute_error
            per_target = {}
            for t in TARGETS:
                actual_col = f"actual_{t}"
                pred_col   = f"pred_{t}"
                if actual_col in df.columns and pred_col in df.columns:
                    mae = float(mean_absolute_error(df[actual_col].dropna(), df[pred_col].dropna()))
                    per_target[t] = {
                        "label": TARGET_LABELS[t],
                        "unit":  TARGET_UNITS[t],
                        "mae":   mae,
                    }
            if per_target:
                global_mae = float(np.mean([v["mae"] for v in per_target.values()]))
                score = (2.5 / (1 + global_mae)) * (len(TARGETS) / 17) * 100
                result["perTarget"]  = per_target
                result["globalMAE"]  = global_mae
                result["score"]      = score
                result["nTargets"]   = len(TARGETS)
                result["totalLabels"] = 17

            # Step-level MAE for the bar chart (per target per step)
            # Also compute the week-start date for each step so the UI can label them.
            step_dates = {}
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
                step_dates = (
                    df.groupby("step")["time"].min()
                    .dt.strftime("%d %b")
                    .to_dict()
                )

            step_data = {}
            for t in TARGETS:
                actual_col = f"actual_{t}"
                pred_col   = f"pred_{t}"
                if actual_col in df.columns and pred_col in df.columns:
                    from sklearn.metrics import mean_absolute_error as mae_fn
                    grp = df.groupby("step").apply(
                        lambda g, ac=actual_col, pc=pred_col: float(mae_fn(g[ac], g[pc])),
                        include_groups=False,
                    ).reset_index(name="mae")
                    grp["date"] = grp["step"].map(step_dates)
                    step_data[t] = grp.to_dict(orient="records")
            result["stepMAE"] = step_data


        except Exception as exc:
            result["csvError"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Model training + caching
# ---------------------------------------------------------------------------

def _load_or_train(target: str, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Returns (model, from_cache: bool).
    If a cached .pkl exists → load it; otherwise train and save.
    """
    cache_path = CACHE_FILES[target]

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                model = pickle.load(f)
            return model, True
        except Exception:
            pass  # corrupt cache → retrain

    # Import model factory from models_04 (sibling of this file)
    sys.path.insert(0, BASE_DIR)
    from models_04 import get_model

    model = get_model(target)
    model.fit(X_train, y_train)

    with open(cache_path, "wb") as f:
        pickle.dump(model, f)

    return model, False


# ---------------------------------------------------------------------------
# Main prediction entry point
# ---------------------------------------------------------------------------

def predict_meteorology_for_location(
    dataset_path: str,
    location: str,
    target_date: str,
) -> dict:
    """
    Predicts next-day meteorological values for `target_date` at `location`.

    Steps:
      1. Preprocess + feature-engineer if features.parquet is missing.
      2. Load/train one model per target (cached).
      3. Use the last known hour of `target_date` in the feature set as input.
      4. Return predictions + competition score.
    """
    t0 = time.time()

    # -----------------------------------------------------------------------
    # 1. Ensure features exist
    # -----------------------------------------------------------------------
    if not os.path.exists(FEATURES_PARQUET):
        import subprocess
        for script in ["01_preprocess.py", "02_feature_engineering.py"]:
            subprocess.run(
                [sys.executable, os.path.join(BASE_DIR, script)],
                check=True,
            )

    df = pd.read_parquet(FEATURES_PARQUET)
    df = df.sort_values(["time", "location"]).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # 2. Filter to the requested location
    # -----------------------------------------------------------------------
    loc_df = df[df["location"] == location].copy()
    if len(loc_df) == 0:
        raise ValueError(f"Location '{location}' not found in features dataset.")

    # -----------------------------------------------------------------------
    # 3. Build train set: all rows up to (and including) target_date
    # -----------------------------------------------------------------------
    target_dt = pd.to_datetime(target_date)
    train_df  = loc_df[loc_df["time"] <= target_dt].dropna(subset=TARGETS)

    if len(train_df) == 0:
        raise ValueError(
            f"No training data available before {target_date} for location '{location}'."
        )

    feature_cols = _get_feature_cols(df)
    X_train = train_df[feature_cols]

    # -----------------------------------------------------------------------
    # 4. Input row: the last hour of target_date (features available at t)
    # -----------------------------------------------------------------------
    day_rows = loc_df[loc_df["time"].dt.date == target_dt.date()]
    if len(day_rows) == 0:
        # Fall back to last available hour before end of day
        day_rows = loc_df[loc_df["time"] <= target_dt + pd.Timedelta(hours=23)]

    if len(day_rows) == 0:
        raise ValueError(f"No data found for date {target_date} at location '{location}'.")

    input_row = day_rows.iloc[[-1]]  # last hour of the day
    X_input   = input_row[feature_cols]

    # -----------------------------------------------------------------------
    # 5. Train / load cached model; predict
    # -----------------------------------------------------------------------
    predictions = []
    from_cache_all = True
    cache_status = {}

    for target in TARGETS:
        y_train = train_df[target]
        model, from_cache = _load_or_train(target, X_train, y_train)
        cache_status[target] = from_cache
        if not from_cache:
            from_cache_all = False

        pred = float(model.predict(X_input)[0])

        # Rain: clip negatives + threshold
        if target == "target_rain":
            pred = max(0.0, pred)
            if pred < 0.1:
                pred = 0.0

        # Actual value for the next day (target column = shift(-24h) of the raw var)
        actual_val = input_row[target].iloc[0]
        actual = None if pd.isna(actual_val) else round(float(actual_val), 4)

        predictions.append({
            "target":    target,
            "label":     TARGET_LABELS[target],
            "unit":      TARGET_UNITS[target],
            "predicted": round(pred, 4),
            "actual":    actual,
        })

    # -----------------------------------------------------------------------
    # 6. Competition score (based on cached walk-forward MAE if available)
    # -----------------------------------------------------------------------
    metrics = get_cached_metrics()
    global_mae = metrics.get("globalMAE")
    score      = metrics.get("score")
    per_target_mae = {
        t: metrics["perTarget"].get(t, {}).get("mae")
        for t in TARGETS
    } if metrics.get("perTarget") else {}

    duration_ms = int((time.time() - t0) * 1000)

    return {
        "location":    location,
        "targetDate":  target_date,
        "predictions": predictions,
        "globalMAE":   global_mae,
        "score":       score,
        "perTargetMAE": per_target_mae,
        "nTargets":    len(TARGETS),
        "totalLabels": 17,
        "fromCache":   from_cache_all,
        "cacheStatus": cache_status,
        "trainingSamples": len(train_df),
        "inputTime":   str(input_row["time"].iloc[0]),
        "durationMs":  duration_ms,
    }
