import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


DEFAULT_UNDER_PENALTY = 0.6
MAX_GLOBAL_UPLIFT = 2.5
MAX_REGION_UPLIFT = 2.0

# Lightweight in-memory cache for merged daily data keyed by file mtimes.
_DATA_CACHE: Dict[str, pd.DataFrame] = {}
_STATIC_VE_CACHE: Dict[str, float] = {}
_NOTEBOOK_SUMMARY_CACHE: Dict[str, dict] = {}


def _file_hash(path: str) -> str:
    stat = os.stat(path)
    return f"{path}:{stat.st_mtime:.0f}:{stat.st_size}"


def _safe_under_penalty(value: float) -> float:
    try:
        q = float(value)
    except Exception:
        q = DEFAULT_UNDER_PENALTY
    if not (0.0 < q < 1.0):
        q = DEFAULT_UNDER_PENALTY
    return q


def _learn_uplift(y_true: np.ndarray, y_pred: np.ndarray, q: float, cap: float) -> float:
    residuals = y_true - y_pred
    pos_residuals = residuals[residuals > 0]
    if len(pos_residuals) == 0:
        return 0.0
    return float(min(np.quantile(pos_residuals, q), cap))


def _load_merged_daily(meteorology_path: str, accidents_path: str) -> pd.DataFrame:
    cache_key = f"{_file_hash(meteorology_path)}|{_file_hash(accidents_path)}"
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key].copy()

    accidents_df = pd.read_csv(accidents_path)
    met_df = pd.read_csv(meteorology_path)

    accidents_df.columns = accidents_df.columns.str.strip()
    met_df.columns = met_df.columns.str.strip()

    accidents_df["time"] = pd.to_datetime(accidents_df["time"], errors="coerce")
    met_df["time"] = pd.to_datetime(met_df["time"], errors="coerce")
    accidents_df = accidents_df.dropna(subset=["time", "location"]).copy()
    met_df = met_df.dropna(subset=["time", "location"]).copy()

    if "accidents" in accidents_df.columns:
        accidents_df = accidents_df.rename(columns={"accidents": "accident_count"})
    if "accident_count" not in accidents_df.columns:
        raise ValueError("Accidents dataset must contain 'accidents' or 'accident_count' column.")

    cutoff_date = accidents_df["time"].min()
    met_df = met_df[met_df["time"] >= cutoff_date].copy()

    accidents_df["date"] = accidents_df["time"].dt.floor("D")
    met_df["date"] = met_df["time"].dt.floor("D")

    agg_ops = {
        "temperature_2m": "mean",
        "relative_humidity_2m": "mean",
        "dew_point_2m": "mean",
        "rain": "sum",
        "cloud_cover": "mean",
        "cloud_cover_low": "mean",
        "cloud_cover_mid": "mean",
        "cloud_cover_highh": "mean",
        "wind_speed_10m": "mean",
        "wind_direction_10m": "mean",
        "wind_gusts_10m": "mean",
        "wind_direction_100m": "mean",
        "wind_speed_100m": "mean",
        "pressure_msl": "mean",
        "surface_pressure": "mean",
    }

    available_agg_ops = {k: v for k, v in agg_ops.items() if k in met_df.columns}
    if "snow_fall" in met_df.columns:
        available_agg_ops["snow_fall"] = "max"

    if len(available_agg_ops) == 0:
        raise ValueError("No valid meteorology feature columns were found for Level 4 aggregation.")

    met_daily = met_df.groupby(["location", "date"]).agg(available_agg_ops).reset_index()

    merged = pd.merge(
        accidents_df[["location", "date", "accident_count"]],
        met_daily,
        on=["location", "date"],
        how="inner",
    )

    if "snow_fall" not in merged.columns:
        merged["snow_fall"] = 0.0
    merged["snow_fall"] = merged["snow_fall"].fillna(0.0)

    merged = merged.sort_values(["date", "location"]).reset_index(drop=True)
    _DATA_CACHE[cache_key] = merged.copy()
    return merged


def _build_features(merged_df: pd.DataFrame):
    model_df = merged_df.copy().sort_values(["date", "location"]).reset_index(drop=True)
    model_df["day_of_week"] = model_df["date"].dt.dayofweek
    model_df["month"] = model_df["date"].dt.month
    model_df["is_weekend"] = (model_df["day_of_week"] >= 5).astype(int)
    model_df["has_snow"] = (model_df["snow_fall"] > 0).astype(int)

    exclude_cols = {"accident_count", "date", "location", "snow_fall"}
    weather_cols = [
        c
        for c in model_df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols and c not in {"day_of_week", "month", "is_weekend", "has_snow"}
    ]

    core_cols = [
        "temperature_2m",
        "relative_humidity_2m",
        "rain",
        "wind_speed_10m",
        "wind_gusts_10m",
        "pressure_msl",
        "surface_pressure",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_highh",
    ]
    core_cols = [c for c in core_cols if c in model_df.columns]

    weather_cols = sorted(set(weather_cols) | set(core_cols))
    base_features = weather_cols + ["has_snow", "day_of_week", "month", "is_weekend"]

    model_df_global = pd.get_dummies(
        model_df[["location", "date", "accident_count"] + base_features],
        columns=["location"],
        drop_first=False,
        dtype=float,
    )

    global_features = [c for c in model_df_global.columns if c not in ["date", "accident_count"]]
    regional_features = base_features.copy()

    return model_df, model_df_global, global_features, regional_features


def _compute_static_mean_vehicle_error_all_days(merged_df: pd.DataFrame) -> float:
    """Compute a static benchmark: mean daily Vehicle Error across all available days.

    Method: chronological day-by-day backtest using the global no-uplift model,
    then average the per-day mean Vehicle Error.
    """
    model_df, model_df_global, global_features, _ = _build_features(merged_df)
    unique_dates = np.array(sorted(model_df["date"].unique()))

    day_mean_errors: List[float] = []
    for selected_dt in unique_dates:
        train_global = model_df_global[model_df_global["date"] < selected_dt].copy()
        test_global = model_df_global[model_df_global["date"] == selected_dt].copy()

        if len(train_global) < 30 or len(test_global) == 0:
            continue

        model = LinearRegression()
        model.fit(train_global[global_features], train_global["accident_count"])
        pred_acc = np.maximum(0, model.predict(test_global[global_features]))

        actual_veh = 3 * test_global["accident_count"].to_numpy()
        pred_veh = 3 * pred_acc
        day_mean_errors.append(float(np.mean(np.abs(actual_veh - pred_veh))))

    return float(np.mean(day_mean_errors)) if day_mean_errors else float("nan")


def _compute_notebook_equivalent_summary(merged_df: pd.DataFrame, under_penalty: float) -> dict:
    """Reproduce notebook final summary using the same chronological test-window split."""
    model_df, model_df_global, global_features, regional_features = _build_features(merged_df)
    unique_dates = np.array(sorted(model_df["date"].unique()))
    if len(unique_dates) < 12:
        return {
            "rows": [],
            "splitDate": None,
            "testStartDate": None,
            "testEndDate": None,
            "testDays": 0,
        }

    test_days = max(10, int(round(0.15 * len(unique_dates))))
    split_date = unique_dates[-test_days]

    train_df = model_df[model_df["date"] < split_date].copy()
    test_df = model_df[model_df["date"] >= split_date].copy()
    train_global = model_df_global[model_df_global["date"] < split_date].copy()
    test_global = model_df_global[model_df_global["date"] >= split_date].copy()

    if len(train_df) < 30 or len(test_df) == 0:
        return {
            "rows": [],
            "splitDate": str(pd.to_datetime(split_date).date()),
            "testStartDate": None,
            "testEndDate": None,
            "testDays": 0,
        }

    q = _safe_under_penalty(under_penalty)

    global_model = LinearRegression()
    X_train_g = train_global[global_features]
    y_train_g = train_global["accident_count"]
    X_test_g = test_global[global_features]

    train_unique_dates = np.array(sorted(train_global["date"].unique()))
    calib_days = max(7, int(round(0.2 * len(train_unique_dates))))
    calib_start = train_unique_dates[-calib_days]

    inner_train_global = train_global[train_global["date"] < calib_start]
    calib_global = train_global[train_global["date"] >= calib_start]

    if len(inner_train_global) > 0 and len(calib_global) > 0:
        calib_global_model = LinearRegression()
        calib_global_model.fit(inner_train_global[global_features], inner_train_global["accident_count"])
        calib_pred = np.maximum(0, calib_global_model.predict(calib_global[global_features]))
        global_uplift = _learn_uplift(
            calib_global["accident_count"].to_numpy(),
            calib_pred,
            q,
            MAX_GLOBAL_UPLIFT,
        )
    else:
        global_uplift = 0.0

    global_model.fit(X_train_g, y_train_g)
    global_pred_raw = np.maximum(0, global_model.predict(X_test_g))
    global_pred = global_pred_raw + global_uplift

    regional_preds: List[pd.DataFrame] = []
    for loc in sorted(test_df["location"].unique()):
        loc_train = train_df[train_df["location"] == loc].copy()
        loc_test = test_df[test_df["location"] == loc].copy()
        if len(loc_test) == 0:
            continue

        if len(loc_train) < 8:
            base_mean = loc_train["accident_count"].mean() if len(loc_train) > 0 else train_df["accident_count"].mean()
            pred_raw = np.repeat(base_mean, len(loc_test))
            uplift_r = 0.0
        else:
            region_dates = np.array(sorted(loc_train["date"].unique()))
            region_calib_days = max(3, int(round(0.2 * len(region_dates))))
            region_calib_start = region_dates[-region_calib_days]

            loc_inner_train = loc_train[loc_train["date"] < region_calib_start]
            loc_calib = loc_train[loc_train["date"] >= region_calib_start]

            model_r = LinearRegression()
            if len(loc_inner_train) > 0 and len(loc_calib) > 0:
                model_r.fit(loc_inner_train[regional_features], loc_inner_train["accident_count"])
                calib_pred_r = np.maximum(0, model_r.predict(loc_calib[regional_features]))
                uplift_r = _learn_uplift(
                    loc_calib["accident_count"].to_numpy(),
                    calib_pred_r,
                    q,
                    MAX_REGION_UPLIFT,
                )
            else:
                uplift_r = 0.0

            model_r.fit(loc_train[regional_features], loc_train["accident_count"])
            pred_raw = np.maximum(0, model_r.predict(loc_test[regional_features]))

        loc_test = loc_test.copy()
        loc_test["pred_regional_raw"] = pred_raw
        loc_test["pred_regional"] = pred_raw + uplift_r
        regional_preds.append(loc_test[["location", "date", "pred_regional_raw", "pred_regional"]])

    if not regional_preds:
        return {
            "rows": [],
            "splitDate": str(pd.to_datetime(split_date).date()),
            "testStartDate": str(test_df["date"].min().date()),
            "testEndDate": str(test_df["date"].max().date()),
            "testDays": int(test_df["date"].nunique()),
        }

    regional_pred_df = pd.concat(regional_preds, ignore_index=True)

    eval_df = test_df[["location", "date", "accident_count"]].copy().reset_index(drop=True)
    eval_df["pred_global_raw"] = global_pred_raw
    eval_df["pred_global"] = global_pred
    eval_df = eval_df.merge(regional_pred_df, on=["location", "date"], how="left")

    summary = _summarize_rows(eval_df).round(4)
    return {
        "rows": summary.to_dict(orient="records"),
        "splitDate": str(pd.to_datetime(split_date).date()),
        "testStartDate": str(test_df["date"].min().date()),
        "testEndDate": str(test_df["date"].max().date()),
        "testDays": int(test_df["date"].nunique()),
    }


def get_level4_options(meteorology_path: str, accidents_path: str) -> dict:
    merged_df = _load_merged_daily(meteorology_path, accidents_path)
    locations = sorted(merged_df["location"].astype(str).unique().tolist())
    valid_dates = []
    for raw_date in merged_df["date"].tolist():
        ts = pd.to_datetime(raw_date, errors="coerce")
        if not pd.isna(ts):
            valid_dates.append(pd.Timestamp(ts).date())
    dates = sorted(set(valid_dates))

    static_key = f"{_file_hash(meteorology_path)}|{_file_hash(accidents_path)}"
    if static_key not in _STATIC_VE_CACHE:
        _STATIC_VE_CACHE[static_key] = _compute_static_mean_vehicle_error_all_days(merged_df)
    static_mean_ve = _STATIC_VE_CACHE[static_key]

    notebook_key = f"{static_key}|up:{DEFAULT_UNDER_PENALTY:.4f}"
    if notebook_key not in _NOTEBOOK_SUMMARY_CACHE:
        _NOTEBOOK_SUMMARY_CACHE[notebook_key] = _compute_notebook_equivalent_summary(
            merged_df,
            under_penalty=DEFAULT_UNDER_PENALTY,
        )
    notebook_summary = _NOTEBOOK_SUMMARY_CACHE[notebook_key]

    return {
        "locations": locations,
        "minDate": str(dates[0]) if dates else None,
        "maxDate": str(dates[-1]) if dates else None,
        "staticMeanVehicleErrorAllDays": static_mean_ve,
        "notebookEquivalent": notebook_summary,
    }


def _summarize_rows(eval_df: pd.DataFrame) -> pd.DataFrame:
    actual_veh = 3 * eval_df["accident_count"]
    global_raw_veh = 3 * eval_df["pred_global_raw"]
    global_adj_veh = 3 * eval_df["pred_global"]
    regional_raw_veh = 3 * eval_df["pred_regional_raw"]
    regional_adj_veh = 3 * eval_df["pred_regional"]

    summary = pd.DataFrame(
        [
            {
                "Model": "Global",
                "Mode": "No uplift",
                "Total Vehicle Error": (actual_veh - global_raw_veh).abs().sum(),
                "Mean Vehicle Error": (actual_veh - global_raw_veh).abs().mean(),
                "Median Vehicle Error": (actual_veh - global_raw_veh).abs().median(),
                "P90 Vehicle Error": (actual_veh - global_raw_veh).abs().quantile(0.90),
                "Underestimation Rate": (eval_df["pred_global_raw"] < eval_df["accident_count"]).mean(),
                "Total Pred Vehicles": global_raw_veh.sum(),
            },
            {
                "Model": "Global",
                "Mode": "With uplift",
                "Total Vehicle Error": (actual_veh - global_adj_veh).abs().sum(),
                "Mean Vehicle Error": (actual_veh - global_adj_veh).abs().mean(),
                "Median Vehicle Error": (actual_veh - global_adj_veh).abs().median(),
                "P90 Vehicle Error": (actual_veh - global_adj_veh).abs().quantile(0.90),
                "Underestimation Rate": (eval_df["pred_global"] < eval_df["accident_count"]).mean(),
                "Total Pred Vehicles": global_adj_veh.sum(),
            },
            {
                "Model": "Regional",
                "Mode": "No uplift",
                "Total Vehicle Error": (actual_veh - regional_raw_veh).abs().sum(),
                "Mean Vehicle Error": (actual_veh - regional_raw_veh).abs().mean(),
                "Median Vehicle Error": (actual_veh - regional_raw_veh).abs().median(),
                "P90 Vehicle Error": (actual_veh - regional_raw_veh).abs().quantile(0.90),
                "Underestimation Rate": (eval_df["pred_regional_raw"] < eval_df["accident_count"]).mean(),
                "Total Pred Vehicles": regional_raw_veh.sum(),
            },
            {
                "Model": "Regional",
                "Mode": "With uplift",
                "Total Vehicle Error": (actual_veh - regional_adj_veh).abs().sum(),
                "Mean Vehicle Error": (actual_veh - regional_adj_veh).abs().mean(),
                "Median Vehicle Error": (actual_veh - regional_adj_veh).abs().median(),
                "P90 Vehicle Error": (actual_veh - regional_adj_veh).abs().quantile(0.90),
                "Underestimation Rate": (eval_df["pred_regional"] < eval_df["accident_count"]).mean(),
                "Total Pred Vehicles": regional_adj_veh.sum(),
            },
        ]
    )

    total_actual_vehicles = float(actual_veh.sum())
    summary["Total Actual Vehicles"] = total_actual_vehicles
    summary["Vehicle Bias (Pred-Actual)"] = summary["Total Pred Vehicles"] - summary["Total Actual Vehicles"]
    return summary


def predict_accidents_for_day(
    meteorology_path: str,
    accidents_path: str,
    selected_date: str,
    location: Optional[str] = None,
    under_penalty: float = DEFAULT_UNDER_PENALTY,
) -> dict:
    t0 = time.time()

    merged_df = _load_merged_daily(meteorology_path, accidents_path)
    model_df, model_df_global, global_features, regional_features = _build_features(merged_df)

    selected_dt = pd.to_datetime(selected_date).floor("D")
    train_df = model_df[model_df["date"] < selected_dt].copy()
    test_df = model_df[model_df["date"] == selected_dt].copy()

    if location:
        test_df = test_df[test_df["location"].astype(str) == str(location)].copy()

    if len(test_df) == 0:
        available_dates = sorted(model_df["date"].dt.strftime("%Y-%m-%d").unique().tolist())
        raise ValueError(
            f"No rows found for selectedDate={selected_dt.date()} and location={location or 'ALL'}. "
            f"Available dates include: {available_dates[:3]} ... {available_dates[-3:]}"
        )

    if len(train_df) < 30:
        raise ValueError("Not enough historical rows before selectedDate to train Level 4 models.")

    q = _safe_under_penalty(under_penalty)

    # Global training/test frame
    train_global = model_df_global[model_df_global["date"] < selected_dt].copy()
    test_global = model_df_global[model_df_global["date"] == selected_dt].copy()
    if location:
        location_col = f"location_{location}"
        if location_col in test_global.columns:
            test_global = test_global[test_global[location_col] == 1.0].copy()

    global_model = LinearRegression()
    X_train_g = train_global[global_features]
    y_train_g = train_global["accident_count"]
    X_test_g = test_global[global_features]

    train_unique_dates = np.array(sorted(train_global["date"].unique()))
    calib_days = max(7, int(round(0.2 * len(train_unique_dates))))
    calib_start = train_unique_dates[-calib_days]

    inner_train_global = train_global[train_global["date"] < calib_start]
    calib_global = train_global[train_global["date"] >= calib_start]

    if len(inner_train_global) > 0 and len(calib_global) > 0:
        calib_global_model = LinearRegression()
        calib_global_model.fit(inner_train_global[global_features], inner_train_global["accident_count"])
        calib_pred = np.maximum(0, calib_global_model.predict(calib_global[global_features]))
        global_uplift = _learn_uplift(
            calib_global["accident_count"].to_numpy(),
            calib_pred,
            q,
            MAX_GLOBAL_UPLIFT,
        )
    else:
        global_uplift = 0.0

    global_model.fit(X_train_g, y_train_g)
    global_pred_raw = np.maximum(0, global_model.predict(X_test_g))
    global_pred = global_pred_raw + global_uplift

    # Regional models
    regional_preds: List[pd.DataFrame] = []
    for loc in sorted(test_df["location"].unique()):
        loc_train = train_df[train_df["location"] == loc].copy()
        loc_test = test_df[test_df["location"] == loc].copy()

        if len(loc_train) < 8:
            base_mean = loc_train["accident_count"].mean() if len(loc_train) > 0 else train_df["accident_count"].mean()
            pred_raw = np.repeat(base_mean, len(loc_test))
            uplift_r = 0.0
        else:
            region_dates = np.array(sorted(loc_train["date"].unique()))
            region_calib_days = max(3, int(round(0.2 * len(region_dates))))
            region_calib_start = region_dates[-region_calib_days]

            loc_inner_train = loc_train[loc_train["date"] < region_calib_start]
            loc_calib = loc_train[loc_train["date"] >= region_calib_start]

            model_r = LinearRegression()
            if len(loc_inner_train) > 0 and len(loc_calib) > 0:
                model_r.fit(loc_inner_train[regional_features], loc_inner_train["accident_count"])
                calib_pred_r = np.maximum(0, model_r.predict(loc_calib[regional_features]))
                uplift_r = _learn_uplift(
                    loc_calib["accident_count"].to_numpy(),
                    calib_pred_r,
                    q,
                    MAX_REGION_UPLIFT,
                )
            else:
                uplift_r = 0.0

            model_r.fit(loc_train[regional_features], loc_train["accident_count"])
            pred_raw = np.maximum(0, model_r.predict(loc_test[regional_features]))

        loc_test = loc_test.copy()
        loc_test["pred_regional_raw"] = pred_raw
        loc_test["pred_regional"] = pred_raw + uplift_r
        regional_preds.append(loc_test[["location", "date", "pred_regional_raw", "pred_regional"]])

    regional_pred_df = pd.concat(regional_preds, ignore_index=True)

    eval_df = test_df[["location", "date", "accident_count"]].copy().reset_index(drop=True)
    eval_df["pred_global_raw"] = global_pred_raw
    eval_df["pred_global"] = global_pred
    eval_df = eval_df.merge(regional_pred_df, on=["location", "date"], how="left")

    summary = _summarize_rows(eval_df).round(4)
    best_row = summary.sort_values("Total Vehicle Error").iloc[0]

    rows = []
    for row in eval_df.sort_values("location").itertuples(index=False):
        rows.append(
            {
                "location": str(row.location),
                "date": str(pd.to_datetime(row.date).date()),
                "actualAccidents": float(row.accident_count),
                "actualVehicles": float(3 * row.accident_count),
                "globalNoUplift": float(row.pred_global_raw),
                "globalWithUplift": float(row.pred_global),
                "regionalNoUplift": float(row.pred_regional_raw),
                "regionalWithUplift": float(row.pred_regional),
                "globalVehicleErrorNoUplift": float(abs(3 * row.accident_count - 3 * row.pred_global_raw)),
                "globalVehicleErrorWithUplift": float(abs(3 * row.accident_count - 3 * row.pred_global)),
                "regionalVehicleErrorNoUplift": float(abs(3 * row.accident_count - 3 * row.pred_regional_raw)),
                "regionalVehicleErrorWithUplift": float(abs(3 * row.accident_count - 3 * row.pred_regional)),
            }
        )

    duration_ms = int((time.time() - t0) * 1000)

    return {
        "selectedDate": str(selected_dt.date()),
        "location": location or "All",
        "underPenalty": q,
        "globalUplift": float(global_uplift),
        "summary": summary.to_dict(orient="records"),
        "bestSetup": f"{best_row['Model']} - {best_row['Mode']}",
        "rows": rows,
        "durationMs": duration_ms,
    }
