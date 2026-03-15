import argparse
import os

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(MODULE_DIR, "saved_models")
TIME_COL = "time"
TEMP_COL = "temperature_2m"
DEW_COL = "dew_point_2m"
HUMIDITY_COL = "relative_humidity_2m"


def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _add_feature_if(df, name, value):
    if value is not None:
        df[name] = value


def load_temperature_data(filepath, excluded_features=None):
    excluded_features = excluded_features or ["dayofweek", "day", "wind_speed_10m", "time", "year"]

    raw_df = pd.read_csv(filepath)
    raw_df.columns = [str(col).strip() for col in raw_df.columns]

    time_col = TIME_COL
    temp_col = TEMP_COL
    dew_col = DEW_COL
    humidity_col = HUMIDITY_COL

    if temp_col not in raw_df.columns:
        raise KeyError(f"Could not find '{temp_col}'. Available columns: {list(raw_df.columns)}")

    data = raw_df.copy()
    if time_col in data.columns:
        data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
        data = data.sort_values(time_col).reset_index(drop=True)

    for col in data.columns:
        if col != time_col and col != "location":
            data[col] = _safe_numeric(data[col])

    base_drop_cols = [c for c in [temp_col, time_col, "location"] if c in data.columns]
    feature_df = data.drop(columns=base_drop_cols).copy()
    feature_df = feature_df.dropna(axis=1, how="all")

    for h in [1, 2, 3, 6, 12]:
        _add_feature_if(feature_df, f"temp_lag_{h}", data[temp_col].shift(h))
        if dew_col in data.columns:
            _add_feature_if(feature_df, f"dew_lag_{h}", data[dew_col].shift(h))

    for h in [24, 48, 72, 96]:
        _add_feature_if(feature_df, f"temp_lag_{h}", data[temp_col].shift(h))
        _add_feature_if(feature_df, f"temp_roll_mean_{h}", data[temp_col].shift(1).rolling(h).mean())
        _add_feature_if(feature_df, f"temp_roll_std_{h}", data[temp_col].shift(1).rolling(h).std())
        _add_feature_if(feature_df, f"temp_roll_min_{h}", data[temp_col].shift(1).rolling(h).min())
        _add_feature_if(feature_df, f"temp_roll_max_{h}", data[temp_col].shift(1).rolling(h).max())
        if dew_col in data.columns:
            _add_feature_if(feature_df, f"dew_lag_{h}", data[dew_col].shift(h))

    if time_col in data.columns:
        _add_feature_if(feature_df, "hour", data[time_col].dt.hour)
        _add_feature_if(feature_df, "dayofweek", data[time_col].dt.dayofweek)
        _add_feature_if(feature_df, "month", data[time_col].dt.month)
        _add_feature_if(feature_df, "day", data[time_col].dt.day)
        _add_feature_if(feature_df, "year", data[time_col].dt.year)

    if dew_col in data.columns:
        _add_feature_if(feature_df, "dew_lag_3", data[dew_col].shift(3))

    if humidity_col in data.columns:
        _add_feature_if(feature_df, "humidity_lag_3", data[humidity_col].shift(3))
        _add_feature_if(feature_df, "humidity_roll_mean_6", data[humidity_col].shift(1).rolling(6).mean())
        _add_feature_if(feature_df, "humidity_change", data[humidity_col] - data[humidity_col].shift(1))

    _add_feature_if(feature_df, "monthly_temp_mean", data[temp_col].shift(1).rolling(720).mean())

    if "cloud_cover_low" in data.columns:
        _add_feature_if(
            feature_df,
            "cloud_cover_low_bin",
            pd.cut(data["cloud_cover_low"], bins=[0, 10, 40, 80, 100], labels=[0, 1, 2, 3]).astype(float),
        )

    if "pressure_msl" in data.columns and "surface_pressure" in data.columns:
        _add_feature_if(feature_df, "pressure_gap", data["pressure_msl"] - data["surface_pressure"])

    selected_feature_cols = [col for col in feature_df.columns if col not in excluded_features]
    feature_df = feature_df[selected_feature_cols].copy()

    target = data[temp_col].shift(-1)
    model_data = feature_df.copy()
    model_data["target"] = target
    trainable = model_data.dropna().copy()

    if len(trainable) == 0:
        raise ValueError("No trainable rows were produced after preprocessing.")

    X = trainable.drop(columns="target")
    y = trainable["target"]

    context = pd.DataFrame(index=trainable.index)
    if time_col in data.columns:
        context["time"] = data.loc[trainable.index, time_col]
    else:
        raise ValueError("A valid 'time' column is required for day-based predictions.")

    if "location" in data.columns:
        context["location"] = data.loc[trainable.index, "location"].astype(str)

    context["actual_next_temperature"] = y

    return X, y, context, feature_df


def get_temperature_prediction_options(filepath):
    df = pd.read_csv(filepath)
    df.columns = [str(col).strip() for col in df.columns]
    if "time" not in df.columns:
        raise ValueError("Dataset must contain a 'time' column.")

    df = df.dropna(subset=["time"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    available_days = pd.Index(df["time"].dt.date.unique()).sort_values()
    locations = []
    if "location" in df.columns:
        locations = sorted(df["location"].dropna().astype(str).unique().tolist())

    return {
        "minDate": str(available_days.min()) if len(available_days) else None,
        "maxDate": str(available_days.max()) if len(available_days) else None,
        "locations": locations,
        "savedModels": list_saved_temperature_models(),
    }


def list_saved_temperature_models():
    if not os.path.exists(ARTIFACT_DIR):
        return []

    saved_models = []
    for file_name in sorted(os.listdir(ARTIFACT_DIR)):
        if not file_name.endswith(".joblib"):
            continue
        artifact_path = os.path.join(ARTIFACT_DIR, file_name)
        try:
            artifact = load(artifact_path)
        except Exception:
            continue

        metadata = artifact.get("metadata", {})
        validation = artifact.get("validation") or metadata.get("validation_metrics") or {}
        saved_models.append(
            {
                "value": file_name,
                "label": os.path.splitext(file_name)[0],
                "artifactPath": artifact_path,
                "modelName": artifact.get("modelName") or metadata.get("model_name") or os.path.splitext(file_name)[0],
                "validation": validation,
            }
        )

    return saved_models


def _regression_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _load_raw_weather_data(filepath):
    raw_df = pd.read_csv(filepath)
    raw_df.columns = [str(col).strip() for col in raw_df.columns]
    if TIME_COL not in raw_df.columns:
        raise ValueError("Dataset must contain a 'time' column.")

    raw_df[TIME_COL] = pd.to_datetime(raw_df[TIME_COL], errors="coerce")
    raw_df = raw_df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)

    for col in raw_df.columns:
        if col != TIME_COL and col != "location":
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    return raw_df


def _build_profile_bundle(raw_df):
    prof_df = raw_df.copy()
    prof_df["month"] = prof_df[TIME_COL].dt.month
    prof_df["day"] = prof_df[TIME_COL].dt.day
    prof_df["hour"] = prof_df[TIME_COL].dt.hour

    numeric_cols = prof_df.select_dtypes(include=[np.number]).columns.tolist()
    profile_cols = [col for col in numeric_cols if col not in {"month", "day", "hour"}]

    mdh = prof_df.groupby(["month", "day", "hour"])[profile_cols].mean()
    mh = prof_df.groupby(["month", "hour"])[profile_cols].mean()
    h = prof_df.groupby(["hour"])[profile_cols].mean()
    overall = prof_df[profile_cols].mean()

    return {"mdh": mdh, "mh": mh, "h": h, "overall": overall}


def _lookup_profile_row(profile_bundle, ts):
    key_mdh = (ts.month, ts.day, ts.hour)
    key_mh = (ts.month, ts.hour)
    key_h = ts.hour

    if key_mdh in profile_bundle["mdh"].index:
        return profile_bundle["mdh"].loc[key_mdh]
    if key_mh in profile_bundle["mh"].index:
        return profile_bundle["mh"].loc[key_mh]
    if key_h in profile_bundle["h"].index:
        return profile_bundle["h"].loc[key_h]
    return profile_bundle["overall"]


def _build_profile_history(profile_bundle, start_time, periods, column_name):
    values = []
    for ts in pd.date_range(start_time, periods=periods, freq="h"):
        profile_row = _lookup_profile_row(profile_bundle, ts)
        if column_name in profile_row.index and not pd.isna(profile_row[column_name]):
            values.append(float(profile_row[column_name]))
    return values


def _build_recursive_feature_row(base_row, feature_columns, feature_time, profile_row, temperature_history, dew_history, humidity_history):
    row = base_row.copy()

    for col in feature_columns:
        if col in profile_row.index:
            row[col] = float(profile_row[col])

    for h in [1, 2, 3, 6, 12, 24, 48, 72, 96]:
        if f"temp_lag_{h}" in feature_columns and len(temperature_history) >= h:
            row[f"temp_lag_{h}"] = float(temperature_history[-h])
        if f"dew_lag_{h}" in feature_columns and len(dew_history) >= h:
            row[f"dew_lag_{h}"] = float(dew_history[-h])

    for h in [24, 48, 72, 96]:
        if len(temperature_history) >= h:
            window = np.array(temperature_history[-h:], dtype=float)
            if f"temp_roll_mean_{h}" in feature_columns:
                row[f"temp_roll_mean_{h}"] = float(np.mean(window))
            if f"temp_roll_std_{h}" in feature_columns:
                row[f"temp_roll_std_{h}"] = float(np.std(window))
            if f"temp_roll_min_{h}" in feature_columns:
                row[f"temp_roll_min_{h}"] = float(np.min(window))
            if f"temp_roll_max_{h}" in feature_columns:
                row[f"temp_roll_max_{h}"] = float(np.max(window))

    if "hour" in feature_columns:
        row["hour"] = feature_time.hour
    if "dayofweek" in feature_columns:
        row["dayofweek"] = feature_time.dayofweek
    if "month" in feature_columns:
        row["month"] = feature_time.month
    if "day" in feature_columns:
        row["day"] = feature_time.day
    if "year" in feature_columns:
        row["year"] = feature_time.year

    if "monthly_temp_mean" in feature_columns and len(temperature_history) > 0:
        window = temperature_history[-720:] if len(temperature_history) >= 720 else temperature_history
        row["monthly_temp_mean"] = float(np.mean(window))

    if "humidity_lag_3" in feature_columns and len(humidity_history) >= 3:
        row["humidity_lag_3"] = float(humidity_history[-3])
    if "humidity_roll_mean_6" in feature_columns and len(humidity_history) >= 6:
        row["humidity_roll_mean_6"] = float(np.mean(humidity_history[-6:]))
    if "humidity_change" in feature_columns and len(humidity_history) >= 2:
        row["humidity_change"] = float(humidity_history[-1] - humidity_history[-2])

    if "cloud_cover_low_bin" in feature_columns and "cloud_cover_low" in profile_row.index:
        cloud_cover_low = float(profile_row["cloud_cover_low"])
        if cloud_cover_low < 10:
            row["cloud_cover_low_bin"] = 0.0
        elif cloud_cover_low < 40:
            row["cloud_cover_low_bin"] = 1.0
        elif cloud_cover_low < 80:
            row["cloud_cover_low_bin"] = 2.0
        else:
            row["cloud_cover_low_bin"] = 3.0

    if "pressure_gap" in feature_columns and "pressure_msl" in profile_row.index and "surface_pressure" in profile_row.index:
        row["pressure_gap"] = float(profile_row["pressure_msl"] - profile_row["surface_pressure"])

    return row


def _forecast_future_with_profiles(filepath, location, selected_day, last_hist_time, X_hist, y_hist, feature_columns, imputer, model_bundle):
    raw_df = _load_raw_weather_data(filepath)
    if location and "location" in raw_df.columns:
        raw_df = raw_df.loc[raw_df["location"].astype(str).eq(str(location))].copy()

    history_raw = raw_df.loc[raw_df[TIME_COL] <= last_hist_time].copy()
    if history_raw.empty:
        raise ValueError("No historical raw rows available for future forecasting.")

    profile_bundle = _build_profile_bundle(raw_df)
    base_row = X_hist.iloc[-1].copy()

    temperature_history = history_raw[TEMP_COL].dropna().astype(float).tolist()
    dew_history = history_raw[DEW_COL].dropna().astype(float).tolist() if DEW_COL in history_raw.columns else []
    humidity_history = history_raw[HUMIDITY_COL].dropna().astype(float).tolist() if HUMIDITY_COL in history_raw.columns else []

    hours_until_selected_day = int((selected_day - (last_hist_time + pd.Timedelta(hours=1)).normalize()).total_seconds() // 3600)
    if hours_until_selected_day > 96:
        profile_history_start = selected_day - pd.Timedelta(hours=96)
        temperature_history = _build_profile_history(profile_bundle, profile_history_start, 96, TEMP_COL)
        if DEW_COL in raw_df.columns:
            dew_history = _build_profile_history(profile_bundle, profile_history_start, 96, DEW_COL)
        if HUMIDITY_COL in raw_df.columns:
            humidity_history = _build_profile_history(profile_bundle, profile_history_start, 96, HUMIDITY_COL)
        target_times = pd.date_range(selected_day, selected_day + pd.Timedelta(hours=23), freq="h")
    else:
        target_times = pd.date_range(last_hist_time + pd.Timedelta(hours=1), selected_day + pd.Timedelta(hours=23), freq="h")

    selected_predictions = []

    for target_time in target_times:
        feature_time = target_time - pd.Timedelta(hours=1)
        profile_row = _lookup_profile_row(profile_bundle, feature_time)
        feature_row = _build_recursive_feature_row(
            base_row=base_row,
            feature_columns=feature_columns,
            feature_time=feature_time,
            profile_row=profile_row,
            temperature_history=temperature_history,
            dew_history=dew_history,
            humidity_history=humidity_history,
        )

        future_X = pd.DataFrame([feature_row], index=[feature_time]).reindex(columns=feature_columns)
        future_X_imputed = pd.DataFrame(imputer.transform(future_X), index=future_X.index, columns=future_X.columns)
        predicted_temp = float(_predict_model_bundle(model_bundle, future_X_imputed)[0])

        target_profile = _lookup_profile_row(profile_bundle, target_time)
        profile_temp = target_profile.get(TEMP_COL)
        if profile_temp is not None and not pd.isna(profile_temp):
            blend_weight = 0.0 if hours_until_selected_day <= 24 else min(0.65, hours_until_selected_day / (24.0 * 60.0))
            predicted_temp = float((1.0 - blend_weight) * predicted_temp + blend_weight * float(profile_temp))

        temperature_history.append(predicted_temp)

        if DEW_COL in target_profile.index:
            dew_history.append(float(target_profile[DEW_COL]))
        if HUMIDITY_COL in target_profile.index:
            humidity_history.append(float(target_profile[HUMIDITY_COL]))

        if target_time.normalize() == selected_day:
            row = {"time": target_time.isoformat(), "predictedTemperature": predicted_temp, "actualTemperature": None, "absoluteError": None}
            if location:
                row["location"] = str(location)
            selected_predictions.append(row)

    if not selected_predictions:
        raise ValueError("No forecast hours were generated for selected date.")

    return selected_predictions


def _predict_model_bundle(model_bundle, X_pred_imputed):
    if model_bundle["type"] == "vote":
        predictions = [model.predict(X_pred_imputed) for _, model in model_bundle["models"]]
        return np.mean(np.column_stack(predictions), axis=1)

    return model_bundle["model"].predict(X_pred_imputed)


def _load_selected_saved_model(saved_model_name):
    artifact_path = os.path.join(ARTIFACT_DIR, saved_model_name)
    if not os.path.exists(artifact_path):
        raise ValueError(f"Saved model not found: {saved_model_name}")

    artifact = load(artifact_path)
    artifact["artifactPath"] = artifact_path
    return artifact


def _artifact_feature_columns(artifact):
    return artifact.get("featureColumns") or artifact.get("feature_columns")


def _artifact_validation(artifact):
    metadata = artifact.get("metadata", {})
    return artifact.get("validation") or metadata.get("validation_metrics") or {}


def _artifact_model_name(artifact):
    metadata = artifact.get("metadata", {})
    return artifact.get("modelName") or metadata.get("model_name") or os.path.splitext(os.path.basename(artifact["artifactPath"]))[0]


def _artifact_model_bundle(artifact):
    if "modelBundle" in artifact:
        return artifact["modelBundle"]
    if "model" in artifact:
        return {"type": "single", "family": "saved", "model": artifact["model"]}
    raise ValueError("Saved artifact does not contain a usable model.")


def predict_temperature_for_day(
    filepath,
    selected_date,
    location="",
    excluded_features=None,
    saved_model_name=None,
):
    if not saved_model_name:
        raise ValueError("A saved model must be selected for temperature prediction.")

    selected_artifact = _load_selected_saved_model(saved_model_name)
    excluded_features = selected_artifact.get("excludedFeatures") or selected_artifact.get("excluded_features") or excluded_features

    excluded_features = excluded_features or ["dayofweek", "day", "wind_speed_10m", "time", "year"]
    X, y, context, _ = load_temperature_data(filepath=filepath, excluded_features=excluded_features)

    selected_day = pd.to_datetime(selected_date, errors="coerce")
    if pd.isna(selected_day):
        raise ValueError(f"Invalid selected date: {selected_date}")
    selected_day = selected_day.normalize()

    context = context.copy()
    context["prediction_day"] = context["time"].dt.normalize()

    has_location_col = "location" in context.columns
    location_filter = None
    if location and has_location_col:
        location_filter = context["location"].astype(str).eq(str(location))
        if int(location_filter.sum()) == 0:
            raise ValueError(f"Location '{location}' was not found in the processed dataset.")

    prediction_mask = context["prediction_day"].eq(selected_day)
    if location_filter is not None:
        prediction_mask &= location_filter

    if location_filter is not None:
        max_hist_day = context.loc[location_filter, "prediction_day"].max()
    else:
        max_hist_day = context["prediction_day"].max()
    is_future_forecast = selected_day > max_hist_day

    if is_future_forecast:
        train_mask = context["prediction_day"].le(max_hist_day)
    else:
        train_mask = context["prediction_day"].lt(selected_day)

    if location_filter is not None:
        train_mask &= location_filter

    if (not is_future_forecast) and prediction_mask.sum() == 0:
        if location:
            raise ValueError(f"No rows found for {location} on {selected_day.date()}.")
        raise ValueError(f"No rows found on {selected_day.date()}.")

    if train_mask.sum() < 200:
        raise ValueError(
            f"Not enough historical rows before {selected_day.date()} to predict"
            f"{f' for {location}' if location else ''}. Found {int(train_mask.sum())}."
        )

    X_hist = X.loc[train_mask]
    y_hist = y.loc[train_mask]
    X_pred = X.loc[prediction_mask] if not is_future_forecast else pd.DataFrame(columns=X.columns)
    y_actual = y.loc[prediction_mask] if not is_future_forecast else pd.Series(dtype=float)
    pred_context = context.loc[prediction_mask].copy() if not is_future_forecast else pd.DataFrame()

    loaded_from_cache = True
    validation = _artifact_validation(selected_artifact)
    model_name = _artifact_model_name(selected_artifact)
    imputer = selected_artifact.get("imputer")
    if imputer is None:
        raise ValueError("Saved artifact does not contain an imputer.")
    model_bundle = _artifact_model_bundle(selected_artifact)
    feature_columns = _artifact_feature_columns(selected_artifact)
    if not feature_columns:
        raise ValueError("Saved artifact does not contain feature columns.")

    if is_future_forecast:
        last_hist_time = context.loc[train_mask, "time"].max()
        if pd.isna(last_hist_time):
            raise ValueError("Could not determine last historical timestamp for future forecasting.")

        target_end = selected_day + pd.Timedelta(hours=23)
        if target_end <= last_hist_time:
            raise ValueError("Selected day is not in the future.")
        hourly_rows = _forecast_future_with_profiles(
            filepath=filepath,
            location=location,
            selected_day=selected_day,
            last_hist_time=last_hist_time,
            X_hist=X_hist,
            y_hist=y_hist,
            feature_columns=feature_columns,
            imputer=imputer,
            model_bundle=model_bundle,
        )
        y_pred = np.array([row["predictedTemperature"] for row in hourly_rows], dtype=float)
        pred_context = pd.DataFrame(hourly_rows)
    else:
        X_pred_imputed = pd.DataFrame(imputer.transform(X_pred.reindex(columns=feature_columns)), index=X_pred.index, columns=feature_columns)
        y_pred = _predict_model_bundle(model_bundle, X_pred_imputed)

    day_metrics = _regression_metrics(y_actual, y_pred) if not is_future_forecast else None

    hourly_rows = []
    for idx in range(len(pred_context)):
        time_value = pred_context.iloc[idx]["time"]
        if hasattr(time_value, "isoformat"):
            time_value = time_value.isoformat()
        row = {
            "time": time_value,
            "predictedTemperature": float(y_pred[idx]),
            "actualTemperature": float(y_actual.iloc[idx]) if not is_future_forecast else None,
            "absoluteError": float(abs(y_actual.iloc[idx] - y_pred[idx])) if not is_future_forecast else None,
        }
        if "location" in pred_context.columns:
            row["location"] = str(pred_context.iloc[idx]["location"])
        hourly_rows.append(row)

    return {
        "selectedDate": str(selected_day.date()),
        "location": str(location),
        "forecastMode": "future" if is_future_forecast else "historical",
        "loadedFromCache": loaded_from_cache,
        "lastHistoricalDate": str(max_hist_day.date()),
        "modelName": model_name,
        "validation": validation,
        "dayMetrics": day_metrics,
        "predictedAverage": float(np.mean(y_pred)),
        "actualAverage": float(np.mean(y_actual)) if not is_future_forecast else None,
        "predictedMin": float(np.min(y_pred)),
        "predictedMax": float(np.max(y_pred)),
        "hourly": hourly_rows,
        "trainingSamples": int(len(X_hist)),
        "totalHours": int(len(hourly_rows)),
        "artifactPath": selected_artifact.get("artifactPath"),
    }


def _build_parser():
    parser = argparse.ArgumentParser(description="Temperature prediction helper for Level 2 UI.")
    parser.add_argument("--dataset-path", required=True, help="Path to meteorology dataset.")
    parser.add_argument("--selected-date", required=True, help="Date to predict (YYYY-MM-DD).")
    parser.add_argument("--location", default="", help="Optional location filter.")
    parser.add_argument("--saved-model", required=True, help="Saved model filename from level2/saved_models.")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    result = predict_temperature_for_day(
        filepath=args.dataset_path,
        selected_date=args.selected_date,
        location=args.location,
        saved_model_name=args.saved_model,
    )

    print("Level 2 temperature prediction")
    print(f"Date: {result['selectedDate']}")
    print(f"Model: {result['modelName']}")
    print(f"Predicted avg: {result['predictedAverage']:.3f}")
    if result["actualAverage"] is None:
        print("Actual avg: N/A (future forecast)")
        print("MAE: N/A (future forecast)")
    else:
        print(f"Actual avg: {result['actualAverage']:.3f}")
        print(f"MAE: {result['dayMetrics']['mae']:.3f}")


if __name__ == "__main__":
    main()
