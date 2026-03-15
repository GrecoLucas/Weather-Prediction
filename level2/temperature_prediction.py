import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _add_feature_if(df, name, value):
    if value is not None:
        df[name] = value


def load_temperature_data(filepath, excluded_features=None):
    excluded_features = excluded_features or ["dayofweek", "day", "wind_speed_10m", "time", "year"]

    raw_df = pd.read_csv(filepath)
    raw_df.columns = [str(col).strip() for col in raw_df.columns]

    time_col = "time"
    temp_col = "temperature_2m"
    dew_col = "dew_point_2m"
    humidity_col = "relative_humidity_2m"

    if temp_col not in raw_df.columns:
        raise KeyError(f"Could not find '{temp_col}'. Available columns: {list(raw_df.columns)}")

    data = raw_df.copy()
    if time_col in data.columns:
        data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
        data = data.sort_values(time_col).reset_index(drop=True)

    for col in data.columns:
        if col != time_col:
            data[col] = _safe_numeric(data[col])

    base_drop_cols = [c for c in [temp_col, time_col] if c in data.columns]
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

    if "location" in raw_df.columns:
        context["location"] = raw_df.loc[trainable.index, "location"].astype(str)

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
    }


def _make_model(model_family):
    family = model_family.lower()

    if family == "rf":
        return RandomForestRegressor(
            n_estimators=450,
            max_depth=16,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    if family == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )

    if family == "lgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            force_col_wise=True,
            verbosity=-1,
            random_state=42,
            n_jobs=-1,
        )

    if family == "lr":
        return LinearRegression()

    raise ValueError(f"Unsupported model family: {model_family}")


def _fit_model(model_family, model, X_train, y_train, X_val, y_val):
    family = model_family.lower()

    if family == "xgb":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif family == "lgbm":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="l2")
    else:
        model.fit(X_train, y_train)

    return model


def _regression_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _build_future_feature_row(base_row, columns, temperature_history, row_time):
    row = base_row.copy()

    for h in [1, 2, 3, 6, 12, 24, 48, 72, 96]:
        if f"temp_lag_{h}" in columns and len(temperature_history) >= h:
            row[f"temp_lag_{h}"] = float(temperature_history[-h])

    for h in [24, 48, 72, 96]:
        if len(temperature_history) >= h:
            window = np.array(temperature_history[-h:], dtype=float)
            if f"temp_roll_mean_{h}" in columns:
                row[f"temp_roll_mean_{h}"] = float(np.mean(window))
            if f"temp_roll_std_{h}" in columns:
                row[f"temp_roll_std_{h}"] = float(np.std(window))
            if f"temp_roll_min_{h}" in columns:
                row[f"temp_roll_min_{h}"] = float(np.min(window))
            if f"temp_roll_max_{h}" in columns:
                row[f"temp_roll_max_{h}"] = float(np.max(window))

    if "hour" in columns:
        row["hour"] = row_time.hour
    if "dayofweek" in columns:
        row["dayofweek"] = row_time.dayofweek
    if "month" in columns:
        row["month"] = row_time.month
    if "day" in columns:
        row["day"] = row_time.day
    if "year" in columns:
        row["year"] = row_time.year

    if "monthly_temp_mean" in columns and len(temperature_history) > 0:
        window = temperature_history[-720:] if len(temperature_history) >= 720 else temperature_history
        row["monthly_temp_mean"] = float(np.mean(window))

    return row


def _predict_with_family(model_family, X_train_imputed, y_train, X_val_imputed, y_val, X_hist_imputed, X_pred_imputed):
    family = model_family.lower()

    if family == "vote":
        base_families = ["rf", "xgb", "lgbm"]
        val_pred_list = []
        pred_list = []
        for item in base_families:
            model = _make_model(item)
            model = _fit_model(item, model, X_train_imputed, y_train, X_val_imputed, y_val)
            val_pred_list.append(model.predict(X_val_imputed))

            # Refit on all historical rows before final prediction.
            model = _fit_model(item, model, X_hist_imputed, pd.concat([y_train, y_val]), X_val_imputed, y_val)
            pred_list.append(model.predict(X_pred_imputed))

        val_pred = np.mean(np.column_stack(val_pred_list), axis=1)
        y_pred = np.mean(np.column_stack(pred_list), axis=1)
        model_name = "Voting (RF + XGB + LGBM)"
        return val_pred, y_pred, model_name

    model = _make_model(family)
    model = _fit_model(family, model, X_train_imputed, y_train, X_val_imputed, y_val)
    val_pred = model.predict(X_val_imputed)

    # Refit on all available historical rows before forecasting.
    model = _fit_model(family, model, X_hist_imputed, pd.concat([y_train, y_val]), X_val_imputed, y_val)
    y_pred = model.predict(X_pred_imputed)
    model_name = model.__class__.__name__
    return val_pred, y_pred, model_name


def predict_temperature_for_day(
    filepath,
    selected_date,
    location="",
    model_family="lgbm",
    excluded_features=None,
):
    X, y, context, _ = load_temperature_data(filepath=filepath, excluded_features=excluded_features)

    selected_day = pd.to_datetime(selected_date, errors="coerce")
    if pd.isna(selected_day):
        raise ValueError(f"Invalid selected date: {selected_date}")
    selected_day = selected_day.normalize()

    context = context.copy()
    context["prediction_day"] = context["time"].dt.normalize()

    prediction_mask = context["prediction_day"].eq(selected_day)
    if location and "location" in context.columns:
        prediction_mask &= context["location"].astype(str).eq(str(location))

    max_hist_day = context["prediction_day"].max()
    is_future_forecast = selected_day > max_hist_day

    if is_future_forecast:
        train_mask = context["prediction_day"].le(max_hist_day)
    else:
        train_mask = context["prediction_day"].lt(selected_day)

    if (not is_future_forecast) and prediction_mask.sum() == 0:
        if location:
            raise ValueError(f"No rows found for {location} on {selected_day.date()}.")
        raise ValueError(f"No rows found on {selected_day.date()}.")

    if train_mask.sum() < 200:
        raise ValueError(
            f"Not enough historical rows before {selected_day.date()} to train. Found {int(train_mask.sum())}."
        )

    X_hist = X.loc[train_mask]
    y_hist = y.loc[train_mask]
    X_pred = X.loc[prediction_mask] if not is_future_forecast else pd.DataFrame(columns=X.columns)
    y_actual = y.loc[prediction_mask] if not is_future_forecast else pd.Series(dtype=float)
    pred_context = context.loc[prediction_mask].copy() if not is_future_forecast else pd.DataFrame()

    split_idx = max(1, int(len(X_hist) * 0.9))
    if split_idx >= len(X_hist):
        split_idx = len(X_hist) - 1
    if split_idx <= 0:
        raise ValueError("Not enough historical rows to create validation split.")

    X_train = X_hist.iloc[:split_idx]
    y_train = y_hist.iloc[:split_idx]
    X_val = X_hist.iloc[split_idx:]
    y_val = y_hist.iloc[split_idx:]

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_val_imputed = pd.DataFrame(imputer.transform(X_val), index=X_val.index, columns=X_val.columns)
    X_hist_imputed = pd.DataFrame(imputer.transform(X_hist), index=X_hist.index, columns=X_hist.columns)
    if is_future_forecast:
        last_hist_time = context.loc[train_mask, "time"].max()
        if pd.isna(last_hist_time):
            raise ValueError("Could not determine last historical timestamp for future forecasting.")

        target_end = selected_day + pd.Timedelta(hours=23)
        if target_end <= last_hist_time:
            raise ValueError("Selected day is not in the future.")

        future_times = pd.date_range(last_hist_time + pd.Timedelta(hours=1), target_end, freq="h")
        output_times = [ts for ts in future_times if ts.normalize() == selected_day]

        if not output_times:
            raise ValueError("No forecast hours were generated for selected date.")

        base_row = X_hist.iloc[-1].copy()
        temp_history = y_hist.tail(1000).tolist()

        future_rows = []
        for ts in future_times:
            new_row = _build_future_feature_row(base_row, X.columns, temp_history, ts)
            future_rows.append(new_row)
            temp_history.append(float(temp_history[-1]))

        X_future = pd.DataFrame(future_rows, index=future_times)
        X_future = X_future.reindex(columns=X.columns)
        X_future_imputed = pd.DataFrame(imputer.transform(X_future), index=X_future.index, columns=X_future.columns)

        val_pred, all_future_pred, model_name = _predict_with_family(
            model_family=model_family,
            X_train_imputed=X_train_imputed,
            y_train=y_train,
            X_val_imputed=X_val_imputed,
            y_val=y_val,
            X_hist_imputed=X_hist_imputed,
            X_pred_imputed=X_future_imputed,
        )

        output_pred = pd.Series(all_future_pred, index=future_times).loc[output_times]
        y_pred = output_pred.values
        pred_context = pd.DataFrame({"time": output_times})
        if location:
            pred_context["location"] = str(location)
    else:
        X_pred_imputed = pd.DataFrame(imputer.transform(X_pred), index=X_pred.index, columns=X_pred.columns)
        val_pred, y_pred, model_name = _predict_with_family(
            model_family=model_family,
            X_train_imputed=X_train_imputed,
            y_train=y_train,
            X_val_imputed=X_val_imputed,
            y_val=y_val,
            X_hist_imputed=X_hist_imputed,
            X_pred_imputed=X_pred_imputed,
        )

    validation = _regression_metrics(y_val, val_pred)
    day_metrics = _regression_metrics(y_actual, y_pred) if not is_future_forecast else None

    hourly_rows = []
    for idx in range(len(pred_context)):
        row = {
            "time": pred_context.iloc[idx]["time"].isoformat(),
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
    }


def _build_parser():
    parser = argparse.ArgumentParser(description="Temperature prediction helper for Level 2 UI.")
    parser.add_argument("--dataset-path", required=True, help="Path to meteorology dataset.")
    parser.add_argument("--selected-date", required=True, help="Date to predict (YYYY-MM-DD).")
    parser.add_argument("--location", default="", help="Optional location filter.")
    parser.add_argument(
        "--model-family",
        default="lgbm",
        choices=["lr", "rf", "xgb", "lgbm", "vote"],
        help="Temperature model family.",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    result = predict_temperature_for_day(
        filepath=args.dataset_path,
        selected_date=args.selected_date,
        location=args.location,
        model_family=args.model_family,
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
