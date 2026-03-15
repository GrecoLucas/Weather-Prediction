import numpy as np
import pandas as pd
import os
import hashlib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


CANDIDATE_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "rain",
    "cloud_cover",
    "wind_speed_10m",
    "pressure_msl",
]


_LEVEL3_CACHE = {}
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
_PORTABLE_ARTIFACT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "level3_model_artifact.pkl")


def get_snowfall_prediction_options(filepath):
    df = pd.read_csv(filepath)
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(subset=["time", "location"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    available_days = pd.Index(df["time"].dt.date.unique()).sort_values()
    locations = sorted(df["location"].astype(str).unique().tolist())

    return {
        "locations": locations,
        "minDate": str(available_days.min()) if len(available_days) else None,
        "maxDate": str(available_days.max()) if len(available_days) else None,
    }


def _anomaly_percentage(pred):
    return float((pred == -1).mean() * 100)


def _build_models():
    return {
        "LocalOutlierFactor": LocalOutlierFactor(n_neighbors=35, contamination=0.04, novelty=True),
    }


def _dataset_signature(filepath):
    stat = os.stat(filepath)
    return (os.path.abspath(filepath), stat.st_mtime_ns, stat.st_size)


def _dataset_content_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _context_cache_path(signature):
    signature_raw = f"{signature[0]}|{signature[1]}|{signature[2]}"
    signature_hash = hashlib.sha256(signature_raw.encode("utf-8")).hexdigest()[:16]
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"level3_context_{signature_hash}.pkl")


def _load_context_from_disk(signature):
    cache_path = _context_cache_path(signature)
    if not os.path.exists(cache_path):
        return None

    try:
        cached = pd.read_pickle(cache_path)
    except Exception:
        return None

    if not isinstance(cached, dict):
        return None

    if cached.get("signature") != signature:
        return None

    eval_df = cached.get("eval_df")
    if not isinstance(eval_df, pd.DataFrame) or eval_df.empty:
        return None

    best_model_name = cached.get("best_model_name")
    training_samples = cached.get("training_samples")
    if not best_model_name or training_samples is None:
        return None

    return {
        "eval_df": eval_df,
        "best_model_name": best_model_name,
        "training_samples": int(training_samples),
    }


def _save_context_to_disk(signature, context):
    cache_path = _context_cache_path(signature)
    payload = {
        "signature": signature,
        "eval_df": context["eval_df"],
        "best_model_name": context["best_model_name"],
        "training_samples": context["training_samples"],
    }
    pd.to_pickle(payload, cache_path)


def _load_portable_context(filepath):
    if not os.path.exists(_PORTABLE_ARTIFACT_PATH):
        return None

    try:
        artifact = pd.read_pickle(_PORTABLE_ARTIFACT_PATH)
    except Exception:
        return None

    if not isinstance(artifact, dict):
        return None

    expected_hash = artifact.get("dataset_hash")
    if not expected_hash:
        return None

    current_hash = _dataset_content_hash(filepath)
    if current_hash != expected_hash:
        return None

    eval_df = artifact.get("eval_df")
    best_model_name = artifact.get("best_model_name")
    training_samples = artifact.get("training_samples")

    if not isinstance(eval_df, pd.DataFrame) or eval_df.empty:
        return None
    if not best_model_name or training_samples is None:
        return None

    return {
        "eval_df": eval_df,
        "best_model_name": best_model_name,
        "training_samples": int(training_samples),
    }


def _save_portable_context(filepath, context):
    payload = {
        "dataset_hash": _dataset_content_hash(filepath),
        "eval_df": context["eval_df"],
        "best_model_name": context["best_model_name"],
        "training_samples": context["training_samples"],
    }
    pd.to_pickle(payload, _PORTABLE_ARTIFACT_PATH)


def _get_cached_level3_context(filepath):
    signature = _dataset_signature(filepath)
    if signature in _LEVEL3_CACHE:
        return _LEVEL3_CACHE[signature]

    portable_context = _load_portable_context(filepath)
    if portable_context is not None:
        _LEVEL3_CACHE.clear()
        _LEVEL3_CACHE[signature] = portable_context
        return portable_context

    disk_context = _load_context_from_disk(signature)
    if disk_context is not None:
        _LEVEL3_CACHE.clear()
        _LEVEL3_CACHE[signature] = disk_context
        return disk_context

    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]

    if "time" not in df.columns:
        raise ValueError("Dataset must include a 'time' column.")
    if "location" not in df.columns:
        raise ValueError("Dataset must include a 'location' column.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    feature_cols = [c for c in CANDIDATE_FEATURES if c in df.columns]
    if not feature_cols:
        raise ValueError("No expected weather feature columns were found.")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    order = np.argsort(df["time"].values)
    X_all_raw = X.iloc[order].copy()
    ordered_df = df.iloc[order].copy()

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all_raw)

    model_outputs = []
    for model_name, model in _build_models().items():
        model.fit(X_all_scaled)
        pred = model.predict(X_all_scaled)
        model_outputs.append(
            {
                "modelName": model_name,
                "model": model,
                "pred": pred,
                "anomalyPct": _anomaly_percentage(pred),
            }
        )

    best = min(model_outputs, key=lambda m: m["anomalyPct"])
    best_model_name = best["modelName"]
    best_model = best["model"]
    best_pred = best["pred"]

    if hasattr(best_model, "score_samples"):
        raw_scores = -best_model.score_samples(X_all_scaled)
    elif hasattr(best_model, "decision_function"):
        raw_scores = -best_model.decision_function(X_all_scaled)
    else:
        raw_scores = np.where(best_pred == -1, 1.0, 0.0)

    eval_df = ordered_df.copy()
    eval_df["anomaly_score"] = raw_scores
    eval_df["is_anomaly_raw"] = (best_pred == -1).astype(int)

    snow_like_mask = pd.Series(True, index=eval_df.index)
    if "temperature_2m" in eval_df.columns:
        snow_like_mask &= eval_df["temperature_2m"] <= 3
    if "relative_humidity_2m" in eval_df.columns:
        snow_like_mask &= eval_df["relative_humidity_2m"] >= 75
    if "rain" in eval_df.columns:
        snow_like_mask &= eval_df["rain"] >= 0.1

    eval_df["is_anomaly"] = ((eval_df["is_anomaly_raw"] == 1) & snow_like_mask).astype(int)
    if eval_df["is_anomaly"].sum() == 0:
        eval_df["is_anomaly"] = eval_df["is_anomaly_raw"]

    context = {
        "eval_df": eval_df,
        "best_model_name": best_model_name,
        "training_samples": int(len(eval_df)),
    }

    _LEVEL3_CACHE.clear()
    _LEVEL3_CACHE[signature] = context

    try:
        _save_context_to_disk(signature, context)
    except Exception:
        pass

    try:
        _save_portable_context(filepath, context)
    except Exception:
        pass

    return context


def train_and_save_level3_model(filepath):
    context = _get_cached_level3_context(filepath)
    _save_portable_context(filepath, context)
    return {
        "ok": True,
        "artifactPath": _PORTABLE_ARTIFACT_PATH,
        "trainingSamples": int(context["training_samples"]),
        "modelName": str(context["best_model_name"]),
    }


def predict_snowfall_for_district(filepath, location):
    context = _get_cached_level3_context(filepath)
    eval_df = context["eval_df"]
    best_model_name = context["best_model_name"]

    district_df = eval_df[eval_df["location"].astype(str).eq(str(location))].copy().sort_values("time").reset_index(drop=True)
    if district_df.empty:
        raise ValueError(f"No rows found for district/location: {location}.")

    scores = district_df["anomaly_score"].to_numpy(dtype=float)
    score_min = float(np.min(scores)) if len(scores) else 0.0
    score_max = float(np.max(scores)) if len(scores) else 1.0
    if score_max > score_min:
        confidences = (scores - score_min) / (score_max - score_min)
    else:
        confidences = np.zeros_like(scores)

    district_df["confidence"] = confidences
    predicted_snow = district_df["is_anomaly"] == 1
    snowy_rows = district_df[predicted_snow].copy()
    has_observed_snow_column = "snowfall" in district_df.columns

    hourly_rows = []
    for _, row in snowy_rows.iterrows():
        observed_snow = None
        if has_observed_snow_column:
            observed_snow = bool(float(row.get("snowfall", 0) or 0) > 0)

        hourly_rows.append(
            {
                "time": row["time"].isoformat(),
                "confidence": float(row["confidence"]),
                "temperature": float(row.get("temperature_2m", np.nan)) if pd.notna(row.get("temperature_2m", np.nan)) else None,
                "humidity": float(row.get("relative_humidity_2m", np.nan)) if pd.notna(row.get("relative_humidity_2m", np.nan)) else None,
                "rain": float(row.get("rain", np.nan)) if pd.notna(row.get("rain", np.nan)) else None,
                "observedSnow": observed_snow,
            }
        )

    monthly_counts = []
    if len(snowy_rows):
        month_counts_series = snowy_rows["time"].dt.to_period("M").value_counts().sort_index()
        monthly_counts = [
            {"month": str(period), "count": int(count)}
            for period, count in month_counts_series.items()
        ]

    snowy_hours = int(len(snowy_rows))
    total_hours = int(len(district_df))
    snowy_rate = float((snowy_hours / total_hours) if total_hours else 0.0)
    first_snow_hour = hourly_rows[0]["time"] if hourly_rows else None
    last_snow_hour = hourly_rows[-1]["time"] if hourly_rows else None

    return {
        "location": str(location),
        "modelName": f"{best_model_name} (Level 3 Best Anomaly Model)",
        "chosenParams": {
            "temperature_threshold": 3,
            "humidity_threshold": 75,
            "rain_threshold": 0.1,
        },
        "snowyHours": snowy_hours,
        "totalHours": total_hours,
        "snowyRate": snowy_rate,
        "firstSnowHour": first_snow_hour,
        "lastSnowHour": last_snow_hour,
        "confidence": float(np.mean(snowy_rows["confidence"])) if snowy_hours else 0.0,
        "monthlyCounts": monthly_counts,
        "hasObservedSnow": has_observed_snow_column,
        "validation": {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        },
        "trainingSamples": int(context["training_samples"]),
        "hourly": hourly_rows,
    }


# Backward-compatible aliases
def get_prediction_options(filepath):
    return get_snowfall_prediction_options(filepath)


def predict_snow_for_day(filepath, selected_date, location):
    return predict_snowfall_for_district(filepath=filepath, location=location)
