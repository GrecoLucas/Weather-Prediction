import argparse
import os
import pickle

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# ── Level 1 model cache ──────────────────────────────────────────────────────
_MODEL_CACHE = {}  # (file_hash, model_family, profile) -> cached dict


def _file_hash(filepath):
    """Lightweight hash from mtime + size to detect dataset changes."""
    s = os.stat(filepath)
    return f"{s.st_mtime:.0f}_{s.st_size}"


def _get_or_build_level1_model(filepath, processed_df, model_family="all", profile="balanced"):
    """Return a cached global model for the requested model family/profile.

    Trains on all locations combined using an 80/10/10 split, then caches to
    disk and memory. If model_family='all', picks the best model by validation
    F1. If model_family is one of rf/xgb/lgbm, uses only that model.
    """
    fhash = _file_hash(filepath)
    cache_key = (fhash, str(model_family), str(profile))

    if cache_key in _MODEL_CACHE:
        print(f"[Level1] In-memory global model hit ({model_family}/{profile}).")
        return _MODEL_CACHE[cache_key]

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(filepath)), ".cache", "level1")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"global_{model_family}_{profile}_{fhash}.pkl")

    if os.path.exists(cache_file):
        print(f"[Level1] Loading global model from disk: {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        _MODEL_CACHE[cache_key] = cached
        return cached

    print(f"[Level1] Training global model on all locations ({model_family}/{profile})...")
    X_all = processed_df.drop(columns=["rain_class"])
    y_all = processed_df["rain_class"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    scale_pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    candidates = get_models_and_params(
        scale_pos_weight=scale_pos_weight,
        model_family=model_family,
        profile=profile,
    )

    best_name, best_val_f1, best_model, best_params_dict = None, -1.0, None, None
    for name, (model, params) in candidates.items():
        fitted = _fit_model(name, model, X_train, y_train, X_val, y_val)
        val_f1 = evaluate_split(fitted, X_val, y_val)["f1"]
        if val_f1 > best_val_f1:
            best_val_f1, best_name, best_model, best_params_dict = val_f1, name, fitted, params

    train_metrics = evaluate_split(best_model, X_train, y_train)
    val_metrics = evaluate_split(best_model, X_val, y_val)
    test_metrics = evaluate_split(best_model, X_test, y_test)

    cached = {
        "model": best_model,
        "modelName": best_name,
        "chosenParams": best_params_dict,
        "trainMetrics": train_metrics,
        "valMetrics": val_metrics,
        "testMetrics": test_metrics,
        "trainingSamples": int(len(X_train)),
    }

    with open(cache_file, "wb") as f:
        pickle.dump(cached, f)
    _MODEL_CACHE[cache_key] = cached
    print(f"[Level1] Global model saved → {cache_file}. Best: {best_name}, Val F1: {best_val_f1:.4f}")
    return cached


# ─────────────────────────────────────────────────────────────────────────────


def load_and_preprocess_data(filepath, return_context=False):
    """Loads dataset and performs initial preprocessing, cleaning, and feature engineering."""
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {filepath}")
        return None
    print(f"Initial Dataset Shape: {df.shape}")

    # Clean column names (remove leading/trailing spaces)
    df.columns = [col.strip() for col in df.columns]

    # Data Cleaning: Drop rows with missing values
    df = df.dropna()
    print(f"After dropna: {df.shape}")

    # Feature Engineering: Parse 'time' to extract month and hour
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    df['hour'] = df['time'].dt.hour

    # Cyclical encoding for hour and month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Target Variable: classify whether it is raining or not
    df['rain_class'] = (df['rain'] > 0.0).astype(int)

    # Categorical Encoding: Encode 'location' feature
    le = LabelEncoder()
    df['location_encoded'] = le.fit_transform(df['location'])
    print(f"Locations encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    context_df = df[['time', 'location', 'rain', 'rain_class']].copy()

    # Drop target-leaking and highly correlated features (based on EDA)
    cols_to_drop = [
        'time', 'location', 'rain', 'wind_speed_10m', 'wind_speed_100m',
        'precipitation', 'snowfall'  # Only drop if present
    ]
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df_processed = df.drop(columns=cols_to_drop)

    # Standard scaling for numerical features (except target and encoded/cyclical/categorical)
    feature_cols = [
        col for col in df_processed.columns
        if col not in ['rain_class', 'location_encoded', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
        and df_processed[col].dtype in [np.float64, np.int64]
    ]
    scaler = StandardScaler()
    df_processed[feature_cols] = scaler.fit_transform(df_processed[feature_cols])

    # Reorder columns: features first, then target
    feature_order = [col for col in df_processed.columns if col != 'rain_class'] + ['rain_class']
    df_processed = df_processed[feature_order]

    if return_context:
        return df_processed.reset_index(drop=True), context_df.reset_index(drop=True)

    return df_processed

def get_models_and_params(scale_pos_weight=4.0, model_family="all", profile="balanced"):
    """Returns model configurations with optional family filtering and profile tuning."""
    if profile == "recall":
        xgb_scale_pos_weight = max(1.0, scale_pos_weight * 1.15)
    elif profile == "precision":
        xgb_scale_pos_weight = max(1.0, scale_pos_weight * 0.85)
    else:
        xgb_scale_pos_weight = max(1.0, scale_pos_weight)

    all_models = {
        "RandomForest": (
            RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                n_jobs=-1,
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=10
            ),
            {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_leaf": 10
            }
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=42,
                scale_pos_weight=xgb_scale_pos_weight,
                eval_metric='logloss',
                n_estimators=400,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1,
                reg_alpha=0.1,
                early_stopping_rounds=30,
                n_jobs=-1
            ),
            {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "max_depth": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1,
                "reg_alpha": 0.1,
                "early_stopping_rounds": 30
            }
        ),
        "LightGBM": (
            LGBMClassifier(
                random_state=42,
                class_weight='balanced',
                verbosity=-1,
                n_estimators=200,
                max_depth=8,
                learning_rate=0.2,
                num_leaves=63
            ),
            {
                "n_estimators": 200,
                "max_depth": 8,
                "learning_rate": 0.2,
                "num_leaves": 63
            }
        )
    }

    family_map = {
        "all": ["RandomForest", "XGBoost", "LightGBM"],
        "rf": ["RandomForest"],
        "xgb": ["XGBoost"],
        "lgbm": ["LightGBM"]
    }

    selected_names = family_map.get(model_family, family_map["all"])
    return {name: all_models[name] for name in selected_names}


def _fit_model(name, model, X_train, y_train, X_val=None, y_val=None):
    if name == "XGBoost" and X_val is not None and y_val is not None and len(X_val) > 0:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)
    return model


def evaluate_split(model, X, y):
    """Returns F1, precision, recall, confusion matrix and classification report for one split."""
    y_pred = model.predict(X)
    return {
        "f1": f1_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "classification_report": classification_report(y, y_pred, digits=4)
    }


def train_and_evaluate(df, model_family="all", profile="balanced"):
    """Splits data (80/10/10), trains model configs, and evaluates on all splits."""
    X = df.drop(columns=['rain_class'])
    y = df['rain_class']

    # 80% train, 10% val, 10% test — stratified to preserve class ratios
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]} samples")

    # Compute class imbalance ratio for XGBoost
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    models_params = get_models_and_params(
        scale_pos_weight=scale_pos_weight,
        model_family=model_family,
        profile=profile
    )

    results = {}
    for name, (model, chosen_params) in models_params.items():
        print(f"\n{'='*50}\nModel: {name}\n{'='*50}")

        model = _fit_model(name, model, X_train, y_train, X_val, y_val)

        results[name] = {
            "model": model,
            "chosen_params": chosen_params,
            "train": evaluate_split(model, X_train, y_train),
            "validation": evaluate_split(model, X_val, y_val),
            "test": evaluate_split(model, X_test, y_test)
        }
        print(f"  Val F1: {results[name]['validation']['f1']:.4f} | "
              f"Test F1: {results[name]['test']['f1']:.4f}")

    return results, X.columns, (len(X_train), len(X_val), len(X_test))


def generate_evaluation_report(results, split_sizes, output_path="model_metrics.txt"):
    """Write a formatted report comparing all models, ranked by Validation F1."""
    ranking = sorted(results.keys(), key=lambda n: results[n]['validation']['f1'], reverse=True)
    best_name = ranking[0]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("RAIN PREDICTION — MODEL COMPARISON REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset Split Sizes:\n"
                f"  Train      : {split_sizes[0]}\n"
                f"  Validation : {split_sizes[1]}\n"
                f"  Test       : {split_sizes[2]}\n\n")

        # ── Per-model detailed results ────────────────────────────────
        for name in ranking:
            r = results[name]
            f.write(f"\n{'=' * 60}\n")
            f.write(f"MODEL: {name}\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"Chosen Hyperparameters : {r['chosen_params']}\n\n")
            for split_key in ["train", "validation", "test"]:
                m = r[split_key]
                label = split_key.capitalize()
                f.write(f"  --- {label} Set ---\n")
                f.write(f"  F1-Score  : {m['f1']:.4f}\n")
                f.write(f"  Precision : {m['precision']:.4f}\n")
                f.write(f"  Recall    : {m['recall']:.4f}\n")
                f.write(f"  Confusion Matrix:\n{m['confusion_matrix']}\n")
                f.write(f"  Classification Report:\n{m['classification_report']}\n")

        # ── Comparison summary table ──────────────────────────────────
        f.write("\n" + "=" * 60 + "\n")
        f.write("MODEL COMPARISON SUMMARY (ranked by Validation F1)\n")
        f.write("=" * 60 + "\n")
        header = f"{'Rank':<6}{'Model':<15}{'Val F1':<10}{'Test F1':<10}{'Val Prec':<12}{'Val Rec':<10}"
        f.write(header + "\n" + "-" * 60 + "\n")
        for i, name in enumerate(ranking, 1):
            r = results[name]
            f.write(
                f"{i:<6}{name:<15}"
                f"{r['validation']['f1']:<10.4f}"
                f"{r['test']['f1']:<10.4f}"
                f"{r['validation']['precision']:<12.4f}"
                f"{r['validation']['recall']:<10.4f}\n"
            )

        # ── Best model section ────────────────────────────────────────
        f.write(f"\n{'=' * 60}\n")
        f.write(f"BEST MODEL: {best_name}\n")
        f.write(f"  Validation F1          : {results[best_name]['validation']['f1']:.4f}\n")
        f.write(f"  Test F1                : {results[best_name]['test']['f1']:.4f}\n")
        f.write(f"  Chosen Hyperparameters : {results[best_name]['chosen_params']}\n")

        # Overfitting check for best model
        f1_train = results[best_name]['train']['f1']
        f1_val = results[best_name]['validation']['f1']
        f.write(f"\n  Overfitting Analysis ({best_name}):\n")
        f.write(f"    Train F1: {f1_train:.4f} | Validation F1: {f1_val:.4f}\n")
        if f1_train - f1_val > 0.10:
            f.write("    WARNING: Model may be overfitting.\n")
        else:
            f.write("    No significant overfitting detected.\n")
        f.write("=" * 60 + "\n")


def get_prediction_options(filepath):
    """Returns available locations and date range for the Level 1 prediction UI."""
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


def predict_rain_for_day(filepath, selected_date, location, model_family="all", profile="balanced"):
    """Predict rain for *selected_date* / *location* using a pre-trained cached model.

    The model is trained once per (dataset, model_family, profile) on ALL
    locations and then saved to .cache/level1/. Subsequent calls reuse the
    cached model instead of retraining.
    """
    processed_df, context_df = load_and_preprocess_data(filepath, return_context=True)
    if processed_df is None:
        raise ValueError(f"Could not load dataset from {filepath}")

    prediction_day = pd.to_datetime(selected_date, errors="coerce")
    if pd.isna(prediction_day):
        raise ValueError(f"Invalid selected date: {selected_date}")
    prediction_day = prediction_day.normalize()

    context_df = context_df.copy()
    context_df["prediction_day"] = context_df["time"].dt.normalize()

    prediction_mask = context_df["prediction_day"].eq(prediction_day)
    if location:
        prediction_mask &= context_df["location"].astype(str).eq(str(location))

    if prediction_mask.sum() == 0:
        raise ValueError(f"No rows found for {location} on {prediction_day.date()}.")

    # Load (or build once) global model for the requested family/profile
    cached = _get_or_build_level1_model(
        filepath,
        processed_df,
        model_family=model_family,
        profile=profile,
    )
    final_model = cached["model"]

    prediction_df = processed_df.loc[prediction_mask].reset_index(drop=True)
    prediction_context = context_df.loc[prediction_mask].reset_index(drop=True)

    X_pred = prediction_df.drop(columns=["rain_class"])
    y_actual = prediction_df["rain_class"]

    hourly_pred = final_model.predict(X_pred)
    if hasattr(final_model, "predict_proba"):
        hourly_prob = final_model.predict_proba(X_pred)[:, 1]
    else:
        hourly_prob = hourly_pred.astype(float)

    hourly_rows = []
    for idx in range(len(prediction_context)):
        hourly_rows.append(
            {
                "time": prediction_context.loc[idx, "time"].isoformat(),
                "predictedRain": bool(hourly_pred[idx]),
                "confidence": float(hourly_prob[idx]),
                "observedRain": bool(y_actual.iloc[idx]),
                "observedRainAmount": float(prediction_context.loc[idx, "rain"]),
            }
        )

    average_confidence = float(np.mean(hourly_prob)) if len(hourly_prob) else 0.0
    rainy_hours = int(np.sum(hourly_pred))
    will_rain = rainy_hours > 0
    observed_rain = bool(y_actual.max())

    return {
        "selectedDate": str(prediction_day.date()),
        "location": str(location),
        "modelName": cached["modelName"],
        "chosenParams": cached["chosenParams"],
        "willRain": will_rain,
        "confidence": average_confidence,
        "rainyHours": rainy_hours,
        "totalHours": int(len(hourly_rows)),
        "observedRain": observed_rain,
        "globalTrainF1": float(cached["trainMetrics"]["f1"]),
        "globalValF1": float(cached["valMetrics"]["f1"]),
        "globalTestF1": float(cached["testMetrics"]["f1"]),
        "trainingSamples": cached["trainingSamples"],
        "hourly": hourly_rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rain classification pipeline and export metrics report.")
    parser.add_argument("--dataset-path", default=None, help="Path to meteorology CSV dataset.")
    parser.add_argument("--model-family", choices=["all", "rf", "xgb", "lgbm"], default="all")
    parser.add_argument("--profile", choices=["balanced", "recall", "precision"], default="balanced")
    parser.add_argument("--output-path", default=None, help="Path to save model metrics report.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dataset_path = os.path.join(script_dir, "..", "data/meteorology_dataset.csv")
    filepath = args.dataset_path if args.dataset_path else default_dataset_path
    report_path = args.output_path if args.output_path else os.path.join(script_dir, "model_metrics.txt")

    processed_data = load_and_preprocess_data(filepath)

    if processed_data is not None:
        results, features, split_sizes = train_and_evaluate(
            processed_data,
            model_family=args.model_family,
            profile=args.profile
        )
        generate_evaluation_report(results, split_sizes, output_path=report_path)
        print(f"\nPipeline complete. Report saved to: {report_path}")
