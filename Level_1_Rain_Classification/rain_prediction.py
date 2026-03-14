import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import os

def load_and_preprocess_data(filepath):
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

    return df_processed

def get_models_and_params(scale_pos_weight=4.0):
    """Returns one fixed hyperparameter configuration per model."""
    return {
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
                scale_pos_weight=scale_pos_weight,
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


def train_and_evaluate(df):
    """Splits data (80/10/10), trains fixed model configs, and evaluates on all splits."""
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
    models_params = get_models_and_params(scale_pos_weight=scale_pos_weight)

    results = {}
    for name, (model, chosen_params) in models_params.items():
        print(f"\n{'='*50}\nModel: {name}\n{'='*50}")

        if name == "XGBoost":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

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


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, "..", "metherology_dataset.csv")

    processed_data = load_and_preprocess_data(filepath)

    if processed_data is not None:
        results, features, split_sizes = train_and_evaluate(processed_data)
        report_path = os.path.join(script_dir, "model_metrics.txt")
        generate_evaluation_report(results, split_sizes, output_path=report_path)
        print(f"\nPipeline complete. Report saved to: {report_path}")
