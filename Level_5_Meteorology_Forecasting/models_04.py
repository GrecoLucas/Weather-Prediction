"""
models_04.py
------------
Model factory: returns a fast, lightweight model per target.

Imported by 03_walk_forward.py — not meant to be run directly.

Changes vs previous version:
  Temperature : unchanged — LightGBM with num_leaves=63 remains optimal.
  Rain        : tweedie_variance_power 1.5 → 1.3
                Rationale: hourly rain in Portugal is dominated by short,
                intense convective events (80% zeros, P99=3.3mm, max=22.8mm).
                power=1.5 (compound Poisson-Gamma) fits daily totals well;
                power=1.3 (closer to Poisson) better matches the sparse,
                bursty nature of hourly precipitation.
"""

import lightgbm as lgb
import xgboost as xgb

# ---------------------------------------------------------------------------
# Target variable names (2 targets)
# ---------------------------------------------------------------------------
TARGETS = [
    "target_temperature_2m",
    "target_rain",
]

TARGET_SHORT = {
    "target_temperature_2m": "Temperature (C)",
    "target_rain":           "Rain (mm)",
}


def get_model(target: str):
    """
    Return a scikit-learn-compatible, unfitted model for `target`.

    Temperature : LightGBM — captures diurnal cycle and feature interactions
                  efficiently. num_leaves=63 provides enough capacity for
                  temperature extremes without overfitting.

    Rain        : XGBoost Tweedie (power=1.3) — purpose-built for zero-heavy
                  right-skewed distributions. power=1.3 is closer to Poisson
                  and better suited for the bursty, sparse nature of hourly
                  precipitation vs the daily-total use case of power=1.5.
    """
    if target == "target_temperature_2m":
        return lgb.LGBMRegressor(
            n_estimators      = 600,
            learning_rate     = 0.05,
            num_leaves        = 63,
            max_depth         = -1,
            min_child_samples = 20,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            reg_alpha         = 0.05,
            reg_lambda        = 0.1,
            random_state      = 42,
            n_jobs            = -1,
            verbose           = -1,
        )

    elif target == "target_rain":
        return xgb.XGBRegressor(
            n_estimators           = 600,
            learning_rate          = 0.05,
            max_depth              = 6,
            objective              = "reg:tweedie",
            tweedie_variance_power = 1.3,    # CHANGED: 1.5 → 1.3 (better for hourly sparse rain)
            subsample              = 0.8,
            colsample_bytree       = 0.8,
            reg_alpha              = 0.1,
            reg_lambda             = 1.0,
            random_state           = 42,
            n_jobs                 = -1,
            verbosity              = 0,
        )

    else:
        raise ValueError(f"Unknown target: {target}")