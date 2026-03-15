"""
models_04.py
------------
Model factory: returns a fast, lightweight model per target.

Imported by 03_walk_forward.py -- not meant to be run directly.
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
    "target_temperature_2m" : "Temperature (C)",
    "target_rain"           : "Rain (mm)",
}


def get_model(target: str):
    """
    Return a scikit-learn-compatible, unfitted model for `target`.

    Temperature: LightGBM with num_leaves=63 -- strong diurnal cycle,
                 heavy feature interaction, expressive enough for extremes.

    Rain:        XGBoost Tweedie (power=1.5) -- purpose-built for zero-heavy
                 right-skewed distributions. Outperforms HistGBM and two-stage
                 classifiers on sparse rain data (R2 was negative with the
                 LogisticRegression+HistGBM two-stage approach).
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
        # Tweedie regression natively handles the zero-heavy distribution:
        #   - power=1.5 sits between Poisson (1.0) and Gamma (2.0),
        #     matching the compound Poisson-Gamma nature of rainfall.
        #   - Predictions are always >= 0 (no clip needed for sign).
        #   - Avoids the 2.2x false-positive rate of the two-stage classifier.
        return xgb.XGBRegressor(
            n_estimators           = 600,
            learning_rate          = 0.05,
            max_depth              = 6,
            objective              = "reg:tweedie",
            tweedie_variance_power = 1.5,
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