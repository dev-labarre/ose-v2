"""
Model configuration presets for controlled rebalancing experiments.
"""

from typing import Dict

# Baseline mirrors previous defaults (no explicit tilt)
BASELINE: Dict = {
    "name": "baseline",
    "model_params": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "colsample_bytree": 1.0,
        "colsample_bylevel": 1.0,
        "subsample": 1.0,
        "min_split_loss": 0.0,
        "lambda": 1.0,
        "alpha": 0.0,
    },
}

# Config A: column sampling tilt to expose signal/text slightly more often
COL_TILT_MILD: Dict = {
    "name": "col_tilt_mild",
    "model_params": {
        "n_estimators": 220,
        "max_depth": 6,
        "learning_rate": 0.09,
        "colsample_bytree": 0.7,   # more feature subsampling
        "colsample_bylevel": 0.8,
        "subsample": 0.85,
    },
}

# Config B: mild regularization on financial splits (gamma/lambda/alpha)
FIN_PEN_MILD: Dict = {
    "name": "fin_pen_mild",
    "model_params": {
        "n_estimators": 220,
        "max_depth": 6,
        "learning_rate": 0.09,
        "colsample_bytree": 0.75,
        "colsample_bylevel": 0.85,
        "subsample": 0.85,
        "min_split_loss": 0.05,  # gamma
        "lambda": 1.5,
        "alpha": 0.2,            # L1
    },
}

# Config C: depth/leaf softness to avoid deep financial splits
DEPTH_SOFT: Dict = {
    "name": "depth_soft",
    "model_params": {
        "n_estimators": 240,
        "max_depth": 5,
        "learning_rate": 0.09,
        "colsample_bytree": 0.75,
        "colsample_bylevel": 0.85,
        "subsample": 0.8,
        "min_child_weight": 3.0,
        "min_split_loss": 0.04,
    },
}

PRESETS = {
    BASELINE["name"]: BASELINE,
    COL_TILT_MILD["name"]: COL_TILT_MILD,
    FIN_PEN_MILD["name"]: FIN_PEN_MILD,
    DEPTH_SOFT["name"]: DEPTH_SOFT,
}

