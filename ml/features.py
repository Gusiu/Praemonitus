"""Feature sets and preprocessing. Everything here runs inside the model Pipeline,
so training and the application compute derived features with the same code."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml.data import CATEGORICAL_FEATURES, INPUT_FEATURES, NUMERIC_FEATURES, PRODUCT_TYPES

DERIVED_FEATURES = ["temp_diff_k", "power_w", "strain_min_nm"]
FEATURE_SETS = {
    "raw": NUMERIC_FEATURES,
    "physics": NUMERIC_FEATURES + DERIVED_FEATURES,
}


def compute_derived(X: pd.DataFrame) -> pd.DataFrame:
    """Derived physical quantities, computed only from values known at prediction time.

    temp_diff_k   = process temperature - air temperature (heat dissipation margin)
    power_w       = torque [Nm] * angular speed [rad/s] = torque * rpm * 2*pi / 60
    strain_min_nm = tool wear [min] * torque [Nm] (tool overstrain)
    """
    return pd.DataFrame(
        {
            "temp_diff_k": X["process_temp_k"] - X["air_temp_k"],
            "power_w": X["torque_nm"] * X["rotational_speed_rpm"] * 2 * np.pi / 60,
            "strain_min_nm": X["tool_wear_min"] * X["torque_nm"],
        },
        index=X.index,
    )


def select_raw(X: pd.DataFrame) -> pd.DataFrame:
    """Keep only the input features (drops identifiers or anything else passed in)."""
    return X[INPUT_FEATURES].copy()


def add_physics_features(X: pd.DataFrame) -> pd.DataFrame:
    """Input features plus the derived physical features."""
    base = select_raw(X)
    return pd.concat([base, compute_derived(base)], axis=1)


def _raw_names(transformer, input_features):
    return np.asarray(INPUT_FEATURES, dtype=object)


def _physics_names(transformer, input_features):
    return np.asarray(INPUT_FEATURES + DERIVED_FEATURES, dtype=object)


def feature_step(feature_set: str) -> FunctionTransformer:
    """First Pipeline step: select inputs and (optionally) compute derived features."""
    if feature_set == "raw":
        return FunctionTransformer(select_raw, feature_names_out=_raw_names)
    if feature_set == "physics":
        return FunctionTransformer(add_physics_features, feature_names_out=_physics_names)
    raise ValueError(f"Unknown feature set: {feature_set}")


def preprocessor(feature_set: str, scale: bool) -> ColumnTransformer:
    """One-hot encode the product type; scale numeric columns only when the model needs it."""
    numeric = FEATURE_SETS[feature_set]
    numeric_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric_step, numeric),
            (
                "cat",
                OneHotEncoder(categories=[list(PRODUCT_TYPES)], handle_unknown="error"),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
