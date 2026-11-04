"""Loading, quality checks and the train/test split for the AI4I 2020 dataset.

Column renaming happens here and nowhere else, so every other module works
with the snake_case names.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
EXPECTED_SHA256 = "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e"
EXPECTED_ROWS = 10_000

RANDOM_STATE = 42
TEST_SIZE = 0.2

RAW_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

RENAME_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

TARGET = "machine_failure"
ID_COLUMNS = ["udi", "product_id"]
FAILURE_MODE_COLUMNS = ["twf", "hdf", "pwf", "osf", "rnf"]
CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
INPUT_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
PRODUCT_TYPES = ("L", "M", "H")

# Thresholds published in the UCI dataset description.
HDF_TEMP_DIFF_K = 8.6
HDF_SPEED_RPM = 1380
PWF_POWER_MIN_W = 3500
PWF_POWER_MAX_W = 9000
OSF_STRAIN_LIMITS = {"L": 11_000, "M": 12_000, "H": 13_000}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: str | Path = DATA_PATH, verify_checksum: bool = True) -> pd.DataFrame:
    """Read the original CSV, verify its checksum and rename columns to snake_case."""
    path = Path(path)
    if verify_checksum:
        actual = sha256_file(path)
        if actual != EXPECTED_SHA256:
            raise ValueError(
                f"Checksum mismatch for {path}: expected {EXPECTED_SHA256}, got {actual}"
            )
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df[RAW_COLUMNS].rename(columns=RENAME_MAP)


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split. This is the first operation after the quality checks.

    The returned frames keep every column (including identifiers and failure-mode
    flags) so that error analysis can use the flags later; feature selection
    happens inside the model pipeline.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )
    return train_df, test_df


def features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return only the model inputs and the label."""
    return df[INPUT_FEATURES].copy(), df[TARGET].copy()


def column_dictionary() -> pd.DataFrame:
    """Data dictionary: name, type, unit, description and role of each column."""
    rows = [
        ("UDI", "udi", "int", "-", "Row identifier (1-10000)", "identifier"),
        ("Product ID", "product_id", "str", "-",
         "Variant letter followed by a serial number", "identifier"),
        ("Type", "type", "category", "-",
         "Product quality variant: L (low, 50%), M (medium, 30%), H (high, 20%)", "feature"),
        ("Air temperature [K]", "air_temp_k", "float", "K",
         "Ambient air temperature", "feature"),
        ("Process temperature [K]", "process_temp_k", "float", "K",
         "Process temperature", "feature"),
        ("Rotational speed [rpm]", "rotational_speed_rpm", "int", "rpm",
         "Spindle rotational speed", "feature"),
        ("Torque [Nm]", "torque_nm", "float", "Nm", "Torque", "feature"),
        ("Tool wear [min]", "tool_wear_min", "int", "min",
         "Accumulated tool usage time", "feature"),
        ("Machine failure", "machine_failure", "int (0/1)", "-",
         "1 if the run ended with a machine failure", "label"),
        ("TWF", "twf", "int (0/1)", "-", "Tool wear failure flag", "excluded (defines label)"),
        ("HDF", "hdf", "int (0/1)", "-", "Heat dissipation failure flag",
         "excluded (defines label)"),
        ("PWF", "pwf", "int (0/1)", "-", "Power failure flag", "excluded (defines label)"),
        ("OSF", "osf", "int (0/1)", "-", "Overstrain failure flag", "excluded (defines label)"),
        ("RNF", "rnf", "int (0/1)", "-", "Random failure flag", "excluded (defines label)"),
    ]
    return pd.DataFrame(
        rows, columns=["original_name", "name", "dtype", "unit", "description", "role"]
    )


def quality_summary(df: pd.DataFrame) -> dict:
    """Basic data quality facts used in the notebook."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rows_without_ids": int(df.drop(columns=ID_COLUMNS).duplicated().sum()),
        "udi_unique": bool(df["udi"].is_unique),
        "product_id_unique": bool(df["product_id"].is_unique),
        "product_id_prefix_matches_type": bool(
            (df["product_id"].str[0] == df["type"]).all()
        ),
        "failures": int(df[TARGET].sum()),
        "failure_rate": float(df[TARGET].mean()),
        "always_no_failure_accuracy": float(1 - df[TARGET].mean()),
    }


def rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the deterministic failure-mode rules (HDF, PWF, OSF) from the inputs."""
    temp_diff = df["process_temp_k"] - df["air_temp_k"]
    power = df["torque_nm"] * df["rotational_speed_rpm"] * 2 * np.pi / 60
    strain = df["tool_wear_min"] * df["torque_nm"]
    strain_limit = df["type"].map(OSF_STRAIN_LIMITS)
    return pd.DataFrame(
        {
            "hdf_rule": ((temp_diff < HDF_TEMP_DIFF_K)
                         & (df["rotational_speed_rpm"] < HDF_SPEED_RPM)).astype(int),
            "pwf_rule": ((power < PWF_POWER_MIN_W) | (power > PWF_POWER_MAX_W)).astype(int),
            "osf_rule": (strain > strain_limit).astype(int),
        },
        index=df.index,
    )


def label_consistency(df: pd.DataFrame) -> dict:
    """Compare the label with the failure-mode flags and with the published rules."""
    flags = df[FAILURE_MODE_COLUMNS]
    any_flag = flags.any(axis=1)
    failure = df[TARGET] == 1
    rules = rule_flags(df)
    explained_by_rules = failure & rules.any(axis=1)
    unexplained = failure & ~rules.any(axis=1)
    return {
        "flag_counts": {c: int(df[c].sum()) for c in FAILURE_MODE_COLUMNS},
        "flag_counts_with_failure": {
            c: int((df[c].astype(bool) & failure).sum()) for c in FAILURE_MODE_COLUMNS
        },
        "rule_agreement": {
            mode: int((rules[f"{mode}_rule"] == df[mode]).sum()) for mode in ("hdf", "pwf", "osf")
        },
        "failures_explained_by_rules": int(explained_by_rules.sum()),
        "failures_not_explained_by_rules": int(unexplained.sum()),
        "unexplained_with_twf": int((unexplained & (df["twf"] == 1)).sum()),
        "unexplained_without_any_flag": int((unexplained & ~any_flag).sum()),
        "failures_without_any_flag": int((failure & ~any_flag).sum()),
        "flags_without_failure": int((~failure & any_flag).sum()),
        "rnf_without_failure": int(((df["rnf"] == 1) & ~failure).sum()),
        "tool_wear_range_at_twf": (
            int(df.loc[df["twf"] == 1, "tool_wear_min"].min()),
            int(df.loc[df["twf"] == 1, "tool_wear_min"].max()),
        ),
    }


def failure_cause(df: pd.DataFrame) -> pd.Series:
    """Single label per row describing which failure mode explains it (for error analysis).

    Priority: deterministic modes (HDF, PWF, OSF) first, then TWF, then RNF,
    then 'no flag' for failures with no mode at all. Non-failures get 'none'.
    """
    cause = pd.Series("none", index=df.index, dtype=object)
    failure = df[TARGET] == 1
    cause[failure] = "no flag"
    for mode in ("rnf", "twf", "osf", "pwf", "hdf"):
        cause[failure & (df[mode] == 1)] = mode.upper()
    return cause
