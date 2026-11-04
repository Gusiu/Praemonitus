"""Model comparison, threshold tuning, final training and model export.

Run ``python -m ml.train`` to reproduce models/model.joblib and
models/model_card.json deterministically from data/raw/ai4i2020.csv.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    TunedThresholdClassifierCV,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ml import data as D
from ml.evaluate import (
    COST_RATIOS,
    CV_SCORING,
    DEFAULT_COST_RATIO,
    cost_scorer,
    cv_summary,
    decision_scorers,
    metrics_at_threshold,
)
from ml.features import FEATURE_SETS, feature_step, preprocessor

MODELS_DIR = D.PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
CARD_PATH = MODELS_DIR / "model_card.json"

N_SPLITS = 5
N_THRESHOLDS = 200

# Frozen after the cross-validated comparison on the training set (notebook chapter 6).
SELECTED_MODEL = "rf"
SELECTED_FEATURE_SET = "physics"

MODEL_SPECS = {
    "dummy": {
        "label": "Dummy (most frequent)",
        "estimator": DummyClassifier(strategy="most_frequent"),
        "scale": False,
        "grid": {},
        "interpretability": "n/a",
    },
    "logreg": {
        "label": "Logistic regression",
        "estimator": LogisticRegression(class_weight="balanced", max_iter=2000),
        "scale": True,
        "grid": {"clf__C": [0.1, 1.0, 10.0]},
        "interpretability": "high (coefficients)",
    },
    "tree": {
        "label": "Decision tree",
        "estimator": DecisionTreeClassifier(class_weight="balanced", random_state=D.RANDOM_STATE),
        "scale": False,
        "grid": {"clf__max_depth": [3, 4, 5, 6, 8]},
        "interpretability": "high (rules)",
    },
    "rf": {
        "label": "Random forest",
        "estimator": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=D.RANDOM_STATE,
            n_jobs=-1,
        ),
        "scale": False,
        "grid": {},
        "interpretability": "low (ensemble)",
    },
    "hgb": {
        "label": "Hist gradient boosting",
        "estimator": HistGradientBoostingClassifier(random_state=D.RANDOM_STATE),
        "scale": False,
        "grid": {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, None]},
        "interpretability": "low (ensemble)",
    },
}


def cv_splitter() -> StratifiedKFold:
    """The same folds for every model."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=D.RANDOM_STATE)


def build_pipeline(model_key: str, feature_set: str) -> Pipeline:
    spec = MODEL_SPECS[model_key]
    return Pipeline(
        [
            ("features", feature_step(feature_set)),
            ("prep", preprocessor(feature_set, scale=spec["scale"])),
            ("clf", clone(spec["estimator"])),
        ]
    )


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the verified data and split it before anything else."""
    return D.split_train_test(D.load_raw())


def compare_models(train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Grid search with PR-AUC for every model and feature set on the same folds.

    Recall, precision and F1 in this table use the default 0.5 threshold; the
    cost-based threshold is evaluated separately in :func:`compare_thresholds`.
    """
    X, y = D.features_and_target(train_df)
    rows, best = [], {}
    for model_key, spec in MODEL_SPECS.items():
        for feature_set in FEATURE_SETS:
            search = GridSearchCV(
                build_pipeline(model_key, feature_set),
                param_grid=spec["grid"] or [{}],
                scoring=CV_SCORING,
                refit="pr_auc",
                cv=cv_splitter(),
                n_jobs=-1 if model_key != "rf" else 1,
            )
            search.fit(X, y)
            idx = search.best_index_
            params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
            rows.append(
                {
                    "model": model_key,
                    "label": spec["label"],
                    "feature_set": feature_set,
                    "best_params": json.dumps(params),
                    **cv_summary(search.cv_results_, idx, CV_SCORING),
                }
            )
            best[(model_key, feature_set)] = search.best_estimator_
    table = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)
    return table, best


def tuned(estimator, fn_cost: float = DEFAULT_COST_RATIO) -> TunedThresholdClassifierCV:
    """Wrap a pipeline so that the threshold minimises the cost on training folds."""
    return TunedThresholdClassifierCV(
        clone(estimator),
        scoring=cost_scorer(fn_cost),
        cv=cv_splitter(),
        thresholds=N_THRESHOLDS,
        refit=True,
        store_cv_results=True,
    )


def _prediction_latency_ms(estimator, X: pd.DataFrame, repeats: int = 30) -> float:
    row = X.iloc[[0]]
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        estimator.predict_proba(row)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def compare_thresholds(
    train_df: pd.DataFrame, best: dict, fn_cost: float = DEFAULT_COST_RATIO
) -> pd.DataFrame:
    """Nested CV: the outer folds score the model with a threshold tuned on inner folds.

    This gives honest estimates of recall, false alarms and cost at the
    cost-based threshold, plus the spread of thresholds between folds.
    """
    X, y = D.features_and_target(train_df)
    rows = []
    for (model_key, feature_set), estimator in best.items():
        if model_key == "dummy":
            continue
        result = cross_validate(
            tuned(estimator, fn_cost),
            X,
            y,
            cv=cv_splitter(),
            scoring=decision_scorers(fn_cost),
            return_estimator=True,
            n_jobs=1 if model_key == "rf" else -1,
        )
        thresholds = [est.best_threshold_ for est in result["estimator"]]
        row = {
            "model": model_key,
            "label": MODEL_SPECS[model_key]["label"],
            "feature_set": feature_set,
            "threshold_mean": float(np.mean(thresholds)),
            "threshold_std": float(np.std(thresholds)),
        }
        for metric in decision_scorers(fn_cost):
            values = result[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        row["latency_ms"] = _prediction_latency_ms(estimator, X)
        row["interpretability"] = MODEL_SPECS[model_key]["interpretability"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cost_per_1000_mean").reset_index(drop=True)


def fit_final(train_df: pd.DataFrame, best: dict, fn_cost: float = DEFAULT_COST_RATIO):
    """The frozen choice: selected pipeline with a cost-tuned threshold, fitted on all training data."""
    X, y = D.features_and_target(train_df)
    model = tuned(best[(SELECTED_MODEL, SELECTED_FEATURE_SET)], fn_cost)
    model.fit(X, y)
    return model


def cost_sensitivity(train_df: pd.DataFrame, best: dict, ratios=COST_RATIOS) -> tuple[pd.DataFrame, dict]:
    """Threshold and cross-validated cost of the selected model for several FN:FP ratios."""
    X, y = D.features_and_target(train_df)
    rows, models = [], {}
    for ratio in ratios:
        model = tuned(best[(SELECTED_MODEL, SELECTED_FEATURE_SET)], ratio).fit(X, y)
        models[ratio] = model
        rows.append(
            {
                "cost_ratio": f"{ratio}:1",
                "threshold": float(model.best_threshold_),
                "cv_cost_per_1000": float(-model.best_score_),
            }
        )
    return pd.DataFrame(rows), models


def evaluate_on_test(model, test_df: pd.DataFrame, fn_cost: float = DEFAULT_COST_RATIO) -> dict:
    """The single use of the test set, after the model and threshold are frozen."""
    X, y = D.features_and_target(test_df)
    proba = model.predict_proba(X)[:, 1]
    return metrics_at_threshold(y, proba, model.best_threshold_, fn_cost=fn_cost)


def training_ranges(train_df: pd.DataFrame) -> dict:
    return {
        col: {"min": float(train_df[col].min()), "max": float(train_df[col].max())}
        for col in D.NUMERIC_FEATURES
    }


def save_model(model, card: dict, model_path: Path = MODEL_PATH, card_path: Path = CARD_PATH) -> dict:
    """Write the estimator and its card; the card stores the SHA-256 of the model file."""
    model_path = Path(model_path)
    card_path = Path(card_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path, compress=3)
    file_hash = D.sha256_file(model_path)
    card = {"model_id": file_hash[:12], **card, "model_sha256": file_hash}
    card_path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    return card


def build_card(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cv_row: dict,
    threshold_row: dict,
    test_metrics: dict,
    sensitivity: pd.DataFrame,
) -> dict:
    return {
        "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": MODEL_SPECS[SELECTED_MODEL]["label"],
        "model_key": SELECTED_MODEL,
        "feature_set": SELECTED_FEATURE_SET,
        "best_params": json.loads(cv_row["best_params"]),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "data_sha256": D.EXPECTED_SHA256,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "input_features": D.INPUT_FEATURES,
        "model_features": FEATURE_SETS[SELECTED_FEATURE_SET] + ["type"],
        "training_ranges": training_ranges(train_df),
        "threshold": float(model.best_threshold_),
        "cost_ratio_fn_fp": DEFAULT_COST_RATIO,
        "cost_sensitivity": sensitivity.to_dict(orient="records"),
        "cv_metrics": {
            "pr_auc_mean": cv_row["pr_auc_mean"],
            "pr_auc_std": cv_row["pr_auc_std"],
            "roc_auc_mean": cv_row["roc_auc_mean"],
            "recall_at_threshold_mean": threshold_row["recall_mean"],
            "precision_at_threshold_mean": threshold_row["precision_mean"],
            "fp_per_1000_mean": threshold_row["fp_per_1000_mean"],
            "cost_per_1000_mean": threshold_row["cost_per_1000_mean"],
        },
        "test_metrics": test_metrics,
    }


def run(verbose: bool = True) -> dict:
    """Full reproducible pipeline: split, compare, tune, fit, test once, save."""
    train_df, test_df = load_split()
    cv_table, best = compare_models(train_df)
    threshold_table = compare_thresholds(train_df, best)
    model = fit_final(train_df, best)
    sensitivity, _ = cost_sensitivity(train_df, best)
    test_metrics = evaluate_on_test(model, test_df)

    def pick(table):
        mask = (table["model"] == SELECTED_MODEL) & (table["feature_set"] == SELECTED_FEATURE_SET)
        return table[mask].iloc[0].to_dict()

    card = build_card(
        model, train_df, test_df, pick(cv_table), pick(threshold_table), test_metrics, sensitivity
    )
    card = save_model(model, card)
    if verbose:
        pd.set_option("display.width", 200)
        print(cv_table[["label", "feature_set", "pr_auc_mean", "pr_auc_std", "roc_auc_mean"]])
        print(threshold_table[["label", "feature_set", "threshold_mean", "recall_mean",
                               "fp_per_1000_mean", "cost_per_1000_mean"]])
        print(sensitivity)
        print(json.dumps(test_metrics, indent=2))
        print(f"Saved {MODEL_PATH} (model_id {card['model_id']})")
    return {
        "train_df": train_df,
        "test_df": test_df,
        "cv_table": cv_table,
        "best": best,
        "threshold_table": threshold_table,
        "model": model,
        "sensitivity": sensitivity,
        "test_metrics": test_metrics,
        "card": card,
    }


if __name__ == "__main__":
    run()
