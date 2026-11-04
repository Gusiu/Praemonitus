"""Metrics, the cost-based scorer, plots and error analysis."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import export_text  # noqa: E402

DEFAULT_COST_RATIO = 10
COST_RATIOS = (5, 10, 20)


def total_cost(y_true, y_pred, fn_cost: float = DEFAULT_COST_RATIO, fp_cost: float = 1.0) -> float:
    """Cost of a set of decisions: fn_cost per missed failure, fp_cost per false alarm."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn_cost * fn + fp_cost * fp)


def cost_per_1000(y_true, y_pred, fn_cost: float = DEFAULT_COST_RATIO) -> float:
    """Cost normalised to 1000 runs, so folds of different size are comparable."""
    return total_cost(y_true, y_pred, fn_cost=fn_cost) / len(y_true) * 1000


def cost_scorer(fn_cost: float = DEFAULT_COST_RATIO):
    """Scorer for TunedThresholdClassifierCV; sklearn maximises, so cost is negated."""
    return make_scorer(cost_per_1000, greater_is_better=False, fn_cost=fn_cost)


def _fp_per_1000(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(((y_pred == 1) & (y_true == 0)).sum() / len(y_true) * 1000)


def decision_scorers(fn_cost: float = DEFAULT_COST_RATIO) -> dict:
    """Scorers evaluated on hard decisions (used for the tuned-threshold models)."""
    return {
        "recall": make_scorer(recall_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "fp_per_1000": make_scorer(_fp_per_1000),
        "cost_per_1000": make_scorer(cost_per_1000, fn_cost=fn_cost),
    }


CV_SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "accuracy": "accuracy",
}


def metrics_at_threshold(y_true, proba, threshold: float, fn_cost: float = DEFAULT_COST_RATIO) -> dict:
    """All reported metrics for one set of probabilities and one threshold."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, proba)) if has_both else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, proba)) if has_both else float("nan"),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fp_per_1000": float(fp / len(y_true) * 1000),
        "cost": float(fn_cost * fn + fp),
        "cost_per_1000": float((fn_cost * fn + fp) / len(y_true) * 1000),
    }


def cv_summary(cv_results: dict, index: int, metrics) -> dict:
    """Mean and standard deviation over folds for one GridSearchCV candidate."""
    out = {}
    for metric in metrics:
        out[f"{metric}_mean"] = float(cv_results[f"mean_test_{metric}"][index])
        out[f"{metric}_std"] = float(cv_results[f"std_test_{metric}"][index])
    return out


def error_breakdown(y_true, y_pred, cause: pd.Series) -> pd.DataFrame:
    """Detected and missed failures per failure mode, plus false alarms."""
    frame = pd.DataFrame(
        {"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred), "cause": cause.values}
    )
    failures = frame[frame["y_true"] == 1]
    table = (
        failures.groupby("cause")
        .agg(failures=("y_true", "size"), detected=("y_pred", "sum"))
        .assign(missed=lambda t: t["failures"] - t["detected"])
        .assign(recall=lambda t: t["detected"] / t["failures"])
    )
    order = [c for c in ["HDF", "PWF", "OSF", "TWF", "RNF", "no flag"] if c in table.index]
    table = table.loc[order]
    table.loc["all failures"] = [
        len(failures),
        int(failures["y_pred"].sum()),
        int(len(failures) - failures["y_pred"].sum()),
        float(failures["y_pred"].mean()),
    ]
    table[["failures", "detected", "missed"]] = table[["failures", "detected", "missed"]].astype(int)
    return table


def tree_thresholds(tree, feature_names) -> pd.DataFrame:
    """All split thresholds of a fitted decision tree, grouped by feature."""
    inner = tree.tree_
    rows = [
        (feature_names[f], float(t))
        for f, t in zip(inner.feature, inner.threshold)
        if f >= 0
    ]
    return pd.DataFrame(rows, columns=["feature", "threshold"]).sort_values(["feature", "threshold"])


def tree_rules_text(tree, feature_names, decimals: int = 1) -> str:
    return export_text(tree, feature_names=list(feature_names), decimals=decimals)


def parse_rule_thresholds(text: str) -> list[tuple[str, float]]:
    """Extract (feature, threshold) pairs from export_text output."""
    pattern = re.compile(r"([a-z_]+)\s*(?:<=|>)\s*(-?[0-9.]+)")
    return [(m.group(1), float(m.group(2))) for m in pattern.finditer(text)]


def save_figure(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")


def plot_confusion(y_true, y_pred, title: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["no failure", "failure"], ax=ax, colorbar=False
    )
    ax.set_title(title)
    return fig


def plot_pr_curves(y_true, probas: dict, title: str):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, proba in probas.items():
        PrecisionRecallDisplay.from_predictions(y_true, proba, name=name, ax=ax)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return fig
