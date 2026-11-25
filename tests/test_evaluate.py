import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from ml import evaluate as E

# Eight runs; at threshold 0.5: TN 4, FP 1 (p=0.6), FN 1 (p=0.2), TP 2.
Y_TRUE = [0, 0, 0, 1, 1, 0, 1, 0]
PROBA = [0.1, 0.4, 0.35, 0.8, 0.2, 0.6, 0.9, 0.05]


def test_metrics_at_threshold_hand_computed():
    m = E.metrics_at_threshold(Y_TRUE, PROBA, 0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (4, 1, 1, 2)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(0.75)
    # 12 of the 15 (failure, non-failure) pairs are ranked correctly.
    assert m["roc_auc"] == pytest.approx(0.8)
    # Failures at ranks 1, 2 and 6 by probability: (1 + 1 + 3/6) / 3.
    assert m["pr_auc"] == pytest.approx(2.5 / 3)
    assert m["fp_per_1000"] == pytest.approx(125.0)
    assert m["cost"] == pytest.approx(10 * 1 + 1)
    assert m["cost_per_1000"] == pytest.approx(11 / 8 * 1000)


def test_probability_equal_to_threshold_is_a_failure():
    m = E.metrics_at_threshold(Y_TRUE, PROBA, 0.8)
    assert (m["tp"], m["fn"], m["fp"]) == (2, 1, 0)


def test_ranking_metrics_need_both_classes():
    m = E.metrics_at_threshold([0, 0, 0], [0.1, 0.7, 0.2], 0.5)
    assert math.isnan(m["pr_auc"])
    assert math.isnan(m["roc_auc"])
    assert (m["fp"], m["cost"]) == (1, 1.0)


def test_cost_functions():
    y_true = [0, 1, 1, 0]
    y_pred = [1, 0, 1, 0]  # one false alarm and one missed failure
    assert E.total_cost(y_true, y_pred) == 11.0
    assert E.total_cost(y_true, y_pred, fn_cost=5, fp_cost=2) == 7.0
    assert E.cost_per_1000(y_true, y_pred) == pytest.approx(2750.0)


def test_cost_scorer_is_negated_cost_per_1000():
    X = np.zeros((4, 1))
    y = np.array([0, 1, 1, 0])
    always_failure = DummyClassifier(strategy="constant", constant=1).fit(X, y)
    # Two false alarms in four runs: cost 2, that is 500 per 1000 runs.
    assert E.cost_scorer()(always_failure, X, y) == pytest.approx(-500.0)


def test_error_breakdown_counts_failures_per_mode():
    y_true = [1, 1, 1, 1, 1, 0, 0]
    y_pred = [1, 0, 1, 1, 0, 1, 0]
    cause = pd.Series(["HDF", "HDF", "PWF", "OSF", "no flag", "none", "none"])
    table = E.error_breakdown(y_true, y_pred, cause)
    assert list(table.index) == ["HDF", "PWF", "OSF", "no flag", "all failures"]
    assert list(table.columns) == ["failures", "detected", "missed", "recall"]
    assert table.loc["HDF"].tolist() == [2, 1, 1, 0.5]
    assert table.loc["no flag"].tolist() == [1, 0, 1, 0.0]
    assert table.loc["all failures"].tolist() == [5, 3, 2, 0.6]


def test_plots_are_saved_as_png(tmp_path):
    y_true = np.array(Y_TRUE)
    y_pred = (np.array(PROBA) >= 0.5).astype(int)
    figures = {
        "confusion.png": E.plot_confusion(y_true, y_pred, "Confusion matrix"),
        "pr.png": E.plot_pr_curves(y_true, {"model": PROBA, "constant": [0.5] * 8}, "PR"),
    }
    for name, fig in figures.items():
        path = tmp_path / "figures" / name
        E.save_figure(fig, path)
        plt.close(fig)
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
