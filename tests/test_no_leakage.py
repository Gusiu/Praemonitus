import pandas as pd
import pytest

from ml import data as D
from ml.features import FEATURE_SETS
from ml.train import MODEL_SPECS, build_pipeline

FORBIDDEN = set(D.ID_COLUMNS) | set(D.FAILURE_MODE_COLUMNS) | {D.TARGET}


def test_feature_lists_have_no_ids_or_flags():
    assert FORBIDDEN.isdisjoint(D.INPUT_FEATURES)
    for columns in FEATURE_SETS.values():
        assert FORBIDDEN.isdisjoint(columns)


@pytest.mark.parametrize("model_key", list(MODEL_SPECS))
@pytest.mark.parametrize("feature_set", list(FEATURE_SETS))
def test_preprocessing_is_inside_pipeline(model_key, feature_set):
    pipe = build_pipeline(model_key, feature_set)
    assert [name for name, _ in pipe.steps] == ["features", "prep", "clf"]


def test_pipeline_ignores_forbidden_columns():
    df = D.load_raw().sample(400, random_state=0)
    pipe = build_pipeline("tree", "physics").fit(df, df[D.TARGET])
    used = set(pipe[:-1].get_feature_names_out())
    assert FORBIDDEN.isdisjoint(used)
    shuffled = df.copy()
    for col in FORBIDDEN - {D.TARGET}:
        shuffled[col] = shuffled[col].sample(frac=1, random_state=1).values
    pd.testing.assert_series_equal(
        pd.Series(pipe.predict_proba(df)[:, 1]), pd.Series(pipe.predict_proba(shuffled)[:, 1])
    )


def test_scaler_fitted_on_training_folds_only():
    df = D.load_raw()
    train, test = D.split_train_test(df)
    pipe = build_pipeline("logreg", "raw").fit(*D.features_and_target(train))
    scaler = pipe.named_steps["prep"].named_transformers_["num"]
    assert scaler.mean_ == pytest.approx(train[D.NUMERIC_FEATURES].mean().values)
