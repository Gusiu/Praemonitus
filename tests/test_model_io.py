import numpy as np
import pytest
from sklearn.model_selection import FixedThresholdClassifier

from ml import data as D
from ml.predictor import ModelIntegrityError, load_card, load_verified_model
from ml.train import CARD_PATH, MODEL_PATH, build_pipeline, save_model


@pytest.fixture(scope="module")
def small_model():
    train, test = D.split_train_test(D.load_raw())
    X, y = D.features_and_target(train)
    model = FixedThresholdClassifier(build_pipeline("tree", "physics"), threshold=0.3).fit(X, y)
    return model, D.features_and_target(test)[0]


def test_identical_predictions_after_reload(small_model, tmp_path):
    model, X_test = small_model
    card = save_model(model, {"threshold": 0.3}, tmp_path / "m.joblib", tmp_path / "c.json")
    loaded = load_verified_model(tmp_path / "m.joblib", card)
    np.testing.assert_array_equal(model.predict_proba(X_test), loaded.predict_proba(X_test))
    np.testing.assert_array_equal(model.predict(X_test), loaded.predict(X_test))


def test_tampered_file_is_rejected(small_model, tmp_path):
    model, _ = small_model
    path = tmp_path / "m.joblib"
    card = save_model(model, {"threshold": 0.3}, path, tmp_path / "c.json")
    with open(path, "ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ModelIntegrityError):
        load_verified_model(path, card)


def test_shipped_model_matches_card():
    card = load_card(CARD_PATH)
    assert D.sha256_file(MODEL_PATH) == card["model_sha256"]
    assert card["model_id"] == card["model_sha256"][:12]
    assert card["data_sha256"] == D.EXPECTED_SHA256
    model = load_verified_model(MODEL_PATH, card)
    assert model.best_threshold_ == pytest.approx(card["threshold"])
