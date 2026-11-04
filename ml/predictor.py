"""Shared prediction core for the Streamlit app and the console interface."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import pandas as pd

import database
from ml.data import INPUT_FEATURES, sha256_file
from ml.features import compute_derived
from ml.validation import ProcessInput, out_of_range_fields, validate_input

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
CARD_PATH = PROJECT_ROOT / "models" / "model_card.json"


class ModelIntegrityError(RuntimeError):
    """The model file does not match the checksum stored in the model card."""


@dataclass
class PredictionResult:
    probability: float
    threshold: float
    prediction: int
    out_of_range: bool
    out_of_range_fields: list[str]
    derived: dict
    model_id: str
    latency_ms: float
    inputs: dict = field(default_factory=dict)
    record_id: int | None = None

    @property
    def label(self) -> str:
        return "FAILURE" if self.prediction == 1 else "no failure"

    def to_dict(self) -> dict:
        return asdict(self)


def load_card(card_path: str | Path = CARD_PATH) -> dict:
    return json.loads(Path(card_path).read_text(encoding="utf-8"))


def load_verified_model(model_path: str | Path = MODEL_PATH, card: dict | None = None):
    """Compare the file hash with the card first, then unpickle.

    joblib files are pickles and can execute code when loaded, so only our own,
    verified file is ever loaded.
    """
    card = card if card is not None else load_card()
    actual = sha256_file(model_path)
    if actual != card["model_sha256"]:
        raise ModelIntegrityError(
            f"SHA-256 of {model_path} ({actual}) does not match the model card "
            f"({card['model_sha256']}); refusing to load."
        )
    return joblib.load(model_path)


class Predictor:
    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        card_path: str | Path = CARD_PATH,
        db_path: str | Path | None = None,
    ):
        self.card = load_card(card_path)
        self.model = load_verified_model(model_path, self.card)
        self.model_id = self.card["model_id"]
        self.threshold = float(self.card["threshold"])
        self.ranges = self.card["training_ranges"]
        self._db_path = db_path

    @property
    def db_path(self):
        """Explicit path, or the database default resolved at call time."""
        return self._db_path if self._db_path is not None else database.DEFAULT_DB_PATH

    def _predict_validated(self, values: ProcessInput) -> tuple[float, dict, list[str], float]:
        start = time.perf_counter()
        frame = pd.DataFrame([values.model_dump()])[INPUT_FEATURES]
        probability = float(self.model.predict_proba(frame)[0, 1])
        latency_ms = (time.perf_counter() - start) * 1000
        derived = {k: float(v) for k, v in compute_derived(frame).iloc[0].items()}
        flagged = out_of_range_fields(values, self.ranges)
        return probability, derived, flagged, latency_ms

    def predict(self, raw: dict, source: str = "cli", save: bool = True) -> PredictionResult:
        """Validate, predict and (by default) log the call. Raises InputError on bad input."""
        values = validate_input(raw)
        probability, derived, flagged, latency_ms = self._predict_validated(values)
        result = PredictionResult(
            probability=probability,
            threshold=self.threshold,
            prediction=int(probability >= self.threshold),
            out_of_range=bool(flagged),
            out_of_range_fields=flagged,
            derived=derived,
            model_id=self.model_id,
            latency_ms=latency_ms,
            inputs=values.model_dump(),
        )
        if save:
            result.record_id = database.save_prediction(
                model_id=self.model_id,
                source=source,
                inputs=result.inputs,
                probability=result.probability,
                threshold=result.threshold,
                prediction=result.prediction,
                out_of_range=result.out_of_range,
                latency_ms=result.latency_ms,
                db_path=self.db_path,
            )
        return result
