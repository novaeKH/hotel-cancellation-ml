import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from hotel_booking.data import build_features

MODEL_DIR = Path(__file__).resolve().parents[2] / "models/trained"


class CancellationPredictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        self.model = CatBoostClassifier()
        self.model.load_model(str(model_dir / "late_cancellation.cbm"))

    def predict_batch(self, bookings: pd.DataFrame) -> pd.DataFrame:
        features = build_features(bookings)
        probabilities = self.model.predict_proba(features)[:, 1]
        return pd.DataFrame(
            {"late_cancellation_probability": probabilities}, index=bookings.index
        )

    def predict(self, booking: dict) -> dict:
        probability = self.predict_batch(pd.DataFrame([booking])).iloc[0, 0]
        return {
            "late_cancellation_probability": float(probability),
            "trained_before": self.metadata["trained_before"],
        }
