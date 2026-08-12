import json 
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import build_features

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / 'models'

class CancellationPredictor:
    def __init__(self):
        self.preprocessor = joblib.load(
            MODEL_DIR / 'preprocessor.joblib'
        )

        self.input_cols = list(
            self.preprocessor.feature_names_in_
        )

        self.model = joblib.load(
            MODEL_DIR / 'lightgbm_v1.joblib'
        )

        with open(
            MODEL_DIR / 'model_metadata.json',
            'r',
            encoding = 'utf-8'
        ) as f:
            self.metadata = json.load(f)

        self.threshold = float(
            self.metadata['threshold']
        )

        self.categorical_cols = self.metadata[
            'categorical_cols'
        ]

        self.drop_cols = self.metadata[
            'drop_cols'
        ]

    def predict(self, data: pd.DataFrame) -> dict: # Если объект один
        if len(data) != 1:
            raise ValueError(
                "predict() выдает только одно предсказание"
            )

        result = self.predict_batch(data).iloc[0]

        return {
            "prediction": int(result["prediction"]),
            "cancellation_probability": float(
                result["cancellation_probability"]
            ),
            "threshold": self.threshold,
            "model_version": "v1",
        }

    def _prepare_input(self, data: pd.DataFrame) -> pd.DataFrame:
        X = build_features(data)

        missing_cols = [
            col for col in self.input_cols
            if col not in X.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Отсутствуют необходимые columns: {missing_cols}"
            )

        X = X[self.input_cols].copy()

        for col in self.categorical_cols:
            X[col] = (
                X[col]
                .astype("object")
                .where(X[col].notna(), np.nan)
            )

        return X

    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:  # Если объектов много
        X = self._prepare_input(data)

        X_enc = self.preprocessor.transform(X)

        probabilities = self.model.predict_proba(
            X_enc
        )[:, 1]

        predictions = (
            probabilities >= self.threshold
        ).astype(int)

        return pd.DataFrame(
            {
                "prediction": predictions,
                "cancellation_probability": probabilities,
                "model_version": "v1",
            },
            index=data.index,
        )