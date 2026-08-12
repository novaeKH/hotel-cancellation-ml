import json
from pathlib import Path

from catboost import CatBoostClassifier
import pandas as pd

from src.features import build_features

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"


class CancellationPredictor:
    def __init__(self):
        with open(
            MODEL_DIR / "model_metadata.json",
            "r",
            encoding="utf-8"
        ) as file:
            metadata = json.load(file)

        self.threshold = float(metadata["threshold"])
        self.feature_cols = metadata["feature_cols"]
        self.categorical_cols = metadata["categorical_cols"]
        self.model_version = metadata["version"]

        self.model = CatBoostClassifier()
        self.model.load_model(
            MODEL_DIR / "catboost_v2.cbm"
        )

    def predict(self, data: pd.DataFrame) -> dict:
        if len(data) != 1:
            raise ValueError(
                "Для одного запроса нужна одна строка данных"
            )

        result = self.predict_batch(data).iloc[0]

        return {
            "prediction": int(result["prediction"]),
            "cancellation_probability": float(
                result["cancellation_probability"]
            ),
            "threshold": self.threshold,
            "model_version": self.model_version,
        }

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        features = build_features(data)

        missing_cols = [
            col for col in self.feature_cols
            if col not in features.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Не хватает признаков: {missing_cols}"
            )

        features = features[self.feature_cols].copy()

        for col in self.categorical_cols:
            features[col] = (
                features[col]
                .astype("string")
                .fillna("Missing")
            )

        return features

    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        features = self.prepare_data(data)
        probabilities = self.model.predict_proba(features)[:, 1]

        predictions = (
            probabilities >= self.threshold
        ).astype(int)

        return pd.DataFrame(
            {
                "prediction": predictions,
                "cancellation_probability": probabilities,
                "model_version": self.model_version,
            },
            index=data.index,
        )
