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

    def predict(self, data: pd.DataFrame) -> dict:
        X = build_features(data)
        X = X[self.input_cols].copy()

        for col in self.categorical_cols:
            X[col] = (
                X[col]
                .astype('object')
                .where(X[col].notna(), np.nan)
            )

        X_enc = self.preprocessor.transform(X)

        probability = self.model.predict_proba(
            X_enc
        )[:,1][0]

        prediction = int(
            probability >= self.threshold
        )

        return {
            'prediction': prediction,
            'cancellation_probability': float(probability),
            'threshold': self.threshold,
            'model_version': 'v1'
        }