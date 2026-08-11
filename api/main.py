import pandas as pd

from fastapi import FastAPI

from api.schemas import BookingRequest, PredictionResponse
from src.inference import CancellationPredictor


app = FastAPI(
    title="Hotel Cancellation API",
    version="1.0.0",
)

predictor = CancellationPredictor()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": "v1",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(booking: BookingRequest):
    data = pd.DataFrame(
        [booking.model_dump()]
    )

    return predictor.predict(data)