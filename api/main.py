from fastapi import FastAPI
from hotel_booking.inference import CancellationPredictor

from api.schemas import BookingRequest, PredictionResponse

app = FastAPI(title="Late Cancellation Demo", version="3.0.0")
predictor = CancellationPredictor()


@app.get("/health")
def health():
    return {"status": "ok", "trained_before": predictor.metadata["trained_before"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(booking: BookingRequest):
    return predictor.predict(booking.model_dump())
