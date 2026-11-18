from datetime import date

from pydantic import BaseModel, Field


class BookingRequest(BaseModel):
    hotel: str
    lead_time: int = Field(ge=1)
    arrival_date: date
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: float | None = None
    babies: int
    meal: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    reserved_room_type: str
    agent: int | None = None
    company: int | None = None
    customer_type: str


class PredictionResponse(BaseModel):
    late_cancellation_probability: float
    trained_before: str
