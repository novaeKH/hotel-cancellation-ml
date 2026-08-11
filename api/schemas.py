from pydantic import BaseModel


class BookingRequest(BaseModel):
    hotel: str
    lead_time: int

    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int

    stays_in_weekend_nights: int
    stays_in_week_nights: int

    adults: int
    children: float | None = None
    babies: int

    meal: str
    country: str | None = None
    market_segment: str
    distribution_channel: str

    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int

    reserved_room_type: str
    deposit_type: str

    agent: int | None = None
    company: int | None = None

    customer_type: str
    adr: float

    required_car_parking_spaces: int
    total_of_special_requests: int


class PredictionResponse(BaseModel):
    prediction: int
    cancellation_probability: float
    threshold: float
    model_version: str