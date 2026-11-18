from pathlib import Path

import numpy as np
import pandas as pd

MONTHS = {
    name: number
    for number, name in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        1,
    )
}
BASE_COLUMNS = [
    "hotel",
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "reserved_room_type",
    "agent",
    "company",
    "customer_type",
]


def arrival_dates(data: pd.DataFrame) -> pd.Series:
    if "arrival_date" in data:
        return pd.to_datetime(data["arrival_date"])
    return pd.to_datetime(
        {
            "year": data["arrival_date_year"],
            "month": data["arrival_date_month"].map(MONTHS),
            "day": data["arrival_date_day_of_month"],
        }
    )


def load_bookings(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    data = raw.copy()
    data["booking_id"] = np.arange(len(data))
    data["arrival_date"] = arrival_dates(data)
    data["status_date"] = pd.to_datetime(data["reservation_status_date"])
    data["booking_date"] = data["arrival_date"] - pd.to_timedelta(
        data["lead_time"], unit="D"
    )
    data["tie_break"] = np.random.default_rng(42).random(len(data))
    active = (data["lead_time"] >= 1) & (
        data["is_canceled"].eq(0) | data["status_date"].ge(data["arrival_date"])
    )
    return data, data.loc[active].copy()


def build_features(data: pd.DataFrame, calendar: bool = True) -> pd.DataFrame:
    features = data[BASE_COLUMNS].copy()
    features["children"] = pd.to_numeric(features["children"])
    for column in ["agent", "company"]:
        features[f"has_{column}"] = features[column].notna().astype(int)
        features[column] = features[column].astype("Int64").astype(str)
    features["total_nights"] = (
        features["stays_in_weekend_nights"] + features["stays_in_week_nights"]
    )
    features["total_guests"] = (
        features["adults"] + features["children"].fillna(0) + features["babies"]
    )
    if calendar:
        dates = arrival_dates(data)
        features["arrival_month"] = dates.dt.month.astype(str)
        features["arrival_weekday"] = dates.dt.dayofweek
        features["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
        features["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
    for column in features.select_dtypes("object"):
        features[column] = features[column].fillna("Missing")
    return features


def training_history(
    data: pd.DataFrame, cutoff: pd.Timestamp, months: int | None = None
) -> pd.DataFrame:
    # Only terminal outcomes already available at the training date enter the fit.
    available = data["arrival_date"].lt(cutoff) & data["status_date"].lt(cutoff)
    if months is not None:
        available &= data["arrival_date"].ge(cutoff - pd.DateOffset(months=months))
    return data.loc[available]


def monthly_cohort(data: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return data.loc[
        data["arrival_date"].ge(cutoff)
        & data["arrival_date"].lt(cutoff + pd.offsets.MonthBegin())
    ]
