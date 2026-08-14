import pandas as pd


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    df["has_agent"] = df["agent"].notna().astype(int)
    df["has_company"] = df["company"].notna().astype(int)

    for col in ["agent", "company"]:
        df[col] = df[col].astype("Int64").astype("string")

    df["country"] = df["country"].fillna("Unknown")
    df["country_group"] = "Foreign"
    df.loc[df["country"] == "PRT", "country_group"] = "Domestic"
    df.loc[df["country"] == "Unknown", "country_group"] = "Unknown"

    df["total_guests"] = (
        df["adults"] + df["children"].fillna(0) + df["babies"]
    )
    df["total_nights"] = (
        df["stays_in_week_nights"] + df["stays_in_weekend_nights"]
    )

    df["has_previous_cancellation"] = (
        df["previous_cancellations"] > 0
    ).astype(int)

    return df
