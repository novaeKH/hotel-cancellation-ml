import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from hotel_booking.data import (
    build_features,
    load_bookings,
    monthly_cohort,
    training_history,
)
from hotel_booking.modeling import (
    evaluate_predictions,
    fit_model,
    seasonal_probabilities,
)


def describe_data(raw, active, report):
    quality = {
        "raw_rows": len(raw),
        "duplicate_rows": int(
            raw.drop(
                columns=[
                    "booking_id",
                    "arrival_date",
                    "status_date",
                    "booking_date",
                    "tie_break",
                ]
            )
            .duplicated()
            .sum()
        ),
        "same_day_bookings": int(raw["lead_time"].eq(0).sum()),
        "already_canceled": int(
            (
                (raw["lead_time"] >= 1)
                & raw["is_canceled"].eq(1)
                & raw["status_date"].lt(raw["arrival_date"])
            ).sum()
        ),
        "eligible_rows": len(active),
        "late_cancellations": int(active["is_canceled"].sum()),
        "late_cancellation_rate": active["is_canceled"].mean(),
        "missing_target": int(raw["is_canceled"].isna().sum()),
        "missing_children": int(raw["children"].isna().sum()),
        "zero_guests": int(
            raw[["adults", "children", "babies"]].fillna(0).sum(axis=1).eq(0).sum()
        ),
        "arrival_start": str(raw["arrival_date"].min().date()),
        "arrival_end": str(raw["arrival_date"].max().date()),
    }
    (report / "data_quality.json").write_text(json.dumps(quality, indent=2))
    raw.isna().mean().rename("missing_share").to_csv(report / "missingness.csv")
    for name, data in [("all", raw), ("active", active)]:
        monthly = data.assign(month=data["arrival_date"].dt.to_period("M").astype(str))
        monthly.groupby(["hotel", "month"])["is_canceled"].agg(
            bookings="size", cancellations="sum", cancellation_rate="mean"
        ).reset_index().to_csv(report / f"monthly_{name}.csv", index=False)
    raw.assign(month=raw["booking_date"].dt.to_period("M").astype(str)).groupby(
        "month"
    ).agg(rows=("booking_id", "size"), median_lead_time=("lead_time", "median")).to_csv(
        report / "booking_cohorts.csv"
    )


def select_parameters(data, report):
    rows = []
    validation_months = pd.to_datetime(["2017-01-01", "2017-02-01"])
    for depth in [4, 6]:
        for history_months in [0, 12]:
            cohorts, predictions = [], []
            for cutoff in validation_months:
                train = training_history(data, cutoff, history_months or None)
                cohort = monthly_cohort(data, cutoff)
                model = fit_model(train, depth=depth)
                cohorts.append(cohort)
                predictions.append(model.predict_proba(build_features(cohort))[:, 1])
            row = {
                "depth": depth,
                "history_months": history_months,
                **evaluate_predictions(pd.concat(cohorts), np.concatenate(predictions)),
            }
            rows.append(row)
            print("Validation", row, flush=True)
    results = pd.DataFrame(rows).sort_values(
        ["average_precision", "history_months", "depth"], ascending=[False, True, True]
    )
    results.to_csv(report / "validation_results.csv", index=False)
    best = results.iloc[0]
    selection = {
        "depth": int(best["depth"]),
        "history_months": int(best["history_months"]),
    }
    (report / "selection.json").write_text(json.dumps(selection, indent=2))
    return selection


def evaluate_months(data, selection, report, model_dir):
    depth = selection["depth"]
    months = selection["history_months"] or None
    test_months = pd.date_range("2017-03-01", "2017-08-01", freq="MS")
    frozen_train = training_history(data, test_months[0], months)
    frozen = fit_model(frozen_train, depth=depth)
    predictions, metrics, windows = [], [], []
    for cutoff in test_months:
        train = training_history(data, cutoff, months)
        cohort = monthly_cohort(data, cutoff)
        calendar_features = build_features(cohort)
        monthly_model = (
            frozen if cutoff == test_months[0] else fit_model(train, depth=depth)
        )
        no_calendar = fit_model(train, depth=depth, calendar=False)
        logistic = fit_model(train, family="logistic")
        probabilities = {
            "seasonal_rate": seasonal_probabilities(train, cohort),
            "logistic": logistic.predict_proba(calendar_features)[:, 1],
            "catboost_frozen": frozen.predict_proba(calendar_features)[:, 1],
            "catboost_monthly": monthly_model.predict_proba(calendar_features)[:, 1],
            "catboost_no_calendar": no_calendar.predict_proba(
                build_features(cohort, calendar=False)
            )[:, 1],
        }
        windows.append(
            {
                "month": str(cutoff.date()),
                "train_rows": len(train),
                "train_arrival_end": str(train["arrival_date"].max().date()),
                "last_known_outcome": str(train["status_date"].max().date()),
                "test_rows": len(cohort),
                "test_positives": int(cohort["is_canceled"].sum()),
                "logistic_iterations": int(logistic[-1].n_iter_[0]),
            }
        )
        for name, probability in probabilities.items():
            metrics.append(
                {
                    "model": name,
                    "month": str(cutoff.date()),
                    **evaluate_predictions(cohort, probability),
                }
            )
            predictions.append(
                cohort[
                    ["booking_id", "hotel", "arrival_date", "is_canceled", "tie_break"]
                ].assign(model=name, probability=probability)
            )
        print(
            "Test month",
            str(cutoff.date()),
            "train",
            len(train),
            "rows",
            len(cohort),
            flush=True,
        )
    pd.DataFrame(metrics).to_csv(report / "monthly_metrics.csv", index=False)
    pd.DataFrame(windows).to_csv(report / "training_windows.csv", index=False)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_parquet(report / "predictions.parquet", index=False)
    summary, segments = [], []
    for name, group in predictions.groupby("model"):
        summary.append(
            {
                "model": name,
                **evaluate_predictions(group, group["probability"].to_numpy()),
            }
        )
        for hotel, segment in group.groupby("hotel"):
            segments.append(
                {
                    "model": name,
                    "hotel": hotel,
                    **evaluate_predictions(segment, segment["probability"].to_numpy()),
                }
            )
    pd.DataFrame(summary).to_csv(report / "test_metrics.csv", index=False)
    pd.DataFrame(segments).to_csv(report / "hotel_metrics.csv", index=False)
    monthly_model.save_model(str(model_dir / "late_cancellation.cbm"))
    feature_names = build_features(data).columns.tolist()
    pd.Series(monthly_model.feature_importances_, index=feature_names).sort_values(
        ascending=False
    ).rename("importance").to_csv(report / "feature_importance.csv")
    metadata = {
        "task": "late_cancellation",
        "prediction_time": "end_of_day_before_arrival",
        "trained_before": "2017-08-01",
        "features": feature_names,
        **selection,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=Path, default=Path("data/raw/hotel_bookings.csv")
    )
    args = parser.parse_args()
    report, model_dir = Path("reports/evaluation"), Path("models/trained")
    report.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    raw, active = load_bookings(args.data_path)
    describe_data(raw, active, report)
    selection = select_parameters(active, report)
    evaluate_months(active, selection, report, model_dir)
    (report / "run.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "catboost_iterations": 400,
                "learning_rate": 0.05,
                "validation_months": ["2017-01", "2017-02"],
                "evaluation_months": [
                    "2017-03",
                    "2017-04",
                    "2017-05",
                    "2017-06",
                    "2017-07",
                    "2017-08",
                ],
                "daily_review_share": 0.10,
                "quota_rounding": "ceil per hotel and arrival date",
                "raw_sha256": hashlib.sha256(args.data_path.read_bytes()).hexdigest(),
                "raw_rows": len(raw),
                "eligible_rows": len(active),
                "python": platform.python_version(),
                "packages": {
                    package: version(package)
                    for package in ["numpy", "pandas", "scikit-learn", "catboost"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
