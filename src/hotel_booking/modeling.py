import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hotel_booking.data import build_features


def fit_model(train, family="catboost", depth=4, calendar=True):
    features = build_features(train, calendar=calendar)
    if family == "logistic":
        numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        preparation = ColumnTransformer(
            [
                ("numeric", numeric, make_column_selector(dtype_include=np.number)),
                (
                    "categorical",
                    categorical,
                    make_column_selector(dtype_include="object"),
                ),
            ],
            sparse_threshold=1.0,
        )
        model = make_pipeline(
            preparation, LogisticRegression(C=1.0, max_iter=1500, random_state=42)
        )
    else:
        model = CatBoostClassifier(
            iterations=400,
            depth=depth,
            learning_rate=0.05,
            random_seed=42,
            loss_function="Logloss",
            thread_count=4,
            verbose=False,
            allow_writing_files=False,
            cat_features=features.select_dtypes("object").columns.tolist(),
        )
    model.fit(features, train["is_canceled"])
    return model


def seasonal_probabilities(train, cohort, smoothing=50):
    historical = train.assign(month=train["arrival_date"].dt.month)
    group_rates = historical.groupby(["hotel", "month"])["is_canceled"].agg(
        ["sum", "size"]
    )
    hotel_rates = historical.groupby("hotel")["is_canceled"].mean()
    prior = group_rates.index.get_level_values("hotel").map(hotel_rates)
    rates = (group_rates["sum"] + smoothing * prior) / (group_rates["size"] + smoothing)
    keys = pd.MultiIndex.from_arrays([cohort["hotel"], cohort["arrival_date"].dt.month])
    probabilities = rates.reindex(keys).to_numpy()
    fallback = cohort["hotel"].map(hotel_rates).fillna(historical["is_canceled"].mean())
    return np.where(np.isnan(probabilities), fallback, probabilities)


def evaluate_predictions(cohort, probabilities, budget=0.10):
    ranking = cohort[["hotel", "arrival_date", "is_canceled", "tie_break"]].copy()
    ranking["probability"] = probabilities
    ranking = ranking.sort_values(["probability", "tie_break"], ascending=[False, True])
    groups = ranking.groupby(["hotel", "arrival_date"])
    quota = np.ceil(groups["is_canceled"].transform("size") * budget)
    selected = ranking.loc[groups.cumcount() < quota]
    target = cohort["is_canceled"]
    precision = selected["is_canceled"].mean()
    recall = selected["is_canceled"].sum() / target.sum()
    return {
        "rows": len(cohort),
        "positives": int(target.sum()),
        "positive_rate": target.mean(),
        "roc_auc": roc_auc_score(target, probabilities),
        "average_precision": average_precision_score(target, probabilities),
        "brier": brier_score_loss(target, probabilities),
        "reviewed": len(selected),
        "review_share": len(selected) / len(cohort),
        "precision_at_budget": precision,
        "recall_at_budget": recall,
        "lift_at_budget": precision / target.mean(),
    }
