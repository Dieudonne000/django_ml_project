from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "dummy-data" / "vehicles_ml_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parent / "regression_model.pkl"

UI_FEATURES = ["year", "kilometers_driven", "seating_capacity", "estimated_income"]
TARGET = "selling_price"

NUMERIC_CANDIDATES = [
    "year",
    "kilometers_driven",
    "seating_capacity",
    "estimated_income",
    "client_age",
]

CATEGORICAL_CANDIDATES = [
    "manufacturer",
    "color",
    "body_type",
    "engine_type",
    "transmission",
    "fuel_type",
    "vehicle_condition",
    "client_gender",
    "province",
    "district",
    "income_level",
    "client_profession",
    "season",
]


def _build_defaults(df: pd.DataFrame, features: list[str]) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            defaults[col] = float(df[col].median())
        else:
            mode_vals = df[col].mode(dropna=True)
            defaults[col] = str(mode_vals.iloc[0]) if not mode_vals.empty else "Unknown"
    return defaults


def train_and_save_regression_model():
    df = pd.read_csv(DATASET_PATH)

    numeric_features = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    categorical_features = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
    all_features = numeric_features + categorical_features

    X = df[all_features].copy()
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = ExtraTreesRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )
    best_pipe = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    best_pipe.fit(X_train, y_train)
    best_predictions = best_pipe.predict(X_test)
    best_r2 = r2_score(y_test, best_predictions)
    best_name = "ExtraTrees"

    defaults = _build_defaults(df, all_features)

    bundle = {
        "model": best_pipe,
        "model_name": best_name,
        "features": all_features,
        "defaults": defaults,
        "ui_features": UI_FEATURES,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    r2 = round(best_r2 * 100, 2)
    comparison_df = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Predicted": np.round(best_predictions, 2),
            "Difference": np.round(y_test.values - best_predictions, 2),
        }
    )

    return bundle, r2, comparison_df


def get_regression_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    bundle, _, _ = train_and_save_regression_model()
    return bundle


def predict_regression_price(
    bundle: dict,
    year: int,
    kilometers_driven: float,
    seating_capacity: int,
    estimated_income: float,
) -> float:
    row = dict(bundle.get("defaults", {}))
    row.update(
        {
            "year": int(year),
            "kilometers_driven": float(kilometers_driven),
            "seating_capacity": int(seating_capacity),
            "estimated_income": float(estimated_income),
        }
    )
    features = bundle["features"]
    X_pred = pd.DataFrame([{f: row.get(f) for f in features}], columns=features)
    pred = float(bundle["model"].predict(X_pred)[0])
    return pred


def evaluate_regression_model():
    _, r2, comparison_df = train_and_save_regression_model()
    return {
        "r2": r2,
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
    }
