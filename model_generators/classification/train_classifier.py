from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "dummy-data" / "vehicles_ml_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parent / "classification_model.pkl"

FEATURES = ["year", "kilometers_driven", "seating_capacity", "estimated_income"]
TARGET = "income_level"


def train_and_save_classification_model():
    df = pd.read_csv(DATASET_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    predictions = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, predictions) * 100, 2)
    comparison_df = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Predicted": predictions,
            "Match": y_test.values == predictions,
        }
    )
    return model, accuracy, comparison_df


def get_classification_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    model, _, _ = train_and_save_classification_model()
    return model


def evaluate_classification_model():
    _, accuracy, comparison_df = train_and_save_classification_model()
    return {
        "accuracy": accuracy,
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            justify="center",
            index=False,
        ),
    }

