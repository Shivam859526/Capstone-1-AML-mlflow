"""Prediction utilities for the AML detection project.

This loads the complete trained Random Forest imbalanced-learn pipeline.
Input must contain the same raw feature columns used during training,
excluding the target column `Is_laundering`.
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


# Assumes this file is stored in <project_root>/src/prediction.py
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "Random_Forest.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at: {MODEL_PATH}. "
        "Run the training script first and confirm the model filename."
    )

# This is the full pipeline: preprocessing -> SMOTE (training only) -> classifier.
model = joblib.load(MODEL_PATH)

# Read the exact input feature order recorded by sklearn when the pipeline was fit.
try:
    FEATURE_COLUMNS = list(model.feature_names_in_)
except AttributeError:
    FEATURE_COLUMNS = list(model.named_steps["preprocessor"].feature_names_in_)


def _convert_date_time(features: pd.DataFrame) -> pd.DataFrame:
    """Match the Date/Time conversions used during training."""
    if "Date" in features.columns:
        # Training converted dates to YYYYMMDD integers.
        parsed_date = pd.to_datetime(features["Date"], errors="coerce")
        date_as_number = pd.to_numeric(
            parsed_date.dt.strftime("%Y%m%d"), errors="coerce"
        )
        # Keep already-numeric YYYYMMDD values if pandas cannot parse them.
        numeric_date = pd.to_numeric(features["Date"], errors="coerce")
        features["Date"] = date_as_number.fillna(numeric_date).astype("float64")

    if "Time" in features.columns:
        # Training converted HH:MM:SS to HHMMSS integers.
        parsed_time = pd.to_datetime(
            features["Time"].astype("string"),
            format="%H:%M:%S",
            errors="coerce",
        )
        time_as_number = pd.to_numeric(
            parsed_time.dt.strftime("%H%M%S"), errors="coerce"
        )
        numeric_time = pd.to_numeric(features["Time"], errors="coerce")
        features["Time"] = time_as_number.fillna(numeric_time).astype("float64")

    return features


def prepare_features(transaction: dict[str, Any]) -> pd.DataFrame:
    """Build a one-row DataFrame with the exact columns expected by the model.

    Supply all original feature columns from the training dataset, except
    `Is_laundering`. Do not supply manually one-hot-encoded columns.
    """
    if not isinstance(transaction, dict):
        raise TypeError("transaction must be a dictionary.")

    payload = {
        key: value
        for key, value in transaction.items()
        if key != "Is_laundering"
    }

    missing = [column for column in FEATURE_COLUMNS if column not in payload]
    if missing:
        raise ValueError(
            "Missing required transaction feature(s): "
            + ", ".join(missing)
        )

    # Keep only trained features, in the same order as during training.
    features = pd.DataFrame(
        [{column: payload[column] for column in FEATURE_COLUMNS}],
        columns=FEATURE_COLUMNS,
    )

    features = _convert_date_time(features)
    return features


def predict_fraud(transaction: dict[str, Any]) -> dict[str, Any]:
    """Predict whether a transaction is suspicious/laundering."""
    features = prepare_features(transaction)

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    positive_class_index = list(model.classes_).index(1)
    laundering_probability = float(probabilities[positive_class_index])

    return {
        "is_laundering": prediction,
        "laundering_probability": round(laundering_probability, 4),
        "result": "Suspicious / Potential Laundering" if prediction == 1 else "Normal",
    }
