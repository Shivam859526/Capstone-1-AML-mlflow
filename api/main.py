from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import prediction function and trained model
try:
    from prediction import predict_fraud
except Exception as exc:
    predict_fraud = None
    MODEL_IMPORT_ERROR = str(exc)
else:
    MODEL_IMPORT_ERROR = None


app = FastAPI(
    title="AML Detection API",
    description="API for detecting potentially suspicious transactions",
    version="1.0.0",
)


class Transaction(BaseModel):
    Time: str = Field(..., examples=["10:30:45"])
    Date: str = Field(..., examples=["2022-09-15"])
    Sender_account: str = Field(..., examples=[" account_1001"])
    Receiver_account: str = Field(..., examples=["account_2001"])
    Amount: float = Field(..., ge=0, examples=[5000.0])
    Payment_currency: str = Field(..., examples=["USD"])
    Received_currency: str = Field(..., examples=["USD"])
    Sender_bank_location: str = Field(..., examples=["USA"])
    Receiver_bank_location: str = Field(..., examples=["USA"])
    Payment_type: str = Field(..., examples=["Wire"])
    Laundering_type: str = Field(..., examples=["Unknown"])


@app.get("/")
def home():
    return {
        "message": "AML Detection API is running",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
    }


@app.get("/health")
def health():
    if predict_fraud is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model could not be loaded: {MODEL_IMPORT_ERROR}",
        )

    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(transaction: Transaction):
    if predict_fraud is None:
        raise HTTPException(
            status_code=503,
            detail=f"Prediction model is unavailable: {MODEL_IMPORT_ERROR}",
        )

    try:
        transaction_data = transaction.model_dump()
        result = predict_fraud(transaction_data)
        return {
            "success": True,
            "prediction": result,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )
