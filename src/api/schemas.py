from datetime import datetime
from pydantic import BaseModel, Field
from typing import List


class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_length=30, max_length=30)
    model: str = "LOG_REG"


class PredictResponse(BaseModel):
    prediction_id: str
    prediction: int
    probability_malignant: float
    probability_benign: float
    created_at: datetime
    model_version: str


class PredictionRecord(BaseModel):
    prediction_id: str
    features: List[float]
    prediction: int
    probability_malignant: float
    probability_benign: float
    created_at: datetime
    model_version: str


class TrainResponse(BaseModel):
    model: str
    trained: bool
    train_samples: int
    message: str


class EvaluateResponse(BaseModel):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    eval_samples: int
    message: str
