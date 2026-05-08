import configparser
import os
import pickle
from uuid import uuid4
import numpy as np
from datetime import datetime

from src.api.schemas import PredictionRecord
from src.api.repositories.prediction_repository import PredictionRepository


class PredictionService:
    def __init__(
        self,
        config_path: str = "config.ini",
        model_version: str = "1.0.0",
        prediction_repository: "PredictionRepository" = None,
    ) -> None:
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.model_version = model_version
        self.prediction_repository = prediction_repository
        self._classifiers = {}
        self._scaler = None

    def _load_artifacts(self):
        if self._scaler is None:
            scaler_path = self.config["SPLIT_DATA"]["scaler"]
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)

        for name in ["LOG_REG"]:
            if name not in self._classifiers:
                path = self.config[name]["path"]
                if os.path.isfile(path):
                    with open(path, "rb") as f:
                        self._classifiers[name] = pickle.load(f)

    def predict(self, features: list[float], model: str = "LOG_REG") -> dict:
        self._load_artifacts()
        if model not in self._classifiers:
            raise ValueError(f"Unknown model: {model}")
        if model not in self._classifiers:
            raise ValueError(f"Model {model} not trained yet")

        X = np.array(features).reshape(1, -1)
        X_scaled = self._scaler.transform(X)

        classifier = self._classifiers[model]
        prediction = int(classifier.predict(X_scaled)[0])

        proba = classifier.predict_proba(X_scaled)[0]
        probability_malignant = float(proba[1])
        probability_benign = float(proba[0])

        return {
            "prediction": prediction,
            "probability_malignant": probability_malignant,
            "probability_benign": probability_benign,
        }

    async def predict_and_save(
        self, features: list[float], model: str = "LOG_REG"
    ) -> PredictionRecord:
        result = self.predict(features, model)

        prediction_record = PredictionRecord(
            prediction_id=str(uuid4()),
            features=features,
            prediction=result["prediction"],
            probability_malignant=result["probability_malignant"],
            probability_benign=result["probability_benign"],
            created_at=datetime.utcnow(),
            model_version=self.model_version,
        )

        if self.prediction_repository:
            await self.prediction_repository.save(prediction_record)

        return prediction_record

    async def get_last_predictions(self, limit: int = 10) -> list[PredictionRecord]:
        if not self.prediction_repository:
            return []

        return await self.prediction_repository.get_last(limit)
