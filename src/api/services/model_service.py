import configparser
import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.api.repositories.dataset_repository import DatasetRepository
from src.api.schemas import TrainResponse, EvaluateResponse


class ModelService:
    def __init__(
        self,
        config_path: str = "config.ini",
        dataset_repository: DatasetRepository = None,
    ) -> None:
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.dataset_repository = dataset_repository
        self._scaler = None
        self._classifiers = {}

    def _get_scaler(self) -> StandardScaler:
        if self._scaler is None:
            scaler_path = self.config["SPLIT_DATA"]["scaler"]
            if os.path.isfile(scaler_path):
                with open(scaler_path, "rb") as f:
                    self._scaler = pickle.load(f)
            else:
                self._scaler = StandardScaler()
        return self._scaler

    async def train(self, model: str = "LOG_REG") -> TrainResponse:
        if self.dataset_repository is None:
            return TrainResponse(
                model=model,
                trained=False,
                train_samples=0,
                message="Dataset repository not available",
            )

        X_train, y_train = self.dataset_repository.get_train_data()

        scaler = self._get_scaler()
        X_train_scaled = scaler.fit_transform(X_train)

        max_iter = self.config.getint("LOG_REG", "max_iter")
        classifier = LogisticRegression(max_iter=max_iter)
        classifier.fit(X_train_scaled, y_train)

        scaler_path = self.config["SPLIT_DATA"]["scaler"]
        model_path = self.config["LOG_REG"]["path"]

        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        with open(model_path, "wb") as f:
            pickle.dump(classifier, f)

        self._scaler = scaler
        self._classifiers[model] = classifier

        return TrainResponse(
            model=model,
            trained=True,
            train_samples=len(y_train),
            message=f"Model {model} trained successfully on {len(y_train)} samples",
        )

    async def evaluate(self, model: str = "LOG_REG") -> EvaluateResponse:
        if self.dataset_repository is None:
            return EvaluateResponse(
                model=model,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                eval_samples=0,
                message="Dataset repository not available",
            )

        X_eval, y_eval = self.dataset_repository.get_eval_data()

        scaler_path = self.config["SPLIT_DATA"]["scaler"]
        model_path = self.config["LOG_REG"]["path"]

        if not os.path.isfile(scaler_path) or not os.path.isfile(model_path):
            return EvaluateResponse(
                model=model,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                eval_samples=len(y_eval),
                message=f"Model {model} not trained yet. Call /train first.",
            )

        if self._scaler is None:
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)

        if model not in self._classifiers:
            with open(model_path, "rb") as f:
                self._classifiers[model] = pickle.load(f)

        X_eval_scaled = self._scaler.transform(X_eval)
        y_pred = self._classifiers[model].predict(X_eval_scaled)

        acc = accuracy_score(y_eval, y_pred)
        prec = precision_score(y_eval, y_pred, average="binary", zero_division=0)
        rec = recall_score(y_eval, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_eval, y_pred, average="binary", zero_division=0)

        return EvaluateResponse(
            model=model,
            accuracy=round(acc, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            eval_samples=len(y_eval),
            message=f"Evaluation complete on {len(y_eval)} samples",
        )
