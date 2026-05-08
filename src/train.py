import configparser
import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from src.logger import Logger


class Model:
    def __init__(self) -> None:
        logger = Logger()
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        self.log = logger.get_logger(__name__)
        self.X_train = pd.read_csv(self.config["SPLIT_DATA"]["x_train"], index_col=0)
        self.y_train = pd.read_csv(self.config["SPLIT_DATA"]["y_train"], index_col=0)
        self.X_test = pd.read_csv(self.config["SPLIT_DATA"]["x_test"], index_col=0)
        self.y_test = pd.read_csv(self.config["SPLIT_DATA"]["y_test"], index_col=0)
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.experiments_path = "experiments"
        os.makedirs(self.experiments_path, exist_ok=True)
        scaler_path = self.config["SPLIT_DATA"]["scaler"]
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        self.log.info(f"{scaler_path} is saved")
        self.log.info("Model is ready")

    def log_reg(self, predict: bool = False) -> bool:
        max_iter = self.config.getint("LOG_REG", "max_iter")
        path = self.config["LOG_REG"]["path"]
        classifier = LogisticRegression(max_iter=max_iter)
        classifier.fit(self.X_train, self.y_train.values.ravel())
        if predict:
            y_pred = classifier.predict(self.X_test)
            self.log.info(f"LOG_REG accuracy: {accuracy_score(self.y_test, y_pred)}")
        return self._save(classifier, path)

    def _save(self, classifier, path: str) -> bool:
        with open(path, "wb") as f:
            pickle.dump(classifier, f)
        self.log.info(f"{path} is saved")
        return os.path.isfile(path)


if __name__ == "__main__":
    model = Model()
    model.log_reg(predict=False)
