import configparser
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import Logger

TEST_SIZE = 0.2


class DataMaker:

    def __init__(self) -> None:
        logger = Logger()
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        self.log = logger.get_logger(__name__)
        self.X_path = self.config["SPLIT_DATA"]["x_path"]
        self.y_path = self.config["SPLIT_DATA"]["y_path"]
        self.X_train_path = self.config["SPLIT_DATA"]["x_train"]
        self.y_train_path = self.config["SPLIT_DATA"]["y_train"]
        self.X_test_path = self.config["SPLIT_DATA"]["x_test"]
        self.y_test_path = self.config["SPLIT_DATA"]["y_test"]
        os.makedirs(os.path.dirname(self.X_train_path), exist_ok=True)
        self.log.info("DataMaker is ready")

    def split_data(self, test_size=TEST_SIZE) -> bool:
        X = pd.read_csv(self.X_path, index_col=0)
        y = pd.read_csv(self.y_path, index_col=0)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        self._save(X_train, self.X_train_path)
        self._save(y_train, self.y_train_path)
        self._save(X_test, self.X_test_path)
        self._save(y_test, self.y_test_path)
        self.log.info("Train and test data is ready")
        return all(os.path.isfile(p) for p in [
            self.X_train_path, self.y_train_path,
            self.X_test_path, self.y_test_path,
        ])

    def _save(self, df: pd.DataFrame, path: str) -> bool:
        df.reset_index(drop=True).to_csv(path, index=True)
        self.log.info(f"{path} is saved")
        return os.path.isfile(path)


if __name__ == "__main__":
    DataMaker().split_data()
