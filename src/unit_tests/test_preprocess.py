import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from src.preprocess import DataMaker


class TestDataMaker(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.config_content = """[SPLIT_DATA]
x_path = {temp_dir}/X.csv
y_path = {temp_dir}/y.csv
x_train = {temp_dir}/X_train.csv
y_train = {temp_dir}/y_train.csv
x_test = {temp_dir}/X_test.csv
y_test = {temp_dir}/y_test.csv
""".format(temp_dir=self.temp_dir)
        self.config_path = os.path.join(self.temp_dir, "config.ini")
        with open(self.config_path, "w") as f:
            f.write(self.config_content)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("src.preprocess.configparser.ConfigParser")
    @patch("src.preprocess.Logger")
    def test_save_creates_file(self, mock_logger_class, mock_config_class):
        mock_logger = MagicMock()
        mock_logger_class.return_value.get_logger.return_value = mock_logger
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config
        mock_config.read.return_value = None

        dm = DataMaker.__new__(DataMaker)
        dm.log = mock_logger
        dm.config = mock_config
        dm.X_train_path = os.path.join(self.temp_dir, "test_save.csv")

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = dm._save(df, dm.X_train_path)

        self.assertTrue(result)
        self.assertTrue(os.path.isfile(dm.X_train_path))
        saved_df = pd.read_csv(dm.X_train_path)
        self.assertEqual(len(saved_df), 3)

    @patch("src.preprocess.pd.read_csv")
    @patch("src.preprocess.train_test_split")
    @patch("src.preprocess.DataMaker._save")
    @patch("src.preprocess.configparser.ConfigParser")
    @patch("src.preprocess.Logger")
    def test_split_data_calls_save_for_all_files(
        self, mock_logger_class, mock_config_class, mock_save, mock_split, mock_read_csv
    ):
        mock_logger = MagicMock()
        mock_logger_class.return_value.get_logger.return_value = mock_logger
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config
        mock_config.read.return_value = None
        mock_config.__getitem__ = lambda self, key: {
            "SPLIT_DATA": {
                "x_path": os.path.join(self.temp_dir, "X.csv"),
                "y_path": os.path.join(self.temp_dir, "y.csv"),
                "x_train": os.path.join(self.temp_dir, "x_train.csv"),
                "y_train": os.path.join(self.temp_dir, "y_train.csv"),
                "x_test": os.path.join(self.temp_dir, "x_test.csv"),
                "y_test": os.path.join(self.temp_dir, "y_test.csv"),
            }
        }[key]

        X = pd.DataFrame({"a": range(10)})
        y = pd.DataFrame({"target": range(10)})
        mock_split.return_value = (X[:8], X[8:], y[:8], y[8:])

        dm = DataMaker.__new__(DataMaker)
        dm.log = mock_logger
        dm.config = mock_config
        dm.X_path = os.path.join(self.temp_dir, "X.csv")
        dm.y_path = os.path.join(self.temp_dir, "y.csv")
        dm.X_train_path = os.path.join(self.temp_dir, "x_train.csv")
        dm.y_train_path = os.path.join(self.temp_dir, "y_train.csv")
        dm.X_test_path = os.path.join(self.temp_dir, "x_test.csv")
        dm.y_test_path = os.path.join(self.temp_dir, "y_test.csv")

        def read_csv_side_effect(path, **kwargs):
            if "X.csv" in str(path):
                return pd.DataFrame({"a": range(10)})
            return pd.DataFrame({"target": range(10)})
        mock_read_csv.side_effect = read_csv_side_effect

        dm.split_data()

        self.assertEqual(mock_save.call_count, 4)


if __name__ == "__main__":
    unittest.main()
