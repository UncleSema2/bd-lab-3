import sys
from unittest.mock import MagicMock

sys.modules["cassandra"] = MagicMock()
sys.modules["cassandra.cluster"] = MagicMock()
sys.modules["cassandra.query"] = MagicMock()

import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np

from src.api.services.prediction_service import PredictionService
from src.api.schemas import PredictionRecord


class TestPredictionService(unittest.TestCase):

    def setUp(self) -> None:
        self.mock_repo = MagicMock()
        self.mock_repo.save = AsyncMock()

    @patch("src.api.services.prediction_service.configparser.ConfigParser")
    def test_init_sets_model_version(self, mock_config_class):
        mock_config_class.return_value.read.return_value = None
        service = PredictionService(model_version="2.0.0")
        self.assertEqual(service.model_version, "2.0.0")

    @patch("src.api.services.prediction_service.pickle.load")
    @patch("src.api.services.prediction_service.open", create=True)
    @patch("src.api.services.prediction_service.os.path.isfile")
    @patch("src.api.services.prediction_service.configparser.ConfigParser")
    def test_load_artifacts_loads_scaler_and_classifier(
        self, mock_config_class, mock_isfile, mock_open, mock_pickle
    ):
        mock_config = MagicMock()
        mock_config.__getitem__ = lambda self, key: {
            "SPLIT_DATA": {"scaler": "scaler.pkl"},
            "LOG_REG": {"path": "model.pkl"},
        }[key]
        mock_config_class.return_value = mock_config

        mock_isfile.return_value = True
        mock_pickle.return_value = MagicMock()
        mock_open.return_value.__enter__.return_value.read.side_effect = [
            b"scaler",
            b"model",
        ]

        service = PredictionService()
        service._load_artifacts()

        self.assertIsNotNone(service._scaler)
        self.assertIn("LOG_REG", service._classifiers)

    @patch("src.api.services.prediction_service.pickle.load")
    @patch("src.api.services.prediction_service.open", create=True)
    @patch("src.api.services.prediction_service.os.path.isfile")
    @patch("src.api.services.prediction_service.configparser.ConfigParser")
    def test_predict_returns_prediction_dict(
        self, mock_config_class, mock_isfile, mock_open, mock_pickle
    ):
        mock_config = MagicMock()
        mock_config.__getitem__ = lambda self, key: {
            "SPLIT_DATA": {"scaler": "scaler.pkl"},
            "LOG_REG": {"path": "model.pkl"},
        }[key]
        mock_config_class.return_value = mock_config

        mock_isfile.return_value = True

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[0.5] * 30])

        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = [1]
        mock_classifier.predict_proba.return_value = [[0.2, 0.8]]

        mock_pickle.side_effect = [mock_scaler, mock_classifier]
        mock_open.return_value.__enter__.return_value.read.side_effect = [
            b"scaler",
            b"model",
        ]

        service = PredictionService()
        result = service.predict([1.0] * 30, model="LOG_REG")

        self.assertEqual(result["prediction"], 1)
        self.assertAlmostEqual(result["probability_malignant"], 0.8)
        self.assertAlmostEqual(result["probability_benign"], 0.2)

    @patch("src.api.services.prediction_service.pickle.load")
    @patch("src.api.services.prediction_service.open", create=True)
    @patch("src.api.services.prediction_service.os.path.isfile")
    @patch("src.api.services.prediction_service.configparser.ConfigParser")
    def test_predict_unknown_model_raises_error(
        self, mock_config_class, mock_isfile, mock_open, mock_pickle
    ):
        mock_config = MagicMock()
        mock_config.__getitem__ = lambda self, key: {
            "SPLIT_DATA": {"scaler": "scaler.pkl"},
            "LOG_REG": {"path": "model.pkl"},
        }[key]
        mock_config_class.return_value = mock_config

        mock_isfile.return_value = True

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[0.5] * 30])

        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = [1]
        mock_classifier.predict_proba.return_value = [[0.2, 0.8]]

        mock_pickle.side_effect = [mock_scaler, mock_classifier]
        mock_open.return_value.__enter__.return_value.read.side_effect = [
            b"scaler",
            b"model",
        ]

        service = PredictionService()
        with self.assertRaises(ValueError) as ctx:
            service.predict([1.0] * 30, model="UNKNOWN")
        self.assertIn("Unknown model", str(ctx.exception))

    @patch("src.api.services.prediction_service.uuid4")
    @patch("src.api.services.prediction_service.pickle.load")
    @patch("src.api.services.prediction_service.open", create=True)
    @patch("src.api.services.prediction_service.os.path.isfile")
    @patch("src.api.services.prediction_service.configparser.ConfigParser")
    def test_predict_and_save_calls_repository(
        self, mock_config_class, mock_isfile, mock_open, mock_pickle, mock_uuid
    ):
        mock_config = MagicMock()
        mock_config.__getitem__ = lambda self, key: {
            "SPLIT_DATA": {"scaler": "scaler.pkl"},
            "LOG_REG": {"path": "model.pkl"},
        }[key]
        mock_config_class.return_value = mock_config

        mock_isfile.return_value = True
        mock_uuid.return_value = "mock-uuid-1234"

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[0.5] * 30])

        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = [1]
        mock_classifier.predict_proba.return_value = [[0.2, 0.8]]

        mock_pickle.side_effect = [mock_scaler, mock_classifier]
        mock_open.return_value.__enter__.return_value.read.side_effect = [
            b"scaler",
            b"model",
        ]

        service = PredictionService(prediction_repository=self.mock_repo)

        result = asyncio.get_event_loop().run_until_complete(
            service.predict_and_save([1.0] * 30, model="LOG_REG")
        )

        self.assertIsInstance(result, PredictionRecord)
        self.mock_repo.save.assert_called_once()

    @patch("src.api.services.prediction_service.configparser.ConfigParser")
    def test_get_last_predictions_without_repo_returns_empty(self, mock_config_class):
        mock_config_class.return_value.read.return_value = None
        service = PredictionService()
        service.prediction_repository = None

        result = asyncio.get_event_loop().run_until_complete(
            service.get_last_predictions(limit=10)
        )
        self.assertEqual(result, [])

    @patch("src.api.services.prediction_service.configparser.ConfigParser")
    def test_get_last_predictions_with_repo(self, mock_config_class):
        mock_config_class.return_value.read.return_value = None
        service = PredictionService(prediction_repository=self.mock_repo)
        self.mock_repo.get_last = AsyncMock(return_value=[])

        asyncio.get_event_loop().run_until_complete(
            service.get_last_predictions(limit=5)
        )
        self.mock_repo.get_last.assert_called_once_with(5)


if __name__ == "__main__":
    unittest.main()
