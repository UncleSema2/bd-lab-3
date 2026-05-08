import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import asyncio

from src.api.services.model_service import ModelService


class TestModelService(unittest.TestCase):

    def setUp(self) -> None:
        self.mock_repo = MagicMock()
        self.mock_repo.get_train_data = MagicMock()
        self.mock_repo.get_eval_data = MagicMock()

    @patch("src.api.services.model_service.configparser.ConfigParser")
    def test_init_sets_repo(self, mock_config_class):
        mock_config_class.return_value.read.return_value = None
        service = ModelService(dataset_repository=self.mock_repo)
        self.assertIsNotNone(service.dataset_repository)

    @patch("src.api.services.model_service.os.makedirs")
    @patch("src.api.services.model_service.pickle.dump")
    @patch("src.api.services.model_service.open", create=True)
    @patch("src.api.services.model_service.configparser.ConfigParser")
    def test_train_without_repo_returns_not_trained(
        self, mock_config_class, mock_open, mock_pickle_dump, mock_makedirs
    ):
        mock_config_class.return_value.read.return_value = None
        service = ModelService()

        result = asyncio.get_event_loop().run_until_complete(service.train())

        self.assertFalse(result.trained)
        self.assertEqual(result.train_samples, 0)

    @patch("src.api.services.model_service.LogisticRegression")
    @patch("src.api.services.model_service.StandardScaler")
    @patch("src.api.services.model_service.pickle.dump")
    @patch("src.api.services.model_service.open", create=True)
    @patch("src.api.services.model_service.os.makedirs")
    @patch("src.api.services.model_service.os.path.isfile")
    @patch("src.api.services.model_service.configparser.ConfigParser")
    def test_train_with_repo_trains_model(
        self, mock_config_class, mock_isfile, mock_makedirs,
        mock_open, mock_pickle_dump, mock_scaler_class, mock_log_reg_class
    ):
        mock_config = MagicMock()
        mock_config.__getitem__ = lambda self, key: {
            "SPLIT_DATA": {"scaler": "scaler.pkl"},
            "LOG_REG": {"path": "model.pkl", "max_iter": "1000"},
        }[key]
        mock_config.getint.return_value = 1000
        mock_config_class.return_value = mock_config

        mock_isfile.return_value = False
        mock_open.return_value.__enter__.return_value.write.side_effect = [None, None]

        mock_scaler = MagicMock()
        mock_scaler.fit_transform.return_value = np.random.rand(10, 30)
        mock_scaler_class.return_value = mock_scaler

        mock_classifier = MagicMock()
        mock_classifier.fit = MagicMock()
        mock_log_reg_class.return_value = mock_classifier

        self.mock_repo.get_train_data.return_value = (
            np.random.rand(10, 30),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        )

        service = ModelService(dataset_repository=self.mock_repo)
        service._scaler = mock_scaler

        result = asyncio.get_event_loop().run_until_complete(service.train())

        self.assertTrue(result.trained)
        self.assertEqual(result.train_samples, 10)
        mock_classifier.fit.assert_called_once()

    @patch("src.api.services.model_service.configparser.ConfigParser")
    def test_evaluate_without_repo_returns_zero_metrics(self, mock_config_class):
        mock_config_class.return_value.read.return_value = None
        service = ModelService()

        result = asyncio.get_event_loop().run_until_complete(service.evaluate())

        self.assertEqual(result.accuracy, 0.0)
        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.message, "Dataset repository not available")

    @patch("src.api.services.model_service.pickle.load")
    @patch("src.api.services.model_service.open", create=True)
    @patch("src.api.services.model_service.os.path.isfile")
    @patch("src.api.services.model_service.configparser.ConfigParser")
    def test_evaluate_model_not_trained(self, mock_config_class, mock_isfile, mock_open, mock_pickle):
        mock_config = MagicMock()
        mock_config.__getitem__ = lambda self, key: {
            "SPLIT_DATA": {"scaler": "scaler.pkl"},
            "LOG_REG": {"path": "model.pkl"},
        }[key]
        mock_config_class.return_value = mock_config

        mock_isfile.return_value = False
        self.mock_repo.get_eval_data.return_value = (
            np.random.rand(5, 30),
            np.array([0, 1, 0, 1, 0])
        )

        service = ModelService(dataset_repository=self.mock_repo)

        result = asyncio.get_event_loop().run_until_complete(service.evaluate())

        self.assertEqual(result.accuracy, 0.0)
        self.assertIn("not trained yet", result.message)

    @patch("src.api.services.model_service.pickle.load")
    @patch("src.api.services.model_service.open", create=True)
    @patch("src.api.services.model_service.os.path.isfile")
    @patch("src.api.services.model_service.configparser.ConfigParser")
    def test_evaluate_returns_metrics(self, mock_config_class, mock_isfile, mock_open, mock_pickle):
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
        mock_classifier.predict.return_value = np.array([1, 0, 1, 0, 1])

        def pickle_load_side_effect(f):
            if "scaler" in str(f):
                return mock_scaler
            return mock_classifier
        mock_pickle.side_effect = pickle_load_side_effect

        self.mock_repo.get_eval_data.return_value = (
            np.random.rand(5, 30),
            np.array([1, 0, 1, 0, 1])
        )

        service = ModelService(dataset_repository=self.mock_repo)

        result = asyncio.get_event_loop().run_until_complete(service.evaluate())

        self.assertGreaterEqual(result.accuracy, 0.0)
        self.assertLessEqual(result.accuracy, 1.0)
        self.assertEqual(result.eval_samples, 5)


if __name__ == "__main__":
    unittest.main()
