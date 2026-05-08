import sys
from unittest.mock import MagicMock

sys.modules["cassandra"] = MagicMock()
sys.modules["cassandra.cluster"] = MagicMock()
sys.modules["cassandra.query"] = MagicMock()

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import UUID
import asyncio

from src.api.repositories.prediction_repository import PredictionRepository
from src.api.schemas import PredictionRecord


class TestPredictionRepository(unittest.TestCase):

    def setUp(self) -> None:
        self.mock_cluster = MagicMock()
        self.mock_session = MagicMock()
        self.mock_cluster.connect.return_value = self.mock_session

    @patch("src.api.repositories.prediction_repository.Cluster")
    def test_init_creates_keyspace_and_table(self, mock_cluster_class):
        mock_cluster_class.return_value = self.mock_cluster
        PredictionRepository(cluster=self.mock_cluster, keyspace="test_keyspace")
        self.mock_session.execute.assert_called()
        self.mock_session.set_keyspace.assert_called_with("test_keyspace")

    @patch("src.api.repositories.prediction_repository.Cluster")
    def test_save_inserts_prediction(self, mock_cluster_class):
        mock_cluster_class.return_value = self.mock_cluster
        repo = PredictionRepository(cluster=self.mock_cluster, keyspace="test_keyspace")

        prediction = PredictionRecord(
            prediction_id="550e8400-e29b-41d4-a716-446655440000",
            features=[1.0] * 30,
            prediction=1,
            probability_malignant=0.8,
            probability_benign=0.2,
            created_at=datetime.utcnow(),
            model_version="v1",
        )

        asyncio.get_event_loop().run_until_complete(repo.save(prediction))
        self.mock_session.execute.assert_called()

    @patch("src.api.repositories.prediction_repository.Cluster")
    def test_get_last_returns_predictions(self, mock_cluster_class):
        mock_cluster_class.return_value = self.mock_cluster
        repo = PredictionRepository(cluster=self.mock_cluster, keyspace="test_keyspace")

        mock_row = MagicMock()
        mock_row.prediction_id = UUID("550e8400-e29b-41d4-a716-446655440000")
        mock_row.features = [1.0] * 30
        mock_row.prediction = 1
        mock_row.probability_malignant = 0.8
        mock_row.probability_benign = 0.2
        mock_row.created_at = datetime.utcnow()
        mock_row.model_version = "v1"

        self.mock_session.execute.return_value = [mock_row]

        result = asyncio.get_event_loop().run_until_complete(repo.get_last(limit=10))

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], PredictionRecord)
        self.assertEqual(result[0].prediction, 1)


if __name__ == "__main__":
    unittest.main()
