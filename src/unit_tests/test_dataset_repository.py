import sys
from unittest.mock import MagicMock
sys.modules['cassandra'] = MagicMock()
sys.modules['cassandra.cluster'] = MagicMock()
sys.modules['cassandra.query'] = MagicMock()

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.api.repositories.dataset_repository import DatasetRepository


class TestDatasetRepository(unittest.TestCase):

    def setUp(self) -> None:
        self.mock_cluster = MagicMock()
        self.mock_session = MagicMock()
        self.mock_cluster.connect.return_value = self.mock_session

    @patch("src.api.repositories.dataset_repository.Cluster")
    def test_init_connects_to_keyspace(self, mock_cluster_class):
        mock_cluster_class.return_value = self.mock_cluster
        DatasetRepository(cluster=self.mock_cluster, keyspace="test_keyspace")
        self.mock_cluster.connect.assert_called_once_with("test_keyspace")

    @patch("src.api.repositories.dataset_repository.Cluster")
    def test_get_train_data_returns_arrays(self, mock_cluster_class):
        mock_cluster_class.return_value = self.mock_cluster
        repo = DatasetRepository(cluster=self.mock_cluster, keyspace="test_keyspace")

        mock_row = MagicMock()
        for col in DatasetRepository.FEATURE_COLS:
            setattr(mock_row, col, 0.5)
        mock_row.target = 1
        self.mock_session.execute.return_value = [mock_row] * 5

        features, targets = repo.get_train_data()

        self.assertEqual(features.shape, (5, 30))
        self.assertEqual(targets.shape, (5,))
        self.assertEqual(targets[0], 1)

    @patch("src.api.repositories.dataset_repository.Cluster")
    def test_get_eval_data_returns_arrays(self, mock_cluster_class):
        mock_cluster_class.return_value = self.mock_cluster
        repo = DatasetRepository(cluster=self.mock_cluster, keyspace="test_keyspace")

        mock_row = MagicMock()
        for col in DatasetRepository.FEATURE_COLS:
            setattr(mock_row, col, 0.3)
        mock_row.target = 0
        self.mock_session.execute.return_value = [mock_row] * 3

        features, targets = repo.get_eval_data()

        self.assertEqual(features.shape, (3, 30))
        self.assertEqual(targets.shape, (3,))
        self.assertEqual(targets[0], 0)

    @patch("src.api.repositories.dataset_repository.Cluster")
    def test_feature_cols_count(self, mock_cluster_class):
        self.assertEqual(len(DatasetRepository.FEATURE_COLS), 30)


if __name__ == "__main__":
    unittest.main()
