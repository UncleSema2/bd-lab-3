from dataclasses import dataclass
from typing import Tuple

import numpy as np
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement


@dataclass
class DataRow:
    features: np.ndarray
    target: int


class DatasetRepository:
    FEATURE_COLS = [
        "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
        "mean_smoothness", "mean_compactness", "mean_concavity",
        "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
        "radius_error", "texture_error", "perimeter_error", "area_error",
        "smoothness_error", "compactness_error", "concavity_error",
        "concave_points_error", "symmetry_error", "fractal_dimension_error",
        "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
        "worst_smoothness", "worst_compactness", "worst_concavity",
        "worst_concave_points", "worst_symmetry", "worst_fractal_dimension",
    ]

    def __init__(self, cluster: Cluster, keyspace: str) -> None:
        self.cluster = cluster
        self.keyspace = keyspace
        self.session = cluster.connect(keyspace)

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        query = SimpleStatement(
            f"SELECT {', '.join(self.FEATURE_COLS)}, target FROM train_data"
        )
        rows = self.session.execute(query)

        features = []
        targets = []
        for row in rows:
            features.append([getattr(row, col) for col in self.FEATURE_COLS])
            targets.append(row.target)

        return np.array(features), np.array(targets)

    def get_eval_data(self) -> Tuple[np.ndarray, np.ndarray]:
        query = SimpleStatement(
            f"SELECT {', '.join(self.FEATURE_COLS)}, target FROM eval_data"
        )
        rows = self.session.execute(query)

        features = []
        targets = []
        for row in rows:
            features.append([getattr(row, col) for col in self.FEATURE_COLS])
            targets.append(row.target)

        return np.array(features), np.array(targets)
