from uuid import UUID
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

from src.api.schemas import PredictionRecord


class PredictionRepository:
    def __init__(self, cluster: Cluster, keyspace: str) -> None:
        self.cluster = cluster
        self.keyspace = keyspace
        self.session = None
        self._init_database()

    def _init_database(self):
        self.session = self.cluster.connect()

        self.session.execute(
            f"CREATE KEYSPACE IF NOT EXISTS {self.keyspace} "
            "WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'}"
        )

        self.session.set_keyspace(self.keyspace)

        self.session.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id UUID PRIMARY KEY,
                features LIST<double>,
                prediction int,
                probability_malignant double,
                probability_benign double,
                created_at timestamp,
                model_version text
            )
            """
        )

    async def save(self, prediction: PredictionRecord) -> None:
        stmt = SimpleStatement(
            """
            INSERT INTO predictions (
                prediction_id, features, prediction,
                probability_malignant, probability_benign,
                created_at, model_version
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
        )
        self.session.execute(
            stmt,
            (
                UUID(prediction.prediction_id),
                prediction.features,
                prediction.prediction,
                prediction.probability_malignant,
                prediction.probability_benign,
                prediction.created_at,
                prediction.model_version,
            ),
        )

    async def get_last(self, limit: int) -> list[PredictionRecord]:
        stmt = SimpleStatement(
            f"SELECT * FROM predictions ORDER BY created_at DESC LIMIT {limit}"
        )
        rows = self.session.execute(stmt)

        result = []
        for row in rows:
            result.append(
                PredictionRecord(
                    prediction_id=str(row.prediction_id),
                    features=row.features,
                    prediction=row.prediction,
                    probability_malignant=row.probability_malignant,
                    probability_benign=row.probability_benign,
                    created_at=row.created_at,
                    model_version=row.model_version,
                )
            )

        return result
