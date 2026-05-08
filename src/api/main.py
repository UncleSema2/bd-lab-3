import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from src.api.routes import router
from src.api.repositories.prediction_repository import PredictionRepository
from src.api.repositories.dataset_repository import DatasetRepository
from src.api.services.prediction_service import PredictionService
from src.api.services.model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    cassandra_host = os.environ["CASSANDRA_HOST"]
    cassandra_port = int(os.environ["CASSANDRA_PORT"])
    cassandra_username = os.environ["CASSANDRA_USERNAME"]
    cassandra_password = os.environ["CASSANDRA_PASSWORD"]
    cassandra_keyspace = os.environ["CASSANDRA_KEYSPACE"]

    model_version = os.getenv("MODEL_VERSION", "1.0.0")

    auth_provider = None
    if cassandra_username and cassandra_password:
        auth_provider = PlainTextAuthProvider(
            username=cassandra_username,
            password=cassandra_password,
        )

    cluster = Cluster(
        [cassandra_host],
        port=cassandra_port,
        auth_provider=auth_provider,
    )
    cluster.connect()

    repository = PredictionRepository(cluster, cassandra_keyspace)
    dataset_repository = DatasetRepository(cluster, cassandra_keyspace)

    prediction_service = PredictionService(
        config_path="config.ini",
        model_version=model_version,
        prediction_repository=repository,
    )

    model_service = ModelService(
        config_path="config.ini",
        dataset_repository=dataset_repository,
    )

    app.state.cluster = cluster
    app.state.prediction_service = prediction_service
    app.state.model_service = model_service

    try:
        yield
    finally:
        cluster.shutdown()


app = FastAPI(
    title="Breast Cancer Classifier",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
