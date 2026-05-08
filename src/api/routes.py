from fastapi import APIRouter, HTTPException, Request
from fastapi.params import Query

from src.api.schemas import (
    PredictRequest,
    PredictResponse,
    PredictionRecord,
    TrainResponse,
    EvaluateResponse,
)

router = APIRouter()


def get_service(request: Request):
    return request.app.state.prediction_service


def get_model_service(request: Request):
    return request.app.state.model_service


@router.post("/train", response_model=TrainResponse)
async def train(request: Request, model: str = "LOG_REG"):
    service = get_model_service(request)
    return await service.train(model=model)


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: Request, model: str = "LOG_REG"):
    service = get_model_service(request)
    return await service.evaluate(model=model)


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, data: PredictRequest):
    service = get_service(request)

    try:
        result = await service.predict_and_save(
            features=data.features,
            model=data.model,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return PredictResponse(**result.model_dump())


@router.get("/predictions", response_model=list[PredictionRecord])
async def get_last_predictions(
    request: Request,
    limit: int = Query(10, ge=1, le=100),
):
    service = get_service(request)
    return await service.get_last_predictions(limit)
