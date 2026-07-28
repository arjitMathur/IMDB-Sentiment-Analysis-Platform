"""Prediction routes — POST /predict and POST /batch."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
    PredictionResult,
)
from src.sentiment.registry import ModelRegistry

router = APIRouter()
_registry = ModelRegistry()


def _get_model(model_key: str):
    """Resolve a model from the registry, raising 404 if unavailable."""
    if model_key:
        model = _registry.load(model_key)
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_key}' not found or not trained yet. "
                       f"Available: {_registry.list_available()}",
            )
        return model

    model = _registry.get_default()
    if not model:
        raise HTTPException(
            status_code=503,
            detail="No trained models available. Train a model first.",
        )
    return model


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict sentiment for a single text",
    responses={404: {"description": "Model not found"}, 503: {"description": "No models available"}},
)
async def predict(request: PredictRequest) -> PredictResponse:
    """Predict whether a review is positive or negative.

    Optionally specify a model (baseline, lstm, transformer).
    If no model is specified, the default model is used.
    """
    model = _get_model(request.model)
    result = model.predict(request.text)

    return PredictResponse(
        model=model.name,
        prediction=PredictionResult(**result),
    )


@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Predict sentiment for a batch of texts (max 32)",
    responses={404: {"description": "Model not found"}, 503: {"description": "No models available"}},
)
async def batch_predict(request: BatchPredictRequest) -> BatchPredictResponse:
    """Predict sentiment for multiple reviews in one request.

    Maximum 32 texts per batch. Each text is limited to 5000 characters.
    """
    model = _get_model(request.model)

    predictions = []
    for text in request.texts:
        result = model.predict(text)
        predictions.append(PredictionResult(**result))

    return BatchPredictResponse(
        model=model.name,
        predictions=predictions,
        count=len(predictions),
    )
