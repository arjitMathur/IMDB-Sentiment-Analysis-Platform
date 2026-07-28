"""Health and metrics routes."""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import AllMetricsResponse, HealthResponse, MetricsResponse
from src.sentiment.registry import ModelRegistry

router = APIRouter()
_registry = ModelRegistry()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health() -> HealthResponse:
    """Check API health and list available models."""
    return HealthResponse(
        status="healthy",
        models_available=_registry.list_available(),
        models_registered=_registry.list_all(),
    )


@router.get(
    "/metrics",
    response_model=AllMetricsResponse,
    summary="Get benchmark metrics for all models",
)
async def metrics() -> AllMetricsResponse:
    """Return saved benchmark metrics for all registered models."""
    results = []
    for key in _registry.list_all():
        m = _registry.get_metrics(key)
        results.append(MetricsResponse(model=key, metrics=m))

    return AllMetricsResponse(models=results)


@router.get(
    "/models",
    summary="List all registered models and their availability",
)
async def list_models() -> dict:
    """List all models with their training status."""
    available = set(_registry.list_available())
    return {
        "models": [
            {"key": key, "available": key in available}
            for key in _registry.list_all()
        ]
    }
