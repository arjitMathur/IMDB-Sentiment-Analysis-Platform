"""Pydantic schemas for the FastAPI sentiment analysis API.

Defines request/response models with validation, examples, and documentation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ============================================================
# Request Schemas
# ============================================================


class PredictRequest(BaseModel):
    """Single text prediction request."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Review text to classify (max 5000 characters).",
        examples=["This movie was absolutely amazing! I loved every moment."],
    )
    model: str = Field(
        default="",
        description="Model to use. Leave empty for default. Options: baseline, lstm, transformer.",
        examples=["lstm"],
    )


class BatchPredictRequest(BaseModel):
    """Batch prediction request (up to 32 texts)."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of review texts to classify (max 32).",
    )
    model: str = Field(
        default="",
        description="Model to use. Leave empty for default.",
    )


# ============================================================
# Response Schemas
# ============================================================


class PredictionResult(BaseModel):
    """Prediction result for a single text."""

    label: str = Field(description="Predicted sentiment: 'positive' or 'negative'.")
    confidence: float = Field(description="Confidence in the predicted label (0.0 - 1.0).")
    probabilities: dict[str, float] = Field(description="Class probabilities.")
    latency_ms: float = Field(default=0.0, description="Inference latency in milliseconds.")


class PredictResponse(BaseModel):
    """Response for single prediction endpoint."""

    success: bool = True
    model: str = Field(description="Model used for prediction.")
    prediction: PredictionResult


class BatchPredictResponse(BaseModel):
    """Response for batch prediction endpoint."""

    success: bool = True
    model: str = Field(description="Model used for prediction.")
    predictions: list[PredictionResult]
    count: int = Field(description="Number of predictions returned.")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    models_available: list[str] = Field(description="List of trained model keys.")
    models_registered: list[str] = Field(description="List of all registered model keys.")


class MetricsResponse(BaseModel):
    """Metrics for a single model."""

    model: str
    metrics: dict | None = Field(description="Saved benchmark metrics, or null if unavailable.")


class AllMetricsResponse(BaseModel):
    """Aggregated metrics for all models."""

    models: list[MetricsResponse]


class ErrorResponse(BaseModel):
    """Error response."""

    success: bool = False
    error: str = Field(description="Human-readable error message.")
