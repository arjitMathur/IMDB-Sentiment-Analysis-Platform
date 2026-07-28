"""FastAPI application entrypoint for the Sentiment Analysis API.

Run with:
    uvicorn api.main:app --reload
    uvicorn api.main:app --host 0.0.0.0 --port 8000

API docs available at:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)
"""

from __future__ import annotations

from fastapi import FastAPI

from api.middleware import setup_middleware
from api.routes import health, predict

app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "Production-grade sentiment analysis API supporting multiple models "
        "(TF-IDF baseline, LSTM, DistilBERT ONNX). "
        "Provides single and batch prediction endpoints with model selection."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware (CORS, logging, optional API key)
setup_middleware(app)

# Routes — all under /api/v1
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(health.router, prefix="/api/v1", tags=["Health & Metrics"])


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API docs."""
    return {
        "message": "Sentiment Analysis API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
