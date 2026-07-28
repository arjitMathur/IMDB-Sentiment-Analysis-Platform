"""API middleware — CORS, request logging, optional API key auth."""

from __future__ import annotations

import time

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from src.sentiment.config import get_settings
from src.sentiment.logger import get_logger

logger = get_logger(__name__)


def setup_middleware(app: FastAPI) -> None:
    """Configure all middleware on the FastAPI application."""

    # CORS — allow all origins for demo, restrict in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next) -> Response:
        """Log every request with method, path, status, and latency."""
        start = time.time()
        response: Response = await call_next(request)
        elapsed_ms = (time.time() - start) * 1000

        logger.info(
            "%s %s → %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.middleware("http")
    async def check_api_key(request: Request, call_next) -> Response:
        """Optional API key authentication.

        If API_KEY is set in environment/settings, all non-health endpoints
        require an X-API-Key header. If API_KEY is empty, auth is disabled.
        """
        settings = get_settings()

        if settings.api_key:
            # Skip auth for health, docs, and openapi schema
            exempt_paths = {"/api/v1/health", "/docs", "/redoc", "/openapi.json"}
            if request.url.path not in exempt_paths:
                provided_key = request.headers.get("X-API-Key", "")
                if provided_key != settings.api_key:
                    from fastapi.responses import JSONResponse
                    return JSONResponse(
                        status_code=401,
                        content={"success": False, "error": "Invalid or missing API key."},
                    )

        return await call_next(request)
