"""Tests for the FastAPI sentiment analysis API.

Uses FastAPI TestClient for integration-style testing.
Models are mocked to avoid needing trained artifacts.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ============================================================
# Health & Info Endpoints
# ============================================================


class TestHealthEndpoints:
    """Tests for health, metrics, and model listing."""

    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "docs" in data

    def test_health_returns_200(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_available" in data
        assert "models_registered" in data

    def test_metrics_returns_200(self):
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data

    def test_models_list_returns_200(self):
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        # All registered models should appear
        keys = [m["key"] for m in data["models"]]
        assert "baseline" in keys
        assert "lstm" in keys
        assert "transformer" in keys


# ============================================================
# Prediction Endpoints
# ============================================================


class TestPredictEndpoint:
    """Tests for POST /api/v1/predict."""

    @patch("api.routes.predict._registry")
    def test_predict_with_mocked_model(self, mock_registry):
        mock_model = MagicMock()
        mock_model.name = "MockModel"
        mock_model.predict.return_value = {
            "label": "positive",
            "confidence": 0.95,
            "probabilities": {"positive": 0.95, "negative": 0.05},
            "latency_ms": 1.2,
        }
        mock_registry.get_default.return_value = mock_model

        response = client.post("/api/v1/predict", json={"text": "Great movie!"})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["prediction"]["label"] == "positive"
        assert data["model"] == "MockModel"

    @patch("api.routes.predict._registry")
    def test_predict_with_specific_model(self, mock_registry):
        mock_model = MagicMock()
        mock_model.name = "Baseline"
        mock_model.predict.return_value = {
            "label": "negative",
            "confidence": 0.8,
            "probabilities": {"positive": 0.2, "negative": 0.8},
            "latency_ms": 0.5,
        }
        mock_registry.load.return_value = mock_model

        response = client.post("/api/v1/predict", json={"text": "Bad movie!", "model": "baseline"})
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"]["label"] == "negative"

    @patch("api.routes.predict._registry")
    def test_predict_model_not_found(self, mock_registry):
        mock_registry.load.return_value = None
        mock_registry.list_available.return_value = []

        response = client.post("/api/v1/predict", json={"text": "test", "model": "nonexistent"})
        assert response.status_code == 404

    @patch("api.routes.predict._registry")
    def test_predict_no_models_available(self, mock_registry):
        mock_registry.get_default.return_value = None

        response = client.post("/api/v1/predict", json={"text": "test"})
        assert response.status_code == 503

    def test_predict_empty_text_rejected(self):
        response = client.post("/api/v1/predict", json={"text": ""})
        assert response.status_code == 422  # Pydantic validation

    def test_predict_text_too_long(self):
        response = client.post("/api/v1/predict", json={"text": "a" * 5001})
        assert response.status_code == 422

    def test_predict_missing_text(self):
        response = client.post("/api/v1/predict", json={})
        assert response.status_code == 422


# ============================================================
# Batch Prediction
# ============================================================


class TestBatchEndpoint:
    """Tests for POST /api/v1/batch."""

    @patch("api.routes.predict._registry")
    def test_batch_predict(self, mock_registry):
        mock_model = MagicMock()
        mock_model.name = "MockModel"
        mock_model.predict.return_value = {
            "label": "positive",
            "confidence": 0.9,
            "probabilities": {"positive": 0.9, "negative": 0.1},
            "latency_ms": 1.0,
        }
        mock_registry.get_default.return_value = mock_model

        response = client.post("/api/v1/batch", json={"texts": ["Good", "Great"]})
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_empty_list_rejected(self):
        response = client.post("/api/v1/batch", json={"texts": []})
        assert response.status_code == 422

    def test_batch_too_many_texts(self):
        response = client.post("/api/v1/batch", json={"texts": ["x"] * 33})
        assert response.status_code == 422
