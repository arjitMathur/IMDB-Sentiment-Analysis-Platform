"""Tests for the model registry.

Uses mocked model availability to avoid needing actual trained models.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import json

import pytest

from src.sentiment.registry import ModelRegistry


# ============================================================
# Listing
# ============================================================


class TestListing:
    """Tests for list_available() and list_all()."""

    def test_list_all_returns_known_keys(self):
        registry = ModelRegistry()
        keys = registry.list_all()
        assert "baseline" in keys
        assert "lstm" in keys
        assert "transformer" in keys

    def test_list_available_empty_when_no_models(self, tmp_path):
        registry = ModelRegistry(model_dir=tmp_path / "empty")
        mock_model = MagicMock()
        mock_model.is_available.return_value = False
        with patch.object(registry, "_instantiate", return_value=mock_model):
            available = registry.list_available()
        assert available == []


# ============================================================
# Loading
# ============================================================


class TestLoading:
    """Tests for model loading."""

    def test_load_unknown_key_returns_none(self, tmp_path):
        registry = ModelRegistry(model_dir=tmp_path)
        result = registry.load("nonexistent_model")
        assert result is None

    def test_load_unavailable_returns_none(self, tmp_path):
        registry = ModelRegistry(model_dir=tmp_path)
        mock_model = MagicMock()
        mock_model.is_available.return_value = False
        with patch.object(registry, "_instantiate", return_value=mock_model):
            result = registry.load("baseline")
        assert result is None

    def test_load_caches_model(self, tmp_path):
        """Once loaded, subsequent calls return the cached instance."""
        registry = ModelRegistry(model_dir=tmp_path)

        # Manually insert a mock into the cache
        mock_model = MagicMock()
        mock_model.name = "test"
        registry._cache["baseline"] = mock_model

        result = registry.load("baseline")
        assert result is mock_model

    def test_clear_cache(self, tmp_path):
        registry = ModelRegistry(model_dir=tmp_path)
        registry._cache["baseline"] = MagicMock()
        registry.clear_cache()
        assert len(registry._cache) == 0


# ============================================================
# Default Model
# ============================================================


class TestDefaultModel:
    """Tests for get_default()."""

    def test_get_default_returns_none_when_nothing_trained(self, tmp_path):
        registry = ModelRegistry(model_dir=tmp_path)
        mock_model = MagicMock()
        mock_model.is_available.return_value = False
        with patch.object(registry, "_instantiate", return_value=mock_model):
            result = registry.get_default()
        assert result is None

    def test_get_default_returns_cached(self, tmp_path):
        registry = ModelRegistry(model_dir=tmp_path)
        mock_model = MagicMock()
        mock_model.name = "test"

        # Pre-cache the configured default model
        with patch("src.sentiment.registry.get_settings") as mock_settings:
            mock_settings.return_value.default_model = "lstm"
            mock_settings.return_value.model_dir = tmp_path
            registry._cache["lstm"] = mock_model
            result = registry.get_default()
            assert result is mock_model


# ============================================================
# Metrics
# ============================================================


class TestMetrics:
    """Tests for get_metrics()."""

    def test_get_metrics_returns_none_when_missing(self, tmp_path):
        registry = ModelRegistry(model_dir=tmp_path)
        assert registry.get_metrics("baseline") is None

    def test_get_metrics_reads_json(self, tmp_path):
        # Create metrics file
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        metrics = {"test_accuracy": 0.89, "model": "baseline"}
        with open(baseline_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)

        registry = ModelRegistry(model_dir=tmp_path)
        loaded = registry.get_metrics("baseline")
        assert loaded["test_accuracy"] == 0.89

    def test_get_metrics_unknown_key(self, tmp_path):
        registry = ModelRegistry(model_dir=tmp_path)
        assert registry.get_metrics("unknown") is None
