"""Lightweight model registry for discovering, loading, and caching models.

The registry scans the models/ directory and provides a unified interface
for the API and Streamlit app to load any available model without hardcoding
filenames or import paths.

Usage:
    registry = ModelRegistry()
    model = registry.load("baseline")  # Returns loaded SentimentModel
    model.predict("Great movie!")

    # Or get whatever is available
    model = registry.get_default()
"""

from __future__ import annotations

import json
from pathlib import Path

from src.sentiment.config import get_settings
from src.sentiment.logger import get_logger
from src.sentiment.models import SentimentModel

logger = get_logger(__name__)

# Map of model keys to their (module_path, class_name) for lazy importing.
# This avoids loading TensorFlow/ONNX until a model is actually requested.
_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "baseline": ("src.sentiment.models.baseline", "BaselineModel"),
    "lstm": ("src.sentiment.models.lstm_infer", "LSTMModel"),
    "transformer": ("src.sentiment.models.transformer_infer", "TransformerModel"),
}


class ModelRegistry:
    """Discovers, loads, and caches sentiment models.

    Models are lazy-loaded on first request and cached for subsequent calls.
    Missing models are handled gracefully (returns None, doesn't crash).
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        settings = get_settings()
        self._model_dir = model_dir or settings.model_dir
        self._cache: dict[str, SentimentModel] = {}

    def list_available(self) -> list[str]:
        """List model keys that have saved artifacts on disk.

        Returns:
            List of available model keys (e.g. ["baseline", "lstm"]).
        """
        available = []
        for key in _MODEL_REGISTRY:
            try:
                model = self._instantiate(key)
                if model.is_available():
                    available.append(key)
            except Exception:
                continue
        return available

    def list_all(self) -> list[str]:
        """List all registered model keys (whether trained or not).

        Returns:
            List of all model keys (e.g. ["baseline", "lstm", "transformer"]).
        """
        return list(_MODEL_REGISTRY.keys())

    def load(self, model_key: str) -> SentimentModel | None:
        """Load a model by key, returning None if unavailable.

        Models are cached after first load.

        Args:
            model_key: One of "baseline", "lstm", "transformer".

        Returns:
            Loaded SentimentModel or None if not available.
        """
        if model_key in self._cache:
            return self._cache[model_key]

        if model_key not in _MODEL_REGISTRY:
            logger.warning("Unknown model key: %s", model_key)
            return None

        try:
            model = self._instantiate(model_key)
            if not model.is_available():
                logger.info("Model '%s' artifacts not found — not trained yet.", model_key)
                return None

            model.load()
            self._cache[model_key] = model
            logger.info("Model '%s' loaded and cached.", model_key)
            return model

        except Exception as e:
            logger.error("Failed to load model '%s': %s", model_key, e)
            return None

    def get_default(self) -> SentimentModel | None:
        """Load the default model (from settings), falling back to any available.

        Returns:
            A loaded SentimentModel, or None if nothing is trained.
        """
        settings = get_settings()

        # Try configured default first
        model = self.load(settings.default_model)
        if model:
            return model

        # Fall back to first available
        for key in _MODEL_REGISTRY:
            model = self.load(key)
            if model:
                logger.info("Default model unavailable, falling back to '%s'.", key)
                return model

        logger.warning("No trained models found.")
        return None

    def get_metrics(self, model_key: str) -> dict | None:
        """Load saved metrics for a model.

        Args:
            model_key: One of "baseline", "lstm", "transformer".

        Returns:
            Dict of metrics or None if not found.
        """
        metrics_dirs = {
            "baseline": self._model_dir / "baseline",
            "lstm": self._model_dir / "lstm",
            "transformer": self._model_dir / "transformer_onnx",
        }

        metrics_dir = metrics_dirs.get(model_key)
        if not metrics_dir:
            return None

        metrics_path = metrics_dir / "metrics.json"
        if not metrics_path.exists():
            return None

        with open(metrics_path) as f:
            return json.load(f)

    def clear_cache(self) -> None:
        """Clear all cached model instances."""
        self._cache.clear()
        logger.info("Model cache cleared.")

    def _instantiate(self, model_key: str) -> SentimentModel:
        """Lazily import and instantiate a model class."""
        module_path, class_name = _MODEL_REGISTRY[model_key]

        import importlib
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        return model_class()
