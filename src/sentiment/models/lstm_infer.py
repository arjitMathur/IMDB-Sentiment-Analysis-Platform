"""LSTM inference wrapper — implements the SentimentModel protocol.

This module handles loading a trained LSTM model + tokenizer
and exposing the standard predict() interface. It is intentionally
separate from lstm.py (training) to keep inference lightweight.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.sentiment.config import get_settings
from src.sentiment.logger import get_logger
from src.sentiment.preprocessing import clean_text

logger = get_logger(__name__)


class LSTMModel:
    """LSTM sentiment classifier — inference wrapper.

    Implements the SentimentModel protocol defined in models/__init__.py.

    Attributes:
        _model: Loaded Keras model.
        _tokenizer: Loaded Keras Tokenizer.
        _max_len: Sequence padding length (must match training).
        _model_dir: Directory containing model artifacts.
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        settings = get_settings()
        self._model_dir = model_dir or (settings.model_dir / "lstm")
        self._max_len = settings.max_len
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return "LSTM"

    def load(self) -> None:
        """Load model and tokenizer from disk."""
        model_path = self._model_dir / "sentiment_model.keras"
        tokenizer_path = self._model_dir / "tokenizer.pickle"

        if not model_path.exists():
            raise FileNotFoundError(
                f"LSTM model not found at {model_path}. "
                "Train the model first."
            )
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                "Train the model first."
            )

        self._model = load_model(model_path)

        with open(tokenizer_path, "rb") as f:
            self._tokenizer = pickle.load(f)  # noqa: S301

        # Load max_len from metrics if available
        metrics_path = self._model_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
                self._max_len = metrics.get("max_len", self._max_len)

        logger.info("LSTM model loaded from %s", self._model_dir)

    def predict(self, text: str) -> dict:
        """Predict sentiment for a single text.

        Args:
            text: Raw review text (will be cleaned internally).

        Returns:
            Dict with keys: label, confidence, probabilities, latency_ms.
        """
        self._check_loaded()
        cleaned = clean_text(text)

        if not cleaned:
            return {
                "label": "negative",
                "confidence": 0.5,
                "probabilities": {"positive": 0.5, "negative": 0.5},
                "latency_ms": 0.0,
            }

        start = time.time()
        sequence = self._tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=self._max_len)
        prob = float(self._model.predict(padded, verbose=0)[0][0])
        latency_ms = (time.time() - start) * 1000

        label = "positive" if prob > 0.5 else "negative"
        confidence = prob if label == "positive" else 1.0 - prob

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "positive": round(prob, 4),
                "negative": round(1.0 - prob, 4),
            },
            "latency_ms": round(latency_ms, 2),
        }

    def save(self) -> None:
        """No-op — LSTM is saved during training via train_lstm()."""
        logger.info("LSTM model artifacts are saved during training.")

    def is_available(self) -> bool:
        """Check if saved model artifacts exist."""
        return (
            (self._model_dir / "sentiment_model.keras").exists()
            and (self._model_dir / "tokenizer.pickle").exists()
        )

    def _check_loaded(self) -> None:
        """Raise if model hasn't been loaded."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "LSTM model not loaded. Call load() first."
            )
