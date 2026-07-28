"""Tests for the LSTM model inference wrapper.

Uses a tiny model trained in-fixture for speed.
Training tests are integration-level (marked slow) since they require
actual Keras training which takes seconds even on small data.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer

from src.sentiment.models.lstm_infer import LSTMModel


def _create_tiny_lstm(model_dir: Path) -> None:
    """Create and save a minimal LSTM model for testing.

    This creates a 1-layer LSTM with tiny dimensions so tests run in < 1s.
    The model will produce garbage predictions, but the interface works.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    # Tiny model
    model = Sequential([
        Embedding(input_dim=101, output_dim=8),
        LSTM(units=4),
        Dense(units=1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Quick fit so weights aren't random zeros
    X = np.array([[1, 2, 3, 0, 0]] * 4)
    y = np.array([1, 0, 1, 0])
    model.fit(X, y, epochs=1, verbose=0)

    model.save(model_dir / "sentiment_model.keras")

    # Tiny tokenizer
    tokenizer = Tokenizer(num_words=100)
    tokenizer.fit_on_texts(["good movie", "bad film", "great", "terrible awful"])
    with open(model_dir / "tokenizer.pickle", "wb") as f:
        pickle.dump(tokenizer, f)

    # Metrics
    import json
    with open(model_dir / "metrics.json", "w") as f:
        json.dump({"max_len": 5, "model": "LSTM"}, f)


@pytest.fixture
def lstm_dir(tmp_path) -> Path:
    """Create a temp directory with a tiny trained LSTM."""
    d = tmp_path / "lstm"
    _create_tiny_lstm(d)
    return d


@pytest.fixture
def loaded_model(lstm_dir) -> LSTMModel:
    """Provide a loaded LSTMModel instance."""
    model = LSTMModel(model_dir=lstm_dir)
    model.load()
    return model


# ============================================================
# Loading
# ============================================================


class TestLoading:
    """Tests for model loading."""

    def test_load_succeeds(self, lstm_dir):
        model = LSTMModel(model_dir=lstm_dir)
        model.load()  # Should not raise

    def test_load_missing_model_raises(self, tmp_path):
        model = LSTMModel(model_dir=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError, match="not found"):
            model.load()

    def test_is_available_true(self, lstm_dir):
        model = LSTMModel(model_dir=lstm_dir)
        assert model.is_available()

    def test_is_available_false(self, tmp_path):
        model = LSTMModel(model_dir=tmp_path / "nonexistent")
        assert not model.is_available()

    def test_name(self, loaded_model):
        assert loaded_model.name == "LSTM"


# ============================================================
# Prediction Interface
# ============================================================


class TestPrediction:
    """Tests for the predict() interface contract."""

    def test_predict_returns_required_keys(self, loaded_model):
        result = loaded_model.predict("good movie")
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_predict_label_is_valid(self, loaded_model):
        result = loaded_model.predict("some review")
        assert result["label"] in ("positive", "negative")

    def test_predict_confidence_in_range(self, loaded_model):
        result = loaded_model.predict("some review")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_probabilities_sum_to_one(self, loaded_model):
        result = loaded_model.predict("some review")
        probs = result["probabilities"]
        assert abs(probs["positive"] + probs["negative"] - 1.0) < 0.01

    def test_predict_includes_latency(self, loaded_model):
        result = loaded_model.predict("good movie")
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0

    def test_predict_empty_string(self, loaded_model):
        result = loaded_model.predict("")
        assert result["confidence"] == 0.5

    def test_predict_html_input(self, loaded_model):
        """Preprocessing should handle HTML before the model sees it."""
        result = loaded_model.predict("<br />Great movie!<br />")
        assert result["label"] in ("positive", "negative")

    def test_predict_before_load_raises(self, tmp_path):
        model = LSTMModel(model_dir=tmp_path / "lstm")
        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict("test")
