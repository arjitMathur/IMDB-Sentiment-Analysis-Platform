"""Tests for the transformer ONNX inference wrapper.

Since we can't easily create a real ONNX model in tests without PyTorch,
these tests focus on:
    - Interface contract (when model IS available)
    - Error handling (when model is NOT available)
    - is_available() checks

A mock-based approach tests the predict interface without a real model.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.sentiment.models.transformer_infer import TransformerModel


# ============================================================
# Availability & Error Handling
# ============================================================


class TestAvailability:
    """Tests for model availability checks."""

    def test_is_available_false_when_missing(self, tmp_path):
        model = TransformerModel(model_dir=tmp_path / "nonexistent")
        assert not model.is_available()

    def test_load_missing_raises(self, tmp_path):
        model = TransformerModel(model_dir=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            model.load()

    def test_predict_before_load_raises(self, tmp_path):
        model = TransformerModel(model_dir=tmp_path / "transformer")
        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict("test")

    def test_name(self):
        model = TransformerModel()
        assert model.name == "DistilBERT (ONNX)"


# ============================================================
# Prediction Interface (mocked)
# ============================================================


class TestPredictionMocked:
    """Tests with mocked ONNX session to verify predict() interface."""

    def _make_loaded_model(self, tmp_path) -> TransformerModel:
        """Create a TransformerModel with mocked internals."""
        model = TransformerModel(model_dir=tmp_path)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[101, 2023, 2003, 1037, 3231, 102] + [0] * 250]),
            "attention_mask": np.array([[1, 1, 1, 1, 1, 1] + [0] * 250]),
        }
        model._tokenizer = mock_tokenizer

        # Mock ONNX session — returns logits [neg_score, pos_score]
        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[0.1, 2.5]])]  # strongly positive
        model._session = mock_session

        return model

    def test_predict_returns_required_keys(self, tmp_path):
        model = self._make_loaded_model(tmp_path)
        result = model.predict("Great movie!")
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "latency_ms" in result

    def test_predict_label_is_valid(self, tmp_path):
        model = self._make_loaded_model(tmp_path)
        result = model.predict("Great movie!")
        assert result["label"] in ("positive", "negative")

    def test_predict_positive_logits(self, tmp_path):
        model = self._make_loaded_model(tmp_path)
        result = model.predict("Great movie!")
        # logits [0.1, 2.5] → softmax → positive > 0.5
        assert result["label"] == "positive"

    def test_predict_negative_logits(self, tmp_path):
        model = self._make_loaded_model(tmp_path)
        model._session.run.return_value = [np.array([[2.5, 0.1]])]  # strongly negative
        result = model.predict("Terrible movie!")
        assert result["label"] == "negative"

    def test_predict_confidence_in_range(self, tmp_path):
        model = self._make_loaded_model(tmp_path)
        result = model.predict("Some review")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_probabilities_sum_to_one(self, tmp_path):
        model = self._make_loaded_model(tmp_path)
        result = model.predict("Some review")
        probs = result["probabilities"]
        assert abs(probs["positive"] + probs["negative"] - 1.0) < 0.01

    def test_predict_empty_string(self, tmp_path):
        model = self._make_loaded_model(tmp_path)
        result = model.predict("")
        assert result["confidence"] == 0.5

    def test_predict_latency_non_negative(self, tmp_path):
        model = self._make_loaded_model(tmp_path)
        result = model.predict("Some review")
        assert result["latency_ms"] >= 0
