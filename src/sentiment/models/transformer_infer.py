"""DistilBERT ONNX inference wrapper — NO PyTorch required.

Uses onnxruntime for inference and HuggingFace tokenizer for text encoding.
This keeps the deployment footprint small (no 2GB+ PyTorch install).

Implements the SentimentModel protocol.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.sentiment.config import get_settings
from src.sentiment.logger import get_logger
from src.sentiment.preprocessing import clean_text

logger = get_logger(__name__)


class TransformerModel:
    """DistilBERT sentiment classifier using ONNX Runtime.

    Implements the SentimentModel protocol defined in models/__init__.py.
    Does NOT require PyTorch — only onnxruntime and transformers (tokenizer).
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        settings = get_settings()
        self._model_dir = model_dir or (settings.model_dir / "transformer_onnx")
        self._session = None
        self._tokenizer = None
        self._max_length: int = 256

    @property
    def name(self) -> str:
        return "DistilBERT (ONNX)"

    def load(self) -> None:
        """Load ONNX model and tokenizer from disk."""
        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_path = self._model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}. "
                "Train and export the model first."
            )

        # Load tokenizer
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
        except Exception as e:
            raise FileNotFoundError(
                f"Tokenizer not found in {self._model_dir}. "
                "Ensure tokenizer files were saved during training."
            ) from e

        # Load ONNX session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(onnx_path), sess_options)

        # Load max_length from metrics if available
        metrics_path = self._model_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
                self._max_length = metrics.get("max_length", self._max_length)

        logger.info("Transformer ONNX model loaded from %s", self._model_dir)

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

        # Tokenize
        encoding = self._tokenizer(
            cleaned,
            return_tensors="np",
            max_length=self._max_length,
            truncation=True,
            padding="max_length",
        )

        # Run ONNX inference
        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)

        outputs = self._session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )

        logits = outputs[0][0]
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        latency_ms = (time.time() - start) * 1000

        neg_prob, pos_prob = float(probs[0]), float(probs[1])
        label = "positive" if pos_prob > 0.5 else "negative"
        confidence = pos_prob if label == "positive" else neg_prob

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "positive": round(pos_prob, 4),
                "negative": round(neg_prob, 4),
            },
            "latency_ms": round(latency_ms, 2),
        }

    def save(self) -> None:
        """No-op — transformer is saved during training/export."""
        logger.info("Transformer artifacts are saved during training.")

    def is_available(self) -> bool:
        """Check if ONNX model and tokenizer exist."""
        return (
            (self._model_dir / "model.onnx").exists()
            and (self._model_dir / "tokenizer_config.json").exists()
        )

    def _check_loaded(self) -> None:
        """Raise if model hasn't been loaded."""
        if self._session is None or self._tokenizer is None:
            raise RuntimeError(
                "Transformer model not loaded. Call load() first."
            )
