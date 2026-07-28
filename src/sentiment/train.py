"""Unified training entrypoint for all models.

Usage:
    python -m src.sentiment.train baseline
    python -m src.sentiment.train lstm
    python -m src.sentiment.train all
"""

from __future__ import annotations

import sys

from src.sentiment.data import load_imdb_data
from src.sentiment.logger import get_logger
from src.sentiment.models.baseline import BaselineModel

logger = get_logger(__name__)


def train_baseline() -> dict:
    """Train the TF-IDF + Logistic Regression baseline."""
    logger.info("=" * 60)
    logger.info("Training Baseline Model (TF-IDF + Logistic Regression)")
    logger.info("=" * 60)

    data = load_imdb_data()
    model = BaselineModel()

    metrics = model.train(
        train_texts=data.train.texts,
        train_labels=data.train.labels,
        val_texts=data.val.texts,
        val_labels=data.val.labels,
    )

    # Evaluate on test set
    test_metrics = model.evaluate(data.test.texts, data.test.labels)
    metrics.update(test_metrics)

    model.save()
    model.save_metrics(metrics)

    logger.info("Baseline — Test Accuracy: %.4f, F1: %.4f", metrics["test_accuracy"], metrics["test_f1"])
    return metrics


def train_lstm_model() -> dict:
    """Train the LSTM model."""
    from src.sentiment.models.lstm import train_lstm

    logger.info("=" * 60)
    logger.info("Training LSTM Model")
    logger.info("=" * 60)

    data = load_imdb_data()

    metrics = train_lstm(
        train_texts=data.train.texts,
        train_labels=data.train.labels,
        val_texts=data.val.texts,
        val_labels=data.val.labels,
    )

    logger.info("LSTM — Val Accuracy: %.4f", metrics["val_accuracy"])
    return metrics


def train_all() -> dict[str, dict]:
    """Train all available models."""
    results = {}

    logger.info("Training all models...")

    results["baseline"] = train_baseline()
    results["lstm"] = train_lstm_model()

    logger.info("=" * 60)
    logger.info("All models trained successfully!")
    for name, metrics in results.items():
        acc_key = "test_accuracy" if "test_accuracy" in metrics else "val_accuracy"
        logger.info("  %s: %s = %.4f", name, acc_key, metrics[acc_key])
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "baseline":
        train_baseline()
    elif target == "lstm":
        train_lstm_model()
    elif target == "all":
        train_all()
    else:
        print(f"Unknown target: {target}. Use: baseline, lstm, all")
        sys.exit(1)
