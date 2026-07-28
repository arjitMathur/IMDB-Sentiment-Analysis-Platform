"""Model evaluation utilities.

Provides a unified evaluation function that works with any model
implementing the SentimentModel protocol.
"""

from __future__ import annotations

import time

from src.sentiment.logger import get_logger
from src.sentiment.models import SentimentModel

logger = get_logger(__name__)


def evaluate_model(
    model: SentimentModel,
    texts: list[str],
    labels: list[int],
) -> dict:
    """Evaluate a model on a test set using the predict() interface.

    This is a model-agnostic evaluator — works with any SentimentModel.

    Args:
        model: Any model implementing SentimentModel protocol.
        texts: List of raw test texts (cleaning handled by model).
        labels: Corresponding binary labels (0/1).

    Returns:
        Dict with accuracy, per-class metrics, and avg latency.
    """
    correct = 0
    total = len(texts)
    latencies: list[float] = []
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    logger.info("Evaluating %s on %d samples...", model.name, total)
    start = time.time()

    for text, label in zip(texts, labels):
        result = model.predict(text)
        predicted = 1 if result["label"] == "positive" else 0

        if predicted == label:
            correct += 1

        if predicted == 1 and label == 1:
            true_pos += 1
        elif predicted == 1 and label == 0:
            false_pos += 1
        elif predicted == 0 and label == 0:
            true_neg += 1
        else:
            false_neg += 1

        if "latency_ms" in result:
            latencies.append(result["latency_ms"])

    elapsed = time.time() - start
    accuracy = correct / total if total > 0 else 0.0

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "model": model.name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_samples": total,
        "correct": correct,
        "true_positives": true_pos,
        "false_positives": false_pos,
        "true_negatives": true_neg,
        "false_negatives": false_neg,
        "total_eval_time_seconds": round(elapsed, 2),
    }

    if latencies:
        metrics["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2)
        metrics["p95_latency_ms"] = round(sorted(latencies)[int(len(latencies) * 0.95)], 2)

    logger.info(
        "%s — Accuracy: %.4f, F1: %.4f, Avg Latency: %.2fms",
        model.name,
        accuracy,
        f1,
        metrics.get("avg_latency_ms", 0),
    )

    return metrics
