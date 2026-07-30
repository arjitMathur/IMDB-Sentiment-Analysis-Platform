"""DistilBERT training script for sentiment analysis.

This script is designed to run on GPU-enabled platforms (Colab/Kaggle).
It fine-tunes distilbert-base-uncased and exports the result to ONNX.

Usage (on Colab/Kaggle):
    python -m src.sentiment.models.transformer

After training, copy the ONNX model + tokenizer to models/transformer_onnx/
for inference with transformer_infer.py (no PyTorch needed at deployment).
"""

from __future__ import annotations

import os

# Prevent HuggingFace transformers from importing TensorFlow (Keras 3 conflict)
os.environ["TRANSFORMERS_NO_TF"] = "1"

import json
import time
from pathlib import Path

from src.sentiment.config import get_settings
from src.sentiment.logger import get_logger

logger = get_logger(__name__)


def train_transformer(
    model_dir: Path | None = None,
    model_name: str = "distilbert-base-uncased",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 256,
) -> dict:
    """Fine-tune DistilBERT on IMDB and export to ONNX.

    Requires PyTorch and transformers — intended for GPU environments only.

    Args:
        model_dir: Where to save artifacts. Defaults to models/transformer_onnx/.
        model_name: HuggingFace model identifier.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate for AdamW.
        max_length: Maximum token sequence length.

    Returns:
        Dict of training and export metrics.
    """
    # Lazy imports — these are heavy and only needed during training
    try:
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        raise ImportError(
            "Transformer training requires PyTorch and transformers. "
            "Install with: pip install -r requirements-train.txt"
        ) from e

    settings = get_settings()
    model_dir = model_dir or (settings.model_dir / "transformer_onnx")
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fine-tuning %s on IMDB...", model_name)
    start = time.time()

    # Load dataset
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    # Training
    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=100,
        report_to="none",
    )

    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    elapsed = time.time() - start

    # Save tokenizer
    tokenizer.save_pretrained(str(model_dir))

    # Export to ONNX
    logger.info("Exporting model to ONNX...")
    _export_to_onnx(model, tokenizer, model_dir, max_length)

    # Save metrics
    metrics = {
        "model": "DistilBERT (ONNX)",
        "base_model": model_name,
        "test_accuracy": eval_results.get("eval_accuracy", 0.0),
        "test_f1": eval_results.get("eval_f1", 0.0),
        "test_loss": eval_results.get("eval_loss", 0.0),
        "training_time_seconds": round(elapsed, 2),
        "num_epochs": num_epochs,
        "max_length": max_length,
    }

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Transformer trained and exported — Accuracy: %.4f", metrics["test_accuracy"])
    return metrics


def _export_to_onnx(model, tokenizer, model_dir: Path, max_length: int) -> None:
    """Export a HuggingFace model to ONNX format."""
    import torch

    onnx_path = model_dir / "model.onnx"

    # Move model to CPU for ONNX export (avoids device mismatch)
    model = model.cpu()

    dummy_input = tokenizer(
        "This is a test review",
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size"},
            },
            opset_version=14,
        )

    logger.info("ONNX model exported to %s", onnx_path)


if __name__ == "__main__":
    train_transformer()
