"""Download pre-trained model artifacts from HuggingFace Hub.

On cloud deployments (e.g., HuggingFace Spaces), models aren't stored locally.
This module downloads them on first launch and caches them in models/.

Usage:
    from src.sentiment.model_downloader import ensure_models
    ensure_models()  # Downloads only if not already present
"""

from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from src.sentiment.config import get_settings
from src.sentiment.logger import get_logger

logger = get_logger(__name__)

# HuggingFace repository where model artifacts are stored
HF_REPO_ID = "arjitMathur/imdb-sentiment-models"


def ensure_models(model_dir: Path | None = None) -> None:
    """Download model artifacts if they don't exist locally.

    Checks each model subdirectory and downloads from HuggingFace Hub
    only if the key file is missing.
    """
    settings = get_settings()
    model_dir = model_dir or settings.model_dir

    _ensure_baseline(model_dir / "baseline")
    _ensure_lstm(model_dir / "lstm")
    _ensure_transformer(model_dir / "transformer_onnx")


def _ensure_baseline(target_dir: Path) -> None:
    """Download baseline model if not present."""
    if (target_dir / "tfidf_vectorizer.joblib").exists():
        logger.info("Baseline model already present.")
        return

    logger.info("Downloading baseline model from HuggingFace Hub...")
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in [
        "baseline/tfidf_vectorizer.joblib",
        "baseline/logistic_regression.joblib",
        "baseline/metrics.json",
    ]:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
        shutil.copy2(path, target_dir / Path(filename).name)

    logger.info("Baseline model downloaded.")


def _ensure_lstm(target_dir: Path) -> None:
    """Download LSTM model if not present."""
    if (target_dir / "sentiment_model.keras").exists():
        logger.info("LSTM model already present.")
        return

    logger.info("Downloading LSTM model from HuggingFace Hub...")
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in [
        "lstm/sentiment_model.keras",
        "lstm/tokenizer.pickle",
        "lstm/metrics.json",
    ]:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
        shutil.copy2(path, target_dir / Path(filename).name)

    logger.info("LSTM model downloaded.")


def _ensure_transformer(target_dir: Path) -> None:
    """Download transformer ONNX model if not present."""
    if (target_dir / "model.onnx").exists():
        logger.info("Transformer model already present.")
        return

    logger.info("Downloading transformer ONNX model from HuggingFace Hub...")
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in [
        "transformer_onnx/model.onnx",
        "transformer_onnx/tokenizer.json",
        "transformer_onnx/tokenizer_config.json",
        "transformer_onnx/vocab.txt",
        "transformer_onnx/special_tokens_map.json",
        "transformer_onnx/metrics.json",
    ]:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
        shutil.copy2(path, target_dir / Path(filename).name)

    logger.info("Transformer model downloaded.")
