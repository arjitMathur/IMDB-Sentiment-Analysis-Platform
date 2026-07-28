"""Centralized configuration for the sentiment analysis platform.

All magic numbers, paths, and hyperparameters live here.
Supports override via environment variables or .env file.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


# Project root is two levels up from this file (src/sentiment/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application-wide settings. Override via env vars or .env file."""

    # ---- Paths ----
    model_dir: Path = PROJECT_ROOT / "models"
    data_cache_dir: Path = PROJECT_ROOT / "data"

    # ---- Preprocessing ----
    max_len: int = 200
    num_words: int = 5000

    # ---- LSTM Hyperparameters ----
    embedding_dim: int = 128
    lstm_units: int = 64
    dropout_rate: float = 0.2
    epochs: int = 9
    batch_size: int = 128
    patience: int = 3
    test_size: float = 0.20
    val_test_split: float = 0.50

    # ---- Model Selection ----
    default_model: str = "lstm"

    # ---- API ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""
    api_workers: int = 1

    # ---- Streamlit ----
    streamlit_port: int = 8501
    api_base_url: str = "http://localhost:8000"

    # ---- Logging ----
    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Factory for settings. Allows easy mocking in tests."""
    return Settings()
