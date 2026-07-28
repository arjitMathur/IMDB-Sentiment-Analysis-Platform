"""Keras LSTM model — training and architecture definition.

Preserves the original model architecture (Embedding → LSTM → Dense)
but adds:
    - Centralized preprocessing via clean_text()
    - Common SentimentModel interface
    - Proper save/load with metrics tracking
    - Typed, documented API
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.sentiment.config import get_settings
from src.sentiment.logger import get_logger

logger = get_logger(__name__)


def build_lstm_model(
    num_words: int = 5000,
    embedding_dim: int = 128,
    lstm_units: int = 64,
    dropout_rate: float = 0.2,
) -> Sequential:
    """Build and compile the LSTM model.

    Architecture matches the original project:
        Embedding(5001, 128) → LSTM(64, dropout=0.2) → Dense(1, sigmoid)

    Args:
        num_words: Vocabulary size for the embedding layer.
        embedding_dim: Dimensionality of word embeddings.
        lstm_units: Number of LSTM hidden units.
        dropout_rate: Dropout rate for LSTM regularization.

    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential([
        Embedding(input_dim=num_words + 1, output_dim=embedding_dim),
        LSTM(units=lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate),
        Dense(units=1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("LSTM model built — %d parameters", model.count_params())
    return model


def train_lstm(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    model_dir: Path | None = None,
    num_words: int | None = None,
    max_len: int | None = None,
    embedding_dim: int | None = None,
    lstm_units: int | None = None,
    dropout_rate: float | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    patience: int | None = None,
) -> dict:
    """Train the LSTM model end-to-end.

    Handles tokenization, padding, training, evaluation, and saving.
    All texts should already be cleaned via clean_text().

    Args:
        train_texts: Cleaned training texts.
        train_labels: Binary training labels.
        val_texts: Cleaned validation texts.
        val_labels: Binary validation labels.
        model_dir: Where to save artifacts. Defaults to models/lstm/.
        num_words: Tokenizer vocabulary size.
        max_len: Sequence padding length.
        embedding_dim: Embedding dimensions.
        lstm_units: LSTM hidden units.
        dropout_rate: Dropout rate.
        epochs: Maximum training epochs.
        batch_size: Training batch size.
        patience: Early stopping patience.

    Returns:
        Dict of training metrics and history.
    """
    settings = get_settings()
    model_dir = model_dir or (settings.model_dir / "lstm")
    num_words = num_words or settings.num_words
    max_len = max_len or settings.max_len
    embedding_dim = embedding_dim or settings.embedding_dim
    lstm_units = lstm_units or settings.lstm_units
    dropout_rate = dropout_rate or settings.dropout_rate
    epochs = epochs or settings.epochs
    batch_size = batch_size or settings.batch_size
    patience = patience or settings.patience

    # Tokenize
    logger.info("Fitting tokenizer on %d training texts...", len(train_texts))
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(train_texts)

    X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=max_len)
    X_val = pad_sequences(tokenizer.texts_to_sequences(val_texts), maxlen=max_len)
    Y_train = np.array(train_labels)
    Y_val = np.array(val_labels)

    logger.info("Training shapes — X_train: %s, X_val: %s", X_train.shape, X_val.shape)

    # Build & train
    model = build_lstm_model(num_words, embedding_dim, lstm_units, dropout_rate)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    start = time.time()
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        callbacks=[early_stopping],
        verbose=1,
    )
    elapsed = time.time() - start

    # Evaluate on validation
    val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0)

    # Save artifacts
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "sentiment_model.keras"
    tokenizer_path = model_dir / "tokenizer.pickle"
    metrics_path = model_dir / "metrics.json"

    model.save(model_path)
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    metrics = {
        "model": "LSTM",
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "training_time_seconds": round(elapsed, 2),
        "epochs_completed": len(history.history["loss"]),
        "num_words": num_words,
        "max_len": max_len,
        "embedding_dim": embedding_dim,
        "lstm_units": lstm_units,
        "dropout_rate": dropout_rate,
        "batch_size": batch_size,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("LSTM model saved to %s", model_dir)
    logger.info("Validation accuracy: %.4f, loss: %.4f", val_acc, val_loss)

    return metrics
