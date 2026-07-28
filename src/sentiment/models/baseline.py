"""TF-IDF + Logistic Regression baseline model.

A fast, interpretable classical ML baseline that serves two purposes:
    1. Provides a performance floor for the LSTM and transformer models to beat.
    2. Demonstrates that strong results are achievable without deep learning.

Implements the SentimentModel protocol defined in models/__init__.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.sentiment.config import get_settings
from src.sentiment.logger import get_logger
from src.sentiment.preprocessing import clean_text

logger = get_logger(__name__)


class BaselineModel:
    """TF-IDF + Logistic Regression sentiment classifier.

    Attributes:
        _vectorizer: Fitted TF-IDF vectorizer.
        _classifier: Trained Logistic Regression model.
        _model_dir: Directory where model artifacts are saved/loaded.
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        settings = get_settings()
        self._model_dir = model_dir or (settings.model_dir / "baseline")
        self._vectorizer: TfidfVectorizer | None = None
        self._classifier: LogisticRegression | None = None

    @property
    def name(self) -> str:
        return "TF-IDF + Logistic Regression"

    # ---- Training ----

    def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str] | None = None,
        val_labels: list[int] | None = None,
        max_features: int = 50_000,
        ngram_range: tuple[int, int] = (1, 2),
        max_iter: int = 1000,
        C: float = 1.0,
    ) -> dict:
        """Train the baseline model.

        Args:
            train_texts: List of already-cleaned training texts.
            train_labels: Corresponding binary labels (0/1).
            val_texts: Optional validation texts for evaluation.
            val_labels: Optional validation labels.
            max_features: Maximum vocabulary size for TF-IDF.
            ngram_range: N-gram range for TF-IDF (default unigrams + bigrams).
            max_iter: Max iterations for logistic regression solver.
            C: Inverse regularization strength.

        Returns:
            Dict of training metrics.
        """
        logger.info("Training baseline model (TF-IDF + LogReg)...")
        start = time.time()

        # Fit TF-IDF vectorizer
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        X_train = self._vectorizer.fit_transform(train_texts)
        logger.info("TF-IDF vocabulary size: %d", len(self._vectorizer.vocabulary_))

        # Train logistic regression
        self._classifier = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=42,
        )
        self._classifier.fit(X_train, train_labels)

        elapsed = time.time() - start
        logger.info("Training completed in %.2fs", elapsed)

        # Compute metrics
        train_preds = self._classifier.predict(X_train)
        metrics: dict = {
            "model": self.name,
            "train_accuracy": float(accuracy_score(train_labels, train_preds)),
            "train_f1": float(f1_score(train_labels, train_preds)),
            "training_time_seconds": round(elapsed, 2),
            "vocab_size": len(self._vectorizer.vocabulary_),
            "max_features": max_features,
            "ngram_range": list(ngram_range),
            "C": C,
        }

        if val_texts and val_labels:
            X_val = self._vectorizer.transform(val_texts)
            val_preds = self._classifier.predict(X_val)
            metrics["val_accuracy"] = float(accuracy_score(val_labels, val_preds))
            metrics["val_f1"] = float(f1_score(val_labels, val_preds))
            logger.info("Validation accuracy: %.4f", metrics["val_accuracy"])

        return metrics

    def evaluate(self, texts: list[str], labels: list[int]) -> dict:
        """Evaluate the model on a test set.

        Args:
            texts: List of already-cleaned test texts.
            labels: Corresponding binary labels.

        Returns:
            Dict of evaluation metrics.
        """
        self._check_loaded()
        X = self._vectorizer.transform(texts)
        preds = self._classifier.predict(X)

        report = classification_report(labels, preds, target_names=["negative", "positive"], output_dict=True)

        return {
            "test_accuracy": float(accuracy_score(labels, preds)),
            "test_f1": float(f1_score(labels, preds)),
            "classification_report": report,
        }

    # ---- Inference ----

    def predict(self, text: str) -> dict:
        """Predict sentiment for a single text.

        Args:
            text: Raw review text (will be cleaned internally).

        Returns:
            Dict with keys: label, confidence, probabilities.
        """
        self._check_loaded()
        cleaned = clean_text(text)

        if not cleaned:
            return {
                "label": "negative",
                "confidence": 0.5,
                "probabilities": {"positive": 0.5, "negative": 0.5},
            }

        start = time.time()
        X = self._vectorizer.transform([cleaned])
        proba = self._classifier.predict_proba(X)[0]
        latency_ms = (time.time() - start) * 1000

        # Classes are [0, 1] → [negative, positive]
        neg_prob, pos_prob = float(proba[0]), float(proba[1])
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

    # ---- Persistence ----

    def save(self) -> None:
        """Save model artifacts to disk."""
        self._check_loaded()
        self._model_dir.mkdir(parents=True, exist_ok=True)

        vectorizer_path = self._model_dir / "tfidf_vectorizer.joblib"
        classifier_path = self._model_dir / "logistic_regression.joblib"

        joblib.dump(self._vectorizer, vectorizer_path)
        joblib.dump(self._classifier, classifier_path)

        logger.info("Baseline model saved to %s", self._model_dir)

    def save_metrics(self, metrics: dict) -> None:
        """Save metrics JSON alongside model artifacts."""
        self._model_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self._model_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved to %s", metrics_path)

    def load(self) -> None:
        """Load model artifacts from disk."""
        vectorizer_path = self._model_dir / "tfidf_vectorizer.joblib"
        classifier_path = self._model_dir / "logistic_regression.joblib"

        if not vectorizer_path.exists() or not classifier_path.exists():
            raise FileNotFoundError(
                f"Baseline model artifacts not found in {self._model_dir}. "
                "Train the model first with `train()`."
            )

        self._vectorizer = joblib.load(vectorizer_path)
        self._classifier = joblib.load(classifier_path)
        logger.info("Baseline model loaded from %s", self._model_dir)

    def is_available(self) -> bool:
        """Check if saved model artifacts exist."""
        return (
            (self._model_dir / "tfidf_vectorizer.joblib").exists()
            and (self._model_dir / "logistic_regression.joblib").exists()
        )

    # ---- Private ----

    def _check_loaded(self) -> None:
        """Raise if model hasn't been trained or loaded."""
        if self._vectorizer is None or self._classifier is None:
            raise RuntimeError(
                "Model not loaded. Call train() or load() first."
            )
