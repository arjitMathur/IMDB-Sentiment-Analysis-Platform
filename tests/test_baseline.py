"""Tests for the TF-IDF + Logistic Regression baseline model.

Covers:
    - Training and prediction
    - Save/load round-trip
    - Predict interface contract
    - Edge cases (empty input, untrained model)
    - is_available() checks
"""

import pytest

from src.sentiment.models.baseline import BaselineModel


# Small synthetic dataset for fast tests
TRAIN_TEXTS = [
    "this movie was absolutely wonderful and amazing",
    "great film with excellent performances",
    "i loved every minute of this movie",
    "best movie i have ever seen truly remarkable",
    "fantastic story and brilliant acting",
    "terrible movie waste of time",
    "awful film with bad acting",
    "i hated this movie so boring",
    "worst movie ever do not watch",
    "horrible plot and dreadful performances",
]
TRAIN_LABELS = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

VAL_TEXTS = [
    "amazing movie loved it",
    "terrible waste of time",
]
VAL_LABELS = [1, 0]


@pytest.fixture
def trained_model(tmp_path) -> BaselineModel:
    """Provide a trained baseline model using a temp directory."""
    model = BaselineModel(model_dir=tmp_path / "baseline")
    model.train(TRAIN_TEXTS, TRAIN_LABELS)
    return model


@pytest.fixture
def untrained_model(tmp_path) -> BaselineModel:
    """Provide an untrained baseline model."""
    return BaselineModel(model_dir=tmp_path / "baseline")


# ============================================================
# Training
# ============================================================


class TestTraining:
    """Tests for model training."""

    def test_train_returns_metrics(self, tmp_path):
        model = BaselineModel(model_dir=tmp_path / "baseline")
        metrics = model.train(TRAIN_TEXTS, TRAIN_LABELS)

        assert "train_accuracy" in metrics
        assert "train_f1" in metrics
        assert "training_time_seconds" in metrics
        assert metrics["model"] == "TF-IDF + Logistic Regression"

    def test_train_with_validation(self, tmp_path):
        model = BaselineModel(model_dir=tmp_path / "baseline")
        metrics = model.train(TRAIN_TEXTS, TRAIN_LABELS, VAL_TEXTS, VAL_LABELS)

        assert "val_accuracy" in metrics
        assert "val_f1" in metrics

    def test_train_accuracy_reasonable(self, tmp_path):
        model = BaselineModel(model_dir=tmp_path / "baseline")
        metrics = model.train(TRAIN_TEXTS, TRAIN_LABELS)

        # On a clean synthetic dataset, should get high accuracy
        assert metrics["train_accuracy"] >= 0.8


# ============================================================
# Prediction Interface
# ============================================================


class TestPrediction:
    """Tests for the predict() interface."""

    def test_predict_returns_required_keys(self, trained_model):
        result = trained_model.predict("This movie was amazing!")
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_predict_label_is_string(self, trained_model):
        result = trained_model.predict("Great film")
        assert result["label"] in ("positive", "negative")

    def test_predict_confidence_in_range(self, trained_model):
        result = trained_model.predict("Great film")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_probabilities_sum_to_one(self, trained_model):
        result = trained_model.predict("Great film")
        probs = result["probabilities"]
        assert abs(probs["positive"] + probs["negative"] - 1.0) < 0.01

    def test_predict_positive_review(self, trained_model):
        result = trained_model.predict("This movie was wonderful and amazing!")
        assert result["label"] == "positive"

    def test_predict_negative_review(self, trained_model):
        result = trained_model.predict("Terrible movie, absolutely awful!")
        assert result["label"] == "negative"

    def test_predict_empty_string(self, trained_model):
        result = trained_model.predict("")
        assert result["label"] in ("positive", "negative")
        assert result["confidence"] == 0.5

    def test_predict_includes_latency(self, trained_model):
        result = trained_model.predict("Good movie")
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0

    def test_predict_handles_html(self, trained_model):
        """Preprocessing should clean HTML before prediction."""
        result = trained_model.predict("<br /><br />Amazing movie!")
        assert result["label"] in ("positive", "negative")
        # Should not crash


# ============================================================
# Save / Load Round-Trip
# ============================================================


class TestPersistence:
    """Tests for save/load functionality."""

    def test_save_creates_files(self, trained_model, tmp_path):
        trained_model.save()
        model_dir = tmp_path / "baseline"
        assert (model_dir / "tfidf_vectorizer.joblib").exists()
        assert (model_dir / "logistic_regression.joblib").exists()

    def test_load_and_predict(self, trained_model, tmp_path):
        trained_model.save()

        # Load into a fresh instance
        loaded = BaselineModel(model_dir=tmp_path / "baseline")
        loaded.load()

        result = loaded.predict("Great movie!")
        assert result["label"] in ("positive", "negative")
        assert "confidence" in result

    def test_save_load_predictions_match(self, trained_model, tmp_path):
        original_result = trained_model.predict("Amazing film!")
        trained_model.save()

        loaded = BaselineModel(model_dir=tmp_path / "baseline")
        loaded.load()
        loaded_result = loaded.predict("Amazing film!")

        assert original_result["label"] == loaded_result["label"]
        assert abs(original_result["confidence"] - loaded_result["confidence"]) < 0.001

    def test_save_metrics(self, trained_model, tmp_path):
        metrics = {"test_accuracy": 0.88}
        trained_model.save_metrics(metrics)
        assert (tmp_path / "baseline" / "metrics.json").exists()

    def test_is_available_false_before_save(self, untrained_model):
        assert not untrained_model.is_available()

    def test_is_available_true_after_save(self, trained_model):
        trained_model.save()
        assert trained_model.is_available()


# ============================================================
# Error Handling
# ============================================================


class TestErrorHandling:
    """Tests for error cases."""

    def test_predict_before_train_raises(self, untrained_model):
        with pytest.raises(RuntimeError, match="not loaded"):
            untrained_model.predict("Some text")

    def test_load_missing_files_raises(self, untrained_model):
        with pytest.raises(FileNotFoundError, match="not found"):
            untrained_model.load()

    def test_evaluate_before_train_raises(self, untrained_model):
        with pytest.raises(RuntimeError, match="not loaded"):
            untrained_model.evaluate(["text"], [1])
