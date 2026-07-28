"""Tests for the data loading and splitting module.

Tests are split into two categories:
    - Unit tests: Fast, mock the HuggingFace download, always run.
    - Integration tests: Actually download the dataset, marked @pytest.mark.slow.

Run fast tests:   pytest tests/test_data.py
Run all tests:    pytest tests/test_data.py -m ""
Skip slow tests:  pytest tests/test_data.py -m "not slow"
"""

from unittest.mock import MagicMock, patch

import pytest

from src.sentiment.data import IMDBData, SplitData, load_imdb_data


# ============================================================
# SplitData Unit Tests
# ============================================================


class TestSplitData:
    """Tests for the SplitData container."""

    def test_len(self):
        split = SplitData(texts=["a", "b", "c"], labels=[1, 0, 1])
        assert len(split) == 3

    def test_repr_shows_counts(self):
        split = SplitData(texts=["a", "b", "c"], labels=[1, 0, 1])
        r = repr(split)
        assert "n=3" in r
        assert "pos=2" in r
        assert "neg=1" in r

    def test_empty_split(self):
        split = SplitData(texts=[], labels=[])
        assert len(split) == 0


# ============================================================
# IMDBData Unit Tests
# ============================================================


class TestIMDBData:
    """Tests for the IMDBData container."""

    def test_repr_contains_all_splits(self):
        data = IMDBData(
            train=SplitData(texts=["a"], labels=[1]),
            val=SplitData(texts=["b"], labels=[0]),
            test=SplitData(texts=["c"], labels=[1]),
        )
        r = repr(data)
        assert "train=" in r
        assert "val=" in r
        assert "test=" in r


# ============================================================
# load_imdb_data Unit Tests (mocked download)
# ============================================================


def _make_mock_dataset():
    """Create a mock that mimics HuggingFace dataset structure."""
    # 20 samples for train, 10 for test
    train_texts = [f"review {i}" for i in range(20)]
    train_labels = [i % 2 for i in range(20)]  # alternating 0, 1

    test_texts = [f"test review {i}" for i in range(10)]
    test_labels = [i % 2 for i in range(10)]

    # Mock the train_test_split method
    mock_train_split = MagicMock()
    mock_train_split.__getitem__ = MagicMock(side_effect=lambda key: {
        "train": {"text": train_texts[:18], "label": train_labels[:18]},
        "test": {"text": train_texts[18:], "label": train_labels[18:]},
    }[key])

    mock_train = MagicMock()
    mock_train.train_test_split.return_value = mock_train_split
    mock_train.__getitem__ = MagicMock(side_effect=lambda key: {
        "text": train_texts,
        "label": train_labels,
    }[key])

    mock_test = MagicMock()
    mock_test.__getitem__ = MagicMock(side_effect=lambda key: {
        "text": test_texts,
        "label": test_labels,
    }[key])

    mock_dataset = MagicMock()
    mock_dataset.__getitem__ = MagicMock(side_effect=lambda key: {
        "train": mock_train,
        "test": mock_test,
    }[key])

    return mock_dataset


class TestLoadIMDBDataMocked:
    """Tests using a mocked HuggingFace dataset (no download)."""

    @patch("src.sentiment.data.load_dataset")
    def test_returns_imdb_data(self, mock_load):
        mock_load.return_value = _make_mock_dataset()
        data = load_imdb_data()
        assert isinstance(data, IMDBData)

    @patch("src.sentiment.data.load_dataset")
    def test_splits_are_non_empty(self, mock_load):
        mock_load.return_value = _make_mock_dataset()
        data = load_imdb_data()
        assert len(data.train) > 0
        assert len(data.val) > 0
        assert len(data.test) > 0

    @patch("src.sentiment.data.load_dataset")
    def test_texts_and_labels_same_length(self, mock_load):
        mock_load.return_value = _make_mock_dataset()
        data = load_imdb_data()
        assert len(data.train.texts) == len(data.train.labels)
        assert len(data.val.texts) == len(data.val.labels)
        assert len(data.test.texts) == len(data.test.labels)

    @patch("src.sentiment.data.load_dataset")
    def test_labels_are_binary(self, mock_load):
        mock_load.return_value = _make_mock_dataset()
        data = load_imdb_data()
        all_labels = data.train.labels + data.val.labels + data.test.labels
        assert all(label in (0, 1) for label in all_labels)

    @patch("src.sentiment.data.load_dataset")
    def test_clean_false_skips_preprocessing(self, mock_load):
        mock_load.return_value = _make_mock_dataset()
        data = load_imdb_data(clean=False)
        # Raw text should be unchanged (contains original casing)
        assert any(t.startswith("review") or t.startswith("test") for t in data.train.texts + data.test.texts)

    @patch("src.sentiment.data.load_dataset")
    def test_clean_true_lowercases(self, mock_load):
        mock_load.return_value = _make_mock_dataset()
        data = load_imdb_data(clean=True)
        for text in data.train.texts:
            assert text == text.lower()


# ============================================================
# Integration Tests (actual download)
# ============================================================


@pytest.mark.slow
class TestLoadIMDBDataIntegration:
    """Integration tests that actually download the IMDB dataset.

    Run with: pytest tests/test_data.py -m slow
    """

    def test_full_download_and_split(self):
        data = load_imdb_data(val_ratio=0.1, seed=42)

        # HuggingFace IMDB: 25000 train, 25000 test
        # After val split: 22500 train, 2500 val, 25000 test
        assert len(data.train) == 22500
        assert len(data.val) == 2500
        assert len(data.test) == 25000

    def test_labels_are_balanced(self):
        data = load_imdb_data()
        # IMDB is perfectly balanced
        train_pos = sum(data.train.labels)
        test_pos = sum(data.test.labels)
        assert test_pos == 12500
        assert abs(train_pos - 11250) <= 10  # allow small variance from stratified split

    def test_no_html_in_cleaned_text(self):
        data = load_imdb_data(clean=True)
        # Spot-check first 100 samples for HTML artifacts
        for text in data.train.texts[:100]:
            assert "<br" not in text
            assert "</" not in text
            assert "&amp;" not in text
