"""Data loading and splitting for the sentiment analysis platform.

Replaces the Kaggle credential dependency with HuggingFace `datasets`.
The repository can now be cloned and run without any external credentials.

Data flow:
    HuggingFace hub  →  load_imdb_data()  →  clean_text()  →  ready for any model
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset

from src.sentiment.logger import get_logger
from src.sentiment.preprocessing import clean_texts

logger = get_logger(__name__)


@dataclass
class SplitData:
    """Container for a single data split (train, val, or test).

    Attributes:
        texts: List of cleaned review strings.
        labels: List of integer labels (0 = negative, 1 = positive).
    """

    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.texts)

    def __repr__(self) -> str:
        pos = sum(self.labels)
        neg = len(self.labels) - pos
        return f"SplitData(n={len(self)}, pos={pos}, neg={neg})"


@dataclass
class IMDBData:
    """Container for the full IMDB dataset with train/val/test splits.

    Attributes:
        train: Training split.
        val: Validation split (carved from the original train split).
        test: Test split.
    """

    train: SplitData
    val: SplitData
    test: SplitData

    def __repr__(self) -> str:
        return (
            f"IMDBData(\n"
            f"  train={self.train},\n"
            f"  val={self.val},\n"
            f"  test={self.test}\n"
            f")"
        )


def load_imdb_data(
    val_ratio: float = 0.1,
    seed: int = 42,
    clean: bool = True,
) -> IMDBData:
    """Load the IMDB dataset from HuggingFace and prepare train/val/test splits.

    The HuggingFace IMDB dataset provides 25,000 train and 25,000 test samples.
    We carve a validation set from the training data.

    Args:
        val_ratio: Fraction of training data to use as validation (default 0.1 = 2,500 samples).
        seed: Random seed for reproducible validation split.
        clean: Whether to apply text cleaning. Set to False only for debugging.

    Returns:
        IMDBData with train, val, and test splits.

    Example:
        >>> data = load_imdb_data()
        >>> print(data)
        IMDBData(
          train=SplitData(n=22500, pos=11250, neg=11250),
          val=SplitData(n=2500, pos=1250, neg=1250),
          test=SplitData(n=25000, pos=12500, neg=12500)
        )
    """
    logger.info("Loading IMDB dataset from HuggingFace...")
    dataset = load_dataset("imdb")

    # Split train into train + validation
    train_val = dataset["train"].train_test_split(
        test_size=val_ratio,
        seed=seed,
        stratify_by_column="label",
    )

    train_texts: list[str] = train_val["train"]["text"]
    train_labels: list[int] = train_val["train"]["label"]

    val_texts: list[str] = train_val["test"]["text"]
    val_labels: list[int] = train_val["test"]["label"]

    test_texts: list[str] = dataset["test"]["text"]
    test_labels: list[int] = dataset["test"]["label"]

    # Apply centralized preprocessing
    if clean:
        logger.info("Cleaning text with centralized preprocessing pipeline...")
        train_texts = clean_texts(train_texts)
        val_texts = clean_texts(val_texts)
        test_texts = clean_texts(test_texts)

    data = IMDBData(
        train=SplitData(texts=train_texts, labels=train_labels),
        val=SplitData(texts=val_texts, labels=val_labels),
        test=SplitData(texts=test_texts, labels=test_labels),
    )

    logger.info(
        "Dataset loaded — Train: %d, Val: %d, Test: %d",
        len(data.train),
        len(data.val),
        len(data.test),
    )

    return data
