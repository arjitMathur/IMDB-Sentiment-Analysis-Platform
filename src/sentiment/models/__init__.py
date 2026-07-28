"""Model implementations for the sentiment analysis platform.

All models expose a common interface via the SentimentModel protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SentimentModel(Protocol):
    """Common interface that all sentiment models must implement.

    This protocol ensures the Streamlit app / FastAPI backend never needs
    to know which concrete model is being used.
    """

    @property
    def name(self) -> str:
        """Human-readable model name (e.g. 'TF-IDF + Logistic Regression')."""
        ...

    def load(self) -> None:
        """Load model artifacts from disk."""
        ...

    def predict(self, text: str) -> dict:
        """Run inference on a single text string.

        Args:
            text: Raw review text (cleaning is handled internally).

        Returns:
            dict with keys:
                - label: str ("positive" or "negative")
                - confidence: float (0.0 - 1.0, confidence in predicted label)
                - probabilities: dict {"positive": float, "negative": float}
        """
        ...
