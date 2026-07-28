"""Shared pytest fixtures for the sentiment analysis test suite."""

import pytest

from src.sentiment.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Provide default settings for tests."""
    return Settings()


@pytest.fixture
def sample_reviews() -> list[dict]:
    """Provide sample reviews for testing predictions."""
    return [
        {"text": "This movie was amazing, I loved every moment!", "expected": "positive"},
        {"text": "Terrible waste of time, awful acting.", "expected": "negative"},
        {"text": "It was okay, not great but not bad either.", "expected": None},  # ambiguous
        {"text": "", "expected": None},  # edge case: empty
        {"text": "<br /><br />Great film with <b>bold</b> performances!", "expected": "positive"},  # HTML
    ]


@pytest.fixture
def sample_html_text() -> str:
    """Provide a sample IMDB review with HTML artifacts."""
    return (
        "This movie was terrible.<br /><br />The acting was awful "
        "and the plot made no sense. I can't believe I wasted "
        "2 hours on this. Visit http://example.com for better movies. "
        "Rating: 1/10 &amp; would NOT recommend!!!"
    )
