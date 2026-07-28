"""Centralized text preprocessing for the sentiment analysis platform.

This module is the SINGLE SOURCE OF TRUTH for all text cleaning.
Every model (baseline, LSTM, transformer) must use `clean_text()` before
tokenization or vectorization — both during training and inference.

Why this matters:
    The IMDB dataset contains raw HTML tags (<br />, <p>, etc.), URLs,
    and special characters. If training cleans text differently than
    inference, the model sees a different distribution at prediction time,
    degrading accuracy silently.
"""

from __future__ import annotations

import re


# Pre-compiled regex patterns for performance (compiled once, reused on every call)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")
_NON_ALPHA_RE = re.compile(r"[^a-z0-9\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean raw review text for model consumption.

    This function is intentionally kept simple and deterministic.
    It performs the same transformations regardless of which model
    will consume the output.

    Pipeline:
        1. Strip HTML tags        (<br />, <p>, <b>, etc.)
        2. Remove HTML entities   (&amp;, &lt;, &#39;, etc.)
        3. Remove URLs            (http://..., www....)
        4. Lowercase
        5. Remove non-alphanumeric characters (keep letters, digits, spaces)
        6. Collapse whitespace    (multiple spaces → single space)
        7. Strip leading/trailing whitespace

    Args:
        text: Raw review text, potentially containing HTML artifacts.

    Returns:
        Cleaned, lowercase text suitable for tokenization.

    Examples:
        >>> clean_text("Great movie!<br /><br />Loved it.")
        'great movie loved it'

        >>> clean_text("Visit http://example.com for more &amp; details!")
        'visit for more details'

        >>> clean_text("")
        ''

        >>> clean_text("   ")
        ''
    """
    if not text or not text.strip():
        return ""

    # 1. Remove HTML tags
    text = _HTML_TAG_RE.sub(" ", text)

    # 2. Remove HTML entities (&amp; → space, not &)
    text = _HTML_ENTITY_RE.sub(" ", text)

    # 3. Remove URLs
    text = _URL_RE.sub("", text)

    # 4. Lowercase
    text = text.lower()

    # 5. Remove non-alphanumeric (preserves spaces)
    text = _NON_ALPHA_RE.sub(" ", text)

    # 6. Collapse multiple spaces into one
    text = _MULTI_SPACE_RE.sub(" ", text)

    # 7. Strip
    text = text.strip()

    return text


def clean_texts(texts: list[str]) -> list[str]:
    """Batch-clean a list of review texts.

    Args:
        texts: List of raw review strings.

    Returns:
        List of cleaned strings, same length and order as input.
    """
    return [clean_text(t) for t in texts]
