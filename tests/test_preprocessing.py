"""Tests for the centralized preprocessing module.

Covers:
    - HTML tag removal
    - HTML entity removal
    - URL removal
    - Lowercasing
    - Special character removal
    - Whitespace normalization
    - Edge cases (empty, None-like, unicode, extremely long)
    - Batch processing
    - Idempotency (cleaning already-clean text returns same result)
"""

import pytest

from src.sentiment.preprocessing import clean_text, clean_texts


# ============================================================
# HTML Cleaning
# ============================================================

class TestHTMLRemoval:
    """Tests for HTML tag and entity removal."""

    def test_removes_br_tags(self):
        assert clean_text("Great movie.<br />Loved it.") == "great movie loved it"

    def test_removes_multiple_br_tags(self):
        assert clean_text("Hello<br /><br /><br />World") == "hello world"

    def test_removes_paragraph_tags(self):
        assert clean_text("<p>Good film</p>") == "good film"

    def test_removes_bold_italic_tags(self):
        assert clean_text("<b>Bold</b> and <i>italic</i>") == "bold and italic"

    def test_removes_anchor_tags_with_attributes(self):
        result = clean_text('<a href="http://example.com">click here</a>')
        assert result == "click here"

    def test_removes_html_entities_amp(self):
        assert clean_text("rock &amp; roll") == "rock roll"

    def test_removes_html_entities_lt_gt(self):
        assert clean_text("5 &lt; 10 &gt; 3") == "5 10 3"

    def test_removes_numeric_html_entities(self):
        assert clean_text("quote &#39; here") == "quote here"


# ============================================================
# URL Removal
# ============================================================

class TestURLRemoval:
    """Tests for URL stripping."""

    def test_removes_http_url(self):
        result = clean_text("Visit http://example.com for more")
        assert "example" not in result
        assert "visit" in result
        assert "for more" in result

    def test_removes_https_url(self):
        result = clean_text("See https://www.example.com/path?q=1")
        assert "example" not in result

    def test_removes_www_url(self):
        result = clean_text("Go to www.example.com now")
        assert "example" not in result
        assert "go to" in result


# ============================================================
# Text Normalization
# ============================================================

class TestNormalization:
    """Tests for lowercasing, special chars, and whitespace."""

    def test_lowercases_text(self):
        assert clean_text("GREAT MOVIE") == "great movie"

    def test_mixed_case(self):
        assert clean_text("ThIs Is MiXeD") == "this is mixed"

    def test_removes_punctuation(self):
        assert clean_text("Wow! Amazing... Really?") == "wow amazing really"

    def test_preserves_numbers(self):
        assert clean_text("I give it 10 out of 10") == "i give it 10 out of 10"

    def test_collapses_multiple_spaces(self):
        assert clean_text("too   many    spaces") == "too many spaces"

    def test_strips_leading_trailing_whitespace(self):
        assert clean_text("   hello world   ") == "hello world"

    def test_removes_special_characters(self):
        assert clean_text("movie!! @#$% great^^^") == "movie great"


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    """Tests for boundary conditions and unusual inputs."""

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_whitespace_only(self):
        assert clean_text("   ") == ""

    def test_tabs_and_newlines(self):
        assert clean_text("\t\n\r") == ""

    def test_single_word(self):
        assert clean_text("amazing") == "amazing"

    def test_only_html_tags(self):
        assert clean_text("<br /><br /><p></p>") == ""

    def test_only_special_chars(self):
        assert clean_text("!@#$%^&*()") == ""

    def test_very_long_text(self):
        """Ensure no performance issues with long input."""
        long_text = "word " * 10000
        result = clean_text(long_text)
        assert len(result) > 0
        assert result == ("word " * 10000).strip()

    def test_unicode_characters_removed(self):
        """Non-ASCII characters should be stripped."""
        result = clean_text("café résumé naïve")
        # accented chars are non-alphanumeric in [a-z0-9]
        assert "caf" in result


# ============================================================
# Idempotency
# ============================================================

class TestIdempotency:
    """Cleaning already-clean text should produce the same result."""

    def test_double_clean_is_same(self):
        raw = "Great movie!<br /><br />Loved the <b>acting</b>."
        first = clean_text(raw)
        second = clean_text(first)
        assert first == second

    def test_clean_text_is_stable(self):
        clean = "this is already clean text"
        assert clean_text(clean) == clean


# ============================================================
# Batch Processing
# ============================================================

class TestBatchProcessing:
    """Tests for the clean_texts() batch function."""

    def test_batch_returns_same_length(self):
        inputs = ["Hello", "World", ""]
        results = clean_texts(inputs)
        assert len(results) == 3

    def test_batch_cleans_each_element(self):
        inputs = ["<br />Hello!", "WORLD", ""]
        results = clean_texts(inputs)
        assert results == ["hello", "world", ""]

    def test_batch_empty_list(self):
        assert clean_texts([]) == []

    def test_batch_preserves_order(self):
        inputs = ["third", "first", "second"]
        results = clean_texts(inputs)
        assert results == ["third", "first", "second"]


# ============================================================
# Real-World IMDB Examples
# ============================================================

class TestRealIMDBSamples:
    """Tests using actual patterns found in the IMDB dataset."""

    def test_typical_imdb_review(self):
        raw = (
            "This movie was terrible.<br /><br />The acting was awful "
            "and the plot made no sense. Visit http://example.com for "
            "better movies. Rating: 1/10 &amp; would NOT recommend!!!"
        )
        result = clean_text(raw)

        # Should NOT contain HTML, URLs, or special chars
        assert "<br" not in result
        assert "http" not in result
        assert "&amp;" not in result
        assert "!!!" not in result

        # Should contain the actual words, lowercased
        assert "terrible" in result
        assert "acting" in result
        assert "not recommend" in result

    def test_review_with_nested_tags(self):
        raw = '<div class="review"><p>Excellent <em>performance</em>!</p></div>'
        result = clean_text(raw)
        assert "excellent" in result
        assert "performance" in result
        assert "<" not in result
