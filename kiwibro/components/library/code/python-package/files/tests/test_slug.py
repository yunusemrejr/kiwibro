"""Tests for {{name}}.slugify."""

import pytest

from {{name}} import slugify


# KIWIBRO: replace these cases with tests for your own functions
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, World!", "hello-world"),
        ("  two   spaces  ", "two-spaces"),
        ("Crème Brûlée", "creme-brulee"),
        ("", ""),
        ("---", ""),
        ("Already-slugged", "already-slugged"),
    ],
)
def test_slugify_normalises_text(text: str, expected: str) -> None:
    assert slugify(text) == expected


def test_slugify_cuts_on_a_word_boundary() -> None:
    assert slugify("one two three four", max_length=8) == "one-two"


def test_slugify_keeps_short_slugs_whole() -> None:
    assert slugify("fits", max_length=10) == "fits"


def test_slugify_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="text must be str"):
        slugify(None)  # type: ignore[arg-type]


def test_slugify_rejects_bad_length() -> None:
    with pytest.raises(ValueError, match="max_length must be positive"):
        slugify("text", max_length=0)
