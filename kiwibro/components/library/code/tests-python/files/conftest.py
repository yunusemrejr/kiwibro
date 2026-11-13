"""Shared fixtures for the {{module}} tests."""

import pytest


# KIWIBRO: keep a fixture here only when several test files need it
@pytest.fixture
def numbers() -> dict[str, int]:
    """Values reused by several tests."""
    return {"a": 2, "b": 3, "sum": 5}


@pytest.fixture
def sample_text() -> str:
    """Three lines of text, with a final newline."""
    return "alpha\nbeta\ngamma\n"
