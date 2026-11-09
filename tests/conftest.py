"""Shared pytest fixtures for every test module.

`env` (defined in test_backend) is a Flask app wired to a fake model client and a
temporary data dir and workspace; re-exporting it here makes it available to
all test files without each one importing it.
"""

from __future__ import annotations

import pytest

from test_backend import env  # noqa: F401  (re-exported fixture)


@pytest.fixture(autouse=True)
def hermetic_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test scans this machine's model folders or finds a real llama.cpp build.

    Tests that need another app's folders (see test_engine) delete these switches again."""
    monkeypatch.setenv("KIWIBRO_OWN_MODELS_ONLY", "1")
    monkeypatch.setenv("KIWIBRO_OWN_RUNTIME_ONLY", "1")
