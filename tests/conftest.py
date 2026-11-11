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


@pytest.fixture(autouse=True)
def no_search_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test sends a query to a real search engine.

    Kiwibro may run a search the user asked for by itself, so a test message can lead to
    one. Tests that exercise a provider replace these two functions with their own."""
    import requests

    from kiwibro import search

    def offline(*_args, **_kwargs):
        raise requests.ConnectionError("search providers are switched off in tests")

    monkeypatch.setattr(search, "_get", offline)
    monkeypatch.setattr(search, "_post", offline)
    # ...and none opens the pages of a result: the test that covers page reading puts this back.
    from kiwibro import research

    monkeypatch.setattr(research, "read_results", lambda engine, query, results, **_kwargs: results)
    monkeypatch.setattr(research, "THIN_RESULTS", 0)  # nor searches a second time for a thin first answer
