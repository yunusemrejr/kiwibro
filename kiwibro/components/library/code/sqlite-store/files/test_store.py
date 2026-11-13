"""Tests for the {{table}} store."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from store import Store


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    opened = Store(tmp_path / "{{table}}.db")
    yield opened
    opened.close()


def test_add_returns_the_stored_record(store: Store) -> None:
    record = store.add("  first note  ", "body text")

    assert record.id == 1
    assert record.title == "first note"
    assert store.get(record.id) == record


def test_missing_record_is_none(store: Store) -> None:
    assert store.get(999) is None


def test_list_returns_unarchived_records_oldest_first(store: Store) -> None:
    first = store.add("first")
    second = store.add("second")

    assert [record.id for record in store.list()] == [first.id, second.id]


def test_archive_hides_the_record_but_keeps_it(store: Store) -> None:
    record = store.add("temporary")

    assert store.archive(record.id) is True
    assert store.list() == []
    assert store.list(include_archived=True) == [store.get(record.id)]


def test_archiving_twice_reports_no_change(store: Store) -> None:
    record = store.add("temporary")

    store.archive(record.id)

    assert store.archive(record.id) is False
    assert store.archive(999) is False


def test_search_matches_title_and_body(store: Store) -> None:
    store.add("shopping list", "milk and bread")
    store.add("ideas", "write more tests")

    assert [record.title for record in store.search("bread")] == ["shopping list"]
    assert [record.title for record in store.search("list")] == ["shopping list"]


def test_search_skips_archived_records(store: Store) -> None:
    record = store.add("old plan", "gone")

    store.archive(record.id)

    assert store.search("plan") == []


def test_empty_title_is_rejected(store: Store) -> None:
    with pytest.raises(ValueError, match="title"):
        store.add("   ")


def test_empty_search_is_rejected(store: Store) -> None:
    with pytest.raises(ValueError, match="search text"):
        store.search(" ")


def test_reopening_keeps_the_data_and_the_schema(tmp_path: Path) -> None:
    path = tmp_path / "{{table}}.db"
    with Store(path) as first:
        first.add("kept")
    with Store(path) as second:
        assert [record.title for record in second.list()] == ["kept"]
