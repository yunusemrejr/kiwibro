"""Tests for {{module}}."""

from pathlib import Path

import pytest

import {{module}}


# KIWIBRO: parametrised cases: add the inputs your function must handle
@pytest.mark.parametrize(
    "a, b, expected",
    [
        (0, 0, 0),
        (2, 3, 5),
        (-4, 1, -3),
        (2**31, 1, 2**31 + 1),
    ],
)
def test_add_returns_the_sum(a: int, b: int, expected: int) -> None:
    assert {{module}}.add(a, b) == expected


def test_fixture_values_add_up(numbers: dict[str, int]) -> None:
    assert {{module}}.add(numbers["a"], numbers["b"]) == numbers["sum"]


@pytest.mark.parametrize("text, expected", [("7", 7), (" -12 ", -12), ("0", 0)])
def test_parse_int_reads_whole_numbers(text: str, expected: int) -> None:
    assert {{module}}.parse_int(text) == expected


# KIWIBRO: error cases: check the message too, so the user can act on it
@pytest.mark.parametrize("text", ["", "   ", "seven", "1.5"])
def test_parse_int_rejects_bad_input(text: str) -> None:
    with pytest.raises(ValueError, match="not an integer"):
        {{module}}.parse_int(text)


# KIWIBRO: tmp_path gives each test its own empty folder
def test_count_lines_reads_a_file(tmp_path: Path, sample_text: str) -> None:
    path = tmp_path / "sample.txt"
    path.write_text(sample_text, encoding="utf-8")

    assert {{module}}.count_lines(path) == 3


def test_count_lines_handles_a_file_without_final_newline(tmp_path: Path) -> None:
    path = tmp_path / "last.txt"
    path.write_text("one\ntwo", encoding="utf-8")

    assert {{module}}.count_lines(path) == 2


def test_count_lines_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        {{module}}.count_lines(tmp_path / "missing.txt")
