"""Tests for {{name}}.cli."""

from pathlib import Path

import pytest

from {{name}}.cli import count_text, longest_line, main


def test_count_text_counts_lines_words_and_characters() -> None:
    assert count_text("one two\nthree\n") == (2, 3, 14)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", ""),
        ("short\nlonger line\n", "longer line"),
        ("no newline", "no newline"),
    ],
)
def test_longest_line(text: str, expected: str) -> None:
    assert longest_line(text) == expected


# KIWIBRO: replace these cases with tests for your own subcommands
def test_stats_prints_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world\n", encoding="utf-8")

    assert main(["stats", str(sample)]) == 0
    assert capsys.readouterr().out.split()[:3] == ["1", "2", "12"]


def test_stats_adds_a_total_for_several_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("a\n", encoding="utf-8")
    second.write_text("b\n", encoding="utf-8")

    assert main(["stats", str(first), str(second)]) == 0
    last_line = capsys.readouterr().out.splitlines()[-1]
    assert last_line.split()[:3] == ["2", "2", "4"]
    assert last_line.endswith("total")


def test_longest_prints_the_longest_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("short\na much longer line\n", encoding="utf-8")

    assert main(["longest", str(sample)]) == 0
    assert capsys.readouterr().out.endswith("a much longer line\n")


def test_missing_file_exits_with_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.txt"

    assert main(["stats", str(missing)]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_verbose_logs_to_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("text\n", encoding="utf-8")

    assert main(["-v", "longest", str(sample)]) == 0
    assert "read" in capsys.readouterr().err
