"""Tests for {{name}}."""

from decimal import Decimal
from pathlib import Path

import pytest

from {{name}} import group_totals, main, render_report


def test_group_totals_sums_by_group() -> None:
    rows = [
        {"cat": "a", "n": "2.5"},
        {"cat": "b", "n": "1"},
        {"cat": "a", "n": "0.5"},
    ]
    assert group_totals(rows, "cat", "n") == {"a": Decimal("3.0"), "b": Decimal("1")}


def test_group_totals_rejects_a_non_number() -> None:
    with pytest.raises(SystemExit, match="must hold numbers"):
        group_totals([{"cat": "a", "n": "many"}], "cat", "n")


def test_render_report_has_a_table_and_a_total() -> None:
    text = render_report(Path("sample.csv"), {"a": Decimal("3")}, "n")
    assert "| a | 3.00 |" in text
    assert "| **Total** | **3.00** |" in text


def test_missing_column_names_what_was_found(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sample.write_text("cat,n\na,1\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="the file has: cat, n"):
        main([str(sample), "--group", "missing", "--sum", "n"])


def test_main_writes_the_report(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sample.write_text("cat,n\na,1\nb,2\n", encoding="utf-8")
    out = tmp_path / "report.md"

    assert main([str(sample), "--group", "cat", "--sum", "n", "--out", str(out)]) == 0
    assert "| a | 1.00 |" in out.read_text(encoding="utf-8")


def test_missing_file_exits_with_one(tmp_path: Path) -> None:
    assert main([str(tmp_path / "missing.csv"), "--group", "cat", "--sum", "n"]) == 1
