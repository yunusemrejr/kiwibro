"""Tests for {{name}}."""

import json
from pathlib import Path

import pytest

from {{name}} import main, reshape, to_csv_text

RECORDS = [
    {"name": "alpha", "kind": "tool", "stars": 12},
    {"name": "beta", "kind": "game", "stars": 7},
    {"name": "gamma", "kind": "tool", "stars": 21},
]


def test_reshape_picks_renames_and_filters() -> None:
    out = reshape(RECORDS, ["name", "stars"], {"stars": "votes"}, {"kind": "tool"})
    assert out == [
        {"name": "alpha", "votes": 12},
        {"name": "gamma", "votes": 21},
    ]


def test_reshape_without_options_keeps_everything() -> None:
    assert reshape(RECORDS, None, {}, {}) == RECORDS


def test_to_csv_text_writes_a_header_and_rows() -> None:
    assert to_csv_text([{"name": "alpha", "stars": 12}]) == "name,stars\r\nalpha,12\r\n"
    assert to_csv_text([]) == ""


def test_main_writes_csv(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps(RECORDS), encoding="utf-8")
    out = tmp_path / "out.csv"

    assert main([str(sample), "--pick", "name,kind", "--where", "kind=tool",
                 "--format", "csv", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == "name,kind\nalpha,tool\ngamma,tool\n"


def test_main_writes_json_to_the_terminal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps(RECORDS), encoding="utf-8")

    assert main([str(sample), "--rename", "stars=votes"]) == 0
    assert json.loads(capsys.readouterr().out) == [
        {"name": "alpha", "kind": "tool", "votes": 12},
        {"name": "beta", "kind": "game", "votes": 7},
        {"name": "gamma", "kind": "tool", "votes": 21},
    ]


def test_unknown_field_names_what_was_found(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps(RECORDS), encoding="utf-8")

    with pytest.raises(SystemExit, match="the file has: kind, name, stars"):
        main([str(sample), "--pick", "missing"])


def test_non_list_input_exits_with_one(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    sample.write_text('{"name": "alpha"}', encoding="utf-8")

    assert main([str(sample)]) == 1
