"""disk_usage: what fills a folder, what was written recently, and its failure modes."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from test_backend import Env  # noqa: F401  (the fixture other tool tests share)

from kiwibro import desktop, tools
from kiwibro.sandbox import Workspace
from kiwibro.tools import ToolContext


def _ctx(root: Path) -> ToolContext:
    return ToolContext(
        workspace=Workspace(root), memory=None, search=None, conversation_id="c_disk"  # type: ignore[arg-type]
    )


def _run(root: Path, **args):
    args.setdefault("path", str(root))
    return tools.execute_tool("disk_usage", tools.validate_args("disk_usage", args), _ctx(root))


def _allocated(path: Path) -> int:
    return os.stat(path).st_blocks * 512


def _lines(text: str) -> list[str]:
    return text.splitlines()


# --- what fills the folder ----------------------------------------------------


def test_largest_first_with_its_share_of_the_total(tmp_path: Path):
    (tmp_path / "small.txt").write_bytes(b"x" * 1000)
    (tmp_path / "big.bin").write_bytes(b"x" * 400_000)
    big_dir = tmp_path / "media"
    big_dir.mkdir()
    (big_dir / "clip.bin").write_bytes(b"x" * 200_000)
    (tmp_path / ".hidden").write_bytes(b"x" * 5000)

    result = _run(tmp_path)
    assert result.ok is True
    lines = _lines(result.content)
    total = _allocated(tmp_path / "small.txt") + _allocated(tmp_path / "big.bin") \
        + _allocated(big_dir / "clip.bin") + _allocated(tmp_path / ".hidden")
    assert desktop._human_bytes(total) in lines[0]
    assert "~" not in lines[0]  # a scratch folder is not the home folder
    assert lines[1].startswith("Disk: ")

    listed = [line.split("  ")[-1].strip() for line in lines[2:6]]
    assert listed == ["big.bin", "media/", ".hidden", "small.txt"]  # hidden entries count too
    share = int([line for line in lines[2:6] if line.endswith("big.bin")][0].split()[-2].rstrip("%"))
    assert share == round(100 * _allocated(tmp_path / "big.bin") / total)


def test_top_clamps_and_says_what_it_left_out(tmp_path: Path):
    for index in range(6):
        (tmp_path / f"f{index}.bin").write_bytes(b"x" * (1000 * (index + 1)))
    every = _lines(_run(tmp_path, top=50).content)
    assert sum(1 for line in every if line.endswith(".bin")) == 6
    assert not any("more not listed" in line for line in every)

    five = _lines(_run(tmp_path, top=2).content)
    assert sum(1 for line in five if line.endswith(".bin")) == 2
    assert any("4 more not listed" in line for line in five)


def test_top_out_of_range_is_refused(tmp_path: Path):
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(ValueError, match="top must be between 1 and 50"):
        tools.validate_args("disk_usage", {"path": str(tmp_path), "top": 0})


# --- what was written recently ------------------------------------------------


def test_changed_today_week_and_month(tmp_path: Path):
    (tmp_path / "now.txt").write_text("today")
    (tmp_path / "soon.txt").write_text("this week")
    (tmp_path / "month.txt").write_text("this month, not this week")
    (tmp_path / "old.txt").write_text("long ago")
    now = time.time()
    os.utime(tmp_path / "soon.txt", (now - 3 * 86400, now - 3 * 86400))
    os.utime(tmp_path / "month.txt", (now - 20 * 86400, now - 20 * 86400))
    os.utime(tmp_path / "old.txt", (now - 40 * 86400, now - 40 * 86400))

    today = _run(tmp_path, changed="today")
    assert today.ok is True
    assert "now.txt" in today.content
    assert "soon.txt" not in today.content and "month.txt" not in today.content
    assert "Largest files written since" in today.content

    week = _run(tmp_path, changed="week").content
    assert "now.txt" in week and "soon.txt" in week
    assert "month.txt" not in week and "old.txt" not in week

    month = _run(tmp_path, changed="month").content
    assert "month.txt" in month and "old.txt" not in month


def test_changed_counts_only_new_bytes_and_lists_the_largest_files(tmp_path: Path):
    old_dir = tmp_path / "archive"
    old_dir.mkdir()
    (old_dir / "old.txt").write_bytes(b"x" * 10_000)
    os.utime(old_dir / "old.txt", (time.time() - 40 * 86400,) * 2)
    (tmp_path / "fresh.txt").write_bytes(b"x" * 30_000)
    (tmp_path / "quiet").mkdir()
    (tmp_path / "quiet" / "nothing_new.txt").write_text("x")
    os.utime(tmp_path / "quiet" / "nothing_new.txt", (time.time() - 40 * 86400,) * 2)

    content = _run(tmp_path, changed="today").content
    assert "fresh.txt" in content
    assert "quiet/" not in content  # nothing written there since today
    assert "old.txt" not in content
    assert "fresh.txt" in content.split("Largest files written since", 1)[1]
    assert desktop._human_bytes(_allocated(tmp_path / "fresh.txt")) in content


def test_a_changed_value_outside_the_enum_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="changed must be today, week or month"):
        tools.validate_args("disk_usage", {"path": str(tmp_path), "changed": "yesterday"})


# --- edges --------------------------------------------------------------------


def test_empty_folder_still_reports_the_disk(tmp_path: Path):
    result = _run(tmp_path)
    assert result.ok is True
    assert "0 entries" in result.content.splitlines()[0]
    assert "Disk: " in result.content


def test_an_unreadable_folder_is_skipped_and_counted(tmp_path: Path):
    if os.geteuid() == 0:
        pytest.skip("root reads every folder")
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "secret.txt").write_text("x")
    (tmp_path / "open.txt").write_text("x")
    locked.chmod(0)
    try:
        content = _run(tmp_path).content
    finally:
        locked.chmod(0o755)
    assert "1 folder could not be read" in content
    assert "open.txt" in content


def test_a_file_as_the_path_is_a_plain_error(tmp_path: Path):
    target = tmp_path / "notes.txt"
    target.write_text("x")
    result = _run(tmp_path, path=str(target))
    assert result.ok is False
    assert result.content.startswith("Error: ") and "is a file, not a folder" in result.content


def test_a_missing_path_is_a_plain_error(tmp_path: Path):
    result = _run(tmp_path, path=str(tmp_path / "nowhere"))
    assert result.ok is False
    assert result.content.startswith("Error: ") and "does not exist" in result.content


def test_the_time_budget_cuts_the_walk_short(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    (tmp_path / "big.bin").write_bytes(b"x" * 200_000)
    (tmp_path / "deep").mkdir()
    (tmp_path / "deep" / "inside.bin").write_bytes(b"x" * 300_000)
    monkeypatch.setattr(desktop, "DISK_WALK_SECONDS", 0)
    content = _run(tmp_path).content
    assert "The walk was cut short after 0 seconds" in content
    assert "not measured at all: deep/" in content  # the folder is named, not silently left out
    assert "big.bin" in content  # a file in the folder itself costs no walk


def test_symlinks_are_not_followed(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "payload.bin").write_bytes(b"x" * 500_000)
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    content = _run(tmp_path).content
    assert not any(line.endswith("link") or line.endswith("link/") for line in _lines(content))
    real_line = [line for line in _lines(content) if line.endswith("real/")][0]
    assert "100%" in real_line  # the payload is counted once, under its real folder


# --- through the tool layer ---------------------------------------------------


def test_the_tool_runs_through_the_dispatch_entry_point(tmp_path: Path):
    (tmp_path / "work.bin").write_bytes(b"x" * 100_000)
    assert "disk_usage" in tools.tool_names()
    result = _run(tmp_path)
    assert result.ok is True and "work.bin" in result.content
    assert isinstance(result.preview, str) and result.preview


def test_the_home_folder_is_named_with_a_tilde(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "a.txt").write_text("x")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    content = _run(tmp_path, path=str(home)).content
    assert content.splitlines()[0].startswith("~: ")
    inner = home / "projects"
    inner.mkdir()
    (inner / "b.txt").write_text("x")
    assert _run(tmp_path, path=str(inner)).content.splitlines()[0].startswith("~/projects: ")
