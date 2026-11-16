"""The code/tui-app kit: its files, its logic tests run from a workspace, and the
phrases in a user's message that call for it."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kiwibro import components  # noqa: E402
from kiwibro.sandbox import Workspace  # noqa: E402

LIBRARY = REPO / "kiwibro" / "components" / "library"
KIT = "code/tui-app"


def test_the_kit_is_in_the_library_with_its_blanks():
    kit = components.Library(LIBRARY).get(KIT)
    assert kit.kind == components.KIT
    assert list(kit.blanks) == ["title", "accent", "items"]
    assert kit.file_paths() == ["app.py", "items.json", "README.md", "screen.py", "test_app.py"]


def test_suggest_picks_the_kit_for_a_terminal_app_and_not_for_an_ordinary_terminal():
    for message in ("build a TUI for my todo list", "make a terminal app to browse my notes"):
        found = components.suggest(message)
        assert found is not None and found.name == KIT, message
    for message in ("which terminal command shows the disk usage", "open a terminal"):
        found = components.suggest(message)
        assert found is None or found.name != KIT, message


def test_the_written_app_passes_its_own_logic_tests(tmp_path: Path):
    workspace = Workspace(tmp_path / "ws")
    components.use(KIT, {"title": "Notes", "items": "notes"}, workspace=workspace, path="app")
    folder = workspace.root / "app"
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_app.py"],
        cwd=folder, capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert " passed" in done.stdout and "failed" not in done.stdout
