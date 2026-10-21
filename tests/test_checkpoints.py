"""Regression tests for kiwibro.checkpoints: automatic file snapshots, undo, pruning and the
undo_changes tool."""

from __future__ import annotations

import shutil
import types

import pytest

from kiwibro import checkpoints
from kiwibro.sandbox import Workspace


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return Workspace(root), tmp_path / "data"


def _objects(data):
    root = data / "checkpoints" / "objects"
    return [path for path in root.rglob("*") if path.is_file()] if root.exists() else []


def _ctx(workspace, data, conversation_id="c1"):
    return types.SimpleNamespace(
        workspace=workspace,
        memory=types.SimpleNamespace(db=types.SimpleNamespace(path=str(data / "kiwibro.db"))),
        conversation_id=conversation_id,
    )


# --- snapshot and undo -----------------------------------------------------------------------


def test_overwrite_is_undone_to_the_snapshot_content(env):
    ws, data = env
    target = ws.root / "a.txt"
    target.write_text("old")

    batch = checkpoints.snapshot_call(data, ws, "write_file", {"path": "a.txt", "content": "new"}, "c1")
    assert batch

    target.write_text("new")
    reports = checkpoints.store_for(data).undo("c1")

    assert target.read_text() == "old"
    assert any("restored" in report["action"] for report in reports)


def test_creating_a_new_file_is_undone_by_removing_it(env):
    ws, data = env
    checkpoints.snapshot_call(data, ws, "write_file", {"path": "new.txt", "content": "x"}, "c1")
    (ws.root / "new.txt").write_text("x")

    reports = checkpoints.store_for(data).undo("c1")

    assert not (ws.root / "new.txt").exists()
    assert any("removed" in report["action"] for report in reports)


def test_undo_records_the_pre_undo_state_in_history(env):
    ws, data = env
    (ws.root / "a.txt").write_text("old")
    checkpoints.snapshot_call(data, ws, "write_file", {"path": "a.txt", "content": "new"}, "c1")
    (ws.root / "a.txt").write_text("new")

    store = checkpoints.store_for(data)
    store.undo("c1")

    tools_in_history = {entry["tool"] for entry in store.history("c1")}
    assert "undo_changes" in tools_in_history


def test_undo_steps_two_restores_original_content_across_two_changes(env):
    ws, data = env
    target = ws.root / "a.txt"
    target.write_text("orig")
    checkpoints.snapshot_call(data, ws, "write_file", {"path": "a.txt", "content": "v1"}, "c1")
    target.write_text("v1")
    checkpoints.snapshot_call(data, ws, "write_file", {"path": "a.txt", "content": "v2"}, "c1")
    target.write_text("v2")

    checkpoints.store_for(data).undo("c1", steps=2)

    assert target.read_text() == "orig"


def test_undo_with_display_path_restores_only_that_file(env):
    ws, data = env
    (ws.root / "a.txt").write_text("a0")
    (ws.root / "b.txt").write_text("b0")

    checkpoints.snapshot_call(data, ws, "write_file", {"path": "a.txt", "content": "a1"}, "c1")
    (ws.root / "a.txt").write_text("a1")
    checkpoints.snapshot_call(data, ws, "write_file", {"path": "b.txt", "content": "b1"}, "c1")
    (ws.root / "b.txt").write_text("b1")

    checkpoints.store_for(data).undo("c1", display="a.txt")

    assert (ws.root / "a.txt").read_text() == "a0"
    assert (ws.root / "b.txt").read_text() == "b1"


def test_deleted_folder_is_recreated_with_all_files(env):
    ws, data = env
    folder = ws.root / "folder"
    (folder / "sub").mkdir(parents=True)
    (folder / "one.txt").write_text("1")
    (folder / "two.txt").write_text("2")
    (folder / "sub" / "three.txt").write_text("3")

    checkpoints.snapshot_call(data, ws, "file_manage", {"action": "delete", "path": "folder"}, "c1")
    shutil.rmtree(folder)
    assert not folder.exists()

    checkpoints.store_for(data).undo("c1")

    assert (folder / "one.txt").read_text() == "1"
    assert (folder / "two.txt").read_text() == "2"
    assert (folder / "sub" / "three.txt").read_text() == "3"


def test_moved_file_is_restored_and_destination_removed(env):
    ws, data = env
    (ws.root / "a.txt").write_text("moved")

    checkpoints.snapshot_call(data, ws, "file_manage", {"action": "move", "path": "a.txt", "destination": "b.txt"}, "c1")
    shutil.move(str(ws.root / "a.txt"), str(ws.root / "b.txt"))

    checkpoints.store_for(data).undo("c1")

    assert (ws.root / "a.txt").read_text() == "moved"
    assert not (ws.root / "b.txt").exists()


def test_identical_content_is_stored_once(env):
    ws, data = env
    (ws.root / "x.txt").write_text("same content")
    (ws.root / "y.txt").write_text("same content")

    checkpoints.snapshot_call(data, ws, "write_file", {"path": "x.txt", "content": ""}, "c1")
    checkpoints.snapshot_call(data, ws, "write_file", {"path": "y.txt", "content": ""}, "c1")

    assert len(_objects(data)) == 1


# --- bounds ---------------------------------------------------------------------------------


def test_prune_keeps_at_most_max_rows_and_deletes_orphaned_objects(env, monkeypatch):
    ws, data = env
    monkeypatch.setattr(checkpoints, "MAX_ROWS", 3)
    target = ws.root / "f.txt"
    target.write_text("v0")

    for number in range(1, 6):
        checkpoints.snapshot_call(data, ws, "write_file", {"path": "f.txt", "content": f"v{number}"}, "c1")
        target.write_text(f"v{number}")

    store = checkpoints.store_for(data)
    store.prune()

    assert len(store.history("c1", limit=50)) <= 3
    assert len(_objects(data)) == 3


# --- the undo_changes tool ------------------------------------------------------------------------


def test_undo_tool_validates_steps():
    assert checkpoints.UNDO_CHANGES.validate({"steps": "2"}) == {"path": None, "steps": 2}
    with pytest.raises(ValueError):
        checkpoints.UNDO_CHANGES.validate({"steps": 0})
    with pytest.raises(ValueError):
        checkpoints.UNDO_CHANGES.validate({"steps": 11})


def test_undo_tool_restores_the_last_change(env):
    ws, data = env
    (ws.root / "a.txt").write_text("old")
    checkpoints.snapshot_call(data, ws, "write_file", {"path": "a.txt", "content": "new"}, "c1")
    (ws.root / "a.txt").write_text("new")

    args = checkpoints.UNDO_CHANGES.validate({})
    result = checkpoints.UNDO_CHANGES.execute(args, _ctx(ws, data))

    assert result.ok is True
    assert (ws.root / "a.txt").read_text() == "old"


def test_undo_tool_with_nothing_to_undo_says_so(env):
    ws, data = env
    args = checkpoints.UNDO_CHANGES.validate({})
    result = checkpoints.UNDO_CHANGES.execute(args, _ctx(ws, data, conversation_id="c-empty"))

    assert result.ok is True
    assert "Nothing to undo" in result.content


# --- never raises ----------------------------------------------------------------------------------


def test_snapshot_call_never_raises(env):
    ws, data = env
    assert checkpoints.snapshot_call(data, None, "write_file", {"path": "a.txt", "content": "x"}, "c1") is None
    assert checkpoints.snapshot_call(data, ws, "file_manage", None, "c1") is None


def test_undo_folder_moved_into_existing_folder(tmp_path):
    """Regression: moving a folder into an existing one, then undoing, removes the moved copy."""
    ws = Workspace(tmp_path / "ws")
    data = tmp_path / "data"
    src = ws.root / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("A")
    (src / "sub" / "b.txt").write_text("B")
    (ws.root / "dest").mkdir()
    args = {"action": "move", "path": "src", "destination": "dest"}
    assert checkpoints.snapshot_call(data, ws, "file_manage", args, "move-dir")
    shutil.move(str(src), str(ws.root / "dest" / "src"))
    reports = checkpoints.store_for(data).undo("move-dir")
    assert (ws.root / "src" / "a.txt").read_text() == "A"
    assert (ws.root / "src" / "sub" / "b.txt").read_text() == "B"
    assert not (ws.root / "dest" / "src" / "a.txt").exists()
    assert not (ws.root / "dest" / "src" / "sub" / "b.txt").exists()
    assert all("failed" not in item["action"] for item in reports)


def test_extension_tools_declare_touched_files_for_undo(tmp_path):
    """convert_document creates a new file; undo must remove it (via ExtensionTool.touches)."""
    ws = Workspace(tmp_path / "ws")
    data = tmp_path / "data"
    (ws.root / "notes.md").write_text("# Notes\n", encoding="utf-8")
    (ws.root / "notes.html").write_text("<p>existing</p>", encoding="utf-8")  # forces notes-2.html
    args = {"path": "notes.md", "to": "html", "output": None}
    assert checkpoints.files_for("convert_document", args, ws) == [(ws.root / "notes-2.html", "notes-2.html")]
    assert checkpoints.snapshot_call(data, ws, "convert_document", args, "conv")
    (ws.root / "notes-2.html").write_text("<h1>Notes</h1>", encoding="utf-8")  # what the conversion produces
    checkpoints.store_for(data).undo("conv")
    assert not (ws.root / "notes-2.html").exists()
    assert (ws.root / "notes.html").read_text(encoding="utf-8") == "<p>existing</p>"
