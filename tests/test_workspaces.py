"""Per-conversation working folders: binding, validation and the switch policy."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kiwibro import safety, workspaces
from kiwibro.db import Database
from kiwibro.sandbox import Workspace


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    (home / "Desktop" / "kiwi-site" / "src").mkdir(parents=True)
    (home / ".ssh").mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "data" / "kiwibro.db")
    database.initialize()
    return database


def _ctx(db: Database, conversation_id: str, root: Path) -> SimpleNamespace:
    return SimpleNamespace(memory=SimpleNamespace(db=db), conversation_id=conversation_id, workspace=Workspace(root))


def test_migration_adds_an_optional_folder(db: Database) -> None:
    conversation = db.create_conversation()
    assert conversation["workspace"] is None
    updated = db.update_conversation(conversation["id"], workspace="/home/x/proj", touch=False)
    assert updated["workspace"] == "/home/x/proj"
    assert workspaces.folder_for(db, conversation["id"]) == "/home/x/proj"


def test_switch_show_and_reset(home: Path, db: Database) -> None:
    conversation_id = db.create_conversation()["id"]
    target = home / "Desktop" / "kiwi-site"
    args = workspaces.validate({"action": "switch", "path": "~/Desktop/kiwi-site"})
    result = workspaces.execute(args, _ctx(db, conversation_id, home / "Kiwibro"))
    assert result.ok and "src/" in result.content
    assert workspaces.folder_for(db, conversation_id) == str(target)
    shown = workspaces.execute({"action": "show"}, _ctx(db, conversation_id, target))
    assert "this conversation's folder" in shown.content
    workspaces.execute({"action": "reset"}, _ctx(db, conversation_id, target))
    assert workspaces.folder_for(db, conversation_id) is None


@pytest.mark.parametrize("raw", ["~", "~/.ssh", "/etc", "relative/path", "~/Desktop/missing", ""])
def test_unsafe_or_missing_folders_are_refused(home: Path, raw: str) -> None:
    with pytest.raises(ValueError):
        workspaces.resolve_folder(raw)


def test_switch_is_auto_only_when_the_user_named_the_folder(home: Path) -> None:
    path = str(home / "Desktop" / "kiwi-site")
    asked = safety.decide({}, "workspace", {"action": "switch", "path": path},
                          turn_state={"user_message": "please work on my kiwi-site project on the desktop"})
    assert (asked.mode, asked.level) == ("auto", "low")
    unasked = safety.decide({}, "workspace", {"action": "switch", "path": path},
                            turn_state={"user_message": "fix the failing test"})
    assert (unasked.mode, unasked.level) == ("ask", "high")
    refused = safety.decide({"autonomy": "autonomous"}, "workspace", {"action": "switch", "path": path},
                            turn_state={"user_message": "fix the failing test"})
    assert refused.mode == "deny"
    assert safety.decide({}, "workspace", {"action": "show"}).mode == "auto"


# --- agent integration ----------------------------------------------------------

from test_backend import Env, chat_events, env, tool_call_result  # noqa: E402,F401  (shared app fixture)

from kiwibro.llm import ChatResult  # noqa: E402


def test_agent_works_in_the_conversation_folder(env: Env, tmp_path: Path) -> None:  # noqa: F811
    folder = tmp_path / "proj"
    folder.mkdir()
    conversation_id = env.svc.db.create_conversation()["id"]
    env.svc.db.update_conversation(conversation_id, workspace=str(folder), touch=False)
    env.fake.stream_script = [tool_call_result("write_file", {"path": "hello.txt", "content": "hi"}), ChatResult(content="Done.")]
    chat_events(env, "write hello.txt", conversation_id=conversation_id)
    assert (folder / "hello.txt").read_text() == "hi"
    assert not (env.workspace / "hello.txt").exists()
    assert str(folder) in env.fake.stream_calls[0]["messages"][0]["content"]  # message 0 names it


def test_user_asked_switch_then_work_there(env: Env, home: Path) -> None:  # noqa: F811
    target = home / "Desktop" / "kiwi-site"
    env.fake.stream_script = [
        tool_call_result("workspace", {"action": "switch", "path": str(target)}),
        tool_call_result("write_file", {"path": "index.html", "content": "<h1>kiwi</h1>"}, call_id="call_2"),
        ChatResult(content="Switched and wrote index.html."),
    ]
    events, _raw = chat_events(env, "Please work on my kiwi-site project on the desktop and add an index page.")
    assert "approval" not in [event for event, _ in events]
    assert (target / "index.html").read_text() == "<h1>kiwi</h1>"
    conversation = env.svc.db.list_conversations()[0]
    assert env.client.get(f"/api/conversations/{conversation['id']}").get_json()["conversation"]["workspace"] == str(target)
