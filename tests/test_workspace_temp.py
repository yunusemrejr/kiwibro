"""Every new chat gets a temporary folder of its own; the folder picker's two endpoints.

The app here is started without a forced workspace (the tests elsewhere force one), with HOME and
the system temp folder inside the test's own tmp_path."""

from __future__ import annotations

import io
import stat
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_backend import Env, FakeLLM, chat_events, tool_call_result

from kiwibro import config, create_app, safety, workspaces
from kiwibro.llm import ChatResult
from kiwibro.sandbox import Workspace


@pytest.fixture()
def temp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    home = tmp_path / "home"
    (home / "Desktop" / "site").mkdir(parents=True)
    (home / "Desktop" / "Apple").mkdir()
    (home / "Desktop" / "banana").mkdir()
    (home / "Desktop" / ".hidden").mkdir()
    (home / ".ssh").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("KIWIBRO_WORKSPACE", raising=False)
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    app = create_app({"TESTING": True, "KIWIBRO_DATA_DIR": str(tmp_path / "data"), "KIWIBRO_SERVER_URL": "http://127.0.0.1:9"})
    svc = app.extensions["kiwibro"]
    fake = FakeLLM()
    svc.agent.memory_idle_s = 0.0
    monkeypatch.setattr(svc.llm, "list_models", fake.list_models)
    monkeypatch.setattr(svc.llm, "stream_chat", fake.stream_chat)
    monkeypatch.setattr(svc.llm, "complete", fake.complete)
    monkeypatch.setattr(svc.llm, "complete_json", fake.complete_json)
    return Env(app, svc, fake, home / "Kiwibro")


def _home() -> Path:
    return Path.home()


def _new_chat(env: Env, **body) -> dict:
    response = env.client.post("/api/conversations", json=body)
    assert response.status_code == 201, response.get_json()
    return response.get_json()


def _info(env: Env, chat_id: str) -> dict:
    response = env.client.get("/api/workspace/info", query_string={"conversation_id": chat_id})
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def _set(env: Env, chat_id: str, **body):
    return env.client.post("/api/workspace/set", json={"conversation_id": chat_id, **body})


# --- a folder at creation ----------------------------------------------------------------------


def test_a_new_chat_gets_its_own_temporary_folder(temp_env: Env) -> None:
    chat = _new_chat(temp_env)
    folder = Path(chat["workspace"])
    assert folder == workspaces.temp_root() / f"chat-{chat['id'][2:10]}"
    assert workspaces.temp_root().parent == Path(tempfile.gettempdir()) and folder.is_dir()
    assert stat.S_IMODE(workspaces.temp_root().stat().st_mode) == 0o700
    info = _info(temp_env, chat["id"])
    assert info["abs"] == str(folder) and info["temp"] is True and info["name"] == folder.name


def test_a_too_open_temp_root_is_made_private_again(temp_env: Env) -> None:
    root = workspaces.temp_root()
    root.mkdir(mode=0o777)
    root.chmod(0o777)
    _new_chat(temp_env)
    assert stat.S_IMODE(root.stat().st_mode) == 0o700


def test_a_chat_started_by_its_first_message_works_in_its_temp_folder(temp_env: Env) -> None:
    temp_env.fake.stream_script = [tool_call_result("write_file", {"path": "hello.txt", "content": "hi"}), ChatResult(content="Done.")]
    chat_events(temp_env, "write hello.txt")
    chat = temp_env.svc.db.list_conversations()[0]
    folder = Path(chat["workspace"])
    assert workspaces.in_temp_root(folder) and (folder / "hello.txt").read_text() == "hi"
    assert not (temp_env.workspace / "hello.txt").exists()
    assert str(folder) in temp_env.fake.stream_calls[0]["messages"][0]["content"]  # message 0 names it
    files = temp_env.client.get("/api/workspace", query_string={"conversation_id": chat["id"]}).get_json()
    assert files["root"] == str(folder) and "hello.txt" in str(files["files"])
    shown = temp_env.client.post("/api/workspace/files", json={"conversation_id": chat["id"], "check": ["hello.txt"]})
    assert shown.status_code == 200 and shown.get_json()["present"] == {"hello.txt": "hello.txt"}


def test_the_preview_open_and_attachment_routes_use_the_temp_folder(temp_env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    from kiwibro import desktop

    chat = _new_chat(temp_env)
    folder = Path(chat["workspace"])
    (folder / "page.html").write_text("<h1>hi</h1>", encoding="utf-8")
    served = temp_env.client.get(f"/api/preview/{chat['id']}/page.html")
    assert served.status_code == 200 and b"<h1>hi</h1>" in served.data
    opened: list[str] = []
    monkeypatch.setattr(desktop, "open_path", lambda workspace, target: opened.append(target) or {"opened": target})
    reveal = temp_env.client.post("/api/workspace/open", json={"conversation_id": chat["id"], "action": "reveal", "path": "page.html"})
    assert reveal.status_code == 200 and opened == [str(folder)]
    upload = temp_env.client.post(
        "/api/attachments",
        data={"conversation_id": chat["id"], "file": (io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\0" * 32), "p.png")},
        content_type="multipart/form-data",
    )
    assert upload.status_code == 200, upload.get_json()
    assert (folder / upload.get_json()["attachment"]["path"]).is_file()


def test_old_chats_without_a_folder_keep_the_settings_folder(temp_env: Env) -> None:
    old = temp_env.svc.db.create_conversation()  # as a chat made before this version
    assert old["workspace"] is None
    info = _info(temp_env, old["id"])
    assert info["abs"] == str(temp_env.workspace.resolve()) and info["temp"] is False
    assert workspaces.folder_for(temp_env.svc.db, old["id"]) is None


def test_settings_without_the_key_default_to_temp(temp_env: Env) -> None:
    assert temp_env.svc.settings_provider()["workspace_mode"] == "temp"
    assert temp_env.settings_patch({"workspace_mode": "folder"})["workspace_mode"] == "folder"
    assert temp_env.client.patch("/api/settings", json={"workspace_mode": "nowhere"}).status_code == 400
    assert _new_chat(temp_env)["workspace"] is None  # folder mode: today's behaviour


def test_a_forced_workspace_forces_folder_mode(env: Env) -> None:
    assert env.svc.settings_provider()["workspace_mode"] == "folder"
    env.svc.db.patch_settings({"workspace_mode": "temp"})  # even a stored "temp" does not win
    assert env.svc.settings_provider()["workspace_mode"] == "folder"
    assert _new_chat(env)["workspace"] is None
    assert config.default_settings()["workspace_mode"] == "temp"


def test_a_cleared_temp_folder_is_made_again_but_other_folders_are_not(temp_env: Env) -> None:
    chat = _new_chat(temp_env)
    folder = Path(chat["workspace"])
    (folder / "x.txt").write_text("gone soon")
    (folder / "x.txt").unlink()
    folder.rmdir()
    assert _info(temp_env, chat["id"])["abs"] == str(folder)
    assert folder.is_dir() and list(folder.iterdir()) == []
    # the whole temp root cleared (a restart)
    folder.rmdir()
    folder.parent.rmdir()
    assert workspaces.folder_for(temp_env.svc.db, chat["id"]) == str(folder) and folder.is_dir()
    assert stat.S_IMODE(folder.parent.stat().st_mode) == 0o700
    # a folder of the user's that vanished stays as it is: nothing is made for it
    mine = _home() / "Desktop" / "mine"
    mine.mkdir()
    other = _new_chat(temp_env, workspace=str(mine))
    mine.rmdir()
    assert workspaces.folder_for(temp_env.svc.db, other["id"]) == str(mine) and not mine.exists()


# --- a folder chosen when the chat is made --------------------------------------------------------


def test_a_chat_can_be_made_with_a_chosen_folder(temp_env: Env) -> None:
    site = _home() / "Desktop" / "site"
    chat = _new_chat(temp_env, workspace="~/Desktop/site")
    assert chat["workspace"] == str(site)
    assert _info(temp_env, chat["id"])["temp"] is False
    before = len(temp_env.svc.db.list_conversations())
    for bad in ("relative/dir", "~/.ssh", "~", str(_home() / "nowhere"), ""):
        refused = temp_env.client.post("/api/conversations", json={"workspace": bad})
        assert refused.status_code == 400 and refused.get_json()["error"]["code"] == "bad_request", bad
    assert temp_env.client.post("/api/conversations", json={"workspace": 5}).status_code == 400
    assert len(temp_env.svc.db.list_conversations()) == before  # a refused folder makes no chat


# --- the set endpoint -------------------------------------------------------------------------------


def test_set_chooses_a_folder_and_goes_back_to_temp(temp_env: Env) -> None:
    chat = _new_chat(temp_env)
    own = chat["workspace"]
    chosen = _set(temp_env, chat["id"], path="~/Desktop/site")
    assert chosen.status_code == 200, chosen.get_json()
    body = chosen.get_json()
    assert body["ok"] is True and body["abs"] == str(_home() / "Desktop" / "site")
    assert body["path"] == "~/Desktop/site" and body["temp"] is False and body["name"] == "site"
    assert set(body) >= {"ok", "path", "abs", "name", "branch", "safe_copy", "temp"}
    assert temp_env.svc.db.get_conversation(chat["id"])["workspace"] == body["abs"]
    back = _set(temp_env, chat["id"], temp=True).get_json()
    assert back["temp"] is True and back["abs"] == own
    assert _info(temp_env, chat["id"]) == back


def test_set_temp_recreates_the_folder_and_works_for_old_chats(temp_env: Env) -> None:
    old = temp_env.svc.db.create_conversation()
    body = _set(temp_env, old["id"], temp=True).get_json()
    assert body["temp"] is True and Path(body["abs"]) == workspaces.temp_folder_of(old["id"]) and Path(body["abs"]).is_dir()


def test_set_is_busy_while_a_reply_runs(temp_env: Env) -> None:
    chat = _new_chat(temp_env)
    lock = temp_env.svc.agent._lock_for(chat["id"])
    assert lock.acquire(blocking=False)
    try:
        busy = _set(temp_env, chat["id"], path="~/Desktop/site")
        assert busy.status_code == 409 and busy.get_json()["error"]["code"] == "busy"
        assert _set(temp_env, chat["id"], temp=True).status_code == 409
    finally:
        lock.release()
    assert temp_env.svc.db.get_conversation(chat["id"])["workspace"] == chat["workspace"]
    assert _set(temp_env, chat["id"], path="~/Desktop/site").status_code == 200


@pytest.mark.parametrize("body", [
    {"path": "~/.ssh"}, {"path": "~"}, {"path": "/etc"}, {"path": "relative/dir"}, {"path": "~/Desktop/missing"},
    {"path": ""}, {"path": 7}, {"path": None}, {}, {"temp": False}, {"temp": 1}, {"temp": True, "path": "~/Desktop/site"},
])
def test_set_refuses_what_it_cannot_do(temp_env: Env, body: dict) -> None:
    chat = _new_chat(temp_env)
    refused = _set(temp_env, chat["id"], **body)
    assert refused.status_code == 400 and refused.get_json()["error"]["code"] == "bad_request"
    assert temp_env.svc.db.get_conversation(chat["id"])["workspace"] == chat["workspace"]


def test_set_needs_a_real_chat(temp_env: Env) -> None:
    assert temp_env.client.post("/api/workspace/set", json={"path": "~/Desktop/site"}).status_code == 400
    assert _set(temp_env, "c_nothere", path="~/Desktop/site").status_code == 404


def test_set_is_refused_while_a_safe_copy_is_open(temp_env: Env) -> None:
    chat = _new_chat(temp_env)
    (Path(chat["workspace"]) / "a.txt").write_text("a")
    started = temp_env.client.post("/api/safecopy/start", json={"conversation_id": chat["id"], "folder": chat["workspace"]})
    assert started.status_code == 200, started.get_json()  # a temp folder can be copied
    refused = _set(temp_env, chat["id"], path="~/Desktop/site")
    assert refused.status_code == 403 and "safe copy" in refused.get_json()["error"]["message"]
    assert _info(temp_env, chat["id"])["safe_copy"] is not None
    assert temp_env.client.post("/api/safecopy/end", json={"conversation_id": chat["id"], "discard": True}).status_code == 200
    assert temp_env.svc.db.get_conversation(chat["id"])["workspace"] == chat["workspace"]


# --- the folder list ----------------------------------------------------------------------------------


def _folders(env: Env, **query):
    return env.client.get("/api/workspace/folders", query_string=query)


def test_list_defaults_to_home_and_sorts_without_hidden_folders(temp_env: Env) -> None:
    (_home() / "notes.txt").write_text("a file")
    body = _folders(temp_env).get_json()
    assert body["ok"] is True and body["path"] == str(_home()) and body["parent"] is None
    assert body["allowed"] is False  # the home folder itself cannot be a chat's folder
    assert [f["name"] for f in body["folders"]] == ["Desktop", "Kiwibro"]
    desktop = _folders(temp_env, path="~/Desktop").get_json()
    assert [f["name"] for f in desktop["folders"]] == ["Apple", "banana", "site"]  # case-insensitive, no .hidden
    assert desktop["parent"] == str(_home()) and desktop["allowed"] is True
    assert desktop["folders"][0] == {"name": "Apple", "path": str(_home() / "Desktop" / "Apple"), "allowed": True}


def test_list_is_capped_at_200(temp_env: Env) -> None:
    big = _home() / "many"
    big.mkdir()
    for number in range(205):
        (big / f"d{number:03}").mkdir()
    body = _folders(temp_env, path=str(big)).get_json()
    assert len(body["folders"]) == 200 and body["truncated"] is True and body["folders"][0]["name"] == "d000"


def test_list_allows_only_home_and_the_temp_root(temp_env: Env, tmp_path: Path) -> None:
    chat = _new_chat(temp_env)
    inside = _folders(temp_env, path=str(workspaces.temp_root())).get_json()
    assert inside["parent"] is None and [f["name"] for f in inside["folders"]] == [Path(chat["workspace"]).name]
    assert _folders(temp_env, path=chat["workspace"]).get_json()["allowed"] is True
    for outside in ("/etc", "/", str(tmp_path), tempfile.gettempdir(), "/home", str(_home().parent)):
        refused = _folders(temp_env, path=outside)
        assert refused.status_code == 403 and refused.get_json()["error"]["code"] == "not_allowed", outside
    assert _folders(temp_env, path="relative").status_code == 400
    assert _folders(temp_env, path=str(_home() / "missing")).status_code == 400
    assert _folders(temp_env, path=str(_home() / "Desktop" / "site" / ".." / ".." / "..")).status_code == 403  # .. is resolved first


def test_list_never_follows_a_link_out_of_its_roots(temp_env: Env, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "secret-folder").mkdir(parents=True)
    (_home() / "Desktop" / "escape").symlink_to(elsewhere)
    (_home() / "Desktop" / "loop").symlink_to(_home() / "Desktop" / "loop")
    (_home() / "Desktop" / "dangling").symlink_to(tmp_path / "nothing")
    (_home() / "Desktop" / "inside").symlink_to(_home() / "Desktop" / "site")
    listed = _folders(temp_env, path="~/Desktop").get_json()
    names = {f["name"]: f for f in listed["folders"]}
    assert "escape" not in names and "loop" not in names and "dangling" not in names
    assert names["inside"]["path"] == str(_home() / "Desktop" / "site")  # shown by where it leads
    through = _folders(temp_env, path="~/Desktop/escape")
    assert through.status_code == 403 and "secret-folder" not in through.get_data(as_text=True)


def test_list_marks_what_a_chat_may_not_use(temp_env: Env) -> None:
    (_home() / "Documents").mkdir()
    (_home() / "bin").mkdir()
    body = {f["name"]: f["allowed"] for f in _folders(temp_env).get_json()["folders"]}
    assert body["Documents"] is True
    for name, allowed in body.items():
        try:
            workspaces.resolve_folder(str(_home() / name))
            expected = True
        except ValueError:
            expected = False
        assert allowed is expected, name
    assert _folders(temp_env, path="~/.ssh").get_json()["allowed"] is False


# --- the workspace tool ----------------------------------------------------------------------------------


def _ctx(db, chat_id: str, root: Path, settings: dict | None) -> SimpleNamespace:
    return SimpleNamespace(memory=SimpleNamespace(db=db), conversation_id=chat_id, workspace=Workspace(root), settings=settings)


def test_reset_returns_to_the_temp_folder_or_the_settings_folder(temp_env: Env) -> None:
    db = temp_env.svc.db
    chat = _new_chat(temp_env)
    site = _home() / "Desktop" / "site"
    run = lambda settings: workspaces.execute({"action": "reset"}, _ctx(db, chat["id"], site, settings))  # noqa: E731
    db.update_conversation(chat["id"], workspace=str(site), touch=False)
    result = run({"workspace_mode": "temp"})
    assert result.ok and db.get_conversation(chat["id"])["workspace"] == chat["workspace"]
    assert chat["workspace"] in result.content
    db.update_conversation(chat["id"], workspace=str(site), touch=False)
    run({"workspace_mode": "folder"})
    assert db.get_conversation(chat["id"])["workspace"] is None
    shown = workspaces.execute({"action": "show"}, _ctx(db, chat["id"], Path(chat["workspace"]), None))
    assert "default workspace" in shown.content
    db.update_conversation(chat["id"], workspace=chat["workspace"], touch=False)
    assert "temporary folder" in workspaces.execute({"action": "show"}, _ctx(db, chat["id"], Path(chat["workspace"]), None)).content


def test_the_model_can_switch_and_reset_in_a_chat(temp_env: Env) -> None:
    target = _home() / "Desktop" / "site"
    temp_env.fake.stream_script = [
        tool_call_result("workspace", {"action": "switch", "path": str(target)}),
        tool_call_result("workspace", {"action": "reset"}, call_id="call_2"),
        ChatResult(content="Back."),
    ]
    chat_events(temp_env, "Please work on my site project on the desktop, then go back.")
    chat = temp_env.svc.db.list_conversations()[0]
    assert Path(chat["workspace"]) == workspaces.temp_folder_of(chat["id"])


# --- what depends on the folder -----------------------------------------------------------------------------


def test_the_chat_root_is_the_workspace_not_plain_tmp(temp_env: Env) -> None:
    folder = Path(_new_chat(temp_env)["workspace"])
    assert safety.place_of(str(folder), workspace=folder) == "workspace_root"
    assert safety.place_of(str(folder / "a" / "b.txt"), workspace=folder) == "workspace"
    assert safety.place_of(str(folder / ".git" / "config"), workspace=folder) == "git"
    assert safety.place_of(str(workspaces.temp_root() / "chat-other"), workspace=folder) == "tmp"  # another chat's folder
    workspaces.resolve_folder(str(folder))  # and the picker may choose it


def test_undo_restores_a_file_of_a_temp_folder(temp_env: Env) -> None:
    from kiwibro import checkpoints

    folder = Path(_new_chat(temp_env)["workspace"])
    workspace = Workspace(folder)
    target = folder / "a.txt"
    target.write_text("old")
    assert checkpoints.snapshot_call(temp_env.svc.data_dir, workspace, "write_file", {"path": "a.txt", "content": "new"}, "c1")
    target.write_text("new")
    checkpoints.store_for(temp_env.svc.data_dir).undo("c1")
    assert target.read_text() == "old"


@pytest.mark.skipif(not safety.sandbox_status()["active"], reason="no bubblewrap sandbox on this machine")
def test_a_command_in_the_box_sees_and_writes_the_real_temp_folder(temp_env: Env) -> None:
    from kiwibro import sandbox

    folder = Path(_new_chat(temp_env)["workspace"])
    (folder / "from-host.txt").write_text("seen")
    argv = sandbox.shell_argv("cat from-host.txt && echo && echo written > from-box.txt && pwd", folder, folder)
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["seen", str(folder)]
    assert (folder / "from-box.txt").read_text() == "written\n"
