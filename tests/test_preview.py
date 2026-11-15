"""Showing what Kiwibro made: the files route, the sandboxed preview route, the folder badge and open."""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import pytest

from kiwibro import desktop, preview


@pytest.fixture()
def chat(env):  # noqa: ANN001 - the shared Flask fixture
    conversation = env.svc.db.create_conversation()
    client = env.app.test_client()
    ws = env.workspace
    (ws / "game").mkdir()
    (ws / "game" / "index.html").write_text("<!doctype html><head><title>t</title></head><body>hi</body>", encoding="utf-8")
    (ws / "game" / "style.css").write_text("body{margin:0}", encoding="utf-8")
    (ws / "logo.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    (ws / "notes.py").write_text("print(1)", encoding="utf-8")
    (ws / ".env").write_text("KEY=1", encoding="utf-8")
    (ws / ".hidden").mkdir()
    (ws / ".hidden" / "x.png").write_bytes(b"png")
    return client, conversation["id"], ws


def _files(client, cid, **body):
    reply = client.post("/api/workspace/files", json={"conversation_id": cid, **body})
    assert reply.status_code == 200, reply.get_json()
    return reply.get_json()["files"]


def test_files_lists_only_visual_files_inside_the_workspace(chat, tmp_path) -> None:
    client, cid, ws = chat
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"png")
    (ws / "link.png").symlink_to(outside)
    files = _files(client, cid, written=["game/index.html", "logo.svg", "game/style.css", "notes.py",
                                         ".env", ".hidden/x.png", "link.png", "../outside.png", "/etc/hostname",
                                         "missing.png"])
    assert [f["path"] for f in files] == ["game/index.html", "logo.svg"]
    assert files[0]["kind"] == "html" and files[0]["url"] == f"/api/preview/{cid}/game/index.html"


def test_check_names_the_ordinary_files_that_exist(chat) -> None:
    client, cid, ws = chat
    reply = client.post("/api/workspace/files", json={"conversation_id": cid, "check": [
        "notes.py", "game/style.css", "./game/index.html", ".env", ".hidden/x.png", "../x.py", "missing.py", "game"]})
    assert reply.get_json()["present"] == {
        "notes.py": "notes.py", "game/style.css": "game/style.css", "./game/index.html": "game/index.html"}


def test_named_files_must_be_new_since_the_turn_began(chat) -> None:
    client, cid, ws = chat
    old = ws / "old.png"
    old.write_bytes(b"png")
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    (ws / "fresh.png").write_bytes(b"png")
    files = _files(client, cid, named=["old.png", "fresh.png"], since=time.time() - 60)
    assert [f["path"] for f in files] == ["fresh.png"]


def test_unknown_chat_and_bad_input_are_refused(chat) -> None:
    client, cid, _ = chat
    assert client.post("/api/workspace/files", json={"conversation_id": "c_nope", "written": []}).status_code == 404
    assert client.post("/api/workspace/files", json={"written": []}).status_code == 400
    assert client.post("/api/workspace/files", json={"conversation_id": cid, "written": "a.png"}).status_code == 400


def test_page_is_sandboxed_and_loads_only_from_its_own_folder_tree(chat) -> None:
    client, cid, _ = chat
    reply = client.get(f"/api/preview/{cid}/game/index.html")
    assert reply.status_code == 200
    csp = reply.headers["Content-Security-Policy"]
    assert csp.startswith("sandbox allow-scripts;") and "allow-same-origin" not in csp
    assert f"script-src 'unsafe-inline' http://localhost/api/preview/{cid}/" in csp
    assert "connect-src http://localhost/api/preview/" in csp and "https:" not in csp
    body = reply.get_data(as_text=True)
    assert body.index("kiwibroPreview") < body.index("<title>")  # the monitor runs first
    assert reply.headers["X-Content-Type-Options"] == "nosniff"
    assert client.get(f"/api/preview/{cid}/game/style.css").mimetype == "text/css"


def test_preview_route_refuses_everything_that_is_not_a_safe_workspace_file(chat, tmp_path) -> None:
    client, cid, ws = chat
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"png")
    (ws / "link.png").symlink_to(secret)
    for rel in ("notes.py", ".env", ".hidden/x.png", "link.png", "missing.png", "game", "..%2Fsecret.png"):
        assert client.get(f"/api/preview/{cid}/{rel}").status_code == 404, rel
    assert client.get("/api/preview/c_nope/logo.svg").status_code == 404


def test_svg_served_on_its_own_cannot_run_script(chat) -> None:
    client, cid, _ = chat
    reply = client.get(f"/api/preview/{cid}/logo.svg")
    assert reply.mimetype == "image/svg+xml"
    assert reply.headers["Content-Security-Policy"].startswith("sandbox allow-scripts;")
    assert "script-src" not in reply.headers["Content-Security-Policy"]


def test_a_sandboxed_page_may_read_its_own_files(chat) -> None:
    client, cid, _ = chat
    reply = client.get(f"/api/preview/{cid}/game/style.css", headers={"Origin": "null"})
    assert reply.headers["Access-Control-Allow-Origin"] == "null"
    other = client.get(f"/api/preview/{cid}/game/style.css", headers={"Origin": "https://example.com"})
    assert "Access-Control-Allow-Origin" not in other.headers


def test_badge_shows_folder_and_branch(chat, monkeypatch) -> None:
    client, cid, ws = chat
    info = client.get(f"/api/workspace/info?conversation_id={cid}").get_json()
    assert info["abs"] == str(ws.resolve()) and info["branch"] == "" and info["safe_copy"] is None
    if shutil.which("git"):
        subprocess.run(["git", "init", "-q", "-b", "trunk"], cwd=ws, check=True)
        assert client.get(f"/api/workspace/info?conversation_id={cid}").get_json()["branch"] == "trunk"
    monkeypatch.setenv("HOME", str(ws.parent))
    assert preview.folder_badge(ws)["path"] == "~/workspace"


def test_open_goes_through_open_path(chat, monkeypatch) -> None:
    client, cid, ws = chat
    opened = []
    monkeypatch.setattr(desktop, "open_path", lambda workspace, target: opened.append(target) or {"opened": target})
    ok = client.post("/api/workspace/open", json={"conversation_id": cid, "action": "open", "path": "logo.svg"})
    assert ok.status_code == 200
    ok = client.post("/api/workspace/open", json={"conversation_id": cid, "action": "reveal", "path": "game/index.html"})
    assert ok.status_code == 200
    assert opened == [str((ws / "logo.svg").resolve()), str((ws / "game").resolve())]
    outside = client.post("/api/workspace/open", json={"conversation_id": cid, "action": "open", "path": "/etc/hostname"})
    assert outside.status_code == 400 and len(opened) == 2
    bad = client.post("/api/workspace/open", json={"conversation_id": cid, "action": "run", "path": "logo.svg"})
    assert bad.status_code == 400


def test_open_respects_a_disabled_tool_and_a_program(chat, monkeypatch, env) -> None:
    client, cid, ws = chat
    monkeypatch.setattr(desktop, "open_path", lambda workspace, target: {"opened": target})
    (ws / "run.sh").write_text("echo hi", encoding="utf-8")
    program = client.post("/api/workspace/open", json={"conversation_id": cid, "action": "open", "path": "run.sh"})
    assert program.status_code == 403
    permissions = {**env.svc.settings_provider().get("permissions", {}), "open_path": "deny"}
    env.svc.db.patch_settings({"permissions": permissions, "autonomy": "supervised"})
    denied = client.post("/api/workspace/open", json={"conversation_id": cid, "action": "open", "path": "logo.svg"})
    assert denied.status_code == 403
