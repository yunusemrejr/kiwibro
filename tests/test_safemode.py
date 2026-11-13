"""Tests for kiwibro.safemode: safe copy sessions in the app, the tool gate and the chat messages."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from test_backend import Env, chat_events, tool_call_result

from kiwibro import safemode, safecopy, safety
from kiwibro.llm import ChatResult
from kiwibro.sandbox import SandboxError


def _chat(env: Env) -> str:
    response = env.client.post("/api/conversations", json={"title": "Safe copy"})
    assert response.status_code == 201, response.get_json()
    return response.get_json()["id"]


@pytest.fixture
def project(tmp_path: Path) -> Path:
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "notes.txt").write_text("one\ntwo\n")
    (folder / "old.txt").write_text("old\n")
    return folder


def _start(env: Env, conversation_id: str, folder: Path) -> dict:
    response = env.client.post(
        "/api/safecopy/start", json={"conversation_id": conversation_id, "folder": str(folder)}
    )
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def _session(env: Env, conversation_id: str) -> safecopy.Session:
    session = safemode.current(env.svc.db, conversation_id)
    assert session is not None
    return session


def _wait_turn_end(env: Env, timeout: float = 10.0) -> None:
    """The worker thread records the step and releases the chat a moment after the stream ends."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and env.svc.agent.active_turns():
        time.sleep(0.02)
    assert env.svc.agent.active_turns() == 0


def _run_tool(env: Env, conversation_id: str, name: str, arguments: dict) -> dict:
    env.fake.stream_script = [tool_call_result(name, arguments), ChatResult(content="Done.")]
    events, _raw = chat_events(env, f"please use {name}", conversation_id)
    _wait_turn_end(env)
    return next(payload for event, payload in events if event == "tool_result")


# --- starting, status, ending -----------------------------------------------------------------


def test_start_through_the_api_copies_the_folder(env: Env, project: Path):
    conversation_id = _chat(env)
    result = _start(env, conversation_id, project)
    assert result["active"] is True and result["started"] is True
    assert result["real"] == str(project.resolve())
    assert Path(result["work"]).name == "work"
    assert (Path(result["work"]) / "notes.txt").read_text() == "one\ntwo\n"
    assert Path(result["work"]) != project
    assert result["changed_files"] == 0
    assert result["applied"] is False and result["can_undo"] is False
    assert Path(result["work"]).is_relative_to(env.svc.data_dir)
    remembered = json.loads(env.svc.db.get_conversation(conversation_id)["safe_copy"])
    assert remembered["real"] == str(project.resolve())
    again = _start(env, conversation_id, project)
    assert again["started"] is False and again["work"] == result["work"]
    status = env.client.get(f"/api/safecopy/status?conversation_id={conversation_id}").get_json()
    assert status["active"] is True and status["changed_files"] == 0


def test_start_refuses_the_wrong_folders(env: Env, project: Path):
    conversation_id = _chat(env)
    inside_data = Path(env.svc.data_dir) / "library"
    inside_data.mkdir()
    for folder in (
        "/",
        str(Path.home()),
        str(env.svc.data_dir),
        str(inside_data),
        str(Path(env.svc.data_dir) / "kiwibro.db"),
    ):
        response = env.client.post(
            "/api/safecopy/start", json={"conversation_id": conversation_id, "folder": folder}
        )
        assert response.status_code == 400, (folder, response.get_json())
        assert response.get_json()["error"]["code"] in {"refused_folder", "not_a_folder"}
    assert safemode.current(env.svc.db, conversation_id) is None


def test_end_detaches_and_discard_removes_only_the_copy(env: Env, project: Path):
    conversation_id = _chat(env)
    started = _start(env, conversation_id, project)
    copy = Path(started["work"])
    response = env.client.post(
        "/api/safecopy/end", json={"conversation_id": conversation_id, "discard": True}
    )
    assert response.status_code == 200, response.get_json()
    assert response.get_json()["discarded"] is True
    assert not copy.parent.exists()
    assert (project / "notes.txt").read_text() == "one\ntwo\n"
    assert env.svc.db.get_conversation(conversation_id)["safe_copy"] is None
    assert env.client.get(f"/api/safecopy/status?conversation_id={conversation_id}").get_json()["active"] is False


def test_old_detached_sessions_are_cleaned_at_start(env: Env, project: Path):
    kept = _chat(env)
    _start(env, kept, project)
    detached = _chat(env)
    started = _start(env, detached, project)
    env.client.post("/api/safecopy/end", json={"conversation_id": detached, "discard": False})
    old = Path(started["work"]).parent
    meta_path = old / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["created"] = time.time() - 20 * 86400
    meta_path.write_text(json.dumps(meta))
    removed = safemode.cleanup_at_start(env.svc.db)
    assert removed == [old.name]
    assert not old.exists()
    assert Path(_start(env, kept, project)["work"]).is_relative_to(env.svc.data_dir)
    assert safemode.current(env.svc.db, kept) is not None


# --- the gate on tools ------------------------------------------------------------------------


def test_a_file_writing_tool_lands_in_the_copy(env: Env, project: Path):
    conversation_id = _chat(env)
    started = _start(env, conversation_id, project)
    result = _run_tool(env, conversation_id, "write_file", {"path": "notes.txt", "content": "from the agent\n"})
    assert result["ok"] is True
    assert (Path(started["work"]) / "notes.txt").read_text() == "from the agent\n"
    assert (project / "notes.txt").read_text() == "one\ntwo\n"
    assert safemode.current(env.svc.db, conversation_id).commit("check") is None  # the turn already recorded it


def test_a_path_outside_the_copy_is_refused_for_reads_and_writes(env: Env, project: Path, tmp_path: Path):
    (tmp_path / "secret.txt").write_text("not yours\n")
    conversation_id = _chat(env)
    started = _start(env, conversation_id, project)
    read = _run_tool(env, conversation_id, "read_file", {"path": "../secret.txt"})
    assert read["ok"] is False and "outside" in read["preview"]
    write = _run_tool(env, conversation_id, "write_file", {"path": str(tmp_path / "secret.txt"), "content": "x"})
    assert write["ok"] is False
    assert (tmp_path / "secret.txt").read_text() == "not yours\n"
    assert not (Path(started["work"]).parent / "secret.txt").exists()


def test_each_refused_tool_is_refused_whatever_the_settings(env: Env, project: Path):
    conversation_id = _chat(env)
    _start(env, conversation_id, project)
    env.settings_patch({"autonomy": "autonomous", "permissions": {name: "auto" for name in safemode.REFUSED}})
    for name in sorted(safemode.REFUSED):
        assert safemode.refusal_for(name), name
    assert safemode.refusal_for("some_third_party_tool")
    assert safemode.refusal_for("forget_everything")
    assert safemode.refusal_for("read_file") is None
    assert safemode.refusal_for("web_search") is None
    assert safemode.refusal_for("remember") is None
    assert safemode.refusal_for("component") is None
    assert safemode.refusal_for("run_command") is None
    result = _run_tool(env, conversation_id, "open_path", {"target": str(project)})
    assert result["ok"] is False
    assert result["preview"] == safemode.REFUSED["open_path"]


def test_the_chat_that_has_no_session_is_not_gated(env: Env, project: Path):
    conversation_id = _chat(env)
    assert safemode.gate(env.svc.db, conversation_id, "open_path").active is False
    assert safemode.gate(env.svc.db, conversation_id, "open_path").refusal is None
    assert safemode.copy_folder(env.svc.db, conversation_id) is None
    assert safemode.context_line(env.svc.db, conversation_id) == ""


def test_the_turn_context_says_where_the_agent_works(env: Env, project: Path):
    conversation_id = _chat(env)
    _start(env, conversation_id, project)
    line = safemode.context_line(env.svc.db, conversation_id)
    assert line.startswith("Safe copy: you are working in a copy of ")
    assert "Nothing reaches the real folder until the user reviews the changes." in line


def test_without_a_sandbox_commands_are_refused(env: Env, project: Path, monkeypatch: pytest.MonkeyPatch):
    conversation_id = _chat(env)
    _start(env, conversation_id, project)
    monkeypatch.setattr(safemode, "sandbox_ready", lambda: False)
    assert "sandbox" in (safemode.refusal_for("run_command") or "")
    workspace = safemode.workspace_for(env.svc.db, conversation_id)
    with pytest.raises(SandboxError):
        workspace.run_command("echo hi")
    result = _run_tool(env, conversation_id, "run_command", {"command": "echo hi"})
    assert result["ok"] is False and "sandbox" in result["preview"]


def test_an_approved_escape_cannot_leave_the_sandbox(env: Env, project: Path):
    conversation_id = _chat(env)
    _start(env, conversation_id, project)
    workspace = safemode.workspace_for(env.svc.db, conversation_id)
    decision = safety.Decision(mode="auto", level="high", reasons=("a push needs your keys",), escape=True)
    with safety.applied(decision, approved_by_user=True):
        assert safety.is_unconfined() is True
        with pytest.raises(SandboxError):
            workspace.run_command("git push")


needs_sandbox = pytest.mark.skipif(not safemode.sandbox_ready(), reason="no sandbox on this machine")


@needs_sandbox
def test_a_command_runs_sandboxed_without_network_and_cannot_write_outside(env: Env, project: Path):
    conversation_id = _chat(env)
    started = _start(env, conversation_id, project)
    workspace = safemode.workspace_for(env.svc.db, conversation_id)
    inside = workspace.run_command("echo hello > made.txt && cat made.txt")
    assert inside["exit_code"] == 0 and inside["output"].strip() == "hello"
    assert (Path(started["work"]) / "made.txt").read_text().strip() == "hello"
    assert not (project / "made.txt").exists()
    escape = workspace.run_command(f"echo x > {project}/escaped.txt")
    assert escape["exit_code"] != 0
    assert not (project / "escaped.txt").exists()
    offline = workspace.run_command(
        "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 443), 3)\""
    )
    assert offline["exit_code"] != 0
    result = _run_tool(env, conversation_id, "run_command", {"command": "echo tool > from_tool.txt"})
    assert result["ok"] is True
    assert (Path(started["work"]) / "from_tool.txt").read_text().strip() == "tool"
    assert not (project / "from_tool.txt").exists()
    escape_tool = _run_tool(env, conversation_id, "run_command", {"command": f"echo x > {project}/toolex.txt"})
    assert escape_tool["ok"] is False
    assert not (project / "toolex.txt").exists()
    assert safemode.current(env.svc.db, conversation_id).commit("check") is None  # the turn recorded it


# --- proof, apply, undo -----------------------------------------------------------------------


def test_proof_then_apply_writes_the_change_and_the_chat_carries_the_undo(env: Env, project: Path):
    conversation_id = _chat(env)
    started = _start(env, conversation_id, project)
    work = Path(started["work"])
    _run_tool(env, conversation_id, "write_file", {"path": "notes.txt", "content": "one\nTWO\n"})
    (work / "old.txt").unlink()
    proof = env.client.get(f"/api/safecopy/proof?conversation_id={conversation_id}").get_json()["proof"]
    kinds = {change["path"]: change["kind"] for change in proof["changes"]}
    assert kinds == {"notes.txt": "modified", "old.txt": "deleted"}
    assert proof["conflicts"] == []
    assert "+TWO" in proof["diff"]
    assert proof["revert"]["verified"] is True
    applied = env.client.post(
        "/api/safecopy/apply",
        json={"conversation_id": conversation_id, "fingerprint": proof["fingerprint"]},
    )
    assert applied.status_code == 200, applied.get_json()
    assert applied.get_json()["applied"]["written"] == ["notes.txt"]
    assert (project / "notes.txt").read_text() == "one\nTWO\n"
    assert (project / "old.txt").read_text() == "old\n"
    message = env.svc.db.list_messages(conversation_id)[-1]
    assert message["role"] == "assistant" and message["meta"]["kind"] == "safe_copy"
    assert "Safe copy applied: 1 file written" in message["content"]
    assert "Undo this in a terminal:" in message["content"]
    assert proof["revert"]["command"] in message["content"]
    assert (
        "These files were deleted in the copy. Kiwibro never deletes anything: "
        "run these yourself if you want them gone:"
    ) in message["content"]
    assert "rm -- " in message["content"] and "old.txt" in message["content"]
    status = env.client.get(f"/api/safecopy/status?conversation_id={conversation_id}").get_json()
    assert status["applied"] is True and status["can_undo"] is True


def test_undo_restores_the_old_bytes(env: Env, project: Path):
    conversation_id = _chat(env)
    _start(env, conversation_id, project)
    _run_tool(env, conversation_id, "write_file", {"path": "notes.txt", "content": "changed\n"})
    _run_tool(env, conversation_id, "write_file", {"path": "new.txt", "content": "fresh\n"})
    proof = env.client.get(f"/api/safecopy/proof?conversation_id={conversation_id}").get_json()["proof"]
    env.client.post(
        "/api/safecopy/apply",
        json={"conversation_id": conversation_id, "fingerprint": proof["fingerprint"]},
    )
    assert (project / "notes.txt").read_text() == "changed\n"
    assert (project / "new.txt").read_text() == "fresh\n"
    response = env.client.post("/api/safecopy/undo", json={"conversation_id": conversation_id})
    assert response.status_code == 200, response.get_json()
    assert response.get_json()["undo"]["restored"] == ["notes.txt"]
    assert (project / "notes.txt").read_text() == "one\ntwo\n"
    assert (project / "new.txt").read_text() == "fresh\n"
    message = env.svc.db.list_messages(conversation_id)[-1]
    assert "Safe copy undone: 1 file put back" in message["content"]
    assert "rm -- " in message["content"] and "new.txt" in message["content"]


def test_a_stale_fingerprint_writes_nothing(env: Env, project: Path):
    conversation_id = _chat(env)
    _start(env, conversation_id, project)
    _run_tool(env, conversation_id, "write_file", {"path": "notes.txt", "content": "changed\n"})
    proof = env.client.get(f"/api/safecopy/proof?conversation_id={conversation_id}").get_json()["proof"]
    _run_tool(env, conversation_id, "write_file", {"path": "notes.txt", "content": "changed again\n"})
    response = env.client.post(
        "/api/safecopy/apply",
        json={"conversation_id": conversation_id, "fingerprint": proof["fingerprint"]},
    )
    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "stale"
    assert (project / "notes.txt").read_text() == "one\ntwo\n"


def test_a_real_edit_since_the_copy_blocks_apply(env: Env, project: Path):
    conversation_id = _chat(env)
    _start(env, conversation_id, project)
    _run_tool(env, conversation_id, "write_file", {"path": "notes.txt", "content": "from the agent\n"})
    (project / "notes.txt").write_text("the user wrote this\n")
    proof = env.client.get(f"/api/safecopy/proof?conversation_id={conversation_id}").get_json()["proof"]
    assert proof["conflicts"] == ["notes.txt"]
    response = env.client.post(
        "/api/safecopy/apply",
        json={"conversation_id": conversation_id, "fingerprint": proof["fingerprint"]},
    )
    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "conflict"
    assert (project / "notes.txt").read_text() == "the user wrote this\n"


def test_the_routes_need_an_open_session_and_a_conversation(env: Env):
    conversation_id = _chat(env)
    missing = env.client.get("/api/safecopy/proof?conversation_id=c_missing")
    assert missing.status_code == 404
    assert missing.get_json()["error"]["code"] == "not_found"
    no_session = env.client.get(f"/api/safecopy/proof?conversation_id={conversation_id}")
    assert no_session.status_code == 404
    assert no_session.get_json()["error"]["message"] == "No safe copy is open in this chat."
    assert env.client.post("/api/safecopy/undo", json={"conversation_id": conversation_id}).status_code == 404
    assert env.client.post("/api/safecopy/start", json={"conversation_id": conversation_id}).status_code == 400
