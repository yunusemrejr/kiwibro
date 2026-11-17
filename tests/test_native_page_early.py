"""The native turn can inspect a newly saved page before asking the model to continue."""

from __future__ import annotations

import pytest

from kiwibro import safemode, tools
from kiwibro.llm import ChatResult
from kiwibro.mind import KIWIMIND
from kiwibro.sandbox import Workspace
import test_backend
import test_page_loop
from test_backend import Env, SSEStream, chat_events, tool_call_result
from test_page_loop import FIXED, PAGE, looks, shown, write
from test_pagecheck import CLEAN, script_error

env = test_backend.env
chrome = test_page_loop.chrome
ASK = "Save index.html. Check the saved page using look_at_page."


@pytest.fixture
def native_page(env: Env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(env.svc.llm, "resolve_model", lambda _requested=None: KIWIMIND)
    monkeypatch.setattr(env.svc.llm, "model_info", lambda _model: {
        "id": KIWIMIND, "engine": "native-gguf", "tools": True, "context": 8192,
    })

    def permissions(look: str = "auto") -> None:
        modes = dict.fromkeys(tools.tool_names(), "deny")
        modes.update(write_file="auto", look_at_page=look)
        env.settings_patch({"autonomy": "supervised", "tool_profile": "full",
                            "max_tool_rounds": 5, "permissions": modes})

    permissions()
    return permissions


@pytest.mark.parametrize("request_text", [ASK, "Save index.html and check the saved render."])
def test_native_explicit_render_is_checked_in_the_write_round(env: Env, native_page, chrome, request_text: str):
    chrome(CLEAN)
    env.fake.stream_script = [write("index.html", FIXED, "call_1")]
    events, _ = chat_events(env, request_text)
    outcomes = [(data["name"], data["ok"]) for event, data in events if event == "tool_result"]
    assert outcomes == [("write_file", True), ("look_at_page", True)]
    assert "No problems found" in looks(env)[0]["content"]
    assert (env.workspace / "screenshots" / "look-index.png").is_file()
    assert len(env.fake.stream_calls) == 1


def test_native_early_render_result_reaches_repair_round(env: Env, native_page, chrome):
    chrome(script_error(2), CLEAN)
    env.fake.stream_script = [write("index.html", PAGE, "call_1"), write("index.html", FIXED, "call_2")]
    events, _ = chat_events(env, ASK)
    checks = [data["name"] for event, data in events if event == "tool_result"]
    assert checks == ["write_file", "look_at_page", "write_file", "look_at_page"]
    assert len(env.fake.stream_calls) == 2
    assert any("spin is not defined" in message.get("content", "")
               for message in env.fake.stream_calls[1]["messages"] if message["role"] == "tool")


@pytest.mark.parametrize("look", ["ask", "deny"])
def test_native_automatic_render_respects_permissions(env: Env, native_page, chrome, look: str):
    native_page(look)
    chrome(CLEAN)
    env.fake.stream_script = [write("index.html", FIXED, "call_1"), ChatResult(content="Saved index.html.")]
    events, _ = chat_events(env, ASK)
    assert len(env.fake.stream_calls) == 2
    assert not looks(env)
    assert not any(event == "tool_call" and data["name"] == "look_at_page" for event, data in events)


def test_native_model_requested_render_can_still_be_approved(env: Env, native_page, chrome):
    native_page("ask")
    chrome(CLEAN)
    env.fake.stream_script = [
        write("index.html", FIXED, "call_1"),
        tool_call_result("look_at_page", {"path": "index.html"}, "call_2"),
        ChatResult(content="Saved and checked index.html."),
    ]
    stream = SSEStream(env.client.post("/api/chat", json={"message": ASK}, buffered=False)).start()
    approval = stream.wait_for("approval", timeout=4)
    assert approval["tool"] == "look_at_page"
    assert not looks(env)
    response = env.new_client().post(f"/api/approvals/{approval['approval_id']}", json={"decision": "allow"})
    assert response.status_code == 200
    assert any(event == "tool_result" and data["name"] == "look_at_page" and data["ok"]
               for event, data in stream.finish(timeout=4))


def test_native_early_render_preserves_requested_reply_and_pending_file(env: Env, native_page, chrome):
    chrome(CLEAN)
    env.fake.stream_script = [
        write("index.html", FIXED, "call_1"),
        write("notes.txt", "The page is interactive.", "call_2"),
        ChatResult(content="The keyboard shortcut is Shift+N."),
    ]
    events, _ = chat_events(env, ASK + " Save notes.txt with a sentence. Tell me the keyboard shortcut.")
    assert (env.workspace / "notes.txt").read_text() == "The page is interactive."
    assert "Shift+N" in shown(events)
    assert [data["name"] for event, data in events if event == "tool_result"] == [
        "write_file", "look_at_page", "write_file",
    ]


def test_native_render_finding_leaves_the_requested_answer_in_place(env: Env, native_page, chrome):
    chrome(script_error(2))
    env.fake.stream_script = [
        write("index.html", PAGE, "call_1"),
        ChatResult(content="The missing function is spin."),
    ]
    events, _ = chat_events(env, ASK + " Tell me the missing function.")
    assert "The missing function is spin." in shown(events)
    assert "Still wrong: a script error" in shown(events)
    assert len(looks(env)) == 1


def test_native_render_respects_an_explicit_skip(env: Env, native_page, chrome):
    chrome(CLEAN)
    env.fake.stream_script = [write("index.html", FIXED, "call_1"), ChatResult(content="Saved index.html.")]
    chat_events(env, ASK + " Do not render it yet.")
    assert not looks(env)


def test_native_implicit_render_keeps_late_check(env: Env, native_page, chrome):
    chrome(CLEAN)
    env.fake.stream_script = [write("index.html", FIXED, "call_1"), ChatResult(content="Saved index.html.")]
    chat_events(env, "Save index.html.")
    assert len(env.fake.stream_calls) == 2
    assert looks(env)


def test_native_render_instruction_in_sample_code_keeps_late_check(env: Env, native_page, chrome):
    chrome(CLEAN)
    env.fake.stream_script = [write("index.html", FIXED, "call_1"), ChatResult(content="Saved index.html.")]
    chat_events(env, "Save index.html. Include this copy:\n```text\nCheck the saved render.\n```")
    assert len(env.fake.stream_calls) == 2
    assert looks(env)


def test_native_multiple_pending_pages_defer_automatic_render(env: Env, native_page, chrome):
    chrome(CLEAN)
    batch = write("index.html", FIXED, "call_1")
    batch.tool_calls.extend(write("about.html", FIXED, "call_2").tool_calls)
    env.fake.stream_script = [batch, ChatResult(content="Saved index.html and about.html.")]
    chat_events(env, "Save index.html and about.html. Check the saved render.")
    assert len(env.fake.stream_calls) == 2
    assert not any(message.get("tool_name") == "look_at_page" for message in env.fake.stream_calls[1]["messages"])
    assert looks(env)


def test_native_render_honors_steering(env: Env, native_page, chrome, monkeypatch: pytest.MonkeyPatch):
    chrome(CLEAN)
    env.fake.stream_script = [write("index.html", FIXED, "call_1"), ChatResult(content="Saved index.html.")]
    conversation = env.svc.db.create_conversation()
    initial_stream = env.fake.stream_chat

    def steer_during_write(**kwargs):
        if not env.fake.stream_calls:
            response = env.new_client().post("/api/chat/steer", json={
                "conversation_id": conversation["id"], "message": "Skip the render check. Just save the page.",
            })
            assert response.status_code == 200
        return initial_stream(**kwargs)

    monkeypatch.setattr(env.svc.llm, "stream_chat", steer_during_write)
    events, _ = chat_events(env, ASK, conversation_id=conversation["id"])
    assert any(event == "steering" and data["text"].startswith("Skip the render") for event, data in events)
    assert (env.workspace / "index.html").is_file()
    assert not looks(env)


def test_native_automatic_render_checks_current_workspace(env: Env, native_page, chrome,
                                                          monkeypatch: pytest.MonkeyPatch, tmp_path):
    chrome(CLEAN)
    env.fake.stream_script = [write("index.html", FIXED, "call_1"), ChatResult(content="Saved index.html.")]
    workspace = env.svc.agent._workspace
    other_workspace = Workspace(tmp_path / "other")
    execute = tools.execute_tool
    switched = False

    def run(name, args, context):
        nonlocal switched
        result = execute(name, args, context)
        switched = name == "write_file" or switched
        return result

    monkeypatch.setattr(tools, "execute_tool", run)
    monkeypatch.setattr(env.svc.agent, "_workspace",
                        lambda conversation_id=None: other_workspace if switched else workspace(conversation_id))
    chat_events(env, ASK)
    assert (env.workspace / "index.html").is_file()
    assert not looks(env)


def test_native_automatic_render_obeys_safe_copy_gate(env: Env, native_page, chrome, monkeypatch: pytest.MonkeyPatch):
    chrome(CLEAN)
    env.fake.stream_script = [write("index.html", FIXED, "call_1"), ChatResult(content="Saved index.html.")]
    gate = safemode.gate
    monkeypatch.setattr(safemode, "gate", lambda db, cid, name: (
        safemode.Gate(active=True, refusal="Rendering is restricted here.")
        if name == "look_at_page" else gate(db, cid, name)
    ))
    chat_events(env, ASK)
    assert (env.workspace / "index.html").is_file()
    assert not looks(env)
