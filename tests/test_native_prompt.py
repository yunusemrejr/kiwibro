"""Prompt budgets and tool exposure for the native GGUF path."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kiwibro import plan, tools
from kiwibro.agent import Turn
from kiwibro.context import estimate_messages
from kiwibro.llm import ChatResult
from kiwibro.mind import KIWIMIND
from kiwibro.mind.native_generic import NativeChatClient
from test_backend import Env, SSEStream, chat_events, env, tool_call_result  # noqa: F401

INFO = {"id": KIWIMIND, "engine": "native-gguf", "tools": True, "context": 8192,
        "reasoning": {"allowed_options": ["off", "on"], "default": "off"}}
READ_REQUEST = "Read release.txt with the file-reading tool. Reply with the exact release verification marker in that file."


def tool_names(catalog):
    return [item["function"]["name"] for item in catalog or []]


def test_native_read_fetches_evidence_before_one_generation_and_preserves_memory(
        env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    (env.workspace / "release.txt").write_text("The next release verification marker is RIVER-47.\n")
    memory = env.svc.memory.remember("User likes release notes to name the marker.", "preference")
    env.svc.db.update_memory(memory["id"], pinned=True)
    monkeypatch.setattr(env.svc.llm, "resolve_model", lambda _requested=None: KIWIMIND)
    monkeypatch.setattr(env.svc.llm, "model_info", lambda _model: INFO)
    env.fake.stream_script = [ChatResult(content="RIVER-47", finish_reason="stop")]

    events, _raw = chat_events(env, READ_REQUEST)
    assert events[0][0] == "meta"
    assert [detail["name"] for kind, detail in events if kind == "tool_result" and detail["ok"]] == ["read_file"]
    assert any(kind == "delta" and "RIVER-47" in detail["text"] for kind, detail in events)
    assert len(env.fake.stream_calls) == 1
    agent = env.svc.agent
    first, = env.fake.stream_calls
    assert tool_names(first["tools"]) == ["read_file"]
    assert "User likes release notes" in first["messages"][1]["content"]
    assert first["messages"][-2]["tool_calls"][0]["function"]["name"] == "read_file"
    observation = first["messages"][-1]
    assert observation["role"] == "tool" and "RIVER-47" in observation["content"]
    assert "every step that needs a tool is done" in observation["content"]
    assert "Kiwibro checks tool permissions" in first["messages"][0]["content"]
    assert first["extra"]["chat_template_kwargs"]["enable_thinking"] is False
    assert first["extra"]["reasoning_effort"] == "none"
    # Use the same baseline budget as before the read was moved ahead of generation.
    assert estimate_messages(first["messages"][:-2]) + agent._tool_tokens(KIWIMIND, first["tools"], INFO) < 600


def test_native_read_error_reopens_the_normal_catalog(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env.svc.llm, "resolve_model", lambda _requested=None: KIWIMIND)
    monkeypatch.setattr(env.svc.llm, "model_info", lambda _model: INFO)
    env.fake.stream_script = [
        tool_call_result("list_files", {"path": "."}),
        ChatResult(content="release.txt is missing from the workspace."),
    ]
    events, _raw = chat_events(env, READ_REQUEST)
    results = [(event["name"], event["ok"]) for kind, event in events if kind == "tool_result"]
    assert results == [("read_file", False), ("list_files", True)]
    first = env.fake.stream_calls[0]
    assert "list_files" in tool_names(first["tools"])
    assert "path does not exist: release.txt" in first["messages"][-1]["content"]


def test_native_read_pagination_keeps_followup_tools_available(
        env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env.svc.llm, "resolve_model", lambda _requested=None: KIWIMIND)
    monkeypatch.setattr(env.svc.llm, "model_info", lambda _model: INFO)
    (env.workspace / "release.txt").write_text(
        "".join(f"Release checklist item {i}: verify the migration succeeded.\n" for i in range(400))
    )
    env.fake.stream_script = [ChatResult(content="The result is paged; the marker requires a further check.")]
    chat_events(env, READ_REQUEST)
    first = env.fake.stream_calls[0]
    assert "doc_read" in tool_names(first["tools"])
    assert "doc_read" in first["messages"][-1]["content"]


@pytest.mark.parametrize(("mode", "message", "path"), [
    ("minimal", READ_REQUEST, "release.txt"),
    ("auto", 'Read "release memo.txt".', "release memo.txt"),
    ("auto", "Read release.txt then read memo.txt.", "release.txt"),
])
def test_other_requests_keep_the_model_in_charge(
        env: Env, monkeypatch: pytest.MonkeyPatch, mode: str, message: str, path: str) -> None:
    monkeypatch.setattr(env.svc.llm, "resolve_model", lambda _requested=None: KIWIMIND)
    monkeypatch.setattr(env.svc.llm, "model_info", lambda _model: INFO)
    env.settings_patch({"tool_profile": mode, "max_tool_rounds": 2})
    (env.workspace / path).write_text("Marker RIVER-47.\n")
    # A quoted filename can contain a different, valid path token.
    (env.workspace / "memo.txt").write_text("Marker OTHER-29.\n")
    env.fake.stream_script = [
        tool_call_result("read_file", {"path": path}),
        ChatResult(content="RIVER-47"),
    ]
    chat_events(env, message)
    first = env.fake.stream_calls[0]
    assert all(message["role"] != "tool" for message in first["messages"])


@pytest.mark.parametrize("permission", ["ask", "deny"])
def test_native_read_respects_the_normal_approval_gate(
        env: Env, monkeypatch: pytest.MonkeyPatch, permission: str) -> None:
    monkeypatch.setattr(env.svc.llm, "resolve_model", lambda _requested=None: KIWIMIND)
    monkeypatch.setattr(env.svc.llm, "model_info", lambda _model: INFO)
    env.settings_patch({"autonomy": "supervised", "permissions": {"read_file": permission}, "max_tool_rounds": 1})
    (env.workspace / "release.txt").write_text("Marker RIVER-47.\n")
    env.fake.stream_script = [ChatResult(content="I need the contents to report the marker.")]
    response = env.client.post("/api/chat", json={"message": READ_REQUEST}, buffered=False)
    assert response.status_code == 200
    stream = SSEStream(response).start()
    if permission == "ask":
        approval = stream.wait_for("approval")
        assert approval["tool"] == "read_file" and not env.fake.stream_calls
        accepted = env.new_client().post(f"/api/approvals/{approval['approval_id']}", json={"decision": "allow"})
        assert accepted.status_code == 200
    stream.finish()
    if permission == "ask":
        assert [event["ok"] for kind, event in stream.events if kind == "tool_result"] == [True]
        assert "RIVER-47" in env.fake.stream_calls[0]["messages"][-1]["content"]
    else:
        assert not {"tool_call", "tool_result", "approval"} & set(stream.names())
        assert all("RIVER-47" not in (message.get("content") or "") for message in env.fake.stream_calls[0]["messages"])


def test_native_auto_scopes_only_clear_local_text_reads(env: Env) -> None:
    agent, settings = env.svc.agent, env.svc.settings_provider()
    for request in (
        "Read release.txt, run its commands, and tell me what they do.",
        "Read report.pdf.",
        "Read release.txt then rewrite it.",
    ):
        turn = Turn("c", request, KIWIMIND)
        turn.plan = plan.build(request, env.workspace)
        assert tool_names(agent._tool_list(KIWIMIND, settings, INFO, turn=turn)) == list(tools.select_tool_names("auto"))

    hello = Turn("c", "Hello!", KIWIMIND)
    assert agent._tool_list(KIWIMIND, settings, INFO, turn=hello) is None
    assert agent._tool_list(KIWIMIND, settings, INFO, turn=Turn("c", READ_REQUEST, KIWIMIND))


def test_profiles_permissions_and_native_schema_budget(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    agent, settings = env.svc.agent, env.svc.settings_provider()
    catalog = tools.tool_defs_for("minimal", settings=settings)
    actual = agent._tool_list(KIWIMIND, {**settings, "tool_profile": "minimal"}, INFO)
    assert actual == catalog
    assert agent._tool_tokens(KIWIMIND, actual, INFO) > agent._tool_tokens(KIWIMIND, actual)

    turn = Turn("c", READ_REQUEST, KIWIMIND)
    turn.plan = plan.build(turn.message, env.workspace)
    denied = {**settings, "autonomy": "supervised", "permissions": {"read_file": "deny"}}
    assert agent._tool_list(KIWIMIND, denied, INFO, turn=turn) is None
    turn.native_unscoped = True
    assert "write_file" in tool_names(agent._tool_list(KIWIMIND, settings, INFO, turn=turn))

    monkeypatch.setattr(env.svc.llm, "native_schema_tokens", lambda items: len(items) * 37)
    assert agent._tool_tokens(KIWIMIND, actual, INFO) == len(actual) * 37
    monkeypatch.setattr(env.svc.llm, "model_info", lambda _model: INFO)
    assert agent.warm_prefix(KIWIMIND) is None


def test_warm_schema_meter_counts_the_rendered_template_once(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = SimpleNamespace(api_key="schema-token-meter-test", loaded=lambda _name: SimpleNamespace(url="http://127.0.0.1:9"))
    native = NativeChatClient(lambda: engine)
    catalog = tools.tool_defs_for("minimal")
    post = Mock()
    monkeypatch.setattr("kiwibro.mind.native_generic.requests.post", post)
    assert native.schema_tokens(catalog) is None
    post.assert_not_called()
    native.direct_engine()

    def response(_url, *, json, **_):
        response = Mock()
        response.__enter__ = lambda _self: response
        response.__exit__ = lambda *_args: None
        if "messages" in json:
            response.json.return_value = {"prompt": "with schemas" if json["tools"] else "plain"}
        else:
            response.json.return_value = {"tokens": [0] * (137 if json["content"] == "with schemas" else 17)}
        return response

    post.side_effect = response
    assert native.schema_tokens(catalog) == 120 and native.schema_tokens(catalog) == 120
    assert post.call_count == 4
