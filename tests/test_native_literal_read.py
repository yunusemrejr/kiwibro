"""Exact native Auto reads can return a checked single-field value without generation."""

from unittest.mock import Mock

import pytest

from kiwibro import tools
from kiwibro.llm import ChatResult
from kiwibro.mind import KIWIMIND
import test_backend
from test_backend import Env, SSEStream, chat_events
from test_native_prompt import INFO

env = test_backend.env
ASK = "Read release.txt with the file reading tool. Reply with only its version."
VALUE = "1.2.3-rc.4"


@pytest.fixture
def native_read(env: Env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(env.svc.llm, "resolve_model", lambda _requested=None: KIWIMIND)
    monkeypatch.setattr(env.svc.llm, "model_info", lambda _model: INFO)


@pytest.mark.parametrize(("request_text", "source", "value"), [
    (ASK, f"Version: {VALUE}\n", VALUE),
    ("Read release.txt. REPLY WITH ONLY THE RELEASE STATE.", "release   State:\tReady to Ship\n", "Ready to Ship"),
    ("Read release.txt. Reply with only its build number.", "Build number: 0", "0"),
])
def test_native_literal_read_keeps_tool_evidence_and_skips_inference(
        env: Env, native_read, monkeypatch: pytest.MonkeyPatch, request_text: str, source: str, value: str,
):
    (env.workspace / "release.txt").write_text(source, encoding="utf-8")
    readiness = Mock()
    monkeypatch.setattr(env.svc.agent, "readiness", readiness)
    extract = Mock()
    monkeypatch.setattr(env.svc.agent, "_start_memory_extraction", extract)
    tokenize = Mock(side_effect=AssertionError("A direct field lookup needs no model template"))
    monkeypatch.setattr(env.svc.llm, "native_schema_tokens", tokenize)
    conversation = env.svc.db.create_conversation()
    for _ in range(8):
        env.svc.db.add_message(conversation["id"], "user", "Previous long unrelated text. " * 500)
        env.svc.db.add_message(conversation["id"], "assistant", "Previous reply.")
    compact = Mock(side_effect=AssertionError("A direct field lookup needs no history compaction"))
    monkeypatch.setattr(env.svc.agent.context, "_compact_segment", compact)

    events, _ = chat_events(env, request_text, conversation_id=conversation["id"])
    assert events[0][0] == "meta" and events[-1][0] == "done"
    assert [data["text"] for kind, data in events if kind == "delta"] == [value]
    assert [(data["name"], data["ok"]) for kind, data in events if kind == "tool_result"] == [("read_file", True)]
    assert all(step["done"] for step in events[-1][1]["plan"])
    stored = env.svc.db.list_messages(conversation["id"])
    assert stored[-2]["role"] == "tool" and stored[-2]["content"].startswith(source.rstrip("\n"))
    assert "Kiwibro plan:" in stored[-2]["content"]
    assert stored[-1]["content"] == value and stored[-1]["meta"]["checked_by_kiwibro"]
    assert not env.fake.stream_calls
    readiness.before_turn.assert_not_called()
    compact.assert_not_called()
    tokenize.assert_not_called()
    extract.assert_not_called()


@pytest.mark.parametrize(("request_text", "path", "source"), [
    (ASK, "release.txt", "Version: 12\nVersion: 13\n"),
    (ASK, "release.txt", "Release version: 12\n"),
    (ASK, "release.txt", "Version: **12**\n"),
    (ASK, "release.txt", "Version: [Stable](https://example.test)\n"),
    (ASK, "release.txt", "Version: https://example.test\n"),
    (ASK.replace("release.txt", "release.md"), "release.md", "Version: 12\n"),
    (ASK.replace("only its", "the"), "release.txt", "Version: 12\n"),
    (ASK + " Then read the notes file notes.txt.", "release.txt", "Version: 12\n"),
])
def test_native_other_reads_keep_generation(
        env: Env, native_read, request_text: str, path: str, source: str,
):
    (env.workspace / path).write_text(source, encoding="utf-8")
    env.fake.stream_script = [ChatResult(content="This reply comes from the model.")]
    events, _ = chat_events(env, request_text)
    assert any(kind == "delta" and data["text"] == "This reply comes from the model." for kind, data in events)
    assert env.fake.stream_calls


def test_native_literal_read_does_not_bypass_approval(env: Env, native_read):
    (env.workspace / "release.txt").write_text(f"Version: {VALUE}\n", encoding="utf-8")
    env.settings_patch({"autonomy": "supervised", "permissions": {"read_file": "ask"}})
    stream = SSEStream(env.client.post("/api/chat", json={"message": ASK}, buffered=False)).start()
    approval = stream.wait_for("approval")
    assert approval["tool"] == "read_file" and not env.fake.stream_calls
    response = env.new_client().post(f"/api/approvals/{approval['approval_id']}", json={"decision": "allow"})
    assert response.status_code == 200
    assert [body["text"] for event, body in stream.finish() if event == "delta"] == [VALUE]
    assert not env.fake.stream_calls


def test_native_literal_read_keeps_guarded_values_out_of_direct_replies(env: Env, native_read):
    token = "sk-proj-" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    (env.workspace / "release.txt").write_text(f"Version: {token}\n", encoding="utf-8")
    env.fake.stream_script = [ChatResult(content="The file value was redacted.")]
    events, raw = chat_events(env, ASK)
    assert token not in raw
    assert "The file value was redacted." in raw and env.fake.stream_calls
    observed = [message["content"] for message in env.fake.stream_calls[0]["messages"] if message["role"] == "tool"]
    assert "[SECRET_" in observed[0] and token not in observed[0]


def test_native_literal_read_keeps_user_steering_in_charge(env: Env, native_read, monkeypatch: pytest.MonkeyPatch):
    (env.workspace / "release.txt").write_text(f"Version: {VALUE}\n", encoding="utf-8")
    conversation = env.svc.db.create_conversation()
    run = tools.execute_tool

    def read_and_steer(name, args, ctx):
        result = run(name, args, ctx)
        if name == "read_file":
            response = env.new_client().post("/api/chat/steer", json={
                "conversation_id": conversation["id"], "message": "Please describe the source in a sentence instead.",
            })
            assert response.status_code == 200
        return result

    monkeypatch.setattr(tools, "execute_tool", read_and_steer)
    env.fake.stream_script = [ChatResult(content=f"The file names {VALUE} as its version.")]
    events, _ = chat_events(env, ASK, conversation_id=conversation["id"])
    assert [data["text"] for kind, data in events if kind == "delta"] == [f"The file names {VALUE} as its version."]
    assert env.fake.stream_calls
    assert any(kind == "steering" for kind, _ in events)
