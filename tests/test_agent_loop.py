"""Agent loop control for small models: failing-call coaching and the tool pause."""

from __future__ import annotations

import json

from test_backend import Env, chat_events

from kiwibro.agent import FAILURE_COACH_AT, FAILURE_PAUSE_AT
from kiwibro.llm import ChatResult


def _call(path: str, index: int) -> ChatResult:
    call = {"id": f"call_{index}", "type": "function",
            "function": {"name": "list_files", "arguments": json.dumps({"path": path})}}
    return ChatResult(tool_calls=[call], usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25})


def _tool_messages(env: Env) -> list[dict]:
    conversation_id = env.svc.db.list_conversations()[0]["id"]
    return [m for m in env.svc.db.list_messages(conversation_id) if m["role"] == "tool"]


def test_failing_calls_get_coached_then_paused(env: Env) -> None:
    guesses = ["/tmp", "/Desktop", "/", "/etc", "/usr", "/opt", "/srv", "/var"]
    env.fake.stream_script = [_call(path, i) for i, path in enumerate(guesses)] + [
        ChatResult(content="I could not find those folders; the workspace is empty."),
    ]
    events, _raw = chat_events(env, "List the files in your workspace.")
    tool_messages = _tool_messages(env)
    assert all("outside the workspace; use a path relative to it" in m["content"] for m in tool_messages[:FAILURE_PAUSE_AT])
    assert "[Kiwibro: the last 3 tool calls failed" in tool_messages[FAILURE_COACH_AT - 1]["content"]
    paused = tool_messages[FAILURE_PAUSE_AT:]
    assert paused and all(m["content"].startswith("Tool use is paused") for m in paused)
    assert events[-1][0] == "done"


def test_a_success_resets_the_failure_streak(env: Env) -> None:
    env.fake.stream_script = [_call("/tmp", 0), _call("/etc", 1), _call(".", 2), _call("/usr", 3),
                              ChatResult(content="Done.")]
    chat_events(env, "List the files in your workspace.")
    contents = [m["content"] for m in _tool_messages(env)]
    assert not any("[Kiwibro: the last" in content for content in contents)  # never 3 failures in a row
