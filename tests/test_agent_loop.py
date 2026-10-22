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


# --- degeneration guard ---------------------------------------------------------

from kiwibro import inference  # noqa: E402
from test_backend import tool_call_result  # noqa: E402


def test_degeneration_detector() -> None:
    assert inference.is_degenerate("I will create the files now.\n\n" + "...\n\n\n" * 60)
    assert inference.is_degenerate("Let me use the workspace path. " * 20)
    assert inference.is_degenerate("\n".join(["The workspace is given."] * 9) + "\n" + "x" * 300)
    prose = ("Kiwis are flightless birds native to New Zealand. They are nocturnal and have long beaks "
             "with nostrils at the tip, which they use to find food in the soil. ") * 3
    code = "\n".join(f"def handler_{i}(event):\n    return process(event, retries={i})" for i in range(12))
    assert not inference.is_degenerate(prose)
    assert not inference.is_degenerate(code)
    assert inference.trim_degenerate("I will create a.txt.\n\n" + "...\n\n" * 50) == "I will create a.txt."


def test_degenerate_reply_is_cut_and_retried_once(env: Env) -> None:
    junk = "I need to create the file. Let me use the workspace path.\n\n" + "...\n\n\n" * 200
    env.fake.stream_script = [
        ChatResult(content=junk),
        tool_call_result("write_file", {"path": "a.txt", "content": "kiwi\n"}),
        ChatResult(content="Created a.txt."),
    ]
    events, _raw = chat_events(env, "Create a.txt containing kiwi.")
    calls = env.fake.stream_calls
    assert len(calls) == 3
    assert calls[1]["messages"][-1]["content"].startswith("Your last reply started repeating itself")
    assert (env.workspace / "a.txt").read_text() == "kiwi\n"
    rewrites = [payload for event, payload in events if event == "rewrite"]
    assert rewrites and "..." not in rewrites[0]["text"]
    stored = [m["content"] for m in env.svc.db.list_messages(env.svc.db.list_conversations()[0]["id"])]
    assert not any("...\n\n\n..." in str(content) for content in stored)  # junk never stored


def test_degenerating_twice_ends_the_round_honestly(env: Env) -> None:
    junk = "...\n\n\n" * 200
    env.fake.stream_script = [ChatResult(content=junk), ChatResult(content=junk)]
    events, _raw = chat_events(env, "Do the thing.")
    assert len(env.fake.stream_calls) == 2
    assert events[-1][0] == "done"
    text = "".join(payload.get("text", "") for event, payload in events if event == "delta")
    assert "I got stuck repeating myself" in text
