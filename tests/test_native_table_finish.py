"""Native table workflows finish from successful evidence after blocked repeats."""

from __future__ import annotations

import pytest

from kiwibro import tools
from kiwibro.llm import ChatResult
from kiwibro.mind import KIWIMIND
from test_backend import Env, chat_events, env, tool_call_result  # noqa: F401


TABLE = "shipments/priority-status.tsv"
ARGS = {"path": TABLE, "where": 'status == "failed"', "aggregate": "count"}
REQUEST = f"Count the failed status rows out of the total in {TABLE}. Give both counts and the source file."
ANSWER = f"Two of seven rows failed in {TABLE}."


def setup_table(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env.svc.llm, "resolve_model", lambda _requested=None: KIWIMIND)
    monkeypatch.setattr(env.svc.llm, "model_info", lambda _model: {
        "id": KIWIMIND, "engine": "native-gguf", "tools": True, "context": 8192,
    })
    permissions = {name: "deny" for name in tools.tool_names()}
    permissions.update(analyze_table="auto", read_file="auto", write_file="auto")
    env.settings_patch({"autonomy": "supervised", "tool_profile": "full", "max_tool_rounds": 9,
                        "permissions": permissions})
    file = env.workspace / TABLE
    file.parent.mkdir()
    file.write_text("job\tstatus\nA\tfailed\nB\tready\nC\tready\nD\tqueued\nE\tqueued\nF\tfailed\nG\tready\n")


def analysis_rounds(args=ARGS):
    return [tool_call_result("analyze_table", args, call_id=f"table_{index}") for index in (1, 2, 3)]


def test_native_exact_successful_table_repeats_trigger_one_text_only_finish(
        env: Env, monkeypatch: pytest.MonkeyPatch,
):
    setup_table(env, monkeypatch)
    env.fake.stream_script = [*analysis_rounds(), ChatResult(content=ANSWER)]

    events, _ = chat_events(env, REQUEST)
    results = [(body["name"], body["ok"]) for event, body in events if event == "tool_result"]
    assert results == [("analyze_table", True), ("analyze_table", True), ("analyze_table", False)]
    assert len(env.fake.stream_calls) == 4
    finish = env.fake.stream_calls[-1]
    assert finish["tools"] is None
    assert any(message["role"] == "tool" and "2 rows matched" in message["content"]
               for message in finish["messages"])
    assert events[-1][0] == "done" and any(event == "delta" and ANSWER in body["text"] for event, body in events)


def test_native_table_repeat_keeps_a_pending_executable_step(env: Env, monkeypatch: pytest.MonkeyPatch):
    setup_table(env, monkeypatch)
    env.fake.stream_script = [
        *analysis_rounds(),
        tool_call_result("write_file", {"path": "result.txt", "content": ANSWER}, call_id="save"),
        ChatResult(content="Saved the requested count and source in result.txt."),
    ]

    events, _ = chat_events(env, REQUEST + " Save the result to result.txt.")
    assert env.fake.stream_calls[3]["tools"] is not None
    assert (env.workspace / "result.txt").read_text() == ANSWER
    assert any(event == "tool_result" and body["name"] == "write_file" and body["ok"] for event, body in events)


def test_native_table_repeat_keeps_a_mixed_rounds_nonduplicate_action(
        env: Env, monkeypatch: pytest.MonkeyPatch,
):
    setup_table(env, monkeypatch)
    rounds = analysis_rounds()
    rounds[-1].tool_calls.extend(tool_call_result("read_file", {"path": TABLE}, call_id="read").tool_calls)
    env.fake.stream_script = [*rounds, ChatResult(content=ANSWER)]

    events, _ = chat_events(env, REQUEST)
    assert env.fake.stream_calls[-1]["tools"] is not None
    assert any(event == "tool_result" and body["name"] == "read_file" and body["ok"] for event, body in events)


def test_native_failures_alone_do_not_trigger_successful_table_finish(env: Env, monkeypatch: pytest.MonkeyPatch):
    setup_table(env, monkeypatch)
    env.fake.stream_script = [
        *analysis_rounds({**ARGS, "path": "shipments/missing.tsv"}),
        tool_call_result("analyze_table", ARGS, call_id="different_table"),
        ChatResult(content=ANSWER),
    ]

    events, _ = chat_events(env, REQUEST)
    assert env.fake.stream_calls[3]["tools"] is not None
    results = [body["ok"] for event, body in events if event == "tool_result" and body["name"] == "analyze_table"]
    assert results == [False, False, False, True]
