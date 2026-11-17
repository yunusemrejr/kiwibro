"""Native table workflows finish from successful evidence after blocked repeats."""

from __future__ import annotations

import pytest

from kiwibro import tools
from kiwibro.llm import ChatResult
from kiwibro.mind import KIWIMIND
import test_backend
from test_backend import Env, chat_events, tool_call_result


env = test_backend.env
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
    permissions.update(analyze_table="auto", calculate="auto", list_files="auto", read_file="auto", write_file="auto")
    env.settings_patch({"autonomy": "supervised", "tool_profile": "full", "max_tool_rounds": 9,
                        "permissions": permissions})
    file = env.workspace / TABLE
    file.parent.mkdir()
    file.write_text("job\tstatus\nA\tfailed\nB\tready\nC\tready\nD\tqueued\nE\tqueued\nF\tfailed\nG\tready\n")


def analysis_rounds(args=ARGS, *, count=3):
    return [tool_call_result("analyze_table", args, call_id=f"table_{index}") for index in range(count)]


@pytest.mark.parametrize("missing", [True, False])
def test_native_analysis_lists_nearest_workspace_folder_only_for_a_missing_path(
        env: Env, monkeypatch: pytest.MonkeyPatch, missing: bool,
):
    setup_table(env, monkeypatch)
    invalid = ({"path": "shipments/guessed/status.tsv"} if missing else
               {"path": TABLE, "where": "imaginary_column == 0"})
    env.fake.stream_script = [
        tool_call_result("analyze_table", invalid, call_id="first"),
        tool_call_result("analyze_table", ARGS, call_id="actual"),
        ChatResult(content=ANSWER),
    ]

    events, _ = chat_events(env, REQUEST)
    results = [(body["name"], body["ok"]) for event, body in events if event == "tool_result"]
    expected = [("analyze_table", False), ("analyze_table", True)]
    if missing:
        expected.insert(1, ("list_files", True))
        listing = next(body for event, body in events if event == "tool_call" and body["name"] == "list_files")
        assert listing["args"] == {"path": "shipments"}
        assert f"file\t{TABLE}\t" in env.fake.stream_calls[1]["messages"][-1]["content"]
    assert results == expected


def test_native_exact_successful_table_repeats_trigger_one_text_only_finish(
        env: Env, monkeypatch: pytest.MonkeyPatch,
):
    setup_table(env, monkeypatch)
    env.fake.stream_script = [*analysis_rounds(count=2), ChatResult(content=ANSWER)]

    events, _ = chat_events(env, REQUEST)
    results = [(body["name"], body["ok"]) for event, body in events if event == "tool_result"]
    assert results == [("analyze_table", True), ("analyze_table", False)]
    assert len(env.fake.stream_calls) == 3
    finish = env.fake.stream_calls[-1]
    assert finish["tools"] is None
    assert any(message["role"] == "tool" and "2 rows matched" in message["content"]
               for message in finish["messages"])
    assert events[-1][0] == "done" and any(event == "delta" and ANSWER in body["text"] for event, body in events)


def test_native_table_repeat_keeps_a_pending_executable_step(env: Env, monkeypatch: pytest.MonkeyPatch):
    setup_table(env, monkeypatch)
    env.fake.stream_script = [
        *analysis_rounds(count=2),
        tool_call_result("write_file", {"path": "result.txt", "content": ANSWER}, call_id="save"),
        ChatResult(content="Saved the requested count and source in result.txt."),
    ]

    events, _ = chat_events(env, REQUEST + " Save the result to result.txt.")
    assert env.fake.stream_calls[2]["tools"] is not None
    assert (env.workspace / "result.txt").read_text() == ANSWER
    assert any(event == "tool_result" and body["name"] == "write_file" and body["ok"] for event, body in events)


def test_native_table_repeat_keeps_a_mixed_rounds_nonduplicate_action(
        env: Env, monkeypatch: pytest.MonkeyPatch,
):
    setup_table(env, monkeypatch)
    rounds = analysis_rounds(count=2)
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


@pytest.mark.parametrize("changed", ["workspace_write", "external_write", "unrelated_write"])
def test_native_exact_table_analysis_can_rerun_after_mutation(
        env: Env, monkeypatch: pytest.MonkeyPatch, changed: str,
):
    setup_table(env, monkeypatch)
    new_rows = "job\tstatus\nA\tfailed\nB\tfailed\nC\tfailed\n"
    first, second, third = analysis_rounds()
    mutation = []
    if changed == "external_write":
        stream = env.svc.llm.stream_chat

        def update_table(**kwargs):
            if len(env.fake.stream_calls) == 1:
                (env.workspace / TABLE).write_text(new_rows)
            return stream(**kwargs)

        monkeypatch.setattr(env.svc.llm, "stream_chat", update_table)
    else:
        changed_path = TABLE if changed == "workspace_write" else "other.txt"
        mutation = [tool_call_result("write_file", {"path": changed_path, "content": new_rows}, call_id="save")]
    answer = ANSWER if changed == "unrelated_write" else f"All three rows failed in {TABLE}."
    env.fake.stream_script = [first, *mutation, second, third, ChatResult(content=answer)]

    events, _ = chat_events(env, REQUEST)
    results = [body["ok"] for event, body in events if event == "tool_result" and body["name"] == "analyze_table"]
    assert results == [True, True, False]
    finish = env.fake.stream_calls[-1]
    assert finish["tools"] is None
    if changed != "unrelated_write":
        assert any(message["role"] == "tool" and "3 rows matched" in message["content"]
                   for message in finish["messages"])
    assert any(event == "delta" and answer in body["text"] for event, body in events)


def test_native_table_repeat_runs_when_the_user_requested_it(env: Env, monkeypatch: pytest.MonkeyPatch):
    setup_table(env, monkeypatch)
    env.fake.stream_script = [*analysis_rounds(count=2), ChatResult(content=ANSWER)]

    events, _ = chat_events(env, f"Analyze the failed rows in {TABLE} twice and give me the count.")
    assert [body["ok"] for event, body in events if event == "tool_result" and body["name"] == "analyze_table"] == [
        True, True,
    ]
    assert env.fake.stream_calls[-1]["tools"] is not None


@pytest.mark.parametrize("table_path", [TABLE, "shipments/missing.tsv", None])
def test_native_exact_calculation_repeat_finishes_only_after_successful_table_analysis(
        env: Env, monkeypatch: pytest.MonkeyPatch, table_path: str | None,
):
    setup_table(env, monkeypatch)
    profile = [tool_call_result("analyze_table", {"path": table_path}, call_id="profile")] if table_path else []
    calculations = [
        tool_call_result("calculate", {"expression": expression}, call_id=f"calc_{index}")
        for index, expression in enumerate(("2/7", "28.57% of 7", "28.57% of 7", "28.57% of 7"))
    ]
    env.fake.stream_script = [*profile, *calculations, ChatResult(content=ANSWER)]

    events, _ = chat_events(env, REQUEST)
    results = [body["ok"] for event, body in events if event == "tool_result" and body["name"] == "calculate"]
    assert results == [True, True, True, False]
    finish = env.fake.stream_calls[len(profile) + len(calculations)]
    assert (finish["tools"] is None) == (table_path == TABLE)
    if table_path == TABLE:
        observed = [row["content"] for row in finish["messages"] if row["role"] == "tool"]
        assert any("Rows: 7" in row and "- status (text)" in row and "failed (2)" in row for row in observed)
        assert all("2 rows matched" not in row for row in observed)
        assert len(env.fake.stream_calls) == len(profile) + len(calculations) + 1
        assert any(event == "delta" and ANSWER in body["text"] for event, body in events)
