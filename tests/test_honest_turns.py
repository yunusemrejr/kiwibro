"""What Kiwibro puts right in code, whatever the model says: a damaged tool call never runs, a written file that
does not parse is named, citation markers need a source, and a reply written without the internet says so."""

from __future__ import annotations

from typing import Any

import pytest
from test_backend import Env, chat_events, tool_call_result

from kiwibro import connectivity, prompts
from kiwibro.agent import Turn
from kiwibro.llm import ChatResult

BROKEN_PYTHON = "def reverse_words(text):\n    words = text.split()\n    return ' '.join(reversed(words)}\n"
FIXED_PYTHON = "def reverse_words(text):\n    words = text.split()\n    return ' '.join(reversed(words))\n"
USAGE = {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}


def _messages(env: Env, role: str | None = None) -> list[dict[str, Any]]:
    stored = env.svc.db.list_messages(env.svc.db.list_conversations()[0]["id"])
    return [m for m in stored if role is None or m["role"] == role]


def _final_text(events: list[tuple[str, dict[str, Any]]]) -> str:
    """The reply as the screen ends up showing it: deltas appended, a rewrite replacing what came before."""
    shown = ""
    for name, data in events:
        if name == "delta":
            shown += data.get("text", "")
        elif name == "rewrite":
            shown = data.get("text", "")
    return shown


# --- a damaged tool call is refused, never run -------------------------------------------------


def test_a_damaged_call_never_writes_and_the_model_is_told_to_make_it_again(env: Env) -> None:
    cut_off = '<tool_call>\n{"name": "write_file", "arguments": {"path": "a.py", "content": "def f(x):\\n    return x'
    env.fake.stream_script = [
        ChatResult(content=cut_off, usage=USAGE),
        tool_call_result("write_file", {"path": "a.py", "content": "def f(x):\n    return x\n"}, "call_2"),
        ChatResult(content="Saved a.py."),
    ]
    chat_events(env, "Write a function f that returns its argument and save it as a.py")
    results = [m["content"] for m in _messages(env, "tool")]
    assert results[0].startswith("Error: this call was damaged (it stops inside a string), so it did not run")
    assert "Make the call again, with the arguments as valid JSON." in results[0]
    assert (env.workspace / "a.py").read_text() == "def f(x):\n    return x\n"  # only the repeated, whole call wrote


def test_a_call_with_closers_in_its_string_writes_the_clean_file(env: Env) -> None:
    leaked = '<tool_call>\n{"name": "write_file", "arguments": {"path": "a.py", "content": "x = 1\\n}}\n</tool_call>'
    env.fake.stream_script = [ChatResult(content=leaked, usage=USAGE), ChatResult(content="Saved a.py.")]
    chat_events(env, "Save x = 1 as a.py")
    assert (env.workspace / "a.py").read_text() == "x = 1\n"


# --- a written file that does not parse -------------------------------------------------------


def test_a_broken_file_is_flagged_in_the_tool_result_and_named_under_the_reply(env: Env) -> None:
    env.fake.stream_script = [
        tool_call_result("write_file", {"path": "words.py", "content": BROKEN_PYTHON}),
        ChatResult(content="Done, words.py is ready."),
    ]
    events, _raw = chat_events(env, "Write a Python function that reverses the words in a sentence and save it as words.py")
    result = _messages(env, "tool")[0]["content"]
    assert "Syntax error in words.py, line 3: " in result
    assert result.rstrip().endswith("Fix the file before you say it is done.")
    line = next(line for line in _final_text(events).splitlines() if line.startswith("Checked by Kiwibro"))
    assert line.startswith("Checked by Kiwibro: `words.py` has a syntax error on line 3 (")
    assert _messages(env, "assistant")[-1]["content"] == line
    assert _messages(env, "assistant")[-1]["meta"]["checked_by_kiwibro"] is True


def test_a_file_mended_after_the_check_is_not_reported(env: Env) -> None:
    env.fake.stream_script = [
        tool_call_result("write_file", {"path": "words.py", "content": BROKEN_PYTHON}, "call_1"),
        tool_call_result("write_file", {"path": "words.py", "content": FIXED_PYTHON}, "call_2"),
        ChatResult(content="Done, words.py is ready."),
    ]
    events, _raw = chat_events(env, "Write a Python function that reverses the words in a sentence and save it as words.py")
    results = [m["content"] for m in _messages(env, "tool")]
    assert "Syntax error in words.py" in results[0] and "Syntax error" not in results[1]
    assert "Checked by Kiwibro" not in _final_text(events)
    assert not any(m["meta"].get("checked_by_kiwibro") for m in _messages(env, "assistant"))


def test_a_file_that_parses_gets_no_line(env: Env) -> None:
    env.fake.stream_script = [
        tool_call_result("write_file", {"path": "words.py", "content": FIXED_PYTHON}),
        ChatResult(content="Done, words.py is ready."),
    ]
    events, _raw = chat_events(env, "Save the function that reverses the words in a sentence as words.py")
    assert "Syntax error" not in _messages(env, "tool")[0]["content"]
    assert "Checked by Kiwibro" not in _final_text(events)


def test_a_file_broken_by_a_later_write_is_reported_with_its_latest_state(env: Env) -> None:
    env.fake.stream_script = [
        tool_call_result("write_file", {"path": "words.py", "content": FIXED_PYTHON}, "call_1"),
        tool_call_result("write_file", {"path": "words.py", "content": BROKEN_PYTHON}, "call_2"),
        ChatResult(content="Done."),
    ]
    events, _raw = chat_events(env, "Save the function that reverses the words in a sentence as words.py")
    assert "Checked by Kiwibro: `words.py` has a syntax error on line 3" in _final_text(events)


# --- a reply written without the internet ---------------------------------------------------------


def _turn(message: str, **fields: Any) -> Turn:
    turn = Turn(conversation_id="c_honest", message=message, requested_model=None)
    for name, value in fields.items():
        setattr(turn, name, value)
    return turn


@pytest.fixture()
def wants_web(monkeypatch: pytest.MonkeyPatch):
    from kiwibro import readings

    monkeypatch.setattr(readings, "wants_search", lambda reading: True)


def test_no_connection_when_the_computer_is_offline_and_the_web_was_wanted(env: Env, offline, wants_web) -> None:
    assert env.svc.agent._no_connection(_turn("latest python version", reading=object())) is True
    assert env.svc.agent._no_connection(_turn("hello", web_tried=1)) is True


def test_no_connection_needs_the_web_to_have_been_wanted(env: Env, offline, monkeypatch: pytest.MonkeyPatch) -> None:
    from kiwibro import readings

    monkeypatch.setattr(readings, "wants_search", lambda reading: False)
    assert env.svc.agent._no_connection(_turn("hello", reading=object())) is False
    assert env.svc.agent._no_connection(_turn("hello")) is False


def test_no_connection_never_when_a_web_tool_worked(env: Env, offline, wants_web) -> None:
    assert env.svc.agent._no_connection(_turn("latest python version", reading=object(), tools_ok=["web_search"])) is False


def test_no_connection_when_every_web_call_failed_for_that_reason_only(env: Env, wants_web, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(connectivity.ENV_SWITCH, "0")  # the probe says online
    agent = env.svc.agent
    assert agent._no_connection(_turn("x", reading=object(), web_tried=2, web_offline=2)) is True
    assert agent._no_connection(_turn("x", reading=object(), web_tried=2, web_offline=1, web_failed=1)) is False
    assert agent._no_connection(_turn("x", reading=object(), web_tried=1, web_failed=1)) is False
    assert agent._no_connection(_turn("x", reading=object())) is False  # online, nothing failed


def test_an_offline_reply_starts_with_the_line_and_loses_its_unsourced_markers(env: Env, offline) -> None:
    env.fake.stream_script = [
        ChatResult(content="The latest stable Python is 3.13 [1], released in October [2].", usage=USAGE),
        ChatResult(content="The latest stable Python is 3.13 [1], released in October [2].", usage=USAGE),
        ChatResult(content="The latest stable Python is 3.13 [1], released in October [2].", usage=USAGE),
    ]
    events, _raw = chat_events(env, "Search the web: what is the latest stable Python version?")
    shown = _final_text(events)
    assert shown.startswith(prompts.NO_CONNECTION_LINE + "\n\n")
    assert shown.count(prompts.NO_CONNECTION_LINE) == 1
    assert "[1]" not in shown and "[2]" not in shown
    assert "3.13" in shown  # the model's words stay; only the unsourced marks and the first line change
    stored = _messages(env, "assistant")[-1]["content"]
    assert stored.startswith(prompts.NO_CONNECTION_LINE)


def test_a_small_talk_reply_offline_is_left_alone(env: Env, offline) -> None:
    env.fake.stream_script = [ChatResult(content="Hi, I am fine, thanks.", usage=USAGE)]
    events, _raw = chat_events(env, "Hi, how are you?")
    assert _final_text(events) == "Hi, I am fine, thanks."


def test_markers_are_kept_when_the_turn_has_the_sources(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(connectivity.ENV_SWITCH, "0")
    agent = env.svc.agent
    turn = _turn("what happened", tools_ok=["web_search"], evidence_items=[{"title": "a"}, {"title": "b"}])
    result = ChatResult(content="It happened [1] and again [2], and once more [3].", usage=USAGE)
    agent._honest_reply(turn, result)
    assert result.content == "It happened [1] and again [2], and once more."


def test_the_line_is_added_once_per_turn(env: Env, offline, wants_web) -> None:
    agent = env.svc.agent
    turn = _turn("latest python version", reading=object())
    first = ChatResult(content="From memory: 3.13.", usage=USAGE)
    agent._honest_reply(turn, first)
    second = ChatResult(content="From memory: 3.13, I think.", usage=USAGE)
    agent._honest_reply(turn, second)
    assert first.content == f"{prompts.NO_CONNECTION_LINE}\n\nFrom memory: 3.13."
    assert second.content == "From memory: 3.13, I think."


def test_a_repeated_offline_search_gets_the_offline_error_not_a_pacing_refusal(env: Env, offline) -> None:
    searches = [tool_call_result("web_search", {"query": "latest stable python"}, f"call_{n}") for n in (1, 2)]
    env.fake.stream_script = [*searches, ChatResult(content="I could not look it up.", usage=USAGE)]
    chat_events(env, "Search the web: what is the latest stable Python version?")
    results = [m["content"] for m in _messages(env, "tool")]
    assert results and all(result == f"Error: {connectivity.OFFLINE_ERROR}" for result in results[:2])
