"""Situational guidance: chosen per turn by code, never in message 0."""

from __future__ import annotations

from test_backend import Env, chat_events

from kiwibro import guidance, prompts
from kiwibro.llm import ChatResult


def test_small_talk_gets_no_guidance() -> None:
    assert guidance.select("hi!") == []
    assert guidance.select("thanks a lot") == []
    assert guidance.note([]) == ""


def test_each_kind_of_request_gets_its_own_line() -> None:
    assert guidance.select("what's the latest news on the Mars mission") == ["web"]
    assert guidance.select("fix the failing test in app.py") == ["code"]
    assert guidance.select("what is the average of 3, 9 and 12") == ["compute"]
    assert guidance.select("take a screenshot") == ["desktop"]
    assert guidance.select("draw a chart of my spending") == ["visual"]
    assert guidance.select("what did I tell you last time about my trip") == ["memory"]
    assert guidance.select("What can you do?") == ["capabilities"]
    assert guidance.select("look at this", attachments=True) == ["attachments"]
    assert "steps" in guidance.select("research X, then write a doc", plan_kinds=["web", "write"], plan_steps=2)


def test_tool_lines_are_dropped_when_agent_mode_is_off() -> None:
    assert guidance.select("search the web for the latest kernel", tools=False) == []
    assert guidance.select("make a table of planets", tools=False) == ["visual"]


def test_message_zero_is_small_and_stable_while_the_note_adapts(env: Env) -> None:
    env.fake.stream_script = [ChatResult(content="Hello!")]
    _events, _raw = chat_events(env, "hi")
    first = env.fake.stream_calls[-1]["messages"]
    conversation_id = env.svc.db.list_conversations()[0]["id"]
    env.fake.stream_script = [ChatResult(content="A chart.")]
    chat_events(env, "draw a chart of my spending", conversation_id=conversation_id)
    second = env.fake.stream_calls[-1]["messages"]

    assert first[0]["content"] == second[0]["content"]  # KV-cache prefix untouched
    assert guidance.HEADER not in first[-1]["content"]  # a greeting gets nothing
    assert guidance.HEADER in second[-1]["content"]
    assert guidance.SNIPPETS["visual"] in second[-1]["content"]
    # Situational text lives only in the guidance snippets, not in the static prompt.
    for snippet in guidance.SNIPPETS.values():
        assert snippet not in second[0]["content"]
    assert len(prompts.SYSTEM_PROMPT) + len(prompts.TOOLS_BLOCK) < 2000


def test_only_memories_that_matched_count_as_used(env: Env) -> None:
    coffee = env.svc.memory.remember("User drinks black coffee every morning")
    garden = env.svc.memory.remember("User grows tomatoes in the garden")
    env.svc.db.mark_memories_used([garden["id"]])  # recently used, but unrelated below
    picked = env.svc.memory.retrieve("how should I make my coffee")
    assert [m["id"] for m in picked] == [coffee["id"]]
    assert env.svc.db.get_memory(coffee["id"])["use_count"] == 1
    assert env.svc.db.get_memory(garden["id"])["use_count"] == 1  # unchanged
