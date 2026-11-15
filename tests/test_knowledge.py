"""The read-only knowledge base: what it holds, when it speaks and when it stays silent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kiwibro import agent, knowledge, tools

FILES = sorted(knowledge.BASE.glob("*.md"))


def test_the_base_stays_under_its_size_limit():
    assert len(FILES) >= 20
    assert knowledge.MAX_BYTES == 200 * 1024 * 1024
    assert sum(path.stat().st_size for path in FILES) < knowledge.MAX_BYTES


@pytest.mark.parametrize("path", FILES, ids=lambda path: path.stem)
def test_every_file_is_a_topic_with_short_entries(path):
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    assert lines[0].startswith("# ")
    titles = [line[3:] for line in lines if line.startswith("## ")]
    assert len(titles) >= 20 and len(set(titles)) == len(titles)
    assert not any(line.startswith("```") for line in lines)
    found = knowledge._parse(content, path.stem)
    assert len(found) == len(titles), "an entry without a body"
    assert max(len(entry.text) for entry in found) <= 700


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("what is the capital of Australia?", "Canberra"),
        ("What's the chemical symbol for gold", "Au"),
        ("when did world war 1 start", "1914"),
        ("when did world war 2 end", "1945"),
        ("how do I convert celsius to fahrenheit", "add 32"),
        ("what does chmod do?", "permission"),
        ("can you tell me what a mutex is", "lock"),
        ("what is quantization of a model?", "4-bit"),
        ("what is the capital of France?", "Paris"),
        ("what is the speed of light", "299"),
    ],
)
def test_a_question_the_base_covers_gets_its_entry(question, answer):
    note = knowledge.note(question)
    assert note.startswith(knowledge.NOTE_HEADER)
    assert answer in note


@pytest.mark.parametrize(
    "message",
    [
        "hi",
        "thanks a lot",
        "write a python script that renames files by date",
        "how much disk space do I have",
        "which file should I edit",
        "is the server running",
        "what time is it",
        "what does this error mean?",
        "what's the size of the downloads folder",
        "where is the bug",
        "what's left to do",
        "what version is installed",
        "which tab is open",
        "how many tests failed",
        "what are you working on",
        "what is the latest python version",
        "what's 17 times 23",
        "what is the capital of Wakanda?",
    ],
)
def test_a_message_about_anything_else_gets_nothing(message):
    assert knowledge.note(message) == ""


def test_the_entry_named_after_the_subject_is_the_one_given():
    assert [entry.title.split(":")[0] for entry in knowledge.search("capital of Spain")] == ["Spain"]


def test_at_most_two_entries_are_given():
    assert 1 <= len(knowledge.search("git stash")) <= knowledge.MAX_ENTRIES


def test_british_and_american_spelling_find_the_same_entry():
    assert knowledge.search("quantisation model") == knowledge.search("quantization model") != []


def test_the_turn_note_carries_the_entry_and_small_talk_carries_none():
    asked = SimpleNamespace(message="what is the capital of Australia?", conversation_id="c1")
    assert "Canberra" in agent.AgentManager._knowledge_part(asked)
    assert agent.AgentManager._knowledge_part(SimpleNamespace(message="hello there", conversation_id="c1")) == ""


def test_an_empty_web_search_falls_back_on_the_base(monkeypatch):
    monkeypatch.setattr(tools.research, "weather_report", lambda search, query: "")
    monkeypatch.setattr(tools.research, "news_request", lambda query: None)
    monkeypatch.setattr(tools.research, "search", lambda search, query, limit, deep=False: [])
    ctx = SimpleNamespace(search=None, deep=False, shown_before=(), must_mention=())
    held = tools._execute("web_search", {"query": "capital of Australia", "max_results": 5}, ctx)
    assert held.content.startswith(knowledge.KNOWLEDGE_INSTEAD) and "Canberra" in held.content
    assert held.meta["knowledge"] is True and held.meta["count"] >= 1  # it counts as something found
    nothing = tools._execute("web_search", {"query": "zorblax 9000 price", "max_results": 5}, ctx)
    assert nothing.content == tools.research.NO_RESULTS


def test_a_base_over_the_limit_is_cut_off(monkeypatch, tmp_path):
    for name in ("a", "b"):
        (tmp_path / f"{name}.md").write_text(f"# Topic {name}\n\n## Entry {name}\nA body about {name}.\n", encoding="utf-8")
    monkeypatch.setattr(knowledge, "BASE", tmp_path)
    monkeypatch.setattr(knowledge, "MAX_BYTES", (tmp_path / "a.md").stat().st_size)
    knowledge._index.cache_clear()
    try:
        assert [entry.title for entry in knowledge.entries()] == ["Entry a"]
    finally:
        knowledge._index.cache_clear()
