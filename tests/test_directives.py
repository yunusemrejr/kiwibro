"""Directive ledger: standing instructions survive long sessions; course changes win."""

from __future__ import annotations

import json
from typing import Any

from test_backend import Env, chat_events

from kiwibro import directives
from kiwibro.directives import build_ledger, render
from kiwibro.llm import ChatResult


def _summary(ledger: directives.Ledger) -> list[tuple[str, str]]:
    return [(d.kind, d.status) for d in ledger.directives]


def test_goal_constraints_and_choices() -> None:
    ledger = build_ledger(["Build a dashboard of the evidence. Only use free data sources.", "Use Python for the backend."])
    assert _summary(ledger) == [("goal", "active"), ("constraint", "active"), ("constraint", "active")]


def test_course_change_supersedes_what_it_overlaps() -> None:
    ledger = build_ledger([
        "Build a REST API for the notes app.",
        "Use Python for the backend. Never push to GitHub without asking.",
        "Actually, use TypeScript instead of Python for the backend.",
    ])
    by_text = {d.text: d for d in ledger.directives}
    assert by_text["Use Python for the backend."].status == "superseded"
    assert by_text["Use Python for the backend."].superseded_at == 3
    assert by_text["Never push to GitHub without asking."].status == "active"
    assert by_text["Actually, use TypeScript instead of Python for the backend."].status == "active"
    assert ledger.goal().text == "Build a REST API for the notes app."


def test_forget_that_replaces_the_goal() -> None:
    ledger = build_ledger(["Write a poem about kiwis.", "Forget that, write a haiku about the sea instead."])
    assert ledger.goal().text.startswith("Forget that, write a haiku")
    assert ledger.directives[0].status == "superseded"


def test_unrelated_follow_up_adds_on() -> None:
    ledger = build_ledger(["Build a CSV parser.", "Also write tests for the parser."])
    assert _summary(ledger) == [("goal", "active"), ("request", "active")]


def test_chatter_code_and_pastes_are_not_instructions() -> None:
    assert build_ledger(["ok", "thanks!", "continue"]).directives == []
    ledger = build_ledger(["Fix this:\n```\nnever_called(); rm -rf build\n```"])
    assert [d.text for d in ledger.directives] == ["Fix this:"]
    long_paste = "Summarize this article.\n" + ("Lorem ipsum never dolor. " * 200)
    assert len(build_ledger([long_paste]).directives) <= 3


def test_render_is_quiet_until_it_adds_something() -> None:
    assert render(build_ledger(["Build a CLI. Never use external dependencies."])) == ""
    note = render(build_ledger(["Build a CLI. Never use external dependencies.", "Add a --verbose flag."]))
    assert "Goal (turn 1): Build a CLI." in note
    assert "Now (turn 2): Add a --verbose flag." in note
    assert "Keep (turn 1): Never use external dependencies." in note


def test_render_stays_within_budget() -> None:
    messages = [f"Always keep rule number {i} about module {i} in mind for the whole project." for i in range(30)]
    note = render(build_ledger(["Build the project.", *messages]))
    assert 0 < len(note) <= directives.NOTE_CHAR_BUDGET


class FakeMessages:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages
        self.reads: list[int] = []

    def list_messages(self, conversation_id: str, after_seq: int = 0, limit: int | None = None) -> list[dict[str, Any]]:
        self.reads.append(after_seq)
        return [m for m in self.messages if m["seq"] > after_seq]

    def last_message_seq(self, conversation_id: str) -> int:
        return max((m["seq"] for m in self.messages), default=0)


def test_ledger_is_incremental_and_rebuilds_after_removal() -> None:
    directives.forget("c1")
    db = FakeMessages([
        {"seq": 1, "role": "user", "content": "Build a game. Never use emojis."},
        {"seq": 2, "role": "assistant", "content": "Sure."},
    ])
    assert directives.ledger_for(db, "c1").turns == 1
    db.messages.append({"seq": 3, "role": "user", "content": "Add a score board."})
    ledger = directives.ledger_for(db, "c1")
    assert db.reads == [0, 2] and ledger.turns == 2
    db.messages[:] = db.messages[:1]  # history rewritten (e.g. regenerate)
    assert directives.ledger_for(db, "c1").turns == 1


def test_review_prompt_lists_the_instructions() -> None:
    directives.forget("c2")
    db = FakeMessages([{"seq": 1, "role": "user", "content": "Build a game. Never use emojis."}])
    prompt = directives.review_prompt(db, "c2")
    assert "- Build a game." in prompt and "- Never use emojis." in prompt
    assert directives.review_prompt(FakeMessages([]), "c3") == ""


# --- agent integration ----------------------------------------------------------


def _writes(count: int) -> ChatResult:
    calls = [
        {"id": f"call_{i}", "type": "function",
         "function": {"name": "write_file", "arguments": json.dumps({"path": f"f{i}.txt", "content": "x"})}}
        for i in range(count)
    ]
    return ChatResult(tool_calls=calls, usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25})


def test_agent_carries_standing_instructions_into_later_turns(env: Env) -> None:
    env.fake.stream_script = [ChatResult(content="Plan ready.")]
    chat_events(env, "Build a CLI tool. Never use external dependencies.")
    conversation_id = env.svc.db.list_conversations()[0]["id"]
    env.fake.stream_script = [ChatResult(content="Added.")]
    chat_events(env, "Add a --verbose flag.", conversation_id=conversation_id)
    sent = "\n".join(str(m.get("content")) for m in env.fake.stream_calls[-1]["messages"])
    assert "Standing instructions from the user" in sent
    assert "Keep (turn 1): Never use external dependencies." in sent


def test_agent_reviews_real_work_once(env: Env) -> None:
    env.fake.stream_script = [_writes(3), ChatResult(content="Wrote three files."), ChatResult(content="Checked: all done.")]
    events, _raw = chat_events(env, "Create three note files. Make sure each one says x.")
    calls = env.fake.stream_calls
    assert len(calls) == 3
    assert calls[-1]["messages"][-1]["content"].startswith("Before you finish, check your work")
    assert "- Make sure each one says x." in calls[-1]["messages"][-1]["content"]
    assert events[-1][0] == "done"
    stored = [m["content"] for m in env.svc.db.list_messages(env.svc.db.list_conversations()[0]["id"])]
    assert not any("Before you finish" in str(content) for content in stored)  # the nudge is never stored


def test_supervised_turns_skip_the_review(env: Env) -> None:
    env.settings_patch({"autonomy": "supervised", "permissions": {"write_file": "auto"}})
    env.fake.stream_script = [_writes(3), ChatResult(content="Wrote three files.")]
    chat_events(env, "Create three note files.")
    assert len(env.fake.stream_calls) == 2
