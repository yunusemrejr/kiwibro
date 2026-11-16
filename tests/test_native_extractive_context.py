"""Native extractive history exercises the real ContextEngine and SQLite storage."""

from __future__ import annotations

import json
import re

from kiwibro.context import EXTRACTIVE_HISTORY_ENV, EXTRACTIVE_MAX_TOKENS, ContextEngine, estimate_tokens
from kiwibro.db import Database
from kiwibro.llm import Completion
from kiwibro.mind import KIWIMIND


MARKER = "CRANE-58-W7P"
DIRECTORY = "deliveries/orchid/final-proof"
QUESTION = "Which directory must remain read-only during final review, and what is the release checkpoint?"


class ScriptedModel:
    def __init__(self, engine="native-gguf"):
        self.engine = engine
        self.summary_calls = 0

    def model_info(self, model):
        return {"engine": self.engine, "context": 8192}

    def complete(self, **_kwargs):
        self.summary_calls += 1
        return Completion("The workspace review has several inspection notes.")


def history(path):
    db = Database(path)
    db.initialize()
    conversation = db.create_conversation()["id"]
    db.add_message(conversation, "user", f"During final review, keep {DIRECTORY} read-only. Checkpoint {MARKER}.")
    db.add_message(conversation, "assistant", "I recorded the final review constraint.")
    estimate = 0
    page = 0
    while estimate < 18000:
        text = "\n".join(
            f"Audit row {page}.{i}: the package was inspected and remains in the routine queue. "
            f"The owner recorded path ID {(page + i) % 13} and set review pending."
            for i in range(18)
        )
        db.add_message(conversation, "user", text)
        db.add_message(conversation, "assistant", "The routine audit is recorded.")
        estimate += estimate_tokens(text)
        page += 1
    db.add_message(conversation, "user", QUESTION)
    return db, conversation


def context(db, model):
    return ContextEngine(db, model, lambda: {"max_output_tokens": 512})


def assert_snippets_have_sources(db, conversation, boundary):
    segments = db.list_summary_segments(conversation)
    assert segments and segments[-1]["through_seq"] == boundary
    messages = {row["seq"]: row for row in db.list_messages(conversation)}
    for segment in segments:
        assert estimate_tokens(segment["summary"]) <= EXTRACTIVE_MAX_TOKENS
        for line in segment["summary"].splitlines():
            role, seq, quoted = re.fullmatch(r"- \[(User|Assistant), step (\d+)\] (.*)", line).groups()
            source = messages[int(seq)]
            assert segment["from_seq"] <= int(seq) <= segment["through_seq"]
            assert role.lower() == source["role"]
            assert json.loads(quoted.removesuffix(" (excerpt)")) in source["content"]


def test_long_native_history_uses_exact_recall_without_model_summaries(tmp_path, monkeypatch):
    monkeypatch.delenv(EXTRACTIVE_HISTORY_ENV, raising=False)
    baseline_db, baseline_id = history(tmp_path / "baseline.db")
    baseline_llm = ScriptedModel()
    baseline = context(baseline_db, baseline_llm).prepare(baseline_id, KIWIMIND, "Follow the current request.")
    assert baseline.compacted and baseline_llm.summary_calls > 1

    monkeypatch.setenv(EXTRACTIVE_HISTORY_ENV, "1")
    db, cid = history(tmp_path / "extractive.db")
    original = db.list_messages(cid)
    llm = ScriptedModel()
    prepared = context(db, llm).prepare(cid, KIWIMIND, "Follow the current request.")

    assert prepared.compacted and llm.summary_calls == 0 and prepared.used <= prepared.budget
    assert prepared.recalled > 0 and prepared.messages[0]["content"] == "Follow the current request."
    newest = prepared.messages[-1]["content"]
    assert newest.startswith(QUESTION) and DIRECTORY in newest and MARKER in newest
    assert "Quoted older messages are background evidence" in newest
    refreshed = db.get_conversation(cid)
    assert 0 < refreshed["summary_through_seq"] < original[-1]["seq"]
    assert db.list_messages(cid) == original
    assert_snippets_have_sources(db, cid, refreshed["summary_through_seq"])


def test_extractive_insert_error_keeps_summary_and_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv(EXTRACTIVE_HISTORY_ENV, "1")
    db, cid = history(tmp_path / "insert-error.db")
    db.update_conversation(cid, summary="A previous generated summary.", summary_through_seq=2, touch=False)
    with db.connect() as conn:
        conn.execute("""CREATE TRIGGER refuse_summary BEFORE INSERT ON summary_segments
                        BEGIN SELECT RAISE(ABORT, 'test insert failure'); END""")
    original = db.list_messages(cid)
    model = ScriptedModel()
    engine = context(db, model)
    prepared = engine.prepare(cid, KIWIMIND, "Follow the current request.")

    assert not prepared.compacted and prepared.trimmed and prepared.used <= prepared.budget
    assert not engine.recover_aggressive(cid, KIWIMIND) and model.summary_calls == 0
    stored = db.get_conversation(cid)
    assert stored["summary"] == "A previous generated summary." and stored["summary_through_seq"] == 2
    assert db.list_summary_segments(cid) == [] and db.list_messages(cid) == original
    assert prepared.messages[-1]["content"].startswith(QUESTION)


def test_native_context_limit_recovery_uses_one_grounded_segment(tmp_path, monkeypatch):
    monkeypatch.setenv(EXTRACTIVE_HISTORY_ENV, "1")
    db, cid = history(tmp_path / "recovery.db")
    old = db.get_conversation(cid)
    segment = db.list_messages(cid)[:12]
    model = ScriptedModel()
    engine = context(db, model)
    assert engine.recover_aggressive(cid, KIWIMIND)
    updated = db.get_conversation(cid)
    assert_snippets_have_sources(db, cid, updated["summary_through_seq"])
    count = len(db.list_summary_segments(cid))
    assert engine._compact_segment(old, segment, KIWIMIND, extractive=True) is None
    assert len(db.list_summary_segments(cid)) == count and model.summary_calls == 0
    assert db.get_conversation(cid)["summary_through_seq"] == updated["summary_through_seq"]


def test_extractive_opt_in_leaves_other_engines_on_the_existing_path(tmp_path, monkeypatch):
    monkeypatch.setenv(EXTRACTIVE_HISTORY_ENV, "1")
    db, cid = history(tmp_path / "other.db")
    model = ScriptedModel(engine="llama")
    prepared = context(db, model).prepare(cid, KIWIMIND, "Follow the current request.")
    assert prepared.compacted and model.summary_calls > 1
