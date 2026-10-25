"""Sessions: forking keeps seqs, search index and compaction state; key-term memory relevance;
continuity across chats (earlier-chats note, recall from other chats)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from datetime import datetime, timedelta, timezone

from kiwibro import sessions, terms
from kiwibro.db import Database, fts_query
from kiwibro.memory import MemoryEngine, normalize
from test_multitask_terminal_browser import app_client  # noqa: F401  (re-exported fixture)

MESSAGE_FIELDS = (
    "role", "content", "reasoning", "tool_calls", "tool_call_id", "tool_name",
    "model", "tokens", "est_tokens", "meta", "created_at",
)


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "data" / "kiwibro.db")
    database.initialize()
    return database


def _seed(db: Database) -> dict[str, Any]:
    conv = db.create_conversation(title="Source Chat", model="fake-model")
    cid = conv["id"]
    db.add_message(cid, role="user", content="alpha question")
    db.add_message(cid, role="assistant", content="bravo answer", reasoning="thinking", model="fake-model", tokens=12, meta={"k": 1})
    db.add_message(cid, role="assistant", content="", tool_calls=[{"id": "t1", "name": "lookup"}])
    db.add_message(cid, role="tool", content="charlie result", tool_call_id="t1", tool_name="lookup")
    db.add_message(cid, role="user", content="delta follow up")
    return conv


def test_fork_copies_every_message_with_its_seq(db: Database):
    src = _seed(db)
    fork = db.fork_conversation(src["id"])
    assert fork is not None and fork["id"] != src["id"]

    copied = db.list_messages(fork["id"])
    original = db.list_messages(src["id"])
    assert [m["seq"] for m in copied] == [1, 2, 3, 4, 5]
    for new, old in zip(copied, original):
        assert new["id"] != old["id"]
        for field in MESSAGE_FIELDS:
            assert new[field] == old[field], field

    assert [m["seq"] for m in db.search_messages("charlie", conversation_id=fork["id"])] == [4]
    assert [m["seq"] for m in db.search_messages("alpha", conversation_id=fork["id"])] == [1]
    assert db.conversation_message_count(src["id"]) == 5


def test_fork_through_seq_cuts_the_copy(db: Database):
    src = _seed(db)
    fork = db.fork_conversation(src["id"], through_seq=2)
    assert [m["seq"] for m in db.list_messages(fork["id"])] == [1, 2]
    assert db.search_messages("delta", conversation_id=fork["id"]) == []
    assert db.search_messages("bravo", conversation_id=fork["id"])

    assert db.list_messages(db.fork_conversation(src["id"], through_seq=0)["id"]) == []
    assert len(db.list_messages(db.fork_conversation(src["id"], through_seq=99)["id"])) == 5
    assert len(db.list_messages(src["id"])) == 5


def test_fork_keeps_summary_and_segments_when_cut_reaches_them(db: Database):
    src = _seed(db)
    db.add_summary_segment(src["id"], 1, 2, "first two", est_tokens=5)
    db.add_summary_segment(src["id"], 3, 4, "next two", est_tokens=6)
    db.update_conversation(src["id"], summary="first two; next two", summary_through_seq=4)

    fork = db.fork_conversation(src["id"], through_seq=4)
    assert fork["summary"] == "first two; next two"
    assert fork["summary_through_seq"] == 4
    segments = db.list_summary_segments(fork["id"])
    assert [(s["from_seq"], s["through_seq"], s["summary"], s["est_tokens"]) for s in segments] == [
        (1, 2, "first two", 5),
        (3, 4, "next two", 6),
    ]
    source_ids = {s["id"] for s in db.list_summary_segments(src["id"])}
    assert source_ids.isdisjoint({s["id"] for s in segments})

    full = db.fork_conversation(src["id"])
    assert full["summary"] == "first two; next two"
    assert len(db.list_summary_segments(full["id"])) == 2


def test_fork_drops_summary_and_segments_when_cut_is_before_them(db: Database):
    src = _seed(db)
    db.add_summary_segment(src["id"], 1, 2, "first two")
    db.add_summary_segment(src["id"], 3, 4, "next two")
    db.update_conversation(src["id"], summary="first two; next two", summary_through_seq=4)

    fork = db.fork_conversation(src["id"], through_seq=3)
    assert fork["summary"] == ""
    assert fork["summary_through_seq"] == 0
    assert db.list_summary_segments(fork["id"]) == []
    assert len(db.list_messages(fork["id"])) == 3

    assert db.get_conversation(src["id"])["summary_through_seq"] == 4
    assert len(db.list_summary_segments(src["id"])) == 2


def test_fork_clamps_memory_watermark_to_the_cut(db: Database):
    src = _seed(db)
    db.update_conversation(src["id"], extracted_through_seq=4)
    assert db.fork_conversation(src["id"], through_seq=2)["extracted_through_seq"] == 2
    assert db.fork_conversation(src["id"])["extracted_through_seq"] == 4

    db.update_conversation(src["id"], extracted_through_seq=1)
    assert db.fork_conversation(src["id"], through_seq=3)["extracted_through_seq"] == 1


def test_fork_copies_workspace_and_model_and_names_the_copy(db: Database):
    src = _seed(db)
    db.update_conversation(src["id"], workspace="projects/app")

    fork = db.fork_conversation(src["id"])
    assert fork["title"] == "Source Chat (fork)"
    assert fork["title_auto"] is False
    assert fork["model"] == "fake-model"
    assert fork["workspace"] == "projects/app"

    assert db.fork_conversation(src["id"], title="Branch B")["title"] == "Branch B"


def test_fork_of_missing_conversation_is_none(db: Database):
    assert db.fork_conversation("c_missing") is None
    assert db.list_conversations() == []


def test_api_fork_keeps_seqs_and_validates_seq(app_client):
    client, svc = app_client
    src = _seed(svc.db)

    resp = client.post(f"/api/conversations/{src['id']}/fork", json={"seq": 3})
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["ok"] is True
    conv = data["conversation"]
    assert conv["id"] != src["id"]
    assert conv["title"] == "Source Chat (fork)"
    assert [m["seq"] for m in svc.db.list_messages(conv["id"])] == [1, 2, 3]

    for bad in ({"seq": "3"}, {"seq": True}, {"seq": -1}, {"seq": 1.5}):
        resp_bad = client.post(f"/api/conversations/{src['id']}/fork", json=bad)
        assert resp_bad.status_code == 400, bad
        assert resp_bad.get_json()["error"]["code"] == "bad_request"

    full = client.post(f"/api/conversations/{src['id']}/fork")
    assert full.status_code == 201
    assert len(svc.db.list_messages(full.get_json()["conversation"]["id"])) == 5

    assert client.post("/api/conversations/c_missing/fork", json={"seq": 3}).status_code == 404


# --- key terms & memory relevance ---------------------------------------------------


def test_stem_folds_inflections_to_one_key():
    for group in (("plays", "playing", "played", "play"), ("lives", "living", "lived", "live"),
                  ("uses", "using", "used", "use"), ("string", "strings"), ("studies", "studying", "study")):
        assert len({terms.stem(word) for word in group}) == 1, group


def test_key_terms_skip_function_words_and_tag_topics():
    assert terms.key_terms("is the build passing") == {"build", "pass"}
    assert "#food" in terms.key_terms("recommend a dinner") and "#food" in terms.key_terms("favourite food")
    assert terms.key_terms("what is it?") == set()


FACTS = (
    "User lives in Istanbul", "User works as a backend developer at Acme", "User prefers dark themes in the editor",
    "User's dog is called Pamuk", "User is learning Japanese", "User uses Neovim for Python",
    "User is allergic to peanuts", "User plays the guitar on weekends", "User's favourite food is lahmacun",
)


@pytest.fixture()
def memory(db: Database) -> MemoryEngine:
    for fact in FACTS:
        db.create_memory("fact", fact, normalize(fact))
    return MemoryEngine(db, None, lambda: {})


@pytest.mark.parametrize(("query", "expected"), [
    ("where do I live?", "User lives in Istanbul"),
    ("what instrument do I play", "User plays the guitar on weekends"),
    ("recommend a dinner for me", "User's favourite food is lahmacun"),
    ("which editor do I use", "User uses Neovim for Python"),
])
def test_memory_found_through_inflections_and_topics(memory: MemoryEngine, query: str, expected: str):
    assert expected in [m["content"] for m in memory.retrieve(query)]


@pytest.mark.parametrize("query", ["is the build passing", "what's in the file?", "write a function to sort a list"])
def test_function_words_pull_in_no_memories(memory: MemoryEngine, query: str):
    assert memory.retrieve(query) == []


SELF = "I am Yunus Emre Vurgun. I am a developer and IT specialist from Istanbul."


@pytest.mark.parametrize("query", ["what's my name?", "who am I?", "how old am I?", "hi"])
def test_users_own_self_description_rides_on_every_turn(db: Database, query: str):
    db.create_memory("person", SELF, normalize(SELF))
    block, _memories = MemoryEngine(db, None, lambda: {}).build_block(query)
    assert "User said: I am Yunus Emre Vurgun" in block  # the user's words, never the assistant's "I"


def test_other_first_person_notes_stay_relevance_gated(db: Database):
    note = "My sister is a doctor in Ankara"
    db.create_memory("person", note, normalize(note))
    engine = MemoryEngine(db, None, lambda: {})
    assert "User said: My sister is a doctor" in engine.build_block("what does my sister do?")[0]
    assert engine.build_block("is the build passing")[0] == ""


def test_extraction_sees_only_memories_it_might_change(memory: MemoryEngine, db: Database):
    for fact in ("User drives a 2015 Corolla", "User's laptop is a ThinkPad", "User reads sci-fi novels"):
        db.create_memory("fact", fact, normalize(fact))
    shown = [m["content"] for m in memory._relevant_existing_memories(
        ["User: I just moved to Ankara for a new job at a startup.", "Assistant: Congratulations on the move!"]
    )]
    assert "User lives in Istanbul" in shown  # the memory this window should update
    assert "User works as a backend developer at Acme" in shown  # same topic, different words
    assert not {"User drives a 2015 Corolla", "User's laptop is a ThinkPad", "User reads sci-fi novels"} & set(shown)
    assert len(shown) <= 5


def test_fts_query_drops_function_words():
    assert fts_query("what is the budget") == '"budget"*'


# --- continuity across chats --------------------------------------------------------


@pytest.mark.parametrize("message", [
    "what did we decide yesterday?", "continue where we left off", "you said the budget was fine",
    "in the other chat we talked about Lisbon", "like last time", "dün ne konuşmuştuk",
])
def test_refers_back(message: str):
    assert sessions.refers_back(message)


@pytest.mark.parametrize("message", ["what is 2+2", "write a poem about kiwis", "search the web for PHP news"])
def test_does_not_refer_back(message: str):
    assert not sessions.refers_back(message)


def test_when_reads_days():
    now = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)
    assert sessions.when((now - timedelta(hours=1)).isoformat(), now) == "today"
    assert sessions.when((now - timedelta(days=1)).isoformat(), now) == "yesterday"
    assert sessions.when((now - timedelta(days=3)).isoformat(), now) == "3 days ago"
    assert sessions.when("not a date", now) == ""


def _chat(db: Database, title: str, *messages: tuple[str, str]) -> str:
    cid = db.create_conversation(title=title)["id"]
    for role, content in messages:
        db.add_message(cid, role, content)
    return cid


def test_recent_note_lists_other_chats_with_gist_and_excerpts(db: Database):
    _chat(db, "Lisbon trip", ("user", "hi"), ("user", "let's plan the Lisbon trip budget"),
          ("assistant", "Budget is 900 euros for flights and hotel."))
    _chat(db, "Empty chat")  # no messages: not listed
    current = _chat(db, "New chat", ("user", "what budget did we decide yesterday?"))
    note = sessions.recent_note(db, current, "what budget did we decide yesterday?")
    assert "Your other recent chats" in note and "Lisbon trip (today)" in note
    assert "asked: let's plan the Lisbon trip budget" in note  # the real request, not the greeting
    assert "900 euros" in note and "Empty chat" not in note and "New chat" not in note
    assert sessions.recent_note(db, current, "what is the budget") == ""  # no back-reference


def test_recent_note_prefers_the_summary_and_respects_its_budget(db: Database):
    for index in range(8):
        cid = _chat(db, f"Chat {index}", ("user", f"topic number {index} with enough words"))
        db.update_conversation(cid, summary=f"Summary of chat {index}. " * 20)
    current = _chat(db, "Now", ("user", "continue where we left off"))
    note = sessions.recent_note(db, current, "continue where we left off", max_tokens=120)
    assert "Summary of chat 7" in note
    assert sessions.estimate_tokens(note) <= 120 + 80  # one line may straddle; never the whole list
    assert note.count("\n- ") < sessions.RECENT_CHATS


def test_recall_also_searches_other_chats(db: Database):
    other = _chat(db, "Database choice", ("user", "we picked PostgreSQL for the inventory service"))
    current = _chat(db, "Now", ("user", "which database for inventory?"))
    items = MemoryEngine(db, None, lambda: {}).recall("inventory database", conversation_id=current)
    hits = [item for item in items if item.get("conversation_id") == other]
    assert hits and hits[0]["chat"] == "Database choice"



# --- finding chats by what was said -----------------------------------------------


def test_search_conversations_returns_each_chat_once_with_a_clean_snippet(db: Database):
    a = _chat(db, "Hi", ("user", "summarize the PHP2Shell advisory"),
              ("assistant", "**PHP2Shell** — bulletin: patch PHP now"), ("assistant", "PHP2Shell is critical"))
    _chat(db, "Other", ("user", "kiwi bird facts"))
    matches = db.search_conversations("php2shell")
    assert [m["id"] for m in matches] == [a]
    assert "**" not in matches[0]["snippet"] and "PHP2Shell" in matches[0]["snippet"]
    assert db.search_conversations("") == []


def test_api_conversation_search(app_client):
    client, svc = app_client
    cid = svc.db.create_conversation(title="Hello")["id"]
    svc.db.add_message(cid, "user", "the Lisbon trip budget is 900 euros")
    data = client.get("/api/conversations/search?q=lisbon").get_json()
    assert data["ok"] and [m["id"] for m in data["matches"]] == [cid]
    assert client.get("/api/conversations/search?q=l").get_json()["matches"] == []  # too short


def test_conversation_view_says_whether_a_reply_is_running(app_client):
    client, svc = app_client
    cid = svc.db.create_conversation(title="Unanswered")["id"]
    svc.db.add_message(cid, "user", "Now make it skip hidden files too")
    data = client.get(f"/api/conversations/{cid}").get_json()
    assert data["busy"] is False and data["messages"][-1]["role"] == "user"  # the UI offers Retry


def test_reading_a_chat_for_export_does_not_prefill_it(app_client, monkeypatch):
    client, svc = app_client
    queued: list[str | None] = []
    monkeypatch.setattr(
        svc.agent.readiness, "request", lambda model=None, conversation_id=None: queued.append(conversation_id) or {}
    )
    cid = svc.db.create_conversation(title="Export me")["id"]
    assert client.get(f"/api/conversations/{cid}?warm=0").status_code == 200
    assert queued == []  # an export is not a next turn: no minutes of CPU prefill
    client.get(f"/api/conversations/{cid}")
    assert queued == [cid]  # opening the chat still warms it
