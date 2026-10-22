"""Long-session / independent-memory tests (memory v2, schema v2, context v2).

These tests pin the behaviours that let very long conversations keep working
on small-context models: ranked memory retrieval, the unified recall engine,
budget-scaled prompt blocks and dedupe correctness at scale. They use the
same offline fake-LM pattern as test_backend.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kiwibro import create_app
from kiwibro.context import estimate_tokens
from kiwibro.llm import Completion, LMStudioError
from kiwibro.memory import is_similar, normalize

MODEL = {
    "id": "fake-1",
    "name": "fake-1",
    "type": "llm",
    "state": "loaded",
    "context": 4096,
    "max_context": 4096,
    "tools": True,
    "publisher": "test",
    "quantization": "Q4_K_M",
}


class FakeLLM:
    def __init__(self) -> None:
        self.models = [dict(MODEL)]
        self.memory_ops: Any = {"ops": []}
        self.summary_text = "Summary of the earlier conversation."
        self.complete_calls: list[list[dict[str, Any]]] = []

    def list_models(self, force: bool = False):
        return [dict(m) for m in self.models]

    def complete(self, *, model, messages, max_tokens=900, temperature=0.0, read_timeout=300, retries=1):
        self.complete_calls.append(messages)
        return Completion(self.summary_text)

    def complete_json(self, *, model, messages, max_tokens=900, temperature=0.0, read_timeout=300, retries=1):
        return self.memory_ops


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = create_app(
        {
            "TESTING": True,
            "KIWIBRO_DATA_DIR": str(tmp_path / "data"),
            "KIWIBRO_WORKSPACE": str(workspace),
            "KIWIBRO_LMSTUDIO_URL": "http://127.0.0.1:9",
        }
    )
    svc = app.extensions["kiwibro"]
    fake = FakeLLM()
    monkeypatch.setattr(svc.llm, "list_models", fake.list_models)
    monkeypatch.setattr(svc.llm, "complete", fake.complete)
    monkeypatch.setattr(svc.llm, "complete_json", fake.complete_json)
    return svc, fake


# --- dedupe -----------------------------------------------------------------


def test_is_similar_edge_cases_still_hold():
    assert is_similar("User prefers dark themes", "User prefers dark theme") is True
    assert is_similar("likes kiwi birds and hiking", "hiking and kiwi birds likes") is True
    assert is_similar("User prefers dark themes", "Runs a bakery in Lisbon") is False
    # Length-mismatched strings can never reach the sequence threshold.
    assert is_similar("tea", "User really enjoys drinking tea every single morning") is False
    assert is_similar("", "anything") is False


def test_remember_dedupes_at_scale(env):
    svc, _fake = env
    for i in range(120):
        svc.db.create_memory(
            "fact",
            f"Filler fact {i} about topic {i % 30}",
            normalize(f"Filler fact {i} about topic {i % 30}"),
        )
    svc.memory.remember("User prefers dark themes", "preference")
    svc.memory.remember("User prefers dark theme", "preference")
    matches = [m for m in svc.db.list_memories() if "dark theme" in m["content"]]
    assert len(matches) == 1


# --- ranked retrieval & recall ----------------------------------------------


def test_retrieval_prefers_full_query_coverage_over_recency(env):
    svc, _fake = env
    svc.memory.remember("User keeps bees in Lisbon", "fact")
    # Newer memories sharing only the common word "User" must not win.
    for i in range(10):
        svc.memory.remember(f"User filler note {i} about subject {i}", "fact")
    block, used = svc.memory.build_block("bees Lisbon")
    assert used and used[0]["content"] == "User keeps bees in Lisbon"
    assert "bees in Lisbon" in block


def test_recall_returns_ranked_memories(env):
    svc, _fake = env
    svc.memory.remember("User's dog is named Pixel", "fact")
    for i in range(6):
        svc.memory.remember(f"User note {i} about gardening", "fact")
    items = svc.memory.recall("dog Pixel", limit=3)
    assert items, "recall should find the dog memory"
    assert items[0]["source"] == "memory"
    assert "Pixel" in items[0]["content"]


def test_pinned_memory_leads_retrieval(env):
    svc, _fake = env
    pinned = svc.memory.remember("User is allergic to peanuts", "fact")
    svc.memory.update(pinned["id"], pinned=True)
    svc.memory.remember("User enjoys peanut-free snacks", "preference")
    _block, used = svc.memory.build_block("peanut snacks")
    assert used[0]["id"] == pinned["id"]


# --- budget-scaled prompt block ----------------------------------------------


def test_build_block_respects_small_token_budget(env):
    svc, _fake = env
    topics = [
        "sourdough baking timers", "bicycle gear ratios", "telescope mirror sizes",
        "espresso extraction yields", "bonsai watering schedules", "kayak paddle lengths",
        "vinyl record pressings", "greenhouse humidity targets", "marathon pacing splits",
        "aquarium nitrate levels", "ham radio antenna heights", "sailboat rigging tension",
    ]
    for topic in topics:
        svc.memory.remember(f"Budget fact: user tracks {topic} in a paper notebook", "fact")
    block, _used = svc.memory.build_block("budget fact", max_tokens=120)
    # Header + at least the top bullet, but nowhere near the 800-token default.
    assert block
    assert estimate_tokens(block) <= 160
    default_block, _ = svc.memory.build_block("budget fact")
    assert estimate_tokens(default_block) > estimate_tokens(block)


# --- extraction prompt economy ------------------------------------------------


def test_extraction_sends_relevant_memories_not_all_recent(env):
    svc, fake = env
    svc.memory.remember("User keeps bees in Lisbon", "fact")
    for i in range(30):
        svc.memory.remember(f"Unrelated gardening note {i}", "fact")
    conversation = svc.db.create_conversation()
    svc.db.add_message(conversation["id"], "user", "My bees in Lisbon produced honey today")
    svc.db.add_message(conversation["id"], "assistant", "Lovely!")
    messages = svc.memory._extraction_messages(conversation["id"])
    prompt = messages[-1]["content"]
    assert "bees in Lisbon" in prompt, "the relevant memory must be in the extraction prompt"
    # The whole utility prompt stays viable for small-context models.
    assert estimate_tokens(prompt) < 2500


# --- schema v2: episodic index, segments, token cache ------------------------


def test_schema_v2_objects_and_token_cache(env):
    svc, _fake = env
    conversation = svc.db.create_conversation()
    msg = svc.db.add_message(conversation["id"], "user", "a" * 360)
    # 360 ASCII chars / 3.6 = 100 tokens + 4 overhead, cached at write time
    # and in agreement with the context estimator (single formula, two homes).
    assert msg["est_tokens"] == estimate_tokens("a" * 360) + 4
    assert svc.db.get_conversation(conversation["id"])["extracted_through_seq"] == 0
    rows = svc.db.message_token_rows(conversation["id"])
    assert rows and rows[0]["est_tokens"] == msg["est_tokens"]


def test_v1_database_upgrades_with_backfill(tmp_path: Path):
    """A pre-v2 database gains columns + a backfilled episodic index."""
    import sqlite3

    from kiwibro.db import _SCHEMA_V1, Database

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA_V1)
    conn.execute(
        "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO schema_migrations VALUES (1, '2026-01-01T00:00:00+00:00')")
    conn.execute(
        "INSERT INTO conversations(id, title, created_at, updated_at) "
        "VALUES ('c_old', 'Old chat', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO messages(id, conversation_id, seq, role, content, created_at) "
        "VALUES ('m_old', 'c_old', 1, 'user', 'The launch code is kiwi-42', '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()

    db = Database(path)
    db.initialize()
    hits = db.search_messages("launch code kiwi", conversation_id="c_old")
    assert hits and hits[0]["id"] == "m_old"
    # Re-running initialize is a no-op (guarded migration).
    db.initialize()
    assert db.search_messages("kiwi", conversation_id="c_old")


def test_episodic_search_ranks_and_filters(env):
    svc, _fake = env
    conversation = svc.db.create_conversation()
    other = svc.db.create_conversation()
    svc.db.add_message(conversation["id"], "user", "We chose the name Weka for the boat")
    svc.db.add_message(conversation["id"], "assistant", "Weka is a fine name for a boat")
    for i in range(5):
        svc.db.add_message(conversation["id"], "user", f"Unrelated sailing chatter {i}")
    svc.db.add_message(other["id"], "user", "Weka appears in another chat too")

    hits = svc.db.search_messages("Weka boat name", conversation_id=conversation["id"], limit=2)
    assert hits and "Weka" in hits[0]["content"]
    assert all(h["conversation_id"] == conversation["id"] for h in hits)
    older = svc.db.search_messages("Weka", conversation_id=conversation["id"], before_seq=2)
    assert older and all(h["seq"] < 2 for h in older)


def test_episodic_index_stays_in_sync(env):
    svc, _fake = env
    conversation = svc.db.create_conversation()
    msg = svc.db.add_message(conversation["id"], "user", "original harbor plan")
    svc.db.update_message(msg["id"], content="revised mountain plan")
    assert svc.db.search_messages("harbor", conversation_id=conversation["id"]) == []
    assert svc.db.search_messages("mountain", conversation_id=conversation["id"])
    svc.db.clear_conversation(conversation["id"])
    assert svc.db.search_messages("mountain", conversation_id=conversation["id"]) == []


def test_summary_segments_roundtrip(env):
    svc, _fake = env
    conversation = svc.db.create_conversation()
    svc.db.add_summary_segment(conversation["id"], 1, 40, "First part.", est_tokens=3)
    svc.db.add_summary_segment(conversation["id"], 41, 90, "Second part.", est_tokens=3)
    segments = svc.db.list_summary_segments(conversation["id"])
    assert [s["through_seq"] for s in segments] == [40, 90]
    assert svc.db.segments_through_seq(conversation["id"]) == 90
    svc.db.clear_conversation(conversation["id"])
    assert svc.db.list_summary_segments(conversation["id"]) == []


def test_recall_includes_episodes(env):
    svc, _fake = env
    conversation = svc.db.create_conversation()
    svc.db.add_message(conversation["id"], "user", "Remember the harbor gate code is 4815")
    svc.db.add_message(conversation["id"], "assistant", "Noted: 4815")
    items = svc.memory.recall("harbor gate code", conversation_id=conversation["id"], limit=4)
    episodes = [i for i in items if i["source"] == "episode"]
    assert episodes and "4815" in episodes[0]["content"]


def test_use_count_increments_on_retrieval(env):
    svc, _fake = env
    memory = svc.memory.remember("User sails on weekends", "fact")
    svc.memory.build_block("sails weekends")
    refreshed = svc.db.get_memory(memory["id"])
    assert refreshed["use_count"] >= 1
    assert refreshed["last_used_at"] is not None


# --- context v2: small-context safety, segments, recall, prefix stability ----


def test_utility_limits_scale_with_context():
    from kiwibro.context import utility_limits

    in_4k, out_4k = utility_limits(4096)
    in_32k, out_32k = utility_limits(32768)
    assert in_4k + out_4k < 4096  # the call itself must fit the model
    assert in_32k > in_4k and out_32k >= out_4k
    assert utility_limits(0)[0] >= 700  # defensive floor, never zero


def test_summary_calls_always_fit_small_context(env):
    """Regression: v1 sent a fixed ~4k-token summary prompt to 4k models."""
    svc, fake = env  # MODEL context is 4096 in this module's fixture
    # Long partial summaries force the multi-level reduce to actually run.
    fake.summary_text = "Summary sentence retaining several concrete details and names. " * 8
    conversation = svc.db.create_conversation()
    for i in range(60):
        svc.db.add_message(conversation["id"], "user", f"Topic {i} " + "detail " * 40)
        svc.db.add_message(conversation["id"], "assistant", f"Reply {i} " + "answer " * 40)
    prepared = svc.context.prepare(conversation["id"], "fake-1", "SYSTEM")
    assert prepared.compacted is True
    assert fake.complete_calls, "summariser must have been called"
    for messages in fake.complete_calls:
        prompt_tokens = sum(estimate_tokens(m.get("content") or "") for m in messages)
        assert prompt_tokens + 600 < 4096, f"summary prompt {prompt_tokens} tok cannot fit 4k ctx"
    assert prepared.used <= prepared.budget


def test_compaction_records_summary_segments(env):
    svc, _fake = env
    conversation = svc.db.create_conversation()
    for i in range(60):
        svc.db.add_message(conversation["id"], "user", f"Segment topic {i} " + "word " * 40)
        svc.db.add_message(conversation["id"], "assistant", f"Segment reply {i} " + "text " * 40)
    svc.context.prepare(conversation["id"], "fake-1", "SYSTEM")
    segments = svc.db.list_summary_segments(conversation["id"])
    assert segments, "compaction must persist fine-grained segment summaries"
    refreshed = svc.db.get_conversation(conversation["id"])
    assert segments[-1]["through_seq"] == refreshed["summary_through_seq"]


def test_recall_note_returns_exact_early_fact_after_compaction(env):
    svc, _fake = env
    conversation = svc.db.create_conversation()
    svc.db.add_message(conversation["id"], "user", "Plan the harbor trip; the gate code is 4815")
    svc.db.add_message(conversation["id"], "assistant", "Harbor trip planned, code noted")
    for i in range(50):
        svc.db.add_message(conversation["id"], "user", f"Filler sailing topic {i} " + "chat " * 30)
        svc.db.add_message(conversation["id"], "assistant", f"Filler reply {i} " + "talk " * 30)
    svc.db.add_message(conversation["id"], "user", "What was the harbor gate code again?")
    prepared = svc.context.prepare(conversation["id"], "fake-1", "SYSTEM")
    assert prepared.recalled >= 1
    newest_user = [m for m in prepared.messages if m["role"] == "user"][-1]
    assert "4815" in newest_user["content"], "exact excerpt must ride on the newest user message"
    assert "4815" not in prepared.messages[0]["content"], "volatile recall must not enter system prompt"
    assert prepared.anchored is True
    assert prepared.used <= prepared.budget


def test_prepare_prefix_is_stable_across_turns(env):
    """KV rule: older rendered messages never change when a new turn arrives."""
    svc, _fake = env
    conversation = svc.db.create_conversation()
    svc.db.add_message(conversation["id"], "user", "First question")
    svc.db.add_message(
        conversation["id"], "tool", "tool output " * 800, tool_call_id="c1", tool_name="read_file"
    )
    svc.db.add_message(conversation["id"], "assistant", "First answer")
    svc.db.add_message(conversation["id"], "user", "Second question")
    first = svc.context.prepare(conversation["id"], "fake-1", "SYSTEM")
    svc.db.add_message(conversation["id"], "assistant", "Second answer")
    svc.db.add_message(conversation["id"], "user", "Third question")
    second = svc.context.prepare(conversation["id"], "fake-1", "SYSTEM")
    # Everything before the second user message must be byte-identical.
    def upto_second_user(messages):
        out, users = [], 0
        for m in messages:
            if m["role"] == "user":
                users += 1
                if users == 2:
                    break
            out.append(m)
        return out

    assert upto_second_user(first.messages) == upto_second_user(second.messages)


def test_extend_with_same_turn_context_stays_incremental(env):
    """The agent re-passes its fixed turn note every tool round; extend must
    append, not rebuild (a rebuild would re-render the KV prefix)."""
    svc, _fake = env
    conversation = svc.db.create_conversation()
    svc.db.add_message(conversation["id"], "user", "Question one")
    prepared = svc.context.prepare(
        conversation["id"], "fake-1", "SYSTEM", turn_context="## Note\nvolatile"
    )
    assert "volatile" in prepared.messages[-1]["content"]
    svc.db.add_message(conversation["id"], "assistant", "working…")
    svc.db.add_message(conversation["id"], "tool", "result", tool_call_id="c1")
    extended = svc.context.extend(
        prepared, conversation["id"], "fake-1", "SYSTEM", turn_context="## Note\nvolatile"
    )
    assert extended.messages[0] is prepared.messages[0]  # appended, not rebuilt
    assert len(extended.messages) == len(prepared.messages) + 2
    joined = "".join(str(m.get("content") or "") for m in extended.messages)
    assert joined.count("volatile") == 1  # note placed exactly once


def test_summarize_long_text_map_reduce(env):
    svc, fake = env
    text = "\n\n".join(f"Paragraph {i} about topic {i}. " + "Evidence sentence. " * 30 for i in range(20))
    result = svc.context.summarize_long_text(text, "fake-1")
    assert result
    assert len(fake.complete_calls) >= 2  # chunked, not one oversized call
    for messages in fake.complete_calls:
        prompt_tokens = sum(estimate_tokens(m.get("content") or "") for m in messages)
        assert prompt_tokens + 600 < 4096


def test_message_pagination_keyset(env):
    svc, _fake = env
    conversation = svc.db.create_conversation()
    for i in range(10):
        svc.db.add_message(conversation["id"], "user", f"message {i}")
    page, has_more = svc.db.list_messages_page(conversation["id"], limit=3)
    assert [m["seq"] for m in page] == [8, 9, 10] and has_more is True
    page, has_more = svc.db.list_messages_page(conversation["id"], before_seq=8, limit=3)
    assert [m["seq"] for m in page] == [5, 6, 7] and has_more is True
    page, has_more = svc.db.list_messages_page(conversation["id"], before_seq=2, limit=3)
    assert [m["seq"] for m in page] == [1] and has_more is False


def test_conversation_api_pagination_is_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = create_app(
        {"TESTING": True, "KIWIBRO_DATA_DIR": str(tmp_path / "data"),
         "KIWIBRO_WORKSPACE": str(workspace), "KIWIBRO_LMSTUDIO_URL": "http://127.0.0.1:9"}
    )
    svc = app.extensions["kiwibro"]
    fake = FakeLLM()
    monkeypatch.setattr(svc.llm, "list_models", fake.list_models)
    conversation = svc.db.create_conversation()
    for i in range(10):
        svc.db.add_message(conversation["id"], "user", f"message {i}")
    client = app.test_client()

    full = client.get(f"/api/conversations/{conversation['id']}").get_json()
    assert len(full["messages"]) == 10 and "page" not in full  # back-compat

    paged = client.get(f"/api/conversations/{conversation['id']}?limit=4").get_json()
    assert [m["seq"] for m in paged["messages"]] == [7, 8, 9, 10]
    assert paged["page"] == {"has_more": True, "oldest_seq": 7}
    older = client.get(
        f"/api/conversations/{conversation['id']}?limit=4&before_seq=7"
    ).get_json()
    assert [m["seq"] for m in older["messages"]] == [3, 4, 5, 6]
    assert client.get(f"/api/conversations/{conversation['id']}?limit=0").status_code == 400
    assert client.get(f"/api/conversations/{conversation['id']}?limit=x").status_code == 400


def test_extraction_watermark_retries_after_failure(env, monkeypatch: pytest.MonkeyPatch):
    """A failed extraction must not skip facts: the watermark stays put."""
    svc, fake = env
    conversation = svc.db.create_conversation()
    svc.db.add_message(conversation["id"], "user", "I just moved to Lisbon for work")
    svc.db.add_message(conversation["id"], "assistant", "Congrats on the move!")

    def boom(**kwargs):
        raise LMStudioError("timeout", "slow model")

    monkeypatch.setattr(svc.llm, "complete_json", boom)
    assert svc.memory.extract_and_apply(conversation["id"], "fake-1")["added"] == 0
    assert svc.db.get_conversation(conversation["id"])["extracted_through_seq"] == 0

    monkeypatch.setattr(svc.llm, "complete_json", fake.complete_json)
    fake.memory_ops = {"ops": [{"op": "add", "kind": "fact", "content": "User moved to Lisbon for work"}]}
    counts = svc.memory.extract_and_apply(conversation["id"], "fake-1")
    assert counts["added"] == 1  # the previously failed window was retried
    watermark = svc.db.get_conversation(conversation["id"])["extracted_through_seq"]
    assert watermark == svc.db.last_message_seq(conversation["id"])
    # Nothing pending: a second pass is a no-op without new messages.
    assert svc.memory.extract_and_apply(conversation["id"], "fake-1") == {
        "added": 0, "updated": 0, "removed": 0,
    }


def test_extraction_failure_never_raises(env, monkeypatch: pytest.MonkeyPatch):
    svc, _fake = env

    def boom(**kwargs):
        raise LMStudioError("timeout", "slow model")

    monkeypatch.setattr(svc.llm, "complete_json", boom)
    conversation = svc.db.create_conversation()
    svc.db.add_message(conversation["id"], "user", "hello there")
    assert svc.memory.extract_and_apply(conversation["id"], "fake-1") == {
        "added": 0,
        "updated": 0,
        "removed": 0,
    }


# --- saving memories reliably -------------------------------------------------


def test_new_memory_survives_pruning_at_the_cap(env, monkeypatch: pytest.MonkeyPatch):
    from kiwibro import config

    svc, _fake = env
    monkeypatch.setattr(config, "MEMORY_CAP", 5)
    ids = [svc.memory.remember(f"User owns a {animal} named Bob{n}")["id"]
           for n, animal in enumerate(["cat", "dog", "fish", "horse", "parrot"])]
    svc.db.mark_memories_used(ids)  # every old memory now outranks a fresh one
    saved = svc.memory.remember("User adopted the zebra sanctuary schedule")
    assert svc.db.get_memory(saved["id"]) is not None
    assert svc.db.count_memories() == 5


def test_merge_without_kind_keeps_the_existing_kind(env):
    svc, _fake = env
    svc.memory.remember("Sam leads the Atlas project", "person")
    merged = svc.memory.remember("Sam leads the Atlas project team")
    assert merged["kind"] == "person"
    assert merged["content"] == "Sam leads the Atlas project team"


def test_remember_tool_accepts_an_invented_kind():
    from kiwibro import tools

    assert tools.validate_args("remember", {"content": "User walks daily", "kind": "habit"})["kind"] == "fact"
    assert tools.validate_args("remember", {"content": "User walks daily", "kind": "Person"})["kind"] == "person"
    assert tools.validate_args("remember", {"content": "User walks daily"})["kind"] is None


def test_unparseable_extraction_does_not_block_later_messages(env, monkeypatch: pytest.MonkeyPatch):
    svc, fake = env
    conversation = svc.db.create_conversation()
    svc.db.add_message(conversation["id"], "user", "I just moved to Lisbon for work")
    svc.db.add_message(conversation["id"], "assistant", "Congrats!")
    fake.memory_ops = None  # what complete_json returns for "Nothing worth saving."
    svc.memory.extract_and_apply(conversation["id"], "fake-1")
    assert svc.db.get_conversation(conversation["id"])["extracted_through_seq"] == 0  # one retry
    svc.memory.extract_and_apply(conversation["id"], "fake-1")
    first_window = svc.db.last_message_seq(conversation["id"])
    assert svc.db.get_conversation(conversation["id"])["extracted_through_seq"] == first_window

    svc.db.add_message(conversation["id"], "user", "My sister Ana lives in Porto")
    fake.memory_ops = {"ops": [{"op": "add", "kind": "person", "content": "User's sister Ana lives in Porto"}]}
    assert svc.memory.extract_and_apply(conversation["id"], "fake-1")["added"] == 1


def test_extraction_window_fits_the_transcript(env, monkeypatch: pytest.MonkeyPatch):
    svc, fake = env
    conversation = svc.db.create_conversation()
    for n in range(8):
        svc.db.add_message(conversation["id"], "user", f"MARKER_{n} " + "x" * 1200)
    sent: list[str] = []

    def capture(**kwargs):
        sent.append(kwargs["messages"][-1]["content"])
        return {"ops": []}

    monkeypatch.setattr(svc.llm, "complete_json", capture)
    svc.memory.extract_and_apply(conversation["id"], "fake-1")
    through = svc.db.get_conversation(conversation["id"])["extracted_through_seq"]
    included = [n for n in range(8) if f"MARKER_{n} " in sent[0]]
    assert included == list(range(len(included))) and included  # oldest first, nothing skipped
    seqs = [m["seq"] for m in svc.db.list_messages(conversation["id"])]
    assert through == seqs[len(included) - 1]  # watermark stops at the last message sent


def test_extraction_ops_are_normalised_not_dropped(env):
    svc, _fake = env
    counts = svc.memory._apply_ops([
        {"op": "add", "kind": "habit", "content": "User runs every morning"},
        {"op": "add", "content": "User collects vintage fountain pens " + "and inks " * 40},
    ])
    assert counts["added"] == 2
    kinds = {m["content"][:20]: m["kind"] for m in svc.db.all_memories()}
    assert kinds["User runs every morn"] == "fact"
    assert all(len(m["content"]) <= 300 for m in svc.db.all_memories())


def test_parse_ops_accepts_content_aliases():
    from kiwibro.memory import parse_ops

    assert parse_ops('{"ops":[{"op":"add","memory":"User likes tea"}]}') == [
        {"op": "add", "content": "User likes tea"}
    ]


def test_extraction_never_deletes_a_pinned_memory(env):
    svc, _fake = env
    memory = svc.memory.remember("User is allergic to peanuts")
    svc.db.update_memory(memory["id"], pinned=True)
    assert svc.memory._apply_ops([{"op": "delete", "id": memory["id"]}])["removed"] == 0
    assert svc.db.get_memory(memory["id"]) is not None


def test_recall_matches_singular_and_plural(env):
    svc, _fake = env
    svc.memory.remember("User prefers dark themes", "preference")
    assert svc.db.search_memories("theme")
    svc.memory.remember("User has one cat", "fact")
    assert any("cat" in m["content"] for m in svc.db.search_memories("cats"))


def test_complete_json_ignores_reasoning_and_truncated_replies(monkeypatch: pytest.MonkeyPatch):
    from kiwibro.llm import LMStudioClient

    client = LMStudioClient(lambda: {})
    reply = Completion("")
    monkeypatch.setattr(client, "complete", lambda **kwargs: reply)

    def ask(text: str, finish: str | None = "stop") -> Any:
        reply.text, reply.finish_reason = text, finish
        return client.complete_json(model="m", messages=[])

    thinking = '<think>Format is {"op":"add","content":"User likes tea"}, or [a memory]?</think>'
    answer = '{"ops":[{"op":"add","content":"User lives in Berlin"}]}'
    assert ask(thinking + answer) == {"ops": [{"op": "add", "content": "User lives in Berlin"}]}
    assert ask('Should I add [a memory]?</think>' + answer)["ops"][0]["content"] == "User lives in Berlin"
    assert ask('<think>still thinking {"ops": [1]}') is None
    assert ask('{"ops":[{"op":"add","content":"User lives in Berl', "length") is None
    assert ask(answer, "length") == {"ops": [{"op": "add", "content": "User lives in Berlin"}]}


def test_extraction_only_saves_facts_grounded_in_the_conversation(env):
    from kiwibro.memory import grounded

    transcript = "Hi! I'm Deniz, a backend developer in Izmir. My cat is called Pamuk."
    assert grounded("User's name is Deniz", transcript)
    assert grounded("User has a cat named Pamuk", transcript)
    assert not grounded("User prefers dark themes", transcript)  # copied prompt example
    assert not grounded("User is Ana, a nurse in Porto", transcript)

    svc, _fake = env
    counts = svc.memory._apply_ops(
        [
            {"op": "add", "kind": "fact", "content": "User works as a backend developer in Izmir"},
            {"op": "add", "kind": "preference", "content": "User prefers dark themes"},
            {"op": "add", "kind": "person", "content": "Pamuk"},
            # an "update" of an id that does not exist is a new fact
            {"op": "update", "id": "mem_123", "kind": "fact", "content": "User has a cat named Pamuk"},
        ],
        transcript,
    )
    assert counts["added"] == 2
    assert sorted(m["content"] for m in svc.db.all_memories()) == [
        "User has a cat named Pamuk",
        "User works as a backend developer in Izmir",
    ]


def test_model_written_first_person_memories_name_the_user(env):
    from kiwibro.memory import third_person

    assert third_person("I always use Neovim") == "User said: I always use Neovim"
    assert third_person("My cat is Pamuk") == "User said: My cat is Pamuk"
    assert third_person("Israel trip planned") == "Israel trip planned"
    svc, _fake = env
    conversation = svc.db.create_conversation()
    saved = svc.memory.remember("I always use Neovim", source_conversation_id=conversation["id"])
    assert saved["content"] == "User said: I always use Neovim"
    assert svc.memory.remember("I like tea")["content"] == "I like tea"  # typed by the user in the UI
