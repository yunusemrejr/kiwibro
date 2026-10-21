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
