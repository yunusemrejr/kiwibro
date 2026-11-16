from pathlib import Path

import pytest

from scripts.mind import agent_corpus, agent_supplement


def test_supplement_is_reproducible_and_separate_from_evaluation(tmp_path: Path) -> None:
    base, first, second = (tmp_path / name for name in ("base", "first", "second"))
    agent_corpus.build(base, train_each=1, test_each=1, seed=42)
    frozen = {name: (base / name).read_bytes() for name in ("train.jsonl", "heldout.jsonl", "probes.jsonl")}
    report = agent_supplement.build(first, each=2)
    assert agent_supplement.build(second, each=2) == report
    assert report["train"]["episodes"] == 20
    assert report["thinking"]["episodes"] == 10
    assert (first / "train.jsonl").read_bytes() == (second / "train.jsonl").read_bytes()
    assert (first / "fixtures.jsonl").read_bytes() == (second / "fixtures.jsonl").read_bytes()
    assert not (first / "heldout.jsonl").exists()
    assert not (first / "probes.jsonl").exists()

    original = [row for part in ("train", "heldout") for row in agent_corpus.read_jsonl(base / f"{part}.jsonl")]
    supplement = agent_corpus.read_jsonl(first / "train.jsonl")
    assert {row["id"] for row in supplement}.isdisjoint(row["id"] for row in original)
    original_requests = {agent_corpus.normalize(content if isinstance(content, str) else content[0]["text"])
                         for row in original if (content := row["messages"][1]["content"])}
    assert all(agent_corpus.normalize(row["messages"][1]["content"]) not in original_requests for row in supplement)

    with pytest.raises(ValueError, match="separate output directory"):
        agent_supplement.build(base)
    assert {name: (base / name).read_bytes() for name in frozen} == frozen


def test_reasoning_rows_and_recovery_follow_evidence(tmp_path: Path) -> None:
    agent_supplement.build(tmp_path, each=2)
    rows = {row["id"]: row for row in agent_corpus.read_jsonl(tmp_path / "train.jsonl")}

    def tools(case: str):
        return [call["function"] for message in rows[case]["messages"] for call in message.get("tool_calls", [])]

    assert [call["name"] for call in tools("supp-exact-unavailable-001")] == ["model_info"]
    delegation = tools("supp-named-fallback-001")
    assert [call["name"] for call in delegation] == ["delegate", "delegate"]
    assert delegation[0]["arguments"]["model"] != delegation[1]["arguments"]["model"]
    assert delegation[0]["arguments"]["task"] == delegation[1]["arguments"]["task"]

    repair = rows["supp-render-stop-001"]
    assert [call["name"] for call in tools(repair["id"])] == ["read_file", "apply_patch"]
    assert "No script errors" in repair["messages"][-2]["content"]
    patched = next(call for call in tools(repair["id"]) if call["name"] == "apply_patch")
    assert len(patched["arguments"]["edits"]) == 1

    table = rows["supp-table-schema-001"]
    assert [call["arguments"].get("aggregate") for call in tools(table["id"])] == [
        "sum:mass_kg", None, "count,sum:load_kg",
    ]
    assert "unknown column" in table["messages"][3]["content"]
    assert "2 rows matched" in table["messages"][-2]["content"]

    plain = rows["supp-table-schema-002"]
    assert "thinking" not in plain and all("reasoning" not in message for message in plain["messages"])
    assert table["thinking"] and all(message["reasoning"] for message in table["messages"] if message["role"] == "assistant")
