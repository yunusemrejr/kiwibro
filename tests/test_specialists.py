"""Keep the shipped weights and their measured behavior reproducible offline."""

from __future__ import annotations

import gzip
import json

import pytest

from kiwibro import specialists
from scripts.helpers import train_specialists as training
from test_backend import Env, chat_events


def test_export_matches_colab_report():
    pack = json.loads(gzip.decompress(specialists.MODEL_PATH.read_bytes()))
    report = pack["report"]
    assert len(specialists.MODEL_PATH.read_bytes()) < 80 * 1024
    assert pack["origin"] == report["origin"] == "google-colab"
    assert report["runtime_sha256"] == training.digest(training.RUNTIME.read_bytes())
    raw_corpus = (training.HERE / training.CORPUS_NAME).read_bytes()
    assert report["corpus_sha256"] == training.digest(raw_corpus)
    assert report["trainer_sha256"] == training.digest((training.HERE / "train_specialists.py").read_bytes())
    splits = training.read_corpus(raw_corpus.decode())
    assert set(pack["heads"]) == {head for head, stats in report["heads"].items() if stats["passed"]}
    assert set(pack["heads"]) == set(training.HEADS)

    for head, model in pack["heads"].items():
        advice = set(model["advice_labels"])
        assert advice == set(specialists.HINTS[head])
        for split, rows in splits[head].items():
            if split in ("train", "calibrate"):
                continue
            prompts, labels = zip(*rows)
            labels = [label if label in advice else "none" for label in labels]
            found = [
                specialists.classify(model, specialists.features(text))
                if specialists.admissible(head, text) else "none"
                for text in prompts
            ]
            outcome = training.score(labels, found)
            tracked = report["heads"][head]
            expected = tracked["accepted"] if split == "test" else tracked["regression"][split]
            assert outcome == expected
            reference = [training.baseline(head, text) for text in prompts]
            reference = [label if label in advice else "none" for label in reference]
            assert training.qualifies(outcome, training.score(labels, reference))


@pytest.mark.parametrize("message", [
    "Calculate the force exerted on the support beam and give its units.",
    "Please remember my preference for faster local builds.",
    "Write two notes on our book club's discussion.",
    "Build a general SQL tutorial covering queries and relationships with invented tables.",
    "Assess whether the attached film script reads well and review the roles of the main characters.",
])
def test_unrelated_requests_receive_no_specialist_advice(message):
    assert specialists.note(message) == ""


def test_hints_are_optional_per_tool_context():
    request = "Create a Three.js scene of a tree swaying while cartoon birds fly around it."
    assert "animate by elapsed time" in specialists.note(request, browser=False)
    assert specialists.note(request, browser=False, visual=False, code=False, tables=False, documents=False) == ""


@pytest.mark.parametrize("message, tool, hint", [
    ("Expand the test suite around Unicode normalization and trailing spaces.", "run_command",
     specialists.HINTS["code"]["tests"]),
    ("Give me total training hours and typical session length by job family from the HR worksheet.", "analyze_table",
     specialists.HINTS["data"]["aggregate"]),
    ("What does the signed contract say about the extra inventory charge? Quote the original clause, with its number.",
     "doc_read", specialists.HINTS["data"]["evidence"]),
])
def test_turn_includes_specialist_only_when_tools_are_allowed(env: Env, message, tool, hint):
    env.settings_patch({"autonomy": "supervised"})
    chat_events(env, message)
    assert any(hint in m["content"] for m in env.fake.stream_calls[-1]["messages"])

    env.settings_patch({"permissions": {tool: "deny"}})
    chat_events(env, message)
    assert not any(hint in m["content"] for m in env.fake.stream_calls[-1]["messages"])


def test_browser_advice_requires_evidence_and_avoids_account_changes():
    assert specialists.admissible("browser", "Follow the link on the page and report the heading.")
    assert not specialists.admissible("browser", "Find the first algebra error in my proof.")
    assert not specialists.admissible("browser", "Open my subscription page and cancel my plan.")
    assert not specialists.admissible("browser", "Log in to my account and update my billing address.")


def test_a_missing_or_corrupt_pack_falls_back(monkeypatch, tmp_path):
    path = tmp_path / "model.gz"
    monkeypatch.setattr(specialists, "MODEL_PATH", path)
    specialists._models.cache_clear()
    try:
        assert specialists.note("Build a new WebGL page with snowfall.") == ""
        path.write_bytes(b"invalid gzip")
        specialists._models.cache_clear()
        assert specialists.note("Build a new WebGL page with snowfall.") == ""
    finally:
        specialists._models.cache_clear()
