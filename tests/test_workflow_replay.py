"""The contract checks use scripted decisions. They do not measure model quality."""

import ast
import base64
import gzip
import json
import zipfile
from io import BytesIO

import pytest

from scripts.mind import agent_corpus, calibration_traces, train_workflows, workflow_replay
from scripts.mind.bench_successor import SAMPLING_SEED


PROFILE = {
    "artifact_sha256": "a" * 64, "runtime_id": "test-fixture", "sampling_seed": SAMPLING_SEED,
    "temperature": 0, "cache_prompt": True, "thinking": "off", "max_tokens": 320, "timeout": 180,
}


def scripted_replay(probes, output, *, misses=(), incomplete=(), profile=PROFILE):
    cases = agent_corpus.read_jsonl(probes)
    metadata = {
        "model": "fixture-only", "binding": profile, "thinking": profile["thinking"], "count": len(cases),
        "per_family_and_action_kind": 0, "probes_sha256": train_workflows.digest(probes.read_bytes()),
    }
    records = []
    for row in cases:
        case = row["id"]
        expected = row["expected"]
        calls = [[call["name"], call["arguments"]] for call in expected["calls"]]
        answer = " ".join(expected["required"])
        passed = case not in misses and case not in incomplete
        records.append({
            "case": case, "episode": row["episode"], "expected": expected,
            "calls": calls if passed else [], "answer": answer if passed else "I cannot answer.",
            "error": None, "finish_reason": "length" if case in incomplete else "stop", "passed": passed,
        })
    agent_corpus.jsonl(output, [metadata, *records])


@pytest.fixture
def traces(tmp_path):
    probes, replays = tmp_path / "probes", tmp_path / "replays"
    calibration_traces.build(probes)
    replays.mkdir()
    scripted_replay(probes / "train.jsonl", replays / "train.jsonl", misses={"train-missing_path-ash"},
                    incomplete={"train-dependency-ash"})
    scripted_replay(probes / "calibrate.jsonl", replays / "calibrate.jsonl", misses={"calibrate-blocked-holly"})
    return probes, replays


def test_colab_bundle_keeps_blind_results_and_features_out(tmp_path, traces, monkeypatch):
    probes, replays = traces
    prepared = workflow_replay.prepare_observations(probes, replays, {})
    assert set(prepared["splits"]) == {"train", "calibrate"}
    assert prepared["provenance"]["folds"]["train"]["targets"] == {"location": 1, "none": 70}
    assert prepared["provenance"]["folds"]["calibrate"]["targets"] == {"boundary": 1, "none": 23}
    assert prepared["provenance"]["folds"]["train"]["outcomes"]["incomplete_generation"] == 1
    assert all(row[2] != "train-dependency-ash/call_observed" for row in prepared["splits"]["train"])

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "train.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(train_workflows, "dataset", lambda *_: {"recovery": {}})
    launcher = tmp_path / "colab.py"
    train_workflows.prepare(corpus, launcher, probes, replays)
    payload = next(node.value for node in ast.walk(ast.parse(launcher.read_text()))
                   if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) > 100)
    with zipfile.ZipFile(BytesIO(base64.b64decode(payload))) as archive:
        actual = json.loads(archive.read(train_workflows.REPLAY_INPUT))
        assert set(actual["splits"]) == {"train", "calibrate"}
        assert actual["provenance"] == prepared["provenance"]
        assert not any("test" in path for path in archive.namelist())
        assert b"test-dependency-aspen" not in archive.read(train_workflows.REPLAY_INPUT)


def test_rejects_changed_probes_cross_split_input_and_mixed_artifacts(traces):
    probes, replays = traces
    path = probes / "test.jsonl"
    rows = agent_corpus.read_jsonl(path)
    test_input = workflow_replay.observation(rows[0]["messages"][-1])
    with pytest.raises(ValueError, match="across authored/train and test"):
        workflow_replay.prepare_observations(probes, replays, {"train": [(test_input, "location")]})
    rows[0]["messages"][-1]["content"] = "modified holdout input"
    agent_corpus.jsonl(path, rows)
    with pytest.raises(ValueError, match="unchanged frozen recovery probes"):
        workflow_replay.prepare_observations(probes, replays, {})

    input_path, replay_path = probes / "calibrate.jsonl", replays / "calibrate.jsonl"
    scripted_replay(input_path, replay_path, misses={"calibrate-blocked-holly"},
                    profile={**PROFILE, "artifact_sha256": "b" * 64})
    with pytest.raises(ValueError, match="different artifacts"):
        workflow_replay.prepare_observations(probes, replays, {})


def test_bound_replay_checks_scores_and_requires_every_case(traces):
    probes, replays = traces
    path = replays / "train.jsonl"
    rows = agent_corpus.read_jsonl(path)
    rows[1]["passed"] = True
    agent_corpus.jsonl(path, rows)
    with pytest.raises(ValueError, match="Score changed"):
        workflow_replay.prepare_observations(probes, replays, {})
    rows[1]["passed"] = False
    rows[0]["count"] -= 1
    agent_corpus.jsonl(path, rows[:-1])
    with pytest.raises(ValueError, match="missing probe IDs"):
        workflow_replay.prepare_observations(probes, replays, {})


def test_no_complete_recovery_miss_cannot_calibrate(traces):
    probes, replays = traces
    scripted_replay(probes / "train.jsonl", replays / "train.jsonl", incomplete={"train-blocked-ash"})
    with pytest.raises(ValueError, match="no completed successor misses"):
        workflow_replay.prepare_observations(probes, replays, {})


def test_candidate_advice_is_blind_and_pair_scores_gain_regression_and_incomplete(tmp_path, traces):
    probes, replays = traces
    calibration = workflow_replay.prepare_observations(probes, replays, {})["provenance"]
    pack = tmp_path / "candidate.json.gz"
    # A small hand-written fixture head issues location hints only to failed searches.
    candidate = {
        "origin": "google-colab",
        "report": {"successor_recovery": calibration},
        "heads": {"recovery": {
            "bias": [0, 512], "labels": ["location", "none"], "advice_labels": ["location"],
            "weights": {"w:absent": [2048, 0]}, "thresholds": [0.5, 0.10],
        }},
    }
    pack.write_bytes(gzip.compress(json.dumps(candidate).encode()))
    test = probes / "test.jsonl"
    hinted = tmp_path / "hinted.jsonl"
    plan = workflow_replay.hint_probes(pack, test, hinted)
    assert list(plan["selected"]) == [f"test-missing_path-{name}" for name in calibration_traces.NAMES["test"]]

    base, after = tmp_path / "baseline.jsonl", tmp_path / "assisted.jsonl"
    scripted_replay(test, base, misses={"test-missing_path-aspen", "test-dependency-aspen"})
    scripted_replay(hinted, after, misses={"test-missing_path-birch"},
                    incomplete={"test-missing_path-chestnut"})
    report = workflow_replay.compare(pack, test, base, hinted, after)
    assert report["outcomes"]["gain"] == 1
    assert report["outcomes"]["regression"] == 1
    assert report["outcomes"]["incomplete_pair"] == 1
    assert report["outcomes"]["no_hint_gain"] == 1
    assert report["outcomes"]["correct_hint_for_miss"] == 1
    assert report["outcomes"]["hint_on_solved_failure"] == 4
    assert report["outcomes"]["hint_on_control"] == 0

    scripted_replay(hinted, after, profile={**PROFILE, "max_tokens": 640})
    with pytest.raises(ValueError, match="model and runtime profile"):
        workflow_replay.compare(pack, test, base, hinted, after)


def test_calibration_freezes_only_helpful_labels_for_fresh_blind_pair(tmp_path, traces):
    probes, replays = traces
    legacy = {split: (probes / f"{split}.jsonl").read_bytes() for split in calibration_traces.NAMES}
    calibration_traces.build(probes, fresh_blind=True)
    assert all((probes / f"{split}.jsonl").read_bytes() == old for split, old in legacy.items())
    calibration = probes / "calibrate.jsonl"
    baseline = replays / "calibrate.jsonl"
    calibration_misses = {
        "calibrate-missing_path-holly", "calibrate-blocked-holly", "calibrate-stale_browser-holly",
    }
    scripted_replay(calibration, baseline, misses=calibration_misses)
    provenance = workflow_replay.prepare_observations(probes, replays, {})["provenance"]
    pack = tmp_path / "candidate.json.gz"
    pack.write_bytes(gzip.compress(json.dumps({
        "origin": "google-colab", "report": {"successor_recovery": provenance},
        "heads": {"recovery": {
            "bias": [0, 0, 0, 0, 512], "labels": ["location", "boundary", "dependency", "render", "none"],
            "advice_labels": ["location", "boundary", "dependency", "render"],
            "weights": {
                "w:open": [2048, 0, 0, 0, 0], "w:refused": [0, 2048, 0, 0, 0],
                "w:absent": [0, 0, 2048, 0, 0], "w:mount": [0, 0, 0, 2048, 0],
            },
            "thresholds": [0.5, 0.1],
        }},
    }).encode()))

    cal_hinted, cal_assisted = tmp_path / "cal-hinted.jsonl", tmp_path / "cal-assisted.jsonl"
    workflow_replay.hint_probes(pack, calibration, cal_hinted, split="calibrate")
    scripted_replay(cal_hinted, cal_assisted, misses={"calibrate-blocked-rowan"},
                    incomplete={"calibrate-stale_browser-rowan"})
    blind = probes / "blind.jsonl"
    with pytest.raises(ValueError, match="unchanged frozen recovery probes"):
        workflow_replay.freeze(pack, calibration, baseline, cal_hinted, cal_assisted, probes / "test.jsonl")
    policy = workflow_replay.freeze(pack, calibration, baseline, cal_hinted, cal_assisted, blind)
    assert policy["labels"] == ["location"]
    assert policy["per_hint"]["location"]["gain"] == 1
    assert policy["per_hint"]["boundary"]["gain"] == policy["per_hint"]["boundary"]["regression"] == 1
    assert policy["per_hint"]["dependency"]["hints"] == 4 and policy["per_hint"]["dependency"]["gain"] == 0
    assert policy["per_hint"]["render"]["incomplete_pair"] == 1

    hinted = tmp_path / "blind-hinted.jsonl"
    plan = workflow_replay.hint_probes(pack, blind, hinted, policy=policy)
    assert plan["selected"] == {f"blind-missing_path-{name}": "location" for name in calibration_traces.BLIND_NAMES}
    with pytest.raises(ValueError, match="qualified labels"):
        workflow_replay.hint_probes(pack, blind, hinted, policy={**policy, "labels": ["boundary", "location"]})
    blind_baseline, blind_assisted = tmp_path / "blind-base.jsonl", tmp_path / "blind-assisted.jsonl"
    scripted_replay(blind, blind_baseline, misses={"blind-missing_path-apricot"})
    scripted_replay(hinted, blind_assisted)
    report = workflow_replay.compare(pack, blind, blind_baseline, hinted, blind_assisted, policy=policy)
    assert report["split"] == "blind" and report["frozen_labels"] == ["location"]
    assert report["policy_sha256"] == workflow_replay.object_digest(policy)
    assert report["outcomes"]["gain"] == 1 and report["outcomes"]["hints"] == 6

    scripted_replay(cal_hinted, cal_assisted, misses=calibration_misses)
    abstain = workflow_replay.freeze(pack, calibration, baseline, cal_hinted, cal_assisted, blind)
    assert abstain["labels"] == []
    assert not workflow_replay.hint_probes(pack, blind, hinted, policy=abstain)["selected"]
    assert hinted.read_bytes() == blind.read_bytes()

    baseline.write_bytes(baseline.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="exact calibration baseline"):
        workflow_replay.freeze(pack, calibration, baseline, cal_hinted, cal_assisted, blind)
