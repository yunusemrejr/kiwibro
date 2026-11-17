"""Steering replay uses the same final-result distinction as the live agent."""

import hashlib

from kiwibro import sense, steer
from scripts.mind import agent_corpus, calibration_traces


def test_replay_keeps_a_promise_when_the_same_response_calls_the_tool(tmp_path, monkeypatch):
    # The wording triggers the deterministic promise rule without a probe prediction.
    reader = sense.Sense(engine="python")
    monkeypatch.setattr(steer, "_reader", lambda: reader)
    monkeypatch.setattr(steer, "_probe", lambda *_: ("fine", 0.0))
    calibration_traces.build(tmp_path)
    probes = tmp_path / "train.jsonl"
    cases = agent_corpus.read_jsonl(probes)[:2]
    rows = []
    for case, uses_tool in zip(cases, (True, False), strict=True):
        calls = [[c["name"], c["arguments"]] for c in case["expected"]["calls"]] if uses_tool else []
        rows.append({
            "case": case["id"], "episode": case["episode"], "expected": case["expected"],
            "answer": "I'll list the directory.", "calls": calls, "error": None,
            "finish_reason": "tool_calls" if uses_tool else "stop", "passed": uses_tool,
        })
    replay = tmp_path / "replay.jsonl"
    sha = "a" * 64
    agent_corpus.jsonl(replay, [{
        "probes_sha256": hashlib.sha256(probes.read_bytes()).hexdigest(),
        "count": len(rows), "model": "fixture", "thinking": "off", "binding": {"artifact_sha256": sha},
    }, *rows])

    _, output = calibration_traces.join(probes, replay, sha)

    assert output[0]["passed"] and output[0]["steering"] is None
    assert not output[1]["passed"] and output[1]["steering"]["kind"] == steer.UNKEPT_PROMISE
