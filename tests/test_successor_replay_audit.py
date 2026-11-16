"""Replay diagnostics keep formatting, runtime errors and strict scoring distinct."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from scripts.mind import audit_replay, bench_successor


def test_emphasis_is_removed_from_prose_only():
    answer = r"By **4**, the *visible* total. Compare `**4**` and \*escaped\*."
    assert audit_replay.without_emphasis(answer) == (
        r"By 4, the visible total. Compare `**4**` and \*escaped\*."
    )


def test_audit_preserves_strict_outcomes_and_context_rejections(tmp_path):
    route = tmp_path / "records.jsonl"
    oracle = {"calls": [], "required": ["by 4"], "forbidden": ["GPU check passed"]}
    context = {"error": {"type": "exceed_context_size_error", "n_prompt_tokens": 17_809, "n_ctx": 8_192}}
    metadata = {"model": "test", "probes_sha256": "example", "seed": 1, "thinking": "off", "count": 4}
    rows = [
        {"case": "format", "answer": "It won by **4**.", "passed": False},
        {"case": "forbidden", "answer": "It won by 4. GPU check **passed**.", "passed": True},
        {"case": "context", "answer": "", "passed": False, "error": "HTTP 400: " + json.dumps(context)},
        {"case": "preview", "answer": "x" * 1200, "passed": False},
    ]
    defaults = {"error": None, "finish_reason": "stop", "calls": [], "expected": oracle}
    original = "\n".join(json.dumps(row) for row in [metadata, *({**defaults, **row} for row in rows)])
    route.write_text(original)

    report, _ = audit_replay.audit(route)
    assert report["strict_passed"] == report["asterisk_emphasis_compatible"] == 1
    assert report["emphasis_gains"] == ["format"]
    assert report["emphasis_losses"] == ["forbidden"]
    assert report["context_rejections"] == {"context": {"prompt_tokens": 17809, "context_tokens": 8192}}
    assert report["possibly_truncated_answers"] == ["preview"]
    assert route.read_text() == original


def test_successor_benchmark_can_return_complete_answer():
    session = MagicMock()
    response = session.post.return_value.__enter__.return_value
    response.status_code = 200
    text = "Long answer " * 115 + "Required conclusion."
    event = {"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]}
    response.iter_lines.return_value = [("data: " + json.dumps(event)).encode(), b"data: [DONE]"]
    args = SimpleNamespace(max_tokens=512, thinking="off", timeout=5)
    case = bench_successor.Case("full", "chat", [], text.replace(".", r"\."))
    prefix = ("http://127.0.0.1:1", "model", case, True, None, args)
    preview = bench_successor.complete(session, *prefix)
    full = bench_successor.complete(session, *prefix, answer_limit=None)
    assert preview["answer"] == text[:1200]
    assert full["answer"] == text
    assert preview["passed"] and full["passed"]
