"""Efficiency/context-engine regression tests (v1.2 substrate, session 01a16089-71d5).

Covers: fast estimator parity, age-decayed tool-result caps, head+tail
truncation, whole-segment summary transcripts, usage calibration bounds,
incremental extend(), COUNT-backed memory counting, settings-version cache,
thread-local connection reuse, SSE delta coalescing order.
"""

from __future__ import annotations

import queue
from pathlib import Path
from typing import Any

import pytest

from kiwibro import create_app
from kiwibro.context import (
    TOOL_RESULT_CAP_MID,
    TOOL_RESULT_CAP_OLD,
    build_llm_messages,
    capped_tool_content,
    estimate_tokens,
    format_transcript,
    raw_message_sum,
    tool_cap_for_age,
)


@pytest.fixture()
def svc(tmp_path: Path):
    app = create_app(
        {
            "TESTING": True,
            "KIWIBRO_DATA_DIR": str(tmp_path / "data"),
            "KIWIBRO_WORKSPACE": str(tmp_path / "ws"),
        }
    )
    return app.extensions["kiwibro"]


def test_estimate_tokens_fast_path_parity():
    assert estimate_tokens("a" * 360) == 100
    assert estimate_tokens("ü" * 100) == 90
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    mixed = "hello wörld " * 50
    ascii_chars = sum(1 for ch in mixed if ord(ch) < 128)
    import math

    expected = math.ceil(ascii_chars / 3.6 + (len(mixed) - ascii_chars) * 0.9)
    assert estimate_tokens(mixed) == expected


def test_tool_cap_ages_decay():
    assert tool_cap_for_age(0) == 4000
    assert tool_cap_for_age(1) == 4000
    assert tool_cap_for_age(2) == TOOL_RESULT_CAP_MID
    assert tool_cap_for_age(6) == TOOL_RESULT_CAP_OLD
    assert tool_cap_for_age(99) == TOOL_RESULT_CAP_OLD


def test_build_messages_shrinks_old_tool_results():
    db_messages: list[dict[str, Any]] = []
    for i in range(10):
        db_messages.append({"role": "user", "content": f"q{i}", "seq": i * 2 + 1})
        db_messages.append(
            {"role": "tool", "content": "x" * 6000, "tool_call_id": f"c{i}", "seq": i * 2 + 2}
        )
    built = build_llm_messages("sys", db_messages)
    tools = [m for m in built if m["role"] == "tool"]
    assert len(tools[-1]["content"]) > 3000  # newest stays detailed
    assert len(tools[0]["content"]) <= TOOL_RESULT_CAP_OLD + 40  # oldest shrunk
    full = sum(len(m["content"]) for m in tools)
    assert full < 10 * 4000  # strictly cheaper than the old flat cap


def test_capped_tool_content_head_and_tail_for_small_caps():
    content = "HEAD" + "m" * 5000 + "TAIL"
    capped = capped_tool_content(content, 300)
    assert capped.startswith("HEAD")
    assert capped.endswith("TAIL")
    assert "truncated" in capped
    big = capped_tool_content(content, 4000)
    assert big.startswith("HEAD") and "truncated" in big


def test_format_transcript_represents_whole_segment():
    messages = [
        {"role": "user", "content": f"marker{i} " + "word " * 400, "seq": i} for i in range(30)
    ]
    transcript = format_transcript(messages, cap=6000)
    assert len(transcript) <= 6000
    assert "marker0" in transcript  # oldest no longer silently dropped
    assert "marker29" in transcript


def test_calibration_moves_toward_actual_and_clamps(svc):
    engine = svc.context
    assert engine.calibration_factor("m") == 1.0
    engine.note_usage("m", raw_sum=1000, actual_prompt_tokens=5000)  # wild overcount
    factor = engine.calibration_factor("m")
    assert 1.0 < factor <= 1.4
    engine.note_usage("m", raw_sum=10, actual_prompt_tokens=5000)  # tiny sample ignored
    assert engine.calibration_factor("m") == factor


def test_extend_appends_only_new_messages(svc):
    conv = svc.db.create_conversation(title="t")
    cid = conv["id"]
    svc.db.add_message(cid, "user", "first question here")
    prepared = svc.context.prepare(cid, "m", "sys", allow_compaction=False)
    assert prepared.last_seq == 1
    svc.db.add_message(cid, "assistant", "an answer")
    svc.db.add_message(cid, "user", "follow up question")
    extended = svc.context.extend(prepared, cid, "m", "sys")
    assert extended.last_seq == 3
    assert len(extended.messages) == len(prepared.messages) + 2
    assert extended.raw_sum == raw_message_sum(extended.messages)
    assert extended.messages[-1]["content"] == "follow up question"
    same = svc.context.extend(extended, cid, "m", "sys")
    assert same is extended


def test_count_memories_matches_list(svc):
    svc.memory.remember("User lives in Berlin", "fact")
    svc.memory.remember("User prefers dark themes everywhere", "preference")
    assert svc.db.count_memories() == len(svc.db.list_memories()) == 2
    assert svc.db.count_memories(kind="fact") == 1
    assert svc.db.count_memories(query="Berlin") == 1


def test_settings_cache_invalidates_on_write(svc):
    assert svc.settings_provider()["temperature"] == pytest.approx(0.6)
    svc.db.set_setting("temperature", 0.25)
    assert svc.settings_provider()["temperature"] == pytest.approx(0.25)
    svc.db.set_permission("write_file", "auto")
    assert svc.settings_provider()["permissions"]["write_file"] == "auto"
    # returned dicts are copies: mutating one must not poison the cache
    svc.settings_provider()["permissions"]["write_file"] = "deny"
    assert svc.settings_provider()["permissions"]["write_file"] == "auto"


def test_thread_local_connection_reused(svc):
    with svc.db.connect() as first:
        pass
    with svc.db.connect() as second:
        pass
    assert first is second


def test_sse_coalescing_preserves_order_and_text(svc):
    from kiwibro.agent import Turn

    turn = Turn(conversation_id="c", message="hi", requested_model=None)
    turn.queue = queue.Queue()
    turn.emit("meta", {"conversation_id": "c"})
    for piece in ("Hel", "lo ", "world"):
        turn.emit("delta", {"text": piece})
    turn.emit("done", {"message_id": "m"})
    turn.queue.put(None)
    frames = list(svc.agent.event_stream(turn))
    assert frames[0].startswith("event: meta")
    delta_frames = [f for f in frames if f.startswith("event: delta")]
    assert len(delta_frames) == 1
    assert "Hello world" in delta_frames[0]
    assert frames[-1].startswith("event: done")
