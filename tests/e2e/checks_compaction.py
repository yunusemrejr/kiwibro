"""Context-budget / rolling-compaction checks (tiny fake context window)."""

from __future__ import annotations

import json
from typing import Any

from harness import SSEChat, assert_eq, assert_in, assert_true


LONG = "kiwi " * 520  # ~2600 chars ~ 720 estimated tokens


def _chat(env: "Any", message: str, conversation_id: str | None = None) -> SSEChat:
    payload: dict[str, Any] = {"message": message}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    stream = env.client.chat(payload, timeout=40)
    stream.wait_done(timeout=40)
    stream.wait_closed(timeout=15)
    return stream


def _detail(env: "Any", conversation_id: str) -> dict[str, Any]:
    response = env.client.get(f"/api/conversations/{conversation_id}")
    assert_eq(response.status, 200, "GET conversation")
    return response.data


def _seed_until_compacted(env: "Any", max_turns: int = 6) -> tuple[str, list[SSEChat]]:
    created = env.client.post("/api/conversations", {}).data
    conversation_id = created["id"]
    streams: list[SSEChat] = []
    for turn in range(max_turns):
        stream = _chat(env, f"turn {turn}: {LONG} [[scenario:plain]]", conversation_id)
        assert_eq(stream.of("error"), [], f"turn {turn} must not surface an error")
        streams.append(stream)
        detail = _detail(env, conversation_id)
        if detail["conversation"]["compacted"]:
            return conversation_id, streams
    raise AssertionError("conversation never compacted despite a 2048-token context window")


def compaction_happens(env: "Any") -> str:
    conversation_id, streams = _seed_until_compacted(env)
    detail = _detail(env, conversation_id)
    conversation = detail["conversation"]
    assert_eq(conversation["compacted"], True, "conversation.compacted")
    assert_true(conversation["summary"], "rolling summary stored")
    assert_true(conversation["summary_through_seq"] > 0, "summary_through_seq advanced")
    assert_true(
        any(stream.of("context") for stream in streams),
        "a context event announced the compaction",
    )
    transcript = detail["messages"]
    assert_true(len(transcript) >= 6, "original messages are still in the transcript")
    context = detail["context"]
    assert_true(context["budget"] > 0, "context.budget")
    assert_true(0 <= context["percent"] <= 100, "context.percent clamped")
    usage = env.fake.chat_requests()
    assert_true(
        any("rolling summary" in _system(payload) for payload in env.fake.completion_requests()),
        "summariser was called through the non-streaming endpoint",
    )
    assert_true(usage, "chat requests recorded")
    return (
        f"compacted after {len(streams)} turns; summary_through_seq="
        f"{conversation['summary_through_seq']}; transcript kept {len(transcript)} messages"
    )


def compaction_keeps_chatting(env: "Any") -> str:
    conversation_id, streams = _seed_until_compacted(env)
    follow_up = _chat(env, f"after compaction {LONG} [[scenario:plain]]", conversation_id)
    assert_eq(follow_up.of("error"), [], "chat after compaction must succeed")
    assert_eq(follow_up.text(), "Hello from the fake model.", "answer streamed after compaction")
    detail = _detail(env, conversation_id)
    assert_true(len(detail["messages"]) >= 8, "transcript still growing")
    return "conversation keeps working after rolling compaction"


# ---------------------------------------------------------------------------
# These two checks encode docs/API.md promises that are easy to regress.
# ---------------------------------------------------------------------------


def summary_reaches_the_model(env: "Any") -> str:
    conversation_id, streams = _seed_until_compacted(env)
    detail = _detail(env, conversation_id)
    summary = (detail["conversation"]["summary"] or "").strip()
    assert_true(summary, "a summary exists")
    # Compaction is only useful if the summary is put in front of the model again.
    sent = [
        payload
        for payload in env.fake.chat_requests()
        if conversation_id  # all requests in this phase belong to the same fake
    ]
    last = sent[-1]
    text = json.dumps(last.get("messages") or [], ensure_ascii=False)
    assert_true(
        summary[:40] in text,
        "compaction summary is not sent back to the model: older context is silently lost",
    )
    return "compaction summary is injected into the next request's prompt"


def meta_is_first_event(env: "Any") -> str:
    conversation_id, streams = _seed_until_compacted(env)
    for index, stream in enumerate(streams):
        names = [name for name, _ in stream.events]
        if "context" not in names:
            continue
        # docs/API.md: `meta` is the first SSE event.
        assert_eq(
            names[0],
            "meta",
            f"turn {index} emitted {names[0]!r} before meta: {names[:4]}",
        )
        assert_in("context", names, "context still announced later in the stream")
        return f"turn {index} keeps meta first when compaction runs"
    raise AssertionError("no compacting turn found")


def _system(payload: dict[str, Any]) -> str:
    for message in payload.get("messages") or []:
        if message.get("role") == "system":
            return str(message.get("content") or "")
    return ""
