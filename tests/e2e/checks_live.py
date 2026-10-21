"""Live LM Studio + real-server checks for Kiwibro (task-3).

These hit the real model (`qwen3.8-2b` by default) and the real Flask app. They
are intentionally *serialized*: one model call at a time, because LM Studio is
slow when several agents talk to it at once.

Checks are written as functions taking an ``Env`` (see ``run_live.py``) and
returning a short evidence string. The runner stops at the first hard failure
only for the open-prompt checks; the rest continue so the report is complete.
"""

from __future__ import annotations

import json
import time
from typing import Any

from harness import SSEChat, assert_eq, assert_in, assert_true, expect_status

from checks_chat import chat, conversation, finish, poll, _last_user  # noqa: F401


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _wait_any(stream: SSEChat, names: tuple[str, ...], timeout: float) -> tuple[str, dict[str, Any]]:
    return stream.wait_any(names, timeout=timeout)


def _timed_chat(env: Any, message: str, *, timeout: float = 240.0, conversation_id: str | None = None):
    started = time.monotonic()
    stream = chat(env, message, conversation_id=conversation_id, timeout=timeout)
    meta = stream.wait_for("meta", timeout=timeout)
    ttft: float | None = None
    while True:
        name, data = _wait_any(stream, ("delta", "error", "done"), timeout=timeout)
        if name == "delta":
            if ttft is None:
                ttft = time.monotonic() - started
            continue
        if name == "error":
            raise AssertionError(f"LM Studio error event: {data}")
        break
    stream.wait_closed(timeout=30)
    total = time.monotonic() - started
    return stream, meta, ttft, total



def _chat_until_tool(
    env: Any,
    prompt: str,
    *,
    timeout: float = 240.0,
    attempts: int = 2,
    conversation_id: str | None = None,
) -> tuple[SSEChat, dict[str, Any]]:
    """Send a prompt until the model emits a tool_call (retry if it just answers)."""
    last_events: list[str] = []
    for attempt in range(1, attempts + 1):
        stream = chat(env, prompt, conversation_id=conversation_id, timeout=timeout)
        stream.wait_for("meta", timeout=timeout)
        try:
            name, data = stream.wait_any(("tool_call", "done", "error"), timeout=timeout)
        except TimeoutError:
            stream.close()
            raise
        if name == "error":
            raise AssertionError(f"LM Studio error event: {data}")
        if name == "tool_call":
            return stream, data
        stream.wait_closed(timeout=60)
        last_events = stream.names
        time.sleep(1.0)
    raise AssertionError(
        f"model answered without calling a tool after {attempts} attempt(s); events={last_events}"
    )

def _model_loaded(env: Any, model_id: str) -> None:
    models = env.client.get("/api/models").data["models"]
    ids = [model["id"] for model in models]
    assert_true(model_id in ids, f"model {model_id!r} is available (have {ids})")


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def live_health(env: Any) -> str:
    health = env.client.get("/health")
    expect_status(health, 200, "GET /health")
    data = health.data
    assert_eq(data["lmstudio"], True, "LM Studio reachable")
    assert_eq(data["ok"], True, "health ok")
    assert_eq(data["version"], "1.0.0", "version")
    models = env.client.get("/api/models?refresh=1")
    expect_status(models, 200, "GET /api/models")
    payload = models.data
    assert_eq(payload["lmstudio"], True, "models.lmstudio")
    assert_true(payload["models"], "at least one model listed")
    for model in payload["models"]:
        for key in ("id", "name", "type", "state", "context", "max_context", "tools"):
            assert_in(key, model, f"model field {key}")
    current = payload["current"]
    assert_true(current in {model["id"] for model in payload["models"]}, f"current model {current!r} listed")
    return f"LM Studio up, {len(payload['models'])} chat models, current={current}"


def live_plain_stream(env: Any) -> str:
    prompt = "Reply with exactly this line and nothing else: Kiwibro live check — Grüße 世界 🥝"
    stream, meta, ttft, total = _timed_chat(env, prompt)
    names = stream.names
    assert_eq(names[0], "meta", "meta first")
    assert_in("delta", names, "deltas streamed")
    assert_eq(stream.of("error"), [], "no error event")
    done = stream.of("done")[0]
    assert_eq(done["aborted"], False, "done.aborted")
    assert_true(done["model"], "done.model")
    usage = done.get("usage")
    assert_true(usage and usage["total_tokens"] > 0, f"usage reported ({usage})")
    text = stream.text()
    assert_true(len(text.strip()) > 5, f"answer text non-trivial ({text[:80]!r})")
    assert_in("Grüße", text, "real LM streamed UTF-8 umlauts intact")
    assert_in("世界", text, "real LM streamed CJK intact")
    assert_in("🥝", text, "real LM streamed emoji intact")
    assert_eq(meta["conversation_id"], env.conversation_id, "same conversation")
    return f"TTFT {ttft:.2f}s, total {total:.2f}s, {len(text)} chars, usage={usage['total_tokens']}"


def live_markdown_and_title(env: Any) -> str:
    stream, meta, ttft, total = _timed_chat(
        env, "Answer in markdown: a heading, two bullets and one fenced python snippet that prints kiwi."
    )
    text = stream.text()
    assert_in("#", text, "markdown heading streamed")
    assert_in("```", text, "code fence streamed")
    titles = stream.of("title")
    assert_eq(len(titles), 1, "one auto-title event")
    title = titles[0]["title"]
    assert_true(0 < len(title) <= 40, f"title length {len(title)}")
    detail = conversation(env, env.conversation_id)
    assert_eq(detail["conversation"]["title"], title, "title persisted")
    return f"markdown {len(text)} chars, title {title!r}, total {total:.1f}s"


def live_web_search_tool(env: Any) -> str:
    prompt = (
        "Use the web_search tool to search for recent kiwi bird conservation news. "
        "Then answer with two short bullet points, each with a markdown source link."
    )
    started = time.monotonic()
    stream, tool_call = _chat_until_tool(env, prompt, timeout=300)
    assert_eq(tool_call["name"], "web_search", "model chose web_search")
    assert_eq(tool_call["requires_approval"], False, "web_search auto-allowed")
    result = stream.wait_for("tool_result", timeout=300)
    stream.wait_done(timeout=300)
    stream.wait_closed(timeout=60)
    elapsed = time.monotonic() - started
    assert_eq(stream.of("error"), [], "no error event")
    assert_eq(result["ok"], True, f"web_search ok ({result['preview'][:200]!r})")
    preview = str(result["preview"])
    assert_true("http" in preview or preview.strip(), "search results non-empty")
    text = stream.text()
    assert_true("http" in text, "assistant cited a link")
    return f"web_search auto-run in {elapsed:.1f}s, {len(preview)} char preview, answer cites links"


def live_write_file_approval(env: Any) -> str:
    target = env.workspace / "notes" / "live-check.md"
    if target.exists():
        target.unlink()
    prompt = (
        "Use the write_file tool to create notes/live-check.md with exactly the content "
        "'kiwi live check' and then say done."
    )
    stream, call = _chat_until_tool(env, prompt, timeout=300)
    assert_eq(call["name"], "write_file", "model chose write_file")
    assert_eq(call["requires_approval"], True, "write_file asks first")
    approval = stream.wait_for("approval", timeout=300)
    assert_true(approval["approval_id"].startswith("ap_"), "approval id")
    expect_status(
        env.client.post(f"/api/approvals/{approval['approval_id']}", {"decision": "allow"}),
        200,
        "approve write",
    )
    result = stream.wait_for("tool_result", timeout=300)
    stream.wait_done(timeout=300)
    stream.wait_closed(timeout=60)
    assert_eq(result["ok"], True, "write executed after allow")
    assert_true(target.is_file(), f"{target} exists")
    assert_in("kiwi live check", target.read_text(encoding="utf-8"), "file content")
    return f"approval allow wrote {target.name} (model={stream.of('done')[0]['model']})"


def live_write_file_deny(env: Any) -> str:
    target = env.workspace / "notes" / "live-denied.md"
    if target.exists():
        target.unlink()
    prompt = (
        "Use the write_file tool to create notes/live-denied.md with the content 'must not exist'."
    )
    stream, _call = _chat_until_tool(env, prompt, timeout=300)
    approval = stream.wait_for("approval", timeout=300)
    expect_status(
        env.client.post(f"/api/approvals/{approval['approval_id']}", {"decision": "deny"}),
        200,
        "deny write",
    )
    result = stream.wait_for("tool_result", timeout=300)
    stream.wait_done(timeout=300)
    stream.wait_closed(timeout=60)
    assert_eq(result["ok"], False, "denied write not executed")
    assert_true(not target.exists(), "denied write left no file")
    return "deny blocked the write and told the model"


def live_run_command(env: Any) -> str:
    prompt = "Use the run_command tool to run `printf 'kiwi-live-command\\n'` and show me the output."
    stream, call = _chat_until_tool(env, prompt, timeout=300)
    assert_eq(call["name"], "run_command", "model chose run_command")
    assert_eq(call["risk"], "exec", "exec risk badge")
    approval = stream.wait_for("approval", timeout=300)
    assert_eq(approval["risk"], "exec", "approval risk exec")
    expect_status(
        env.client.post(f"/api/approvals/{approval['approval_id']}", {"decision": "allow"}),
        200,
        "allow command",
    )
    result = stream.wait_for("tool_result", timeout=300)
    stream.wait_done(timeout=300)
    stream.wait_closed(timeout=60)
    assert_eq(result["ok"], True, "command executed")
    text = json.dumps(stream.events, ensure_ascii=False)
    assert_true("kiwi-live-command" in text, "command output reached the stream/model")
    return "run_command ask -> allow -> output returned"


def live_blocked_command(env: Any) -> str:
    env.client.patch("/api/settings", {"permissions": {"run_command": "ask"}})
    prompt = "Use the run_command tool to run `sudo rm -rf /` and report what happened."
    stream, _call = _chat_until_tool(env, prompt, timeout=300)
    approval = stream.wait_for("approval", timeout=300)
    expect_status(
        env.client.post(f"/api/approvals/{approval['approval_id']}", {"decision": "allow"}),
        200,
        "allow blocked command",
    )
    result = stream.wait_for("tool_result", timeout=300)
    stream.wait_done(timeout=300)
    stream.wait_closed(timeout=60)
    assert_eq(result["ok"], False, "blocklisted command refused even after Allow")
    assert_in("blocked", str(result["preview"]).lower(), "blocklist message")
    return "sudo rm -rf / blocked even after approval"


def live_stop(env: Any) -> str:
    created = env.client.post("/api/conversations", {}).data
    conversation_id = created["id"]
    stream = chat(
        env,
        "Write a 600 word story about kiwi birds. Do not stop early.",
        conversation_id=conversation_id,
        timeout=300,
    )
    stream.wait_for("meta", timeout=300)
    stream.wait_for("delta", timeout=300)
    stream.wait_for("delta", timeout=60)
    expect_status(env.client.post("/api/chat/stop", {"conversation_id": conversation_id}), 200, "stop")
    done = stream.wait_done(timeout=60)
    stream.wait_closed(timeout=30)
    assert_eq(done["aborted"], True, "done.aborted after stop")
    detail = conversation(env, conversation_id)
    assert_true(
        any(message["meta"].get("aborted") for message in detail["messages"]),
        "partial answer saved with meta.aborted",
    )
    return f"stop mid-stream after {len(stream.of('delta'))} deltas, partial saved"


def live_model_swap(env: Any) -> str:
    models = env.client.get("/api/models").data
    current = models["current"]
    others = [model["id"] for model in models["models"] if model["id"] != current]
    assert_true(others, "a second model is available for a swap")
    target = others[0]
    before = conversation(env, env.conversation_id)
    memories_before = env.client.get("/api/memories").data["total"]
    expect_status(env.client.post("/api/models/select", {"id": target}), 200, "select model")
    try:
        stream, meta, ttft, total = _timed_chat(env, "Say only: swapped model online.")
        assert_eq(meta["model"], target, f"new turn uses {target}")
        after = conversation(env, env.conversation_id)
        assistant_models = [
            message["model"] for message in after["messages"] if message["role"] == "assistant"
        ]
        assert_in(current, assistant_models, "old message keeps its model")
        assert_in(target, assistant_models, "new message records the new model")
        assert_true(len(after["messages"]) >= len(before["messages"]), "transcript preserved")
        assert_eq(env.client.get("/api/memories").data["total"], memories_before, "memories unchanged")
        return f"swap {current} -> {target}, per-message models kept, memories unchanged"
    finally:
        env.client.post("/api/models/select", {"id": current})


def live_memory_roundtrip(env: Any) -> str:
    before = env.client.get("/api/memories").data["total"]
    prompt = "Remember that my QA kiwi is named Pip and likes concise answer. Use the remember tool."
    stream = chat(env, prompt, timeout=300)
    stream.wait_for("meta", timeout=300)
    # the model may answer without the tool; accept either (remember tool or extraction)
    name, _data = stream.wait_any(("tool_call", "done", "error"), timeout=300)
    if name == "error":
        raise AssertionError(f"LM Studio error event: {_data}")
    if name == "tool_call":
        stream.wait_done(timeout=300)
    stream.wait_closed(timeout=60)
    def has_memory() -> list[dict[str, Any]]:
        items = env.client.get("/api/memories").data["memories"]
        return [item for item in items if "Pip" in item["content"]]

    memories = poll(has_memory, timeout=30, what="live memory extraction")
    assert_true(memories, "a memory about Pip was stored")
    return f"memory extracted on live model ({len(memories)} match, before={before})"


def live_context_compaction(env: Any) -> str:
    """Induce compaction with real 28k-char messages (slow but contract-true)."""
    created = env.client.post("/api/conversations", {}).data
    conversation_id = created["id"]
    filler = "The kiwi is a small flightless bird of New Zealand. " * 500  # ~28k chars
    for turn in range(4):
        message = f"Context seed {turn}. Reply with only OK.\n\n{filler}"
        stream = chat(env, message, conversation_id=conversation_id, timeout=600)
        stream.wait_done(timeout=600)
        stream.wait_closed(timeout=60)
        assert_eq(stream.of("error"), [], f"seed turn {turn} no error")
        detail = conversation(env, conversation_id)
        if detail["conversation"]["compacted"]:
            break
    detail = conversation(env, conversation_id)
    assert_eq(detail["conversation"]["compacted"], True, "conversation compacted")
    assert_true(detail["conversation"]["summary"], "summary stored")
    before = len(detail["messages"])
    follow = chat(env, "Reply with only: still here.", conversation_id=conversation_id, timeout=600)
    follow.wait_done(timeout=600)
    follow.wait_closed(timeout=60)
    assert_eq(follow.of("error"), [], "no context-length error surfaced")
    after = conversation(env, conversation_id)
    assert_true(len(after["messages"]) > before, "conversation continues after compaction")
    assert_true(len(after["conversation"]["summary"]) > 0, "summary retained")
    return (
        f"compacted at turn with summary_through_seq={after['conversation']['summary_through_seq']}, "
        f"{before}->{len(after['messages'])} messages, no context error"
    )


def live_restart_persistence(env: Any) -> str:
    env.client.post("/api/memories", {"content": "QA restart persistence probe.", "kind": "fact"})
    conversation_id = env.conversation_id
    details_before = conversation(env, conversation_id)
    env.restart()
    detail = conversation(env, conversation_id)
    assert_eq(
        [message["id"] for message in detail["messages"]],
        [message["id"] for message in details_before["messages"]],
        "transcript survives restart",
    )
    memories = env.client.get("/api/memories").data["memories"]
    assert_true(any("restart persistence probe" in item["content"] for item in memories), "memory survives")
    return "conversation + memories persisted across a real server restart"


def live_offline_state(env: Any) -> str:
    """Point a second app instance at a closed LM port and verify the calm state."""
    from harness import AppServer, free_port

    closed = free_port()
    with AppServer("live-offline", lmstudio_url=f"http://127.0.0.1:{closed}") as app:
        client = app.client()
        health = client.get("/health")
        expect_status(health, 200, "offline health")
        assert_eq(health.data["lmstudio"], False, "offline health.lmstudio")
        chat_stream = client.chat({"message": "hello"})
        chat_stream.wait_done(timeout=20)
        chat_stream.wait_closed(timeout=10)
        assert_eq(chat_stream.names[0], "meta", "meta first when offline")
        errors = chat_stream.of("error")
        assert_eq(len(errors), 1, "one error event")
        assert_eq(errors[0]["code"], "connection", "connection error code")
        assert_eq(errors[0]["recoverable"], True, "recoverable connection error")
    return "offline app: health lmstudio=false, chat meta -> connection error, no crash"


LIVE_CHECKS: list[tuple[str, Any]] = [
    ("live_health", live_health),
    ("live_plain_stream", live_plain_stream),
    ("live_markdown_and_title", live_markdown_and_title),
    ("live_web_search_tool", live_web_search_tool),
    ("live_write_file_approval", live_write_file_approval),
    ("live_write_file_deny", live_write_file_deny),
    ("live_run_command", live_run_command),
    ("live_blocked_command", live_blocked_command),
    ("live_stop", live_stop),
    ("live_model_swap", live_model_swap),
    ("live_memory_roundtrip", live_memory_roundtrip),
    ("live_context_compaction", live_context_compaction),
    ("live_restart_persistence", live_restart_persistence),
    ("live_offline_state", live_offline_state),
]
