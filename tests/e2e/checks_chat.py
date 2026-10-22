"""Chat / SSE / tools / approvals / security checks against the fake LM server."""

from __future__ import annotations

import contextlib
import time
from typing import Any, Iterator

from fake_lm import UTF8_REPLY
from harness import (
    SSEChat,
    assert_eq,
    assert_in,
    assert_true,
    expect_error,
    expect_status,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def chat(env: "Any", message: str, *, conversation_id: str | None = None, timeout: float = 30.0) -> SSEChat:
    payload: dict[str, Any] = {"message": message}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    stream = env.client.chat(payload, timeout=timeout)
    assert_eq(stream.response.status_code, 200, "POST /api/chat status")
    return stream


def finish(stream: SSEChat, timeout: float = 30.0) -> list[tuple[str, dict[str, Any]]]:
    stream.wait_done(timeout=timeout)
    stream.wait_closed(timeout=timeout)
    return stream.events


def poll(predicate, timeout: float = 10.0, interval: float = 0.1, what: str = "condition") -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {what}")


def conversation(env: "Any", conversation_id: str) -> dict[str, Any]:
    response = env.client.get(f"/api/conversations/{conversation_id}")
    expect_status(response, 200, "GET conversation")
    return response.data


def approve(env: "Any", approval: dict[str, Any], decision: str = "allow") -> None:
    response = env.client.post(
        f"/api/approvals/{approval['approval_id']}", {"decision": decision}
    )
    expect_status(response, 200, f"POST approval {decision}")


# ---------------------------------------------------------------------------
# streaming basics
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def autonomy(env: "Any", mode: str) -> Iterator[None]:
    """Run a check under an autonomy mode, then restore the default (smart)."""
    env.client.patch("/api/settings", {"autonomy": mode})
    try:
        yield
    finally:
        env.client.patch("/api/settings", {"autonomy": "smart"})


def plain_stream(env: "Any") -> str:
    stream = chat(env, "Say hi [[scenario:plain]]")
    events = finish(stream)
    names = [name for name, _ in events]
    assert_eq(names[0], "meta", "first SSE event is meta")
    assert_in("delta", names, "delta events")
    assert_in("done", names, "done event")
    assert_eq(names[-1], "done", "done is the final event")
    assert_eq(stream.text(), "Hello from the fake model.", "streamed text")
    meta = stream.of("meta")[0]
    assert_true(meta["conversation_id"].startswith("c_"), "meta.conversation_id")
    assert_eq(meta["model"], "qwen3.8-2b", "meta.model")
    assert_true(meta["user_message_id"].startswith("m_"), "meta.user_message_id")
    assert_in("used", meta["context"], "meta.context.used")
    done = stream.of("done")[0]
    assert_eq(done["aborted"], False, "done.aborted")
    assert_eq(done["model"], "qwen3.8-2b", "done.model")
    assert_eq(done["usage"]["completion_tokens"], 7, "usage completion tokens")
    assert_eq(done["usage"]["prompt_tokens"], 42, "usage prompt tokens")

    detail = conversation(env, meta["conversation_id"])
    roles = [message["role"] for message in detail["messages"]]
    assert_eq(roles, ["user", "assistant"], "persisted user + assistant messages")
    assistant = detail["messages"][1]
    assert_eq(assistant["content"], "Hello from the fake model.", "assistant content persisted")
    assert_eq(assistant["model"], "qwen3.8-2b", "assistant model stored")
    assert_eq(detail["conversation"]["model"], "qwen3.8-2b", "conversation model stored")
    return "meta->status->delta*->title->done; transcript persisted"


def utf8_stream(env: "Any") -> str:
    """Regression: LM Studio SSE has no charset; characters must not turn to mojibake."""
    stream = chat(env, "[[scenario:utf8]]")
    finish(stream)
    text = stream.text()
    assert_eq(text, UTF8_REPLY, "UTF-8 streamed text is byte-clean")
    return "non-ASCII (umlauts, CJK, emoji) survives streaming"


def markdown_stream(env: "Any") -> str:
    stream = chat(env, "[[scenario:markdown]]")
    finish(stream)
    text = stream.text()
    assert_in("```python", text, "code fence in deltas")
    assert_in("**bold**", text, "bold markdown in deltas")
    assert_in("[link](https://example.com/kiwi)", text, "link markdown in deltas")
    return f"{len(text)} chars of markdown streamed intact"


def reasoning_stream(env: "Any") -> str:
    stream = chat(env, "[[scenario:reasoning]]")
    finish(stream)
    assert_true(stream.of("reasoning"), "reasoning events emitted")
    assert_in("Let me think", "".join(r.get("text", "") for r in stream.of("reasoning")), "reasoning text")
    assert_eq(stream.text(), "The kiwi is a bird.", "answer text after reasoning")
    meta = stream.of("meta")[0]
    detail = conversation(env, meta["conversation_id"])
    assistant = next(message for message in detail["messages"] if message["role"] == "assistant")
    assert_in("Let me think", assistant["reasoning"], "reasoning persisted on the message")
    return "reasoning streamed and persisted separately from answer"


def no_usage_stream(env: "Any") -> str:
    stream = chat(env, "[[scenario:no_usage]]")
    finish(stream)
    done = stream.of("done")[0]
    assert_eq(done["usage"], None, "done.usage is null when LM reports none")
    return "graceful when the model reports no usage"


def empty_stream(env: "Any") -> str:
    stream = chat(env, "[[scenario:empty]]")
    finish(stream)
    assert_eq(stream.of("delta"), [], "no delta events")
    done = stream.of("done")[0]
    meta = stream.of("meta")[0]
    assert_eq(done["message_id"], meta["user_message_id"], "done falls back to the user message id")
    return "empty model output terminates cleanly"


# ---------------------------------------------------------------------------
# titles and memory
# ---------------------------------------------------------------------------


def title_event(env: "Any") -> str:
    stream = chat(env, "Tell me about kiwi birds [[scenario:plain]]")
    finish(stream)
    titles = stream.of("title")
    assert_eq(len(titles), 1, "one title event")
    title = titles[0]["title"]
    assert_true(0 < len(title) <= 40, f"title length ({title!r})")
    assert_eq(titles[0]["id"], stream.of("meta")[0]["conversation_id"], "title id")
    detail = conversation(env, titles[0]["id"])
    assert_eq(detail["conversation"]["title"], title, "title persisted")
    return f"auto-title {title!r} emitted before done"


def title_echo_rejected(env: "Any") -> str:
    """Models sometimes echo the prompt; the title must not contain prompt scaffolding."""
    stream = chat(env, "Fix the CSV export [[scenario:title_echo]]")
    finish(stream)
    titles = stream.of("title")
    assert_true(titles, "a title event is still emitted")
    title = titles[0]["title"]
    assert_true(len(title) <= 40, f"title length ({title!r})")
    lowered = title.lower()
    for bad in ("first user message", "write the title", "you name conversations", "here is"):
        assert_true(bad not in lowered, f"title must not echo prompt text: {title!r}")
    return f"prompt echo rejected, deterministic title {title!r}"


def memory_extraction(env: "Any") -> str:
    env.client.delete("/api/memories")
    stream = chat(env, "I prefer brief German answers [[scenario:plain]]")
    finish(stream)
    # memory extraction is allowed to run after `done`; poll the REST surface.
    def extracted():
        items = env.client.get("/api/memories").data["memories"]
        return [item for item in items if "Grüße" in item["content"]]

    memories = poll(extracted, timeout=12, what="memory extraction")
    assert_true(memories, "memory extracted after the turn")
    content = memories[0]["content"]
    assert_true("Grüße 🥝" in content, f"UTF-8 memory round-trip ({content!r})")
    assert_eq(memories[0]["kind"], "preference", "memory kind from model")
    has_event = any(name == "memory" for name, _ in stream.events)
    return (
        "memory extracted (async, after done) and visible via REST"
        + ("; optional memory event emitted" if has_event else "; no memory event (docs: optional, UI polls)")
    )


# ---------------------------------------------------------------------------
# approvals & tools
# ---------------------------------------------------------------------------


def _tool_write_allow(env: "Any") -> str:
    stream = chat(env, "Please save a note [[scenario:tool_write]]")
    meta = stream.wait_for("meta", timeout=20)
    tool_call = stream.wait_for("tool_call", timeout=20)
    assert_eq(tool_call["name"], "write_file", "tool name")
    assert_eq(tool_call["requires_approval"], True, "write needs approval")
    assert_eq(tool_call["risk"], "write", "risk badge")
    approval = stream.wait_for("approval", timeout=20)
    assert_eq(approval["tool"], "write_file", "approval tool")
    assert_true(approval["approval_id"].startswith("ap_"), "approval id prefix")
    assert_eq(approval["timeout_s"], 120, "approval timeout_s")
    assert_in("notes/hello.txt", approval["summary"], "approval summary includes target")

    approve(env, approval, "allow")
    result = stream.wait_for("tool_result", timeout=30)
    assert_eq(result["ok"], True, f"tool executed ({result})")
    assert_true(result["duration_ms"] >= 0, "duration reported")
    assert_true(len(result["preview"]) <= 400, "tool_result.preview <= 400 chars")
    finish(stream)
    assert_eq(stream.names[-1], "done", "done after tool round")
    assert_true(stream.text(), "final assistant text after tool result")

    path = env.workspace / "notes" / "hello.txt"
    assert_true(path.is_file(), "file written inside the workspace")
    assert_in("Kiwi says hi", path.read_text(encoding="utf-8"), "file content")

    detail = conversation(env, meta["conversation_id"])
    roles = [message["role"] for message in detail["messages"]]
    assert_in("tool", roles, "tool message persisted")
    assert_eq(roles[-1], "assistant", "final assistant reply persisted")
    # an approval can only be resolved once
    again = env.client.post(
        f"/api/approvals/{approval['approval_id']}", {"decision": "allow"}
    )
    expect_error(again, 404, "not_found", "approval cannot be resolved twice")
    return "tool_call -> approval -> tool_result -> final answer; file on disk"


def tool_write_allow(env: "Any") -> str:
    with autonomy(env, "supervised"):  # the per-call approval flow under test
        return _tool_write_allow(env)


def _tool_write_deny(env: "Any") -> str:
    stream = chat(env, "Write it anyway [[scenario:tool_write_deny]]")
    stream.wait_for("meta", timeout=20)
    stream.wait_for("tool_call", timeout=20)
    approval = stream.wait_for("approval", timeout=20)
    approve(env, approval, "deny")
    result = stream.wait_for("tool_result", timeout=20)
    assert_eq(result["ok"], False, "denied tool is not executed")
    assert_in("Denied", result["preview"], "preview says denied")
    finish(stream)
    assert_true(not (env.workspace / "denied.txt").exists(), "denied write did not touch disk")
    return "deny blocks execution and reports it to the model"


def tool_write_deny(env: "Any") -> str:
    with autonomy(env, "supervised"):  # the per-call approval flow under test
        return _tool_write_deny(env)


def _tool_write_always(env: "Any") -> str:
    try:
        stream = chat(env, "Two files please [[scenario:tool_always_flow]]")
        stream.wait_for("meta", timeout=20)
        first = stream.wait_for("approval", timeout=20)
        approve(env, first, "always")
        stream.wait_for("tool_result", timeout=30)
        second_call = stream.wait_for("tool_call", timeout=30)
        assert_eq(second_call["requires_approval"], False, "`always` auto-approves the same tool")
        result = stream.wait_for("tool_result", timeout=30)
        assert_eq(result["ok"], True, "second write executed")
        finish(stream)
        assert_eq(len(stream.of("approval")), 1, "only one approval card for two writes")
        assert_true((env.workspace / "always-1.txt").is_file(), "first file written")
        assert_true((env.workspace / "always-2.txt").is_file(), "second file written")
        settings = env.client.get("/api/settings").data["settings"]
        assert_eq(settings["permissions"]["write_file"], "auto", "always persisted as auto permission")

        # a brand new conversation must not ask again
        other = chat(env, "One more file [[scenario:tool_write]]")
        other.wait_for("meta", timeout=20)
        call = other.wait_for("tool_call", timeout=20)
        assert_eq(call["requires_approval"], False, "permission persists across conversations")
        assert_eq(other.of("approval"), [], "no new approval card")
        finish(other)
        return "always upgrades to auto, persists across turns and conversations"
    finally:
        env.client.patch("/api/settings", {"permissions": {"write_file": "ask"}})


def tool_write_always(env: "Any") -> str:
    with autonomy(env, "supervised"):  # the per-call approval flow under test
        return _tool_write_always(env)


def tool_errors(env: "Any") -> str:
    # unknown tool
    stream = chat(env, "Use something odd [[scenario:tool_unknown]]")
    stream.wait_for("meta", timeout=20)
    call = stream.wait_for("tool_call", timeout=20)
    assert_eq(call["requires_approval"], False, "unknown tool cannot ask for approval")
    result = stream.wait_for("tool_result", timeout=20)
    assert_eq(result["ok"], False, "unknown tool fails cleanly")
    finish(stream)

    # bad arguments
    stream = chat(env, "Search for nothing [[scenario:tool_bad_args]]")
    stream.wait_for("meta", timeout=20)
    stream.wait_for("tool_call", timeout=20)
    result = stream.wait_for("tool_result", timeout=20)
    assert_eq(result["ok"], False, "invalid arguments fail cleanly")
    assert_in("query", str(result["preview"]).lower(), "validation message points at the argument")
    finish(stream)
    return "unknown tool and invalid args become tool_result errors, not crashes"


def tool_remember(env: "Any") -> str:
    env.client.delete("/api/memories")
    # The fact must be in the user's words: the tool refuses memories nobody stated.
    stream = chat(env, "Remember: the QA kiwi is named Pip and likes concise answers [[scenario:tool_remember]]")
    stream.wait_for("meta", timeout=20)
    call = stream.wait_for("tool_call", timeout=20)
    assert_eq(call["name"], "remember", "remember tool")
    assert_eq(call["requires_approval"], False, "remember is auto-allowed")
    result = stream.wait_for("tool_result", timeout=20)
    assert_eq(result["ok"], True, "remember executed")
    finish(stream)
    memories = env.client.get("/api/memories").data["memories"]
    assert_true(any("Pip" in memory["content"] for memory in memories), "remember tool wrote memory")
    return "remember tool writes memory immediately (auto permission)"


# ---------------------------------------------------------------------------
# security probes through the real tool pipeline
# ---------------------------------------------------------------------------


def _security_write_traversal(env: "Any") -> str:
    stream = chat(env, "Escape the sandbox [[scenario:tool_write_traversal]]")
    stream.wait_for("meta", timeout=20)
    call = stream.wait_for("tool_call", timeout=20)
    assert_eq(call["requires_approval"], False, "a call that cannot run is never offered for approval")
    result = stream.wait_for("tool_result", timeout=20)
    finish(stream)
    assert_eq(stream.of("approval"), [], "no approval card for a path outside the workspace")
    assert_eq(result["ok"], False, "traversal write refused")
    assert_in("outside the workspace", str(result["preview"]).lower(), "sandbox error text")
    escaped = env.app.run_dir / "pwned.txt"
    assert_true(not escaped.exists(), f"nothing written outside the workspace ({escaped})")
    return "write_file '../../pwned.txt' refused before any approval"


def security_write_traversal(env: "Any") -> str:
    with autonomy(env, "supervised"):  # the per-call approval flow under test
        return _security_write_traversal(env)


def security_read_traversal(env: "Any") -> str:
    stream = chat(env, "Read passwd [[scenario:tool_read_traversal]]")
    stream.wait_for("meta", timeout=20)
    call = stream.wait_for("tool_call", timeout=20)
    assert_eq(call["requires_approval"], False, "read_file is auto")
    result = stream.wait_for("tool_result", timeout=20)
    finish(stream)
    assert_eq(result["ok"], False, "traversal read refused")
    assert_true("root:" not in str(result["preview"]), "no /etc/passwd content leaked")
    return "read_file '../../../etc/passwd' refused without leaking content"


def security_ssrf(env: "Any") -> str:
    started = time.monotonic()
    stream = chat(env, "Fetch metadata [[scenario:tool_fetch_ssrf]]")
    stream.wait_for("meta", timeout=20)
    call = stream.wait_for("tool_call", timeout=20)
    assert_eq(call["requires_approval"], False, "fetch_page is auto")
    result = stream.wait_for("tool_result", timeout=20)
    finish(stream)
    elapsed = time.monotonic() - started
    assert_eq(result["ok"], False, "link-local fetch refused")
    assert_in("local or private", str(result["preview"]).lower(), "SSRF guard message")
    assert_true(elapsed < 8, f"SSRF guard refuses before any network wait ({elapsed:.1f}s)")
    return "fetch_page 169.254.169.254 refused by the SSRF guard"


def security_command_blocklist(env: "Any") -> str:
    stream = chat(env, "Run something [[scenario:tool_run_blocklist]]")
    stream.wait_for("meta", timeout=20)
    call = stream.wait_for("tool_call", timeout=20)
    assert_eq(call.get("level"), "critical", "command graded critical")
    assert_eq(call["requires_approval"], False, "critical commands are never offered for approval")
    result = stream.wait_for("tool_result", timeout=20)
    finish(stream)
    assert_eq(stream.of("approval"), [], "no approval card for a blocked command")
    assert_eq(result["ok"], False, "blocklisted command refused")
    assert_in("blocked", str(result["preview"]).lower(), "blocked message")
    return "run_command 'sudo rm -rf /' denied up front by the safety policy (no approval card)"


def smart_autonomy_write(env: "Any") -> str:
    stream = chat(env, "Save it [[scenario:tool_write]]")
    stream.wait_for("meta", timeout=20)
    call = stream.wait_for("tool_call", timeout=20)
    assert_eq(call["requires_approval"], False, "smart autonomy runs a workspace write without asking")
    assert_in(call.get("level"), ("low", "medium"), "risk level reported on tool_call")
    assert_true(isinstance(call.get("reasons"), list), "risk reasons reported on tool_call")
    result = stream.wait_for("tool_result", timeout=30)
    finish(stream)
    assert_eq(stream.of("approval"), [], "no approval card")
    assert_eq(result["ok"], True, "write executed")
    assert_true((env.workspace / "notes" / "hello.txt").is_file(), "file written")
    return f"smart mode auto-ran a {call.get('level')}-risk workspace write with reasons attached"


# ---------------------------------------------------------------------------
# busy / abort / disconnect
# ---------------------------------------------------------------------------


def busy_409_and_stop(env: "Any") -> str:
    created = env.client.post("/api/conversations", {}).data
    conversation_id = created["id"]
    stream = chat(env, "Take your time [[scenario:stop_slow]]", conversation_id=conversation_id)
    stream.wait_for("meta", timeout=20)
    stream.wait_for("delta", timeout=20)

    second = env.client.post(
        "/api/chat", {"conversation_id": conversation_id, "message": "second request"}
    )
    expect_error(second, 409, "busy", "second turn on the same conversation")
    expect_status(env.client.post("/api/chat/stop", {"conversation_id": conversation_id}), 200, "stop")
    done = stream.wait_done(timeout=20)
    stream.wait_closed(timeout=10)
    assert_eq(done["aborted"], True, "done.aborted after stop")
    detail = conversation(env, conversation_id)
    aborted = [message for message in detail["messages"] if message["meta"].get("aborted")]
    assert_true(aborted, "partial assistant text saved with meta.aborted")

    # the lock must be released: a new turn is accepted
    retry = chat(env, "fresh start [[scenario:plain]]", conversation_id=conversation_id)
    finish(retry)
    return "409 busy while streaming; stop aborts; partial saved; lock released"


def client_disconnect_aborts(env: "Any") -> str:
    created = env.client.post("/api/conversations", {}).data
    conversation_id = created["id"]
    stream = chat(env, "disconnect me [[scenario:stop_slow]]", conversation_id=conversation_id)
    stream.wait_for("meta", timeout=20)
    stream.wait_for("delta", timeout=20)
    stream.close()

    def aborted_saved() -> bool:
        detail = conversation(env, conversation_id)
        return any(message["meta"].get("aborted") for message in detail["messages"])

    poll(aborted_saved, timeout=15, what="server-side abort after client disconnect")
    retry = chat(env, "back again [[scenario:plain]]", conversation_id=conversation_id)
    finish(retry)
    return "disconnect aborts the turn, saves partial text, frees the conversation"


# ---------------------------------------------------------------------------
# model behaviour / recovery
# ---------------------------------------------------------------------------


def context_limit_recovery(env: "Any") -> str:
    created = env.client.post("/api/conversations", {}).data
    conversation_id = created["id"]
    first = chat(env, "first message [[scenario:plain]]", conversation_id=conversation_id)
    finish(first)
    env.fake.reset()
    stream = chat(
        env, "a long question [[scenario:context_400_once]]", conversation_id=conversation_id
    )
    finish(stream)
    errors = stream.of("error")
    assert_eq(errors, [], "context-limit 400 recovered without surfacing an error")
    assert_eq(stream.text(), "Recovered after context compaction.", "retry answer streamed")
    conversation_id = stream.of("meta")[0]["conversation_id"]
    detail = conversation(env, conversation_id)
    assert_true(
        detail["conversation"]["summary_through_seq"] > 0,
        "aggressive recovery advanced the summary boundary",
    )
    announced = bool(stream.of("context"))
    summary_calls = [
        payload
        for payload in env.fake.completion_requests()
        if "rolling summary" in _first_system(payload)
    ]
    assert_true(summary_calls, "summariser ran before the retry")
    attempts = [payload for payload in env.fake.chat_requests() if "context_400_once" in _last_user(payload)]
    assert_eq(len(attempts), 2, "one failed attempt + one retry")
    return (
        "HTTP 400 context-limit -> summarise -> retry succeeds"
        + ("" if announced else "; recovery did not emit a `context` event (UI chip missing)")
    )


def context_limit_unrecoverable(env: "Any") -> str:
    created = env.client.post("/api/conversations", {}).data
    conversation_id = created["id"]
    first = chat(env, "warm up [[scenario:plain]]", conversation_id=conversation_id)
    finish(first)
    stream = chat(
        env, "too much [[scenario:context_400_always]]", conversation_id=conversation_id
    )
    finish(stream)
    errors = stream.of("error")
    assert_eq(len(errors), 1, "one terminal error event")
    assert_eq(errors[0]["code"], "context", "error code context")
    assert_eq(errors[0]["recoverable"], False, "context errors are not blindly retryable")
    assert_true("context" in errors[0]["message"].lower(), "calm error message")
    # conversation is not wedged
    retry = chat(env, "still works [[scenario:plain]]", conversation_id=conversation_id)
    finish(retry)
    return "persistent context failures surface one error event, conversation still usable"


def tools_unsupported_retry(env: "Any") -> str:
    stream = chat(env, "no tools here [[scenario:tools_reject_once]]")
    finish(stream)
    assert_eq(stream.text(), "Text answer without tools.", "answer streamed after retry without tools")
    assert_eq(stream.of("error"), [], "no error surfaced")
    attempts = [
        payload for payload in env.fake.chat_requests() if "tools_reject_once" in _last_user(payload)
    ]
    assert_eq(len(attempts), 2, "two attempts: with tools then without")
    assert_true(attempts[0].get("tools"), "first attempt sends tools")
    assert_true(not attempts[1].get("tools"), "retry drops tools")
    return "400 'tools unsupported' retried once without tools"


def model_swap_preserves_state(env: "Any") -> str:
    memories_before = env.client.get("/api/memories").data["total"]
    created_conversation = env.client.post("/api/conversations", {}).data
    conversation_id = created_conversation["id"]
    first = chat(env, "hello qwen [[scenario:plain]]", conversation_id=conversation_id)
    finish(first)
    expect_status(
        env.client.post("/api/models/select", {"id": "spark-x2.6-4b"}), 200, "switch model"
    )
    try:
        second = chat(env, "hello spark [[scenario:plain]]", conversation_id=conversation_id)
        finish(second)
        assert_eq(second.of("meta")[0]["model"], "spark-x2.6-4b", "new turn uses the new model")
        assert_eq(env.fake.last_chat_request().get("model"), "spark-x2.6-4b", "LM request model")
        detail = conversation(env, conversation_id)
        assistant_models = [
            message["model"] for message in detail["messages"] if message["role"] == "assistant"
        ]
        assert_eq(assistant_models[0], "qwen3.8-2b", "old message keeps its model")
        assert_eq(assistant_models[1], "spark-x2.6-4b", "new message records the new model")
        assert_eq(
            env.client.get("/api/memories").data["total"], memories_before, "memories unchanged by swap"
        )
        return "mid-chat model swap: transcript per-message models preserved, memories unchanged"
    finally:
        env.client.post("/api/models/select", {"id": "qwen3.8-2b"})


# ---------------------------------------------------------------------------
# model capability edge cases
# ---------------------------------------------------------------------------


def no_tools_model(env: "Any") -> str:
    """A model without tool_use must be called without tools (no 400 loop)."""
    stream = chat(env, "plain answer please [[scenario:plain]]")
    finish(stream)
    assert_eq(stream.text(), "Hello from the fake model.", "answer streamed")
    assert_eq(env.fake.last_request_has_tools(), False, "no tools sent to a non-tool model")
    return "tools omitted for models lacking tool_use capability"


# ---------------------------------------------------------------------------
# internals used by assertions
# ---------------------------------------------------------------------------


def _first_system(payload: dict[str, Any]) -> str:
    for message in payload.get("messages") or []:
        if message.get("role") == "system":
            return str(message.get("content") or "")
    return ""


def _last_user(payload: dict[str, Any]) -> str:
    for message in reversed(payload.get("messages") or []):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""
