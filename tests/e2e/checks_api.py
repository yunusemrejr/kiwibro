"""Black-box API checks that do not require a live LM Studio.

These run against a real ``python3 app.py`` child process. Phase "offline"
points the app at a closed port to exercise the LM-offline contract; phase
"api" points it at the fake LM server from ``fake_lm.py``.
"""

from __future__ import annotations

import time
from typing import Any

from harness import (
    assert_eq,
    assert_in,
    assert_true,
    error_code,
    expect_error,
    expect_status,
)


# ---------------------------------------------------------------------------
# health / bootstrap / models
# ---------------------------------------------------------------------------


def health_offline(env: "Any") -> str:
    response = env.client.get("/health")
    expect_status(response, 200, "GET /health")
    data = response.data or {}
    assert_eq(data.get("ok"), True, "health.ok")
    assert_eq(data.get("lmstudio"), False, "health.lmstudio offline")
    assert_eq(data.get("version"), "2.0.0", "health.version")
    assert_true(data.get("lmstudio_error"), "health.lmstudio_error present")
    assert_eq(response.headers.get("cache-control"), "no-store", "health cache-control")
    return "lmstudio=false with calm error message"


def bootstrap_offline(env: "Any") -> str:
    response = env.client.get("/api/bootstrap")
    expect_status(response, 200, "GET /api/bootstrap")
    data = response.data or {}
    assert_eq(data.get("health", {}).get("lmstudio"), False, "bootstrap.health.lmstudio")
    assert_eq(data.get("models"), [], "bootstrap.models empty when offline")
    assert_eq(data.get("conversations"), [], "bootstrap.conversations")
    assert_eq(data.get("memories_count"), 0, "bootstrap.memories_count")
    assert_eq(data.get("current_model"), None, "bootstrap.current_model")
    workspace = data.get("workspace")
    assert_true(workspace and workspace.startswith(str(env.app.workspace)), "bootstrap.workspace")
    assert_in("permissions", data.get("settings") or {}, "bootstrap.settings.permissions")
    return "200 with models=[] and workspace set"


def models_offline(env: "Any") -> str:
    response = env.client.get("/api/models")
    expect_status(response, 200, "GET /api/models")
    data = response.data or {}
    assert_eq(data.get("ok"), True, "models.ok")
    assert_eq(data.get("lmstudio"), False, "models.lmstudio")
    assert_eq(data.get("models"), [], "models.models")
    return "200, lmstudio=false, models=[]"


def chat_offline_is_graceful(env: "Any") -> str:
    started = time.monotonic()
    chat = env.client.chat({"message": "hello there"})
    assert_eq(chat.response.status_code, 200, "POST /api/chat offline status")
    chat.wait_done(timeout=20)
    chat.wait_closed(timeout=10)
    elapsed = time.monotonic() - started
    assert_true(elapsed < 10, f"offline chat should fail fast (took {elapsed:.1f}s)")
    names = chat.names
    assert_true(names and names[0] == "meta", f"first SSE event must be meta, got {names}")
    errors = chat.of("error")
    assert_eq(len(errors), 1, "exactly one error event")
    error = errors[0]
    assert_eq(error.get("code"), "connection", "error.code")
    assert_eq(error.get("recoverable"), True, "error.recoverable")
    assert_true("LM Studio" in str(error.get("message")), "error message mentions LM Studio")
    conversation_id = chat.of("meta")[0]["conversation_id"]
    conversation = env.client.get(f"/api/conversations/{conversation_id}")
    expect_status(conversation, 200, "GET conversation after offline chat")
    roles = [message["role"] for message in conversation.data["messages"]]
    assert_eq(roles, ["user"], "user message is persisted, no assistant message")
    return f"meta -> error(connection, recoverable) in {elapsed:.2f}s"


def models_select_validation(env: "Any") -> str:
    models = env.client.get("/api/models").data
    assert_eq(len(models.get("models") or []), 3, "fake exposes 3 chat models")
    ids = [model["id"] for model in models["models"]]
    assert_in("qwen3.8-2b", ids, "qwen listed")
    assert_true("embedding-kiwi" not in ids, "embedding model is excluded")
    tiny = next(model for model in models["models"] if model["id"] == "tiny-context")
    assert_eq(tiny.get("context"), 2048, "loaded model context")
    not_loaded = next(model for model in models["models"] if model["id"] == "spark-x2.6-4b")
    assert_eq(not_loaded.get("context"), 32768, "unloaded model context = min(max,32768)")
    expect_error(
        env.client.post("/api/models/select", {"id": "does-not-exist"}),
        400,
        "unknown_model",
        "select unknown model",
    )
    expect_error(env.client.post("/api/models/select", {}), 400, "bad_request", "select without id")
    selected = env.client.post("/api/models/select", {"id": "spark-x2.6-4b"})
    expect_status(selected, 200, "select valid model")
    assert_eq(selected.data.get("current"), "spark-x2.6-4b", "selected model echoed")
    health = env.client.get("/health").data
    assert_eq(health.get("model"), "spark-x2.6-4b", "health reports selected model")
    # restore the default for later checks
    env.client.post("/api/models/select", {"id": "qwen3.8-2b"})
    return "list excludes embeddings; unknown id -> 400; selection persists"


# ---------------------------------------------------------------------------
# conversations
# ---------------------------------------------------------------------------


def conversations_crud(env: "Any") -> str:
    client = env.client
    created = client.post("/api/conversations", {"title": "QA conversation"})
    expect_status(created, 201, "POST /api/conversations")
    conversation = created.data
    assert_eq(conversation["title"], "QA conversation", "created title")
    assert_true(conversation["id"].startswith("c_"), "conversation id prefix")
    conversation_id = conversation["id"]

    listing = client.get("/api/conversations").data
    assert_true(any(c["id"] == conversation_id for c in listing["conversations"]), "conversation listed")

    patched = client.patch(f"/api/conversations/{conversation_id}", {"title": "Renamed", "pinned": True})
    expect_status(patched, 200, "PATCH conversation")
    assert_eq(patched.data["conversation"]["title"], "Renamed", "patched title")
    assert_eq(patched.data["conversation"]["pinned"], True, "patched pinned")
    assert_eq(patched.data["conversation"]["title_auto"], False, "manual title disables auto-title")

    first = client.get("/api/conversations").data["conversations"][0]
    assert_eq(first["id"], conversation_id, "pinned conversation sorts first")

    for payload, code, what in (
        ({"title": ""}, "bad_request", "empty title"),
        ({"pinned": "yes"}, "bad_request", "non-bool pinned"),
        ({}, "bad_request", "empty patch"),
    ):
        expect_error(client.patch(f"/api/conversations/{conversation_id}", payload), 400, code, what)

    cleared = client.post(f"/api/conversations/{conversation_id}/clear")
    expect_status(cleared, 200, "clear conversation")
    detail = client.get(f"/api/conversations/{conversation_id}").data
    assert_eq(detail["messages"], [], "clear removed messages")
    assert_eq(detail["conversation"]["summary"], None, "clear removed summary")

    expect_error(client.get("/api/conversations/c_missing"), 404, "not_found", "unknown conversation")
    expect_error(client.delete("/api/conversations/c_missing"), 404, "not_found", "delete unknown")
    accept = client.delete(f"/api/conversations/{conversation_id}")
    expect_status(accept, 200, "delete conversation")
    expect_error(client.get(f"/api/conversations/{conversation_id}"), 404, "not_found", "deleted is gone")
    return "create/list/patch/pin/clear/delete + validation"


def conversation_cascade(env: "Any") -> str:
    """Deleting a conversation removes its messages (cascade) but not memories."""
    client = env.client
    memory = client.post("/api/memories", {"content": "Survives conversation deletion.", "kind": "fact"})
    expect_status(memory, 200, "create memory")
    conversation = client.post("/api/conversations", {}).data
    conv_id = conversation["id"]
    chat = client.chat({"conversation_id": conv_id, "message": "note it [[scenario:plain]]"}, timeout=30)
    chat.wait_done(25)
    chat.wait_closed(10)
    detail = client.get(f"/api/conversations/{conv_id}").data
    assert_true(len(detail["messages"]) >= 2, "chat persisted user+assistant")
    expect_status(client.delete(f"/api/conversations/{conv_id}"), 200, "delete conversation")
    memories = client.get("/api/memories").data["memories"]
    assert_true(any(item["id"] == memory.data["memory"]["id"] for item in memories), "memory survived")
    return "messages cascaded, memories untouched"


# ---------------------------------------------------------------------------
# memories
# ---------------------------------------------------------------------------


def memories_crud(env: "Any") -> str:
    client = env.client
    expect_status(client.delete("/api/memories"), 200, "reset memories")

    created = client.post("/api/memories", {"content": "Prefers dark themes and kiwi green.", "kind": "preference"})
    expect_status(created, 200, "POST /api/memories")
    memory = created.data["memory"]
    assert_true(memory["id"].startswith("mem_"), "memory id prefix")
    assert_eq(memory["kind"], "preference", "memory kind")
    assert_eq(memory["pinned"], False, "memory not pinned")

    duplicate = client.post("/api/memories", {"content": "prefers dark themes and kiwi green!", "kind": "preference"})
    expect_status(duplicate, 200, "POST duplicate memory")
    assert_eq(duplicate.data["memory"]["id"], memory["id"], "near-duplicate merged into existing memory")
    total = client.get("/api/memories").data["total"]
    assert_eq(total, 1, "dedupe kept one memory")

    utf8 = client.post("/api/memories", {"content": "Heißt Jürgen und mag Grüße 🥝", "kind": "person"})
    expect_status(utf8, 200, "UTF-8 memory")
    assert_true("Grüße 🥝" in utf8.data["memory"]["content"], "UTF-8 memory round-trips")

    pinned = client.patch(
        f"/api/memories/{memory['id']}",
        {"content": "Prefers dark themes, kiwi green and compact UI.", "pinned": True},
    )
    expect_status(pinned, 200, "PATCH memory")
    assert_eq(pinned.data["memory"]["pinned"], True, "memory pinned")
    assert_eq(
        client.get("/api/memories").data["memories"][0]["id"], memory["id"], "pinned memory sorts first"
    )

    search = client.get("/api/memories?q=kiwi%20green")
    expect_status(search, 200, "GET memories?q=")
    assert_true(any(item["id"] == memory["id"] for item in search.data["memories"]), "FTS search finds memory")
    filtered = client.get("/api/memories?kind=person")
    assert_eq(filtered.data["total"], 1, "kind filter")
    assert_eq(filtered.data["memories"][0]["kind"], "person", "kind filter content")

    # hostile search text must not 500
    for hostile in ("!!!", "kiwi AND OR NOT", '"quoted"', "'; DROP TABLE memories; --"):
        response = client.request("GET", "/api/memories?q=" + requests_quote(hostile))
        expect_status(response, 200, f"hostile search {hostile!r}")

    for payload, code, what in (
        ({"content": "x"}, "bad_request", "too short content"),
        ({"content": "valid content", "kind": "nonsense"}, "bad_request", "unknown kind"),
    ):
        expect_error(client.post("/api/memories", payload), 400, code, what)
    expect_error(client.patch("/api/memories/mem_missing", {"content": "abc"}), 404, "not_found", "patch unknown")
    expect_error(client.delete("/api/memories/mem_missing"), 404, "not_found", "delete unknown")

    deleted = client.delete("/api/memories")
    expect_status(deleted, 200, "DELETE all memories")
    assert_true(deleted.data["deleted"] >= 2, "deleted count")
    assert_eq(client.get("/api/memories").data["total"], 0, "memories empty")
    return "CRUD + dedupe + FTS + kind filter + hostile queries"


def requests_quote(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


# ---------------------------------------------------------------------------
# settings & workspace
# ---------------------------------------------------------------------------


def settings_validation(env: "Any") -> str:
    client = env.client
    settings = client.get("/api/settings").data["settings"]
    for key in (
        "model",
        "lmstudio_url",
        "agent_enabled",
        "memory_enabled",
        "max_output_tokens",
        "temperature",
        "workspace",
        "user_name",
        "permissions",
    ):
        assert_in(key, settings, f"settings.{key}")
    assert_eq(settings["permissions"]["write_file"], "ask", "default write_file permission")
    assert_eq(settings["permissions"]["read_file"], "auto", "default read_file permission")

    invalid_bodies = [
        ({"temperature": 3}, "temperature range"),
        ({"temperature": "hot"}, "temperature type"),
        ({"max_output_tokens": 100}, "max_output_tokens range"),
        ({"max_output_tokens": 4096.5}, "max_output_tokens type"),
        ({"permissions": {"write_file": "sometimes"}}, "permission mode"),
        ({"permissions": {"format_disk": "auto"}}, "unknown permission tool"),
        ({"workspace": "/"}, "workspace root"),
        ({"lmstudio_url": "ftp://example.com"}, "lmstudio scheme"),
        ({"unknown_key": 1}, "unknown setting"),
        ({"agent_enabled": "yes"}, "agent_enabled type"),
    ]
    for payload, what in invalid_bodies:
        expect_error(client.patch("/api/settings", payload), 400, "bad_request", what)

    updated = client.patch(
        "/api/settings",
        {"temperature": 0.3, "max_output_tokens": 1024, "agent_enabled": False, "memory_enabled": False},
    )
    expect_status(updated, 200, "PATCH valid settings")
    merged = updated.data["settings"]
    assert_eq(merged["temperature"], 0.3, "temperature persisted")
    assert_eq(merged["max_output_tokens"], 1024, "max_output_tokens persisted")
    assert_eq(merged["agent_enabled"], False, "agent_enabled persisted")
    assert_eq(merged["memory_enabled"], False, "memory_enabled persisted")

    permissions = client.patch("/api/settings", {"permissions": {"run_command": "deny"}})
    expect_status(permissions, 200, "PATCH permissions")
    assert_eq(permissions.data["settings"]["permissions"]["run_command"], "deny", "permission persisted")
    # restore defaults for later phases
    client.patch(
        "/api/settings",
        {
            "agent_enabled": True,
            "memory_enabled": True,
            "max_output_tokens": 2048,
            "temperature": 0.6,
            "permissions": {"run_command": "ask"},
        },
    )
    return "range/type/whitelist validation + persistence"


def workspace_listing(env: "Any") -> str:
    workspace = env.workspace
    (workspace / "notes").mkdir(parents=True, exist_ok=True)
    (workspace / "notes" / "kiwi.md").write_text("# Kiwi\n", encoding="utf-8")
    (workspace / "README.txt").write_text("hello", encoding="utf-8")
    (workspace / ".hidden").mkdir(exist_ok=True)
    (workspace / ".hidden" / "secret.txt").write_text("no", encoding="utf-8")
    response = env.client.get("/api/workspace")
    expect_status(response, 200, "GET /api/workspace")
    data = response.data
    assert_eq(data["root"], str(workspace.resolve()), "workspace root")
    paths = [entry["path"] for entry in data["files"]]
    assert_in("README.txt", paths, "workspace lists files")
    assert_in("notes/kiwi.md", paths, "workspace recurses depth 2")
    assert_true(not any(path.startswith(".hidden") for path in paths), "dotdirs pruned")
    assert_true(len(paths) <= 200, "workspace entry cap")
    return f"{len(paths)} entries, dotdirs pruned"


# ---------------------------------------------------------------------------
# persistence across a server restart
# ---------------------------------------------------------------------------


def restart_persistence(env: "Any") -> str:
    client = env.client
    conversation = client.post("/api/conversations", {"title": "Persistent chat"}).data
    conversation_id = conversation["id"]
    chat = client.chat(
        {"conversation_id": conversation_id, "message": "remember this [[scenario:plain]]"},
        timeout=30,
    )
    chat.wait_done(25)
    chat.wait_closed(15)
    memory = client.post("/api/memories", {"content": "Survives a server restart.", "kind": "fact"})
    expect_status(memory, 200, "create memory before restart")
    client.patch("/api/settings", {"user_name": "Yemre"})
    env.fake.reset()

    env.restart()

    conversations = env.client.get("/api/conversations").data["conversations"]
    found = [item for item in conversations if item["id"] == conversation_id]
    assert_true(found, "conversation survives restart")
    detail = env.client.get(f"/api/conversations/{conversation_id}").data
    assert_eq([message["role"] for message in detail["messages"]], ["user", "assistant"], "messages survive")
    memories = env.client.get("/api/memories").data["memories"]
    assert_true(any("Survives a server restart" in item["content"] for item in memories), "memory survives")
    settings = env.client.get("/api/settings").data["settings"]
    assert_eq(settings.get("user_name"), "Yemre", "settings survive restart")
    health = env.client.get("/health")
    expect_status(health, 200, "health after restart")
    return "conversation, messages, memories and settings survive a restart"


# ---------------------------------------------------------------------------
# headers, host guard, error shapes, static shell
# ---------------------------------------------------------------------------


def api_headers(env: "Any") -> str:
    response = env.client.get("/api/settings")
    expect_status(response, 200, "GET /api/settings")
    assert_eq(response.headers.get("cache-control"), "no-store", "api cache-control")
    assert_eq(response.headers.get("x-content-type-options"), "nosniff", "api nosniff")
    assert_eq(response.headers.get("x-frame-options"), "SAMEORIGIN", "frame options")

    chat = env.client.chat({"message": "[[scenario:empty]]"})
    chat.wait_done(20)
    chat.wait_closed(10)
    headers = chat.response.headers
    assert_in("text/event-stream", headers.get("content-type", ""), "SSE content-type")
    connection = headers.get("connection", "")
    assert_true("keep-alive" in connection, "SSE Connection header advertises keep-alive")
    cache_control = headers.get("cache-control", "")
    # docs/API.md says no-cache; the app's global API middleware uses the stronger
    # no-store. Either is correct for an event stream; record the actual value.
    assert_true(
        "no-cache" in cache_control or "no-store" in cache_control,
        f"SSE cache-control carries a no-cache directive (got {cache_control!r})",
    )
    return (
        f"no-store/nosniff/frame ok; SSE Connection={connection!r} "
        f"cache-control={cache_control!r} (Werkzeug also appends close)"
    )


def host_header_guard(env: "Any") -> str:
    client = env.client
    evil = client.get("/health", headers={"Host": "evil.example"})
    expect_status(evil, 403, "Host: evil.example")
    assert_eq(error_code(evil), "forbidden_host", "Host rejection code")
    rebound = client.get("/health", headers={"Host": "127.0.0.1.evil.example"})
    expect_status(rebound, 403, "Host: 127.0.0.1.evil.example")
    for host in (f"127.0.0.1:{env.app.port}", f"localhost:{env.app.port}", f"[::1]:{env.app.port}"):
        allowed = client.get("/health", headers={"Host": host})
        expect_status(allowed, 200, f"Host: {host}")
    return "loopback hosts allowed (with port), DNS-rebinding hosts rejected"


def error_shapes(env: "Any") -> str:
    client = env.client
    unknown = client.get("/api/nope")
    expect_error(unknown, 404, "not_found", "unknown API route")
    wrong_method = client.request("DELETE", "/health")
    expect_error(wrong_method, 405, "method_not_allowed", "wrong method")
    bad_json = client.request(
        "POST", "/api/chat", data=b"{not json", headers={"Content-Type": "application/json"}
    )
    expect_error(bad_json, 400, "bad_request", "malformed JSON")
    huge = client.request(
        "POST", "/api/chat", data=b"x" * (4 * 1024 * 1024 + 64), headers={"Content-Type": "application/json"}
    )
    # MAX_CONTENT_LENGTH is 4 MiB; Flask surfaces the rejected body as 400 here
    # (the 413 mapping exists but JSON parsing happens first). Never a 500.
    assert_true(huge.status in (400, 413), f"oversized body rejected ({huge.status})")
    assert_true(error_code(huge) in ("bad_request", "too_large"), "oversized body JSON error")
    return (
        f"404/405/400 JSON envelopes; {huge.status} {error_code(huge)} "
        "for a >4 MiB body (docs imply 413)"
    )


def chat_validation(env: "Any") -> str:
    client = env.client
    expect_error(client.post("/api/chat", {}), 400, "bad_request", "missing message")
    expect_error(client.post("/api/chat", {"message": "   "}), 400, "bad_request", "blank message")
    expect_error(client.post("/api/chat", {"message": "x" * 32001}), 400, "message_too_long", "huge message")
    expect_error(
        client.post("/api/chat", {"message": "hi", "conversation_id": "c_missing"}),
        404,
        "not_found",
        "unknown conversation",
    )
    expect_error(client.post("/api/chat/stop", {}), 400, "bad_request", "stop without id")
    stopped = client.post("/api/chat/stop", {"conversation_id": "c_missing"})
    expect_status(stopped, 200, "stop is idempotent")
    expect_error(
        client.post("/api/approvals/ap_missing", {"decision": "allow"}), 404, "not_found", "unknown approval"
    )
    expect_error(
        client.post("/api/approvals/ap_missing", {"decision": "maybe"}), 400, "bad_request", "bad decision"
    )
    return "request validation covers 400/404 paths"


def static_shell(env: "Any") -> str:
    client = env.client
    index = client.get("/")
    expect_status(index, 200, "GET /")
    for needle in ("<title>Kiwibro</title>", 'id="composer"', 'id="transcript"', 'src="/static/js/app.js"'):
        assert_in(needle, index.text, f"index.html contains {needle}")
    css = client.get("/static/css/app.css")
    expect_status(css, 200, "GET app.css")
    app_js = client.get("/static/js/app.js")
    expect_status(app_js, 200, "GET app.js")
    favicon = client.get("/favicon.ico")
    expect_status(favicon, 200, "GET /favicon.ico")
    traversal = client.get("/static/../app.py")
    assert_true(traversal.status in (404, 403, 400), f"static traversal blocked ({traversal.status})")
    assert_true("create_app" not in traversal.text, "python source never served")
    return "shell, CSS/JS, favicon served; traversal blocked"
