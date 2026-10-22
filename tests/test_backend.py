"""Offline pytest suite for the Kiwibro backend.

Every test uses a temporary data dir/workspace and an in-process fake LM Studio
client: no network, no real LM Studio, no writes outside the repo sandbox.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiwibro import config, tools
from kiwibro.agent import derive_title, validate_llm_title
from kiwibro.context import (
    compute_budget,
    context_stats,
    estimate_messages,
    estimate_tokens,
)
from kiwibro.db import fts_query
from kiwibro.llm import (
    ChatResult,
    Completion,
    ContextLimitError,
    LMStudioClient,
    LMStudioError,
    ToolsUnsupportedError,
    classify_http_error,
    normalize_model,
    parse_json_loose,
)
from kiwibro.memory import is_similar, normalize, parse_ops
from kiwibro.sandbox import SandboxError, Workspace, check_command
from kiwibro.search import SearchEngine, SearchError, assert_public_host
from kiwibro import create_app

MODEL = {
    "id": "fake-1",
    "name": "fake-1",
    "type": "llm",
    "state": "loaded",
    "context": 32768,
    "max_context": 32768,
    "tools": True,
    "publisher": "test",
    "quantization": "Q4_K_M",
}


# ---------------------------------------------------------------------------
# fakes & helpers
# ---------------------------------------------------------------------------


class FakeLLM:
    """Stands in for LMStudioClient methods; scripted per test."""

    def __init__(self) -> None:
        self.models: list[dict[str, Any]] = [dict(MODEL)]
        self.stream_script: list[Any] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.title_text = "Kiwi Facts"
        self.summary_text = "Summary of the earlier conversation."
        self.memory_ops: Any = None
        self.offline = False

    def list_models(self, force: bool = False) -> list[dict[str, Any]]:
        if self.offline:
            raise LMStudioError("connection", "LM Studio is not reachable.")
        return [dict(model) for model in self.models]

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 900,
        temperature: float = 0.0,
        read_timeout: int = 300,
        retries: int = 1,
    ) -> Completion:
        system = messages[0]["content"] if messages else ""
        if "You name conversations" in system:
            return Completion(self.title_text)
        if "rolling summary" in system:
            return Completion(self.summary_text)
        return Completion("ok")

    def complete_json(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 900,
        temperature: float = 0.0,
        read_timeout: int = 300,
        retries: int = 1,
    ) -> Any:
        return self.memory_ops

    def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.6,
        max_tokens: int = 2048,
        on_delta: Any = None,
        on_reasoning: Any = None,
        abort: Any = None,
        extra: Any = None,
        on_rewrite: Any = None,
    ) -> ChatResult:
        self.stream_calls.append(
            {"model": model, "messages": messages, "tools": tools, "abort": abort, "extra": extra}
        )
        item = self.stream_script.pop(0) if self.stream_script else ChatResult(
            content="Hello from the fake model.",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        if isinstance(item, tuple) and item[0] == "partial_error":
            if on_delta:
                on_delta("partial text before the error")
            raise LMStudioError("stream", "The connection to LM Studio was interrupted.")
        if isinstance(item, tuple) and item[0] == "abort_partial":
            if on_delta:
                on_delta("partial text before abort")
            abort.set()
            from kiwibro.llm import TurnAborted

            raise TurnAborted()
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item()
        if item.reasoning and on_reasoning:
            on_reasoning(item.reasoning)
        if item.content and on_delta:
            on_delta(item.content)
        return item


class Env:
    def __init__(self, app, svc, fake: FakeLLM, workspace: Path) -> None:
        self.app = app
        self.svc = svc
        self.fake = fake
        self.workspace = workspace
        self.client = app.test_client()
        self.last_chat_response: Any = None

    def new_client(self):
        return self.app.test_client()

    def settings_patch(self, payload: dict[str, Any]):
        response = self.client.patch("/api/settings", json=payload)
        assert response.status_code == 200, response.get_json()
        return response.get_json()["settings"]

    def collections(self) -> dict[str, Any]:
        return {
            "conversations": self.svc.db.list_conversations(),
            "memories": self.svc.db.list_memories(),
        }


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    data_dir = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = create_app(
        {
            "TESTING": True,
            "KIWIBRO_DATA_DIR": str(data_dir),
            "KIWIBRO_WORKSPACE": str(workspace),
            "KIWIBRO_LMSTUDIO_URL": "http://127.0.0.1:9",
        }
    )
    svc = app.extensions["kiwibro"]
    fake = FakeLLM()
    monkeypatch.setattr(svc.llm, "list_models", fake.list_models)
    monkeypatch.setattr(svc.llm, "stream_chat", fake.stream_chat)
    monkeypatch.setattr(svc.llm, "complete", fake.complete)
    monkeypatch.setattr(svc.llm, "complete_json", fake.complete_json)
    return Env(app, svc, fake, workspace)


def parse_frame(block: str) -> tuple[str, dict[str, Any]] | None:
    event = "message"
    data = ""
    for line in block.splitlines():
        if line.startswith("event: "):
            event = line[7:].strip()
        elif line.startswith("data: "):
            data += line[6:]
        elif line.startswith(":"):
            continue
    if not data:
        return None
    return event, json.loads(data)


def collect_sse(response: Any, timeout: float = 15.0) -> tuple[list[tuple[str, dict[str, Any]]], str]:
    events: list[tuple[str, dict[str, Any]]] = []
    raw = ""
    buffer = ""
    deadline = time.monotonic() + timeout
    for chunk in response.response:
        if time.monotonic() > deadline:
            raise AssertionError("SSE stream timed out")
        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        raw += text
        buffer += text
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            parsed = parse_frame(block)
            if parsed is None:
                continue
            events.append(parsed)
            if parsed[0] in ("done", "error"):
                return events, raw
    return events, raw


class SSEStream:
    """Consume an SSE response on a background thread so tests can act mid-stream."""

    def __init__(self, response: Any) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.raw = ""
        self.done = threading.Event()
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, args=(response,), daemon=True)

    def start(self) -> "SSEStream":
        self.thread.start()
        return self

    def _run(self, response: Any) -> None:
        buffer = ""
        try:
            for chunk in response.response:
                text = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
                self.raw += text
                buffer += text
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    parsed = parse_frame(block)
                    if parsed is None:
                        continue
                    with self._lock:
                        self.events.append(parsed)
                    if parsed[0] in ("done", "error"):
                        self.done.set()
        finally:
            self.done.set()

    def wait_for(self, name: str, timeout: float = 10.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                for event, payload in self.events:
                    if event == name:
                        return payload
            if self.done.is_set():
                with self._lock:
                    for event, payload in self.events:
                        if event == name:
                            return payload
                break
            time.sleep(0.01)
        raise AssertionError(f"event {name} not seen; events={[e for e, _ in self.events]}")

    def finish(self, timeout: float = 15.0) -> list[tuple[str, dict[str, Any]]]:
        assert self.done.wait(timeout), f"stream did not finish: {[e for e, _ in self.events]}"
        time.sleep(0.05)
        with self._lock:
            return list(self.events)

    def names(self) -> list[str]:
        with self._lock:
            return [event for event, _ in self.events]


def chat_events(env: Env, message: str, conversation_id: str | None = None, model: str | None = None):
    body: dict[str, Any] = {"message": message}
    if conversation_id:
        body["conversation_id"] = conversation_id
    if model:
        body["model"] = model
    response = env.client.post("/api/chat", json=body, buffered=False)
    assert response.status_code == 200, response.get_data(as_text=True)
    env.last_chat_response = response
    return collect_sse(response)


def tool_call_result(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> ChatResult:
    return ChatResult(
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
        usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    )


# ---------------------------------------------------------------------------
# token estimation & budget
# ---------------------------------------------------------------------------


def test_estimate_tokens_handles_ascii_and_unicode():
    assert estimate_tokens("") == 0
    ascii_cost = estimate_tokens("a" * 360)
    unicode_cost = estimate_tokens("ü" * 100)
    assert ascii_cost == 100
    assert unicode_cost == 90
    assert estimate_tokens(None) == 0


def test_to_llm_message_preserves_roles(monkeypatch: pytest.MonkeyPatch):
    from kiwibro.context import to_llm_message

    assert to_llm_message({"role": "assistant", "content": "answer"})["role"] == "assistant"
    assert to_llm_message({"role": "user", "content": "question"})["role"] == "user"
    assert to_llm_message({"role": "assistant", "content": ""}) is None
    tool = to_llm_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": {}}}],
        }
    )
    assert tool["role"] == "assistant"
    assert tool["tool_calls"][0]["function"]["arguments"] == "{}"
    assert to_llm_message({"role": "tool", "content": "result", "tool_call_id": "c1"})["role"] == "tool"


def test_estimate_messages_includes_overhead_and_safety():
    messages = [
        {"role": "user", "content": "x" * 36},
        {"role": "assistant", "content": "y" * 36, "tool_calls": [{"function": {"name": "t", "arguments": "{}"}}]},
    ]
    tokens = estimate_messages(messages)
    raw = estimate_tokens("x" * 36) + estimate_tokens("y" * 36)
    assert tokens > raw * 1.1


def test_compute_budget_and_stats():
    context = 32768
    budget = compute_budget(context, 2048)
    reserve = min(2048, max(512, context // 8))
    assert budget == context - reserve - 512
    assert compute_budget(600, 2048) == 1024
    assert context_stats(1000, 4000) == {"used": 1000, "budget": 4000, "percent": 25}
    assert context_stats(9000, 4000)["percent"] == 100


def test_normalize_model_metadata():
    raw = {
        "id": "m1",
        "type": "llm",
        "state": "loaded",
        "loaded_context_length": 8192,
        "max_context_length": 131072,
        "capabilities": ["tool_use"],
        "publisher": "me",
        "quantization": "Q8_0",
    }
    model = normalize_model(raw)
    assert model["context"] == 8192
    assert model["tools"] is True
    unloaded = normalize_model({**raw, "state": "not-loaded"})
    assert unloaded["context"] == 32768
    assert normalize_model({"id": "embed", "type": "embeddings"}) is None
    assert normalize_model({"id": "vlm", "type": "vlm"})["tools"] is False


def test_fts_query_is_sanitised():
    assert fts_query("kiwi bird") == '"kiwi" OR "bird"'
    assert fts_query("") is None
    assert fts_query('quote" and *') == '"quote" OR "and"'


def test_parse_json_loose_variants():
    assert parse_json_loose('```json\n{"ops": []}\n```') == {"ops": []}
    assert parse_json_loose('Sure! {"a": 1} trailing') == {"a": 1}
    assert parse_json_loose("no json here") is None
    # Seen live from a 4B model: a trailing comma used to make the parser fall back to the
    # inner object, so the extracted memory was silently dropped.
    assert parse_json_loose('{"ops": [{"op": "add", "kind": "preference", "content": "User prefers short answers"},]}') == {
        "ops": [{"op": "add", "kind": "preference", "content": "User prefers short answers"}]
    }
    assert parse_json_loose("{'ops': []}") == {"ops": []}
    assert parse_json_loose('[{"op": "add"}]') == [{"op": "add"}]


# ---------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------


def test_db_conversations_messages_and_cascade(env: Env):
    db = env.svc.db
    conversation = db.create_conversation(title="Test chat", model="fake-1")
    second = db.create_conversation()
    first_msg = db.add_message(conversation["id"], "user", "hello")
    second_msg = db.add_message(conversation["id"], "assistant", "hi", model="fake-1", tokens=3)
    assert (first_msg["seq"], second_msg["seq"]) == (1, 2)
    assert db.conversation_message_count(conversation["id"]) == 2
    listed = db.list_conversations()
    assert listed[0]["message_count"] == 2
    assert [c["id"] for c in listed] == [conversation["id"], second["id"]]
    assert db.get_message(first_msg["id"])["content"] == "hello"
    assert db.get_conversation("missing") is None
    assert db.delete_conversation(conversation["id"]) is True
    assert db.list_messages(conversation["id"]) == []
    cleared = db.clear_conversation(second["id"])
    assert cleared["summary"] == ""


def test_db_pinned_ordering_and_update(env: Env):
    db = env.svc.db
    one = db.create_conversation(title="One")
    two = db.create_conversation(title="Two")
    db.update_conversation(one["id"], pinned=True, title="Pinned one")
    listed = db.list_conversations()
    assert listed[0]["id"] == one["id"] and listed[0]["pinned"] is True
    assert listed[1]["id"] == two["id"]


def test_db_memories_fts_search_and_usage(env: Env):
    db = env.svc.db
    memory = db.create_memory("preference", "Prefers dark themes", normalize("Prefers dark themes"))
    db.create_memory("fact", "Lives in Istanbul", normalize("Lives in Istanbul"))
    assert db.count_memories() == 2
    found = db.search_memories("dark")
    assert [m["id"] for m in found] == [memory["id"]]
    updated = db.update_memory(memory["id"], content="Prefers dark mode", normalized=normalize("Prefers dark mode"))
    assert updated["content"] == "Prefers dark mode"
    assert db.search_memories("mode")[0]["id"] == memory["id"]
    assert db.delete_memory(memory["id"]) is True
    assert db.search_memories("mode") == []
    assert db.delete_all_memories() == 1
    db.add_usage("c_1", "fake-1", 10, 5)
    assert db.usage_totals()["prompt_tokens"] == 10


def test_db_settings_and_permissions(env: Env):
    db = env.svc.db
    defaults = config.default_settings()
    merged = db.get_settings(defaults)
    assert merged["permissions"]["write_file"] == "ask"
    db.patch_settings({"temperature": 0.9, "permissions": {"write_file": "auto"}})
    merged = db.get_settings(defaults)
    assert merged["temperature"] == 0.9
    assert merged["permissions"]["write_file"] == "auto"
    assert merged["permissions"]["web_search"] == "auto"


# ---------------------------------------------------------------------------
# context compaction
# ---------------------------------------------------------------------------


def test_compaction_summarises_old_messages(env: Env):
    env.fake.models = [{**MODEL, "context": 3000}]
    conversation = env.svc.db.create_conversation(title="Long")
    for index in range(12):
        env.svc.db.add_message(conversation["id"], "user", f"user message {index} " + "word " * 55)
        env.svc.db.add_message(conversation["id"], "assistant", f"reply {index} " + "detail " * 55)
    prepared = env.svc.context.prepare(conversation["id"], "fake-1", "SYSTEM")
    assert prepared.compacted is True
    refreshed = env.svc.db.get_conversation(conversation["id"])
    assert refreshed["summary"] == "Summary of the earlier conversation."
    assert refreshed["summary_through_seq"] > 0
    remaining = env.svc.db.list_messages(conversation["id"], after_seq=refreshed["summary_through_seq"])
    assert len(remaining) >= 2
    assert prepared.used <= prepared.budget
    system_prompt = prepared.messages[0]
    assert system_prompt["role"] == "system"
    assert "Earlier conversation summary" in system_prompt["content"]
    assert "Summary of the earlier conversation." in system_prompt["content"]


def test_compaction_falls_back_to_hard_trim(env: Env, monkeypatch: pytest.MonkeyPatch):
    env.fake.models = [{**MODEL, "context": 2000}]
    conversation = env.svc.db.create_conversation(title="Huge")
    for index in range(20):
        env.svc.db.add_message(conversation["id"], "user", f"question {index} " + "filler " * 60)
        env.svc.db.add_message(conversation["id"], "assistant", f"answer {index} " + "filler " * 60)

    def failing_complete(**kwargs):
        raise LMStudioError("server", "no summary for you")

    monkeypatch.setattr(env.svc.llm, "complete", failing_complete)
    prepared = env.svc.context.prepare(conversation["id"], "fake-1", "SYSTEM")
    assert prepared.trimmed is True
    non_system = [m for m in prepared.messages if m["role"] != "system"]
    assert 1 <= len(non_system) <= 6
    assert non_system[-1]["role"] == "assistant"


def test_context_recovery_from_context_limit(env: Env):
    conversation = env.svc.db.create_conversation(title="Recovery")
    for index in range(8):
        env.svc.db.add_message(conversation["id"], "user", f"u{index} " + "x " * 50)
        env.svc.db.add_message(conversation["id"], "assistant", f"a{index} " + "y " * 50)
    assert env.svc.context.recover_aggressive(conversation["id"], "fake-1") is True
    refreshed = env.svc.db.get_conversation(conversation["id"])
    assert refreshed["summary_through_seq"] > 0
    assert env.svc.context.hard_recover(conversation["id"], keep=2) is True


# ---------------------------------------------------------------------------
# memory engine
# ---------------------------------------------------------------------------


def test_memory_normalisation_and_dedupe():
    assert normalize("  User   PREFERS dark-themes! ") == "user prefers dark themes"
    assert is_similar("User prefers dark themes", "User prefers dark theme") is True
    assert is_similar("likes kiwi birds and hiking", "hiking and kiwi birds likes") is True
    assert is_similar("User prefers dark themes", "Runs a bakery in Lisbon") is False


def test_memory_parse_ops_robustness():
    ops = parse_ops('```json\n{"ops": [{"op": "add", "kind": "fact", "content": "Likes tea"}]}\n```')
    assert ops == [{"op": "add", "kind": "fact", "content": "Likes tea"}]
    assert parse_ops({"memories": [{"op": "delete", "id": "mem_1"}]}) == [{"op": "delete", "id": "mem_1"}]
    assert parse_ops("not json") == []
    assert parse_ops({"ops": [{"op": "explode"}, {"op": "add", "content": "ok"}]}) == [{"op": "add", "content": "ok"}]


def test_memory_extraction_applies_max_three_ops(env: Env):
    env.fake.memory_ops = {
        "ops": [
            {"op": "add", "kind": "preference", "content": "User prefers dark themes"},
            {"op": "add", "kind": "fact", "content": "User lives in Istanbul"},
            {"op": "add", "kind": "project", "content": "User is building Kiwibro"},
            {"op": "add", "kind": "person", "content": "User's friend is Ada"},
        ]
    }
    conversation = env.svc.db.create_conversation()
    env.svc.db.add_message(conversation["id"], "user", "I prefer dark themes and live in Istanbul")
    env.svc.db.add_message(conversation["id"], "assistant", "Noted.")
    counts = env.svc.memory.extract_and_apply(conversation["id"], "fake-1")
    assert counts == {"added": 3, "updated": 0, "removed": 0}
    assert env.svc.db.count_memories() == 3


def test_memory_dedupes_near_duplicate_on_extraction(env: Env):
    env.svc.memory.remember("User prefers dark themes", "preference")
    env.fake.memory_ops = {
        "ops": [{"op": "add", "kind": "preference", "content": "User prefers dark theme"}]
    }
    conversation = env.svc.db.create_conversation()
    env.svc.db.add_message(conversation["id"], "user", "dark mode please")
    counts = env.svc.memory.extract_and_apply(conversation["id"], "fake-1")
    assert counts["added"] == 0
    assert counts["updated"] == 1
    assert env.svc.db.count_memories() == 1


def test_memory_retrieval_renders_and_marks_used(env: Env):
    memory = env.svc.memory.remember("User prefers dark themes", "preference")
    block, used = env.svc.memory.build_block("dark themes")
    assert "User prefers dark themes" in block
    assert used and used[0]["id"] == memory["id"]
    refreshed = env.svc.db.get_memory(memory["id"])
    assert refreshed["last_used_at"] is not None
    assert env.svc.memory.build_block("x", enabled=False) == ("", [])


def test_memory_never_raises_when_llm_fails(env: Env, monkeypatch: pytest.MonkeyPatch):
    def boom(**kwargs):
        raise RuntimeError("LLM exploded")

    monkeypatch.setattr(env.svc.llm, "complete_json", boom)
    conversation = env.svc.db.create_conversation()
    env.svc.db.add_message(conversation["id"], "user", "hello")
    counts = env.svc.memory.extract_and_apply(conversation["id"], "fake-1")
    assert counts == {"added": 0, "updated": 0, "removed": 0}


# ---------------------------------------------------------------------------
# sandbox
# ---------------------------------------------------------------------------


def test_sandbox_path_containment(tmp_path: Path):
    root = tmp_path / "ws"
    root.mkdir()
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    workspace = Workspace(root)
    with pytest.raises(SandboxError):
        workspace.read_file("../outside.txt")
    with pytest.raises(SandboxError):
        workspace.read_file(str(tmp_path / "outside.txt"))
    (root / "notes.txt").write_text("inside", encoding="utf-8")
    assert workspace.read_file("notes.txt")["content"] == "inside"
    if hasattr(Path, "symlink_to"):
        link = root / "escape"
        try:
            link.symlink_to(tmp_path / "outside.txt")
        except OSError:  # pragma: no cover - platform dependent
            pass
        else:
            with pytest.raises(SandboxError):
                workspace.read_file("escape")


def test_sandbox_file_tools(tmp_path: Path):
    root = tmp_path / "ws"
    root.mkdir()
    workspace = Workspace(root)
    written = workspace.write_file("sub/notes.txt", "hello\nworld\n")
    assert written["path"] == "sub/notes.txt"
    read = workspace.read_file("sub/notes.txt")
    assert read["content"] == "hello\nworld\n"
    assert workspace.read_file("sub/notes.txt", start=2, end=2)["content"] == "world"
    replaced = workspace.replace_in_file("sub/notes.txt", "world", "kiwi")
    assert replaced["replacements"] == 1
    with pytest.raises(SandboxError):
        workspace.replace_in_file("sub/notes.txt", "missing", "new")
    with pytest.raises(SandboxError):
        workspace.write_file("sub/notes.txt", "x" * (config.MAX_FILE_WRITE_BYTES + 1))
    entries = workspace.list_files(".")
    assert entries[0]["path"] == "sub" and entries[0]["type"] == "dir"
    assert workspace.list_files("sub")[0]["path"] == "sub/notes.txt"
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        (root / "leak").symlink_to(outside)
    except OSError:  # pragma: no cover - platform dependent
        pass
    else:
        assert all(entry["path"] != "leak" for entry in workspace.list_files("."))
        assert workspace.public(outside) == "(outside workspace)"


def test_sandbox_command_blocklist_and_execution(tmp_path: Path):
    root = tmp_path / "ws"
    root.mkdir()
    workspace = Workspace(root)
    for blocked in ("sudo ls", "rm -rf /", "mkfs.ext4 /dev/sda", "shutdown -h now", ":(){ :|:& };:"):
        with pytest.raises(SandboxError):
            check_command(blocked)
    with pytest.raises(SandboxError):
        workspace.run_command("sudo ls")
    result = workspace.run_command("echo hello-from-sandbox")
    assert result["exit_code"] == 0
    assert "hello-from-sandbox" in result["output"]
    assert result["cwd"] == "."
    with pytest.raises(SandboxError):
        workspace.run_command("pwd", cwd="..")


# ---------------------------------------------------------------------------
# search parsers & SSRF guard
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code
        self.headers = {"Content-Type": "text/html"}


BRAVE_HTML = """
<html><body>
<div class="snippet noscript-hide" id="llm-snippet"><div class="title">AI answer</div></div>
<div class="snippet" data-type="web">
  <a href="https://example.com/kiwi"><div class="site-name-content">Example</div></a>
  <div class="title">Kiwi Facts</div>
  <div class="generic-snippet"><div class="content">Kiwis are flightless birds from New Zealand.</div></div>
</div>
</body></html>
"""

BING_RSS = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel><title>t</title>
<item><title>Kiwi - Wikipedia</title><link>https://en.wikipedia.org/wiki/Kiwi</link>
<description>Kiwi are flightless birds.</description></item>
</channel></rss>"""

WIKI_JSON = {
    "query": {
        "pages": {
            "1": {
                "pageid": 1,
                "title": "Kiwi (bird)",
                "index": 1,
                "extract": "Kiwi are flightless birds endemic to New Zealand.",
            }
        }
    }
}


def test_brave_parser_uses_current_markup(monkeypatch: pytest.MonkeyPatch):
    from kiwibro import search as search_module

    monkeypatch.setattr(search_module, "_get", lambda *a, **k: FakeResponse(BRAVE_HTML))
    results = search_module.brave_search("kiwi", 5)
    assert results == [
        {
            "title": "Kiwi Facts",
            "url": "https://example.com/kiwi",
            "snippet": "Kiwis are flightless birds from New Zealand.",
        }
    ]


def test_bing_html_parser_and_rss_last_resort(monkeypatch: pytest.MonkeyPatch):
    from kiwibro import search as search_module

    blocked = FakeResponse("<html><body>no results</body></html>")
    monkeypatch.setattr(search_module, "_get", lambda *a, **k: blocked)
    assert search_module.bing_search("kiwi", 3) == []

    monkeypatch.setattr(
        search_module, "_get", lambda *a, **k: FakeResponse(BING_RSS) if k.get("params", {}).get("format") == "rss" else blocked
    )
    results = search_module.bing_rss_search("kiwi", 3)
    assert results[0]["title"] == "Kiwi - Wikipedia"
    assert results[0]["url"].startswith("https://en.wikipedia.org")
    # The RSS provider is the last-resort entry after the documented chain.
    assert search_module.PROVIDERS[-1][0] == "bing_rss"


def test_search_provider_cooldown_skips_rate_limited_provider(monkeypatch: pytest.MonkeyPatch):
    from kiwibro import search as search_module

    engine = SearchEngine()
    calls: list[str] = []

    def rate_limited(query, limit):
        calls.append("limited")
        raise SearchError("Brave returned HTTP 429")

    def working(query, limit):
        calls.append("working")
        return [{"title": "Result", "url": "https://example.com", "snippet": "text"}]

    monkeypatch.setattr(search_module, "PROVIDERS", (("brave", rate_limited), ("wikipedia", working)))
    assert engine.web_search("one", 3)[0]["title"] == "Result"
    assert engine.web_search("two", 3)[0]["title"] == "Result"
    assert calls == ["limited", "working", "working"]


def test_wikipedia_parser_uses_extracts(monkeypatch: pytest.MonkeyPatch):
    from kiwibro import search as search_module

    class JsonResponse(FakeResponse):
        def json(self):
            return WIKI_JSON

    monkeypatch.setattr(search_module, "_get", lambda *a, **k: JsonResponse(""))
    results = search_module.wikipedia_search("kiwi", 3)
    assert results[0]["snippet"].startswith("Kiwi are flightless birds")
    assert results[0]["url"] == "https://en.wikipedia.org/wiki/Kiwi_%28bird%29"


def test_ddg_reports_captcha(monkeypatch: pytest.MonkeyPatch):
    from kiwibro import search as search_module

    monkeypatch.setattr(search_module, "_post", lambda *a, **k: FakeResponse("captcha", 202))
    with pytest.raises(SearchError):
        search_module.ddg_search("kiwi", 3)


def test_search_chain_cache_and_snippet_enrichment(monkeypatch: pytest.MonkeyPatch):
    from kiwibro import search as search_module

    engine = SearchEngine()
    calls: list[str] = []

    def provider_one(query, limit):
        calls.append("one")
        raise SearchError("blocked")

    def provider_two(query, limit):
        calls.append("two")
        return [{"title": "Result", "url": "https://example.com", "snippet": ""}]

    monkeypatch.setattr(search_module, "PROVIDERS", (("one", provider_one), ("two", provider_two)))
    monkeypatch.setattr(engine, "fetch_snippet", lambda url: "A real description.")
    first = engine.web_search("kiwi", 3)
    second = engine.web_search("kiwi", 3)
    assert calls == ["one", "two"]
    assert first[0]["snippet"] == "A real description."
    assert second == first


def test_ssrf_guard_rejects_local_targets():
    for host in ("localhost", "127.0.0.1", "10.0.0.8", "192.168.1.10", "::1", "169.254.1.1"):
        with pytest.raises(SearchError):
            assert_public_host(host)
    assert_public_host("1.1.1.1")
    with pytest.raises(SearchError):
        SearchEngine().fetch_page("file:///etc/passwd")
    with pytest.raises(SearchError):
        SearchEngine().fetch_page("http://127.0.0.1:8777/health")


# ---------------------------------------------------------------------------
# LM client wire behaviour (UTF-8 + tool call fragments + retry without tools)
# ---------------------------------------------------------------------------


def _sse_response(lines: list[bytes]):
    class Response:
        status_code = 200
        encoding = None
        headers = {"Content-Type": "text/event-stream"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_lines(self, decode_unicode: bool = False):
            yield from lines

        def close(self):
            pass

    return Response()


def test_lm_client_decodes_utf8_and_accumulates_tool_calls(monkeypatch: pytest.MonkeyPatch):
    content_chunk = json.dumps(
        {"choices": [{"delta": {"content": "Kia ora — kiwi 👋"}}]}, ensure_ascii=False
    )
    tool_first = json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_9",
                                "type": "function",
                                "function": {"name": "write_file", "arguments": ""},
                            }
                        ]
                    }
                }
            ]
        }
    )
    tool_second = json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"path": "café.txt"}'}}
                        ]
                    }
                }
            ]
        }
    )
    usage_chunk = json.dumps(
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
    )
    lines = [
        f"data: {content_chunk}".encode("utf-8"),
        f"data: {tool_first}".encode("utf-8"),
        f"data: {tool_second}".encode("utf-8"),
        f"data: {usage_chunk}".encode("utf-8"),
        b"data: [DONE]",
    ]
    monkeypatch.setattr("kiwibro.llm.requests.post", lambda *a, **k: _sse_response(lines))
    deltas: list[str] = []
    client = LMStudioClient(lambda: {"lmstudio_url": "http://fake"})
    result = client.stream_chat(
        model="fake-1", messages=[{"role": "user", "content": "hi"}], on_delta=deltas.append
    )
    assert result.content == "Kia ora — kiwi 👋"
    assert deltas == ["Kia ora — kiwi 👋"]
    assert result.tool_calls[0]["function"]["name"] == "write_file"
    assert json.loads(result.tool_calls[0]["function"]["arguments"]) == {"path": "café.txt"}
    assert result.usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


def test_lm_client_retries_without_tools(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, Any]] = []

    class RejectResponse:
        status_code = 400
        encoding = None
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @property
        def text(self):
            return '{"error": "model does not support tools"}'

    success_lines = [
        f"data: {json.dumps({'choices': [{'delta': {'content': 'no tools needed'}}]})}".encode("utf-8"),
        b"data: [DONE]",
    ]

    def fake_post(url, **kwargs):
        calls.append(kwargs)
        if kwargs.get("json", {}).get("tools"):
            return RejectResponse()
        return _sse_response(success_lines)

    monkeypatch.setattr("kiwibro.llm.requests.post", fake_post)
    client = LMStudioClient(lambda: {"lmstudio_url": "http://fake"})
    result = client.stream_chat(
        model="fake-1",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}],
    )
    assert len(calls) == 2
    assert result.tools_disabled is True
    assert result.content == "no tools needed"


def test_lm_error_classification():
    assert isinstance(classify_http_error(400, "n_ctx is too small"), ContextLimitError)
    assert isinstance(classify_http_error(400, "this model does not support tools"), ToolsUnsupportedError)
    assert classify_http_error(500, "boom").kind == "server"
    assert classify_http_error(404, "no route").kind == "http"


# ---------------------------------------------------------------------------
# titles & tools
# ---------------------------------------------------------------------------


def test_derive_title_is_deterministic():
    assert derive_title("Use the web_search tool to search for 'kiwi bird facts'. Then explain.") == "Kiwi bird facts"
    assert derive_title("Create a file called notes.txt with a haiku") == "Create a file called notes.txt"
    assert derive_title("Reply with exactly: Kia ora — hello 👋") == "Kia ora — hello 👋"
    assert derive_title("please can you write a short poem about kiwis?") == "Write a short poem about"
    assert derive_title("") == "New chat"
    assert derive_title("  hello   world  ") == "World"
    assert derive_title("One kiwi bird fact and") == "One kiwi bird fact"
    assert derive_title("Tell me about kiwis with") == "Kiwis"


def test_validate_llm_title_rejects_echoes():
    assert validate_llm_title("Kiwi Bird Facts") == "Kiwi Bird Facts"
    assert validate_llm_title('"Kiwi Facts"') == "Kiwi Facts"
    assert validate_llm_title("The user wants me to write a chat title") is None
    assert validate_llm_title("Title: A very long title that has way too many words in it") is None
    assert validate_llm_title("one") is None
    assert validate_llm_title("") is None


def test_tool_args_validation():
    args = tools.validate_args("web_search", {"query": "kiwi", "max_results": 8})
    assert args["max_results"] == 8
    with pytest.raises(ValueError):
        tools.validate_args("web_search", {"query": "kiwi", "max_results": 99})
    with pytest.raises(ValueError):
        tools.validate_args("web_search", {"query": ""})
    with pytest.raises(ValueError):
        tools.validate_args("fetch_page", {"url": "file:///etc/passwd"})
    with pytest.raises(ValueError):
        tools.validate_args("read_file", {})
    validated = tools.validate_args("write_file", {"path": "a.txt", "content": ""})
    assert validated["content"] == ""
    assert tools.permission_mode({}, "write_file") == "ask"
    assert tools.permission_mode({}, "web_search") == "auto"


# ---------------------------------------------------------------------------
# API: health, bootstrap, models, conversations, memories, settings, workspace
# ---------------------------------------------------------------------------


def test_health_and_bootstrap(env: Env):
    health = env.client.get("/health")
    assert health.status_code == 200
    payload = health.get_json()
    assert payload["ok"] is True and payload["lmstudio"] is True
    assert payload["version"] == config.VERSION
    assert payload["model"] == "fake-1"
    assert isinstance(payload["started_at"], int) and payload["started_at"] > 0
    assert health.headers["Cache-Control"] == "no-store"

    bootstrap = env.client.get("/api/bootstrap").get_json()
    assert bootstrap["current_model"] == "fake-1"
    assert bootstrap["models"][0]["id"] == "fake-1"
    assert bootstrap["settings"]["permissions"]["write_file"] == "ask"
    assert bootstrap["memories_count"] == 0
    assert isinstance(bootstrap["conversations"], list)


def test_system_prompt_identity_and_capabilities(env: Env):
    settings = env.svc.settings_provider()
    prompt = env.svc.agent._system_prompt(settings, model="fake-1", model_info={"tools": True})
    assert "You are Kiwibro" in prompt
    assert "local AI agent" in prompt
    assert "LM Studio" in prompt
    assert "fake-1" in prompt
    assert "## Workspace" in prompt
    assert "web_search" in prompt and "run_command" in prompt
    assert "permission" in prompt.lower()
    # KV-cache rule: nothing that changes per turn lives in message 0.
    assert "Local time" not in prompt and "## Memory" not in prompt
    assert "context note" in prompt

    disabled = env.svc.agent._system_prompt(
        {**settings, "agent_enabled": False}, model="fake-1", model_info={"tools": True}
    )
    assert "Agent mode is off" in disabled
    assert "web_search, fetch_page" not in disabled

    # Models without native tool calling get the text tool protocol, byte-stable per settings.
    unsupported = env.svc.agent._system_prompt(settings, model="fake-1", model_info={"tools": False})
    assert "<tool_call>" in unsupported and "web_search(query: string" in unsupported
    assert unsupported == env.svc.agent._system_prompt(settings, model="fake-1", model_info={"tools": False})


def test_health_offline_is_calm(env: Env, monkeypatch: pytest.MonkeyPatch):
    def offline(**kwargs):
        raise LMStudioError("connection", "LM Studio is not reachable.")

    monkeypatch.setattr(env.svc.llm, "list_models", offline)
    health = env.client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["lmstudio"] is False
    models = env.client.get("/api/models").get_json()
    assert models["lmstudio"] is False and models["models"] == []
    bootstrap = env.client.get("/api/bootstrap")
    assert bootstrap.status_code == 200
    assert bootstrap.get_json()["models"] == []


def test_host_header_must_be_loopback(env: Env):
    response = env.client.get("/health", headers={"Host": "evil.example.com"})
    assert response.status_code == 403
    assert response.get_json()["error"]["code"] == "forbidden_host"


def test_conversations_api(env: Env):
    created = env.client.post("/api/conversations", json={})
    assert created.status_code == 201
    conversation = created.get_json()
    assert conversation["title"] == "New chat" and conversation["messages"] == []
    conversation_id = conversation["id"]

    patched = env.client.patch(
        f"/api/conversations/{conversation_id}", json={"title": "Kiwis", "pinned": True}
    ).get_json()
    assert patched["conversation"]["title"] == "Kiwis"
    assert patched["conversation"]["pinned"] is True
    listed = env.client.get("/api/conversations").get_json()["conversations"]
    assert listed[0]["id"] == conversation_id and listed[0]["pinned"] is True

    detail = env.client.get(f"/api/conversations/{conversation_id}").get_json()
    assert detail["messages"] == []
    assert set(detail["context"]) == {"used", "budget", "percent"}

    cleared = env.client.post(f"/api/conversations/{conversation_id}/clear")
    assert cleared.status_code == 200 and cleared.get_json()["conversation"]["summary"] is None

    assert env.client.delete(f"/api/conversations/{conversation_id}").get_json() == {"ok": True}
    missing = env.client.get(f"/api/conversations/{conversation_id}")
    assert missing.status_code == 404
    assert missing.get_json()["error"]["code"] == "not_found"


def test_models_api_and_selection(env: Env):
    env.fake.models = [
        dict(MODEL),
        {
            "id": "other",
            "name": "other",
            "type": "llm",
            "state": "not-loaded",
            "context": 16384,
            "max_context": 65536,
            "tools": False,
            "publisher": "",
            "quantization": "",
        },
    ]
    payload = env.client.get("/api/models").get_json()
    assert [model["id"] for model in payload["models"]] == ["fake-1", "other"]
    assert payload["current"] == "fake-1"

    selected = env.client.post("/api/models/select", json={"id": "other"})
    assert selected.get_json() == {"ok": True, "current": "other"}
    assert env.svc.settings_provider()["model"] == "other"
    unknown = env.client.post("/api/models/select", json={"id": "ghost"})
    assert unknown.status_code == 400
    assert unknown.get_json()["error"]["code"] == "unknown_model"


def test_memories_api_crud(env: Env):
    created = env.client.post("/api/memories", json={"content": "User prefers dark themes", "kind": "preference"})
    assert created.status_code == 200
    memory = created.get_json()["memory"]
    assert memory["kind"] == "preference"
    assert set(memory) == {
        "id",
        "kind",
        "content",
        "pinned",
        "confidence",
        "created_at",
        "updated_at",
        "last_used_at",
    }
    listed = env.client.get("/api/memories?q=dark").get_json()
    assert listed["total"] == 1 and listed["memories"][0]["id"] == memory["id"]

    patched = env.client.patch(
        f"/api/memories/{memory['id']}", json={"content": "User prefers dark mode", "pinned": True}
    ).get_json()
    assert patched["memory"]["pinned"] is True
    assert patched["memory"]["content"] == "User prefers dark mode"
    assert env.client.delete(f"/api/memories/{memory['id']}").get_json() == {"ok": True}
    assert env.client.get("/api/memories").get_json()["total"] == 0
    created_again = env.client.post("/api/memories", json={"content": "Another fact"})
    assert created_again.status_code == 200
    deleted = env.client.delete("/api/memories").get_json()
    assert deleted == {"ok": True, "deleted": 1}


def test_memories_api_validation(env: Env):
    assert env.client.post("/api/memories", json={"content": "x"}).status_code == 400
    assert env.client.post("/api/memories", json={"content": "ok fact", "kind": "nonsense"}).status_code == 400
    assert env.client.get("/api/memories?kind=nonsense").status_code == 400
    assert env.client.patch("/api/memories/mem_missing", json={"pinned": True}).status_code == 404


def test_settings_api_and_validation(env: Env):
    payload = env.client.get("/api/settings").get_json()
    settings = payload["settings"]
    assert settings["agent_enabled"] is True and settings["memory_enabled"] is True
    assert payload["data_dir"].endswith("data")

    updated = env.settings_patch(
        {"temperature": 1.2, "max_output_tokens": 4096, "permissions": {"run_command": "deny"}}
    )
    assert updated["temperature"] == 1.2
    assert updated["permissions"]["run_command"] == "deny"
    assert env.client.patch("/api/settings", json={"temperature": 5}).status_code == 400
    assert env.client.patch("/api/settings", json={"max_output_tokens": 1}).status_code == 400
    assert env.client.patch("/api/settings", json={"permissions": {"x": "auto"}}).status_code == 400
    assert env.client.patch("/api/settings", json={"permissions": {"read_file": "sometimes"}}).status_code == 400
    assert env.client.patch("/api/settings", json={"workspace": "/"}).status_code == 400
    assert env.client.patch("/api/settings", json={"workspace": "relative/path"}).status_code == 400
    assert env.client.patch("/api/settings", json={"lmstudio_url": "ftp://nope"}).status_code == 400
    assert env.client.patch("/api/settings", json={"unknown": 1}).status_code == 400


def test_workspace_listing(env: Env):
    (env.workspace / "notes.md").write_text("# Notes", encoding="utf-8")
    (env.workspace / "sub").mkdir()
    (env.workspace / "sub" / "inner.txt").write_text("inner", encoding="utf-8")
    payload = env.client.get("/api/workspace").get_json()
    assert payload["root"] == str(env.workspace.resolve())
    paths = {entry["path"]: entry for entry in payload["files"]}
    assert paths["notes.md"]["type"] == "file" and paths["notes.md"]["size"] == 7
    assert paths["sub"]["type"] == "dir"
    assert paths["sub/inner.txt"]["type"] == "file"


def test_root_serves_template(env: Env):
    response = env.client.get("/")
    assert response.status_code == 200
    assert b"Kiwibro" in response.data


# ---------------------------------------------------------------------------
# API: chat SSE
# ---------------------------------------------------------------------------


def test_chat_sse_event_sequence_and_persistence(env: Env):
    env.fake.stream_script = [
        ChatResult(
            content="Kia ora — kiwi 👋",
            usage={"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
        )
    ]
    events, raw = chat_events(env, "Hello kiwi")
    names = [event for event, _ in events]
    assert names[0] == "meta"
    assert names[-1] == "done"
    assert "delta" in names and "title" in names
    meta = events[0][1]
    assert meta["model"] == "fake-1"
    assert meta["title"] == "New chat"
    assert set(meta["context"]) == {"used", "budget", "percent"}
    assert env.fake.stream_calls[0]["tools"] is not None
    done = events[-1][1]
    assert done["aborted"] is False and done["model"] == "fake-1"
    assert done["usage"] == {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42}

    # SSE framing is exact: every non-comment block begins with `event:` then `data:`.
    for block in raw.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        lines = block.splitlines()
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")

    conversation_id = meta["conversation_id"]
    assert env.last_chat_response.headers["Cache-Control"] == "no-cache"
    detail = env.client.get(f"/api/conversations/{conversation_id}").get_json()
    assert detail["conversation"]["title"] == "Kiwi"
    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["content"] == "Kia ora — kiwi 👋"
    assert detail["messages"][1]["model"] == "fake-1"
    assert env.svc.db.usage_totals()["turns"] == 1


def test_chat_meta_is_first_even_when_compaction_fires(env: Env):
    env.fake.models = [{**MODEL, "context": 2500}]
    created = env.client.post("/api/conversations", json={}).get_json()
    conversation_id = created["id"]
    for index in range(12):
        env.svc.db.add_message(conversation_id, "user", f"old question {index} " + "filler " * 55)
        env.svc.db.add_message(conversation_id, "assistant", f"old answer {index} " + "filler " * 55)
    env.fake.stream_script = [ChatResult(content="Fresh answer")]
    events, _raw = chat_events(env, "fresh question", conversation_id=conversation_id)
    names = [event for event, _ in events]
    assert names[0] == "meta", names
    assert "context" in names
    assert names.index("context") > names.index("meta")
    assert names[-1] == "done"


def test_chat_offline_emits_meta_then_calm_error(env: Env):
    env.fake.offline = True
    response = env.client.post("/api/chat", json={"message": "hi"}, buffered=False)
    assert response.status_code == 200
    events, _raw = collect_sse(response)
    names = [event for event, _ in events]
    assert names[0] == "meta"
    assert names[-1] == "error"
    error = events[-1][1]
    assert error["code"] == "connection"
    assert error["recoverable"] is True
    assert "not reachable" in error["message"]


def test_chat_oversized_body_returns_413(env: Env):
    response = env.client.post(
        "/api/chat",
        data="x" * (env.app.config["MAX_CONTENT_LENGTH"] + 10),
        content_type="application/json",
    )
    assert response.status_code == 413
    assert response.get_json()["error"]["code"] == "too_large"


def test_sse_chat_headers(env: Env):
    env.fake.stream_script = [ChatResult(content="Headers")]
    events, _raw = chat_events(env, "header check")
    assert events[-1][0] == "done"
    headers = env.last_chat_response.headers
    assert headers["Content-Type"].startswith("text/event-stream")
    assert headers["Cache-Control"] == "no-cache"
    assert headers["X-Accel-Buffering"] == "no"
    assert "keep-alive" in (headers.get("Connection") or "")


def test_chat_validation_errors(env: Env):
    assert env.client.post("/api/chat", json={"message": ""}).status_code == 400
    assert env.client.post("/api/chat", json={"message": "x" * 32001}).status_code == 400
    missing = env.client.post("/api/chat", json={"message": "hi", "conversation_id": "c_missing"})
    assert missing.status_code == 404
    assert missing.get_json()["error"]["code"] == "not_found"
    assert env.client.post("/api/chat/stop", json={}).status_code == 400


def test_chat_tool_call_runs_with_auto_permission(env: Env):
    env.settings_patch({"permissions": {"write_file": "auto"}})
    env.fake.stream_script = [
        tool_call_result("write_file", {"path": "notes.txt", "content": "hello from tool"}),
        ChatResult(content="Done: wrote notes.txt.", usage={"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48}),
    ]
    events, _raw = chat_events(env, "Create notes.txt using write_file")
    names = [event for event, _ in events]
    assert "tool_call" in names and "tool_result" in names
    tool_call = next(payload for event, payload in events if event == "tool_call")
    assert tool_call["name"] == "write_file"
    assert tool_call["requires_approval"] is False
    tool_result = next(payload for event, payload in events if event == "tool_result")
    assert tool_result["ok"] is True
    assert (env.workspace / "notes.txt").read_text(encoding="utf-8") == "hello from tool"
    assert len(env.fake.stream_calls) == 2

    messages = env.svc.db.list_messages(env.svc.db.list_conversations()[0]["id"])
    roles = [message["role"] for message in messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert messages[2]["tool_name"] == "write_file"
    assert messages[3]["content"] == "Done: wrote notes.txt."


def _approval_chat(env: Env, decision: str):
    env.settings_patch({"autonomy": "supervised"})  # the per-call approval flow under test
    env.fake.stream_script = [
        tool_call_result("write_file", {"path": "approved.txt", "content": "approved content"}),
        ChatResult(content="Approved and written."),
    ]
    response = env.client.post("/api/chat", json={"message": "write approved.txt"}, buffered=False)
    assert response.status_code == 200
    stream = SSEStream(response).start()
    approval = stream.wait_for("approval")
    assert approval["tool"] == "write_file"
    assert approval["timeout_s"] == int(config.APPROVAL_TIMEOUT_S)
    assert approval["summary"]
    resolved = env.new_client().post(
        f"/api/approvals/{approval['approval_id']}", json={"decision": decision}
    )
    assert resolved.status_code == 200
    assert resolved.get_json() == {"ok": True, "decision": decision}
    events = stream.finish()
    payloads = {event: payload for event, payload in events}
    assert events[-1][0] == "done"
    return payloads, (env.workspace / "approved.txt")


def test_chat_approval_allow(env: Env):
    payloads, file = _approval_chat(env, "allow")
    assert payloads["tool_result"]["ok"] is True
    assert file.read_text(encoding="utf-8") == "approved content"
    assert env.svc.settings_provider()["permissions"]["write_file"] == "ask"


def test_chat_approval_deny(env: Env):
    payloads, file = _approval_chat(env, "deny")
    assert payloads["tool_result"]["ok"] is False
    assert not file.exists()
    assert env.svc.settings_provider()["permissions"]["write_file"] == "ask"


def test_chat_approval_always_persists_auto(env: Env):
    payloads, file = _approval_chat(env, "always")
    assert payloads["tool_result"]["ok"] is True
    assert file.read_text(encoding="utf-8") == "approved content"
    assert env.svc.settings_provider()["permissions"]["write_file"] == "auto"

    # Next turn: same tool runs without an approval event.
    env.fake.stream_script = [
        tool_call_result("write_file", {"path": "second.txt", "content": "second"}),
        ChatResult(content="Second write done."),
    ]
    events, _raw = chat_events(env, "write second.txt")
    assert "approval" not in [event for event, _ in events]
    assert (env.workspace / "second.txt").read_text(encoding="utf-8") == "second"


def test_chat_smart_autonomy_runs_workspace_write_without_asking(env: Env):
    env.fake.stream_script = [
        tool_call_result("write_file", {"path": "smart.txt", "content": "smart"}),
        ChatResult(content="Written."),
    ]
    events, _raw = chat_events(env, "write smart.txt")
    assert "approval" not in [event for event, _ in events]
    call = next(payload for event, payload in events if event == "tool_call")
    assert (call["requires_approval"], call["level"], call["source"]) == (False, "low", "policy")
    assert (env.workspace / "smart.txt").read_text(encoding="utf-8") == "smart"


def test_chat_critical_command_is_denied_without_asking(env: Env):
    env.fake.stream_script = [
        tool_call_result("run_command", {"command": "rm -rf ~"}),
        ChatResult(content="I won't do that."),
    ]
    events, _raw = chat_events(env, "clean up my home folder")
    assert "approval" not in [event for event, _ in events]
    call = next(payload for event, payload in events if event == "tool_call")
    assert call["level"] == "critical" and call["reasons"]
    assert next(payload for event, payload in events if event == "tool_result")["ok"] is False
    messages = env.svc.db.list_messages(env.svc.db.list_conversations()[0]["id"])
    assert "Blocked by Kiwibro" in next(m for m in messages if m["role"] == "tool")["content"]


def test_chat_denied_tool_is_not_executed(env: Env):
    env.settings_patch({"permissions": {"write_file": "deny"}})
    env.fake.stream_script = [
        tool_call_result("write_file", {"path": "denied.txt", "content": "nope"}),
        ChatResult(content="I could not write the file."),
    ]
    events, _raw = chat_events(env, "write denied.txt")
    names = [event for event, _ in events]
    assert "approval" not in names
    tool_result = next(payload for event, payload in events if event == "tool_result")
    assert tool_result["ok"] is False
    assert not (env.workspace / "denied.txt").exists()


def test_chat_busy_returns_409_and_stop_aborts(env: Env):
    gate = threading.Event()
    entered = threading.Event()

    def blocking_stream() -> ChatResult:
        entered.set()
        assert gate.wait(5)
        return ChatResult(content="late reply")

    env.fake.stream_script = [blocking_stream]
    response = env.client.post("/api/chat", json={"message": "slow turn"}, buffered=False)
    assert response.status_code == 200
    stream = SSEStream(response).start()
    assert entered.wait(5), "fake stream never started"
    conversation_id = stream.wait_for("meta")["conversation_id"]

    busy = env.new_client().post("/api/chat", json={"message": "second", "conversation_id": conversation_id})
    assert busy.status_code == 409
    assert busy.get_json()["error"]["code"] == "busy"

    stopped = env.new_client().post("/api/chat/stop", json={"conversation_id": conversation_id})
    assert stopped.get_json() == {"ok": True}
    gate.set()
    events = stream.finish()
    assert events[-1][0] == "done"
    assert events[-1][1]["aborted"] is True


def test_delete_conversation_during_turn_closes_quietly(env: Env):
    gate = threading.Event()
    entered = threading.Event()

    def blocking_stream() -> ChatResult:
        entered.set()
        assert gate.wait(5)
        return ChatResult(content="late reply")

    env.fake.stream_script = [blocking_stream]
    response = env.client.post("/api/chat", json={"message": "delete me"}, buffered=False)
    assert response.status_code == 200
    stream = SSEStream(response).start()
    assert entered.wait(5)
    conversation_id = stream.wait_for("meta")["conversation_id"]
    deleted = env.new_client().delete(f"/api/conversations/{conversation_id}")
    assert deleted.get_json() == {"ok": True}
    assert env.new_client().get(f"/api/conversations/{conversation_id}").status_code == 404
    gate.set()
    stream.finish()
    internal = [payload for event, payload in stream.events if event == "error" and payload.get("code") == "internal"]
    assert internal == []


def test_chat_stop_saves_partial_text(env: Env):
    env.fake.stream_script = [("abort_partial",)]
    events, _raw = chat_events(env, "tell me a story")
    assert events[-1][0] == "done"
    assert events[-1][1]["aborted"] is True
    conversation = env.svc.db.list_conversations()[0]
    messages = env.svc.db.list_messages(conversation["id"])
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "partial text before abort"
    assert messages[-1]["meta"]["aborted"] is True


def test_chat_stream_error_saves_partial_and_reports(env: Env):
    env.fake.stream_script = [("partial_error",)]
    events, _raw = chat_events(env, "This will fail")
    names = [event for event, _ in events]
    assert names[-1] == "error"
    error = events[-1][1]
    assert error["code"] == "stream"
    assert error["recoverable"] is True
    conversation = env.svc.db.list_conversations()[0]
    last = env.svc.db.list_messages(conversation["id"])[-1]
    assert last["content"] == "partial text before the error"


def test_chat_title_is_deterministic_in_stream(env: Env):
    env.fake.stream_script = [ChatResult(content="Hello!")]
    events, _raw = chat_events(env, "Create a file called notes.txt")
    title_event = next(payload for event, payload in events if event == "title")
    assert title_event["title"] == "Create a file called notes.txt"


def test_memory_extraction_runs_after_done_without_memory_event(env: Env):
    env.fake.memory_ops = {
        "ops": [{"op": "add", "kind": "fact", "content": "User is testing Kiwibro"}]
    }
    env.fake.stream_script = [ChatResult(content="Noted.")]
    events, _raw = chat_events(env, "I am testing Kiwibro")
    assert "memory" not in [event for event, _ in events]
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and env.svc.db.count_memories() == 0:
        time.sleep(0.02)
    assert env.svc.db.count_memories() == 1
    assert env.svc.db.list_memories()[0]["content"] == "User is testing Kiwibro"


def test_model_swap_keeps_memories_and_transcripts(env: Env):
    env.fake.models = [dict(MODEL), {**MODEL, "id": "fake-2", "state": "not-loaded"}]
    env.client.post("/api/memories", json={"content": "User is a backend engineer"})
    env.fake.stream_script = [ChatResult(content="First reply")]
    events, _raw = chat_events(env, "first", model="fake-1")
    conversation_id = events[0][1]["conversation_id"]
    assert env.client.post("/api/models/select", json={"id": "fake-2"}).status_code == 200

    env.fake.stream_script = [ChatResult(content="Second reply")]
    chat_events(env, "second", conversation_id=conversation_id)
    detail = env.client.get(f"/api/conversations/{conversation_id}").get_json()
    assert [message["content"] for message in detail["messages"] if message["role"] == "assistant"] == [
        "First reply",
        "Second reply",
    ]
    assert detail["messages"][-1]["model"] == "fake-2"
    assert env.client.get("/api/memories").get_json()["total"] == 1


def test_remember_tool_writes_memory_immediately(env: Env):
    from kiwibro.memory import MemoryEngine

    engine: MemoryEngine = env.svc.memory
    context = tools.ToolContext(
        workspace=Workspace(env.workspace),
        memory=engine,
        search=env.svc.search,
        conversation_id="c_test",
    )
    result = tools.execute_tool("remember", {"content": "User's cat is named Pixel", "kind": "person"}, context)
    assert result.ok is True
    assert env.svc.db.count_memories() == 1
    assert env.svc.db.list_memories()[0]["kind"] == "person"


def test_tools_never_raise_on_workspace_escape(env: Env):
    context = tools.ToolContext(
        workspace=Workspace(env.workspace),
        memory=env.svc.memory,
        search=env.svc.search,
        conversation_id="c_test",
    )
    result = tools.execute_tool("read_file", {"path": "../../etc/passwd"}, context)
    assert result.ok is False
    assert "outside the workspace" in result.content
