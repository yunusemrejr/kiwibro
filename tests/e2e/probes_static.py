"""Fast in-process probes for the backend safety/logic units.

These complement the black-box HTTP checks: they pin down the pure functions
that the tool pipeline relies on (sandbox, SSRF guard, validation, context math,
memory dedupe, DB behaviour) without starting a server. Every probe here is an
independent positive assertion; no network access is required.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from harness import REPO_ROOT, assert_eq, assert_in, assert_true

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kiwibro import config  # noqa: E402
from kiwibro import context as context_engine  # noqa: E402
from kiwibro import db as db_module  # noqa: E402
from kiwibro import llm as llm_module  # noqa: E402
from kiwibro import memory as memory_engine  # noqa: E402
from kiwibro import sandbox, search  # noqa: E402
from kiwibro.agent import derive_title, validate_llm_title  # noqa: E402


# ---------------------------------------------------------------------------
# sandbox
# ---------------------------------------------------------------------------


def sandbox_paths(env: Any = None) -> str:
    with tempfile.TemporaryDirectory() as temp:
        workspace = sandbox.Workspace(Path(temp) / "ws")
        workspace.root.mkdir(parents=True, exist_ok=True)
        (workspace.root / "ok.txt").write_text("hello", encoding="utf-8")
        assert_eq(workspace.read_file("ok.txt")["content"], "hello", "read inside workspace")
        for bad in ("../escape.txt", "/etc/passwd", "a/../../b.txt"):
            try:
                workspace.resolve(bad)
            except sandbox.SandboxError:
                pass
            else:
                raise AssertionError(f"path {bad!r} escaped the workspace")
        # symlink pointing outside must be rejected after resolution
        link = workspace.root / "link-out"
        link.symlink_to("/etc/passwd")
        try:
            workspace.read_file("link-out")
        except sandbox.SandboxError:
            pass
        else:
            raise AssertionError("symlink escape was not rejected")
        written = workspace.write_file("nested/dir/file.txt", "content")
        assert_eq(written["path"], "nested/dir/file.txt", "relative path in write result")
        assert_true((workspace.root / "nested" / "dir" / "file.txt").is_file(), "nested write")
        try:
            workspace.write_file("huge.txt", "x" * (config.MAX_FILE_WRITE_BYTES + 1))
        except sandbox.SandboxError as exc:
            assert_in("too large", str(exc).lower(), "write size cap message")
        else:
            raise AssertionError("oversized write was not rejected")
        listed = workspace.list_files(".")
        paths = [entry["path"] for entry in listed]
        assert_in("ok.txt", paths, "list_files sees files")
        assert_true(not any(path.startswith(".") for path in paths), "hidden dirs pruned")
    return "traversal, absolute paths, symlink escapes, write cap all refused"


def sandbox_commands(env: Any = None) -> str:
    blocked = [
        "sudo ls",
        "rm -rf /",
        "mkfs.ext4 /dev/sda1",
        "shutdown now",
        ":(){ :|:& };:",
        "dd if=/dev/zero of=/dev/sda",
        "chmod 777 /",
        "kill -9 1",
    ]
    for command in blocked:
        try:
            sandbox.check_command(command)
        except sandbox.SandboxError:
            continue
        raise AssertionError(f"command {command!r} was not blocked")
    safe = ["ls -la", "pwd", "echo 'kiwi'", "git status", "python3 -c 'print(1)'"]
    for command in safe:
        sandbox.check_command(command)

    with tempfile.TemporaryDirectory() as temp:
        workspace = sandbox.Workspace(Path(temp) / "ws")
        workspace.root.mkdir(parents=True, exist_ok=True)
        result = workspace.run_command("printf 'kiwi\\n'")
        assert_eq(result["exit_code"], 0, "safe command exit code")
        assert_in("kiwi", result["output"], "safe command output")
        timed = workspace.run_command("sleep 2", timeout=1)
        assert_eq(timed["timed_out"], True, "timeout enforced")
        capped = workspace.run_command("python3 -c \"print('x' * 7000)\"")
        assert_true(len(capped["output"].encode()) <= config.MAX_COMMAND_OUTPUT_BYTES + 20, "output cap")
        assert_eq(workspace.public(workspace.root / "x"), "x", "public() relativises")
    return "blocklist, safe commands, timeout and output cap"


# ---------------------------------------------------------------------------
# SSRF guard (no network needed)
# ---------------------------------------------------------------------------


def ssrf_guard(env: Any = None) -> str:
    rejected = [
        "localhost",
        "localhost.localdomain",
        "127.0.0.1",
        "127.1.2.3",
        "0.0.0.0",
        "10.0.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "::1",
        "[::1]",
        "fe80::1",
        "fd00::1",
        "metadata.google.internal",
    ]
    failures = []
    for host in rejected:
        try:
            search.assert_public_host(host.strip("[]"))
        except search.SearchError:
            continue
        failures.append(host)
    assert_eq(failures, [], "private/loopback hosts all rejected")
    for host in ("93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"):
        search.assert_public_host(host)
    # fetch_page refuses without touching the network (would raise immediately)
    engine = search.SearchEngine()
    for url in (
        "http://127.0.0.1:1234/api/v0/models",
        "http://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
        "http://[::1]/",
    ):
        started = time.monotonic()
        try:
            engine.fetch_page(url)
        except search.SearchError:
            pass
        else:
            raise AssertionError(f"fetch_page did not refuse {url!r}")
        assert_true(time.monotonic() - started < 2, f"refusal for {url!r} is instant")
    return f"{len(rejected)} private targets + file:// and ::1 refused before sockets"


# ---------------------------------------------------------------------------
# context math
# ---------------------------------------------------------------------------


def context_math(env: Any = None) -> str:
    assert_eq(context_engine.compute_budget(32768, 2048), 30208, "budget for 32K context")
    assert_eq(context_engine.compute_budget(2048, 2048), 1024, "budget never below 1024")
    assert_eq(context_engine.compute_budget(1000, 256), 1024, "small context floor")
    stats = context_engine.context_stats(400, 1000)
    assert_eq(stats["percent"], 40, "percent")
    assert_eq(context_engine.context_stats(5000, 1000)["percent"], 100, "percent clamped at 100")
    assert_eq(context_engine.context_stats(-5, 1000)["percent"], 0, "percent clamped at 0")
    assert_eq(context_engine.estimate_tokens(""), 0, "empty tokens")
    ascii_tokens = context_engine.estimate_tokens("a" * 360)
    assert_true(ascii_tokens == 100, f"ascii estimate ~3.6 chars/token (got {ascii_tokens})")
    # tool transcripts are capped when rebuilt for the model
    long_tool = {"role": "tool", "content": "x" * 9000, "tool_call_id": "call_1"}
    rebuilt = context_engine.to_llm_message(long_tool)
    assert_true(len(rebuilt["content"]) < 9000, "tool content capped for the prompt")
    # assistant tool-call arguments normalised to a JSON string
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": {"path": "a"}}}],
    }
    rebuilt_assistant = context_engine.to_llm_message(assistant)
    assert_true(isinstance(rebuilt_assistant["tool_calls"][0]["function"]["arguments"], str), "args stringified")
    # orphan tool messages are trimmed from the start of a window
    window = [{"role": "tool", "content": "x"}, {"role": "user", "content": "hi"}]
    assert_eq(context_engine._clean_leading_boundary(window)[0]["role"], "user", "orphan tool dropped")
    return "budget, percent clamps, token estimate, tool caps, window boundaries"


# ---------------------------------------------------------------------------
# memory dedupe & extraction parsing
# ---------------------------------------------------------------------------


def memory_units(env: Any = None) -> str:
    assert_true(memory_engine.is_similar("Prefers dark themes", "prefers dark theme!"), "dedupe case/punctuation")
    assert_true(memory_engine.is_similar("Likes kiwi green UI", "Likes kiwi-green UI"), "dedupe punctuation")
    assert_true(
        not memory_engine.is_similar("Prefers dark themes", "Works on a Rust project"),
        "unrelated memories stay separate",
    )
    assert_eq(memory_engine.normalize("  Hello,   World!! "), "hello world", "normalization")
    assert_eq(memory_engine.validate_content("  spaced   out  "), "spaced out", "content collapse")
    long_content = memory_engine.validate_content("x" * 400)
    assert_eq(len(long_content), 300, "content capped at 300")
    for bad in ("", "x", "  "):
        try:
            memory_engine.validate_content(bad)
        except ValueError:
            continue
        raise AssertionError(f"memory content {bad!r} should be rejected")
    ops = memory_engine.parse_ops(
        '```json\n{"ops":[{"op":"add","kind":"fact","content":"Uses Ubuntu 24.04"}]}\n```'
    )
    assert_eq(ops[0]["op"], "add", "fenced JSON ops parsed")
    assert_eq(memory_engine.parse_ops("not json at all"), [], "garbage parses to []")
    assert_eq(memory_engine.parse_ops('{"ops":[]}'), [], "empty ops")
    return "similarity thresholds, normalization, caps, defensive JSON ops parsing"


# ---------------------------------------------------------------------------
# DB / FTS
# ---------------------------------------------------------------------------


def database_units(env: Any = None) -> str:
    with tempfile.TemporaryDirectory() as temp:
        database = db_module.Database(Path(temp) / "test.db")
        database.initialize()
        conversation = database.create_conversation(title="DB test", model="m1")
        user = database.add_message(conversation["id"], "user", "hello")
        assistant = database.add_message(conversation["id"], "assistant", "hi")
        assert_eq((user["seq"], assistant["seq"]), (1, 2), "monotonic seq")
        assert_eq(database.conversation_message_count(conversation["id"]), 2, "message count")
        memory = database.create_memory("fact", "Prefers dark themes", "prefers dark themes")
        assert_eq(database.search_memories("dark")[0]["id"], memory["id"], "FTS find")
        database.update_memory(memory["id"], content="Prefers light themes", normalized="prefers light themes")
        assert_eq(database.search_memories("light")[0]["id"], memory["id"], "FTS updated on edit")
        assert_eq(database.search_memories("dark"), [], "stale FTS row removed")
        assert_eq(database.search_memories("!!! ??? ---"), [], "hostile FTS query is safe")
        assert_eq(db_module.fts_query("dark themes"), '"dark" OR "themes"', "FTS query quoting")
        assert_true(database.delete_conversation(conversation["id"]), "conversation deleted")
        assert_eq(database.list_messages(conversation["id"]), [], "cascade removed messages")
        assert_eq(database.delete_all_memories(), 1, "forget all count")
    return "seq allocation, FTS sync on edits, hostile queries, cascade delete"


# ---------------------------------------------------------------------------
# settings & llm helpers
# ---------------------------------------------------------------------------


def settings_units(env: Any = None) -> str:
    defaults = config.default_settings(lmstudio_url="http://127.0.0.1:1234/", workspace="/tmp/ws")
    assert_eq(defaults["lmstudio_url"], "http://127.0.0.1:1234", "trailing slash trimmed")
    cleaned = config.validate_settings_patch(
        {"temperature": 0.3, "max_output_tokens": 1024, "permissions": {"write_file": "auto"}},
        defaults,
    )
    assert_eq(cleaned["temperature"], 0.3, "valid patch")
    failures = [
        {"temperature": -1},
        {"temperature": 2.5},
        {"max_output_tokens": 100},
        {"permissions": {"write_file": "maybe"}},
        {"workspace": "/"},
        {"lmstudio_url": "file:///etc/passwd"},
        {"surprise": True},
    ]
    for patch in failures:
        try:
            config.validate_settings_patch(patch, defaults)
        except ValueError:
            continue
        raise AssertionError(f"settings patch {patch!r} should be rejected")
    return "settings validation accepts/rejects the documented ranges"


def llm_helpers(env: Any = None) -> str:
    model = llm_module.normalize_model(
        {
            "id": "m",
            "type": "llm",
            "state": "loaded",
            "loaded_context_length": 2048,
            "max_context_length": 8192,
            "capabilities": ["tool_use"],
        }
    )
    assert_eq(model["context"], 2048, "loaded context")
    assert_eq(model["tools"], True, "tool capability")
    assert_eq(llm_module.normalize_model({"id": "e", "type": "embeddings"}), None, "embeddings excluded")
    parsed = llm_module.parse_json_loose('prose before ```json\n{"ops": []}\n``` after')
    assert_eq(parsed, {"ops": []}, "loose JSON extraction")
    assert_eq(llm_module.parse_json_loose("nothing"), None, "unparseable JSON")
    assert_eq(
        type(llm_module.classify_http_error(400, "context length exceeded")).__name__,
        "ContextLimitError",
        "context 400 classified",
    )
    assert_eq(
        type(llm_module.classify_http_error(400, "tools not supported")).__name__,
        "ToolsUnsupportedError",
        "tools 400 classified",
    )
    assert_eq(llm_module.classify_http_error(500, "boom").kind, "server", "5xx classified")
    return "model normalisation, loose JSON, HTTP error classification"


def title_derivation(env: Any = None) -> str:
    cases = {
        "Please can you explain the kiwi bird?": "Explain the kiwi bird",
        "Search the web for kiwi conservation news": "Kiwi conservation news",
        "  hello   world  ": "World",
        "": "New chat",
    }
    for message, expected in cases.items():
        actual = derive_title(message)
        assert_eq(actual, expected, f"derive_title({message!r})")
    assert_eq(validate_llm_title("Kiwi Facts", prompt_text="x" * 10), "Kiwi Facts", "clean LLM title")
    for bad in (
        "First user message: hello there",
        "You name conversations and write titles",
        "Title:",
        "a",
        "one two three four five six seven",
        "Sure, here is a title",
    ):
        assert_eq(validate_llm_title(bad, prompt_text="You name conversations"), None, f"reject {bad!r}")
    return "deterministic titles and prompt-echo rejection"


# ---------------------------------------------------------------------------
# registry used by run_e2e.py --audit / probes section
# ---------------------------------------------------------------------------

PROBES: list[tuple[str, Callable[..., str]]] = [
    ("sandbox_paths", sandbox_paths),
    ("sandbox_commands", sandbox_commands),
    ("ssrf_guard", ssrf_guard),
    ("context_math", context_math),
    ("memory_units", memory_units),
    ("database_units", database_units),
    ("settings_units", settings_units),
    ("llm_helpers", llm_helpers),
    ("title_derivation", title_derivation),
]
