"""Tests for the 2.x tool layer: registry/profiles, codebase, research,
desktop degradation, delegation picker, snippet sandbox, safeguards."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kiwibro import codebase, config, delegate, desktop, research, tools  # noqa: E402
from kiwibro.sandbox import SandboxError, Workspace  # noqa: E402
from kiwibro.search import SearchEngine, dedupe_results, normalize_url, rerank_results  # noqa: E402
from kiwibro.tools import ToolContext  # noqa: E402


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.py").write_text("import os\n\n\ndef hello(name):\n    return name\n\n\nclass Thing:\n    def run(self):\n        pass\n", encoding="utf-8")
    (root / "b.md").write_text("# Title\n\n## Section\nhello world\n", encoding="utf-8")
    sub = root / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("needle in a haystack\nsecond line\n", encoding="utf-8")
    return Workspace(root)


# --- registry & profiles -------------------------------------------------------


def test_registry_permissions_parity():
    builtin = set(tools.tool_names()) - set(getattr(tools, "_EXTENSIONS", {}))
    assert builtin == set(config.DEFAULT_PERMISSIONS)
    assert len(tools.tool_names()) >= 35
    for name in tools.tool_names():
        spec = tools.get_spec(name)
        assert spec.description and spec.parameters["type"] == "object"
        assert tools.phase_for(name)[0]
        assert isinstance(tools.summary_for(name, {}), str)


def test_tool_profiles():
    assert tools.select_tool_names("minimal") == [
        n for n in tools.tool_names()
        if n in {"web_search", "fetch_page", "read_file", "list_files", "write_file", "replace_in_file", "run_command", "remember"}
    ]
    assert len(tools.select_tool_names("full")) == len(tools.tool_names())
    # auto = one fixed set: the tool list renders before every message (KV-cache prefix)
    auto = tools.select_tool_names("auto")
    assert auto == [n for n in tools.tool_names() if n in tools.AUTO_TOOLS]
    assert 12 <= len(auto) <= 22 and "web_research" in auto and "grep_files" in auto
    settings = {"tool_profile": "auto", "autonomy": "supervised", "permissions": {"git": "deny"}}
    line = tools.more_tools_line(settings)
    assert "apply_patch" in line and "git" not in line.split(":", 1)[1].replace(",", " ").split()
    assert not any(name in line.split(":", 1)[1] for name in ("web_research,", "grep_files,"))
    assert tools.more_tools_line({"tool_profile": "full"}) == ""
    assert len(tools.tool_defs_for("minimal")) == 8
    denied = tools.tool_defs_for("full", settings={"autonomy": "supervised", "permissions": {"run_command": "deny"}})
    assert all(item["function"]["name"] != "run_command" for item in denied)
    # Automatic modes ignore the per-tool toggles (they are Manual mode's).
    automatic = tools.tool_defs_for("full", settings={"autonomy": "smart", "permissions": {"run_command": "deny"}})
    assert any(item["function"]["name"] == "run_command" for item in automatic)


def test_validation_rejects_bad_args():
    with pytest.raises(ValueError):
        tools.validate_args("multi_search", {"queries": []})
    with pytest.raises(ValueError):
        tools.validate_args("delegate", {"task": "x", "capability": "telepathy"})
    with pytest.raises(ValueError):
        tools.validate_args("apply_patch", {"path": "a", "edits": [{"old": "x"}]})
    with pytest.raises(ValueError):
        tools.validate_args("browser", {"action": "open", "url": "ftp://x"})
    with pytest.raises(ValueError):
        tools.validate_args("terminal_session", {"action": "status"})
    ok = tools.validate_args("task_board", {"action": "add", "title": "t", "priority": "high"})
    assert ok["action"] == "add" and ok["priority"] == "high"


# --- codebase ---------------------------------------------------------------------


def test_glob_outline_and_atomic_patch(ws: Workspace):
    found = codebase.glob_files(ws, "**/*.py")
    assert [item["path"] for item in found["results"]] == ["a.py"]
    outline = codebase.file_outline(ws, "a.py")
    joined = "\n".join(outline["entries"])
    assert "def hello" in joined and "class Thing" in joined
    patched = codebase.apply_patch(ws, "a.py", [{"old": "def hello", "new": "def greet"}])
    assert patched["edits"] == 1 and "def greet" in patched["diff"]
    assert not (ws.root / ".kiwibro-backups").exists()  # undo lives in kiwibro.checkpoints
    with pytest.raises(SandboxError):
        codebase.apply_patch(ws, "a.py", [{"old": "missing", "new": "x"}])
    # failed patch changed nothing
    assert "def greet(name)" in (ws.root / "a.py").read_text(encoding="utf-8")


def test_file_manage_and_mass_delete_guard(ws: Workspace, tmp_path: Path):
    codebase.file_manage(ws, "mkdir", "newdir")
    assert (ws.root / "newdir").is_dir()
    codebase.file_manage(ws, "copy", "b.md", destination="newdir/b2.md")
    assert (ws.root / "newdir/b2.md").exists()
    with pytest.raises(SandboxError):
        codebase.file_manage(ws, "delete", "newdir")  # non-empty, not recursive
    big = ws.root / "big"
    big.mkdir()
    for i in range(codebase.MAX_RECURSIVE_DELETE_ENTRIES + 5):
        (big / f"f{i}.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SandboxError):
        codebase.file_manage(ws, "delete", "big", recursive=True)
    deleted = codebase.file_manage(ws, "delete", "newdir/b2.md")
    assert deleted["action"] == "delete" and not (ws.root / "newdir/b2.md").exists()
    with pytest.raises(SandboxError):
        codebase.file_manage(ws, "delete", ".")


def test_run_snippet_python(ws: Workspace):
    result = codebase.run_snippet(ws, "python", "print(6 * 7)\nopen('out.txt', 'w').write('hi')")
    assert result["exit_code"] == 0 and "42" in result["output"]
    assert any(path.endswith("out.txt") for path in result["created_files"])
    timeout = codebase.run_snippet(ws, "python", "import time; time.sleep(5)", timeout=1)
    assert timeout["timed_out"] is True
    with pytest.raises(SandboxError):
        codebase.run_snippet(ws, "python", "   ")


def test_git_info(ws: Workspace):
    subprocess.run(["git", "init", "-q"], cwd=ws.root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"], cwd=ws.root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"], cwd=ws.root, check=True)
    log = codebase.git_info(ws, "log", limit=5)
    assert "init" in log["output"]
    status = codebase.git_info(ws, "status")
    assert status["action"] == "status"
    with pytest.raises(SandboxError):
        codebase.git_info(ws, "show", ref="bad ref with spaces")


# --- search helpers & research -------------------------------------------------------


def test_url_normalise_dedupe_rerank():
    assert normalize_url("https://www.Example.com/a/") == "https://example.com/a"
    deduped = dedupe_results([
        {"title": "A", "url": "https://www.x.com/a/", "snippet": ""},
        {"title": "B", "url": "https://x.com/a", "snippet": ""},
    ])
    assert len(deduped) == 1
    ranked = rerank_results("python release", [
        {"title": "Cooking", "url": "u1", "snippet": "food"},
        {"title": "Python release notes", "url": "u2", "snippet": "python"},
    ])
    assert ranked[0]["url"] == "u2"


class FakeSearch:
    def web_search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        return [{"title": f"R {query}", "url": f"https://x.com/{query.replace(' ', '-')}", "snippet": "s"}]

    def fetch_page(self, url: str, max_chars: int = 4000) -> dict[str, Any]:
        return {"title": "T", "url": url, "text": f"body of {url}", "truncated": False}


def test_research_pack():
    bundle = research.web_research(FakeSearch(), "kiwi fruit", depth=1)  # type: ignore[arg-type]
    assert "[1]" in bundle["evidence"] and "body of https://x.com" in bundle["evidence"]
    multi = research.multi_search(FakeSearch(), ["one", "two"])  # type: ignore[arg-type]
    assert multi["count"] == 2


def test_fetch_cache(monkeypatch: pytest.MonkeyPatch):
    engine = SearchEngine()
    calls = {"n": 0}

    def fake_get(url: str, **kwargs):
        calls["n"] += 1
        return url, "text/html", "utf-8", b"<html><title>T</title><body><p>hello world this is text</p></body></html>"

    monkeypatch.setattr(engine, "_guarded_get", fake_get)
    first = engine.fetch_page("https://example.com/x")
    second = engine.fetch_page("https://example.com/x")
    assert first["text"] == second["text"] and calls["n"] == 1


# --- desktop degradation ---------------------------------------------------------------


def test_desktop_system_info_and_missing_tools(ws: Workspace, monkeypatch: pytest.MonkeyPatch):
    info = desktop.system_info(ws)
    assert info["workspace_files"] >= 3 and "os" in info
    monkeypatch.setattr(desktop.shutil, "which", lambda _name: None)
    with pytest.raises(desktop.DesktopError):
        desktop.clipboard("read")
    with pytest.raises(desktop.DesktopError):
        desktop.open_path(ws, "https://example.com")
    with pytest.raises(desktop.DesktopError):
        desktop.screenshot(ws)


# --- delegation ---------------------------------------------------------------------------


MODELS = [
    {"id": "big-70b", "name": "Big 70B", "state": "loaded", "context": 32768, "max_context": 32768, "type": "llm", "vision": False, "capabilities": ["tool_use"], "tools": True},
    {"id": "tiny-vl-2b", "name": "Tiny VL 2B", "state": "not-loaded", "context": 8192, "max_context": 8192, "type": "vlm", "vision": True, "capabilities": ["vision"], "tools": False},
    {"id": "qwen-coder-7b", "name": "Qwen Coder 7B", "state": "not-loaded", "context": 32768, "max_context": 131072, "type": "llm", "vision": False, "capabilities": [], "tools": False},
]


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def models_status(self, force: bool = False):
        return True, [dict(m) for m in MODELS], None

    def list_models(self, force: bool = False):
        return [dict(m) for m in MODELS]

    def complete(self, *, model: str, messages: Any, **kwargs: Any):
        self.calls.append(model)
        from kiwibro.llm import Completion

        return Completion(text=f"answer from {model}", usage=None)


def test_pick_helper_smallest_capable():
    assert delegate.pick_helper(MODELS, "vision")["id"] == "tiny-vl-2b"
    assert delegate.pick_helper(MODELS, "coding")["id"] == "qwen-coder-7b"
    assert delegate.pick_helper(MODELS, "long_context")["id"] == "qwen-coder-7b"
    assert delegate.pick_helper(MODELS, "audio") is None
    assert "vision" in delegate.capabilities_of(MODELS[1])


def test_run_helper_keeps_current_when_adequate():
    llm = FakeLLM()
    result = delegate.run_helper(llm, "general", [{"role": "user", "content": "hi"}], current_model="big-70b")  # type: ignore[arg-type]
    assert result["model"] == "big-70b"
    vision = delegate.run_helper(llm, "vision", [{"role": "user", "content": "look"}], current_model="big-70b")  # type: ignore[arg-type]
    assert vision["model"] == "tiny-vl-2b"
    with pytest.raises(delegate.DelegateError):
        delegate.run_helper(llm, "audio", [{"role": "user", "content": "listen"}], current_model="big-70b")  # type: ignore[arg-type]


def test_study_text_map_reduce():
    llm = FakeLLM()
    text = "\n\n".join(f"Paragraph {i} about topic {i}." + " filler" * 120 for i in range(6))
    result = delegate.study_text(llm, text, "What topics appear?", current_model="big-70b")  # type: ignore[arg-type]
    assert result["chunks"] >= 2 and "answer from" in result["text"]
    chunks = delegate.chunk_text("short")
    assert chunks == ["short"]


def test_execute_tool_end_to_end(ws: Workspace):
    class FakeMemory:
        def retrieve(self, query: str, limit: int = 6):
            return [{"content": "User likes kiwis", "kind": "preference"}]

    ctx = ToolContext(
        workspace=ws, memory=FakeMemory(), search=FakeSearch(),  # type: ignore[arg-type]
        conversation_id="c1", llm=FakeLLM(), current_model="big-70b",  # type: ignore[arg-type]
    )
    recall = tools.execute_tool("recall", tools.validate_args("recall", {"query": "kiwis"}), ctx)
    assert recall.ok and "kiwis" in recall.content
    info = tools.execute_tool("model_info", {}, ctx)
    assert info.ok and "tiny-vl-2b" in info.content and "vision" in info.content
    snippet = tools.execute_tool(
        "run_python", tools.validate_args("run_python", {"code": "print(2+2)"}), ctx
    )
    assert snippet.ok and "4" in snippet.content
    board = tools.execute_tool(
        "task_board", tools.validate_args("task_board", {"action": "add", "title": "write tests"}), ctx
    )
    assert board.ok and "write tests" in board.content
    missing = tools.execute_tool("read_file", {"path": "nope.txt", "start": None, "end": None}, ctx)
    assert missing.ok is False  # errors are results, never exceptions
