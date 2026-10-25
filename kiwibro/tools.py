"""Tool catalog, argument validation, permission evaluation and execution.

Tools are grouped into categories (``core``, ``research``, ``coding``,
``desktop``) so the agent can send a task-shaped subset of schemas to small
local models instead of the whole catalog — see :func:`tool_defs_for`.
Read-only/web tools are marked ``parallel=True``: when a model asks for
several in one round the agent may run them concurrently.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from . import codebase, config, delegate, desktop, research, safety
from .delegate import DelegateError
from .desktop import DesktopError
from .llm import LMStudioClient
from .memory import MemoryEngine
from .sandbox import TERMINALS, SandboxError, Workspace
from .search import SearchEngine, SearchError

log = logging.getLogger(__name__)


@dataclass
class ToolContext:
    workspace: Workspace
    memory: MemoryEngine
    search: SearchEngine
    conversation_id: str
    llm: LMStudioClient | None = None
    current_model: str | None = None


@dataclass
class ToolResult:
    ok: bool
    content: str
    preview: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    risk: str
    phase: str
    label: str
    summary: Callable[[dict[str, Any]], str]
    category: str = "core"
    parallel: bool = False


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _str(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _int(description: str, low: int, high: int) -> dict[str, Any]:
    return {"type": "integer", "description": description, "minimum": low, "maximum": high}


def _bool(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description}


def _str_list(description: str, max_items: int) -> dict[str, Any]:
    return {
        "type": "array",
        "description": description,
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": max_items,
    }


_SPECS: tuple[ToolSpec, ...] = (
    # --- core web ------------------------------------------------------------
    ToolSpec(
        name="web_search",
        description="Search the web and return titles, URLs and snippets. Use for current facts.",
        parameters=_schema(
            {
                "query": _str("Search query"),
                "max_results": _int("How many results (1-8, default 5)", 1, 8),
            },
            ["query"],
        ),
        risk="web",
        phase="web",
        label="Searching the web",
        summary=lambda args: f'Search the web for "{args.get("query", "")}"',
        parallel=True,
    ),
    ToolSpec(
        name="fetch_page",
        description="Fetch a public web page and return its readable text.",
        parameters=_schema(
            {
                "url": _str("http(s) URL to read"),
                "max_chars": _int("Max characters of text (200-8000)", 200, 8000),
            },
            ["url"],
        ),
        risk="web",
        phase="reading",
        label="Reading a web page",
        summary=lambda args: f"Read {args.get('url', '')}",
        parallel=True,
    ),
    # --- research (batch/parallel web) ----------------------------------------
    ToolSpec(
        name="multi_search",
        description=(
            "Search the web with several queries at once (parallel). Returns one merged, "
            "deduplicated, ranked result list. Prefer this over repeated web_search calls "
            "when a question has multiple angles."
        ),
        parameters=_schema(
            {
                "queries": _str_list("1-4 search queries", 4),
                "max_results": _int("Results per query (1-8, default 5)", 1, 8),
            },
            ["queries"],
        ),
        risk="web",
        phase="web",
        label="Searching the web",
        summary=lambda args: "Search: " + "; ".join(str(q) for q in args.get("queries", []))[:160],
        category="research",
        parallel=True,
    ),
    ToolSpec(
        name="fetch_pages",
        description=(
            "Fetch several web pages at once (parallel) and return a numbered evidence "
            "pack. Prefer this over repeated fetch_page calls."
        ),
        parameters=_schema(
            {
                "urls": _str_list("1-5 http(s) URLs", 5),
                "max_chars": _int("Max characters per page (200-8000)", 200, 8000),
            },
            ["urls"],
        ),
        risk="web",
        phase="reading",
        label="Reading web pages",
        summary=lambda args: f"Read {len(args.get('urls', []))} pages",
        category="research",
        parallel=True,
    ),
    ToolSpec(
        name="web_research",
        description=(
            "Research a question in one call: searches, ranks results, fetches the top "
            "sources and returns numbered evidence with URLs. Cite sources as [1], [2]…"
        ),
        parameters=_schema(
            {
                "query": _str("Research question or search query"),
                "depth": _int("How many top sources to fetch (1-5, default 3)", 1, 5),
            },
            ["query"],
        ),
        risk="web",
        phase="web",
        label="Researching the web",
        summary=lambda args: f'Research "{args.get("query", "")}"',
        category="research",
        parallel=True,
    ),
    ToolSpec(
        name="browse_page",
        description=(
            "Open a web page like a text browser: readable text plus a numbered list of "
            "links. Follow a link by calling browse_page or fetch_page on its URL."
        ),
        parameters=_schema(
            {
                "url": _str("http(s) URL to browse"),
                "max_chars": _int("Max characters of text (200-8000)", 200, 8000),
            },
            ["url"],
        ),
        risk="web",
        phase="reading",
        label="Browsing a web page",
        summary=lambda args: f"Browse {args.get('url', '')}",
        category="research",
        parallel=True,
    ),
    # --- core files -------------------------------------------------------------
    ToolSpec(
        name="read_file",
        description="Read a text file from the workspace, optionally a line range.",
        parameters=_schema(
            {
                "path": _str("Workspace-relative file path"),
                "start": {"type": "integer", "description": "First line (1-based)"},
                "end": {"type": "integer", "description": "Last line (inclusive)"},
            },
            ["path"],
        ),
        risk="read",
        phase="reading",
        label="Reading a file",
        summary=lambda args: f"Read {args.get('path', '')}",
        parallel=True,
    ),
    ToolSpec(
        name="list_files",
        description="List files and folders in the workspace (depth 2, up to 200 entries).",
        parameters=_schema({"path": _str("Directory (default workspace root)")}),
        risk="read",
        phase="reading",
        label="Listing files",
        summary=lambda args: f"List files in {args.get('path', '.')}",
        parallel=True,
    ),
    ToolSpec(
        name="write_file",
        description="Create or overwrite a text file inside the workspace.",
        parameters=_schema(
            {
                "path": _str("Workspace-relative file path"),
                "content": _str("Full file content"),
            },
            ["path", "content"],
        ),
        risk="write",
        phase="writing",
        label="Writing a file",
        summary=lambda args: (
            f"Write {len(str(args.get('content', '')))} characters to {args.get('path', '')}"
        ),
    ),
    ToolSpec(
        name="replace_in_file",
        description="Replace exactly one occurrence of old text with new text in a workspace file.",
        parameters=_schema(
            {
                "path": _str("Workspace-relative file path"),
                "old": _str("Exact text to replace"),
                "new": _str("Replacement text"),
            },
            ["path", "old", "new"],
        ),
        risk="write",
        phase="writing",
        label="Editing a file",
        summary=lambda args: f"Edit {args.get('path', '')}",
    ),
    ToolSpec(
        name="run_command",
        description="Run a shell command in the workspace (60s limit, output capped).",
        parameters=_schema(
            {
                "command": _str("Command line to run"),
                "cwd": _str("Working directory inside the workspace"),
            },
            ["command"],
        ),
        risk="exec",
        phase="running",
        label="Running a command",
        summary=lambda args: f"Run: {str(args.get('command', ''))[:120]}",
    ),
    ToolSpec(
        name="remember",
        description=(
            "Long-term memory about the user: save a fact they told you, update one that changed "
            "(old = words of the outdated memory), or forget one they ask you to forget."
        ),
        parameters=_schema(
            {
                "content": _str("The fact, self-contained, e.g. 'User prefers Python' (forget: words of the memory)"),
                "action": {
                    "type": "string",
                    "enum": ["save", "update", "forget"],
                    "description": "save (default), update or forget",
                },
                "old": _str("update only: words of the outdated memory, e.g. 'lives in Istanbul'"),
                "kind": {
                    "type": "string",
                    "enum": list(config.MEMORY_KINDS),
                    "description": "Memory kind",
                },
            },
            ["content"],
        ),
        risk="read",
        phase="tool",
        label="Saving a memory",
        summary=lambda args: f"{_MEMORY_VERBS.get(str(args.get('action') or 'save'), 'Remember')}: "
        f"{str(args.get('content', ''))[:80]}",
    ),
    # --- coding / codebase --------------------------------------------------------
    ToolSpec(
        name="read_files",
        description=(
            "Read several workspace files in one call (up to 8), each with a header. "
            "Prefer this over repeated read_file calls when exploring a codebase."
        ),
        parameters=_schema(
            {
                "paths": _str_list("Workspace-relative file paths (1-8)", 8),
                "start": {"type": "integer", "description": "First line for every file (1-based)"},
                "end": {"type": "integer", "description": "Last line for every file (inclusive)"},
            },
            ["paths"],
        ),
        risk="read",
        phase="reading",
        label="Reading files",
        summary=lambda args: "Read " + ", ".join(str(p) for p in args.get("paths", []))[:160],
        category="coding",
        parallel=True,
    ),
    ToolSpec(
        name="glob_files",
        description="Find workspace files by glob pattern, e.g. **/*.py or src/**/*.md.",
        parameters=_schema(
            {
                "pattern": _str("Glob pattern, relative to the workspace"),
                "path": _str("Directory to search under (default workspace root)"),
            },
            ["pattern"],
        ),
        risk="read",
        phase="reading",
        label="Finding files",
        summary=lambda args: f"Find {args.get('pattern', '')}",
        category="coding",
        parallel=True,
    ),
    ToolSpec(
        name="grep_files",
        description=(
            "Search workspace file contents (grep). Supports regex, a filename glob "
            "filter and context lines. Use before editing code you have not read."
        ),
        parameters=_schema(
            {
                "pattern": _str("Text or regular expression to find"),
                "path": _str("File or directory to search (default workspace root)"),
                "glob": _str("Filename glob filter, e.g. *.py"),
                "regex": _bool("Treat pattern as a regular expression (default true)"),
                "case_sensitive": _bool("Case-sensitive search (default false)"),
                "context_lines": _int("Context lines around matches (0-3)", 0, 3),
                "max_results": _int("Max matches (1-100, default 50)", 1, 100),
            },
            ["pattern"],
        ),
        risk="read",
        phase="reading",
        label="Searching files",
        summary=lambda args: f"Search files for “{str(args.get('pattern', ''))[:80]}”",
        category="coding",
        parallel=True,
    ),
    ToolSpec(
        name="file_outline",
        description=(
            "Outline a file without reading it all: Python classes/functions/imports, "
            "JS symbols, or Markdown headings, with line numbers."
        ),
        parameters=_schema({"path": _str("Workspace-relative file path")}, ["path"]),
        risk="read",
        phase="reading",
        label="Outlining a file",
        summary=lambda args: f"Outline {args.get('path', '')}",
        category="coding",
        parallel=True,
    ),
    ToolSpec(
        name="apply_patch",
        description=(
            "Apply several exact-text edits to one file atomically: every edit must "
            "match exactly once or nothing changes. Returns a diff. Prefer this over "
            "many replace_in_file calls on the same file."
        ),
        parameters=_schema(
            {
                "path": _str("Workspace-relative file path"),
                "edits": {
                    "type": "array",
                    "description": "Edits to apply in order (1-20)",
                    "minItems": 1,
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "properties": {
                            "old": _str("Exact text to replace"),
                            "new": _str("Replacement text"),
                        },
                        "required": ["old", "new"],
                        "additionalProperties": False,
                    },
                },
            },
            ["path", "edits"],
        ),
        risk="write",
        phase="writing",
        label="Patching a file",
        summary=lambda args: f"Apply {len(args.get('edits', []))} edits to {args.get('path', '')}",
        category="coding",
    ),
    ToolSpec(
        name="file_manage",
        description="Create a folder, or move/copy/delete a file or folder in the workspace.",
        parameters=_schema(
            {
                "action": {
                    "type": "string",
                    "enum": ["mkdir", "move", "copy", "delete"],
                    "description": "What to do",
                },
                "path": _str("Source path (workspace-relative)"),
                "destination": _str("Destination for move/copy"),
                "recursive": _bool("Delete non-empty folders (delete only)"),
            },
            ["action", "path"],
        ),
        risk="write",
        phase="writing",
        label="Managing files",
        summary=lambda args: (
            f"{str(args.get('action', '')).capitalize()} {args.get('path', '')}"
            + (f" → {args['destination']}" if args.get("destination") else "")
        ),
        category="coding",
    ),
    ToolSpec(
        name="git",
        description=(
            "Inspect a git repository in the workspace (read-only): status, diff stat, "
            "recent log, branches, or show a commit. Use run_command for git writes."
        ),
        parameters=_schema(
            {
                "action": {
                    "type": "string",
                    "enum": ["status", "diff", "log", "show", "branch"],
                    "description": "What to inspect (default status)",
                },
                "path": _str("Repository folder (default workspace root)"),
                "ref": _str("Commit/branch for diff or show"),
                "limit": _int("Log entries (1-50, default 10)", 1, 50),
            },
        ),
        risk="read",
        phase="reading",
        label="Inspecting git",
        summary=lambda args: f"git {args.get('action', 'status')}",
        category="coding",
        parallel=True,
    ),
    ToolSpec(
        name="multi_command",
        description=(
            "Run several shell commands in order (e.g. build then test) with combined "
            "output. Stops at the first failure unless stop_on_error is false."
        ),
        parameters=_schema(
            {
                "commands": _str_list("Command lines to run in order (1-8)", 8),
                "cwd": _str("Working directory inside the workspace"),
                "stop_on_error": _bool("Stop after the first failing command (default true)"),
            },
            ["commands"],
        ),
        risk="exec",
        phase="running",
        label="Running commands",
        summary=lambda args: "Run: " + " && ".join(str(c) for c in args.get("commands", []))[:160],
        category="coding",
    ),
    ToolSpec(
        name="terminal_session",
        description=(
            "Persistent/background terminals in the workspace. start: launch a long-running "
            "process (server, watcher) and get a session id. run: stateful command keeping "
            "cwd. status/input/kill/list: inspect, type into, stop, or list sessions."
        ),
        parameters=_schema(
            {
                "action": {
                    "type": "string",
                    "enum": ["start", "run", "status", "input", "kill", "list"],
                    "description": "Terminal action",
                },
                "command": _str("Command line (start/run)"),
                "session_id": _str("Session id (run/status/input/kill)"),
                "input": _str("Text to type into the session (input)"),
                "cwd": _str("Working directory inside the workspace"),
                "wait_ms": _int("How long to wait for output, ms (0-5000)", 0, 5000),
            },
            ["action"],
        ),
        risk="exec",
        phase="running",
        label="Using a terminal",
        summary=lambda args: (
            f"Terminal {args.get('action', '')}: {str(args.get('command') or args.get('session_id') or '')[:100]}"
        ),
        category="coding",
    ),
    # --- desktop / computer use -----------------------------------------------------
    ToolSpec(
        name="system_info",
        description=(
            "Read-only snapshot of this machine: OS, CPU, load, disk space, desktop "
            "session, local time, and workspace size. Use for environment questions."
        ),
        parameters=_schema({}),
        risk="read",
        phase="tool",
        label="Checking the system",
        summary=lambda args: "Check system information",
        category="desktop",
        parallel=True,
    ),
    ToolSpec(
        name="open_path",
        description=(
            "Open a workspace file/folder or an http(s) URL in the user's desktop "
            "(default app/browser). Use when the user asks to open or show something."
        ),
        parameters=_schema({"target": _str("Workspace path or http(s) URL")}, ["target"]),
        risk="exec",
        phase="tool",
        label="Opening on the desktop",
        summary=lambda args: f"Open {args.get('target', '')}",
        category="desktop",
    ),
    ToolSpec(
        name="clipboard",
        description="Read the desktop clipboard text, or write text to it.",
        parameters=_schema(
            {
                "action": {
                    "type": "string",
                    "enum": ["read", "write"],
                    "description": "Read or write",
                },
                "text": _str("Text to copy (write only)"),
            },
            ["action"],
        ),
        risk="write",
        phase="tool",
        label="Using the clipboard",
        summary=lambda args: (
            "Read the clipboard" if args.get("action") == "read"
            else f"Copy {len(str(args.get('text', '')))} characters to the clipboard"
        ),
        category="desktop",
    ),
    ToolSpec(
        name="notify",
        description="Show a desktop notification (e.g. when a long task finishes).",
        parameters=_schema(
            {
                "title": _str("Notification title"),
                "body": _str("Notification body (optional)"),
            },
            ["title"],
        ),
        risk="read",
        phase="tool",
        label="Sending a notification",
        summary=lambda args: f"Notify: {args.get('title', '')}",
        category="desktop",
    ),
    ToolSpec(
        name="screenshot",
        description="Take a screenshot of the desktop and save it in the workspace.",
        parameters=_schema(
            {"filename": _str("File name/path in the workspace (default screenshots/…png)")}
        ),
        risk="exec",
        phase="tool",
        label="Taking a screenshot",
        summary=lambda args: "Take a screenshot",
        category="desktop",
    ),
    # --- agent meta -------------------------------------------------------------------
    ToolSpec(
        name="task_board",
        description=(
            "Persistent task board for multi-step/autonomous work (survives restarts). "
            "add tasks with priorities/dependencies, update status as you go "
            "(todo → in_progress → done, blocked when stuck), list or summarise at any "
            "time. Keep exactly one task in_progress."
        ),
        parameters=_schema(
            {
                "action": {
                    "type": "string",
                    "enum": ["list", "add", "update", "delete", "clear", "summary"],
                    "description": "Board action (default list)",
                },
                "title": _str("Task title (add/update)"),
                "task_id": _str("Task id (update/delete)"),
                "status": {
                    "type": "string",
                    "enum": ["todo", "in_progress", "blocked", "done"],
                    "description": "Task status filter or new status",
                },
                "notes": _str("Task notes"),
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Task priority",
                },
                "depends_on": _str_list("Task ids this task depends on", 10),
                "only_done": _bool("Clear only finished tasks (clear)"),
            },
        ),
        risk="read",
        phase="tool",
        label="Updating the task board",
        summary=lambda args: f"Tasks: {args.get('action', 'list')}"
        + (f" — {args['title']}" if args.get("title") else ""),
    ),
    ToolSpec(
        name="recall",
        description=(
            "Search long-term memory and earlier conversation for relevant facts. Use "
            "when the user references something from before that is not in view."
        ),
        parameters=_schema(
            {
                "query": _str("What to recall, e.g. 'database choice' or a person's name"),
                "limit": _int("Max items (1-12, default 6)", 1, 12),
            },
            ["query"],
        ),
        risk="read",
        phase="tool",
        label="Recalling memory",
        summary=lambda args: f'Recall "{str(args.get("query", ""))[:80]}"',
        parallel=True,
    ),
    # --- browser tabs -------------------------------------------------------------------
    ToolSpec(
        name="browser",
        description=(
            "Multi-tab text browser with history. open a URL into a tab (rich Markdown "
            "extraction, tables, optional CSS selector), list/switch/close tabs, go "
            "back, or inspect a page without opening a tab. Use for multi-page site work; "
            "use fetch_page/browse_page for one-off reads."
        ),
        parameters=_schema(
            {
                "action": {
                    "type": "string",
                    "enum": ["open", "list", "switch", "close", "back", "inspect"],
                    "description": "Browser action",
                },
                "url": _str("http(s) URL (open/inspect)"),
                "tab_id": _str("Tab id (switch/close)"),
                "selector": _str("Optional CSS selector to extract only part of the page"),
                "max_chars": _int("Max characters of page text (200-8000)", 200, 8000),
            },
            ["action"],
        ),
        risk="web",
        phase="reading",
        label="Browsing the web",
        summary=lambda args: f"Browser {args.get('action', '')}: {str(args.get('url') or args.get('tab_id') or '')[:80]}",
        category="research",
    ),
    # --- model delegation -----------------------------------------------------------------
    ToolSpec(
        name="model_info",
        description=(
            "List every model installed in the local LM Studio with its capabilities "
            "(vision, tools, coding, reasoning, long context…), load state and context "
            "size. Use before delegate when unsure what is available."
        ),
        parameters=_schema({}),
        risk="read",
        phase="tool",
        label="Checking local models",
        summary=lambda args: "List local models and capabilities",
        parallel=True,
    ),
    ToolSpec(
        name="delegate",
        description=(
            "Ask another local model one bounded side-question, chosen by the capability "
            "you need (vision, reasoning, coding, long_context, fast…). The harness picks "
            "the smallest adequate installed model (or keeps the current one if it already "
            "fits) and returns its answer; you stay in charge of the task. Optionally attach "
            "a workspace image for vision questions."
        ),
        parameters=_schema(
            {
                "task": _str("Complete, self-contained question/instruction for the helper"),
                "capability": {
                    "type": "string",
                    "enum": list(delegate.NEEDS),
                    "description": "Capability the helper must have (default general)",
                },
                "model": _str("Exact model id to use instead of auto-picking"),
                "image_path": _str("Workspace image to show the helper (vision)"),
                "max_tokens": _int("Helper answer budget (64-2048, default 600)", 64, 2048),
            },
            ["task"],
        ),
        risk="read",
        phase="tool",
        label="Asking a helper model",
        summary=lambda args: f"Delegate ({args.get('capability', 'general')}): {str(args.get('task', ''))[:80]}",
    ),
    ToolSpec(
        name="describe_image",
        description=(
            "Describe a workspace image (or answer a question about it) using the smallest "
            "installed vision model. Works even when the current model cannot see images."
        ),
        parameters=_schema(
            {
                "path": _str("Workspace-relative image path"),
                "question": _str("What to ask about the image (default: describe it)"),
            },
            ["path"],
        ),
        risk="read",
        phase="tool",
        label="Looking at an image",
        summary=lambda args: f"Describe {args.get('path', '')}",
        category="desktop",
    ),
    ToolSpec(
        name="study_file",
        description=(
            "Answer a question about a long workspace file by splitting it into chunks, "
            "extracting relevant notes per chunk with a helper model, and merging them. "
            "Use for files too big to read whole; cite part numbers from the notes."
        ),
        parameters=_schema(
            {
                "path": _str("Workspace-relative text file path"),
                "question": _str("What you need to know from the file"),
            },
            ["path", "question"],
        ),
        risk="read",
        phase="reading",
        label="Studying a long file",
        summary=lambda args: f"Study {args.get('path', '')}: {str(args.get('question', ''))[:60]}",
        category="coding",
    ),
    # --- sandboxed code execution -------------------------------------------------------------
    ToolSpec(
        name="run_python",
        description=(
            "Run a short Python snippet in an isolated scratch folder inside the workspace "
            "(no shell, 30s limit, capped output). For calculations, data transforms, and "
            "analysing files/media with code. Created files are listed in the result."
        ),
        parameters=_schema(
            {
                "code": _str("Python code to run"),
                "timeout": _int("Seconds before timeout (1-30)", 1, 30),
            },
            ["code"],
        ),
        risk="exec",
        phase="running",
        label="Running Python",
        summary=lambda args: "Run Python: " + str(args.get("code", "")).splitlines()[0][:100] if str(args.get("code", "")).strip() else "Run Python",
        category="coding",
    ),
    ToolSpec(
        name="run_javascript",
        description=(
            "Run a short JavaScript snippet with Node.js in an isolated scratch folder "
            "inside the workspace (no shell, 30s limit, capped output). For calculations, "
            "JSON/data work and quick prototypes. Created files are listed in the result."
        ),
        parameters=_schema(
            {
                "code": _str("JavaScript code to run (ES module)"),
                "timeout": _int("Seconds before timeout (1-30)", 1, 30),
            },
            ["code"],
        ),
        risk="exec",
        phase="running",
        label="Running JavaScript",
        summary=lambda args: "Run JS: " + str(args.get("code", "")).splitlines()[0][:100] if str(args.get("code", "")).strip() else "Run JS",
        category="coding",
    ),
)

TOOL_SPECS: dict[str, ToolSpec] = {spec.name: spec for spec in _SPECS}

# Extension tools (kiwibro/extensions.py): optional plug-in modules register
# through the same catalog so permissions, profiles and approvals apply alike.
_EXTENSIONS: dict[str, Any] = {}
try:
    from .extensions import all_tools as _extension_tools

    for _ext in _extension_tools():
        if _ext.name in TOOL_SPECS:
            log.warning("extension tool %s shadows a built-in; ignored", _ext.name)
            continue
        _EXTENSIONS[_ext.name] = _ext
        TOOL_SPECS[_ext.name] = ToolSpec(
            name=_ext.name,
            description=_ext.description,
            parameters=_ext.parameters,
            risk=_ext.risk,
            phase=_ext.phase,
            label=_ext.label,
            summary=_ext.summary,
            category=getattr(_ext, "category", "coding"),
            parallel=_ext.risk in ("read", "web"),
        )
except Exception:  # extensions are optional; never break the core catalog
    log.warning("extension tools unavailable", exc_info=True)

PHASES = {name: (spec.phase, spec.label) for name, spec in TOOL_SPECS.items()}

_TOOL_DEFS_CACHE: list[dict[str, Any]] | None = None


def tool_defs() -> list[dict[str, Any]]:
    """OpenAI-style definitions for every tool (cached; copies returned)."""
    global _TOOL_DEFS_CACHE
    if _TOOL_DEFS_CACHE is None:
        _TOOL_DEFS_CACHE = [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in TOOL_SPECS.values()
        ]
    return [dict(item, function=dict(item["function"])) for item in _TOOL_DEFS_CACHE]


def tool_names() -> list[str]:
    return list(TOOL_SPECS)


def get_spec(name: str) -> ToolSpec:
    spec = TOOL_SPECS.get(name)
    if spec is None:
        raise ValueError(f"unknown tool: {name}")
    return spec


def summary_for(name: str, args: dict[str, Any]) -> str:
    spec = TOOL_SPECS.get(name)
    if spec is None:
        return f"Use {name}"
    try:
        return spec.summary(args)
    except Exception:  # pragma: no cover - summaries must never fail
        return f"Use {name}"


def phase_for(name: str) -> tuple[str, str]:
    return PHASES.get(name, ("tool", "Working"))


def permission_mode(settings: dict[str, Any], name: str) -> str:
    return safety.permission(settings, name)


def requires_approval(settings: dict[str, Any], name: str) -> bool:
    return permission_mode(settings, name) == "ask"


def is_parallel_safe(name: str) -> bool:
    """True for read-only/web tools that may run concurrently in one round."""
    spec = TOOL_SPECS.get(name)
    return bool(spec and spec.parallel)


# --- tool profiles ---------------------------------------------------------------

_CORE = tuple(name for name, spec in TOOL_SPECS.items() if spec.category == "core")
_BY_CATEGORY = {
    category: tuple(name for name, spec in TOOL_SPECS.items() if spec.category == category)
    for category in ("research", "coding", "desktop")
}
_MINIMAL = (
    "web_search", "fetch_page", "read_file", "list_files",
    "write_file", "replace_in_file", "run_command", "remember",
)
# The tool list renders into the prompt ahead of every message, so a list that
# changes between turns forces LM Studio to re-prefill the whole conversation
# (measured on this CPU: 0% reuse and 152 s, against 1-4 s with a stable list).
# "auto" is therefore ONE fixed set, identical for every turn and conversation,
# sized for 1-4B models; the rest of the catalog still runs when called by its
# exact name (calls resolve against TOOL_SPECS) and is named in message 0.
AUTO_TOOLS = (
    "web_search", "web_research", "fetch_page",
    "read_file", "list_files", "grep_files", "write_file", "replace_in_file",
    "run_command", "run_python", "study_file", "undo_changes",
    "remember", "recall", "task_board", "doc_search", "doc_read",
    "describe_image", "delegate", "calculate", "workspace",
)


def select_tool_names(profile: str | None) -> list[str]:
    """Tool names a profile offers, in catalog order (never depends on the message)."""
    profile = profile if profile in config.TOOL_PROFILES else config.DEFAULT_TOOL_PROFILE
    if profile == "full":
        return tool_names()
    if profile == "minimal":
        wanted = set(_MINIMAL)
    elif profile in _BY_CATEGORY:
        wanted = set(_CORE) | set(_BY_CATEGORY[profile])
    else:
        wanted = set(AUTO_TOOLS)
    return [name for name in TOOL_SPECS if name in wanted]


def more_tools_line(settings: dict[str, Any]) -> str:
    """One stable prompt line naming catalog tools the profile does not list ('' if none)."""
    offered = set(select_tool_names(settings.get("tool_profile")))
    others = [
        name for name in TOOL_SPECS
        if name not in offered and permission_mode(settings, name) != "deny"
    ]
    if not others:
        return ""
    return (
        "More tools, not listed above, run when called by exact name: "
        + ", ".join(others) + ".\n"
    )


def tool_defs_for(
    profile: str | None = None,
    settings: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Tool definitions for a profile, minus user-denied tools (stable for a given settings)."""
    if settings is not None and profile is None:
        profile = settings.get("tool_profile")
    names = set(select_tool_names(profile))
    if settings is not None:
        names = {name for name in names if permission_mode(settings, name) != "deny"}
    return [item for item in tool_defs() if item["function"]["name"] in names]


# --- argument validation -----------------------------------------------------------


def _as_str(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field_name} is required")
    return value


def _as_int(value: Any, field_name: str, default: int, low: int, high: int) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if isinstance(value, bool) or not low <= number <= high:
        raise ValueError(f"{field_name} must be between {low} and {high}")
    return number


def _as_bool(value: Any, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be true or false")
    return value


def _as_str_list(value: Any, field_name: str, max_items: int) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list")
    items = [_as_str(item, field_name) for item in value]
    if len(items) > max_items:
        raise ValueError(f"{field_name} is limited to {max_items} entries")
    return items


def _as_url(value: Any, field_name: str = "url") -> str:
    url = _as_str(value, field_name)
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"{field_name} must start with http:// or https://")
    return url


def _opt_str(args: dict[str, Any], field_name: str) -> str | None:
    value = args.get(field_name)
    if value is None:
        return None
    return _as_str(value, field_name, allow_empty=True).strip() or None


def validate_args(name: str, raw_args: dict[str, Any]) -> dict[str, Any]:
    get_spec(name)
    if not isinstance(raw_args, dict):
        raise ValueError("tool arguments must be an object")
    if name in _EXTENSIONS:
        validated = _EXTENSIONS[name].validate(raw_args)
        if not isinstance(validated, dict):
            raise ValueError("tool arguments must be an object")
        return validated
    args = raw_args
    if name == "web_search":
        return {
            "query": _as_str(args.get("query"), "query")[:400],
            "max_results": _as_int(args.get("max_results"), "max_results", 5, 1, 8),
        }
    if name == "fetch_page":
        return {"url": _as_url(args.get("url")), "max_chars": _as_int(args.get("max_chars"), "max_chars", 4000, 200, 8000)}
    if name == "multi_search":
        return {
            "queries": [query[:400] for query in _as_str_list(args.get("queries"), "queries", 4)],
            "max_results": _as_int(args.get("max_results"), "max_results", 5, 1, 8),
        }
    if name == "fetch_pages":
        return {
            "urls": [_as_url(url, "urls") for url in _as_str_list(args.get("urls"), "urls", 5)],
            "max_chars": _as_int(args.get("max_chars"), "max_chars", 2500, 200, 8000),
        }
    if name == "web_research":
        return {
            "query": _as_str(args.get("query"), "query")[:400],
            "depth": _as_int(args.get("depth"), "depth", 3, 1, 5),
        }
    if name == "browse_page":
        return {"url": _as_url(args.get("url")), "max_chars": _as_int(args.get("max_chars"), "max_chars", 4000, 200, 8000)}
    if name == "read_file":
        start, end = args.get("start"), args.get("end")
        return {
            "path": _as_str(args.get("path"), "path"),
            "start": _as_int(start, "start", 1, 1, 10_000_000) if start is not None else None,
            "end": _as_int(end, "end", 10_000_000, 1, 10_000_000) if end is not None else None,
        }
    if name == "read_files":
        start, end = args.get("start"), args.get("end")
        return {
            "paths": _as_str_list(args.get("paths"), "paths", 8),
            "start": _as_int(start, "start", 1, 1, 10_000_000) if start is not None else None,
            "end": _as_int(end, "end", 10_000_000, 1, 10_000_000) if end is not None else None,
        }
    if name == "list_files":
        return {"path": _as_str(args.get("path", "."), "path", allow_empty=True) or "."}
    if name == "glob_files":
        return {"pattern": _as_str(args.get("pattern"), "pattern")[:200], "path": _opt_str(args, "path") or "."}
    if name == "grep_files":
        return {
            "pattern": _as_str(args.get("pattern"), "pattern")[:1000],
            "path": _opt_str(args, "path") or ".",
            "glob": _opt_str(args, "glob"),
            "regex": _as_bool(args.get("regex"), "regex", True),
            "case_sensitive": _as_bool(args.get("case_sensitive"), "case_sensitive", False),
            "context_lines": _as_int(args.get("context_lines"), "context_lines", 0, 0, 3),
            "max_results": _as_int(args.get("max_results"), "max_results", 50, 1, 100),
        }
    if name == "file_outline":
        return {"path": _as_str(args.get("path"), "path")}
    if name == "write_file":
        return {
            "path": _as_str(args.get("path"), "path"),
            "content": _as_str(args.get("content"), "content", allow_empty=True),
        }
    if name == "replace_in_file":
        new = args.get("new")
        if not isinstance(new, str):
            raise ValueError("new must be a string")
        return {"path": _as_str(args.get("path"), "path"), "old": _as_str(args.get("old"), "old"), "new": new}
    if name == "apply_patch":
        raw_edits = args.get("edits")
        if not isinstance(raw_edits, list) or not raw_edits:
            raise ValueError("edits must be a non-empty list")
        if len(raw_edits) > 20:
            raise ValueError("edits is limited to 20 entries")
        edits = []
        for edit in raw_edits:
            if not isinstance(edit, dict):
                raise ValueError("each edit must be an object with old and new")
            new = edit.get("new")
            if not isinstance(new, str):
                raise ValueError("each edit's new must be a string")
            edits.append({"old": _as_str(edit.get("old"), "old"), "new": new})
        return {"path": _as_str(args.get("path"), "path"), "edits": edits}
    if name == "file_manage":
        action = args.get("action")
        if action not in codebase.FILE_MANAGE_ACTIONS:
            raise ValueError("action must be mkdir, move, copy or delete")
        return {
            "action": action,
            "path": _as_str(args.get("path"), "path"),
            "destination": _opt_str(args, "destination"),
            "recursive": _as_bool(args.get("recursive"), "recursive", False),
        }
    if name == "git":
        action = args.get("action") or "status"
        if action not in codebase.GIT_ACTIONS:
            raise ValueError("action must be status, diff, log, show or branch")
        return {
            "action": action,
            "path": _opt_str(args, "path") or ".",
            "ref": _opt_str(args, "ref"),
            "limit": _as_int(args.get("limit"), "limit", 10, 1, 50),
        }
    if name == "run_command":
        cwd = args.get("cwd")
        return {
            "command": _as_str(args.get("command"), "command")[:4000],
            "cwd": _as_str(cwd, "cwd", allow_empty=True) or None if cwd is not None else None,
        }
    if name == "multi_command":
        return {
            "commands": [command[:4000] for command in _as_str_list(args.get("commands"), "commands", 8)],
            "cwd": _opt_str(args, "cwd"),
            "stop_on_error": _as_bool(args.get("stop_on_error"), "stop_on_error", True),
        }
    if name == "terminal_session":
        action = args.get("action")
        if action not in ("start", "run", "status", "input", "kill", "list"):
            raise ValueError("action must be start, run, status, input, kill or list")
        validated: dict[str, Any] = {
            "action": action,
            "command": None,
            "session_id": _opt_str(args, "session_id"),
            "input": None,
            "cwd": _opt_str(args, "cwd"),
            "wait_ms": _as_int(args.get("wait_ms"), "wait_ms", 800, 0, 5000),
        }
        if action in ("start", "run"):
            validated["command"] = _as_str(args.get("command"), "command")[:4000]
        if action in ("status", "input", "kill") and not validated["session_id"]:
            raise ValueError("session_id is required")
        if action == "input":
            validated["input"] = _as_str(args.get("input"), "input", allow_empty=True)[:4000]
        return validated
    if name == "system_info":
        return {}
    if name == "open_path":
        return {"target": _as_str(args.get("target"), "target")[:2000]}
    if name == "clipboard":
        action = args.get("action")
        if action not in ("read", "write"):
            raise ValueError("action must be read or write")
        text = args.get("text")
        if action == "write" and not isinstance(text, str):
            raise ValueError("text is required to write the clipboard")
        return {"action": action, "text": text if action == "write" else None}
    if name == "notify":
        body = args.get("body")
        if body is not None and not isinstance(body, str):
            raise ValueError("body must be a string")
        return {"title": _as_str(args.get("title"), "title"), "body": body or ""}
    if name == "screenshot":
        return {"filename": _opt_str(args, "filename")}
    if name == "task_board":
        action = args.get("action") or "list"
        if action not in ("list", "add", "update", "delete", "clear", "summary"):
            raise ValueError("action must be list, add, update, delete, clear or summary")
        status = args.get("status")
        if status is not None and status not in ("todo", "in_progress", "blocked", "done"):
            raise ValueError("status must be todo, in_progress, blocked or done")
        priority = args.get("priority")
        if priority is not None and priority not in ("low", "medium", "high"):
            raise ValueError("priority must be low, medium or high")
        validated = {
            "action": action,
            "title": _opt_str(args, "title"),
            "task_id": _opt_str(args, "task_id"),
            "status": status,
            "notes": _opt_str(args, "notes"),
            "priority": priority,
            "depends_on": (
                _as_str_list(args.get("depends_on"), "depends_on", 10)
                if args.get("depends_on") is not None else None
            ),
            "only_done": _as_bool(args.get("only_done"), "only_done", False),
        }
        if action == "add" and not validated["title"]:
            raise ValueError("title is required to add a task")
        if action in ("update", "delete") and not validated["task_id"]:
            raise ValueError("task_id is required")
        return validated
    if name == "browser":
        action = args.get("action")
        if action not in ("open", "list", "switch", "close", "back", "inspect"):
            raise ValueError("action must be open, list, switch, close, back or inspect")
        validated = {
            "action": action,
            "url": None,
            "tab_id": _opt_str(args, "tab_id"),
            "selector": _opt_str(args, "selector"),
            "max_chars": _as_int(args.get("max_chars"), "max_chars", 6000, 200, 8000),
        }
        if action in ("open", "inspect"):
            validated["url"] = _as_url(args.get("url"))
        if action in ("switch", "close") and not validated["tab_id"]:
            raise ValueError("tab_id is required")
        return validated
    if name == "model_info":
        return {}
    if name == "delegate":
        capability = args.get("capability") or "general"
        if capability not in delegate.NEEDS:
            raise ValueError(f"capability must be one of {', '.join(delegate.NEEDS)}")
        return {
            "task": _as_str(args.get("task"), "task")[:8000],
            "capability": capability,
            "model": _opt_str(args, "model"),
            "image_path": _opt_str(args, "image_path"),
            "max_tokens": _as_int(args.get("max_tokens"), "max_tokens", 600, 64, 2048),
        }
    if name == "describe_image":
        question = args.get("question")
        if question is not None and not isinstance(question, str):
            raise ValueError("question must be a string")
        return {"path": _as_str(args.get("path"), "path"), "question": (question or "")[:1000]}
    if name == "study_file":
        return {
            "path": _as_str(args.get("path"), "path"),
            "question": _as_str(args.get("question"), "question")[:1000],
        }
    if name in ("run_python", "run_javascript"):
        return {
            "code": _as_str(args.get("code"), "code"),
            "timeout": _as_int(args.get("timeout"), "timeout", 30, 1, 30),
        }
    if name == "recall":
        return {
            "query": _as_str(args.get("query"), "query")[:400],
            "limit": _as_int(args.get("limit"), "limit", 6, 1, 12),
        }
    if name == "remember":
        kind = str(args.get("kind") or "").strip().lower() or None
        if kind is not None and kind not in config.MEMORY_KINDS:
            kind = "fact"  # small models invent kinds ("habit"); the fact itself still matters
        action = str(args.get("action") or "save").strip().lower()
        action = {"add": "save", "store": "save", "edit": "update", "change": "update", "replace": "update",
                  "delete": "forget", "remove": "forget"}.get(action, action)
        if action not in ("save", "update", "forget"):
            action = "save"
        old = args.get("old")
        return {
            "content": _as_str(args.get("content"), "content"),
            "kind": kind,
            "action": action,
            "old": str(old).strip()[:300] if isinstance(old, str) and old.strip() else None,
        }
    raise ValueError(f"unknown tool: {name}")


# --- execution ------------------------------------------------------------------------


def _preview(text: str, limit: int = 400) -> str:
    collapsed = " ".join((text or "").split())
    return collapsed[:limit]


def shape_content(text: str, cap: int = 8000) -> str:
    """Head+tail truncation that keeps both ends of long tool output visible."""
    text = text or ""
    if len(text) <= cap:
        return text
    head = (cap * 3) // 5
    tail = cap - head
    return text[:head] + f"\n… [{len(text) - cap} chars omitted] …\n" + text[-tail:]


def execute_tool(name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Run a validated tool call. Never raises for expected tool failures."""
    try:
        return _execute(name, args, ctx)
    except (SandboxError, SearchError, DesktopError, DelegateError, ValueError) as exc:
        return ToolResult(ok=False, content=f"Error: {exc}", preview=_preview(f"Error: {exc}"))
    except Exception:  # pragma: no cover - unexpected tool bugs must not kill the turn
        log.exception("tool %s failed", name)
        return ToolResult(ok=False, content="Error: the tool failed unexpectedly.", preview="Tool failed")


def _search_lines(results: list[dict[str, str]]) -> str:
    return research.format_search_results(results)


def _execute(name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    if name in _EXTENSIONS:
        outcome = _EXTENSIONS[name].execute(args, ctx)
        if not isinstance(outcome, ToolResult):
            raise ValueError(f"extension tool {name} returned an invalid result")
        return outcome
    if name == "web_search":
        results = ctx.search.web_search(args["query"], args["max_results"])
        if not results:
            return ToolResult(ok=True, content=research.NO_RESULTS, preview="No results", meta={"count": 0})
        lines = []
        for index, result in enumerate(results, start=1):
            lines.append(f"{index}. {result['title']}\n{result['url']}")
            if result.get("snippet"):
                lines.append(result["snippet"])
            lines.append("")
        content = "\n".join(lines).strip()
        return ToolResult(ok=True, content=content, preview=_preview(content), meta={"count": len(results), "results": results})
    if name == "multi_search":
        bundle = research.multi_search(ctx.search, args["queries"], max_results=args["max_results"])
        content = _search_lines(bundle["results"])
        return ToolResult(ok=True, content=content, preview=_preview(content),
                          meta={"count": bundle["count"], "results": bundle["results"]})
    if name == "fetch_page":
        page = ctx.search.fetch_page(args["url"], args["max_chars"])
        content = f"{page['title']}\n{page['url']}\n\n{page['text']}"
        return ToolResult(ok=True, content=content, preview=_preview(page["text"]),
                          meta={"title": page["title"], "url": page["url"], "truncated": page["truncated"]})
    if name == "fetch_pages":
        bundle = research.fetch_pages(ctx.search, args["urls"], max_chars=args["max_chars"])
        content = research.format_evidence(bundle["pages"])
        return ToolResult(ok=bundle["ok_count"] > 0, content=shape_content(content, 12000),
                          preview=_preview(content), meta={"count": bundle["count"], "ok_count": bundle["ok_count"]})
    if name == "web_research":
        bundle = research.web_research(ctx.search, args["query"], depth=args["depth"])
        return ToolResult(ok=True, content=shape_content(bundle["evidence"], 12000),
                          preview=_preview(bundle["evidence"]),
                          meta={"count": len(bundle["results"]), "results": bundle["results"]})
    if name == "browse_page":
        page = ctx.search.browse_page(args["url"], args["max_chars"])
        content = f"{page['title']}\n{page['url']}\n\n{page['text']}"
        if page["links"]:
            link_lines = "\n".join(f"[{link['n']}] {link['text']} — {link['url']}" for link in page["links"])
            content += f"\n\nLinks on this page:\n{link_lines}"
        return ToolResult(ok=True, content=shape_content(content, 10000), preview=_preview(page["text"]),
                          meta={"title": page["title"], "url": page["url"], "links": len(page["links"])})
    if name == "read_file":
        result = ctx.workspace.read_file(args["path"], args.get("start"), args.get("end"))
        return ToolResult(ok=True, content=result["content"], preview=_preview(result["content"]),
                          meta={"path": result["path"], "lines": result["total_lines"]})
    if name == "read_files":
        paths: list[str] = args["paths"]
        result = ctx.workspace.read_file(paths[0], args.get("start"), args.get("end"), paths=paths)
        return ToolResult(ok=True, content=result["content"], preview=_preview(result["content"]),
                          meta={"paths": result.get("paths", paths), "lines": result["total_lines"]})
    if name == "list_files":
        entries = ctx.workspace.list_files(args.get("path") or ".")
        if not entries:
            # Spelled out: tiny models read a bare "(empty)" as "try again", not "0 files".
            return ToolResult(ok=True, content="The folder is empty (0 files, 0 folders).", preview="Empty folder", meta={"count": 0})
        content = "\n".join(f"{entry['type']}\t{entry['path']}\t{entry['size']}" for entry in entries)
        return ToolResult(ok=True, content=content, preview=_preview(content), meta={"count": len(entries)})
    if name == "glob_files":
        found = codebase.glob_files(ctx.workspace, args["pattern"], path=args["path"])
        if not found["results"]:
            return ToolResult(ok=True, content=f"No files match {found['pattern']}.", preview="No matches", meta={"count": 0})
        content = "\n".join(f"{item['type']}\t{item['path']}\t{item['size']}" for item in found["results"])
        return ToolResult(ok=True, content=content, preview=_preview(content), meta={"count": found["count"]})
    if name == "grep_files":
        found = ctx.workspace.grep_files(
            args["pattern"], args["path"], glob=args["glob"], regex=args["regex"],
            case_sensitive=args["case_sensitive"], context_lines=args["context_lines"],
            max_results=args["max_results"],
        )
        if not found["matches"]:
            content = f"No matches for “{args['pattern']}” ({found['files_searched']} files searched)."
            return ToolResult(ok=True, content=content, preview="No matches", meta={"count": 0})
        lines = []
        for match in found["matches"]:
            if match.get("context"):
                lines.append(f"{match['path']}:{match['line']}")
                lines.extend(match["context"])
            else:
                lines.append(f"{match['path']}:{match['line']}: {match['text']}")
        if found["truncated"]:
            lines.append("… [more matches exist; narrow the pattern or path]")
        content = shape_content("\n".join(lines), 10000)
        return ToolResult(ok=True, content=content, preview=_preview(content), meta={"count": found["count"]})
    if name == "file_outline":
        outline = codebase.file_outline(ctx.workspace, args["path"])
        body = "\n".join(outline["entries"]) if outline["entries"] else "(no symbols or headings found)"
        content = f"{outline['path']} ({outline['total_lines']} lines)\n{body}"
        return ToolResult(ok=True, content=content, preview=_preview(content), meta={"path": outline["path"]})
    if name == "write_file":
        written = ctx.workspace.write_file(args["path"], args["content"])
        message = f"Wrote {written['bytes']} bytes to {written['path']}."
        return ToolResult(ok=True, content=message, preview=message, meta=written)
    if name == "replace_in_file":
        replaced = ctx.workspace.replace_in_file(args["path"], args["old"], args["new"])
        message = f"Replaced 1 occurrence in {replaced['path']}."
        return ToolResult(ok=True, content=message, preview=message, meta=replaced)
    if name == "apply_patch":
        patched = codebase.apply_patch(ctx.workspace, args["path"], args["edits"])
        content = f"Applied {patched['edits']} edits to {patched['path']} ({patched['bytes']} bytes)."
        if patched["diff"]:
            content += f"\n\n{patched['diff']}"
        return ToolResult(ok=True, content=content, preview=_preview(content), meta=patched)
    if name == "file_manage":
        managed = codebase.file_manage(
            ctx.workspace, args["action"], args["path"],
            destination=args["destination"], recursive=args["recursive"],
        )
        if managed["action"] in ("move", "copy"):
            message = f"{managed['action'].capitalize()}d {managed['path']} → {managed['destination']}."
        elif managed["action"] == "mkdir":
            message = f"Created folder {managed['path']}."
        else:
            message = f"Deleted {managed['path']}."
        return ToolResult(ok=True, content=message, preview=message, meta=managed)
    if name == "git":
        info = codebase.git_info(ctx.workspace, args["action"], path=args["path"], ref=args["ref"], limit=args["limit"])
        content = f"git {info['action']} in {info['cwd']}:\n{info['output']}"
        return ToolResult(ok=True, content=content, preview=_preview(info["output"]), meta=info)
    if name == "run_command":
        result = ctx.workspace.run_command(args["command"], args.get("cwd"))
        content = f"$ {args['command']}\nexit code: {result['exit_code']}\ncwd: {result['cwd']}\n\n{result['output']}"
        return ToolResult(ok=not result["timed_out"] and result["exit_code"] == 0, content=content,
                          preview=_preview(result["output"]),
                          meta={"exit_code": result["exit_code"], "duration_ms": result["duration_ms"]})
    if name == "multi_command":
        pipeline = codebase.multi_command(
            ctx.workspace, args["commands"], cwd=args["cwd"], stop_on_error=args["stop_on_error"]
        )
        return ToolResult(ok=pipeline["ok"], content=pipeline["output"], preview=_preview(pipeline["output"]),
                          meta={"ran": pipeline["ran"], "results": pipeline["results"]})
    if name == "terminal_session":
        return _execute_terminal(args, ctx)
    if name == "system_info":
        info = desktop.system_info(ctx.workspace)
        content = "\n".join(
            f"{key}: {value}" for key, value in info.items() if key != "tools"
        ) + "\navailable: " + ", ".join(name_ for name_, ok in info["tools"].items() if ok)
        return ToolResult(ok=True, content=content, preview=_preview(content), meta=info)
    if name == "open_path":
        opened = desktop.open_path(ctx.workspace, args["target"])
        message = f"Opened {opened['opened']}."
        return ToolResult(ok=True, content=message, preview=message, meta=opened)
    if name == "clipboard":
        clip = desktop.clipboard(args["action"], args.get("text"))
        if args["action"] == "read":
            content = clip["text"] if clip["text"] else "(clipboard is empty)"
            return ToolResult(ok=True, content=content, preview=_preview(content), meta=clip)
        message = f"Copied {clip['written_chars']} characters to the clipboard."
        return ToolResult(ok=True, content=message, preview=message, meta=clip)
    if name == "notify":
        desktop.notify(args["title"], args["body"])
        message = f"Notification shown: {args['title']}"
        return ToolResult(ok=True, content=message, preview=message, meta={"title": args["title"]})
    if name == "screenshot":
        shot = desktop.screenshot(ctx.workspace, args.get("filename"))
        message = f"Screenshot saved to {shot['path']} ({shot['bytes']} bytes, via {shot['backend']})."
        return ToolResult(ok=True, content=message, preview=message, meta=shot)
    if name == "task_board":
        from .tasks import TASK_BOARD

        board_args = {
            key: value for key, value in args.items()
            if key != "action" and value is not None
        }
        outcome = TASK_BOARD.execute(args["action"], board_args, conversation_id=ctx.conversation_id)
        tasks = outcome.get("tasks") or []
        marks = {"done": "[x]", "in_progress": "[~]", "blocked": "[!]", "todo": "[ ]"}
        lines = [
            f"{marks.get(task.get('status', 'todo'), '[ ]')} {task.get('title', '')} ({task.get('id', '')}, {task.get('priority', 'medium')})"
            for task in tasks
        ]
        content = str(outcome.get("summary") or "")
        if lines and args["action"] != "summary":
            content += "\n" + "\n".join(lines)
        return ToolResult(ok=bool(outcome.get("ok")), content=content.strip() or "(no tasks)",
                          preview=_preview(content), meta={"tasks": tasks})
    if name == "browser":
        return _execute_browser(args, ctx)
    if name == "model_info":
        llm = _require_llm(ctx)
        models = delegate.list_model_capabilities(llm)
        lines = []
        for entry in models:
            caps = ", ".join(entry["capabilities"]) if entry["capabilities"] else "general chat"
            size = f" ~{entry['size_hint_b']}B" if entry["size_hint_b"] else ""
            lines.append(f"- {entry['id']} ({entry['state']}, ctx {entry['context']}{size}): {caps}")
        content = "Installed LM Studio models:\n" + "\n".join(lines)
        if ctx.current_model:
            content += f"\nCurrent main model: {ctx.current_model}"
        return ToolResult(ok=True, content=content, preview=_preview(content), meta={"models": models})
    if name == "delegate":
        llm = _require_llm(ctx)
        if args.get("image_path"):
            data_url, _public = delegate.image_data_url(ctx.workspace, args["image_path"])
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": args["task"]},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }]
            need = args["capability"] if args["capability"] != "general" else "vision"
        else:
            messages = [{"role": "user", "content": args["task"]}]
            need = args["capability"]
        result = delegate.run_helper(
            llm, need, messages,
            current_model=ctx.current_model, model=args["model"], max_tokens=args["max_tokens"],
        )
        content = f"[helper model: {result['model']}]\n{result['text']}"
        return ToolResult(ok=True, content=content, preview=_preview(result["text"]), meta=result)
    if name == "describe_image":
        llm = _require_llm(ctx)
        result = delegate.describe_image(
            llm, ctx.workspace, args["path"], args["question"], current_model=ctx.current_model
        )
        content = f"[vision model: {result['model']}]\n{result['text']}"
        return ToolResult(ok=True, content=content, preview=_preview(result["text"]), meta=result)
    if name == "study_file":
        llm = _require_llm(ctx)
        result = delegate.study_file(
            llm, ctx.workspace, args["path"], args["question"], current_model=ctx.current_model
        )
        suffix = f"\n\n(Studied {result['source']} in {result['chunks']} part(s) with {result['model']}"
        if result.get("focused"):
            suffix += f"; used the {result['focused']} parts most relevant to the question.)"
        else:
            suffix += "; file was truncated to fit.)" if result.get("truncated") else ".)"
        return ToolResult(ok=True, content=result["text"] + suffix, preview=_preview(result["text"]), meta=result)
    if name in ("run_python", "run_javascript"):
        language = "python" if name == "run_python" else "javascript"
        snippet = codebase.run_snippet(ctx.workspace, language, args["code"], timeout=args["timeout"])
        content = (
            f"{language} snippet · exit code: {snippet['exit_code']} · {snippet['duration_ms']}ms\n"
            f"scratch: {snippet['scratch']}\n\n{snippet['output']}"
        )
        if snippet["created_files"]:
            content += "\n\nCreated files:\n" + "\n".join(f"- {path}" for path in snippet["created_files"])
        return ToolResult(ok=not snippet["timed_out"] and snippet["exit_code"] == 0,
                          content=content, preview=_preview(snippet["output"]), meta=snippet)
    if name == "recall":
        return _execute_recall(args, ctx)
    if name == "remember":
        return _execute_remember(args, ctx)
    raise ValueError(f"unknown tool: {name}")


def _require_llm(ctx: ToolContext) -> LMStudioClient:
    if ctx.llm is None:
        raise DelegateError("model delegation is unavailable in this context")
    return ctx.llm


def _browser_page_text(page: dict[str, Any]) -> str:
    body = page.get("markdown") or page.get("text") or ""
    text = f"{page.get('title', '')}\n{page.get('url', '')}\n\n{body}".strip()
    links = page.get("links") or []
    if links:
        rendered = []
        for link in links[:40]:
            if isinstance(link, dict):
                rendered.append(f"[{link.get('n', '?')}] {link.get('text', '')} — {link.get('url', '')}")
            else:  # pragma: no cover - defensive
                rendered.append(str(link))
        text += "\n\nLinks:\n" + "\n".join(rendered)
    return text


def _execute_browser(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from .browser import BROWSER

    action = args["action"]
    conv = ctx.conversation_id
    if action == "list":
        tabs = BROWSER.list_tabs(conv)
        if not tabs:
            return ToolResult(ok=True, content="No browser tabs open.", preview="No tabs", meta={"tabs": []})
        lines = [
            f"{'*' if tab.get('active') else '-'} {tab.get('tab_id', '')} · {tab.get('title', '')} · {tab.get('url', '')}"
            for tab in tabs
        ]
        return ToolResult(ok=True, content="\n".join(lines), preview=_preview(lines[0]), meta={"tabs": tabs})
    if action == "close":
        closed = BROWSER.close_tab(args["tab_id"], conv)
        message = f"Closed tab {args['tab_id']}." if closed else f"Tab {args['tab_id']} was not open."
        return ToolResult(ok=bool(closed), content=message, preview=message, meta={"closed": bool(closed)})
    if action == "inspect":
        page = BROWSER.inspect_page(
            args["url"], selector=args["selector"], max_chars=args["max_chars"]
        )
        return ToolResult(ok=True, content=shape_content(_browser_page_text(page), 10000),
                          preview=_preview(page.get("text") or ""), meta={"url": page.get("url")})
    if action == "open":
        outcome = BROWSER.open_tab(
            args["url"], conv, selector=args["selector"], max_chars=args["max_chars"]
        )
        page = outcome["page"]
        content = f"[tab {outcome['tab_id']}]\n" + _browser_page_text(page)
        return ToolResult(ok=True, content=shape_content(content, 10000),
                          preview=_preview(page.get("text") or ""), meta={"tab_id": outcome["tab_id"]})
    if action == "switch":
        outcome = BROWSER.switch_tab(args["tab_id"], conv)
        page = outcome.get("page") or {}
        content = f"[tab {outcome['tab_id']}]\n" + _browser_page_text(page)
        return ToolResult(ok=True, content=shape_content(content, 10000),
                          preview=_preview(page.get("text") or ""), meta={"tab_id": outcome["tab_id"]})
    # back
    outcome = BROWSER.back(conv, max_chars=args["max_chars"])
    page = outcome.get("page") or {}
    if not outcome.get("ok"):
        return ToolResult(ok=False, content=f"Error: {outcome.get('reason', 'cannot go back')}",
                          preview="No previous page", meta=outcome)
    content = f"[tab {outcome['tab_id']}]\n" + _browser_page_text(page)
    return ToolResult(ok=True, content=shape_content(content, 10000),
                      preview=_preview(page.get("text") or ""), meta={"tab_id": outcome["tab_id"]})


def _execute_terminal(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    action = args["action"]
    workspace = ctx.workspace
    if action == "list":
        sessions = TERMINALS.list_sessions(ctx.conversation_id, workspace)
        if not sessions:
            return ToolResult(ok=True, content="No terminal sessions.", preview="No sessions", meta={"count": 0})
        lines = [
            f"{item['id']} · {item['status']} · {item['cwd']} · {item['command'][:80]}"
            for item in sessions
        ]
        return ToolResult(ok=True, content="\n".join(lines), preview=_preview(lines[0]), meta={"count": len(sessions)})
    if action == "start":
        info = TERMINALS.start_background(
            workspace, args["command"], cwd=args["cwd"],
            conversation_id=ctx.conversation_id, wait_ms=args["wait_ms"],
        )
        content = (
            f"Started session {info['id']} ({info['status']}, cwd {info['cwd']}):\n"
            f"$ {info['command']}\n{info.get('output') or '(no output)'}"
        )
        return ToolResult(ok=True, content=shape_content(content), preview=_preview(content), meta=info)
    if action == "run":
        info = TERMINALS.run_stateful(
            workspace, args["command"], session_id=args["session_id"],
            cwd=args["cwd"], conversation_id=ctx.conversation_id,
        )
        content = (
            f"[{info['session_id']}] $ {info['command']}\nexit code: {info['exit_code']} · "
            f"cwd: {info['cwd']}\n\n{info['output']}"
        )
        return ToolResult(ok=not info["timed_out"] and info["exit_code"] == 0,
                          content=content, preview=_preview(info["output"]), meta=info)
    if action == "status":
        info = TERMINALS.status(args["session_id"], workspace)
    elif action == "input":
        info = TERMINALS.send_input(args["session_id"], args["input"] or "", wait_ms=args["wait_ms"], workspace=workspace)
    else:  # kill
        info = TERMINALS.kill(args["session_id"], workspace)
    content = (
        f"Session {info['id']} · {info['status']} · exit code: {info['exit_code']} · "
        f"cwd: {info['cwd']}\n$ {info['command']}\n{info.get('output') or '(no output)'}"
    )
    return ToolResult(ok=True, content=shape_content(content), preview=_preview(content), meta=info)


_MEMORY_VERBS = {"save": "Remember", "update": "Update memory", "forget": "Forget"}
_UNGROUNDED = (
    "Not saved: a memory must be something the user said about themselves, their work or their "
    "preferences in this conversation — not a fact about Kiwibro, the tools or the web."
)


def _user_words(ctx: ToolContext, limit: int = 24, *, latest: bool = False) -> str:
    """What the user said recently in this chat (or only their newest message): memories
    the model writes must be grounded in it."""
    try:
        messages = ctx.memory.db.list_messages(ctx.conversation_id)[-limit:]
    except Exception:  # an unreadable history must not block saving
        return ""
    said = [str(m.get("content") or "") for m in messages if m.get("role") == "user"]
    if latest:
        return said[-1] if said else ""
    return "\n".join(said)


def _execute_remember(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from .memory import grounded  # local: memory imports this package's db/llm layers

    action, content = args.get("action") or "save", args["content"]
    said = _user_words(ctx)
    if action == "forget":
        memory, outcome = ctx.memory.forget(content)
        if outcome == "missing":
            message = f"No saved memory matches “{content[:80]}”."
            return ToolResult(ok=False, content=message, preview="Nothing to forget")
        if outcome == "pinned":
            message = f"Kept (pinned by the user; they can unpin it in Memory): {memory['content']}"
            return ToolResult(ok=False, content=message, preview="Pinned memory kept")
        message = f"Forgot: {memory['content']}"
        return ToolResult(ok=True, content=message, preview=message, meta={"memory_id": memory["id"]})
    old = args.get("old")
    if action == "update":
        latest = _user_words(ctx, latest=True)
        if old and latest and grounded(old, latest) and not grounded(content, latest):
            # The newest message carries the new fact; small models swap the two (measured).
            content, old = old, content
    if ctx.memory.about_assistant(content) or (said and not grounded(content, said)):
        return ToolResult(ok=False, content=_UNGROUNDED, preview="Not saved")
    if action == "update":
        memory, replaced = ctx.memory.revise(
            old, content, args.get("kind"), source_conversation_id=ctx.conversation_id
        )
        message = f"{'Updated memory' if replaced else 'Remembered'}: {memory['content']}"
    else:
        memory = ctx.memory.remember(content, args.get("kind"), source_conversation_id=ctx.conversation_id)
        message = f"Remembered: {memory['content']}"
    return ToolResult(ok=True, content=message, preview=message, meta={"memory_id": memory["id"]})


def _execute_recall(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Recall via the memory engine's seam (memories + episodic history)."""
    query, limit = args["query"], args["limit"]
    recall = getattr(ctx.memory, "recall", None)
    items: list[Any] = []
    if callable(recall):
        try:
            items = list(recall(query, conversation_id=ctx.conversation_id, limit=limit) or [])
        except TypeError:  # older signature tolerance
            items = list(recall(query, limit=limit) or [])  # type: ignore[operator]
    if not items:
        items = list(ctx.memory.retrieve(query, limit=limit))
    if not items:
        return ToolResult(ok=True, content=f"Nothing recalled for “{query}”.", preview="Nothing recalled", meta={"count": 0})
    lines = []
    for item in items[:limit]:
        if isinstance(item, dict):
            source = str(item.get("source") or item.get("kind") or "memory")
            if item.get("chat"):
                source = f"from the chat “{item['chat']}”, {str(item.get('created_at') or '')[:10]}"
            text = str(item.get("content") or item.get("text") or "").strip()
        else:  # pragma: no cover - defensive
            source, text = "memory", str(item)
        if text:
            lines.append(f"- [{source}] {text[:600]}")
    content = "\n".join(lines) if lines else f"Nothing recalled for “{query}”."
    return ToolResult(ok=True, content=content, preview=_preview(content), meta={"count": len(lines)})


# Re-exported for callers that build tool JSON by hand in tests/tools.
TOOL_JSON_SEP = (",", ":")
_ = json  # keep import for signature helpers below


def tool_call_signature(name: str, args: dict[str, Any]) -> str:
    """Stable fingerprint of a tool call, used by the agent's loop guard."""
    try:
        encoded = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        encoded = repr(args)
    return f"{name}:{encoded}"
