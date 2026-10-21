"""Tool catalog, argument validation, permission evaluation and execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from . import config
from .memory import MemoryEngine
from .sandbox import SandboxError, Workspace
from .search import SearchEngine, SearchError

log = logging.getLogger(__name__)

@dataclass
class ToolContext:
    workspace: Workspace
    memory: MemoryEngine
    search: SearchEngine
    conversation_id: str


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


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


TOOL_SPECS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            name="web_search",
            description="Search the web and return titles, URLs and snippets. Use for current facts.",
            parameters=_schema(
                {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "How many results (1-8, default 5)",
                        "minimum": 1,
                        "maximum": 8,
                    },
                },
                ["query"],
            ),
            risk="web",
            phase="web",
            label="Searching the web",
            summary=lambda args: f'Search the web for "{args.get("query", "")}"',
        ),
        ToolSpec(
            name="fetch_page",
            description="Fetch a public web page and return its readable text.",
            parameters=_schema(
                {
                    "url": {"type": "string", "description": "http(s) URL to read"},
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters of text (200-8000)",
                        "minimum": 200,
                        "maximum": 8000,
                    },
                },
                ["url"],
            ),
            risk="web",
            phase="reading",
            label="Reading a web page",
            summary=lambda args: f"Read {args.get('url', '')}",
        ),
        ToolSpec(
            name="read_file",
            description="Read a text file from the workspace, optionally a line range.",
            parameters=_schema(
                {
                    "path": {"type": "string", "description": "Workspace-relative file path"},
                    "start": {"type": "integer", "description": "First line (1-based)"},
                    "end": {"type": "integer", "description": "Last line (inclusive)"},
                },
                ["path"],
            ),
            risk="read",
            phase="reading",
            label="Reading a file",
            summary=lambda args: f"Read {args.get('path', '')}",
        ),
        ToolSpec(
            name="list_files",
            description="List files and folders in the workspace (depth 2, up to 200 entries).",
            parameters=_schema(
                {"path": {"type": "string", "description": "Directory (default workspace root)"}}
            ),
            risk="read",
            phase="reading",
            label="Listing files",
            summary=lambda args: f"List files in {args.get('path', '.')}",
        ),
        ToolSpec(
            name="write_file",
            description="Create or overwrite a text file inside the workspace.",
            parameters=_schema(
                {
                    "path": {"type": "string", "description": "Workspace-relative file path"},
                    "content": {"type": "string", "description": "Full file content"},
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
                    "path": {"type": "string", "description": "Workspace-relative file path"},
                    "old": {"type": "string", "description": "Exact text to replace"},
                    "new": {"type": "string", "description": "Replacement text"},
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
                    "command": {"type": "string", "description": "Command line to run"},
                    "cwd": {"type": "string", "description": "Working directory inside the workspace"},
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
            description="Store a durable fact about the user in long-term memory.",
            parameters=_schema(
                {
                    "content": {"type": "string", "description": "Self-contained fact (3-300 chars)"},
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
            summary=lambda args: f"Remember: {str(args.get('content', ''))[:80]}",
        ),
    )
}

PHASES = {
    "web_search": ("web", "Searching the web"),
    "fetch_page": ("reading", "Reading a page"),
    "read_file": ("reading", "Reading a file"),
    "list_files": ("reading", "Listing files"),
    "write_file": ("writing", "Writing a file"),
    "replace_in_file": ("writing", "Editing a file"),
    "run_command": ("running", "Running a command"),
    "remember": ("tool", "Saving a memory"),
}


def tool_defs() -> list[dict[str, Any]]:
    return [
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
    permissions = settings.get("permissions") or {}
    mode = permissions.get(name)
    if mode not in config.PERMISSION_MODES:
        mode = config.DEFAULT_PERMISSIONS.get(name, "ask")
    return mode


def requires_approval(settings: dict[str, Any], name: str) -> bool:
    return permission_mode(settings, name) == "ask"


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
    if not low <= number <= high:
        raise ValueError(f"{field_name} must be between {low} and {high}")
    return number


def validate_args(name: str, raw_args: dict[str, Any]) -> dict[str, Any]:
    get_spec(name)
    if not isinstance(raw_args, dict):
        raise ValueError("tool arguments must be an object")
    args = raw_args
    if name == "web_search":
        return {
            "query": _as_str(args.get("query"), "query")[:400],
            "max_results": _as_int(args.get("max_results"), "max_results", 5, 1, 8),
        }
    if name == "fetch_page":
        url = _as_str(args.get("url"), "url")
        if not url.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return {"url": url, "max_chars": _as_int(args.get("max_chars"), "max_chars", 4000, 200, 8000)}
    if name == "read_file":
        start = args.get("start")
        end = args.get("end")
        return {
            "path": _as_str(args.get("path"), "path"),
            "start": _as_int(start, "start", 1, 1, 10_000_000) if start is not None else None,
            "end": _as_int(end, "end", 10_000_000, 1, 10_000_000) if end is not None else None,
        }
    if name == "list_files":
        return {"path": _as_str(args.get("path", "."), "path", allow_empty=True) or "."}
    if name == "write_file":
        return {
            "path": _as_str(args.get("path"), "path"),
            "content": _as_str(args.get("content"), "content", allow_empty=True),
        }
    if name == "replace_in_file":
        old = _as_str(args.get("old"), "old")
        new = args.get("new")
        if not isinstance(new, str):
            raise ValueError("new must be a string")
        return {"path": _as_str(args.get("path"), "path"), "old": old, "new": new}
    if name == "run_command":
        cwd = args.get("cwd")
        return {
            "command": _as_str(args.get("command"), "command")[:4000],
            "cwd": _as_str(cwd, "cwd", allow_empty=True) or None if cwd is not None else None,
        }
    if name == "remember":
        content = _as_str(args.get("content"), "content")
        kind = args.get("kind") or "fact"
        if kind not in config.MEMORY_KINDS:
            raise ValueError("kind is not a valid memory kind")
        return {"content": content, "kind": kind}
    raise ValueError(f"unknown tool: {name}")


def _preview(text: str, limit: int = 400) -> str:
    collapsed = " ".join((text or "").split())
    return collapsed[:limit]


def execute_tool(name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Run a validated tool call. Never raises for expected tool failures."""
    try:
        return _execute(name, args, ctx)
    except (SandboxError, SearchError, ValueError) as exc:
        return ToolResult(ok=False, content=f"Error: {exc}", preview=_preview(f"Error: {exc}"))
    except Exception:  # pragma: no cover - unexpected tool bugs must not kill the turn
        log.exception("tool %s failed", name)
        return ToolResult(ok=False, content="Error: the tool failed unexpectedly.", preview="Tool failed")


def _execute(name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    if name == "web_search":
        results = ctx.search.web_search(args["query"], args["max_results"])
        if not results:
            return ToolResult(
                ok=True,
                content="No web results were found.",
                preview="No results",
                meta={"count": 0},
            )
        lines = []
        for index, result in enumerate(results, start=1):
            lines.append(f"{index}. {result['title']}\n{result['url']}")
            if result.get("snippet"):
                lines.append(result["snippet"])
            lines.append("")
        content = "\n".join(lines).strip()
        return ToolResult(
            ok=True,
            content=content,
            preview=_preview(content),
            meta={"count": len(results), "results": results},
        )
    if name == "fetch_page":
        page = ctx.search.fetch_page(args["url"], args["max_chars"])
        content = f"{page['title']}\n{page['url']}\n\n{page['text']}"
        return ToolResult(
            ok=True,
            content=content,
            preview=_preview(page["text"]),
            meta={"title": page["title"], "url": page["url"], "truncated": page["truncated"]},
        )
    if name == "read_file":
        result = ctx.workspace.read_file(args["path"], args.get("start"), args.get("end"))
        return ToolResult(
            ok=True,
            content=result["content"],
            preview=_preview(result["content"]),
            meta={"path": result["path"], "lines": result["total_lines"]},
        )
    if name == "list_files":
        entries = ctx.workspace.list_files(args.get("path") or ".")
        if not entries:
            return ToolResult(ok=True, content="(empty)", preview="(empty)", meta={"count": 0})
        lines = [f"{entry['type']}\t{entry['path']}\t{entry['size']}" for entry in entries]
        content = "\n".join(lines)
        return ToolResult(ok=True, content=content, preview=_preview(content), meta={"count": len(entries)})
    if name == "write_file":
        written = ctx.workspace.write_file(args["path"], args["content"])
        message = f"Wrote {written['bytes']} bytes to {written['path']}."
        return ToolResult(ok=True, content=message, preview=message, meta=written)
    if name == "replace_in_file":
        replaced = ctx.workspace.replace_in_file(args["path"], args["old"], args["new"])
        message = f"Replaced 1 occurrence in {replaced['path']}."
        return ToolResult(ok=True, content=message, preview=message, meta=replaced)
    if name == "run_command":
        result = ctx.workspace.run_command(args["command"], args.get("cwd"))
        content = (
            f"$ {args['command']}\nexit code: {result['exit_code']}\n"
            f"cwd: {result['cwd']}\n\n{result['output']}"
        )
        return ToolResult(
            ok=not result["timed_out"] and result["exit_code"] == 0,
            content=content,
            preview=_preview(result["output"]),
            meta={"exit_code": result["exit_code"], "duration_ms": result["duration_ms"]},
        )
    if name == "remember":
        memory = ctx.memory.remember(args["content"], args["kind"], source_conversation_id=ctx.conversation_id)
        message = f"Remembered: {memory['content']}"
        return ToolResult(ok=True, content=message, preview=message, meta={"memory_id": memory["id"]})
    raise ValueError(f"unknown tool: {name}")
