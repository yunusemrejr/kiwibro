"""Extension tools: the plug-in surface for tool modules that live outside tools.py.

Each extension module (``compute``, ``library``, ``checkpoints``) exposes a
``TOOLS`` tuple of :class:`ExtensionTool` records. ``tools.py`` registers them
at import time through :func:`register_all`, so the validate/execute chains in
``tools.py`` never have to know about individual extension tools.

Extension modules must not import ``kiwibro.tools`` at module level (tools.py
imports this module, so that would be circular). Build results with
:func:`tool_result`, which imports ``ToolResult`` lazily.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

# Modules scanned for a ``TOOLS`` tuple, in registration order.
MODULES = ("compute", "library", "convert", "checkpoints", "remote")


@dataclass(frozen=True)
class ExtensionTool:
    name: str
    description: str
    parameters: dict[str, Any]
    risk: str  # read | write | exec | web
    phase: str  # reading | writing | running | web | tool
    label: str
    summary: Callable[[dict[str, Any]], str]
    validate: Callable[[dict[str, Any]], dict[str, Any]]
    execute: Callable[[dict[str, Any], Any], Any]
    permission: str = "auto"
    category: str = "coding"  # tool-profile group: core | research | coding | desktop
    # Files a call will create or change (for checkpoints/undo); None = nothing to snapshot.
    touches: Callable[[dict[str, Any], Any], list[Path]] | None = None


def schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    """JSON-schema object in the same shape tools.py uses."""
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def preview_text(text: str, limit: int = 400) -> str:
    return " ".join((text or "").split())[:limit]


def tool_result(ok: bool, content: str, preview: str | None = None, meta: dict[str, Any] | None = None) -> Any:
    """Build a ``tools.ToolResult`` without importing tools.py at module load."""
    from .tools import ToolResult

    return ToolResult(
        ok=ok,
        content=content,
        preview=preview if preview is not None else preview_text(content),
        meta=meta or {},
    )


def data_dir_for(ctx: Any) -> Path:
    """Kiwibro's data directory (where kiwibro.db lives) for a ToolContext."""
    return Path(ctx.memory.db.path).expanduser().resolve().parent


def all_tools() -> list[ExtensionTool]:
    collected: list[ExtensionTool] = []
    seen: set[str] = set()
    for module_name in MODULES:
        qualified = f"{__package__}.{module_name}"
        if importlib.util.find_spec(qualified) is None:
            continue  # optional module not present in this checkout
        try:
            module = importlib.import_module(qualified)
        except Exception:  # one broken module must not take the other extensions down
            log.warning("extension module %s failed to import", module_name, exc_info=True)
            continue
        for tool in getattr(module, "TOOLS", ()):
            if tool.name in seen:
                log.warning("duplicate extension tool %s ignored", tool.name)
                continue
            seen.add(tool.name)
            collected.append(tool)
    return collected


def register_all(register: Callable[[ExtensionTool], None]) -> list[str]:
    """Hand every extension tool to ``register``; returns the registered names."""
    names: list[str] = []
    for tool in all_tools():
        register(tool)
        names.append(tool.name)
    return names


def default_permissions() -> dict[str, str]:
    return {tool.name: tool.permission for tool in all_tools()}
