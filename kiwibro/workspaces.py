"""Per-conversation working folders: "work on ~/Desktop/my-app" just works.

A conversation can be bound to its own folder (any existing directory inside
``$HOME`` except the home folder itself, credential and start-up locations and
Kiwibro's own data); unbound conversations use the Settings workspace. Every
file, command and code tool of that conversation is confined to its folder.

The ``workspace`` tool shows the current folder or switches it when the user
asks. The safety policy auto-approves a switch only to a folder the user's
newest message actually mentions, so a drifting or prompt-injected model
cannot wander into other folders on its own.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import safety
from .extensions import ExtensionTool, schema, tool_result

ACTIONS = ("show", "switch", "reset")
_LISTING = 30
# Places a conversation may never be bound to (whatever the user typed).
FORBIDDEN_PLACES = frozenset({"root", "system", "device", "home_root", "secret", "startup", "kiwibro", "dynamic", "outside"})


def folder_for(db: Any, conversation_id: str | None) -> str | None:
    """The folder bound to this conversation, or None for the Settings workspace."""
    if not conversation_id:
        return None
    conversation = db.get_conversation(conversation_id)
    return (conversation or {}).get("workspace") or None


def resolve_folder(raw: str) -> Path:
    """Validate a folder the user asked to work in (raises ValueError with the reason)."""
    text = os.path.expanduser(str(raw or "").strip())
    if not text or not os.path.isabs(text):
        raise ValueError("give the folder as an absolute path or ~/…, e.g. ~/Desktop/my-app")
    path = Path(text).resolve()
    kind = safety.place_of(str(path))
    if kind in FORBIDDEN_PLACES:
        raise ValueError(f"Kiwibro cannot work inside {path} ({safety.describe_place(kind)})")
    if not path.is_dir():
        raise ValueError(f"{path} is not an existing folder")
    return path


def _listing(path: Path) -> str:
    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as exc:
        return f"(cannot list: {exc.strerror or exc})"
    shown = [f"{p.name}/" if p.is_dir() else p.name for p in entries if not p.name.startswith(".")][:_LISTING]
    more = len(entries) - len(shown)
    return ", ".join(shown) + (f" … (+{more} more)" if more > 0 else "") if shown else "(empty)"


def validate(args: dict[str, Any]) -> dict[str, Any]:
    action = args.get("action") or ("switch" if args.get("path") else "show")
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {', '.join(ACTIONS)}")
    if action != "switch":
        return {"action": action}
    return {"action": action, "path": str(resolve_folder(str(args.get("path") or "")))}


def execute(args: dict[str, Any], ctx: Any) -> Any:
    db = ctx.memory.db
    if args["action"] == "switch":
        db.update_conversation(ctx.conversation_id, workspace=args["path"], touch=False)
        path = Path(args["path"])
        return tool_result(True, f"Now working in {path} (this conversation only).\nContents: {_listing(path)}",
                           f"Working in {path}")
    if args["action"] == "reset":
        db.update_conversation(ctx.conversation_id, workspace=None, touch=False)
        return tool_result(True, "Back to the default workspace from Settings for this conversation.", "Default workspace")
    root = ctx.workspace.root
    bound = folder_for(db, ctx.conversation_id)
    origin = "this conversation's folder" if bound else "the default workspace from Settings"
    return tool_result(True, f"Working in {root} ({origin}).\nContents: {_listing(root)}", f"Working in {root}")


def _summary(args: dict[str, Any]) -> str:
    if (args.get("action") or "show") == "switch":
        return f"Work in {args.get('path', '?')}"
    return "Reset to the default workspace" if args.get("action") == "reset" else "Show the working folder"


WORKSPACE = ExtensionTool(
    name="workspace",
    description=(
        "Show or change the folder you work in for this conversation. When the user asks you to work "
        "on a specific folder (e.g. ~/Desktop/my-app), call action=switch with that path first; all file, "
        "command and code tools then operate there. action=show tells you where you are; reset goes back "
        "to the default workspace."
    ),
    parameters=schema(
        {
            "action": {"type": "string", "enum": list(ACTIONS), "description": "show | switch | reset"},
            "path": {"type": "string", "description": "folder to switch to (absolute or ~/…)"},
        },
        ["action"],
    ),
    risk="write",
    phase="reading",
    label="Working folder",
    summary=_summary,
    validate=validate,
    execute=execute,
    permission="auto",
    category="core",
)

TOOLS = (WORKSPACE,)
