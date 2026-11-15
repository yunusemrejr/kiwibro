"""Per-conversation working folders: "work on ~/Desktop/my-app" just works.

A conversation can be bound to its own folder (any existing directory inside
``$HOME`` except the home folder itself, credential and start-up locations and
Kiwibro's own data). With the setting ``workspace_mode`` = "temp" (the default) a new
conversation gets a temporary folder of its own; "folder" keeps new conversations on
the Settings workspace, which is also where conversations with no folder work. Every
file, command and code tool of that conversation is confined to its folder.

The ``workspace`` tool shows the current folder or switches it when the user
asks. The safety policy auto-approves a switch only to a folder the user's
newest message actually mentions, so a drifting or prompt-injected model
cannot wander into other folders on its own.
"""

from __future__ import annotations

import getpass
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

from . import safety
from .extensions import ExtensionTool, schema, tool_result

ACTIONS = ("show", "switch", "reset")
_LISTING = 30
# Places a conversation may never be bound to (whatever the user typed).
FORBIDDEN_PLACES = frozenset({"root", "system", "device", "home_root", "secret", "startup", "kiwibro", "dynamic", "outside"})
_MAX_FOLDERS = 200


def temp_root() -> Path:
    """Where the temporary folders of chats live: <system temp>/kiwibro-<login name>."""
    login = re.sub(r"[^A-Za-z0-9._-]", "_", getpass.getuser())
    return Path(tempfile.gettempdir()).resolve() / f"kiwibro-{login}"


def in_temp_root(path: str | os.PathLike[str]) -> bool:
    return Path(path).is_relative_to(temp_root()) and Path(path) != temp_root()


def temp_folder_of(conversation_id: str) -> Path:
    """The folder a chat gets in temp mode: chat-<first 8 characters of its id after the prefix>."""
    return temp_root() / f"chat-{conversation_id.partition('_')[2][:8]}"


def make_temp_folder(path: Path) -> Path:
    """Create a folder under the temp root; the root itself is private to this user (0700)."""
    root = temp_root()
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        pass
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise OSError(f"{root} exists but is not a folder of yours, so Kiwibro will not work in it")
    if info.st_mode & 0o077:
        root.chmod(0o700)
    path.mkdir(parents=True, exist_ok=True)
    return path


def mode(settings: dict[str, Any]) -> str:
    """"temp" or "folder". Real settings always carry the key (the default is "temp"); settings
    without it, such as a hand-made dict, mean the old behaviour."""
    return "temp" if settings.get("workspace_mode") == "temp" else "folder"


def own_folder(settings: dict[str, Any], conversation_id: str) -> str | None:
    """The folder a chat starts in, or returns to: its temporary one, or None (the Settings folder)."""
    return str(make_temp_folder(temp_folder_of(conversation_id))) if mode(settings) == "temp" else None


def create_conversation(
    db: Any, settings: dict[str, Any], *, title: str, model: str | None, private: bool, workspace: str | None = None
) -> dict[str, Any]:
    """A new conversation with its folder: the one asked for, else its own temporary one (temp mode)."""
    chosen = str(resolve_folder(workspace)) if workspace is not None else None
    conversation = db.create_conversation(title=title, model=model, private=private)
    try:
        folder = chosen or own_folder(settings, conversation["id"])
        if folder:
            conversation = db.update_conversation(conversation["id"], workspace=folder, touch=False) or conversation
    except OSError:
        db.delete_conversation(conversation["id"])  # no half-made chat that would work in the wrong folder
        raise
    return conversation


def folder_for(db: Any, conversation_id: str | None) -> str | None:
    """The folder bound to this conversation, or None for the Settings workspace.

    A temporary folder the system cleared (it clears /tmp at restart) is made again, empty."""
    if not conversation_id:
        return None
    conversation = db.get_conversation(conversation_id)
    bound = (conversation or {}).get("workspace") or None
    if bound and in_temp_root(bound) and not os.path.isdir(bound):
        make_temp_folder(Path(bound))
    return bound


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


def _accepts(path: Path) -> bool:
    try:
        resolve_folder(str(path))
    except ValueError:
        return False
    return True


def _listable(path: Path) -> bool:
    return any(path.is_relative_to(root) for root in (Path.home().resolve(), temp_root()))


def browse(raw: str | None) -> dict[str, Any]:
    """The sub-folders of a folder, for the folder picker. Only the home folder and Kiwibro's
    temp root may be looked into; a link that leads out of them is not shown. ValueError: a
    request that cannot be answered; PermissionError: a place the picker may not list."""
    text = os.path.expanduser(str(raw).strip() if raw else "~")
    if not os.path.isabs(text):
        raise ValueError("give the folder as an absolute path or ~/…")
    path = Path(text).resolve()
    if not _listable(path):
        raise PermissionError("the folder picker only shows folders inside your home folder or Kiwibro's temporary folders")
    if not path.is_dir():
        raise ValueError(f"{path} is not an existing folder")
    try:
        with os.scandir(path) as scan:
            entries = sorted((e for e in scan if not e.name.startswith(".")), key=lambda e: e.name.casefold())
    except OSError as exc:
        raise ValueError(f"cannot list {path}: {exc.strerror or exc}") from None
    folders = []
    for entry in entries:
        try:
            real = Path(entry.path).resolve(strict=True)  # a link is shown by where it really leads
        except (OSError, RuntimeError):  # broken link or link loop
            continue
        if real.is_dir() and _listable(real):
            folders.append({"name": entry.name, "path": str(real), "allowed": _accepts(real)})
    parent = path.parent if _listable(path.parent) else None
    return {
        "path": str(path), "parent": str(parent) if parent else None, "allowed": _accepts(path),
        "folders": folders[:_MAX_FOLDERS], "truncated": len(folders) > _MAX_FOLDERS,
    }


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
        folder = own_folder(getattr(ctx, "settings", None) or {}, ctx.conversation_id)
        db.update_conversation(ctx.conversation_id, workspace=folder, touch=False)
        if folder:
            return tool_result(True, f"Back in this conversation's own temporary folder {folder}.", "Temporary folder")
        return tool_result(True, "Back to the default workspace from Settings for this conversation.", "Default workspace")
    root = ctx.workspace.root
    bound = folder_for(db, ctx.conversation_id)
    origin = ("this conversation's temporary folder" if bound and in_temp_root(bound)
              else "this conversation's folder" if bound else "the default workspace from Settings")
    return tool_result(True, f"Working in {root} ({origin}).\nContents: {_listing(root)}", f"Working in {root}")


def _summary(args: dict[str, Any]) -> str:
    if (args.get("action") or "show") == "switch":
        return f"Work in {args.get('path', '?')}"
    return "Back to the starting folder" if args.get("action") == "reset" else "Show the working folder"


WORKSPACE = ExtensionTool(
    name="workspace",
    description=(
        "Show or change the folder you work in for this conversation. When the user asks you to work "
        "on a specific folder (e.g. ~/Desktop/my-app), call action=switch with that path first; all file, "
        "command and code tools then operate there. action=show tells you where you are; reset goes back "
        "to where this conversation started (its own temporary folder, or the default workspace)."
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
