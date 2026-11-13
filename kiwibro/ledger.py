"""The session ledger: an exact, code-written record of a long conversation's steps.

A prompt loses detail as a chat grows, so every message is kept verbatim and this module
keeps the index over them: one row per step saying what was asked, which file changed,
which command ran and which page was read. Rows are decided here from the tool name and
its arguments, never by a model, so the record stays exact. ``block`` renders the rows the
model is reminded with, and ``step_text`` reads one earlier step back in full.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .db import Database

log = logging.getLogger(__name__)

ASK_CHARS = 160
COMMAND_CHARS = 200
STEP_READ_CHARS = 6000
STEPS_PER_KEY = 4
# A model was trained on these two lines, so they never change.
HEADER = (
    "## Earlier in this session\n"
    "Every earlier step of this conversation is stored in full. To read one again: "
    'recall("words to find"), or recall("step N") with a step number.'
)

_ROLE_LABELS = {"user": "user", "assistant": "Kiwibro", "tool": "tool result"}
_EXIT_CODE_RE = re.compile(r"exit code:\s*(-?\d+)")
_PAREN_EXIT_RE = re.compile(r"\(exit\s+(-?\d+)\)")


def _flat(value: Any, limit: int | None = None) -> str:
    """Whitespace collapsed to one line, cut at ``limit`` characters when one is given."""
    text = " ".join(str(value if value is not None else "").split())
    return text[:limit] if limit else text


def role_label(role: Any) -> str:
    """Who wrote a message, as a recall line names them: user, Kiwibro or tool result."""
    return _ROLE_LABELS.get(str(role or ""), str(role or "message"))


def step_label(step: int, role: Any) -> str:
    """The recall label of one step, e.g. "this chat, step 14, tool result"."""
    return f"this chat, step {int(step)}, {role_label(role)}"


# --- writing ----------------------------------------------------------------


def record_user(
    db: Database, conversation_id: str, seq: int, text: Any
) -> dict[str, Any] | None:
    """One "ask" row for what the user wrote at ``seq``; nothing when they wrote nothing."""
    ask = _flat(text, ASK_CHARS)
    if not ask:
        return None
    return db.add_ledger_entry(conversation_id, int(seq), "ask", ask)


def record_tool(
    db: Database,
    conversation_id: str,
    seq: int,
    name: Any,
    arguments: Any,
    ok: Any,
    result: Any,
) -> list[dict[str, Any]]:
    """The ledger rows one finished tool call earned, from its name and arguments.

    Odd arguments (missing keys, wrong types) never raise: the call is recorded as far as
    it can be read, and an unreadable one leaves the ledger as it was."""
    try:
        rows = _rows_for(str(name or ""), arguments if isinstance(arguments, dict) else {}, bool(ok), result)
    except Exception:  # a ledger row is never worth failing a tool call over
        log.warning("ledger: could not record %s", name, exc_info=True)
        return []
    return [
        db.add_ledger_entry(conversation_id, int(seq), kind, key, detail)
        for kind, key, detail in rows
        if kind and key
    ]


def _row(kind: str, key: str, detail: str) -> list[tuple[str, str, str]]:
    return [(kind, key, detail)] if key else []


def _rows_for(name: str, args: dict[str, Any], ok: bool, result: Any) -> list[tuple[str, str, str]]:
    text = result if isinstance(result, str) else _flat(result)
    if name in ("write_file", "replace_in_file", "apply_patch", "convert_document", "undo_changes",
                "file_manage", "component"):
        # A failed write changed nothing, so it leaves no file row.
        return _file_rows(name, args) if ok else []
    if name in ("run_command", "terminal_session", "run_python", "run_javascript", "remote"):
        return _command_rows(name, args, text, ok)
    if name == "multi_command":
        return _multi_command_rows(args.get("commands"), text, ok)
    if name in ("fetch_page", "browse_page", "fetch_pages", "browser"):
        return _page_rows(name, args, ok)
    if name in ("web_search", "web_research", "multi_search"):
        return _search_rows(name, args, ok)
    if name == "note":
        if _flat(args.get("action")) not in ("save", "append"):
            return []
        title = _flat(args.get("title"), ASK_CHARS) or _flat(args.get("content"), ASK_CHARS)
        return _row("note", title, "" if ok else "failed")
    if name == "remember":
        return _row("note", _flat(args.get("content"), ASK_CHARS), "" if ok else "failed")
    if name == "task_board":
        if _flat(args.get("action")) not in ("add", "update"):
            return []
        title = _flat(args.get("title"), ASK_CHARS) or _flat(args.get("task_id"))
        return _row("task", title, "" if ok else "failed")
    return []


def _file_rows(name: str, args: dict[str, Any]) -> list[tuple[str, str, str]]:
    path = _flat(args.get("path")) or _flat(args.get("file")) or _flat(args.get("name"))
    if name == "write_file":
        return _row("file", path, "written")
    if name in ("replace_in_file", "apply_patch"):
        return _row("file", path, "patched")
    if name == "convert_document":
        output = _flat(args.get("output")) or _flat(args.get("to"))
        return _row("file", path, f"converted to {output}" if output else "converted")
    if name == "undo_changes":
        return _row("file", path or "last change", "undone")
    if name == "component":
        return _row("file", path, "used") if _flat(args.get("action")) == "use" else []
    action, destination = _flat(args.get("action")), _flat(args.get("destination"))
    if action == "mkdir":
        return _row("file", path, "folder made")
    if action == "move":
        return _row("file", path, f"moved to {destination}" if destination else "moved")
    if action == "copy":
        return _row("file", path, f"copied to {destination}" if destination else "copied")
    return []  # delete is not a change worth a line of its own


def _command_rows(name: str, args: dict[str, Any], result: str, ok: bool) -> list[tuple[str, str, str]]:
    if name == "run_python":
        return _command("python snippet", result, ok)
    if name == "run_javascript":
        return _command("javascript snippet", result, ok)
    if name == "terminal_session" and _flat(args.get("action")) not in ("start", "run"):
        return []
    if name == "remote" and _flat(args.get("action")) != "run":
        return []
    return _command(_flat(args.get("command")) or _flat(args.get("session_id")), result, ok)


def _command(
    command: Any, result: str, ok: bool, detail: str | None = None
) -> list[tuple[str, str, str]]:
    key = _flat(command, COMMAND_CHARS)
    if not key:
        return []
    return [("command", key, _exit_detail(result, ok) if detail is None else detail)]


def _multi_command_rows(commands: Any, result: str, ok: bool) -> list[tuple[str, str, str]]:
    """One row per command, each with the exit code of its own section of the result."""
    listed = commands if isinstance(commands, (list, tuple)) else []
    codes = [code for code in _EXIT_CODE_RE.findall(result)]
    rows: list[tuple[str, str, str]] = []
    for index, command in enumerate(listed):
        detail = f"exit {codes[index]}" if index < len(codes) else ("" if ok else "failed")
        rows.extend(_command(command, result, ok, detail))
    return rows


def _exit_detail(result: str, ok: bool) -> str:
    match = _EXIT_CODE_RE.search(result) or _PAREN_EXIT_RE.search(result)
    if match:
        return f"exit {match.group(1)}"
    return "" if ok else "failed"


def _page_rows(name: str, args: dict[str, Any], ok: bool) -> list[tuple[str, str, str]]:
    if name == "fetch_pages":
        urls = args.get("urls")
        return [_page(url, "read", ok) for url in (urls if isinstance(urls, list) else [])]
    if name == "browser" and _flat(args.get("action")) != "open":
        return []
    return [_page(args.get("url"), "read", ok)]


def _search_rows(name: str, args: dict[str, Any], ok: bool) -> list[tuple[str, str, str]]:
    if name == "multi_search":
        queries = args.get("queries")
        return [_page(_quoted(query), "searched", ok) for query in (queries if isinstance(queries, list) else [])]
    return [_page(_quoted(args.get("query")), "searched", ok)]


def _page(value: Any, detail: str, ok: bool) -> tuple[str, str, str]:
    """A page row; the key may come out empty and is dropped by the caller's filter."""
    return ("page", _flat(value), detail if ok else "failed")


def _quoted(value: Any) -> str:
    query = _flat(value, ASK_CHARS)
    return f'"{query}"' if query else ""


# --- reading ----------------------------------------------------------------


def block(db: Database, conversation_id: str, through_seq: int, budget_chars: int = 1400) -> str:
    """The note the model sees: what happened in the steps that scrolled out of view.

    Only steps up to ``through_seq`` are counted, so the note stays a prefix of the chat's
    own history. Over ``budget_chars`` the oldest entries of the longest line go first,
    leaving "and N earlier" behind, because a half-read list is worse than a stated cut."""
    through = int(through_seq or 0)
    if through <= 0:
        return ""
    grouped: dict[str, dict[str, list[tuple[int, str]]]] = {}
    for row in db.list_ledger(conversation_id, through_seq=through):
        key = str(row.get("key") or "")
        if not key:
            continue
        steps = grouped.setdefault(str(row.get("kind") or ""), {}).setdefault(key, [])
        step = (int(row["seq"]), str(row.get("detail") or ""))
        if step not in steps:
            steps.append(step)

    notes: dict[str, list[tuple[int, str]]] = {}
    for kind in ("note", "task"):
        for key, steps in grouped.get(kind, {}).items():
            notes.setdefault(key, []).extend(steps)
    for steps in notes.values():
        steps.sort()
    groups = [
        ("Asked", "ask", grouped.get("ask", {})),
        ("Files changed", "file", grouped.get("file", {})),
        ("Commands", "command", grouped.get("command", {})),
        ("Pages", "page", grouped.get("page", {})),
        ("Notes and tasks", "note", notes),
    ]
    lines = [
        {"title": title, "kind": kind, "items": [(key, steps) for key, steps in entries.items()], "dropped": 0}
        for title, kind, entries in groups
        if entries
    ]
    if not lines:
        return HEADER
    while len(HEADER) + sum(len(_line(line)) + 1 for line in lines) > budget_chars:
        longest = [line for line in lines if len(line["items"]) > 1]
        if not longest:
            break
        line = max(longest, key=lambda candidate: len(_line(candidate)))
        line["items"].pop(0)
        line["dropped"] += 1
    return HEADER + "\n" + "\n".join(_line(line) for line in lines)


def _line(line: dict[str, Any]) -> str:
    separator = ", " if line["kind"] == "file" else "; "
    text = f"{line['title']}: " + separator.join(
        _entry(line["kind"], key, steps) for key, steps in line["items"]
    )
    return text + (f"; and {line['dropped']} earlier" if line["dropped"] else "")


def _entry(kind: str, key: str, steps: list[tuple[int, str]]) -> str:
    shown = steps[-STEPS_PER_KEY:]
    earlier = len(steps) - len(shown)
    seqs = ", ".join(str(seq) for seq, _detail in shown)
    word = "step" if len(shown) == 1 else "steps"
    more = f" and {earlier} more" if earlier else ""
    if kind == "ask":
        return f"({word} {seqs}{more}) {key}"
    label = f'"{key}"' if kind in ("note", "task") else key
    details = {detail for _seq, detail in shown}
    if not any(details):
        return f"{label} ({word} {seqs}{more})"
    if len(details) == 1:
        return f"{label} ({word} {seqs}{more}: {shown[-1][1]})"
    parts = "; ".join(f"step {seq}: {detail}" for seq, detail in shown)
    tail = f"; and {earlier} more" if earlier else ""
    return f"{label} ({parts}{tail})"


def step_text(db: Database, conversation_id: str, step: int) -> str | None:
    """One earlier step in full as a recall line, cut at ``STEP_READ_CHARS``; None if absent."""
    number = int(step)
    message = next(iter(db.list_messages_range(conversation_id, number, number)), None)
    if message is None:
        return None
    line = f"- [{step_label(number, message.get('role'))}] {_flat(message.get('content'))}"
    if len(line) > STEP_READ_CHARS:
        line = line[:STEP_READ_CHARS] + f" [cut: {len(line) - STEP_READ_CHARS} more characters]"
    return line
