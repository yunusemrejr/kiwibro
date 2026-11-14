"""Safe copy sessions inside the app: one folder per chat, the gate on tools, and the chat messages.

A chat that has a safe copy open works only inside the copy the session keeps in Kiwibro's data
folder. Every tool call is checked here before it runs: file and command tools work in the copy,
Kiwibro's own web tools and memory stay available, and anything that acts on the desktop or outside
the copy is refused, whatever autonomy or approval setting is chosen. Commands run in the sandbox
with the copy as the only writable place and no network. Nothing reaches the real folder until the
user clicks apply on a proof, and removals are handed to the user as commands to run.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config, plan, safecopy, safety, tools
from .sandbox import (
    COMMAND_TIMEOUT_S,
    MAX_COMMAND_TIMEOUT_S,
    SandboxError,
    Workspace,
    check_command,
    command_env,
    run_process,
)

log = logging.getLogger(__name__)

ROOT_NAME = "safecopy"
SHELL = shutil.which("bash") or "/bin/sh"
MAX_LISTED_FILES = 30
APPLYING_TOOLS = plan.WRITE_TOOLS | plan.RUN_TOOLS | {"file_manage", "component", "undo_changes"}
_ROOT: Path | None = None

# Tools that act on the whole computer instead of the copy. Each reason is one short line the
# model can pass on as it is.
REFUSED: dict[str, str] = {
    "remote": "Remote control is not available in a safe copy.",
    "open_path": "Opening things on the desktop is not available in a safe copy.",
    "clipboard": "The clipboard is outside the safe copy, so it is not available here.",
    "screenshot": "Screenshots show the whole desktop, so they are not available in a safe copy.",
    "notify": "Notifications are outside the safe copy, so they are not available here.",
    "system_info": "It reports on the whole computer, so it is not available in a safe copy.",
    "disk_usage": "It measures the whole computer, so it is not available in a safe copy.",
    "describe_image": "Desktop tools are not available in a safe copy.",
    "workspace": "The workspace cannot change while a safe copy is open.",
    "terminal_session": "Background terminals do not run inside the safe copy, so they are not available.",
    "delegate": "Handing work to another agent is not available in a safe copy.",
}
COMMAND_TOOLS = frozenset({"run_command", "multi_command", "run_python", "run_javascript"})


@dataclass(frozen=True)
class Gate:
    """What the agent needs to know before it runs one tool call in a chat."""

    active: bool
    refusal: str | None


def configure(data_dir: Path) -> None:
    """The running app's data folder, so copies live beside its database (tests run their own)."""
    global _ROOT
    _ROOT = Path(data_dir) / ROOT_NAME


def sessions_root() -> Path:
    """Where every safe copy lives: Kiwibro's own data folder, never the user's folder."""
    return _ROOT if _ROOT is not None else config.data_dir() / ROOT_NAME


# --- the chat's session ----------------------------------------------------------------------


def attachment(db: Any, conversation_id: str | None) -> dict[str, Any] | None:
    """The session this chat remembers: {session_id, real, previous_workspace}."""
    if not conversation_id:
        return None
    conversation = db.get_conversation(conversation_id)
    if not conversation:
        return None
    raw = conversation.get("safe_copy")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict) or not data.get("session_id"):
        return None
    return data


def current(db: Any, conversation_id: str | None) -> safecopy.Session | None:
    """The open session of a chat, or None."""
    data = attachment(db, conversation_id)
    if data is None:
        return None
    try:
        return safecopy.load(sessions_root(), str(data["session_id"]))
    except safecopy.SafeCopyError:
        log.warning("safe copy %s of chat %s is gone", data.get("session_id"), conversation_id)
        return None


def copy_folder(db: Any, conversation_id: str | None) -> str | None:
    """The copy the chat's tools work in (its work folder), or None when no session is open."""
    session = current(db, conversation_id)
    return str(session.work) if session is not None else None


def workspace_for(db: Any, conversation_id: str | None) -> Workspace | None:
    """The confined workspace of an open session: files and commands stay in the copy."""
    session = current(db, conversation_id)
    return SafeWorkspace(session.work) if session is not None else None


def gate(db: Any, conversation_id: str | None, name: str) -> Gate:
    """Whether a tool may run in this chat, and why not when the chat has a session."""
    if current(db, conversation_id) is None:
        return Gate(active=False, refusal=None)
    return Gate(active=True, refusal=refusal_for(name))


def refusal_for(name: str) -> str | None:
    """The one line that says why a tool cannot run in a safe copy (None: it may run)."""
    if name in REFUSED:
        return REFUSED[name]
    if name in tools._EXTENSIONS:
        return "Third-party tools are not available in a safe copy."
    if name not in tools.TOOL_SPECS:
        return "That tool is not available in a safe copy."
    if name in COMMAND_TOOLS and not sandbox_ready():
        return "Commands need the sandbox, and it is not available on this machine."
    return None


def sandbox_ready() -> bool:
    """Whether commands can be confined here at all; without it nothing runs in a session."""
    try:
        return bool(safety.sandbox_status().get("active"))
    except Exception:  # a broken check must never mean "unconfined"
        return False


def context_line(db: Any, conversation_id: str | None) -> str:
    """The one line that tells the model where it is, near the top of the turn context."""
    data = attachment(db, conversation_id)
    if data is None or current(db, conversation_id) is None:
        return ""
    return (
        f"Safe copy: you are working in a copy of {config.display_path(str(data.get('real', '')))}. "
        "Nothing reaches the real folder until the user reviews the changes."
    )


def commit_turn(db: Any, turn: Any) -> None:
    """Record the step in the session's git history after a turn that wrote or ran something."""
    if getattr(turn, "step_recorded", False):
        return  # a turn that finished recorded its step before it said so
    try:
        turn.step_recorded = True
        conversation_id = getattr(turn, "conversation_id", None)
        session = current(db, conversation_id)
        if session is None:
            return
        used = sorted({name for name in getattr(turn, "tools_ok", []) if name in APPLYING_TOOLS})
        if not used:
            return
        session.commit(f"agent: {', '.join(used)}"[:200])
    except Exception:  # recording a step must never break the turn
        log.warning("could not record the step in the safe copy", exc_info=True)


# --- starting, looking, ending ---------------------------------------------------------------


def start(db: Any, conversation_id: str, folder: str) -> dict[str, Any]:
    """Open a session for this chat (or return the one it already has)."""
    if db.get_conversation(conversation_id) is None:
        raise safecopy.SafeCopyError("not_found", "That chat is not there.")
    session = current(db, conversation_id)
    if session is not None:
        return {**status(db, conversation_id), "started": False}
    real = _checked_folder(folder)
    session_id = f"sc-{uuid.uuid4().hex[:12]}"
    previous = (db.get_conversation(conversation_id) or {}).get("workspace")
    safecopy.start(real, sessions_root(), session_id)
    db.update_conversation(
        conversation_id,
        safe_copy=json.dumps(
            {"session_id": session_id, "real": str(real), "previous_workspace": previous}
        ),
    )
    return {**status(db, conversation_id), "started": True}


def _checked_folder(folder: str) -> Path:
    raw = str(folder or "").strip()
    if not raw:
        raise safecopy.SafeCopyError("not_a_folder", "Pick a folder first.")
    candidate = Path(raw).expanduser()
    if not candidate.is_dir():
        raise safecopy.SafeCopyError("not_a_folder", "That is not a folder, so nothing was copied.")
    real = candidate.resolve()
    data = sessions_root().parent.resolve()
    if real.parent == real:
        raise safecopy.SafeCopyError("refused_folder", "The whole filesystem cannot go into a safe copy.")
    if real == Path.home().resolve():
        raise safecopy.SafeCopyError("refused_folder", "Your home folder itself cannot go into a safe copy.")
    if real == data or data.is_relative_to(real) or real.is_relative_to(data):
        raise safecopy.SafeCopyError("refused_folder", "Kiwibro's own data folder cannot go into a safe copy.")
    return real


def status(db: Any, conversation_id: str) -> dict[str, Any]:
    """A few facts for the page: where the copy is, what waits, what was applied, disk use."""
    view: dict[str, Any] = {
        "active": False,
        "real": None,
        "work": None,
        "changed_files": 0,
        "disk_bytes": 0,
        "left_out": [],
        "applied": False,
        "can_undo": False,
    }
    data = attachment(db, conversation_id)
    session = current(db, conversation_id)
    if data is None or session is None:
        return view
    counts = session.status()
    applies = session.last_apply()
    return {
        "active": True,
        "real": str(session.real),
        "work": str(session.work),
        "changed_files": int(counts["changes"]["total"]),
        "disk_bytes": int(counts["bytes"]),
        "left_out": session.left_out,
        "applied": bool(applies),
        "can_undo": session.can_undo(),
    }


def proof(db: Any, conversation_id: str) -> dict[str, Any]:
    """The proof of what the copy changed, for the page that asks the user to approve it."""
    session = _required(db, conversation_id)
    return session.proof().to_dict()


def apply(db: Any, conversation_id: str, fingerprint: str) -> dict[str, Any]:
    """Write the approved proof into the real folder and say in the chat how to undo it."""
    session = _required(db, conversation_id)
    if not isinstance(fingerprint, str) or not fingerprint:
        raise safecopy.SafeCopyError("stale", "Make a new proof and read it again.")
    applied = session.apply(fingerprint)
    db.add_message(
        conversation_id,
        "assistant",
        apply_message(applied, session.real),
        meta={"kind": "safe_copy"},
    )
    return {
        "written": applied.written,
        "removals": applied.removals,
        "revert": applied.revert,
        "before_commit": applied.before_commit,
        "applied_commit": applied.applied_commit,
        "left_out": applied.left_out,
    }


def undo(db: Any, conversation_id: str) -> dict[str, Any]:
    """Put the last apply's files back and say in the chat what stayed."""
    session = _required(db, conversation_id)
    reverted = session.revert_last()
    db.add_message(
        conversation_id,
        "assistant",
        undo_message(reverted, session.real),
        meta={"kind": "safe_copy"},
    )
    return {"restored": reverted.restored, "kept": reverted.kept, "before_commit": reverted.before_commit}


def end(db: Any, conversation_id: str, discard: bool) -> dict[str, Any]:
    """Detach the chat from its session, and remove the copy when the user asks for that."""
    data = attachment(db, conversation_id)
    if data is None:
        raise safecopy.SafeCopyError("not_found", "No safe copy is open in this chat.")
    session = current(db, conversation_id)
    discarded = False
    if discard and session is not None:
        session.discard()
        discarded = True
    db.update_conversation(conversation_id, safe_copy=None, workspace=data.get("previous_workspace"))
    return {"active": False, "discarded": discarded}


def cleanup_at_start(db: Any, max_age_days: float = 14) -> list[str]:
    """Remove sessions nobody is attached to any more, so the copies do not pile up."""
    keep: set[str] = set()
    try:
        for conversation in db.list_conversations():
            raw = conversation.get("safe_copy")
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except ValueError:
                continue
            if isinstance(data, dict) and data.get("session_id"):
                keep.add(str(data["session_id"]))
        return safecopy.cleanup(sessions_root(), keep, max_age_days=max_age_days)
    except Exception:  # housekeeping must never block startup
        log.warning("could not clean up old safe copies", exc_info=True)
        return []


def _required(db: Any, conversation_id: str) -> safecopy.Session:
    session = current(db, conversation_id)
    if session is None:
        raise safecopy.SafeCopyError("not_found", "No safe copy is open in this chat.")
    return session


# --- what the user reads in the chat ---------------------------------------------------------


def apply_message(applied: safecopy.Applied, real: Path) -> str:
    """The chat message after an apply: what was written, the undo command, the removals to run."""
    lines = [f"Safe copy applied: {_files(len(applied.written))} written to {config.display_path(str(real))}."]
    lines.append("")
    lines.extend(f"- {path}" for path in applied.written[:MAX_LISTED_FILES])
    if len(applied.written) > MAX_LISTED_FILES:
        lines.append(f"- and {len(applied.written) - MAX_LISTED_FILES} more")
    lines.extend(["", "Undo this in a terminal:", "", "```", str(applied.revert.get("command", "")), "```"])
    if applied.removals:
        lines.extend([
            "",
            "These files were deleted in the copy. Kiwibro never deletes anything: "
            "run these yourself if you want them gone:",
            "",
            "```",
            "\n".join(entry["command"] for entry in applied.removals),
            "```",
        ])
    return "\n".join(lines)


def undo_message(reverted: safecopy.Reverted, real: Path) -> str:
    """The chat message after an undo: what came back, and the commands for what stayed."""
    lines = [f"Safe copy undone: {_files(len(reverted.restored))} put back in {config.display_path(str(real))}."]
    if reverted.restored:
        lines.append("")
        lines.extend(f"- {path}" for path in reverted.restored[:MAX_LISTED_FILES])
    if reverted.kept:
        lines.extend([
            "",
            "Files this apply had added stay in your folder. Kiwibro never deletes anything: "
            "run these yourself if you want them gone:",
            "",
            "```",
            "\n".join(entry["command"] for entry in reverted.kept),
            "```",
        ])
    return "\n".join(lines)


def _files(count: int) -> str:
    return "1 file" if count == 1 else f"{count} files"


# --- the confined workspace ------------------------------------------------------------------


class SafeWorkspace(Workspace):
    """The copy: commands run in the sandbox, with the copy as the only writable place and no network."""

    def run_command(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int = COMMAND_TIMEOUT_S,
        *,
        stdin: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not sandbox_ready():
            raise SandboxError("commands need the sandbox, and it is not available on this machine")
        if safety.is_unconfined():  # an approved step that may leave the box: a copy never does
            raise SandboxError("this step was approved to run outside the sandbox, and a safe copy never leaves it")
        check_command(command)
        workdir = self.resolve(cwd or ".", must_exist=True)
        if not workdir.is_dir():
            raise SandboxError(f"cwd is not a directory: {self.public(workdir)}")
        timeout = max(1, min(int(timeout or COMMAND_TIMEOUT_S), MAX_COMMAND_TIMEOUT_S))
        env = command_env(self.root, workdir, env_vars)
        argv = safety.confine([SHELL, "-c", command], writable=[self.root], cwd=workdir, network=False)
        started = time.monotonic()
        try:
            exit_code, output = run_process(argv, cwd=workdir, env=env, timeout=timeout, stdin=stdin)
        except OSError as exc:
            raise SandboxError(f"could not run command: {exc.strerror or exc}") from exc
        duration = int((time.monotonic() - started) * 1000)
        timed_out = exit_code is None
        if timed_out:
            partial = output.strip()
            output = f"Command timed out after {timeout}s."
            if partial:
                clipped, _ = self._truncate_output(partial, max_bytes=160)
                output += f"\nPartial output:\n{clipped}"
            truncated = False
        else:
            output, truncated = self._truncate_output(output)
            if not output.strip():
                output = "(no output)"
            if exit_code < 0:
                output += (
                    f"\n[killed by signal {-exit_code}: the sandbox stops commands that exceed its "
                    "memory or process limits]"
                )
        return {
            "command": command,
            "cwd": self.public(workdir),
            "exit_code": exit_code,
            "output": output,
            "truncated": truncated,
            "duration_ms": duration,
            "timed_out": timed_out,
        }
