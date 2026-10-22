"""Workspace sandbox: path containment, file tools and safe command execution."""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config, safety

log = logging.getLogger(__name__)

MAX_READ_BYTES = config.MAX_FILE_READ_BYTES
MAX_WRITE_BYTES = config.MAX_FILE_WRITE_BYTES
MAX_OUTPUT_BYTES = config.MAX_COMMAND_OUTPUT_BYTES
COMMAND_TIMEOUT_S = 60
MAX_COMMAND_TIMEOUT_S = 120
LIST_MAX_ENTRIES = 200
LIST_MAX_DEPTH = 2
GREP_MAX_MATCHES = 120
SESSION_BUFFER_CHARS = 64 * 1024
MAX_TERMINAL_SESSIONS = 12
PRUNED_DIRS = {"node_modules", "__pycache__", ".git", ".venv", "venv", "dist", "build", ".cache", ".idea", ".vscode"}

BLOCKED_COMMANDS: tuple[tuple[str, str], ...] = (
    (r"\bsudo\b", "sudo"),
    (r"(?:^|[;&|]\s*)su\s", "su"),
    (r"\bdoas\b", "doas"),
    (r"\bmkfs(?:\.[a-z0-9]+)?\b", "mkfs"),
    (r"\bshutdown\b", "shutdown"),
    (r"\breboot\b", "reboot"),
    (r"\bpoweroff\b", "poweroff"),
    (r"\bhalt\b", "halt"),
    (r"\binit\s+[06]\b", "init"),
    (r"\bdd\b[^\n]*\bof\s*=\s*/dev/", "dd to a device"),
    (r":\s*\(\s*\)\s*\{", "fork bomb"),
    (r"\bchmod\s+(?:-[a-zA-Z]+\s+)*777\s+/", "chmod 777 on root"),
    (r"\bchown\s+(?:-[a-zA-Z]+\s+)*[^\s]+\s+/(?:\s|$)", "chown on root"),
    (r"\b(?:systemctl|service)\s+(?:poweroff|reboot|halt|shutdown)", "service power control"),
    (r"\bkill\s+(?:-9\s+)?1\b", "kill init"),
    (r">\s*/dev/(?:sd|nvme|hd)", "write to a block device"),
)
_BLOCKED_RM = re.compile(r"\brm\b[^\n;&|]*\s/(?=\s|$|\*|\.)", re.IGNORECASE)
_BLOCKED_RM_ROOT = re.compile(r"\brm\s+(-[^\s]+\s+)*-[a-z]*[rf][a-z]*\b", re.IGNORECASE)


class SandboxError(Exception):
    """Raised for unsafe paths or commands; message is user-facing."""


def check_command(command: str) -> None:
    if not command or not command.strip():
        raise SandboxError("command is empty")
    if len(command) > 4000:
        raise SandboxError("command is too long")
    for pattern, label in BLOCKED_COMMANDS:
        if re.search(pattern, command, re.IGNORECASE):
            raise SandboxError(f"command blocked for safety ({label})")
    if _BLOCKED_RM.search(command) and _BLOCKED_RM_ROOT.search(command):
        raise SandboxError("command blocked for safety (rm on root)")


_SHELL = shutil.which("bash") or "/bin/sh"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def command_env(root: Path, workdir: Path, env_vars: dict[str, str] | None = None, **extra: str) -> dict[str, str]:
    """Minimal environment for workspace commands.

    HOME is Kiwibro's tool home (safety.tool_home) so caches never land in the
    user's projects; git still sees the user's identity through GIT_CONFIG_GLOBAL.
    A high-risk command the user explicitly approved gets the real HOME, since it
    may need the user's own credentials (npm publish, gh, cloud CLIs)."""
    real_home = Path.home()
    gitconfig = real_home / ".gitconfig"
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(real_home if safety.is_unconfined() else safety.tool_home()),
        **({"GIT_CONFIG_GLOBAL": str(gitconfig)} if gitconfig.is_file() else {}),
        "PWD": str(workdir),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TERM": "dumb",
        **extra,
    }
    for key, value in (env_vars or {}).items():
        if isinstance(key, str) and _ENV_NAME.match(key) and key not in ("PATH", "HOME") and key not in safety.DANGEROUS_ENV:
            env[key] = str(value)[:1000]
    return env


def shell_argv(command: str, root: Path, workdir: Path) -> list[str]:
    """`bash -c command`, confined to the workspace when a sandbox is available."""
    return safety.confine([_SHELL, "-c", command], writable=[root, safety.tool_home()], cwd=workdir)


def kill_group(proc: subprocess.Popen[str], sig: int = signal.SIGKILL) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except OSError:
            pass


def run_process(
    argv: list[str], *, cwd: Path, env: dict[str, str], timeout: float, stdin: str | None = None
) -> tuple[int | None, str]:
    """Run argv in its own process group with stdout+stderr merged.

    Returns (exit code, output); the exit code is None when the timeout fired,
    in which case the whole group is killed so no grandchild keeps the pipes
    (and the turn) hanging."""
    proc = subprocess.Popen(
        argv,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(stdin, timeout=timeout)
        return proc.returncode, output or ""
    except subprocess.TimeoutExpired:
        kill_group(proc)
        try:
            output, _ = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            output = ""
        return None, output or ""


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


class Workspace:
    """A root directory every file/command tool is confined to."""

    def __init__(self, root: str | Path) -> None:
        self._root = str(root)

    @property
    def root(self) -> Path:
        resolved = Path(self._root).expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def resolve(self, raw_path: str | Path | None, *, must_exist: bool = False) -> Path:
        root = self.root
        raw = str(raw_path if raw_path not in (None, "") else ".").strip()
        if "\x00" in raw:
            raise SandboxError("path contains invalid characters")
        candidate = Path(raw)
        if raw == "~" or raw.startswith("~/"):  # models copy the "~/Kiwibro" form from the prompt
            candidate = candidate.expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise SandboxError("path could not be resolved") from exc
        if resolved != root and not resolved.is_relative_to(root):
            raise SandboxError("path is outside the workspace; use a path relative to it, like '.' or 'src/app.py'")
        if must_exist and not resolved.exists():
            raise SandboxError(f"path does not exist: {self.public(resolved)}")
        return resolved

    def public(self, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(self.root)
        except (ValueError, OSError):
            # Never disclose absolute paths for targets outside the workspace.
            return "(outside workspace)"
        text = relative.as_posix()
        return text or "."

    # --- file tools --------------------------------------------------------

    def read_file(
        self,
        path: str,
        start: int | None = None,
        end: int | None = None,
        *,
        line_numbers: bool = False,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        if paths:
            batch_items: list[dict[str, Any]] = []
            chunks: list[str] = []
            total_lines = 0
            total_size = 0
            for item_path in paths[:10]:
                single = self.read_file(item_path, start=start, end=end, line_numbers=line_numbers)
                batch_items.append(single)
                total_lines += int(single["total_lines"])
                total_size += int(single["size"])
                chunks.append(f"=== {single['path']} (lines {single['start']}-{single['end']} of {single['total_lines']}) ===\n{single['content']}")
            joined = "\n\n".join(chunks)
            encoded = joined.encode("utf-8", errors="replace")
            if len(encoded) > MAX_READ_BYTES:
                joined = encoded[:MAX_READ_BYTES].decode("utf-8", errors="replace") + "\n… [batch truncated]"
            return {
                "path": ", ".join(item["path"] for item in batch_items),
                "paths": [item["path"] for item in batch_items],
                "content": joined,
                "start": start or 1,
                "end": end or total_lines,
                "total_lines": total_lines,
                "size": total_size,
            }

        resolved = self.resolve(path, must_exist=True)
        if not resolved.is_file():
            raise SandboxError(f"not a file: {self.public(resolved)}")
        try:
            size = resolved.stat().st_size
        except OSError as exc:
            raise SandboxError("could not stat file") from exc
        if size > MAX_READ_BYTES:
            raise SandboxError(f"file is too large ({size} bytes; limit {MAX_READ_BYTES})")
        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise SandboxError(f"could not read file: {exc.strerror or exc}") from exc
        lines = text.splitlines()
        line_count = len(lines)
        if start is not None or end is not None:
            first = max(1, start or 1)
            last = min(line_count, end or line_count)
            if first > last and line_count > 0:
                raise SandboxError("start line is after end line")
            selected = lines[first - 1 : last]
            if line_numbers:
                selected = [f"{idx}: {line}" for idx, line in enumerate(selected, start=first)]
            return {
                "path": self.public(resolved),
                "content": "\n".join(selected),
                "start": first,
                "end": last,
                "total_lines": line_count,
                "size": size,
            }
        content = "\n".join(f"{idx}: {line}" for idx, line in enumerate(lines, start=1)) if line_numbers else text
        return {
            "path": self.public(resolved),
            "content": content,
            "start": 1,
            "end": line_count,
            "total_lines": line_count,
            "size": size,
        }

    def grep_files(
        self,
        pattern: str,
        path: str = ".",
        *,
        glob: str | None = None,
        regex: bool = True,
        case_sensitive: bool = False,
        context_lines: int = 0,
        max_results: int = 50,
    ) -> dict[str, Any]:
        if not pattern or not str(pattern).strip():
            raise SandboxError("search pattern is empty")
        resolved = self.resolve(path, must_exist=True)
        context_lines = max(0, min(int(context_lines or 0), 5))
        max_results = max(1, min(int(max_results or 50), GREP_MAX_MATCHES))
        flags = 0 if case_sensitive else re.IGNORECASE
        if regex:
            try:
                compiled = re.compile(pattern, flags)
            except re.error as exc:
                raise SandboxError(f"invalid regular expression: {exc}") from exc
        else:
            compiled = re.compile(re.escape(pattern), flags)

        root = self.root
        matches: list[dict[str, Any]] = []
        files_searched = 0
        truncated = False

        def candidate_files(target: Path):
            if target.is_file():
                yield target
                return
            stack = [target]
            while stack:
                current = stack.pop()
                try:
                    children = sorted(current.iterdir(), key=lambda item: item.name.lower())
                except OSError:
                    continue
                for child in children:
                    if child.name.startswith(".") or child.name in PRUNED_DIRS:
                        continue
                    try:
                        resolved_child = child.resolve()
                        if resolved_child != root and not resolved_child.is_relative_to(root):
                            continue
                    except OSError:
                        continue
                    if child.is_dir():
                        stack.append(child)
                    elif child.is_file():
                        yield child

        for file_path in candidate_files(resolved):
            rel = self.public(file_path)
            if glob and not (fnmatch.fnmatch(file_path.name, glob) or fnmatch.fnmatch(rel, glob)):
                continue
            try:
                stat = file_path.stat()
                if stat.st_size > MAX_READ_BYTES or stat.st_size == 0:
                    continue
                raw = file_path.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:2048]:
                continue
            files_searched += 1
            text = raw.decode("utf-8", errors="replace")
            lines = text.splitlines()
            for idx, line in enumerate(lines):
                if compiled.search(line):
                    entry: dict[str, Any] = {
                        "path": rel,
                        "line": idx + 1,
                        "text": line[:400],
                    }
                    if context_lines > 0:
                        lo = max(0, idx - context_lines)
                        hi = min(len(lines), idx + context_lines + 1)
                        entry["context"] = [
                            f"{line_no + 1}{':' if line_no == idx else '-'} {lines[line_no][:300]}"
                            for line_no in range(lo, hi)
                        ]
                    matches.append(entry)
                    if len(matches) >= max_results:
                        truncated = True
                        break
            if truncated:
                break

        return {
            "pattern": pattern,
            "path": self.public(resolved),
            "files_searched": files_searched,
            "matches": matches,
            "count": len(matches),
            "truncated": truncated,
        }

    def list_files(self, path: str = ".", depth: int = LIST_MAX_DEPTH) -> list[dict[str, Any]]:
        resolved = self.resolve(path, must_exist=True)
        if not resolved.is_dir():
            raise SandboxError(f"not a directory: {self.public(resolved)}")
        depth = max(1, min(int(depth), LIST_MAX_DEPTH))
        entries: list[dict[str, Any]] = []
        root = self.root

        def walk(directory: Path, level: int) -> None:
            if len(entries) >= LIST_MAX_ENTRIES:
                return
            try:
                children = sorted(directory.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
            except OSError:
                return
            for child in children:
                if len(entries) >= LIST_MAX_ENTRIES:
                    return
                if child.name.startswith(".") or child.name in PRUNED_DIRS:
                    continue
                try:
                    resolved_child = child.resolve()
                    if resolved_child != root and not resolved_child.is_relative_to(root):
                        continue
                    stat = child.stat()
                except OSError:
                    continue
                is_dir = child.is_dir()
                entries.append(
                    {
                        "path": self.public(child),
                        "type": "dir" if is_dir else "file",
                        "size": 0 if is_dir else int(stat.st_size),
                        "modified": _iso(stat.st_mtime),
                    }
                )
                if is_dir and level < depth:
                    walk(child, level + 1)

        walk(resolved, 1)
        return entries

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        if not isinstance(content, str):
            raise SandboxError("content must be text")
        payload = content.encode("utf-8")
        if len(payload) > MAX_WRITE_BYTES:
            raise SandboxError(f"content is too large ({len(payload)} bytes; limit {MAX_WRITE_BYTES})")
        resolved = self.resolve(path)
        if resolved == self.root:
            raise SandboxError("path must name a file inside the workspace")
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            if resolved.exists() and resolved.is_dir():
                raise SandboxError("path is a directory")
            resolved.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise SandboxError(f"could not write file: {exc.strerror or exc}") from exc
        return {"path": self.public(resolved), "bytes": len(payload)}

    def replace_in_file(
        self,
        path: str,
        old: str = "",
        new: str = "",
        *,
        replace_all: bool = False,
        edits: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        resolved = self.resolve(path, must_exist=True)
        if not resolved.is_file():
            raise SandboxError(f"not a file: {self.public(resolved)}")
        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise SandboxError(f"could not read file: {exc.strerror or exc}") from exc

        operations = edits if edits else [{"old": old, "new": new, "replace_all": replace_all}]
        total_replacements = 0
        updated = text
        for edit in operations:
            target = str(edit.get("old") or "")
            replacement = str(edit.get("new") if edit.get("new") is not None else "")
            all_occurrences = bool(edit.get("replace_all", replace_all))
            if not target:
                raise SandboxError("old text is empty")
            count = updated.count(target)
            if count == 0:
                raise SandboxError("old text was not found in the file")
            if count > 1 and not all_occurrences:
                raise SandboxError(f"old text matched {count} times; it must match exactly once")
            if all_occurrences:
                updated = updated.replace(target, replacement)
                total_replacements += count
            else:
                updated = updated.replace(target, replacement, 1)
                total_replacements += 1

        if len(updated.encode("utf-8")) > MAX_WRITE_BYTES:
            raise SandboxError("resulting file would be too large")
        try:
            resolved.write_text(updated, encoding="utf-8")
        except OSError as exc:
            raise SandboxError(f"could not write file: {exc.strerror or exc}") from exc
        return {"path": self.public(resolved), "replacements": total_replacements}

    # --- shell -------------------------------------------------------------

    @staticmethod
    def _truncate_output(output: str, max_bytes: int = MAX_OUTPUT_BYTES) -> tuple[str, bool]:
        encoded = output.encode("utf-8", errors="replace")
        if len(encoded) <= max_bytes:
            return output, False
        marker = f"\n… [truncated {len(encoded) - max_bytes} bytes] …\n".encode("utf-8")
        if max_bytes <= len(marker) + 64:
            return encoded[:max_bytes].decode("utf-8", errors="replace"), True
        budget = max_bytes - len(marker)
        head_bytes = int(budget * 0.6)
        tail_bytes = budget - head_bytes
        trimmed = encoded[:head_bytes] + marker + encoded[-tail_bytes:]
        return trimmed.decode("utf-8", errors="replace"), True

    def run_command(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int = COMMAND_TIMEOUT_S,
        *,
        stdin: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        check_command(command)
        workdir = self.resolve(cwd or ".", must_exist=True)
        if not workdir.is_dir():
            raise SandboxError(f"cwd is not a directory: {self.public(workdir)}")
        timeout = max(1, min(int(timeout or COMMAND_TIMEOUT_S), MAX_COMMAND_TIMEOUT_S))
        env = command_env(self.root, workdir, env_vars)
        started = time.monotonic()
        try:
            exit_code, output = run_process(
                shell_argv(command, self.root, workdir), cwd=workdir, env=env, timeout=timeout, stdin=stdin
            )
        except OSError as exc:
            raise SandboxError(f"could not run command: {exc.strerror or exc}") from exc
        duration = int((time.monotonic() - started) * 1000)
        timed_out = exit_code is None
        if timed_out:
            partial = output.strip()
            output = f"Command timed out after {timeout}s."
            if partial:
                clipped, _ = self._truncate_output(partial, max_bytes=max(200, MAX_OUTPUT_BYTES - 80))
                output += f"\nPartial output:\n{clipped}"
            truncated = False
        else:
            output, truncated = self._truncate_output(output, MAX_OUTPUT_BYTES)
            if not output.strip():
                output = "(no output)"
            if exit_code < 0:
                output += f"\n[killed by signal {-exit_code}: the sandbox stops commands that exceed its memory or process limits]"
        return {
            "command": command,
            "cwd": self.public(workdir),
            "exit_code": exit_code,
            "output": output,
            "truncated": truncated,
            "duration_ms": duration,
            "timed_out": timed_out,
        }


@dataclass
class TerminalSession:
    id: str
    conversation_id: str | None
    workspace_root: str
    cwd: str
    command: str
    started_at: float = field(default_factory=time.monotonic)
    created_iso: str = field(default_factory=config.now_iso)
    env_vars: dict[str, str] = field(default_factory=dict)
    process: subprocess.Popen[str] | None = None
    output_chunks: list[str] = field(default_factory=list)
    output_len: int = 0
    exit_code: int | None = None
    finished_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def append_output(self, chunk: str) -> None:
        if not chunk:
            return
        with self.lock:
            self.output_chunks.append(chunk)
            self.output_len += len(chunk)
            while self.output_len > SESSION_BUFFER_CHARS and len(self.output_chunks) > 1:
                removed = self.output_chunks.pop(0)
                self.output_len -= len(removed)

    def read_output(self, tail_chars: int = MAX_OUTPUT_BYTES) -> str:
        with self.lock:
            full = "".join(self.output_chunks)
        if len(full) <= tail_chars:
            return full
        return f"… [showing last {tail_chars} chars of {len(full)}]\n" + full[-tail_chars:]

    @property
    def is_running(self) -> bool:
        proc = self.process
        if proc is None:
            return False
        polled = proc.poll()
        if polled is not None:
            self.exit_code = polled
            if self.finished_at is None:
                self.finished_at = time.monotonic()
            return False
        return True

    def to_dict(self, workspace: Workspace | None = None, include_output: bool = False) -> dict[str, Any]:
        running = self.is_running
        end_t = time.monotonic() if running else (self.finished_at or time.monotonic())
        ws = workspace or Workspace(self.workspace_root)
        try:
            rel_cwd = ws.public(Path(self.cwd))
        except Exception:
            rel_cwd = "."
        info: dict[str, Any] = {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "command": self.command,
            "cwd": rel_cwd,
            "status": "running" if running else "exited",
            "running": running,
            "pid": self.process.pid if self.process else None,
            "exit_code": self.exit_code,
            "duration_ms": int((end_t - self.started_at) * 1000),
            "created_at": self.created_iso,
        }
        if include_output:
            info["output"] = self.read_output() or "(no output)"
        return info


class TerminalManager:
    """Manages persistent and background shell sessions confined to the workspace."""

    def __init__(self) -> None:
        self._sessions: dict[str, TerminalSession] = {}
        self._lock = threading.Lock()

    def _prune_unlocked(self) -> None:
        if len(self._sessions) < MAX_TERMINAL_SESSIONS:
            return
        exited = [s for s in self._sessions.values() if not s.is_running]
        exited.sort(key=lambda s: s.started_at)
        for s in exited[: max(1, len(self._sessions) - MAX_TERMINAL_SESSIONS + 1)]:
            self._sessions.pop(s.id, None)

    def start_background(
        self,
        workspace: Workspace,
        command: str,
        *,
        cwd: str | None = None,
        conversation_id: str | None = None,
        wait_ms: int = 800,
        env_vars: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        check_command(command)
        workdir = workspace.resolve(cwd or ".", must_exist=True)
        if not workdir.is_dir():
            raise SandboxError(f"cwd is not a directory: {workspace.public(workdir)}")

        with self._lock:
            self._prune_unlocked()
            running_count = sum(1 for s in self._sessions.values() if s.is_running)
            if running_count >= MAX_TERMINAL_SESSIONS:
                raise SandboxError(f"too many active background terminals ({running_count}); kill one first")
            session_id = f"term_{uuid.uuid4().hex[:10]}"

        env = command_env(workspace.root, workdir, env_vars, PYTHONUNBUFFERED="1")

        try:
            proc = subprocess.Popen(
                shell_argv(command, workspace.root, workdir),
                cwd=str(workdir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            raise SandboxError(f"could not start terminal session: {exc.strerror or exc}") from exc

        session = TerminalSession(
            id=session_id,
            conversation_id=conversation_id,
            workspace_root=str(workspace.root),
            cwd=str(workdir),
            command=command,
            env_vars=dict(env_vars or {}),
            process=proc,
        )
        with self._lock:
            self._sessions[session_id] = session

        def reader() -> None:
            stream = proc.stdout
            if stream is None:
                return
            try:
                while True:
                    chunk = stream.read(1024)
                    if not chunk:
                        break
                    session.append_output(chunk)
            except Exception:
                pass
            finally:
                rc = proc.poll()
                if rc is None:
                    try:
                        rc = proc.wait(timeout=1)
                    except Exception:
                        rc = None
                session.exit_code = rc
                session.finished_at = time.monotonic()

        threading.Thread(target=reader, name=f"kiwibro-{session_id}", daemon=True).start()

        wait_s = max(0.05, min(float(wait_ms or 800) / 1000.0, 10.0))
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and session.is_running:
            time.sleep(0.05)

        return session.to_dict(workspace=workspace, include_output=True)

    def run_stateful(
        self,
        workspace: Workspace,
        command: str,
        *,
        session_id: str | None = None,
        cwd: str | None = None,
        conversation_id: str | None = None,
        timeout: int = COMMAND_TIMEOUT_S,
        stdin: str | None = None,
    ) -> dict[str, Any]:
        """Run a command in a persistent stateful session that preserves cwd and env vars."""
        check_command(command)
        session: TerminalSession | None = None
        if session_id:
            with self._lock:
                session = self._sessions.get(session_id)
        if session is None:
            workdir = workspace.resolve(cwd or ".", must_exist=True)
            if not workdir.is_dir():
                raise SandboxError(f"cwd is not a directory: {workspace.public(workdir)}")
            sid = session_id if (session_id and re.match(r"^[\w\-]{1,32}$", session_id)) else f"term_{uuid.uuid4().hex[:10]}"
            session = TerminalSession(
                id=sid,
                conversation_id=conversation_id,
                workspace_root=str(workspace.root),
                cwd=str(workdir),
                command=command,
            )
            with self._lock:
                self._prune_unlocked()
                self._sessions[sid] = session
        else:
            if cwd:
                workdir = workspace.resolve(cwd, must_exist=True)
                session.cwd = str(workdir)

        marker = f"__KIWIBRO_PWD_{uuid.uuid4().hex[:8]}__"
        wrapped = f"{command}\n__kb_ec=$?\nprintf '\\n{marker}:%s\\n' \"$(pwd)\"\nexit $__kb_ec"
        rel_cwd = workspace.public(Path(session.cwd))
        res = workspace.run_command(
            wrapped,
            cwd=rel_cwd,
            timeout=timeout,
            stdin=stdin,
            env_vars=session.env_vars,
        )
        raw_out = res["output"]
        marker_re = re.compile(rf"\n?{re.escape(marker)}:([^\n]+)\n?")
        match = marker_re.search(raw_out)
        if match:
            new_pwd = match.group(1).strip()
            raw_out = marker_re.sub("", raw_out).strip()
            try:
                resolved_pwd = workspace.resolve(new_pwd, must_exist=True)
                if resolved_pwd.is_dir():
                    session.cwd = str(resolved_pwd)
            except SandboxError:
                pass
        if not raw_out:
            raw_out = "(no output)"
        session.command = command
        session.exit_code = res["exit_code"]
        session.append_output(f"$ {command}\n{raw_out}\n")
        return {
            "session_id": session.id,
            "command": command,
            "cwd": workspace.public(Path(session.cwd)),
            "exit_code": res["exit_code"],
            "output": raw_out,
            "truncated": res["truncated"],
            "duration_ms": res["duration_ms"],
            "timed_out": res["timed_out"],
            "running": False,
        }

    def status(self, session_id: str, workspace: Workspace | None = None, tail_chars: int = MAX_OUTPUT_BYTES) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SandboxError(f"unknown terminal session: {session_id}")
        info = session.to_dict(workspace=workspace, include_output=False)
        info["output"] = session.read_output(tail_chars=tail_chars) or "(no output)"
        return info

    def send_input(self, session_id: str, data: str, wait_ms: int = 400, workspace: Workspace | None = None) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SandboxError(f"unknown terminal session: {session_id}")
        if not session.is_running or session.process is None or session.process.stdin is None:
            raise SandboxError(f"terminal session {session_id} is not running")
        try:
            payload = data if data.endswith("\n") else data + "\n"
            session.process.stdin.write(payload)
            session.process.stdin.flush()
        except OSError as exc:
            raise SandboxError(f"could not write to terminal stdin: {exc}") from exc
        if wait_ms > 0:
            time.sleep(min(float(wait_ms) / 1000.0, 5.0))
        return self.status(session_id, workspace=workspace)

    def kill(self, session_id: str, workspace: Workspace | None = None) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SandboxError(f"unknown terminal session: {session_id}")
        proc = session.process
        if proc is not None and session.is_running:
            kill_group(proc, signal.SIGTERM)
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline and session.is_running:
                time.sleep(0.05)
            if session.is_running:
                kill_group(proc)
            session.exit_code = proc.poll()
            session.finished_at = time.monotonic()
        return session.to_dict(workspace=workspace, include_output=True)

    def list_sessions(self, conversation_id: str | None = None, workspace: Workspace | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._sessions.values())
        if conversation_id:
            items = [s for s in items if s.conversation_id in (None, conversation_id)]
        items.sort(key=lambda s: s.started_at, reverse=True)
        return [s.to_dict(workspace=workspace, include_output=False) for s in items]

    def kill_all(self, conversation_id: str | None = None) -> int:
        with self._lock:
            ids = [
                s.id
                for s in self._sessions.values()
                if s.is_running and (conversation_id is None or s.conversation_id == conversation_id)
            ]
        for sid in ids:
            try:
                self.kill(sid)
            except SandboxError:
                pass
        return len(ids)


TERMINALS = TerminalManager()
