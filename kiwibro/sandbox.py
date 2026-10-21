"""Workspace sandbox: path containment, file tools and safe command execution."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config

log = logging.getLogger(__name__)

MAX_READ_BYTES = config.MAX_FILE_READ_BYTES
MAX_WRITE_BYTES = config.MAX_FILE_WRITE_BYTES
MAX_OUTPUT_BYTES = config.MAX_COMMAND_OUTPUT_BYTES
COMMAND_TIMEOUT_S = 60
LIST_MAX_ENTRIES = 200
LIST_MAX_DEPTH = 2
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
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise SandboxError("path could not be resolved") from exc
        if resolved != root and not resolved.is_relative_to(root):
            raise SandboxError("path is outside the workspace")
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

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> dict[str, Any]:
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
            if first > last:
                raise SandboxError("start line is after end line")
            selected = lines[first - 1 : last]
            return {
                "path": self.public(resolved),
                "content": "\n".join(selected),
                "start": first,
                "end": last,
                "total_lines": line_count,
                "size": size,
            }
        return {
            "path": self.public(resolved),
            "content": text,
            "start": 1,
            "end": line_count,
            "total_lines": line_count,
            "size": size,
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

    def replace_in_file(self, path: str, old: str, new: str) -> dict[str, Any]:
        if not old:
            raise SandboxError("old text is empty")
        resolved = self.resolve(path, must_exist=True)
        if not resolved.is_file():
            raise SandboxError(f"not a file: {self.public(resolved)}")
        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise SandboxError(f"could not read file: {exc.strerror or exc}") from exc
        count = text.count(old)
        if count == 0:
            raise SandboxError("old text was not found in the file")
        if count > 1:
            raise SandboxError(f"old text matched {count} times; it must match exactly once")
        updated = text.replace(old, new, 1)
        if len(updated.encode("utf-8")) > MAX_WRITE_BYTES:
            raise SandboxError("resulting file would be too large")
        try:
            resolved.write_text(updated, encoding="utf-8")
        except OSError as exc:
            raise SandboxError(f"could not write file: {exc.strerror or exc}") from exc
        return {"path": self.public(resolved), "replacements": 1}

    # --- shell -------------------------------------------------------------

    def run_command(self, command: str, cwd: str | None = None, timeout: int = COMMAND_TIMEOUT_S) -> dict[str, Any]:
        check_command(command)
        workdir = self.resolve(cwd or ".", must_exist=True)
        if not workdir.is_dir():
            raise SandboxError(f"cwd is not a directory: {self.public(workdir)}")
        timeout = max(1, min(int(timeout), COMMAND_TIMEOUT_S))
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(self.root),
            "PWD": str(workdir),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TERM": "dumb",
        }
        started = time.monotonic()
        try:
            # shell=True is deliberate: agents run shell command lines (sandboxed, blocklisted).
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            duration = int((time.monotonic() - started) * 1000)
            return {
                "command": command,
                "cwd": self.public(workdir),
                "exit_code": None,
                "output": f"Command timed out after {timeout}s.",
                "truncated": False,
                "duration_ms": duration,
                "timed_out": True,
            }
        except OSError as exc:
            raise SandboxError(f"could not run command: {exc.strerror or exc}") from exc
        duration = int((time.monotonic() - started) * 1000)
        output = (completed.stdout or "") + (completed.stderr or "")
        truncated = False
        encoded = output.encode("utf-8", errors="replace")
        if len(encoded) > MAX_OUTPUT_BYTES:
            output = encoded[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
            truncated = True
        if not output.strip():
            output = "(no output)"
        return {
            "command": command,
            "cwd": self.public(workdir),
            "exit_code": completed.returncode,
            "output": output,
            "truncated": truncated,
            "duration_ms": duration,
            "timed_out": False,
        }
