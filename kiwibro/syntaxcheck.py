"""Does a file that was just written parse?

Kiwibro checks the files the agent writes, because a small model says "the file is ready" about a file that
does not compile. Only formats that can be parsed without running anything are checked: Python
(``compile``, which parses and builds code but never executes it), JSON, TOML, and, with the tools at hand,
shell (``bash -n``) and JavaScript (``node --check``); both only parse. A file of another type, a file too
big to read, or a checker that is not installed gives no answer, which is the same as no problem.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .sandbox import command_env, run_process

try:
    import tomllib
except ImportError:  # Python 3.10
    tomllib = None  # type: ignore[assignment]

MAX_BYTES = 1_000_000
TIMEOUT_S = 5
SHELL_SUFFIXES = {".sh": "bash", ".bash": "bash"}
NODE_SUFFIXES = {".js", ".mjs", ".cjs"}

# Wraps the argv of a parse-only subprocess in the confinement the file tools use (None: as it is).
Confine = Callable[[list[str]], list[str]]


@dataclass(frozen=True)
class Problem:
    """A file that does not parse: its name as the agent wrote it, the line (0 when unknown) and the parser's words."""

    path: str
    line: int
    message: str

    def for_tool(self) -> str:
        """The line added to the tool result, so the model can repair the file in its next round."""
        where = f", line {self.line}" if self.line else ""
        return f"Syntax error in {self.path}{where}: {self.message}. Fix the file before you say it is done."

    def for_reply(self) -> str:
        """The line Kiwibro adds under the final reply when the file is still broken."""
        where = f" on line {self.line}" if self.line else ""
        return f"Checked by Kiwibro: `{self.path}` has a syntax error{where} ({self.message})."


def check(path: Path, shown: str, *, confine: Confine | None = None) -> Problem | None:
    """The problem in the file at ``path`` (called ``shown`` in the messages), or None."""
    suffix = path.suffix.lower()
    if suffix == ".py" or suffix in (".json", ".toml"):
        parse = _python if suffix == ".py" else _json if suffix == ".json" else _toml
        if suffix == ".toml" and tomllib is None:
            return None
        data = _read(path)
        return None if data is None else parse(data, shown)
    if suffix in SHELL_SUFFIXES or suffix in NODE_SUFFIXES:
        program = shutil.which(SHELL_SUFFIXES.get(suffix, "node"))
        if program is None or not path.is_file():
            return None
        argv = [program, "-n" if suffix in SHELL_SUFFIXES else "--check", str(path)]
        return _subprocess(argv, path, shown, node=suffix in NODE_SUFFIXES, confine=confine)
    return None


def _read(path: Path) -> bytes | None:
    try:
        if not path.is_file() or path.stat().st_size > MAX_BYTES:
            return None
        return path.read_bytes()
    except OSError:
        return None


def _python(data: bytes, shown: str) -> Problem | None:
    try:
        compile(data, shown, "exec", dont_inherit=True)  # builds code, never runs it
    except SyntaxError as exc:
        return Problem(shown, int(exc.lineno or 0), exc.msg or "invalid syntax")
    except (ValueError, RecursionError, MemoryError) as exc:  # a null byte; code nested too deep to compile
        return Problem(shown, 0, str(exc) or exc.__class__.__name__)
    return None


def _json(data: bytes, shown: str) -> Problem | None:
    try:
        json.loads(data.decode("utf-8-sig"))
    except json.JSONDecodeError as exc:
        return Problem(shown, exc.lineno, exc.msg)
    except UnicodeDecodeError:
        return Problem(shown, 0, "the file is not UTF-8 text")
    return None


def _toml(data: bytes, shown: str) -> Problem | None:
    try:
        tomllib.loads(data.decode("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        found = re.search(r"\(at line (\d+), column \d+\)\s*$", str(exc))
        message = str(exc)[: found.start()].strip() if found else str(exc)
        return Problem(shown, int(found.group(1)) if found else 0, message)
    except UnicodeDecodeError:
        return Problem(shown, 0, "the file is not UTF-8 text")
    return None


def _subprocess(argv: list[str], path: Path, shown: str, *, node: bool, confine: Confine | None) -> Problem | None:
    root = path.parent
    try:
        code, output = run_process(
            confine(argv) if confine else argv, cwd=root, env=command_env(root, root), timeout=TIMEOUT_S,
        )
    except OSError:
        return None
    if code in (0, None):  # parsed, or too slow to wait for
        return None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if node:
        # "/path/a.js:3" on the first line, "SyntaxError: Unexpected token ';'" further down
        message = next((line.removeprefix("SyntaxError:").strip() for line in lines if line.startswith("SyntaxError:")), "")
        where = re.search(r":(\d+)$", lines[0]) if lines else None
    else:
        # "words.sh: line 3: syntax error near unexpected token `fi'"
        found = next((re.search(r"line (\d+): (.*)", line) for line in lines if "syntax error" in line), None)
        message = found.group(2).removeprefix("syntax error").lstrip(": ").strip() or "syntax error" if found else ""
        where = found
    if not message:  # it did not run, or failed for another reason: no verdict
        return None
    return Problem(shown, int(where.group(1)) if where else 0, message)
