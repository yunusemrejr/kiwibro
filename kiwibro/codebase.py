"""Codebase tools: glob, outlines, atomic multi-edits, file management,
command pipelines and read-only git.

Everything operates through :class:`~kiwibro.sandbox.Workspace`, so every path
stays inside the workspace sandbox. Grep itself lives on
``Workspace.grep_files`` (one implementation); this module covers the rest.
Functions return plain dicts and raise ``SandboxError`` with user-facing
messages on expected failures.
"""

from __future__ import annotations

import ast
import difflib
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from . import safety
from .sandbox import MAX_WRITE_BYTES, PRUNED_DIRS, SandboxError, Workspace, command_env, run_process

log = logging.getLogger(__name__)

MAX_GLOB_RESULTS = 200
GIT_TIMEOUT_S = 15
GIT_OUTPUT_CAP = 8000
DIFF_CAP = 3000
PIPELINE_OUTPUT_CAP = 12000
MAX_PIPELINE_COMMANDS = 8
SNIPPET_TIMEOUT_S = 30
SNIPPET_MAX_CODE_BYTES = 64 * 1024
SNIPPET_OUTPUT_CAP = 8000
SNIPPET_MEMORY_BYTES = 1024**3
SNIPPET_KEEP_RUNS = 20  # older scratch runs are pruned so long sessions stay tidy
MAX_RECURSIVE_DELETE_ENTRIES = 100


# --- glob_files ---------------------------------------------------------------


def glob_files(
    workspace: Workspace,
    pattern: str,
    *,
    path: str = ".",
    max_results: int = MAX_GLOB_RESULTS,
) -> dict[str, Any]:
    """Find workspace files matching a glob like ``**/*.py`` or ``src/*.md``."""
    pattern = (pattern or "").strip()
    if not pattern:
        raise SandboxError("glob pattern is empty")
    if pattern.startswith("/") or ".." in Path(pattern).parts:
        raise SandboxError("glob pattern must stay inside the workspace")
    max_results = max(1, min(int(max_results), MAX_GLOB_RESULTS))
    base = workspace.resolve(path or ".", must_exist=True)
    if not base.is_dir():
        raise SandboxError(f"not a directory: {workspace.public(base)}")
    root = workspace.root
    results: list[dict[str, Any]] = []
    try:
        candidates = sorted(base.glob(pattern), key=lambda item: str(item))
    except (ValueError, NotImplementedError) as exc:
        raise SandboxError(f"invalid glob pattern: {exc}") from exc
    for candidate in candidates:
        if len(results) >= max_results:
            break
        relative_parts = candidate.relative_to(base).parts if candidate != base else ()
        if any(part.startswith(".") or part in PRUNED_DIRS for part in relative_parts[:-1]):
            continue
        try:
            resolved = candidate.resolve()
            stat = candidate.stat()
        except OSError:
            continue
        if resolved != root and not resolved.is_relative_to(root):
            continue
        is_dir = candidate.is_dir()
        results.append(
            {
                "path": workspace.public(candidate),
                "type": "dir" if is_dir else "file",
                "size": 0 if is_dir else int(stat.st_size),
            }
        )
    return {"pattern": pattern, "results": results, "count": len(results)}


# --- file_outline -------------------------------------------------------------


def _python_outline(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [f"(could not parse Python: {exc.msg} at line {exc.lineno})"]
    lines: list[str] = []

    def describe(node: ast.AST, depth: int) -> None:
        indent = "  " * depth
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            lines.append(f"{indent}import · line {node.lineno}: {ast.unparse(node)[:100]}")
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(base) for base in node.bases)
            suffix = f"({bases})" if bases else ""
            lines.append(f"{indent}class {node.name}{suffix} · line {node.lineno}")
            for child in node.body:
                describe(child, depth + 1)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            args = ", ".join(arg.arg for arg in node.args.args)
            lines.append(f"{indent}{kind} {node.name}({args}) · line {node.lineno}")
        elif isinstance(node, ast.Assign) and depth == 0:
            names = ", ".join(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
            if names:
                lines.append(f"{indent}{names} = … · line {node.lineno}")

    for node in tree.body:
        describe(node, 0)
    return lines


_JS_SYMBOL = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(async\s+function|function|class|const|let|var)\s+"
    r"([A-Za-z_$][\w$]*)"
)


def _generic_outline(text: str, suffix: str) -> list[str]:
    lines: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if suffix in (".md", ".markdown", ".rst") and stripped.startswith("#"):
            lines.append(f"{stripped[:120]} · line {number}")
        elif suffix in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
            match = _JS_SYMBOL.match(line)
            if match:
                lines.append(f"{match.group(1)} {match.group(2)} · line {number}")
        if len(lines) >= 200:
            lines.append("… (outline truncated)")
            break
    return lines


def file_outline(workspace: Workspace, path: str) -> dict[str, Any]:
    """Structural outline of a file: symbols and headings with line numbers."""
    resolved = workspace.resolve(path, must_exist=True)
    if not resolved.is_file():
        raise SandboxError(f"not a file: {workspace.public(resolved)}")
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SandboxError(f"could not read file: {exc.strerror or exc}") from exc
    suffix = resolved.suffix.lower()
    entries = _python_outline(text) if suffix == ".py" else _generic_outline(text, suffix)
    return {
        "path": workspace.public(resolved),
        "total_lines": len(text.splitlines()),
        "entries": entries,
    }


# --- apply_patch (atomic multi-edit) -----------------------------------------


def apply_patch(
    workspace: Workspace,
    path: str,
    edits: list[dict[str, str]],
) -> dict[str, Any]:
    """Apply several exact replacements to one file, all-or-nothing.

    Each edit's ``old`` text must match exactly once in the text produced by
    the previous edits. Nothing is written unless every edit applies.
    """
    if not edits:
        raise SandboxError("no edits were given")
    if len(edits) > 20:
        raise SandboxError("too many edits in one patch (max 20)")
    resolved = workspace.resolve(path, must_exist=True)
    if not resolved.is_file():
        raise SandboxError(f"not a file: {workspace.public(resolved)}")
    try:
        original = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SandboxError(f"could not read file: {exc.strerror or exc}") from exc
    updated = original
    for index, edit in enumerate(edits, start=1):
        old = edit.get("old") or ""
        new = edit.get("new")
        if not old:
            raise SandboxError(f"edit {index}: old text is empty")
        if not isinstance(new, str):
            raise SandboxError(f"edit {index}: new text must be a string")
        count = updated.count(old)
        if count == 0:
            raise SandboxError(f"edit {index}: old text was not found (nothing was changed)")
        if count > 1:
            raise SandboxError(
                f"edit {index}: old text matched {count} times; it must match exactly once "
                "(nothing was changed)"
            )
        updated = updated.replace(old, new, 1)
    if len(updated.encode("utf-8")) > MAX_WRITE_BYTES:
        raise SandboxError("resulting file would be too large (nothing was changed)")
    if updated != original:
        try:
            resolved.write_text(updated, encoding="utf-8")
        except OSError as exc:
            raise SandboxError(f"could not write file: {exc.strerror or exc}") from exc
    diff = "\n".join(
        difflib.unified_diff(
            original.splitlines(),
            updated.splitlines(),
            fromfile=workspace.public(resolved),
            tofile=workspace.public(resolved),
            lineterm="",
            n=1,
        )
    )
    return {
        "path": workspace.public(resolved),
        "edits": len(edits),
        "bytes": len(updated.encode("utf-8")),
        "diff": diff[:DIFF_CAP] + ("\n… [diff truncated]" if len(diff) > DIFF_CAP else ""),
    }


# --- file_manage -----------------------------------------------------------------

FILE_MANAGE_ACTIONS = ("mkdir", "move", "copy", "delete")


def file_manage(
    workspace: Workspace,
    action: str,
    path: str,
    *,
    destination: str | None = None,
    recursive: bool = False,
) -> dict[str, Any]:
    """Create/move/copy/delete files and folders inside the workspace."""
    if action not in FILE_MANAGE_ACTIONS:
        raise SandboxError(f"unknown file action: {action}")
    root = workspace.root
    if action == "mkdir":
        target = workspace.resolve(path)
        if target == root:
            raise SandboxError("path must name a folder inside the workspace")
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SandboxError(f"could not create folder: {exc.strerror or exc}") from exc
        return {"action": action, "path": workspace.public(target)}
    source = workspace.resolve(path, must_exist=True)
    if source == root:
        raise SandboxError("the workspace root cannot be moved, copied or deleted")
    if action == "delete":
        try:
            if source.is_dir():
                if recursive:
                    entries = sum(1 for _ in source.rglob("*"))
                    if entries > MAX_RECURSIVE_DELETE_ENTRIES:
                        raise SandboxError(
                            f"refusing to recursively delete {entries} entries at once "
                            f"(limit {MAX_RECURSIVE_DELETE_ENTRIES}); delete subfolders separately"
                        )
                    shutil.rmtree(source)
                else:
                    try:
                        source.rmdir()
                    except OSError as exc:
                        raise SandboxError(
                            "folder is not empty; pass recursive=true to delete it"
                        ) from exc
            else:
                source.unlink()
        except SandboxError:
            raise
        except OSError as exc:
            raise SandboxError(f"could not delete: {exc.strerror or exc}") from exc
        return {"action": action, "path": workspace.public(source)}
    # move / copy
    if not destination:
        raise SandboxError(f"{action} needs a destination")
    target = workspace.resolve(destination)
    if target == root or target == source:
        raise SandboxError("destination must differ from the source")
    if target.exists():
        raise SandboxError(f"destination already exists: {workspace.public(target)}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if action == "move":
            shutil.move(str(source), str(target))
        elif source.is_dir():
            shutil.copytree(str(source), str(target))
        else:
            shutil.copy2(str(source), str(target))
    except OSError as exc:
        raise SandboxError(f"could not {action}: {exc.strerror or exc}") from exc
    return {
        "action": action,
        "path": workspace.public(source),
        "destination": workspace.public(target),
    }


# --- multi_command (pipeline) -------------------------------------------------------


def multi_command(
    workspace: Workspace,
    commands: list[str],
    *,
    cwd: str | None = None,
    stop_on_error: bool = True,
) -> dict[str, Any]:
    """Run commands sequentially in the workspace, combined output.

    Stops after the first failing command unless ``stop_on_error`` is false.
    Each command goes through the same blocklist/timeout rules as
    ``run_command``.
    """
    cleaned = [str(command).strip() for command in commands if str(command or "").strip()]
    if not cleaned:
        raise SandboxError("no commands were given")
    if len(cleaned) > MAX_PIPELINE_COMMANDS:
        raise SandboxError(f"too many commands in one pipeline (max {MAX_PIPELINE_COMMANDS})")
    results: list[dict[str, Any]] = []
    sections: list[str] = []
    all_ok = True
    for command in cleaned:
        result = workspace.run_command(command, cwd)
        ok = not result["timed_out"] and result["exit_code"] == 0
        all_ok = all_ok and ok
        results.append(
            {"command": command, "exit_code": result["exit_code"], "ok": ok,
             "duration_ms": result["duration_ms"]}
        )
        sections.append(f"$ {command}\nexit code: {result['exit_code']}\n{result['output']}")
        if not ok and stop_on_error:
            sections.append("(stopped after the failing command)")
            break
    combined = "\n\n".join(sections)
    if len(combined) > PIPELINE_OUTPUT_CAP:
        head = PIPELINE_OUTPUT_CAP * 3 // 5
        tail = PIPELINE_OUTPUT_CAP - head
        combined = combined[:head] + "\n… [pipeline output truncated] …\n" + combined[-tail:]
    return {"ok": all_ok, "ran": len(results), "results": results, "output": combined}


# --- sandboxed snippet execution --------------------------------------------------

SNIPPET_LANGUAGES = ("python", "javascript")


def echo_last_expression(code: str) -> str:
    """Python code that also prints its final bare expression, the way a REPL does.

    Small models often send ``1234 * 5678`` and get "(no output)" back, then guess
    the number. Only a top-level expression that ends the snippet on its own lines
    is wrapped (the rest of the text, comments included, is kept as written);
    ``None`` results such as ``print(...)`` add nothing."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code  # the interpreter reports it
    last = tree.body[-1] if tree.body else None
    if not isinstance(last, ast.Expr) or last.col_offset != 0 or last.end_lineno is None:
        return code
    lines = code.splitlines(keepends=True)
    if lines[last.end_lineno - 1][last.end_col_offset:].strip().lstrip(";").strip():
        return code  # something else shares its last line
    expression = "".join(lines[last.lineno - 1:last.end_lineno])
    head = "".join(lines[:last.lineno - 1])
    return (f"{head}__kiwibro_value = (\n{expression.rstrip().rstrip(';')}\n)\n"
            "if __kiwibro_value is not None:\n    print(repr(__kiwibro_value))\n")


def run_snippet(
    workspace: Workspace,
    language: str,
    code: str,
    *,
    timeout: int = SNIPPET_TIMEOUT_S,
) -> dict[str, Any]:
    """Run a short Python/JS snippet in an isolated workspace scratch dir.

    No shell: the interpreter runs a temp file with argv, a stripped
    environment, cwd confined to ``.kiwibro-sandbox/<run>/`` inside the
    workspace, a hard timeout and capped output. Files the snippet creates
    in its scratch dir are listed so the agent can read or keep them. This is
    the safe way to compute, transform data, or analyse media with code when
    no model capability fits. Where bubblewrap is available the snippet is
    confined (safety.confine): no network, read-only filesystem except its
    scratch dir, 1 GiB memory and 64 task caps; plots render headless (Agg).
    """
    import time as _time
    import uuid as _uuid

    if language not in SNIPPET_LANGUAGES:
        raise SandboxError(f"unsupported snippet language: {language}")
    code = code or ""
    if not code.strip():
        raise SandboxError("snippet code is empty")
    if len(code.encode("utf-8")) > SNIPPET_MAX_CODE_BYTES:
        raise SandboxError("snippet code is too large (limit 64 KiB); write a file and run it instead")
    timeout = max(1, min(int(timeout or SNIPPET_TIMEOUT_S), SNIPPET_TIMEOUT_S))
    if language == "python":
        interpreter = shutil.which("python3") or shutil.which("python")
        filename, extra = "snippet.py", ["-I"]
        missing = "python3 is not available on this machine"
    else:
        interpreter = shutil.which("node")
        filename, extra = "snippet.mjs", []
        missing = "node is not available on this machine (JavaScript snippets need Node.js)"
    if interpreter is None:
        raise SandboxError(missing)
    runs = workspace.root / ".kiwibro-sandbox"
    scratch = runs / f"run-{_uuid.uuid4().hex[:8]}"
    cache = runs / ".cache"  # shared across runs (e.g. matplotlib's font cache)
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        cache.mkdir(exist_ok=True)
        _prune_runs(runs, keep=SNIPPET_KEEP_RUNS)
        script = scratch / filename
        script.write_text(echo_last_expression(code) if language == "python" else code, encoding="utf-8")
    except OSError as exc:
        raise SandboxError(f"could not prepare snippet: {exc.strerror or exc}") from exc
    env = command_env(
        scratch, scratch, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
        XDG_CACHE_HOME=str(cache), MPLCONFIGDIR=str(cache / "matplotlib"),
    )
    argv = safety.confine(
        [interpreter, *extra, filename],
        writable=[scratch, cache],
        readable=(workspace.root,),
        cwd=scratch,
        network=False,
        memory_bytes=SNIPPET_MEMORY_BYTES,
        tasks=64,
    )
    started = _time.monotonic()
    try:
        exit_code, output = run_process(argv, cwd=scratch, env=env, timeout=timeout)
    except OSError as exc:
        raise SandboxError(f"could not run snippet: {exc.strerror or exc}") from exc
    timed_out = exit_code is None
    if timed_out:
        output += f"\nSnippet timed out after {timeout}s."
    duration = int((_time.monotonic() - started) * 1000)
    created = sorted(
        workspace.public(item)
        for item in scratch.rglob("*")
        if item.is_file() and item.name != filename
    )[:50]
    if len(output) > SNIPPET_OUTPUT_CAP:
        head = SNIPPET_OUTPUT_CAP * 3 // 5
        output = output[:head] + "\n… [output truncated] …\n" + output[-(SNIPPET_OUTPUT_CAP - head):]
    return {
        "language": language,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_ms": duration,
        "output": output.strip() and output or "(no output)",
        "scratch": workspace.public(scratch),
        "created_files": created,
    }


def _prune_runs(runs: Path, *, keep: int) -> None:
    """Delete all but the newest `keep` snippet scratch dirs (only run-* under runs/)."""
    try:
        old = sorted((d for d in runs.iterdir() if d.is_dir() and d.name.startswith("run-")), key=lambda d: d.stat().st_mtime)
    except OSError:
        return
    for directory in old[:-keep] if len(old) > keep else []:
        shutil.rmtree(directory, ignore_errors=True)


# --- read-only git --------------------------------------------------------------

_GIT_REF = re.compile(r"^[A-Za-z0-9._/@^~:-]{1,120}$")
GIT_ACTIONS = ("status", "diff", "log", "show", "branch", "compare")


def git_info(
    workspace: Workspace,
    action: str,
    *,
    path: str = ".",
    ref: str | None = None,
    limit: int = 10,
    remote: str | None = None,
) -> dict[str, Any]:
    """Read-only git inspection inside the workspace (never a shell)."""
    if action not in GIT_ACTIONS:
        raise SandboxError(f"unknown git action: {action}")
    workdir = workspace.resolve(path or ".", must_exist=True)
    if workdir.is_file():
        workdir = workdir.parent
    limit = max(1, min(int(limit), 50))
    if action == "compare":
        return _git_compare(workspace, workdir, "origin" if remote is None else remote)
    if action == "status":
        argv = ["git", "status", "--short", "--branch"]
    elif action == "diff":
        argv = ["git", "diff", "--stat"]
        if ref:
            if not _GIT_REF.match(ref):
                raise SandboxError("invalid git ref")
            argv.append(ref)
    elif action == "log":
        argv = ["git", "log", "--oneline", "--decorate", f"-n{limit}"]
    elif action == "branch":
        argv = ["git", "branch", "--verbose"]
    else:  # show
        target = ref or "HEAD"
        if not _GIT_REF.match(target):
            raise SandboxError("invalid git ref")
        argv = ["git", "show", "--stat", "--oneline", target]
    try:
        completed = subprocess.run(
            argv,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
        )
    except FileNotFoundError as exc:
        raise SandboxError("git is not installed on this machine") from exc
    except subprocess.TimeoutExpired as exc:
        raise SandboxError("git timed out") from exc
    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    if completed.returncode != 0:
        if "not a git repository" in output.lower():
            raise SandboxError("this folder is not a git repository")
        raise SandboxError(f"git {action} failed: {output[:300] or 'unknown error'}")
    return {
        "action": action,
        "cwd": workspace.public(workdir),
        "output": (output or "(no output)")[:GIT_OUTPUT_CAP],
        "truncated": len(output) > GIT_OUTPUT_CAP,
    }


# --- git compare: local copy against the remote --------------------------------------

GIT_REMOTE_TIMEOUT_S = 20
GIT_LIST_CAP = 8
_HEADS = "refs/heads/"
_SHA = re.compile(r"[0-9a-f]{40,64}")
_REMOTE_NAME = re.compile(r"[A-Za-z0-9][\w.-]{0,60}", re.ASCII)
_SHORTHAND = re.compile(r"([A-Za-z0-9_][\w.-]*)/([A-Za-z0-9_][\w.-]*)", re.ASCII)
_REMOTE_HOST = r"[A-Za-z0-9][A-Za-z0-9.-]*"
_REMOTE_PATH = r"[\w./~-]+"
_REMOTE_ADDRESS = re.compile(
    rf"https://{_REMOTE_HOST}(?::\d{{1,5}})?/{_REMOTE_PATH}"
    rf"|ssh://(?:[A-Za-z0-9_][\w.-]*@)?{_REMOTE_HOST}(?::\d{{1,5}})?/{_REMOTE_PATH}"
    rf"|[A-Za-z0-9_][\w.-]*@{_REMOTE_HOST}:{_REMOTE_PATH}",
    re.ASCII,
)


class _CompareError(Exception):
    """A compare that cannot finish; the text follows "Could not compare:"."""


def _remote_address(text: str) -> tuple[str, str | None]:
    """Check a remote name or repository address; return what git should ask and the branch a web link names."""
    if _REMOTE_NAME.fullmatch(text):
        return text, None
    if _SHORTHAND.fullmatch(text):
        return f"https://github.com/{text}", None
    if not _REMOTE_ADDRESS.fullmatch(text):
        raise SandboxError("remote must be a remote name (such as origin), owner/repo, or an https:// address")
    if not text.startswith("https://"):
        # An ssh address offers the user's keys to whatever host it names: only the checkout's
        # own remotes, by name, go over ssh.
        raise SandboxError("for an ssh address, give the remote's name instead (for example origin)")
    prefix = "https://github.com/"
    if not text.startswith(prefix):
        return text, None
    parts = text[len(prefix):].split("/")
    if len(parts) <= 2:
        return text, None
    branch = parts[3] if len(parts) > 3 and parts[2] in ("tree", "blob") else None
    return f"{prefix}{parts[0]}/{parts[1]}", branch


def _git(
    argv: list[str], cwd: Path, *, timeout: int = GIT_TIMEOUT_S, env: dict[str, str] | None = None
) -> tuple[int | None, str, str]:
    """Run git from an argument list; returns (exit code, stdout, stderr); the code is None on timeout."""
    try:
        completed = subprocess.run(
            argv, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, env=env, stdin=subprocess.DEVNULL
        )
    except FileNotFoundError as exc:
        raise SandboxError("git is not installed on this machine") from exc
    except subprocess.TimeoutExpired:
        return None, "", ""
    return completed.returncode, (completed.stdout or "").strip(), (completed.stderr or "").strip()


def _local(argv: list[str], workdir: Path) -> str:
    """Read one value from the local copy; failures become plain words."""
    code, out, err = _git(argv, workdir)
    if code is None:
        raise _CompareError("git did not finish in time.")
    if code != 0:
        if "not a git repository" in err.lower():
            raise _CompareError("this folder is not a git repository.")
        raise _CompareError("the local copy has no commit yet, or git could not read it.")
    return out


def _commits(count: int) -> str:
    return f"{count} commit" if count == 1 else f"{count} commits"


def _compare(workdir: Path, remote: str) -> dict[str, Any]:
    """Compare HEAD with the chosen remote branch; raises _CompareError when that cannot be done."""
    address, named_branch = _remote_address(remote)
    local = _local(["git", "rev-parse", "HEAD"], workdir)
    branch = _local(["git", "rev-parse", "--abbrev-ref", "HEAD"], workdir)
    changed = len(_local(["git", "--no-optional-locks", "status", "--porcelain"], workdir).splitlines())
    last = _local(["git", "log", "-1", "--format=%h %s (%cr)"], workdir)
    if _REMOTE_NAME.fullmatch(remote):
        code, _, _ = _git(["git", "config", "--get", f"remote.{remote}.url"], workdir)
        if code != 0:
            raise _CompareError(f"this checkout has no remote named {remote}.")
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "/bin/true"}
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=10")  # never wait for a passphrase
    code, out, _ = _git(
        ["git", "-c", "protocol.ext.allow=never", "ls-remote", "--symref", "--", address],
        workdir,
        timeout=GIT_REMOTE_TIMEOUT_S,
        env=env,
    )
    if code is None:
        raise _CompareError(f"the remote did not answer within {GIT_REMOTE_TIMEOUT_S} seconds (timed out).")
    if code != 0:
        raise _CompareError(f"the remote {remote} could not be reached, or it does not exist or is private.")
    default = ""
    heads: dict[str, str] = {}
    for line in out.splitlines():
        if line.startswith("ref: "):
            target, _, name = line[5:].partition("\t")
            if name == "HEAD" and target.startswith(_HEADS):
                default = target[len(_HEADS):]
            continue
        sha, _, ref = line.partition("\t")
        if ref.startswith(_HEADS) and _SHA.fullmatch(sha):
            heads[ref[len(_HEADS):]] = sha
    if not heads:
        raise _CompareError("the remote has no branches.")
    if named_branch:
        wanted = named_branch
    elif branch != "HEAD" and branch in heads:
        wanted = branch
    else:
        wanted = default
    if wanted not in heads:
        if named_branch:
            raise _CompareError(f"the remote has no branch named {wanted}.")
        raise _CompareError("the remote's default branch is unknown, so there is nothing to compare with.")
    remote_sha = heads[wanted]
    same = local == remote_sha
    lines = []
    if same:
        lines.append(f"Same commit: yes. The local copy and {remote} ({wanted}) are both at {local[:9]}.")
    else:
        label = "detached HEAD" if branch == "HEAD" else branch
        lines.append(
            f"Same commit: no. The local copy is at {local[:9]} ({label}), "
            f"{remote} ({wanted}) is at {remote_sha[:9]}."
        )
        code, _, _ = _git(["git", "cat-file", "-e", f"{remote_sha}^{{commit}}"], workdir)
        if code is None:
            raise _CompareError("git did not finish in time.")
        if code == 0:
            counts = _local(["git", "rev-list", "--left-right", "--count", f"HEAD...{remote_sha}"], workdir)
            ahead, behind = (int(count) for count in counts.split())
            phrases = []
            if ahead:
                phrases.append(f"{_commits(ahead)} ahead")
            if behind:
                phrases.append(f"{_commits(behind)} behind")
            lines.append(f"The local copy is {' and '.join(phrases)}.")
        else:
            lines.append(
                "The remote commit is not in the local copy (it has not been fetched), "
                "so the remote has work the local copy lacks or the histories differ."
            )
    lines.append(f"Local: {last}")
    uncommitted = "none" if not changed else "1 file" if changed == 1 else f"{changed} files"
    lines.append(f"Uncommitted changes in the local copy: {uncommitted}")
    names = list(heads)
    listed = ", ".join(names[:GIT_LIST_CAP])
    if len(names) > GIT_LIST_CAP:
        listed += f" (and {len(names) - GIT_LIST_CAP} more)"
    lines.append(f"Remote branches: {listed}")
    return {
        "output": "\n".join(lines),
        "same": same,
        "local": local,
        "remote_sha": remote_sha,
        "branch": branch,
        "remote_branch": wanted,
    }


def _git_compare(workspace: Workspace, workdir: Path, remote: str) -> dict[str, Any]:
    """Say whether the local copy and a remote branch are at the same commit."""
    try:
        found = _compare(workdir, remote)
    except _CompareError as exc:
        found = {
            "output": f"Could not compare: {exc}",
            "same": None,
            "local": "",
            "remote_sha": "",
            "branch": "",
            "remote_branch": "",
        }
    return {"action": "compare", "cwd": workspace.public(workdir), "truncated": False, **found}
