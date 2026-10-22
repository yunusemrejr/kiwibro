"""Execution safety: what a tool call may do, and the box commands run in.

One vocabulary for every layer: ``safe < low < medium < high < critical``.

* :func:`classify_command` reads a shell command the way bash does (quotes,
  comments, heredocs, ``$(…)``/backticks, pipelines, wrappers such as ``env``,
  ``xargs`` or ``bash -c``, scripts it runs) and grades what it would touch:
  the workspace, regenerable build output, the rest of ``$HOME``,
  credentials, start-up files, system files.
* :func:`decide` turns a validated tool call into ``auto`` / ``ask`` /
  ``deny`` from that grade, the per-tool permission, the ``autonomy`` setting
  (``supervised`` | ``smart`` | ``autonomous``), per-turn circuit breakers
  against mass edits/removals, the conversation's taint flag and
  ``guard.py``'s content verdict. Critical actions are always denied.
* :func:`confine` wraps an argv so it runs in a transient systemd scope
  (memory/task caps) inside bubblewrap: read-only root, writable workspace,
  private ``/tmp``, own PID namespace, credentials and session sockets
  hidden. Each layer is feature-detected and degrades gracefully.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import fnmatch
import contextvars
import functools
import logging
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator

from . import config

log = logging.getLogger(__name__)

LEVELS = ("safe", "low", "medium", "high", "critical")
_RANK = {level: index for index, level in enumerate(LEVELS)}
_KINDS = ("read", "web", "write", "exec")
_MODE_RANK = {"auto": 0, "ask": 1, "deny": 2}

AUTONOMY_MODES = config.AUTONOMY_MODES
DEFAULT_AUTONOMY = config.DEFAULT_AUTONOMY
SANDBOX_MODES = config.SANDBOX_MODES
DEFAULT_SANDBOX = config.DEFAULT_SANDBOX

# Per-turn circuit breakers: each time a turn's deletions/file changes pass
# another multiple of these limits, further ones are escalated one more level,
# so a looping small model cannot mass-delete or mass-edit unnoticed while
# legitimate big jobs (scaffolding a project) still finish without a human.
MASS_DELETE_LIMIT = 8
MASS_WRITE_LIMIT = config.MAX_WRITES_PER_TURN
MAX_TARGETS = 20  # operands on one delete command before it counts as "mass"
_MAX_DEPTH = 4  # nested `bash -c`, `$(…)`, scripts calling scripts
_SCRIPT_SCAN_BYTES = 64 * 1024


# --------------------------------------------------------------------------
# Values


@dataclass(frozen=True)
class Risk:
    level: str = "safe"
    reasons: tuple[str, ...] = ()
    kind: str = "read"  # read | web | write | exec — drives breakers and taint
    writes: int = 0
    deletes: int = 0
    flags: frozenset[str] = frozenset()  # "secret" (reads credentials), "net" (talks to the network)

    def __or__(self, other: Risk) -> Risk:
        first, second = (other, self) if _RANK[other.level] > _RANK[self.level] else (self, other)
        return Risk(
            level=first.level,
            reasons=first.reasons + tuple(r for r in second.reasons if r not in first.reasons),
            kind=max(self.kind, other.kind, key=_KINDS.index),
            writes=self.writes + other.writes,
            deletes=self.deletes + other.deletes,
            flags=self.flags | other.flags,
        )


SAFE = Risk()
_NET = frozenset({"net"})
_SECRET = frozenset({"secret"})


def _risk(level: str, reason: str, kind: str = "exec", *, flags: frozenset[str] = frozenset(), **counts: int) -> Risk:
    return Risk(level, (reason,), kind, flags=flags, **counts)


def _bump(level: str, steps: int = 1) -> str:
    """`steps` levels up, capped at high: escalation never manufactures `critical`."""
    return level if level == "critical" else LEVELS[min(_RANK[level] + steps, _RANK["high"])]


@dataclass(frozen=True)
class Decision:
    mode: str  # auto | ask | deny
    level: str
    reasons: tuple[str, ...] = ()
    source: str = "permission"  # permission | policy | guard

    def payload(self) -> dict[str, Any]:
        """Additive fields for the `tool_call` / `approval` SSE events."""
        return {"level": self.level, "reasons": list(self.reasons), "source": self.source}

    def denial_text(self) -> str:
        """Tool-result text for a denied call, written for the model to act on."""
        if self.source == "permission":
            return "The user has disabled this tool. Do not retry it."
        why = "; ".join(self.reasons) or f"{self.level} risk"
        return (
            f"Blocked by Kiwibro's safety policy ({self.level} risk: {why}). Do not retry this "
            "action or a variant of it. Use a safer approach that stays inside the workspace, "
            "or finish and tell the user exactly what you need them to do and why."
        )


# --------------------------------------------------------------------------
# Places: where a path points and what touching it means


_SYSTEM_PREFIXES = (
    "/bin", "/boot", "/dev", "/etc", "/lib", "/lib32", "/lib64", "/libx32", "/opt", "/proc",
    "/root", "/run", "/sbin", "/snap", "/srv", "/sys", "/usr", "/var", "/home", "/mnt",
    "/media", "/lost+found", "/cdrom",
)
# On a remote server, deploy targets (/var/www, /srv, /opt, other homes) are ordinary places.
_REMOTE_SYSTEM_PREFIXES = ("/bin", "/boot", "/dev", "/etc", "/lib", "/lib32", "/lib64",
                           "/proc", "/sbin", "/sys", "/usr")
_TEMP_PREFIXES = ("/tmp", "/var/tmp", "/dev/shm")
_HARMLESS_DEVICES = {
    "/dev/null", "/dev/zero", "/dev/stdin", "/dev/stdout", "/dev/stderr", "/dev/tty",
    "/dev/random", "/dev/urandom", "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
}
# Relative to $HOME. Reading these leaks credentials; writing them is tampering.
SECRET_PATHS = (
    ".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".config/gcloud", ".config/gh",
    ".password-store", ".local/share/keyrings", ".mozilla", ".config/google-chrome",
    ".config/chromium", ".config/BraveSoftware", ".thunderbird", ".netrc", ".git-credentials",
    ".pypirc", ".npmrc", ".claude", ".config/rclone", ".pgpass", ".cargo/credentials",
    ".cargo/credentials.toml", ".m2/settings.xml", ".s3cfg", ".vault-token", ".config/glab-cli",
    ".config/microsoft-edge", ".config/hub", ".terraform.d", ".config/doctl", ".config/op",
    ".local/share/kiwibro-secrets",
)
# Absolute (system) places whose contents are secret even when readable.
_SYSTEM_SECRETS = ("/etc/shadow", "/etc/gshadow", "/etc/sudoers", "/etc/sudoers.d", "/etc/ssh", "/root")
_SECRET_NAME = re.compile(
    r"^(?:id_(?:rsa|dsa|ecdsa|ed25519)\*?|.*\.(?:pem|key|p12|pfx|kdbx|keystore|jks|asc|gpg)|credentials.*|"
    r"\.netrc|\.pgpass|.*token.*|.*secret.*|.*passw.*|authorized_keys|known_hosts)$",
    re.I,
)
# Relative to $HOME. Writing these runs code later (log-in, new shell, git, PATH).
STARTUP_PATHS = (
    ".bashrc", ".bash_profile", ".bash_login", ".bash_logout", ".profile", ".zshrc",
    ".zprofile", ".zshenv", ".zlogin", ".xprofile", ".xsessionrc", ".pam_environment",
    ".config/autostart", ".config/systemd", ".config/environment.d", ".local/bin", "bin",
    ".gitconfig", ".config/git", ".bash_history", ".zsh_history", ".local/share/applications",
    ".bash_aliases", ".inputrc", ".config/fish", ".config/zsh", ".zsh", ".oh-my-zsh",
    ".local/lib", ".config/nvim/init.lua", ".vimrc", ".config/nvim", ".emacs", ".emacs.d",
)
REGENERABLE_DIRS = frozenset({
    "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    ".nox", ".cache", ".parcel-cache", ".next", ".nuxt", ".svelte-kit", ".turbo", "dist",
    "build", "out", "target", "coverage", "htmlcov", ".coverage", ".eggs", "venv", ".venv",
    ".gradle", "_build", ".dart_tool", "tmp", "temp",
})
_REGENERABLE_GLOB = re.compile(
    r"^\*\.(?:pyc|pyo|o|obj|a|so|class|log|tmp|temp|bak|swp|orig|rej|cache|map|d|lo)$", re.I
)

_DELETE_LEVEL = {
    "root": "critical", "system": "critical", "home_root": "critical", "secret": "critical",
    "startup": "critical", "device": "critical", "kiwibro": "critical", "home": "high",
    "outside": "high", "git": "high", "workspace_root": "high", "tmp": "low",
    "regenerable": "medium", "workspace": "medium", "dynamic": "medium", "null": "low",
}
_WRITE_LEVEL = {
    "root": "critical", "system": "critical", "device": "critical", "secret": "critical",
    "startup": "critical", "kiwibro": "critical", "home_root": "high", "home": "high",
    "outside": "high", "git": "high", "workspace_root": "medium", "tmp": "low",
    "regenerable": "low", "workspace": "medium", "dynamic": "medium", "null": "safe",
}
_PLACE = {
    "root": "the filesystem root", "system": "system files", "home_root": "your whole home folder",
    "secret": "credential files", "startup": "shell/login start-up files", "device": "a raw device",
    "kiwibro": "Kiwibro's own memory and data", "home": "files outside the workspace",
    "outside": "files outside the workspace", "git": "git internals (.git)",
    "workspace_root": "the whole workspace folder", "tmp": "temporary files",
    "regenerable": "regenerable build output", "workspace": "workspace files",
    "dynamic": "paths only known at run time", "null": "/dev/null",
}
_IN_WORKSPACE = ("workspace", "workspace_root", "regenerable")


@dataclass
class _Ctx:
    workspace: Path | None
    cwd: Path | None
    home: Path
    remote: bool = False
    depth: int = 0
    stdin: str | None = None  # None | "pipe" | "text"
    stdin_text: str = ""
    # Shared by every command of one command line: what earlier commands set up.
    vars: dict[str, str] = field(default_factory=dict)  # D=/ ; for d in ~
    links: dict[str, str] = field(default_factory=dict)  # ln -s TARGET LINK
    files: dict[str, str] = field(default_factory=dict)  # echo '…' > s.sh (static content)

    def child(self, **changes: Any) -> _Ctx:
        values = {**self.__dict__, "stdin": None, "stdin_text": ""}
        values.update(changes)
        return _Ctx(**values)

    def nested(self) -> _Ctx:
        return self.child(depth=self.depth + 1)


@dataclass(frozen=True)
class _Scope:
    kind: str
    path: str
    wildcard: bool = False


_SUBST = "__KB_SUBST__"  # stands for the output of $(…) / backticks / xargs input
_PROCSUB = "__KB_PROCSUB__"  # stands for <(…) / >(…)
_VAR = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)(?::?[-=]([^}]*)|[^}]*)\}|([A-Za-z_][A-Za-z0-9_]*)|[0-9@*#?!$-])")
_REMOTE_HOME = Path("/~")


@functools.lru_cache(maxsize=8)
def _data_dir(_env_key: str) -> str:
    return str(config.data_dir().resolve())


def _scope(raw: str, ctx: _Ctx) -> _Scope:
    text = raw.strip()
    if not text or _SUBST in text or _PROCSUB in text or "`" in text:
        return _Scope("dynamic", text)

    def expand(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(3) or ""
        if name not in ctx.vars and name not in ("HOME", "PWD") and match.group(2) is not None:
            return match.group(2)  # ${UNSET:-default}
        if name == "HOME":
            return str(ctx.home)
        if name == "PWD" and ctx.cwd is not None:
            return str(ctx.cwd)
        if name in ctx.vars and not any(m in ctx.vars[name] for m in (_SUBST, _PROCSUB)):
            return ctx.vars[name]
        raise LookupError(name)

    try:
        text = _VAR.sub(expand, text)
    except LookupError:
        return _Scope("dynamic", raw)
    if text == "~" or text.startswith("~/"):
        text = str(ctx.home) + text[1:]
    elif text.startswith("~"):
        if ctx.remote:
            return _Scope("outside", text)
        text = os.path.expanduser(text)  # ~user
        if text.startswith("~"):
            return _Scope("system", text)
    if not text.startswith("/"):
        if ctx.cwd is None:
            return _Scope("dynamic", raw, any(ch in text for ch in "*?["))
        text = f"{ctx.cwd}/{text}"
    globs = [text.find(ch) for ch in "*?[" if ch in text]
    if globs:
        cut = text.rfind("/", 0, min(globs))
        base, rest = text[:cut] or "/", text[cut + 1 :]
        resolved = os.path.normpath(base) if ctx.remote else os.path.realpath(base)
        path = os.path.join(_follow_links(resolved, ctx), rest)
    else:
        path = _follow_links(os.path.normpath(text) if ctx.remote else os.path.realpath(text), ctx)
    kind = _classify_path(path, ctx)
    if globs:
        kind = max((kind, *_glob_hits(path, ctx)), key=_HARM.index)
    return _Scope(kind, path, bool(globs))


# Kinds from least to most harmful; equal delete levels are ordered by how much the
# kind means (a glob that can reach ~/.ssh is "secret", not merely "home").
_HARM = ("null", "tmp", "regenerable", "workspace", "dynamic", "git", "workspace_root", "home",
         "outside", "device", "system", "kiwibro", "home_root", "root", "startup", "secret")


def _follow_links(path: str, ctx: _Ctx) -> str:
    """Resolve through symlinks created earlier on the same command line."""
    for link, target in sorted(ctx.links.items(), key=lambda item: -len(item[0])):
        if path == link or path.startswith(link + "/"):
            return target + path[len(link) :]
    return path


def _glob_hits(pattern: str, ctx: _Ctx) -> list[str]:
    """Kinds of the dangerous places a glob could reach (bash rules: `*` skips dotfiles)."""
    home = str(ctx.home)
    candidates = [p for p in _SYSTEM_PREFIXES] + [home]
    candidates += [f"{home}/{rel}" for rel in (*SECRET_PATHS, *STARTUP_PATHS)]
    if not ctx.remote:
        candidates.append(_data_dir(os.environ.get("KIWIBRO_DATA_DIR", "")))
    parts = pattern.strip("/").split("/")
    hits = []
    for candidate in candidates:
        cparts = candidate.strip("/").split("/")
        if all(
            fnmatch.fnmatchcase(c, p) and not (c.startswith(".") and not p.startswith("."))
            for c, p in zip(cparts, parts)
        ):
            rest = ["x" if any(ch in p for ch in "*?[") else p for p in parts[len(cparts) :]]
            hits.append(_classify_path("/" + "/".join(cparts[: len(parts)] + rest), ctx))
    return hits


_OTHER_HOME_SECRET = re.compile(r"^/home/[^/]+/(?:" + "|".join(re.escape(p) for p in SECRET_PATHS) + r")(?:/|$)")


def _under(path: str, prefixes: tuple[str, ...] | list[str]) -> bool:
    return any(path == p or path.startswith(p.rstrip("/") + "/") for p in prefixes)


def _classify_path(path: str, ctx: _Ctx) -> str:
    if path in ("/", "/*", "/."):
        return "root"
    if path in _HARMLESS_DEVICES or path.startswith("/proc/self/fd/"):
        return "null"
    if path.startswith("/dev/"):
        return "device"
    if not ctx.remote and _under(path, [_data_dir(os.environ.get("KIWIBRO_DATA_DIR", ""))]):
        return "kiwibro"
    if not ctx.remote and (_under(path, _SYSTEM_SECRETS) or _OTHER_HOME_SECRET.match(path)):
        return "secret"
    workspace = str(ctx.workspace) if ctx.workspace else None
    if workspace and _under(path, [workspace]):
        rel = path[len(workspace) + 1 :]
        if rel in ("", "*", "."):
            return "workspace_root"
        parts = rel.split("/")
        if parts[0] == ".git":
            return "git"
        return "regenerable" if any(part in REGENERABLE_DIRS for part in parts) else "workspace"
    base = path[:-2] if path.endswith("/*") else path
    if workspace and workspace.startswith(base.rstrip("/") + "/") and base != str(ctx.home):
        return "outside"  # an ancestor: touching it touches the workspace and its neighbours
    home = str(ctx.home)
    if path in (home, home + "/*"):
        return "home_root"
    if path.startswith(home + "/"):
        rel = path[len(home) + 1 :]
        if _under(rel, SECRET_PATHS):
            return "secret"
        if _under(rel, STARTUP_PATHS):
            return "startup"
        return "home"
    if _under(path, _TEMP_PREFIXES):
        return "tmp"
    if _under(path, _REMOTE_SYSTEM_PREFIXES if ctx.remote else _SYSTEM_PREFIXES):
        return "system"
    return "outside"


def place_of(path: str, *, workspace: Any = None) -> str:
    """What kind of place an absolute path is (workspace, home, secret, system, …)."""
    root = _workspace_root(workspace)
    return _scope(path, _Ctx(workspace=root, cwd=root, home=Path.home().resolve())).kind


def describe_place(kind: str) -> str:
    return _PLACE.get(kind, kind)


def _mentioned(path: str, message: str) -> bool:
    """Did the user's message name this folder (full path, ~/ form or its own name)?"""
    text = message.lower()
    home = str(Path.home())
    forms = {path, path.replace(home, "~", 1), os.path.basename(path.rstrip("/"))}
    return any(len(form) >= 3 and form.lower() in text for form in forms)


def _delete_risk(targets: list[str], ctx: _Ctx, *, recursive: bool, verb: str) -> Risk:
    risk = Risk(kind="write", deletes=len(targets))
    if len(targets) > MAX_TARGETS:
        risk |= _risk("high", f"{verb} removes {len(targets)} paths at once", "write")
    for target in targets:
        scope = _scope(target, ctx)
        level = _DELETE_LEVEL[scope.kind]
        if recursive and scope.kind in ("workspace", "dynamic"):
            level = "high"
        if ctx.remote and scope.kind in ("home", "outside", "dynamic"):
            level = "high" if recursive else "medium"
        if scope.wildcard and scope.kind == "workspace" and not _REGENERABLE_GLOB.match(scope.path.rsplit("/", 1)[-1]):
            level = "high"
        what = "recursively deletes" if recursive else "deletes"
        risk |= _risk(level, f"{verb} {what} {_PLACE[scope.kind]} ({target})", "write")
    return risk


def _write_risk(targets: list[str], ctx: _Ctx, *, verb: str, create: bool = False) -> Risk:
    risk = Risk(kind="write")
    for target in targets:
        if not target or target.isdigit() or target == "-":
            continue
        scope = _scope(target, ctx)
        level = _WRITE_LEVEL[scope.kind]
        if create and scope.kind in _IN_WORKSPACE:
            level = "low"
        if ctx.remote and scope.kind in ("dynamic", "outside", "home"):
            level = "medium"
        risk |= _risk(level, f"{verb} {_PLACE[scope.kind]} ({target})", "write", writes=1)
    return risk


def _read_risk(targets: list[str], ctx: _Ctx, *, verb: str) -> Risk:
    risk = SAFE
    if ctx.remote:
        return risk
    for target in targets:
        kind = _scope(target, ctx).kind
        if kind == "secret":
            risk |= _risk("high", f"{verb} reads credential files ({target})", "read", flags=_SECRET)
        elif kind == "device":
            risk |= _risk("high", f"{verb} reads a raw device ({target})", "read")
    return risk


# --------------------------------------------------------------------------
# Shell text → pipelines of simple commands


class _ParseError(ValueError):
    pass


def _match_close(text: str, start: int) -> int:
    """Index of the `)` closing the `(` at `start`, honouring quotes and nesting."""
    depth, i, quote = 0, start, None
    while i < len(text):
        ch = text[i]
        if quote == "'":
            quote = None if ch == "'" else quote
        elif ch == "\\":
            i += 1
        elif quote == '"':
            quote = None if ch == '"' else quote
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise _ParseError("unbalanced parentheses")


_HEREDOC = re.compile(r"<<(-?)[ \t]*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
_OPERATOR_CHARS = ";&|<>()"


def _preprocess(text: str) -> tuple[str, list[str], list[str]]:
    """Strip comments, arithmetic and heredoc bodies; pull out substitutions.

    Returns the text with every `$(…)`, backtick and process substitution
    replaced by a placeholder word, plus the substitution bodies and heredoc
    bodies (heredoc `k` is referenced in the text as `<< __KB_HEREDOC_k__`)."""
    text = text.replace("\\\n", "")
    out: list[str] = []
    subs: list[str] = []
    heredocs: list[str] = []
    pending: list[tuple[str, bool]] = []  # (delimiter, strip_tabs) awaiting their bodies
    i, n, quote = 0, len(text), None
    while i < n:
        ch = text[i]
        if quote == "'":
            out.append(ch)
            quote = None if ch == "'" else quote
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            out.append(text[i : i + 2])
            i += 2
            continue
        if text.startswith("$((", i) or (quote is None and text.startswith("((", i)):
            start = i + 1 if text[i] == "$" else i
            close = _match_close(text, start)
            subs.extend(_preprocess(text[start + 2 : close - 1])[1])
            out.append("0" if text[i] == "$" else "true")
            i = close + 1
            continue
        if text.startswith("$(", i) or (quote is None and text.startswith(("<(", ">("), i)):
            close = _match_close(text, i + 1)
            subs.append(text[i + 2 : close])
            out.append(f"{_SUBST}{len(subs) - 1}_" if text[i] == "$" else _PROCSUB)
            i = close + 1
            continue
        if ch == "`":
            close = text.find("`", i + 1)
            if close < 0:
                raise _ParseError("unbalanced backtick")
            subs.append(text[i + 1 : close])
            out.append(f"{_SUBST}{len(subs) - 1}_")
            i = close + 1
            continue
        if quote == '"':
            out.append(ch)
            quote = None if ch == '"' else quote
            i += 1
            continue
        if text.startswith("$'", i):  # ANSI-C quoting: decode it the way bash would
            close = text.find("'", i + 2)
            while close > 0 and text[close - 1] == "\\":
                close = text.find("'", close + 1)
            if close < 0:
                raise _ParseError("unbalanced $'")
            out.append(shlex.quote(_unescape(text[i + 2 : close])))
            i = close + 1
            continue
        if ch in "'\"":
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "#" and (not out or out[-1][-1:] in " \t\n" + _OPERATOR_CHARS):
            while i < n and text[i] != "\n":
                i += 1
            continue
        if text.startswith("<<", i) and not text.startswith("<<<", i):
            match = _HEREDOC.match(text, i)
            if match:
                out.append(f" << __KB_HEREDOC_{len(heredocs) + len(pending)}__ ")
                pending.append((match.group(3), match.group(1) == "-"))
                i = match.end()
                continue
        if ch in "<>":
            # `2>file` / `2>&1`: drop an fd number glued to the operator.
            j = len(out)
            while j > 0 and out[j - 1].isdigit():
                j -= 1
            if j < len(out) and (j == 0 or out[j - 1][-1:] in " \t\n" + _OPERATOR_CHARS):
                del out[j:]
        if ch == "\n":
            out.append(" ; ")
            i += 1
            for delimiter, strip_tabs in pending:
                body: list[str] = []
                while i < n:
                    end = text.find("\n", i)
                    line = text[i : end if end >= 0 else n]
                    i = end + 1 if end >= 0 else n
                    if (line.lstrip("\t") if strip_tabs else line) == delimiter:
                        break
                    body.append(line)
                heredocs.append("\n".join(body))
            pending.clear()
            continue
        out.append(ch)
        i += 1
    if quote:
        raise _ParseError("unbalanced quotes")
    heredocs.extend("" for _ in pending)
    return "".join(out), subs, heredocs


_REDIRECTS = {">", ">>", ">|", "&>", "&>>", ">&", "<", "<<", "<<<", "<>", "<&"}
_KEYWORDS = {"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "!", "{", "}",
             "time", "esac", "in", "coproc"}


@dataclass
class _Cmd:
    words: list[str] = field(default_factory=list)
    redirects: list[tuple[str, str]] = field(default_factory=list)


def _pipelines(text: str) -> list[list[_Cmd]]:
    lexer = shlex.shlex(text, posix=True, punctuation_chars=_OPERATOR_CHARS)
    lexer.whitespace_split = True
    lexer.commenters = ""
    tokens = list(lexer)
    pipelines: list[list[_Cmd]] = []
    pipeline: list[_Cmd] = []
    cmd = _Cmd()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token and all(ch in _OPERATOR_CHARS for ch in token):
            if token in _REDIRECTS:
                cmd.redirects.append((token, tokens[i + 1] if i + 1 < len(tokens) else ""))
                i += 2
                continue
            if cmd.words or cmd.redirects:
                pipeline.append(cmd)
            cmd = _Cmd()
            if token not in ("|", "|&"):
                if pipeline:
                    pipelines.append(pipeline)
                pipeline = []
            i += 1
            continue
        cmd.words.append(token)
        i += 1
    if cmd.words or cmd.redirects:
        pipeline.append(cmd)
    if pipeline:
        pipelines.append(pipeline)
    return pipelines


# --------------------------------------------------------------------------
# Program knowledge


_FORK_BOMB = re.compile(r"(\w+|:)\s*\(\s*\)\s*\{[^}]*\1\s*\|\s*\1\s*&")
_ANSI_C_ESCAPE = re.compile(r"\$'[^']*\\(?:x[0-9a-fA-F]|[0-7]{2,3}|u[0-9a-fA-F]|c.)")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\+?=")
DANGEROUS_ENV = frozenset({
    "LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH", "BASH_ENV", "ENV", "PROMPT_COMMAND",
    "GIT_SSH_COMMAND", "GIT_SSH", "GIT_EXEC_PATH", "GIT_EXTERNAL_DIFF", "GIT_PROXY_COMMAND",
    "GIT_ASKPASS", "SSH_ASKPASS", "GIT_EDITOR", "GIT_SEQUENCE_EDITOR", "GIT_PAGER", "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM", "GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS", "PAGER", "MANPAGER", "EDITOR",
    "VISUAL", "LESSOPEN", "LESSCLOSE", "PYTHONSTARTUP", "NODE_OPTIONS", "PERL5OPT", "RUBYOPT",
})
_SHELLS = frozenset({"sh", "bash", "dash", "zsh", "ksh", "mksh", "ash", "fish", "tcsh", "csh", "rbash"})
# interpreter: (exact inline-code flags, short-flag letter that ends a cluster with code)
_INTERPRETERS: dict[str, tuple[tuple[str, ...], str]] = {
    "python": (("-c",), "c"), "node": (("-e", "--eval", "-p", "--print"), ""),
    "perl": (("-e", "-E"), "eE"), "ruby": (("-e",), "e"), "php": (("-r",), ""),
    "lua": (("-e",), ""), "bun": (("-e", "--eval"), ""), "deno": ((), ""),
    "rscript": (("-e",), ""), "R": (("-e",), ""), "julia": (("-e", "--eval"), ""), "pwsh": (("-c", "-command"), ""),
}
_PRIVILEGE = frozenset({"sudo", "su", "doas", "pkexec", "run0", "sudoedit", "chroot", "sg", "newgrp"})
_CATASTROPHIC = frozenset({
    "mkfs", "mke2fs", "mkswap", "fdisk", "sfdisk", "gdisk", "cfdisk", "parted", "wipefs",
    "shutdown", "reboot", "poweroff", "halt", "telinit", "insmod", "rmmod", "modprobe", "mount",
    "umount", "losetup", "cryptsetup", "lvremove", "vgremove", "pvremove", "mdadm", "dmsetup",
    "hdparm", "swapon", "swapoff", "iptables", "ip6tables", "nft", "ufw", "useradd", "userdel",
    "usermod", "groupadd", "groupdel", "passwd", "chpasswd", "visudo", "chsh", "chfn", "setcap",
    "sysctl", "kexec", "killall5", "efibootmgr", "grub-install", "update-grub", "init",
})
_REMOTE_TOOLS = frozenset({"ssh", "scp", "sftp", "sshfs", "mosh", "sshpass", "autossh", "rsh", "rlogin"})
_SYS_PACKAGERS = frozenset({"apt", "apt-get", "aptitude", "dpkg", "snap", "flatpak", "dnf", "yum",
                            "rpm", "pacman", "zypper", "brew", "port", "nix-env", "apk", "emerge"})
_PACKAGE_QUERIES = frozenset({"list", "search", "show", "info", "policy", "depends", "rdepends",
                              "changelog", "query", "files", "owns", "find", "-l", "-L", "-s",
                              "-S", "-Q", "-Ss", "-Qi", "--list"})
_INFRA = frozenset({
    "kubectl", "helm", "terraform", "tofu", "ansible", "ansible-playbook", "pulumi", "vagrant",
    "aws", "gcloud", "az", "doctl", "flyctl", "fly", "heroku", "vercel", "netlify", "firebase",
    "wrangler", "railway", "gh", "glab", "hub", "docker", "podman", "nerdctl", "docker-compose",
    "lxc", "virsh", "multipass",
})
_INFRA_READ_VERBS = frozenset({
    "get", "describe", "logs", "log", "list", "ls", "show", "status", "version", "--version",
    "help", "--help", "whoami", "plan", "validate", "view", "ps", "images", "inspect", "info",
    "top", "stats", "diff", "history", "search", "explain", "api-resources", "cluster-info",
    "port", "fmt", "output", "graph", "providers", "checks", "browse",
})
_INFRA_WRITE_VERBS = frozenset({
    "delete", "destroy", "rm", "rmi", "prune", "kill", "apply", "deploy", "push", "create", "run",
    "exec", "up", "down", "set", "remove", "login", "start", "stop", "restart", "scale", "rollout",
})
_DATABASES = frozenset({"psql", "mysql", "mariadb", "mongo", "mongosh", "redis-cli", "cqlsh", "sqlite3"})
_DB_WRITES = re.compile(r"\b(?:DROP|TRUNCATE|DELETE|ALTER|UPDATE|FLUSHALL|FLUSHDB|dropDatabase)\b", re.I)
_PUBLISH_VERBS = frozenset({"publish", "unpublish", "deprecate", "owner", "access", "token", "login",
                            "logout", "adduser", "dist-tag", "team", "org", "yank", "upload"})
_READ_ONLY = frozenset({
    "ls", "ll", "la", "dir", "vdir", "cat", "tac", "head", "tail", "less", "more", "wc", "grep",
    "egrep", "fgrep", "rg", "ag", "ack", "fd", "fdfind", "pwd", "echo", "printf", "true", "false",
    "test", "[", "[[", "which", "whereis", "type", "file", "stat", "du", "df", "tree", "realpath",
    "readlink", "basename", "dirname", "date", "cal", "uptime", "whoami", "id", "groups",
    "hostname", "uname", "lsb_release", "nproc", "free", "ps", "pgrep", "pidof", "jq", "yq",
    "sort", "uniq", "cut", "tr", "paste", "join", "comm", "column", "fmt", "fold", "nl", "od",
    "hexdump", "strings", "diff", "cmp", "md5sum", "sha1sum", "sha256sum", "sha512sum", "b2sum",
    "cksum", "seq", "sleep", "wait", "man", "help", "info", "tldr", "locale", "getent", "lscpu",
    "lsblk", "lsusb", "lspci", "lsof", "ss", "netstat", "ip", "tput", "clear", "printenv",
    "history", "exit", "return", "read", "set", "shopt", "unset", "unalias", "umask",
    "cd", "pushd", "popd",
    "ulimit", "hash", "jobs", "fg", "bg", "disown", "dirs", "shift", "getopts", "yes",
    "nvidia-smi", "sensors", "journalctl", "dmesg", "top", "htop", "vmstat", "iostat", "look",
    "numfmt", "expr", "bc", "dc", "factor", "pdftotext", "pdfinfo", "identify", "exiftool",
    "mediainfo", "ffprobe", "notify-send", "base64", "base32", "xxd", "rev", "zcat", "gunzip",
    "ping", "dig", "nslookup", "host", "whois", "traceroute", "tracepath",
})
_DEV_TOOLS = frozenset({
    "make", "cmake", "ninja", "meson", "scons", "bazel", "gradle", "gradlew", "mvn", "ant",
    "dotnet", "msbuild", "sbt", "mix", "rebar3", "stack", "cabal", "dune", "zig", "swift", "gcc",
    "g++", "cc", "c++", "clang", "clang++", "rustc", "javac", "java", "kotlinc", "tsc", "esbuild",
    "swc", "vite", "webpack", "rollup", "parcel", "pytest", "py.test", "tox", "nox", "jest",
    "vitest", "mocha", "ava", "phpunit", "rspec", "ctest", "black", "isort", "autopep8", "yapf",
    "prettier", "ruff", "flake8", "mypy", "pylint", "pyright", "eslint", "shellcheck", "gofmt",
    "rustfmt", "clang-format", "shfmt", "cargo", "go", "rake", "bundle", "composer", "gem",
    "poetry", "pipenv", "pdm", "hatch", "uv", "uvx", "pipx", "conda", "mamba", "micromamba",
    "pip", "npm", "pnpm", "yarn", "npx", "pnpx", "bunx", "corepack", "virtualenv", "twine",
    "ffmpeg", "convert", "magick", "mogrify", "pandoc", "inkscape", "gs", "qpdf", "libreoffice",
    "soffice", "patch", "touch", "mkdir", "zip", "gzip", "bzip2", "xz", "zstd", "7z", "7za",
    "tar", "unzip", "split", "csplit", "trash", "trash-put", "gio", "xdg-open", "open",
    "gnome-open", "kde-open", "sensible-browser", "x-www-browser", "firefox", "google-chrome",
    "chromium", "chromium-browser", "code", "jupyter", "ipython", "streamlit", "flask", "uvicorn",
    "gunicorn", "hugo", "jekyll", "mkdocs", "sphinx-build", "latexmk", "pdflatex", "xelatex",
    "lualatex", "typst", "dot", "plantuml", "mmdc", "playwright", "wasm-pack", "emcc",
    "pkg-config", "ldd", "install",
})
_CHECKERS = frozenset({"ruff", "flake8", "mypy", "pylint", "pyright", "shellcheck", "eslint", "tsc"})
_OPENERS = frozenset({"xdg-open", "open", "gnome-open", "kde-open", "sensible-browser",
                      "x-www-browser", "firefox", "google-chrome", "chromium", "chromium-browser"})


def _prog(word: str) -> str:
    name = re.sub(r"^(python|pypy|pip|perl|ruby|php|lua|node)[0-9.]*$", r"\1", os.path.basename(word))
    return {"nodejs": "node", "pypy": "python", "Rscript": "rscript", "pip3": "pip"}.get(name, name)


def _split_flags(argv: list[str]) -> tuple[set[str], list[str]]:
    """Flags (short clusters expanded) and operands of a GNU-style argv."""
    flags: set[str] = set()
    operands: list[str] = []
    ended = False
    for token in argv[1:]:
        if ended or not token.startswith("-") or token == "-":
            operands.append(token)
        elif token == "--":
            ended = True
        elif token.startswith("--"):
            flags.add(token.split("=", 1)[0])
        else:
            flags.update("-" + ch for ch in token[1:])
    return flags, operands


def _option_values(argv: list[str], *names: str) -> list[str]:
    """Values of options given as `-o v`, `-ov`, `--opt v` or `--opt=v`."""
    values: list[str] = []
    for index, token in enumerate(argv):
        for name in names:
            if token == name and index + 1 < len(argv):
                values.append(argv[index + 1])
            elif name.startswith("--") and token.startswith(name + "="):
                values.append(token.split("=", 1)[1])
            elif not name.startswith("--") and token.startswith(name) and len(token) > len(name):
                values.append(token[len(name) :])
    return values


def _verbs(argv: list[str]) -> list[str]:
    return [a for a in argv[1:] if not a.startswith("-")]


# name: (options that take a value, positional operands before the wrapped command)
_WRAPPERS: dict[str, tuple[frozenset[str], int]] = {
    "nohup": (frozenset(), 0),
    "time": (frozenset({"-o", "-f", "--output", "--format"}), 0),
    "nice": (frozenset({"-n", "--adjustment"}), 0),
    "ionice": (frozenset({"-c", "-n", "-p", "-P", "-u", "--class", "--classdata"}), 0),
    "timeout": (frozenset({"-s", "--signal", "-k", "--kill-after"}), 1),
    "stdbuf": (frozenset({"-i", "-o", "-e", "--input", "--output", "--error"}), 0),
    "setsid": (frozenset(), 0),
    "builtin": (frozenset(), 0),
    "exec": (frozenset({"-a"}), 0),
    "chrt": (frozenset(), 1),
    "taskset": (frozenset(), 1),
    "unbuffer": (frozenset(), 0),
    "busybox": (frozenset(), 0),
    "strace": (frozenset({"-o", "-e", "-p", "-s", "-u", "-E"}), 0),
    "ltrace": (frozenset({"-o", "-e", "-p", "-s", "-u"}), 0),
    "xargs": (frozenset({"-I", "-n", "-L", "-P", "-d", "-s", "-a", "-E", "--max-args",
                         "--max-lines", "--max-procs", "--delimiter", "--arg-file",
                         "--max-chars", "--replace", "--eof"}), 0),
    "flock": (frozenset({"-w", "--timeout", "-E", "--conflict-exit-code"}), 1),
    "env": (frozenset({"-u", "--unset", "-C", "--chdir"}), 0),
    "parallel": (frozenset({"-j", "--jobs", "-S", "--sshlogin", "--delay", "--timeout", "-a", "--arg-file"}), 0),
    "numactl": (frozenset({"-C", "-m", "-N", "-p", "-i", "--physcpubind", "--membind", "--cpunodebind",
                           "--preferred", "--interleave"}), 0),
    "proxychains": (frozenset({"-f"}), 0),
    "proxychains4": (frozenset({"-f"}), 0),
    "fakeroot": (frozenset({"-i", "-s", "-l", "--lib", "--faked"}), 0),
    "systemd-cat": (frozenset({"-t", "-p", "--identifier", "--priority", "--stderr-priority"}), 0),
    "chronic": (frozenset(), 0),
    "torsocks": (frozenset(), 0),
    "firejail": (frozenset(), 0),
}


def _skip_options(argv: list[str], valued: frozenset[str], positionals: int) -> list[str]:
    i = 1
    while i < len(argv) and argv[i].startswith("-") and argv[i] != "-":
        if argv[i] == "--":
            i += 1
            break
        i += 2 if argv[i] in valued else 1
    return argv[i + positionals :]


# --------------------------------------------------------------------------
# Command analysis


def _command_risk(cmd: _Cmd, ctx: _Ctx, heredocs: list[str]) -> Risk:
    risk = SAFE
    stdin_text: str | None = None
    for op, target in cmd.redirects:
        if op in (">", ">>", ">|", "&>", "&>>", "<>") or (op == ">&" and not target.isdigit() and target != "-"):
            risk |= _write_risk([target], ctx, verb="redirects output into")
        elif op == "<":
            risk |= _read_risk([target], ctx, verb="input redirection")
            stdin_text = _script_text(target, ctx)
        elif op == "<<":
            match = re.fullmatch(r"__KB_HEREDOC_(\d+)__", target)
            if match and int(match.group(1)) < len(heredocs):
                stdin_text = heredocs[int(match.group(1))]
        elif op == "<<<":
            stdin_text = target
    if stdin_text is not None:
        ctx = ctx.child(stdin="text", stdin_text=stdin_text)
    words = list(cmd.words)
    while words and words[0] in _KEYWORDS:
        words.pop(0)
    while words and words[-1] in ("}", "fi", "done", "esac"):
        words.pop()
    if words[:1] == ["function"]:
        words = [w for w in words[2:] if w != "{"]  # `function f { body` → body
    if words[:1] == ["for"] and len(words) >= 3 and words[2] == "in":
        ctx.vars[words[1]] = " ".join(words[3:])
        return risk
    if not words or words[0] in ("for", "case", "select"):
        return risk
    content = _static_output(words, ctx.stdin_text if ctx.stdin == "text" else None)
    if content is not None:  # remember scripts written by this line (write-then-run)
        for op, target in cmd.redirects:
            if op in (">", ">>", ">|") and _scope(target, ctx).kind != "dynamic":
                ctx.files[_scope(target, ctx).path] = content
    return risk | _argv_risk(words, ctx)


def _argv_risk(argv: list[str], ctx: _Ctx) -> Risk:
    risk = SAFE
    while argv and _ASSIGNMENT.match(argv[0]):
        risk |= _assignment_risk(argv[0], ctx)
        argv = argv[1:]
    if not argv:
        return risk
    argv = _expand_words(argv, ctx)
    if not argv:
        return risk
    word = argv[0]
    base = os.path.basename(word)
    if word not in ("[", "[[") and any(ch in word for ch in "*?[") and base and not any(ch in base for ch in "*?[$`"):
        risk |= _risk("high", f"the program path `{word}` is a glob")
        word = argv[0] = base  # /???/rm is rm
    if any(marker in word for marker in ("$", "`", _SUBST, _PROCSUB)) or (
        word not in ("[", "[[") and any(ch in word for ch in "*?[")
    ):
        return risk | _risk("high", f"the program to run (`{word}`) is only known at run time")
    name = _prog(word)
    handler = _HANDLERS.get(name)
    if handler is not None:
        return risk | handler(argv, ctx)
    if name in _PRIVILEGE:
        return risk | _risk("high" if ctx.remote else "critical", f"`{name}` escalates privileges")
    if name in _CATASTROPHIC or name.startswith("mkfs."):
        return risk | _risk("critical", f"`{name}` can damage the system or end the session")
    if name in _WRAPPERS:
        return risk | _wrapper_risk(name, argv, ctx)
    if name in _SHELLS:
        return risk | _shell_risk(argv, ctx)
    if name in _INTERPRETERS:
        return risk | _interpreter_risk(argv, ctx, name)
    if name in _REMOTE_TOOLS:
        return risk | _risk("high", f"`{name}` connects to another machine (use the `remote` tool)", flags=_NET)
    if name in _SYS_PACKAGERS:
        if set(argv[1:]) & _PACKAGE_QUERIES:
            return risk | _risk("low", f"queries installed packages with `{name}`", "read")
        return risk | _risk("high", f"`{name}` changes system-wide software")
    if name in _DATABASES:
        if _DB_WRITES.search(" ".join(argv[1:]) + ctx.stdin_text):
            return risk | _risk("high", f"`{name}` changes or deletes database data", "write", writes=1)
        return risk | _risk("medium", f"`{name}` talks to a database")
    if name in _INFRA:
        return risk | _infra_risk(name, argv)
    if name in _READ_ONLY:
        return risk | _read_only_risk(name, argv, ctx)
    if name in _DEV_TOOLS:
        return risk | _dev_tool_risk(name, argv, ctx) | _operand_risk(name, argv, ctx)
    if "/" in word:
        return risk | _program_file_risk(word, ctx) | _operand_risk(name, argv, ctx)
    return risk | _risk("medium", f"runs `{name}`") | _operand_risk(name, argv, ctx)


_SAFE_HOOK_VALUES = frozenset({"", "cat", "less", "more", "head", "tail", "true", "false", "less -R", "cat -v"})
_BRACE = re.compile(r"\{([^{}]*,[^{}]*)\}")
_WORD_VAR = re.compile(r"^\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))$")


def _assignment_risk(word: str, ctx: _Ctx) -> Risk:
    """`NAME=value`: remember it for later words, flag code-injection variables."""
    name, _, value = word.partition("=")
    name = name.rstrip("+")
    ctx.vars[name] = value
    if (name in DANGEROUS_ENV or name.startswith("GIT_CONFIG_")) and value.strip() not in _SAFE_HOOK_VALUES:
        return _risk("high", f"sets {name}, which injects code into the programs it starts")
    if name == "PATH" and any(part and not part.startswith(("/", "$")) for part in value.split(":")):
        return _risk("medium", "puts relative directories on PATH (programs could be shadowed)")
    return SAFE


def _expand_words(argv: list[str], ctx: _Ctx) -> list[str]:
    """Apply what bash would: known `$VAR` words (with word splitting) and `{a,b}` braces."""
    out: list[str] = []
    for word in argv:
        if "$IFS" in word or "${IFS}" in word:
            out.extend(word.replace("${IFS}", " ").replace("$IFS", " ").split())
            continue
        match = _WORD_VAR.match(word)
        name = match and (match.group(1) or match.group(2))
        if name and name in ctx.vars:
            out.extend(ctx.vars[name].split() or [""])
            continue
        if "{" in word and "," in word and not any(ch in word for ch in " \"'():=;"):
            words = [word]
            for _ in range(4):  # nesting depth
                nxt = []
                for item in words:
                    found = _BRACE.search(item)
                    nxt.extend(
                        [item[: found.start()] + part + item[found.end() :] for part in found.group(1).split(",")]
                        if found else [item]
                    )
                if nxt == words:
                    break
                words = nxt[:32]
            out.extend(w for w in words if w)
            continue
        out.append(word)
    return out[:256]


def _operand_risk(name: str, argv: list[str], ctx: _Ctx) -> Risk:
    """Any program handed credential or start-up files as arguments."""
    risk = SAFE
    for token in argv[1:]:
        value = token.split("=", 1)[1] if token.startswith("--") and "=" in token else token
        if not value or not (value[0] in "~/." or "/" in value):
            continue
        kind = _scope(value, ctx).kind
        if kind == "secret" and not ctx.remote:
            risk |= _risk("high", f"`{name}` is handed credential files ({value})", "read", flags=_SECRET)
        elif kind == "startup":
            risk |= _risk("high", f"`{name}` is handed shell/login start-up files ({value})", "write")
    return risk


def _program_file_risk(word: str, ctx: _Ctx) -> Risk:
    """`./tool` or `/path/script`: review the script it would run (by its #! line)."""
    text = _script_text(word, ctx)
    risk = _risk("medium", f"runs the program `{word}`")
    if not text:
        return risk
    first = text.split("\n", 1)[0]
    interpreter = _prog(first[2:].split()[-1]) if first.startswith("#!") and first[2:].split() else "sh"
    if interpreter in _INTERPRETERS:
        return risk | _code_risk(text, interpreter, inline=False, ctx=ctx)
    return risk | _classify(text, ctx.nested())


def _wrapper_risk(name: str, argv: list[str], ctx: _Ctx) -> Risk:
    if name == "flock" and "-c" in argv:
        return _shell_string(argv[argv.index("-c") + 1 : argv.index("-c") + 2], ctx)
    if name == "env":
        split = _option_values(argv, "-S", "--split-string")
        if split:
            return _shell_string(split, ctx)
    valued, positionals = _WRAPPERS[name]
    rest = _skip_options(argv, valued, positionals)
    if name == "env":
        if not rest:
            return _risk("low", "prints environment variables (may contain secrets)", "read")
        if all(_ASSIGNMENT.match(word) for word in rest):
            return _argv_risk(rest, ctx)
    if not rest:
        return SAFE
    if name == "xargs":
        # Arguments arrive on stdin at run time; the command never sees the pipe as input.
        extra = ctx.stdin_text.split() if ctx.stdin == "text" and ctx.stdin_text.strip() else [_SUBST]
        return _argv_risk([*rest, *extra[:64]], ctx.child())
    if name == "parallel":  # `parallel CMD ::: ARGS` runs CMD once per argument
        if ":::" in rest:
            split = rest.index(":::")
            template, args = rest[:split], rest[split + 1 :]
        else:
            template, args = rest, [_SUBST]
        return _argv_risk([*template, *args], ctx.child())
    return _argv_risk(rest, ctx.child(stdin=ctx.stdin, stdin_text=ctx.stdin_text))


def _read_only_risk(name: str, argv: list[str], ctx: _Ctx) -> Risk:
    flags, operands = _split_flags(argv)
    if name in ("cd", "pushd"):
        target = operands[0] if operands else "~"
        scope = _scope(target, ctx) if target != "-" else _Scope("dynamic", "-")
        ctx.cwd = None if scope.kind == "dynamic" else Path(scope.path)
        return SAFE
    if name == "popd":
        ctx.cwd = None
        return SAFE
    if name == "history" and flags & {"-c", "-d", "-w"}:
        return _risk("medium", "rewrites shell history", "write")
    if name == "journalctl" and any(f.startswith("--vacuum") or f == "--rotate" for f in flags):
        return _risk("high", "deletes system logs", "write")
    if name in ("grep", "egrep", "fgrep", "rg", "ag", "ack") and (name != "grep" or flags & {"-r", "-R", "--recursive"}):
        roots = operands[1:] if name in ("grep", "egrep", "fgrep") else operands[1:] or ["."]
        wide = [r for r in roots if _scope(r, ctx).kind in ("home_root", "root")]
        if wide:
            return _risk("medium", f"`{name}` searches your whole home folder or disk ({wide[0]})", "read")
    if name in ("ping", "dig", "nslookup", "host", "whois", "traceroute", "tracepath"):
        return _risk("low", f"`{name}` uses the network", "web", flags=_NET)
    return _read_risk(operands, ctx, verb=f"`{name}`")


def _builtin_risk(argv: list[str], ctx: _Ctx) -> Risk:
    """source / . / export / declare / tee — shell builtins with side effects."""
    name = _prog(argv[0])
    _flags, operands = _split_flags(argv)
    if name in ("source", "."):
        if not operands:
            return SAFE
        if operands[0] == _PROCSUB:
            return _risk("high", "sources shell code produced by another command")
        text = _script_text(operands[0], ctx)
        inner = _classify(text, ctx.nested()) if text else SAFE
        return _risk("medium", f"sources {operands[0]}") | inner
    if name == "tee":
        return _write_risk(operands, ctx, verb="tee writes")
    risk = SAFE
    for operand in operands:  # export / declare / typeset / local / readonly
        if "=" in operand and _ASSIGNMENT.match(operand):
            risk |= _assignment_risk(operand, ctx)
    return risk


def _shell_string(values: list[str], ctx: _Ctx) -> Risk:
    if not values:
        return SAFE
    if _SUBST in values[0] or _PROCSUB in values[0] or values[0].strip().startswith("$"):
        return _risk("high", "runs shell code that is only known at run time (e.g. a download)")
    return _classify(values[0], ctx.nested())


_STDIN_PATHS = frozenset({"-", "/dev/stdin", "/dev/fd/0", "/proc/self/fd/0"})


def _script_text(path: str, ctx: _Ctx) -> str:
    """Contents of a small local script so nested code gets reviewed, or ''."""
    if path in _STDIN_PATHS and ctx.stdin == "text":
        return ctx.stdin_text
    if ctx.depth >= _MAX_DEPTH or path in (_SUBST, _PROCSUB):
        return ""
    scope = _scope(path, ctx)
    if scope.path in ctx.files:
        return ctx.files[scope.path]
    if ctx.remote:
        return ""
    if scope.kind in ("dynamic", "device", "secret", "null") or scope.wildcard:
        return ""
    try:
        with open(scope.path, "rb") as handle:
            data = handle.read(_SCRIPT_SCAN_BYTES + 1)
    except OSError:
        return ""
    if len(data) > _SCRIPT_SCAN_BYTES or b"\x00" in data[:2048]:
        return ""
    return data.decode("utf-8", errors="replace")


def _shell_risk(argv: list[str], ctx: _Ctx) -> Risk:
    i = 1
    while i < len(argv):
        token = argv[i]
        if token == "--":
            i += 1
            break
        if token in ("-o", "+o", "-O", "+O", "--rcfile", "--init-file"):
            i += 2
            continue
        if token.startswith("-") and token != "-":
            if not token.startswith("--") and "c" in token[1:]:
                return _shell_string(argv[i + 1 : i + 2], ctx)
            i += 1
            continue
        break
    operands = argv[i:]
    if operands and operands[0] not in _STDIN_PATHS:
        if operands[0] in (_PROCSUB, _SUBST):
            return _risk("high", "runs shell code produced by another command")
        text = _script_text(operands[0], ctx)
        inner = _classify(text, ctx.nested()) if text else SAFE
        return _risk("medium", f"runs the shell script {operands[0]}") | inner
    if ctx.stdin == "text":
        return _classify(ctx.stdin_text, ctx.nested())
    if ctx.stdin == "pipe":
        return _risk("high", "pipes generated text into a shell")
    return _risk("low", "starts an interactive shell")


# Patterns in code handed to an interpreter (inline or as a script).
_CODE_SECRETS = re.compile(
    r"\.ssh\b|id_(?:rsa|dsa|ecdsa|ed25519)|\.gnupg|\.aws/|\.netrc|\.git-credentials|\.bashrc|"
    r"\.profile\b|\.zshrc|authorized_keys|/etc/(?:shadow|sudoers)|\.config/autostart|crontab|"
    r"\.local/share/keyrings|\.mozilla|google-chrome",
    re.I,
)
_CODE_TREE_DELETE = re.compile(
    r"rmtree|removedirs|rm\s+-[a-zA-Z]*[rR]|\.rm(?:Sync)?\s*\([^;]*recursive|rimraf|remove_tree|"
    r"rm_rf|FileUtils\.rm_r|File::Path|Remove-Item[^\n]*-Recurse|Deno\.remove\s*\([^;]*recursive|"
    r"rmdirSync\s*\([^;]*recursive|unlink\s+glob|unlink\s*\(\s*glob",
    re.I,
)
_CODE_HOME_OR_ROOT = re.compile(
    r"""expanduser\(\s*['"]~|Path\.home\(\)|environ(?:\.get)?\s*[\[(]\s*['"]HOME|process\.env\.HOME|"""
    r"""homedir\(\)|ENV\[['"]HOME|Deno\.env\.get\(\s*['"]HOME|\(\s*['"]/['"]\s*[,)]""",
)
_SPAWNED = re.compile(
    r"(?:os\.system|os\.popen|subprocess\.\w+|\bpopen|execSync|spawnSync|\bspawn|\bsystem|shell_exec|"
    r"passthru|\bexec|IO\.popen|Command)\s*\(?\s*(\[[^\]]*\]|(['\"`]).*?\2)"
)
_LITERAL_TREE_TARGET = re.compile(r"(?:rmtree|rimraf|rm_rf|remove_tree)\s*\(\s*(?:[rbuf]?)['\"]([^'\"]+)['\"]")
_CODE_DELETE = re.compile(r"\bunlink|\bos\.remove\b|\.remove\(|\brmdir|\bfs\.rm|\bshred\b|\btruncate\s*\(", re.I)
_CODE_MANY = re.compile(r"glob|listdir|os\.walk|scandir|iterdir|readdir|<\*>", re.I)
_CODE_SPAWN = re.compile(
    r"os\.system|subprocess|\bpopen|os\.exec|\bexec[lv]p?e?\s*\(|child_process|execSync|"
    r"\b(?:system|exec)\s+['\"]|"
    r"spawnSync|\bspawn\s*\(|\bsystem\s*\(|shell_exec|passthru|proc_open|IO\.popen|Kernel\.exec|"
    r"%x\{|pty\.spawn|Deno\.(?:run|Command)|Bun\.spawn|`[^`]*`",
    re.I,
)
_CODE_HIDDEN = re.compile(
    r"\bb64decode|base64\.|codecs\.decode|\beval\s*\(|\bexec\s*\(|\bcompile\s*\(|__import__|"
    r"fromCharCode|\batob\s*\(|new\s+Function|marshal\.loads|pickle\.loads",
    re.I,
)
_CODE_NET = re.compile(
    r"\bsocket\b|requests\.|urlopen|urllib|http\.client|\bfetch\s*\(|axios|XMLHttpRequest|"
    r"net\.connect|https?\.request|LWP::|Net::HTTP|curl_exec|smtplib|ftplib|paramiko|websocket",
    re.I,
)
_CODE_WRITE = re.compile(
    r"open\s*\([^)]*['\"][wax]|write_text|write_bytes|writeFile|appendFile|createWriteStream|"
    r"fs\.write|\.write\(|shutil\.(?:move|copy)|os\.rename|os\.replace|chmod|chown|mkdir",
    re.I,
)


def _code_risk(code: str, language: str, *, inline: bool, ctx: _Ctx | None = None) -> Risk:
    where = "inline code" if inline else "the script"
    if inline and (_SUBST in code or _PROCSUB in code or code.strip().startswith("$")):
        return _risk("high", f"runs {language} code that is only known at run time")
    risk = _risk("low" if inline else "medium", f"runs {'inline ' if inline else ''}{language} code")
    code = code + "\n" + re.sub(r"""(['"])\s*[+.]?\s*(['"])""", "", code)  # '/.s' + 'sh' → '/.ssh'
    if _CODE_SECRETS.search(code):
        risk |= _risk("high", f"{where} touches credentials or shell start-up files", flags=_SECRET)
    if ctx is not None and ctx.depth < _MAX_DEPTH:
        for match in list(_SPAWNED.finditer(code))[:6]:  # review commands the code spawns
            body = match.group(1)
            command = " ".join(re.findall(r"['\"]([^'\"]*)['\"]", body)) if body.startswith("[") else body[1:-1]
            if command.strip():
                risk |= _classify(command, ctx.nested())
    if _CODE_TREE_DELETE.search(code):
        literal = _LITERAL_TREE_TARGET.findall(code)
        bounded = literal and all(os.path.basename(t.rstrip("/")) in REGENERABLE_DIRS for t in literal) and len(
            literal
        ) == len(_CODE_TREE_DELETE.findall(code))
        risk |= _risk("medium" if bounded else "high", f"{where} deletes directory trees", "write", deletes=1)
        targets = [_scope(t, ctx).kind for t in literal] if ctx is not None else []
        if _CODE_HOME_OR_ROOT.search(code) or any(k in ("root", "home_root", "system", "secret", "startup") for k in targets):
            risk |= _risk("critical", f"{where} deletes your home folder, credentials or system files", "write")
    elif _CODE_DELETE.search(code):
        many = bool(_CODE_MANY.search(code))
        risk |= _risk("high" if many else "medium", f"{where} deletes {'many ' if many else ''}files", "write", deletes=1)
    if _CODE_SPAWN.search(code):
        risk |= _risk("high" if inline else "medium", f"{where} launches other programs")
    if inline and _CODE_HIDDEN.search(code):
        risk |= _risk("high", f"{where} decodes or evaluates hidden code")
    if _CODE_NET.search(code):
        risk |= _risk("medium", f"{where} uses the network")
    if _CODE_WRITE.search(code):
        risk |= _risk("medium", f"{where} writes files", "write", writes=1)
    return risk


def _interpreter_risk(argv: list[str], ctx: _Ctx, name: str) -> Risk:
    exact, cluster_letters = _INTERPRETERS[name]
    in_place = False
    i = 1
    while i < len(argv):
        token = argv[i]
        short = token.startswith("-") and not token.startswith("--") and len(token) > 1
        if name == "python" and short and token[1] in "WX":
            i += 1 if len(token) > 2 else 2
            continue
        if name == "python" and token == "-m":
            return _python_module(argv[i + 1 : i + 2], argv[i + 2 :], ctx)
        if name == "perl" and short and "i" in token[1:]:
            in_place = True
        attached = ""
        if name == "python" and short and "c" in token[1:]:
            attached = token[token.index("c", 1) + 1 :]
        if token in exact or attached or (short and cluster_letters and token[-1] in cluster_letters):
            code_index = i if attached else i + 1
            code = attached or (argv[code_index] if code_index < len(argv) else "")
            risk = _code_risk(code, name, inline=True, ctx=ctx)
            if in_place:
                files = [a for a in argv[code_index + 1 :] if not a.startswith("-")]
                risk |= _write_risk(files, ctx, verb="edits in place")
            return risk
        if name in ("deno", "bun", "node") and token == "eval":
            return _code_risk(argv[i + 1] if i + 1 < len(argv) else "", name, inline=True, ctx=ctx)
        if name == "php" and token == "-S":
            docroot = _option_values(argv, "-t")
            return _serve_risk(docroot[0] if docroot else ".", ctx)
        if name in ("deno", "bun") and token in ("run", "test", "x", "install", "add", "upgrade", "task"):
            return _risk("medium", f"`{name} {token}` runs project code or installs packages")
        if token.startswith("-") and token != "-":
            i += 2 if token in ("-r", "--require", "--import", "--loader", "-I") else 1
            continue
        break
    operands = argv[i:]
    if operands and operands[0] not in _STDIN_PATHS:
        if operands[0] in (_PROCSUB, _SUBST):
            return _risk("high", f"runs {name} code produced by another command")
        text = _script_text(operands[0], ctx)
        risk = _risk("medium", f"runs the {name} script {operands[0]}")
        return risk | _code_risk(text, name, inline=False, ctx=ctx) if text else risk
    if ctx.stdin == "text":
        return _code_risk(ctx.stdin_text, name, inline=True, ctx=ctx)
    if ctx.stdin == "pipe":
        return _risk("high", f"pipes generated text into the {name} interpreter")
    return _risk("low", f"starts an interactive {name} interpreter")


def _python_module(module: list[str], rest: list[str], ctx: _Ctx) -> Risk:
    name = module[0] if module else ""
    if name == "pip":
        return _dev_tool_risk("pip", ["pip", *rest], ctx)
    if name in ("json.tool", "py_compile", "site", "this", "platform", "sysconfig", "tokenize",
                "pydoc", "timeit", "calendar", "doctest", "ast", "dis", "tabnanny"):
        return _risk("low", f"runs `python -m {name}`")
    if name == "http.server":
        directory = _option_values(rest, "-d", "--directory")
        return _serve_risk(directory[0] if directory else ".", ctx)
    return _risk("medium", f"runs `python -m {name}`")


def _serve_risk(directory: str, ctx: _Ctx) -> Risk:
    kind = _scope(directory, ctx).kind
    if kind in _IN_WORKSPACE or kind == "tmp":
        return _risk("medium", "starts a local web server", flags=_NET)
    return _risk("high", f"serves {_PLACE[kind]} over the network ({directory})", flags=_NET)


# --- program-specific handlers ----------------------------------------------


def _rm(argv: list[str], ctx: _Ctx) -> Risk:
    flags, targets = _split_flags(argv)
    if "--no-preserve-root" in flags:
        return _risk("critical", "rm --no-preserve-root", "write", deletes=1)
    return _delete_risk(targets, ctx, recursive=bool(flags & {"-r", "-R", "--recursive"}), verb="rm")


def _rmdir(argv: list[str], ctx: _Ctx) -> Risk:
    return _delete_risk(_split_flags(argv)[1], ctx, recursive=False, verb=_prog(argv[0]))


def _shred(argv: list[str], ctx: _Ctx) -> Risk:
    flags, targets = _split_flags(argv)
    risk = _delete_risk(targets, ctx, recursive=bool(flags & {"-r", "-R"}), verb=_prog(argv[0]))
    return risk | _risk("high", f"`{_prog(argv[0])}` destroys data beyond recovery", "write")


def _find(argv: list[str], ctx: _Ctx) -> Risk:
    roots: list[str] = []
    i = 1
    while i < len(argv) and not argv[i].startswith(("-", "(", "!", ")")):
        roots.append(argv[i])
        i += 1
    roots = roots or ["."]
    expression = argv[i:]
    patterns = _option_values(expression, "-name", "-iname")
    regenerable = bool(patterns) and all(_REGENERABLE_GLOB.match(p) or p in REGENERABLE_DIRS for p in patterns)
    risk = _read_risk(roots, ctx, verb="find")
    if any(_SECRET_NAME.match(p.strip("*")) for p in patterns + _option_values(expression, "-path", "-ipath")):
        risk |= _risk("high", "find searches for credential files", "read", flags=_SECRET)
    deletes = "-delete" in expression
    j = 0
    while j < len(expression):
        if expression[j] in ("-exec", "-execdir", "-ok", "-okdir"):
            end = j + 1
            while end < len(expression) and expression[end] not in (";", "+"):
                end += 1
            sub = [_SUBST if w == "{}" else w for w in expression[j + 1 : end]]
            if sub and _prog(sub[0]) in ("rm", "rmdir", "unlink", "shred"):
                deletes = True
            elif sub:
                risk |= _argv_risk(sub, ctx.child())
            j = end
        elif expression[j] in ("-fprint", "-fprint0", "-fprintf", "-fls") and j + 1 < len(expression):
            risk |= _write_risk([expression[j + 1]], ctx, verb="find writes")
        j += 1
    if deletes:
        for root in roots:
            scope = _scope(root, ctx)
            level = _DELETE_LEVEL[scope.kind]
            if scope.kind in ("workspace", "workspace_root", "dynamic", "home", "outside") and regenerable:
                level = "medium" if scope.kind != "home" and scope.kind != "outside" else "high"
            elif scope.kind in ("workspace", "workspace_root", "dynamic"):
                level = "high"
            risk |= _risk(level, f"find deletes every match under {_PLACE[scope.kind]} ({root})",
                          "write", deletes=1 if level == "medium" else MASS_DELETE_LIMIT)
    return risk


def _copy_move(argv: list[str], ctx: _Ctx) -> Risk:
    name = _prog(argv[0])
    target_dir = _option_values(argv, "-t", "--target-directory")
    flags, operands = _split_flags([a for a in argv if a not in target_dir])
    if name == "rsync" and any(re.match(r"^[^/\s]*:", o) or o.startswith("rsync://") for o in operands):
        return _risk("high", "rsync copies to or from another machine (use the `remote` tool)")
    if target_dir:
        sources, dest = operands, target_dir[0]
    elif len(operands) >= 2:
        sources, dest = operands[:-1], operands[-1]
    else:
        return SAFE
    risk = _read_risk(sources, ctx, verb=name)
    if name == "mv" and dest == "/dev/null":
        return _risk("critical", "moves files into /dev/null (destroys them)", "write", deletes=len(sources))
    risk |= _write_risk([dest], ctx, verb=f"{name} writes into")
    if name == "mv":
        for source in sources:
            scope = _scope(source, ctx)
            if scope.kind == "workspace":
                risk |= _risk("medium", f"mv moves workspace files ({source})", "write")
            elif scope.kind not in ("regenerable", "tmp", "null", "dynamic"):
                risk |= _risk(_DELETE_LEVEL[scope.kind], f"mv takes {_PLACE[scope.kind]} away from where they are ({source})", "write")
    if name == "rsync" and any(f.startswith("--delete") for f in flags):
        risk |= _delete_risk([dest], ctx, recursive=True, verb="rsync --delete")
    if name == "rsync" and "--remove-source-files" in flags:
        risk |= _delete_risk(sources, ctx, recursive=True, verb="rsync --remove-source-files")
    return risk


def _ln(argv: list[str], ctx: _Ctx) -> Risk:
    flags, operands = _split_flags(argv)
    if flags & {"-s", "--symbolic"} and operands:
        target = _scope(operands[0], ctx)
        link = _scope(operands[1], ctx) if len(operands) > 1 else _scope(os.path.basename(operands[0]), ctx)
        if target.kind != "dynamic" and link.kind != "dynamic":
            ctx.links[link.path] = target.path
    return _write_risk(operands[-1:] if len(operands) > 1 else operands, ctx, verb="ln creates a link in", create=True)


def _chmod(argv: list[str], ctx: _Ctx) -> Risk:
    name = _prog(argv[0])
    flags, operands = _split_flags(argv)
    if not operands:
        return SAFE
    spec, targets = ("", operands) if any(f.startswith("--reference") for f in flags) else (operands[0], operands[1:])
    recursive = bool(flags & {"-R", "--recursive"})
    risk = _write_risk(targets, ctx, verb=f"{name} changes permissions of", create=True)
    if name in ("chown", "chgrp"):
        risk |= _risk("high", f"{name} changes file ownership", "write")
    if recursive and re.search(r"(?:^|[^0-7])0?(?:777|000)$", spec):
        risk |= _risk("high", f"{name} -R {spec} breaks or opens up a whole tree", "write")
    if recursive:
        for target in targets:
            kind = _scope(target, ctx).kind
            if kind in ("home_root", "root", "system", "secret", "startup"):
                risk |= _risk("critical", f"{name} -R on {_PLACE[kind]}", "write")
    return risk


def _dd(argv: list[str], ctx: _Ctx) -> Risk:
    outputs = [a[3:] for a in argv[1:] if a.startswith("of=")]
    inputs = [a[3:] for a in argv[1:] if a.startswith("if=")]
    return _read_risk(inputs, ctx, verb="dd") | _write_risk(outputs, ctx, verb="dd writes")


def _truncate(argv: list[str], ctx: _Ctx) -> Risk:
    sizes = set(_option_values(argv, "-s", "--size"))
    targets = [o for o in _split_flags(argv)[1] if o not in sizes]
    return _write_risk(targets, ctx, verb="truncate empties")


def _sed(argv: list[str], ctx: _Ctx) -> Risk:
    in_place = any(t.startswith("--in-place") or (t.startswith("-") and not t.startswith("--") and "i" in t[1:]) for t in argv[1:])
    scripts = _option_values(argv, "-e", "--expression", "-f", "--file")
    operands = _split_flags([a for a in argv if a not in scripts])[1]
    files = operands if scripts else operands[1:]
    script_text = "\n".join(scripts or operands[:1])
    executes = re.search(r"(?:^|[;\n}])\s*e\s+(.+)", script_text)
    replaced = re.search(r"\bs(.)((?:(?!\1).)*)\1((?:(?!\1).)*)\1[a-zA-Z0-9]*e", script_text)
    risk = SAFE
    if executes:
        risk |= _shell_string([executes.group(1)], ctx)
    if replaced:
        risk |= _shell_string([replaced.group(3)], ctx) | _risk("high", "sed s///e runs its replacement as a command")
    if not in_place:
        return risk | _read_risk(files, ctx, verb="sed")
    return risk | _write_risk(files, ctx, verb="sed -i edits")


def _awk(argv: list[str], ctx: _Ctx) -> Risk:
    program = next((a for a in argv[1:] if not a.startswith("-")), "")
    if re.search(r"\bsystem\s*\(|\|\s*\"|\|&|/inet/", program):
        return _risk("high", "the awk program runs shell commands")
    files = [a for a in argv[1:] if not a.startswith("-") and a != program and a != "inplace"]
    risk = _read_risk(files + re.findall(r'getline(?:\s+\w+)?\s*<\s*"([^"]+)"', program), ctx, verb="awk")
    outputs = re.findall(r'print[^;]*>>?\s*"([^"]+)"', program)
    if outputs:
        risk |= _write_risk(outputs, ctx, verb="the awk program writes")
    if "inplace" in argv:
        risk |= _write_risk(files, ctx, verb="awk -i inplace edits")
    return risk


def _curl(argv: list[str], ctx: _Ctx) -> Risk:
    risk = _risk("low", "downloads from the network", "web", flags=_NET)
    data = _option_values(argv, "-d", "--data", "--data-binary", "--data-raw", "--data-urlencode",
                          "--data-ascii", "--json")
    forms = _option_values(argv, "-F", "--form")
    files = [v[1:] for v in data if v.startswith("@") and v != "@-"]
    files += [m.group(1) for v in forms for m in [re.search(r"=[@<]([^;]+)", v)] if m]
    files += _option_values(argv, "-T", "--upload-file")
    for path in files:
        kind = _scope(path, ctx).kind
        risk |= _risk("critical" if kind in ("secret", "kiwibro") else "high", f"uploads a local file to the network ({path})")
    methods = {m.upper() for m in _option_values(argv, "-X", "--request")}
    if data or forms or methods & {"POST", "PUT", "PATCH", "DELETE"}:
        risk |= _risk("medium", "sends data to a remote server")
    risk |= _write_risk([o for o in _option_values(argv, "-o", "--output") if o != "-"], ctx, verb="curl saves into")
    risk |= _write_risk(_option_values(argv, "--output-dir"), ctx, verb="curl saves into", create=True)
    if {"-O", "--remote-name", "-J", "--remote-name-all"} & set(argv):
        risk |= _risk("medium", "saves a download into the current directory", "write", writes=1)
    return risk


def _wget(argv: list[str], ctx: _Ctx) -> Risk:
    risk = _risk("low", "downloads from the network", "web", flags=_NET)
    for path in _option_values(argv, "--post-file", "--body-file"):
        kind = _scope(path, ctx).kind
        risk |= _risk("critical" if kind in ("secret", "kiwibro") else "high", f"uploads a local file to the network ({path})")
    if _option_values(argv, "--post-data", "--body-data", "--method"):
        risk |= _risk("medium", "sends data to a remote server")
    outputs = [o for o in _option_values(argv, "-O", "--output-document") if o != "-"]
    if _option_values(argv, "-O", "--output-document") and not outputs:
        return risk  # -O - : stdout only
    targets = outputs or _option_values(argv, "-P", "--directory-prefix") or ["."]
    risk |= _write_risk(targets, ctx, verb="wget saves into", create=True)
    if {"-r", "--recursive", "-m", "--mirror"} & set(argv):
        risk |= _risk("medium", "mirrors many files from the network", "write")
    return risk


def _netcat(argv: list[str], ctx: _Ctx) -> Risk:
    return replace(_netcat_level(argv, ctx), flags=_NET)


def _netcat_level(argv: list[str], ctx: _Ctx) -> Risk:
    name = _prog(argv[0])
    if name == "socat":
        text = " ".join(argv[1:])
        if re.search(r"\b(?:EXEC|SYSTEM):", text, re.I):
            return _risk("critical", "socat wires a network socket to a program (remote shell)")
        if re.search(r"LISTEN", text, re.I):
            return _risk("high", "opens a listening network socket")
        return _risk("medium", "opens a raw network connection")
    flags = _split_flags(argv)[0]
    if flags & {"-e", "-c", "--exec", "--sh-exec", "--lua-exec"}:
        return _risk("critical", f"`{name}` wires a network socket to a program (remote shell)")
    if flags & {"-l", "--listen"}:
        return _risk("high", "opens a listening network socket")
    return _risk("medium", "opens a raw network connection")


def _kill(argv: list[str], ctx: _Ctx) -> Risk:
    name = _prog(argv[0])
    rest = argv[1:]
    if name == "kill":
        if rest[:1] in (["-l"], ["-L"]):
            return SAFE
        if rest[:1] in (["-s"], ["-n"], ["--signal"]):
            rest = rest[2:]
        elif len(rest) > 1 and re.fullmatch(r"-(?:[0-9]+|[A-Za-z][A-Za-z0-9+]*)", rest[0]):
            rest = rest[1:]
        rest = rest[1:] if rest[:1] == ["--"] else rest
        if any(pid in ("1", "-1") for pid in rest):
            return _risk("critical", "kill targets init or every process")
        if rest and all(pid.startswith("%") or pid in ("$!", "$$") for pid in rest):
            return _risk("low", "stops a job started by this command")
    return _risk("high", f"`{name}` stops processes that may not belong to this task")


def _systemctl(argv: list[str], ctx: _Ctx) -> Risk:
    name = _prog(argv[0])
    verbs = _verbs(argv)
    verb = verbs[0] if verbs else "list-units"
    if name == "loginctl":
        if verb.startswith(("terminate", "kill")) or verb in ("poweroff", "reboot", "suspend", "hibernate"):
            return _risk("critical", f"loginctl {verb} ends sessions")
        return _risk("low", f"loginctl {verb}", "read")
    if name == "service":
        verb = verbs[1] if len(verbs) > 1 else "status"
    if verb in ("poweroff", "reboot", "halt", "suspend", "hibernate", "hybrid-sleep", "kexec",
                "emergency", "rescue", "isolate", "default", "soft-reboot"):
        return _risk("critical", f"systemctl {verb} affects the whole machine")
    if verb in ("status", "show", "cat", "is-active", "is-enabled", "is-failed", "help") or verb.startswith("list-"):
        return _risk("low", f"{name} {verb}", "read")
    return _risk("high", f"{name} {verb} changes services outside the workspace")


def _crontab(argv: list[str], ctx: _Ctx) -> Risk:
    flags = _split_flags(argv)[0]
    if flags == {"-l"}:
        return _risk("low", "lists cron jobs", "read")
    if "-r" in flags:
        return _risk("critical", "crontab -r deletes every scheduled job")
    return _risk("high", "installs persistent scheduled jobs")


def _at(argv: list[str], ctx: _Ctx) -> Risk:
    if _prog(argv[0]) == "atq" or "-l" in argv:
        return _risk("low", "lists scheduled jobs", "read")
    return _risk("high", "schedules commands to run later")


_GIT_READ = frozenset({
    "status", "log", "diff", "show", "ls-files", "ls-tree", "rev-parse", "rev-list", "blame",
    "grep", "describe", "shortlog", "cat-file", "show-ref", "for-each-ref", "help", "version",
    "count-objects", "fsck", "whatchanged", "name-rev", "merge-base", "cherry", "check-ignore",
    "var", "annotate", "range-diff", "verify-commit",
})
_GIT_EXEC_CONFIG = re.compile(
    r"(?i)core\.(?:sshCommand|hooksPath|fsmonitor|pager|editor|askPass|gitProxy)|alias\.|credential\.|"
    r"protocol\.|filter\.|diff\.(?:external|\w+\.textconv|\w+\.command)|gpg\.(?:\w+\.)?program|"
    r"sequence\.editor|merge\.\w+\.driver|uploadpack\.|receivepack\.|remote\.\S+\.(?:uploadpack|receivepack|proxy)"
)
_GIT_EXEC_OPTIONS = ("--upload-pack", "--receive-pack", "--exec", "--extcmd", "-x", "-u")


def _git(argv: list[str], ctx: _Ctx) -> Risk:
    i = 1
    while i < len(argv) and argv[i].startswith("-"):
        if argv[i] in ("-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"):
            if argv[i] == "-c" and i + 1 < len(argv) and _GIT_EXEC_CONFIG.match(argv[i + 1]):
                return _risk("high", f"git -c {argv[i + 1].split('=')[0]} can run arbitrary programs")
            i += 2
            continue
        if argv[i].startswith("--config-env=") and _GIT_EXEC_CONFIG.match(argv[i].split("=", 1)[1]):
            return _risk("high", "git --config-env sets a key that can run arbitrary programs")
        i += 1
    if i >= len(argv):
        return SAFE
    verb, rest = argv[i], argv[i + 1 :]
    flags, operands = _split_flags([verb, *rest])
    if verb in ("ls-remote", "fetch", "clone", "push", "pull", "archive", "difftool", "rebase") and any(
        a == o or a.startswith(o + "=") for a in rest for o in _GIT_EXEC_OPTIONS if not (o in ("-x", "-u") and verb not in ("rebase", "clone", "difftool"))
    ):
        return _risk("high", f"git {verb} is told to run another program")
    if verb == "bisect" and operands[:1] == ["run"]:
        return _risk("high", "git bisect run executes a command repeatedly")
    if verb in ("diff", "archive", "format-patch") and _option_values(rest, "--output", "-o"):
        return _write_risk(_option_values(rest, "--output", "-o"), ctx, verb=f"git {verb} writes")
    if verb in _GIT_READ:
        return SAFE
    if verb in ("fetch", "ls-remote"):
        return _risk("low", f"git {verb} reads from the remote", "web")
    if verb == "push":
        if flags & {"-f", "--force", "--force-with-lease", "--mirror", "--delete", "-d", "--prune"} or any(o.startswith((":", "+")) for o in operands):
            return _risk("high", "git push rewrites or deletes history on the remote")
        return _risk("high", "git push publishes commits outside this machine")
    if verb == "reset" and flags & {"--hard", "--merge", "--keep"}:
        return _risk("high", "git reset --hard discards uncommitted work", "write")
    if verb == "clean":
        if flags & {"-n", "--dry-run"}:
            return SAFE
        return _risk("high", "git clean deletes untracked files", "write", deletes=MASS_DELETE_LIMIT)
    if verb == "restore" and "--staged" not in flags and "-S" not in flags or (
        verb == "checkout" and ("--" in rest or "." in operands or flags & {"-f", "--force"})
    ):
        return _risk("high", f"git {verb} overwrites uncommitted changes", "write")
    if verb == "stash":
        sub = operands[0] if operands else "push"
        if sub in ("drop", "clear"):
            return _risk("high", f"git stash {sub} deletes stashed work", "write")
        return SAFE if sub in ("list", "show") else _risk("medium", f"git stash {sub}", "write")
    if verb == "branch":
        if flags & {"-D", "-d", "--delete", "-M", "-m", "-C", "--move", "--force", "-f"}:
            return _risk("high", "git branch deletes or renames branches", "write")
        return SAFE if not operands else _risk("medium", "git branch creates a branch", "write")
    if verb == "tag":
        if flags & {"-d", "--delete", "-f", "--force"}:
            return _risk("high", "git tag deletes or moves tags", "write")
        return SAFE if not operands or flags & {"-l", "--list"} else _risk("medium", "git tag creates a tag", "write")
    if verb == "remote":
        if not operands or operands[0] in ("-v", "show", "get-url"):
            return SAFE
        return _risk("medium", f"git remote {operands[0]} changes where pushes go", "write")
    if verb == "config":
        if flags & {"--get", "--list", "-l", "--get-all", "--get-regexp", "--show-origin"} or len(operands) < 2:
            return SAFE
        if flags & {"--global", "--system"}:
            return _risk("high", "git config --global changes settings outside the workspace", "write")
        if _GIT_EXEC_CONFIG.match(operands[0]):
            return _risk("high", f"git config {operands[0]} can run arbitrary programs", "write")
        return _risk("medium", "git config changes repository settings", "write")
    if (verb == "reflog" and operands[:1] in (["expire"], ["delete"])) or (
        verb in ("gc", "prune") and any(f.startswith("--prune") for f in flags)
    ) or verb in ("filter-branch", "filter-repo", "update-ref", "replace"):
        return _risk("high", f"git {verb} removes history or recovery points", "write")
    if verb == "submodule" and operands[:1] == ["foreach"]:
        return _risk("high", "git submodule foreach runs commands in every submodule")
    if verb.startswith("credential"):
        return _risk("high", "git credential handles stored passwords")
    if verb == "rm":
        return _risk("medium", "git rm deletes tracked files (recoverable from git)", "write", deletes=len(operands))
    if verb == "clone":
        return _risk("medium", "git clone downloads a repository", "write", writes=1)
    return _risk("medium", f"git {verb} changes the repository", "write")


_DOCKER_HOST_ESCAPE = re.compile(
    r"--privileged|--pid[= ]host|--cap-add|--device|--security-opt|docker\.sock|podman\.sock|"
    r"(?:-v|--volume)[= ]?\s*(?:/|~|\$HOME|\$\{HOME\}|/home|/root|/etc[^\s:]*|/var/run[^\s:]*|/run[^\s:]*):|"
    r"--mount[= ]\S*(?:src|source)=(?:/|~|\$HOME|/home|/root|/etc)(?:[,\s]|$)"
)


def _infra_risk(name: str, argv: list[str]) -> Risk:
    verbs = _verbs(argv)
    shown = " ".join(verbs[:2])
    if name in ("docker", "podman", "nerdctl", "docker-compose") and _DOCKER_HOST_ESCAPE.search(" ".join(argv[1:])):
        return _risk("critical", f"`{name}` gives a container control of the host")
    if not verbs or (set(verbs[:2]) & _INFRA_READ_VERBS and not set(verbs) & _INFRA_WRITE_VERBS):
        return _risk("low", f"`{name} {shown}` reads state from a service", "web")
    return _risk("high", f"`{name} {shown}` acts on services outside this machine")


def _dev_tool_risk(name: str, argv: list[str], ctx: _Ctx) -> Risk:
    verbs = _verbs(argv)
    verb = verbs[0] if verbs else ""
    flags, operands = _split_flags(argv)
    if verb in _PUBLISH_VERBS and name in ("npm", "pnpm", "yarn", "cargo", "gem", "poetry", "uv",
                                           "pdm", "hatch", "twine", "pip", "composer", "bundle"):
        return _risk("high", f"`{name} {verb}` publishes or changes registry accounts")
    if name in ("npm", "pnpm", "yarn") and (flags & {"-g", "--global"} or "--location=global" in argv):
        return _risk("high", f"`{name}` installs packages globally")
    if name == "pip" and ("--break-system-packages" in argv or _option_values(argv, "--prefix", "--root")):
        return _risk("high", "pip modifies the system Python installation")
    if (name == "pip" and verb in ("list", "show", "freeze", "check", "help", "index", "inspect", "debug")) or (
        name in ("npm", "pnpm", "yarn") and verb in ("ls", "list", "view", "info", "outdated", "why", "explain", "doctor", "help")
    ) or flags & {"--version", "-V"} and not verbs:
        return _risk("low", f"`{name} {verb}` reads package metadata", "read")
    if name == "make" and "install" in verbs:
        return _risk("high", "make install copies software outside the workspace")
    if name == "tar":
        text = " ".join(argv[1:])
        if re.search(r"--to-command|--checkpoint-action|--use-compress-program|--info-script|--new-volume-script|(?:^| )-I ", text):
            return _risk("high", "tar runs an external program")
        if "--remove-files" in text:
            return _risk("high", "tar deletes the files it archives", "write", deletes=MASS_DELETE_LIMIT)
        mode = argv[1] if len(argv) > 1 and not argv[1].startswith("-") else "".join(f[1:] for f in flags if not f.startswith("--"))
        if "t" in mode and not set(mode) & {"x", "c", "r", "u"}:
            return SAFE
        return _write_risk(_option_values(argv, "-C", "--directory", "-f", "--file") or ["."], ctx, verb="tar writes into", create=True)
    if name == "unzip":
        if flags & {"-l", "-t", "-Z", "-v"}:
            return SAFE
        return _write_risk(_option_values(argv, "-d") or ["."], ctx, verb="unzip extracts into", create=True)
    if name in ("touch", "mkdir"):
        return _write_risk(operands, ctx, verb=f"{name} creates", create=True)
    if name == "install":
        return _copy_move(argv, ctx)
    if name in ("trash", "trash-put") or (name == "gio" and verb in ("trash", "remove")):
        return _delete_risk(operands[1:] if name == "gio" else operands, ctx, recursive=False, verb=name)
    if name in _OPENERS or (name == "gio" and verb == "open"):
        return _risk("medium", "opens a desktop application")
    if name == "patch":
        return _risk("medium", "patch edits workspace files", "write", writes=1)
    if name in _CHECKERS and "--fix" not in argv and verb not in ("format",) and not (name == "tsc" and "--noEmit" not in argv):
        return _risk("low", f"`{name}` checks code")
    if name in ("npx", "pnpx", "bunx", "uvx") or (name in ("npm", "pnpm") and verb in ("exec", "x", "dlx")):
        return _risk("medium", f"`{name}` downloads and runs a package")
    return _risk("medium", f"runs `{name}{(' ' + verb) if verb else ''}`")


def _eval(argv: list[str], ctx: _Ctx) -> Risk:
    text = " ".join(argv[1:])
    if any(marker in text for marker in ("$", _SUBST, _PROCSUB)):
        return _risk("high", "eval runs code assembled at run time")
    return _classify(text, ctx.nested())


def _command_builtin(argv: list[str], ctx: _Ctx) -> Risk:
    if argv[1:2] in (["-v"], ["-V"]):
        return SAFE
    rest = argv[2:] if argv[1:2] == ["-p"] else argv[1:]
    return _argv_risk(rest, ctx) if rest else SAFE


def _watch(argv: list[str], ctx: _Ctx) -> Risk:
    rest = _skip_options(argv, frozenset({"-n", "--interval", "-d", "--differences", "-g", "-e"}), 0)
    return _classify(" ".join(rest), ctx.nested()) if rest else SAFE


def _script_cmd(argv: list[str], ctx: _Ctx) -> Risk:
    for index, token in enumerate(argv[1:], start=1):
        if token.startswith("--command"):
            return _shell_string(_option_values(argv, "--command"), ctx)
        if token.startswith("-") and not token.startswith("--") and "c" in token[1:]:
            attached = token[token.index("c", 1) + 1 :]
            return _shell_string([attached] if attached else argv[index + 1 : index + 2], ctx)
    return SAFE


def _desktop_input(argv: list[str], ctx: _Ctx) -> Risk:
    return _risk("high", f"`{_prog(argv[0])}` types or clicks on your desktop")


def _desktop_config(argv: list[str], ctx: _Ctx) -> Risk:
    verbs = _verbs(argv)
    if verbs and verbs[0] in ("get", "list-keys", "list-schemas", "list-recursively", "read", "list", "dump", "range", "describe", "query", "--display"):
        return _risk("low", f"reads settings with `{_prog(argv[0])}`", "read")
    return _risk("high", f"`{_prog(argv[0])}` changes desktop or system settings")


def _escape_hatch(argv: list[str], ctx: _Ctx) -> Risk:
    return _risk("high", f"`{_prog(argv[0])}` starts processes outside Kiwibro's sandbox")


def _trap_alias(argv: list[str], ctx: _Ctx) -> Risk:
    """`trap 'cmd' EXIT` and `alias x='cmd'` run their bodies later: review them now."""
    bodies = argv[1:2] if _prog(argv[0]) == "trap" else [a.split("=", 1)[1] for a in argv[1:] if "=" in a]
    return functools.reduce(lambda acc, body: acc | _shell_string([body], ctx), bodies, SAFE)


def _ip(argv: list[str], ctx: _Ctx) -> Risk:
    if argv[1:3] == ["netns", "exec"] and len(argv) > 4:
        return _argv_risk(argv[4:], ctx)
    return _read_only_risk("ip", argv, ctx)


def _mailer(argv: list[str], ctx: _Ctx) -> Risk:
    attachments = _option_values(argv, "-A", "-a", "--attach")
    return _risk("medium", f"`{_prog(argv[0])}` sends email", flags=_NET) | _read_risk(attachments, ctx, verb="mail attaches")


# program: (options whose value is executed as a command, options whose value is an output
# path, how many trailing operands are outputs, flag that turns operands into in-place edits)
_OPTION_EFFECTS: dict[str, tuple[tuple[str, ...], tuple[str, ...], int, str]] = {
    "vim": (("-c", "--cmd", "+"), (), 0, ""), "vi": (("-c", "--cmd", "+"), (), 0, ""),
    "nvim": (("-c", "--cmd", "+"), (), 0, ""), "view": (("-c", "--cmd", "+"), (), 0, ""),
    "ex": (("-c", "--cmd", "+"), (), 0, ""), "gvim": (("-c", "--cmd", "+"), (), 0, ""),
    "gdb": (("-ex", "--eval-command", "-iex"), (), 0, ""), "emacs": (("--eval", "-eval"), (), 0, ""),
    "expect": (("-c",), (), 0, ""), "zip": (("-TT", "--unzip-command"), (), 0, ""),
    "less": (("+",), (), 0, ""), "more": (("+",), (), 0, ""), "man": (("-P", "--pager"), (), 0, ""),
    "sort": (("--compress-program",), ("-o", "--output"), 0, ""), "rg": (("--pre",), (), 0, ""),
    "fd": (("-x", "--exec", "-X", "--exec-batch"), (), 0, ""),
    "fdfind": (("-x", "--exec", "-X", "--exec-batch"), (), 0, ""),
    "tree": ((), ("-o",), 0, ""), "aria2c": ((), ("-o", "--out", "-d", "--dir"), 0, ""),
    "7z": ((), ("-o",), 0, ""), "7za": ((), ("-o",), 0, ""), "yq": ((), (), 0, "-i"),
    "xxd": ((), (), 1, ""), "uniq": ((), (), 1, ""), "pdftotext": ((), (), 1, ""),
}


def _editor(argv: list[str], ctx: _Ctx) -> Risk:
    """Tools whose options can run commands or write files (editors, pagers, sort -o, …)."""
    name = _prog(argv[0])
    if name == "dmesg":
        return _risk("medium", "clears the kernel log", "write") if {"-C", "-c", "--clear"} & set(argv) else SAFE
    executes, outputs, positional, in_place = _OPTION_EFFECTS[name]
    risk = _risk("medium", f"runs `{name}`") if name in ("vim", "vi", "nvim", "gvim", "emacs", "gdb", "expect", "zip", "7z", "7za", "aria2c") else SAFE
    commands = _option_values(argv, *(o for o in executes if o != "+"))
    if "+" in executes:
        commands += [a[1:] for a in argv[1:] if a.startswith("+") and len(a) > 1]
    for command in commands:
        text = command.lstrip(":")
        if text.startswith("!"):
            risk |= _shell_string([text[1:]], ctx)
        elif name in ("fd", "fdfind"):
            flag = next(a for a in argv if a in executes)
            tail = argv[argv.index(flag) + 1 :]
            risk |= _argv_risk([*(tail[: tail.index(";")] if ";" in tail else tail), _SUBST], ctx.child())
        elif name in ("vim", "vi", "nvim", "view", "ex", "gvim") and not re.search(r"!|system|terminal|shell", text):
            continue
        else:
            risk |= _risk("high", f"`{name}` runs commands passed in its options ({command[:40]})")
            if name in ("man", "sort", "rg", "zip"):  # the option value is itself a command line
                risk |= _shell_string([command], ctx)
    targets = _option_values(argv, *outputs)
    operands = _split_flags(argv)[1]
    if positional and len(operands) > positional and not (name == "xxd" and "-r" not in argv):
        targets += operands[-positional:]
    if in_place and any(a == in_place or a.startswith(in_place) for a in argv[1:]):
        targets += [o for o in operands if "/" in o or "." in o][1:] if name == "yq" else operands
    return risk | (_write_risk(targets, ctx, verb=f"`{name}` writes") if targets else SAFE) | _operand_risk(name, argv, ctx)


_HANDLERS: dict[str, Callable[[list[str], _Ctx], Risk]] = {
    "rm": _rm, "rmdir": _rmdir, "unlink": _rmdir, "shred": _shred, "wipe": _shred, "srm": _shred,
    "find": _find, "cp": _copy_move, "mv": _copy_move, "rsync": _copy_move, "ln": _ln,
    "chmod": _chmod, "chown": _chmod, "chgrp": _chmod, "dd": _dd, "truncate": _truncate,
    "sed": _sed, "gsed": _sed, "awk": _awk, "gawk": _awk, "mawk": _awk, "nawk": _awk,
    "curl": _curl, "wget": _wget, "nc": _netcat, "ncat": _netcat, "netcat": _netcat,
    "socat": _netcat, "telnet": _netcat, "kill": _kill, "pkill": _kill, "killall": _kill,
    "xkill": _kill, "systemctl": _systemctl, "service": _systemctl, "loginctl": _systemctl,
    "crontab": _crontab, "at": _at, "batch": _at, "atq": _at, "atrm": _at, "git": _git,
    "eval": _eval, "command": _command_builtin, "watch": _watch, "script": _script_cmd,
    "source": _builtin_risk, ".": _builtin_risk, "tee": _builtin_risk, "export": _builtin_risk,
    "declare": _builtin_risk, "typeset": _builtin_risk, "local": _builtin_risk,
    "readonly": _builtin_risk,
    "xdotool": _desktop_input, "ydotool": _desktop_input, "xte": _desktop_input,
    "wtype": _desktop_input, "gsettings": _desktop_config, "dconf": _desktop_config,
    "update-alternatives": _desktop_config, "chattr": _desktop_config, "setfacl": _desktop_config,
    "systemd-run": _escape_hatch, "nsenter": _escape_hatch, "unshare": _escape_hatch,
    "bwrap": _escape_hatch, "trap": _trap_alias, "alias": _trap_alias, "ip": _ip,
    "vim": _editor, "vi": _editor, "nvim": _editor, "view": _editor, "ex": _editor, "gvim": _editor,
    "gdb": _editor, "emacs": _editor, "expect": _editor, "zip": _editor, "less": _editor,
    "more": _editor, "man": _editor, "sort": _editor, "rg": _editor, "fd": _editor, "fdfind": _editor,
    "tree": _editor, "aria2c": _editor, "7z": _editor, "7za": _editor, "yq": _editor, "xxd": _editor,
    "uniq": _editor, "pdftotext": _editor, "dmesg": _editor, "mail": _mailer, "mailx": _mailer,
    "mutt": _mailer, "sendmail": _mailer,
}


def _static_output(words: list[str], upstream: str | None) -> str | None:
    """What a simple producer writes to stdout, when knowable without running it.

    Lets `echo 'cm0gLXJmIH4=' | base64 -d | sh` be judged on the decoded text."""
    if not words or any(_SUBST in w or _PROCSUB in w for w in words):
        return None
    name, args = _prog(words[0]), words[1:]
    if name == "echo":
        escapes = False
        while args and re.fullmatch(r"-[neE]+", args[0]):
            escapes = escapes or "e" in args[0]
            args = args[1:]
        return _unescape(" ".join(args)) if escapes else " ".join(args)
    if name == "printf":
        return _unescape(" ".join(args))
    if upstream is None:
        return None
    try:
        if name in ("base64", "base32") and set(args) & {"-d", "--decode", "-D"}:
            data = re.sub(r"\s+", "", upstream)
            decoder = base64.b64decode if name == "base64" else base64.b32decode
            return decoder(data + "=" * (-len(data) % (4 if name == "base64" else 8))).decode("utf-8", "replace")
        if name == "xxd" and "-r" in args:
            return bytes.fromhex(re.sub(r"\s+", "", upstream)).decode("utf-8", "replace")
    except (binascii.Error, ValueError):
        return None
    if name == "rev":
        return "\n".join(line[::-1] for line in upstream.splitlines())
    if name == "cat" and not args:
        return upstream
    return None


def _unescape(text: str) -> str:
    """printf / echo -e escapes (\\xHH, \\NNN octal, \\n …) as the shell would expand them."""
    simple = {"n": "\n", "t": "\t", "\\": "\\", "a": "\a", "r": "\r", "e": "\x1b", "0": "\0"}

    def repl(match: re.Match[str]) -> str:
        token = match.group(1)
        if token[0] == "x":
            return chr(int(token[1:], 16))
        if token[0].isdigit() and len(token) > 1:
            return chr(int(token.lstrip("0") or "0", 8) % 256)
        return simple.get(token, token)

    return re.sub(r"\\(x[0-9a-fA-F]{1,2}|0?[0-7]{1,3}|[ntr\\ae])", repl, text)


_PIPE_SOURCES = frozenset({"curl", "wget", "nc", "ncat", "netcat", "socat", "aria2c", "base64",
                           "base32", "xxd", "uudecode", "openssl", "gpg", "zcat", "gunzip", "rev"})


def _pipeline_risk(pipeline: list[_Cmd], ctx: _Ctx, heredocs: list[str]) -> Risk:
    if len(pipeline) == 1:
        return _command_risk(pipeline[0], ctx, heredocs)  # shares ctx so `cd` persists
    risk = SAFE
    upstream_names: set[str] = set()
    text: str | None = None
    for position, cmd in enumerate(pipeline):
        words = list(cmd.words)
        while words and _ASSIGNMENT.match(words[0]):
            words.pop(0)
        if position == 0:
            child = ctx.child()
        elif text is not None:
            child = ctx.child(stdin="text", stdin_text=text)
        else:
            child = ctx.child(stdin="pipe")
        command_risk = _command_risk(cmd, child, heredocs)
        name = _prog(words[0]) if words else ""
        if position and child.stdin == "pipe" and upstream_names & _PIPE_SOURCES and command_risk.level == "high" and (
            name in _SHELLS or name in _INTERPRETERS
        ):
            command_risk |= _risk("high", "runs code that was downloaded or decoded on the fly")
        risk |= command_risk
        text = _static_output(words, text)
        if name:
            upstream_names.add(name)
    return risk


def _classify(text: str, ctx: _Ctx) -> Risk:
    if ctx.depth > _MAX_DEPTH:
        return _risk("high", "nested shell code is too deep to review")
    if not text or not text.strip():
        return SAFE
    if _FORK_BOMB.search(text):
        return _risk("critical", "fork bomb")
    risk = SAFE
    if _ANSI_C_ESCAPE.search(text):
        risk |= _risk("high", "uses escaped characters to hide what runs")
    try:
        clean, subs, heredocs = _preprocess(text)
        pipelines = _pipelines(clean)
    except ValueError as exc:  # _ParseError or shlex's "No closing quotation"
        return risk | _risk("high", f"cannot be parsed safely ({exc})")
    for index, sub in enumerate(subs):
        risk |= _classify(sub, ctx.nested())
        value = _static_value(sub)
        if value is not None:  # `$(echo ~)` is just "~": judge the command on it
            pipelines = _pipelines(clean.replace(f"{_SUBST}{index}_", shlex.quote(value)))
            clean = clean.replace(f"{_SUBST}{index}_", shlex.quote(value))
    for pipeline in pipelines:
        risk |= _pipeline_risk(pipeline, ctx, heredocs)
    if {"secret", "net"} <= risk.flags and not ctx.remote:
        risk |= _risk("critical", "reads credential files and talks to the network (exfiltration)")
    return risk


def _static_value(sub: str) -> str | None:
    """Output of a substitution that is a single echo/printf of literals, else None."""
    try:
        pipelines = _pipelines(_preprocess(sub)[0])
    except ValueError:
        return None
    if len(pipelines) != 1 or len(pipelines[0]) != 1 or pipelines[0][0].redirects:
        return None
    value = _static_output(pipelines[0][0].words, None)
    return value.strip() if value is not None and "\n" not in value.strip() else None


def _workspace_root(workspace: Any) -> Path | None:
    if workspace is None:
        return None
    # A sandbox.Workspace exposes `.root`; plain paths are used as-is (Path.root is "/"!).
    raw = workspace if isinstance(workspace, (str, os.PathLike)) else workspace.root
    try:
        return Path(raw).expanduser().resolve()
    except (OSError, TypeError):
        return None


def classify_command(
    command: str, *, workspace: Any = None, cwd: str | Path | None = None, remote: bool = False
) -> Risk:
    """Grade a shell command line. `workspace` is a path or a sandbox.Workspace;
    `remote=True` grades it for a remote host (no local paths, sudo is high)."""
    if remote:
        ctx = _Ctx(workspace=None, cwd=None, home=_REMOTE_HOME, remote=True)
    else:
        root = _workspace_root(workspace)
        start = root
        if cwd:
            candidate = Path(str(cwd)).expanduser()
            start = (candidate if candidate.is_absolute() else (root or Path.cwd()) / candidate).resolve()
        ctx = _Ctx(workspace=root, cwd=start, home=Path.home().resolve())
    risk = _classify(command, ctx)
    return risk if risk.kind in ("exec", "write") else replace(risk, kind="exec")


# --------------------------------------------------------------------------
# Tool calls → risk


_EXECUTABLE_SUFFIXES = (".sh", ".bash", ".desktop", ".appimage", ".run", ".bin", ".exe", ".jar",
                        ".py", ".pl", ".rb", ".deb", ".rpm", ".snap", ".flatpakref", ".msi")
_READ_TOOLS = frozenset({"read_file", "read_files", "list_files", "glob_files", "grep_files",
                         "file_outline", "git", "system_info", "notify", "recall", "remember",
                         "task_board", "plan", "models", "doc_list", "doc_read", "doc_search"})
_WEB_TOOLS = frozenset({"web_search", "fetch_page", "multi_search", "fetch_pages", "web_research",
                        "browse_page"})


def _file_tool_risk(path: str, root: Path | None, *, verb: str, existing_level: str = "medium") -> Risk:
    ctx = _Ctx(workspace=root, cwd=root, home=Path.home().resolve())
    scope = _scope(path, ctx)
    if scope.kind not in _IN_WORKSPACE:
        return _risk(_WRITE_LEVEL[scope.kind], f"{verb} {_PLACE[scope.kind]} ({path})", "write", writes=1)
    exists = os.path.exists(scope.path)
    return _risk(existing_level if exists else "low", f"{verb} {'existing' if exists else 'new'} file {path}", "write", writes=1)


def assess(name: str, args: dict[str, Any], *, workspace: Any = None, spec_risk: str | None = None) -> Risk:
    """Risk of one (validated) tool call."""
    root = _workspace_root(workspace)
    if name == "run_command":
        return classify_command(str(args.get("command") or ""), workspace=root, cwd=args.get("cwd"))
    if name == "multi_command":
        risk = Risk(kind="exec")
        for command in args.get("commands") or []:
            risk |= classify_command(str(command), workspace=root, cwd=args.get("cwd"))
        return risk
    if name == "terminal_session":
        action = args.get("action")
        text = args.get("command") if action in ("start", "run") else args.get("input") if action == "input" else None
        return classify_command(str(text), workspace=root, cwd=args.get("cwd")) if text else Risk(kind="exec")
    if name == "remote":
        host, action = args.get("host") or "?", args.get("action") or "run"
        if action == "hosts":
            return SAFE
        if action == "upload":
            return _risk("medium", f"copies {args.get('path')} to {host}:{args.get('remote_path')}")
        if action == "download":
            return _file_tool_risk(str(args.get("path") or ""), root, verb=f"downloads from {host} into")
        command = str(args.get("command") or "")
        return classify_command(command, remote=True) | _risk("low", f"runs on the remote host {host}")
    if name in ("run_python", "run_javascript"):
        if sandbox_status()["active"]:
            return _risk("low", "runs code in an isolated sandbox (no network, writes only its scratch folder)")
        language = "python" if name == "run_python" else "node"
        return _risk("medium", "runs code unconfined (no sandbox available)") | _code_risk(
            str(args.get("code") or ""), language, inline=True
        )
    if name == "write_file":
        return _file_tool_risk(str(args.get("path") or ""), root, verb="writes")
    if name in ("replace_in_file", "apply_patch"):
        return _file_tool_risk(str(args.get("path") or ""), root, verb="edits")
    if name == "file_manage":
        ctx = _Ctx(workspace=root, cwd=root, home=Path.home().resolve())
        path, action = str(args.get("path") or ""), args.get("action")
        if action == "delete":
            recursive = bool(args.get("recursive")) and os.path.isdir(_scope(path, ctx).path)
            return _delete_risk([path], ctx, recursive=recursive, verb="file_manage")
        if action == "move":
            return _delete_risk([path], ctx, recursive=False, verb="file_manage move") | _file_tool_risk(
                str(args.get("destination") or ""), root, verb="moves into"
            )
        if action == "copy":
            return _file_tool_risk(str(args.get("destination") or ""), root, verb="copies into")
        return _file_tool_risk(path, root, verb="creates", existing_level="low")
    if name == "open_path":
        target = str(args.get("target") or "")
        if re.match(r"^https?://", target, re.I):
            return _risk("low", "opens a web page in your browser")
        if target.lower().endswith(_EXECUTABLE_SUFFIXES):
            return _risk("high", f"opening {target} may run it as a program")
        return _risk("low", f"opens {target} in its default application")
    if name == "workspace":
        if args.get("action") != "switch":
            return SAFE
        return _risk("low", f"switches this conversation's working folder to {args.get('path')}", "write")
    if name == "clipboard":
        if args.get("action") == "read":
            return _risk("medium", "reads your clipboard (may hold private data)", "read")
        return _risk("low", "replaces your clipboard contents", "write")
    if name == "screenshot":
        return _risk("medium", "captures your screen", "read")
    if name in _READ_TOOLS:
        return SAFE
    if name in _WEB_TOOLS:
        return Risk("low", (), "web")
    kind = spec_risk if spec_risk in _KINDS else "exec"
    level = {"read": "safe", "web": "low"}.get(kind, "medium")
    return Risk(level, () if level == "safe" else (f"`{name}` is a {kind} tool",), kind)


# --------------------------------------------------------------------------
# Policy


def permission(settings: dict[str, Any], name: str) -> str:
    """The user's per-tool permission (the single lookup; tools.permission_mode delegates here)."""
    mode = (settings.get("permissions") or {}).get(name)
    return mode if mode in config.PERMISSION_MODES else config.permission_defaults().get(name, "ask")


def _remote_host_known(host: str, settings: dict[str, Any]) -> bool:
    from .remote import host_known  # lazy: remote imports the sandbox, which imports this module

    return host_known(host, settings)


def _guard_verdict(name: str, args: dict[str, Any], turn_state: dict[str, Any]) -> Any:
    try:
        from . import guard  # optional content-safety layer
    except ImportError:
        return None
    try:
        return guard.inspect_call(name, args, turn_state)
    except Exception:  # the guard must never break a turn
        log.warning("guard.inspect_call failed for %s", name, exc_info=True)
        return None


def _policy_mode(perm: str, level: str, autonomy: str, kind: str, *, remote: bool = False) -> str:
    if autonomy == "supervised":
        return "ask" if perm == "auto" and level == "high" else perm
    if level == "high":
        return "ask" if autonomy == "smart" else "deny"
    if perm == "auto" or autonomy == "autonomous" or level in ("safe", "low"):
        return "auto"
    # smart + "ask" + medium: auto-approve while the blast radius is bounded — never
    # on a remote server (often production; no local sandbox can contain it).
    return "auto" if not remote and (kind != "exec" or sandbox_status()["active"]) else "ask"


def decide(
    settings: dict[str, Any] | None,
    name: str,
    args: dict[str, Any] | None,
    *,
    workspace: Any = None,
    turn_state: dict[str, Any] | None = None,
    risk: str | None = None,
) -> Decision:
    """auto / ask / deny for one validated tool call, with the reasons why."""
    settings = settings or {}
    args = args or {}
    configure(settings)
    perm = permission(settings, name)
    if perm == "deny":
        return Decision("deny", "safe", ("disabled in Settings → Permissions",), "permission")
    try:
        assessed = assess(name, args, workspace=workspace, spec_risk=risk)
    except Exception:  # an analyser bug must fail closed, never open
        log.exception("safety assessment failed for %s", name)
        assessed = _risk("high", "the safety check could not analyse this call")
    state = turn_state if turn_state is not None else {}
    counters = state.setdefault("counters", {"writes": 0, "deletes": 0})
    level, reasons = assessed.level, list(assessed.reasons)
    if level != "critical":
        deletes = counters.get("deletes", 0) + assessed.deletes
        writes = counters.get("writes", 0) + assessed.writes
        if assessed.deletes and deletes > MASS_DELETE_LIMIT:
            level = _bump(level, (deletes - 1) // MASS_DELETE_LIMIT)
            reasons.append(f"mass removal: {deletes} deletions this turn")
        if assessed.writes and writes > MASS_WRITE_LIMIT:
            level = _bump(level, (writes - 1) // MASS_WRITE_LIMIT)
            reasons.append(f"mass edit: {writes} file changes this turn")
        if name == "remote" and args.get("action") != "hosts" and not _remote_host_known(str(args.get("host") or ""), settings):
            level = "high"
            reasons.append(f"{args.get('host')} is not a configured host (add it to ~/.ssh/config or Settings)")
        if name == "workspace" and args.get("action") == "switch" and not _mentioned(
            str(args.get("path") or ""), str(state.get("user_message") or "")
        ):
            level = "high"
            reasons.append("the user's message does not mention this folder")
        if state.get("tainted") and assessed.kind in ("exec", "write") and level != "safe":
            level = _bump(level)
            reasons.append("content read earlier contained instructions aimed at the AI (possible prompt injection)")
    autonomy = settings.get("autonomy") if settings.get("autonomy") in AUTONOMY_MODES else DEFAULT_AUTONOMY
    mode = "deny" if level == "critical" else _policy_mode(perm, level, autonomy, assessed.kind, remote=name == "remote")
    source = "permission" if mode == perm else "policy"
    verdict = _guard_verdict(name, args, state)
    if verdict is not None and getattr(verdict, "mode", None) in _MODE_RANK:
        if _MODE_RANK[verdict.mode] > _MODE_RANK[mode]:
            mode, source = verdict.mode, "guard"
        if getattr(verdict, "level", None) in _RANK and _RANK[verdict.level] > _RANK[level]:
            level = verdict.level
        reasons.extend(r for r in getattr(verdict, "reasons", ()) if r not in reasons)
    if autonomy == "autonomous" and mode == "ask":
        mode = "deny"
        reasons.append("autonomous mode: nobody is watching to approve it")
    if mode != "deny":
        counters["writes"] = counters.get("writes", 0) + assessed.writes
        counters["deletes"] = counters.get("deletes", 0) + assessed.deletes
        counters["actions"] = counters.get("actions", 0) + (assessed.kind in ("write", "exec"))
    return Decision(mode, level, tuple(reasons[:6]), source)


# --------------------------------------------------------------------------
# Confinement


_CONFIG = {"sandbox": os.environ.get("KIWIBRO_SANDBOX", DEFAULT_SANDBOX)}
_UNCONFINED: contextvars.ContextVar[bool] = contextvars.ContextVar("kiwibro_unconfined", default=False)


def configure(settings: dict[str, Any]) -> None:
    """Apply the `sandbox` setting (decide() does this before every tool call)."""
    if settings.get("sandbox") in SANDBOX_MODES:
        _CONFIG["sandbox"] = settings["sandbox"]


def is_unconfined() -> bool:
    """True inside `applied(...)` for a high-risk call the user explicitly approved."""
    return _UNCONFINED.get()


def tool_home() -> Path:
    """HOME for confined commands: a Kiwibro-owned folder in the XDG cache, so npm/pip/
    cargo caches and tool configs never litter the user's projects (and stay warm
    across runs). Writable inside the sandbox; nothing else in $HOME is."""
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache").expanduser()
    path = base / "kiwibro" / "home"
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextlib.contextmanager
def applied(decision: Decision | None, *, approved_by_user: bool = False) -> Iterator[None]:
    """Run a tool call under its decision. A high-risk call that the user
    explicitly approved runs unconfined (e.g. `git push` needs ~/.ssh)."""
    unconfined = bool(approved_by_user and decision and _RANK[decision.level] >= _RANK["high"])
    token = _UNCONFINED.set(unconfined)
    try:
        yield
    finally:
        _UNCONFINED.reset(token)


def _probe(argv: list[str]) -> bool:
    try:
        return subprocess.run(argv, capture_output=True, timeout=5).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _bus() -> dict[str, str]:
    """The user-bus variables systemd-run needs (workspace commands get a minimal env)."""
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return {
        "XDG_RUNTIME_DIR": runtime,
        "DBUS_SESSION_BUS_ADDRESS": os.environ.get("DBUS_SESSION_BUS_ADDRESS") or f"unix:path={runtime}/bus",
    }


@functools.lru_cache(maxsize=1)
def _engines() -> tuple[str | None, str | None]:
    """(bwrap, systemd-run) paths for engines that actually work on this machine."""
    bwrap = shutil.which("bwrap")
    if bwrap and not _probe([bwrap, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
                             "--unshare-pid", "--die-with-parent", "--new-session", "--", "/bin/true"]):
        bwrap = None
    scope = shutil.which("systemd-run")
    if scope and not _probe(["env", "-i", *(f"{k}={v}" for k, v in _bus().items()), scope, "--user",
                             "--scope", "--quiet", "--collect", "-p", "TasksMax=64", "--", "/bin/true"]):
        scope = None
    return bwrap, scope


def sandbox_status() -> dict[str, Any]:
    mode = _CONFIG["sandbox"]
    bwrap, scope = _engines() if mode != "off" else (None, None)
    return {
        "mode": mode,
        "active": bool(bwrap),
        "engine": "bubblewrap" if bwrap else None,
        "resource_limits": bool(bwrap and scope),
        "network": mode != "strict",
    }


@functools.lru_cache(maxsize=1)
def _memory_limit() -> int:
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        total = 8 * 1024**3
    return min(int(total * 0.4), 8 * 1024**3)


def _hidden_paths(writable: list[Path]) -> tuple[list[str], list[str]]:
    """(directories to cover with tmpfs, files to cover with /dev/null), symlinks resolved."""
    home = Path.home().resolve()
    candidates = [home / rel for rel in SECRET_PATHS] + [Path(f"/run/user/{os.getuid()}")]
    dirs: set[str] = set()
    files: set[str] = set()
    for path in candidates:  # runtime dir: D-Bus, systemd --user, Wayland, keyring sockets
        real = Path(os.path.realpath(path))
        if any(w == real or w.is_relative_to(real) for w in writable):
            continue
        if real.is_dir():
            dirs.add(str(real))
        elif real.is_file():
            files.add(str(real))
    for sock in ("/run/docker.sock", "/var/run/docker.sock", "/run/podman/podman.sock"):
        if os.path.exists(sock):
            files.add(os.path.realpath(sock))
    return sorted(dirs), sorted(files)


def _bwrap_argv(
    argv: list[str],
    *,
    writable: list[Path],
    cwd: Path,
    network: bool,
    hide: bool,
    readable: tuple[Path, ...] = (),
    unset: tuple[str, ...] = (),
) -> list[str]:
    wrapped = [
        str(_engines()[0]), "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
        "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup-try",
        "--die-with-parent", "--new-session",
    ]
    if not network:
        wrapped.append("--unshare-net")
    if hide:
        dirs, files = _hidden_paths(writable)
        for directory in dirs:
            wrapped += ["--tmpfs", directory]
        for path in files:
            wrapped += ["--ro-bind", "/dev/null", path]
    for path in readable:  # e.g. a workspace under the private /tmp
        wrapped += ["--ro-bind", str(path), str(path)]
    for path in writable:
        wrapped += ["--bind", str(path), str(path)]
    for key in unset:
        wrapped += ["--unsetenv", key]
    return [*wrapped, "--chdir", str(cwd), "--", *argv]


@functools.lru_cache(maxsize=32)
def _hiding_works(root: str) -> bool:
    """Verify once per workspace that the full mount set is valid here, so an
    unusual layout degrades to minimal confinement instead of breaking commands."""
    argv = _bwrap_argv(["/bin/true"], writable=[Path(root)], cwd=Path(root), network=True, hide=True)
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("sandbox self-test failed (%s); using minimal confinement", exc)
        return False
    if result.returncode != 0:
        log.warning("sandbox self-test failed (%s); using minimal confinement", result.stderr.strip()[:200])
    return result.returncode == 0


def confine(
    argv: list[str],
    *,
    writable: list[Path],
    cwd: Path,
    readable: tuple[Path, ...] = (),
    network: bool | None = None,
    memory_bytes: int | None = None,
    tasks: int = 512,
) -> list[str]:
    """`argv` wrapped in the strongest confinement available (or unchanged).
    `writable` paths are bound read-write, `readable` ones read-only."""
    status = sandbox_status()
    if not status["active"] or _UNCONFINED.get():
        return argv
    scope = _engines()[1]
    bus = _bus()
    socket = re.search(r"unix:path=([^,;]+)", bus["DBUS_SESSION_BUS_ADDRESS"])
    use_scope = bool(scope and socket and os.path.exists(socket.group(1)))
    wrapped = _bwrap_argv(
        argv,
        writable=writable,
        cwd=cwd,
        network=status["network"] if network is None else network,
        hide=all(_hiding_works(str(path)) for path in writable),
        readable=readable,
        unset=tuple(bus) if use_scope else (),  # systemd-run gets the bus; the command does not
    )
    if use_scope:
        wrapped = ["env", *(f"{k}={v}" for k, v in bus.items()), str(scope), "--user", "--scope",
                   "--quiet", "--collect", "-p", f"MemoryMax={memory_bytes or _memory_limit()}",
                   "-p", "MemorySwapMax=0", "-p", f"TasksMax={tasks}", "--", *wrapped]
    return wrapped
