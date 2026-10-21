"""Remote hosts over SSH: run commands and move files on servers the user chose.

The ``remote`` tool uses the system ``ssh``/``scp`` with the user's own keys,
agent and ``~/.ssh/config``. BatchMode means it never prompts (an unknown host
key or a missing key fails fast with a clear message); connect and command
timeouts and capped output keep long sessions responsive. Local paths stay
inside the workspace.

Safety (see safety.decide): hosts from ``~/.ssh/config`` or the
``remote_hosts`` setting are trusted; any other host is graded high (approval
in smart mode, refused in autonomous mode). Commands are graded for a server
(``safety.classify_command(remote=True)``): reading logs is low, deploy-style
changes are medium, ``sudo``/service restarts are high, wiping ``/`` or ``~``
is critical. Remote work runs outside the local sandbox by necessity: the
sandbox hides ``~/.ssh``.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

from .extensions import ExtensionTool, preview_text, schema, tool_result
from .sandbox import MAX_OUTPUT_BYTES, SandboxError, Workspace, run_process

ACTIONS = ("run", "upload", "download", "hosts")
CONNECT_TIMEOUT_S = 10
DEFAULT_TIMEOUT_S = 60
MAX_TIMEOUT_S = 300
# user@host, host:port-free aliases; never starting with "-" (no option injection).
_HOST = re.compile(r"^(?:[A-Za-z0-9._-]{1,64}@)?[A-Za-z0-9][A-Za-z0-9._-]{0,252}$")
_SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", f"ConnectTimeout={CONNECT_TIMEOUT_S}",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2")


def config_hosts(path: Path | None = None) -> list[str]:
    """Concrete Host aliases from ~/.ssh/config (wildcard patterns skipped)."""
    path = path or Path.home() / ".ssh" / "config"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    hosts: list[str] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].lower() == "host":
            hosts += [h for h in parts[1:] if not any(ch in h for ch in "*?!") and h not in hosts]
    return hosts


def host_known(host: str, settings: dict[str, Any]) -> bool:
    """A host the user configured: in ~/.ssh/config or Settings → remote hosts."""
    bare = host.split("@", 1)[-1]
    allowed = set(settings.get("remote_hosts") or []) | set(config_hosts())
    return host in allowed or bare in allowed


def _env() -> dict[str, str]:
    """ssh needs the user's HOME and agent socket; nothing else leaks through."""
    keep = ("HOME", "USER", "LOGNAME", "PATH", "SSH_AUTH_SOCK", "LANG", "LC_ALL")
    return {key: os.environ[key] for key in keep if key in os.environ}


def _remote_path(value: Any) -> str:
    path = str(value or "").strip()
    if not path or path.startswith("-") or "\x00" in path or "\n" in path:
        raise ValueError("remote_path must be a non-empty path on the host")
    return path


def validate(args: dict[str, Any]) -> dict[str, Any]:
    action = args.get("action") or ("run" if args.get("command") else "hosts")
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {', '.join(ACTIONS)}")
    if action == "hosts":
        return {"action": action}
    host = str(args.get("host") or "").strip()
    if not _HOST.match(host):
        raise ValueError("host must be an ssh alias or user@host (letters, digits, . _ -)")
    timeout = args.get("timeout", DEFAULT_TIMEOUT_S)
    if isinstance(timeout, bool) or not isinstance(timeout, int):
        timeout = DEFAULT_TIMEOUT_S
    cleaned: dict[str, Any] = {"action": action, "host": host, "timeout": max(1, min(timeout, MAX_TIMEOUT_S))}
    if action == "run":
        command = str(args.get("command") or "").strip()
        if not command or len(command) > 4000:
            raise ValueError("command must be a non-empty shell command (at most 4000 characters)")
        cleaned["command"] = command
    else:
        local = str(args.get("path") or "").strip()
        if not local:
            raise ValueError("path (the workspace file) is required")
        cleaned.update(path=local, remote_path=_remote_path(args.get("remote_path")))
    return cleaned


def _clip(output: str) -> str:
    output = Workspace._truncate_output(output.strip(), MAX_OUTPUT_BYTES)[0]
    return output or "(no output)"


def _diagnose(output: str) -> str:
    """Turn ssh's failure modes into one actionable line for the model and the user."""
    text = output.lower()
    if "host key verification failed" in text or "no ecdsa host key" in text:
        return "\n[Kiwibro: this host's key is not trusted yet. Ask the user to run `ssh <host>` once in a terminal.]"
    if "permission denied" in text and "publickey" in text:
        return "\n[Kiwibro: no SSH key is accepted by this host. Ask the user to set up key login (ssh-copy-id).]"
    if "could not resolve hostname" in text or "connection timed out" in text or "connection refused" in text:
        return "\n[Kiwibro: the host cannot be reached. Check the host name, VPN or that the server is running.]"
    return ""


def execute(args: dict[str, Any], ctx: Any) -> Any:
    action = args["action"]
    if action == "hosts":
        hosts = config_hosts()
        body = "\n".join(f"- {h}" for h in hosts) or "(no hosts in ~/.ssh/config)"
        return tool_result(True, f"SSH hosts from ~/.ssh/config:\n{body}", f"{len(hosts)} host(s)")
    if not shutil.which("ssh"):
        return tool_result(False, "Error: the ssh client is not installed (sudo apt install openssh-client).")
    host = args["host"]
    workspace: Workspace = ctx.workspace
    try:
        if action == "run":
            argv = ["ssh", "-T", *_SSH_OPTIONS, host, "--", args["command"]]
            label = f"$ {host}: {args['command']}"
        elif action == "upload":
            local = workspace.resolve(args["path"], must_exist=True)
            if not local.is_file():
                raise SandboxError(f"not a file: {workspace.public(local)}")
            argv = ["scp", "-B", "-q", *_SSH_OPTIONS, "--", str(local), f"{host}:{args['remote_path']}"]
            label = f"upload {workspace.public(local)} → {host}:{args['remote_path']}"
        else:
            local = workspace.resolve(args["path"])
            local.parent.mkdir(parents=True, exist_ok=True)
            argv = ["scp", "-B", "-q", *_SSH_OPTIONS, "--", f"{host}:{args['remote_path']}", str(local)]
            label = f"download {host}:{args['remote_path']} → {workspace.public(local)}"
        code, output = run_process(argv, cwd=workspace.root, env=_env(), timeout=args["timeout"])
    except (SandboxError, OSError) as exc:
        return tool_result(False, f"Error: {exc}")
    if code is None:
        return tool_result(False, f"{label}\nTimed out after {args['timeout']}s.\n{_clip(output)}", "Timed out")
    content = f"{label}\n(exit {code})\n{_clip(output)}{_diagnose(output) if code else ''}"
    return tool_result(code == 0, content, preview_text(content))


def _summary(args: dict[str, Any]) -> str:
    action = args.get("action") or "run"
    if action == "run":
        return f"On {args.get('host', '?')}: {str(args.get('command', ''))[:120]}"
    if action == "hosts":
        return "List SSH hosts"
    return f"{action.title()} {args.get('path', '')} ↔ {args.get('host', '?')}:{args.get('remote_path', '')}"


def _touches(args: dict[str, Any], ctx: Any) -> list[Path]:
    if args.get("action") != "download":
        return []
    try:
        return [ctx.workspace.resolve(args["path"])]
    except (SandboxError, KeyError):
        return []


REMOTE = ExtensionTool(
    name="remote",
    description=(
        "Work on a remote server over SSH using the user's own keys and ~/.ssh/config hosts. "
        "action=run executes a shell command on the host; upload/download copy one file between "
        "the workspace and the host; hosts lists configured hosts. Prefer read-only commands first "
        "(ls, cat, tail logs, systemctl status) and change things only when asked."
    ),
    parameters=schema(
        {
            "action": {"type": "string", "enum": list(ACTIONS), "description": "run | upload | download | hosts"},
            "host": {"type": "string", "description": "ssh alias from ~/.ssh/config, or user@host"},
            "command": {"type": "string", "description": "shell command to run on the host (action=run)"},
            "path": {"type": "string", "description": "workspace file (upload source / download destination)"},
            "remote_path": {"type": "string", "description": "path on the host (upload/download)"},
            "timeout": {"type": "integer", "description": f"seconds, 1-{MAX_TIMEOUT_S} (default {DEFAULT_TIMEOUT_S})"},
        },
        ["action"],
    ),
    risk="exec",
    phase="running",
    label="Remote",
    summary=_summary,
    validate=validate,
    execute=execute,
    permission="ask",
    category="coding",
    touches=_touches,
)

TOOLS = (REMOTE,)
