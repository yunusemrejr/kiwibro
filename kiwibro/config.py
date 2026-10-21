"""Paths, constants and settings values for Kiwibro."""

from __future__ import annotations

import functools
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION = "2.0.0"
APP_NAME = "Kiwibro"
# Unix timestamp of process start: lets the desktop launcher notice when the repo
# changed underneath a still-running server and restart it.
STARTED_AT = int(time.time())

DEFAULT_LMSTUDIO_URL = "http://127.0.0.1:1234"
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.6
MAX_OUTPUT_TOKENS_RANGE = (256, 8192)
TEMPERATURE_RANGE = (0.0, 2.0)

MAX_TOOL_ROUNDS = 16
MAX_PARALLEL_TOOLS = 6
APPROVAL_TIMEOUT_S = 120.0
HEARTBEAT_S = 15.0
MEMORY_EXTRACTION_TIMEOUT_S = 8.0
MEMORY_CAP = 400

MAX_MESSAGE_CHARS = 32000
MAX_FILE_READ_BYTES = 200 * 1024
MAX_FILE_WRITE_BYTES = 256 * 1024
MAX_COMMAND_OUTPUT_BYTES = 6000

PERMISSION_MODES = ("auto", "ask", "deny")

DEFAULT_PERMISSIONS: dict[str, str] = {
    "web_search": "auto",
    "fetch_page": "auto",
    "multi_search": "auto",
    "fetch_pages": "auto",
    "web_research": "auto",
    "browse_page": "auto",
    "read_file": "auto",
    "read_files": "auto",
    "list_files": "auto",
    "glob_files": "auto",
    "grep_files": "auto",
    "file_outline": "auto",
    "git": "auto",
    "write_file": "ask",
    "replace_in_file": "ask",
    "apply_patch": "ask",
    "file_manage": "ask",
    "run_command": "ask",
    "multi_command": "ask",
    "terminal_session": "ask",
    "system_info": "auto",
    "open_path": "ask",
    "clipboard": "ask",
    "notify": "auto",
    "screenshot": "ask",
    "browser": "auto",
    "model_info": "auto",
    "delegate": "auto",
    "describe_image": "auto",
    "study_file": "auto",
    "run_python": "ask",
    "run_javascript": "ask",
    "task_board": "auto",
    "recall": "auto",
    "remember": "auto",
}

# Per-turn ceiling on write/exec-class tool executions: small models get a
# hard stop instead of an unbounded edit/run spree in one autonomous turn.
MAX_WRITES_PER_TURN = 30

@functools.lru_cache(maxsize=1)
def _extension_permissions() -> tuple[tuple[str, str], ...]:
    try:
        from .extensions import default_permissions  # lazy: extensions import tool modules
    except ImportError:
        return ()
    return tuple(default_permissions().items())


def permission_defaults() -> dict[str, str]:
    """Every tool's default permission: built-ins plus each extension tool's own."""
    return {**dict(_extension_permissions()), **DEFAULT_PERMISSIONS}


# Tool profiles keep small local models effective: instead of sending every
# tool schema each turn, the agent sends a task-shaped subset (see tools.py).
TOOL_PROFILES = ("auto", "full", "minimal", "coding", "research", "desktop")
DEFAULT_TOOL_PROFILE = "auto"
TOOL_ROUNDS_RANGE = (1, 32)

# Thinking: "auto" thinks only where it pays off (see inference.wants_reasoning).
REASONING_MODES = ("auto", "on", "off")
DEFAULT_REASONING = "auto"

# How much a turn may do without a human (see safety.decide): supervised asks per
# the permission table; smart auto-approves low-risk work and asks for high risk;
# autonomous never waits and denies high risk with a reason the model can act on.
AUTONOMY_MODES = ("supervised", "smart", "autonomous")
DEFAULT_AUTONOMY = "smart"
# Command confinement (see safety.confine): auto = bubblewrap when available,
# strict = also no network, off = run commands unconfined.
SANDBOX_MODES = ("auto", "strict", "off")
DEFAULT_SANDBOX = "auto"

# Settings whose value must be one of a fixed set.
SETTING_CHOICES: dict[str, tuple[str, ...]] = {
    "tool_profile": TOOL_PROFILES,
    "reasoning": REASONING_MODES,
    "autonomy": AUTONOMY_MODES,
    "sandbox": SANDBOX_MODES,
}

MEMORY_KINDS = ("fact", "preference", "project", "person", "decision")


def data_dir() -> Path:
    """Directory for the SQLite DB and logs (override with KIWIBRO_DATA_DIR)."""
    raw = os.environ.get("KIWIBRO_DATA_DIR", "~/.local/share/kiwibro")
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def database_path(directory: Path | None = None) -> Path:
    base = directory or data_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / "kiwibro.db"


def default_workspace() -> Path:
    return Path.home() / "Kiwibro"


def default_settings(lmstudio_url: str | None = None, workspace: str | None = None) -> dict[str, Any]:
    """Fresh settings values (permissions are merged separately)."""
    url = (lmstudio_url or os.environ.get("KIWIBRO_LMSTUDIO_URL") or DEFAULT_LMSTUDIO_URL).rstrip("/")
    ws = workspace or os.environ.get("KIWIBRO_WORKSPACE") or str(default_workspace())
    return {
        "model": None,
        "lmstudio_url": url,
        "agent_enabled": True,
        "memory_enabled": True,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "workspace": str(Path(ws).expanduser()),
        "user_name": "",
        "tool_profile": DEFAULT_TOOL_PROFILE,
        "reasoning": DEFAULT_REASONING,
        "autonomy": DEFAULT_AUTONOMY,
        "sandbox": DEFAULT_SANDBOX,
        "remote_hosts": [],
        "max_tool_rounds": MAX_TOOL_ROUNDS,
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_workspace(value: Any) -> str:
    """Validate that a workspace path is creatable inside $HOME. Returns resolved path."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("workspace must be a non-empty path")
    raw = value.strip()
    if not os.path.isabs(os.path.expanduser(raw)):
        raise ValueError("workspace must be an absolute path")
    target = Path(os.path.expanduser(raw)).resolve()
    home = Path.home().resolve()
    if not target.is_relative_to(home):
        raise ValueError("workspace must be inside your home directory")
    if target == Path(home) or target == Path("/"):
        raise ValueError("workspace must be a directory inside your home directory")
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"workspace is not creatable: {exc}") from exc
    if not target.is_dir():
        raise ValueError("workspace path is not a directory")
    return str(target)


def validate_lmstudio_url(value: Any) -> str:
    from urllib.parse import urlsplit

    if not isinstance(value, str):
        raise ValueError("lmstudio_url must be a string")
    url = value.strip().rstrip("/")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("lmstudio_url must be an http(s) URL")
    return url


def validate_settings_patch(patch: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Validate a settings PATCH body; returns the normalised values to persist."""
    cleaned: dict[str, Any] = {}
    for key, value in patch.items():
        if key == "permissions":
            if not isinstance(value, dict):
                raise ValueError("permissions must be an object")
            perms: dict[str, str] = {}
            for tool, mode in value.items():
                if tool not in permission_defaults():
                    raise ValueError(f"unknown tool: {tool}")
                if mode not in PERMISSION_MODES:
                    raise ValueError(f"invalid permission mode for {tool}")
                perms[tool] = mode
            cleaned["permissions"] = perms
        elif key == "temperature":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("temperature must be a number")
            temp = float(value)
            if not TEMPERATURE_RANGE[0] <= temp <= TEMPERATURE_RANGE[1]:
                raise ValueError("temperature must be between 0 and 2")
            cleaned["temperature"] = round(temp, 3)
        elif key == "max_output_tokens":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("max_output_tokens must be an integer")
            if not MAX_OUTPUT_TOKENS_RANGE[0] <= value <= MAX_OUTPUT_TOKENS_RANGE[1]:
                raise ValueError("max_output_tokens must be between 256 and 8192")
            cleaned["max_output_tokens"] = value
        elif key == "workspace":
            cleaned["workspace"] = validate_workspace(value)
        elif key == "lmstudio_url":
            cleaned["lmstudio_url"] = validate_lmstudio_url(value)
        elif key == "model":
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("model must be a non-empty string or null")
            cleaned["model"] = value.strip() if isinstance(value, str) else None
        elif key in ("agent_enabled", "memory_enabled"):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
            cleaned[key] = value
        elif key == "user_name":
            if not isinstance(value, str) or len(value) > 80:
                raise ValueError("user_name must be a string of at most 80 characters")
            cleaned["user_name"] = value.strip()
        elif key == "remote_hosts":
            if not isinstance(value, list) or len(value) > 32 or not all(
                isinstance(h, str) and re.fullmatch(r"(?:[A-Za-z0-9._-]{1,64}@)?[A-Za-z0-9][A-Za-z0-9._-]{0,252}", h)
                for h in value
            ):
                raise ValueError("remote_hosts must be a list of up to 32 ssh aliases or user@host names")
            cleaned["remote_hosts"] = list(dict.fromkeys(value))
        elif key in SETTING_CHOICES:
            if value not in SETTING_CHOICES[key]:
                raise ValueError(f"{key} must be one of {', '.join(SETTING_CHOICES[key])}")
            cleaned[key] = value
        elif key == "max_tool_rounds":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("max_tool_rounds must be an integer")
            if not TOOL_ROUNDS_RANGE[0] <= value <= TOOL_ROUNDS_RANGE[1]:
                raise ValueError(
                    f"max_tool_rounds must be between {TOOL_ROUNDS_RANGE[0]} and {TOOL_ROUNDS_RANGE[1]}"
                )
            cleaned["max_tool_rounds"] = value
        else:
            raise ValueError(f"unknown setting: {key}")
    return cleaned
