"""Paths, constants and settings values for Kiwibro."""

from __future__ import annotations

import functools
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION = "2.5.0"
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
MEMORY_EXTRACTION_TIMEOUT_S = 8.0  # floor; the real budget follows measured speed (inference.SPEEDS)
# Background memory extraction waits until no reply has streamed for this long. Cancelling
# it frees the CPU only at LM Studio's next 2048-token batch (~20 s on a laptop CPU), so it
# starts in real pauses rather than between quick replies (the remember tool saves at once).
MEMORY_IDLE_S = float(os.environ.get("KIWIBRO_MEMORY_IDLE_S", "20"))
MEMORY_CAP = 400

MAX_MESSAGE_CHARS = 400_000  # longer than the window is fine: the agent pages it into the library
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

# File changes per turn before safety.decide escalates each further one a level
# (and again at every multiple): a runaway edit spree gets stopped (ask / deny in
# autonomous mode) while a legitimate big job, like scaffolding a project, finishes.
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

# Where models run: "lmstudio" calls LM Studio's API; "direct" runs LM Studio's downloaded
# models with llama.cpp itself (see engine.py), so the LM Studio app need not be open.
ENGINES = ("lmstudio", "direct")
DEFAULT_ENGINE = "lmstudio"
DIRECT_THREADS_RANGE = (0, 256)  # 0 = every physical core
# Direct engine graphics acceleration: auto = a GPU build of llama.cpp on the GPU it finds
# (an integrated one too), off = the CPU only.
GPU_MODES = ("auto", "off")
DEFAULT_GPU = "auto"

# Settings whose value must be one of a fixed set.
SETTING_CHOICES: dict[str, tuple[str, ...]] = {
    "engine": ENGINES,
    "direct_gpu": GPU_MODES,
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


def display_path(path: str) -> str:
    """A path the way people (and small models) read it best: `~/Kiwibro`, not /home/user/Kiwibro."""
    home = str(Path.home())
    return "~" + path[len(home):] if path == home or path.startswith(home + "/") else path


def default_workspace() -> Path:
    return Path.home() / "Kiwibro"


def default_settings(lmstudio_url: str | None = None, workspace: str | None = None) -> dict[str, Any]:
    """Fresh settings values (permissions are merged separately)."""
    url = (lmstudio_url or os.environ.get("KIWIBRO_LMSTUDIO_URL") or DEFAULT_LMSTUDIO_URL).rstrip("/")
    ws = workspace or os.environ.get("KIWIBRO_WORKSPACE") or str(default_workspace())
    return {
        "model": None,
        "lmstudio_url": url,
        "lmstudio_token": "",
        "engine": os.environ.get("KIWIBRO_ENGINE") if os.environ.get("KIWIBRO_ENGINE") in ENGINES else DEFAULT_ENGINE,
        "direct_runtime": "",
        "direct_models_dir": "",
        "direct_threads": 0,
        "direct_gpu": DEFAULT_GPU,
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


def _validate_optional_path(value: Any, key: str, *, executable: bool) -> str:
    """'' (automatic) or an absolute path to an existing executable file / folder."""
    if not isinstance(value, str) or len(value) > 4096:
        raise ValueError(f"{key} must be a path")
    raw = value.strip()
    if not raw:
        return ""
    path = Path(os.path.expanduser(raw))
    if not path.is_absolute():
        raise ValueError(f"{key} must be an absolute path")
    if executable and not (path.is_file() and os.access(path, os.X_OK)):
        raise ValueError(f"{key} must be an executable llama-server file")
    if not executable and not path.is_dir():
        raise ValueError(f"{key} must be an existing folder")
    return str(path)


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
        elif key == "direct_runtime":
            # A custom llama.cpp build's llama-server; empty = LM Studio's bundled engine.
            cleaned["direct_runtime"] = _validate_optional_path(value, key, executable=True)
        elif key == "direct_models_dir":
            cleaned["direct_models_dir"] = _validate_optional_path(value, key, executable=False)
        elif key == "direct_threads":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("direct_threads must be an integer")
            if not DIRECT_THREADS_RANGE[0] <= value <= DIRECT_THREADS_RANGE[1]:
                raise ValueError("direct_threads must be between 0 and 256")
            cleaned["direct_threads"] = value
        elif key == "lmstudio_token":
            # API token for an LM Studio server with authentication on (another machine's).
            if not isinstance(value, str) or len(value) > 512 or any(c.isspace() for c in value.strip()):
                raise ValueError("lmstudio_token must be a single token of at most 512 characters")
            cleaned["lmstudio_token"] = value.strip()
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
