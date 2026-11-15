"""Showing what Kiwibro made: workspace files served to sandboxed previews, and the folder badge.

Every file goes through Workspace.resolve (nothing outside the chat's folder, no symlink out of
it), a type allow-list and a size cap. An HTML page is served with a Content-Security-Policy that
sandboxes it (scripts only, an opaque origin, so it cannot reach this app's cookies or API) and
lets it load files from its own folder tree, plus scripts and styles from four public libraries
(SCRIPT_HOSTS: a three.js page names one in a script tag). Nothing else reaches the network.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from . import config, safety
from .sandbox import SandboxError, Workspace

# suffix -> (content type, kind). Kinds shown in a chat: html, svg, image, video, audio.
# "asset" files (styles, scripts, data, fonts, models) are only served to a page that loads them.
TYPES: dict[str, tuple[str, str]] = {
    ".html": ("text/html; charset=utf-8", "html"),
    ".htm": ("text/html; charset=utf-8", "html"),
    ".svg": ("image/svg+xml", "svg"),
    ".png": ("image/png", "image"),
    ".jpg": ("image/jpeg", "image"),
    ".jpeg": ("image/jpeg", "image"),
    ".gif": ("image/gif", "image"),
    ".webp": ("image/webp", "image"),
    ".mp4": ("video/mp4", "video"),
    ".webm": ("video/webm", "video"),
    ".mp3": ("audio/mpeg", "audio"),
    ".wav": ("audio/wav", "audio"),
    ".ogg": ("audio/ogg", "audio"),
    ".css": ("text/css; charset=utf-8", "asset"),
    ".js": ("text/javascript; charset=utf-8", "asset"),
    ".mjs": ("text/javascript; charset=utf-8", "asset"),
    ".json": ("application/json", "asset"),
    ".woff2": ("font/woff2", "asset"),
    ".woff": ("font/woff", "asset"),
    ".ttf": ("font/ttf", "asset"),
    ".glb": ("model/gltf-binary", "asset"),
    ".ico": ("image/x-icon", "asset"),
}
SERVE_MAX_BYTES = 64 * 1024 * 1024
INLINE_MAX_BYTES = 5 * 1024 * 1024  # larger files are listed, not shown in the chat
PAGE_MAX_BYTES = 2 * 1024 * 1024  # a page is read to add the load monitor
MAX_FILES = 40
CHAT_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_REFRESH = re.compile(r"<meta\b[^>]*http-equiv\s*=\s*[\"']?refresh[^>]*>", re.IGNORECASE)
_HEAD = re.compile(r"<head(?:\s[^>]*)?>", re.IGNORECASE)

# Runs first in a page: tells the chat when a script or file was blocked or the page failed,
# so the preview can say so instead of staying blank.
MONITOR = (
    "<script>(function(){var p=function(k,d){try{parent.postMessage({kiwibroPreview:k,detail:String(d||'')},'*')}catch(e){}};"
    "addEventListener('securitypolicyviolation',function(e){p('blocked',(e.blockedURI||'inline')+' '+e.effectiveDirective)});"
    "addEventListener('error',function(e){var t=e.target;"
    "if(t&&t!==window&&(t.src||t.href)){p('failed',t.src||t.href)}else{p('error',e.message)}},true);"
    "addEventListener('unhandledrejection',function(e){p('error',e.reason&&e.reason.message||e.reason)})})();</script>"
)


def entry(workspace: Workspace, raw: str, *, kinds: tuple[str, ...] | None = None) -> dict[str, Any] | None:
    """What a workspace file is, or None when it may not be shown (outside, missing, wrong type)."""
    try:
        resolved = workspace.resolve(raw, must_exist=True)
    except SandboxError:
        return None
    if not resolved.is_file():
        return None
    rel = workspace.public(resolved)
    kind = TYPES.get(resolved.suffix.lower(), ("", ""))[1]
    hidden = any(part.startswith(".") for part in Path(rel).parts)
    if not kind or hidden or safety._SECRET_NAME.match(resolved.name) or (kinds and kind not in kinds):
        return None
    size = resolved.stat().st_size
    if size > SERVE_MAX_BYTES:
        return None
    return {
        "path": rel,
        "abs": str(resolved),
        "name": resolved.name,
        "kind": kind,
        "size": size,
        "inline": size <= INLINE_MAX_BYTES,
        "mtime": resolved.stat().st_mtime,
    }


def present(workspace: Workspace, raw: str) -> str | None:
    """The workspace path of an existing ordinary file of any type, or None (outside, missing, hidden, secret)."""
    try:
        resolved = workspace.resolve(raw, must_exist=True)
    except SandboxError:
        return None
    rel = workspace.public(resolved)
    hidden = any(part.startswith(".") for part in Path(rel).parts)
    if not resolved.is_file() or hidden or safety._SECRET_NAME.match(resolved.name):
        return None
    return rel


def shown_files(workspace: Workspace, written: list[str], named: list[str], since: float) -> list[dict[str, Any]]:
    """The visual files of a turn. `written` come from file tools; `named` appear in command
    output and count only when changed since the turn began."""
    found: dict[str, dict[str, Any]] = {}
    for raw, needs_new in [(item, False) for item in written] + [(item, True) for item in named]:
        info = entry(workspace, raw, kinds=("html", "svg", "image", "video", "audio"))
        if info is None or info["path"] in found or (needs_new and info["mtime"] < since - 2):
            continue
        found[info["path"]] = info
        if len(found) >= MAX_FILES:
            break
    return list(found.values())


def page_bytes(path: Path) -> bytes:
    """An HTML page with the load monitor first in <head> and meta refresh removed."""
    text = _REFRESH.sub("", path.read_text(encoding="utf-8", errors="replace"))
    head = _HEAD.search(text)
    if head:
        text = text[: head.end()] + MONITOR + text[head.end():]
    else:
        text = f"<head>{MONITOR}</head>" + text
    return text.encode("utf-8")


# Public libraries a page may load scripts and styles from. Nothing else (requests, images, media,
# fonts, forms) goes to these or to any other host.
SCRIPT_HOSTS = ("https://cdn.jsdelivr.net", "https://unpkg.com", "https://cdnjs.cloudflare.com", "https://esm.sh")


def policy(base_url: str, kind: str) -> str:
    """The Content-Security-Policy of one served file. `base_url` is the folder tree of the chat:
    a page may load files from there, and scripts and styles from SCRIPT_HOSTS."""
    parts = ["sandbox allow-scripts", "default-src 'none'", "style-src 'unsafe-inline'"]
    if kind == "html":
        parts = [
            "sandbox allow-scripts",
            "default-src 'none'",
            f"script-src 'unsafe-inline' {base_url} {' '.join(SCRIPT_HOSTS)}",
            f"style-src 'unsafe-inline' {base_url} {' '.join(SCRIPT_HOSTS)}",
            f"img-src data: blob: {base_url}",
            f"media-src data: blob: {base_url}",
            f"font-src data: {base_url}",
            f"connect-src {base_url}",
            "form-action 'none'",
            "base-uri 'none'",
        ]
    return "; ".join(parts)


def _git(root: Path, *argv: str) -> str:
    try:
        done = subprocess.run(
            ["git", *argv], cwd=str(root), capture_output=True, text=True, timeout=2, stdin=subprocess.DEVNULL,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def folder_badge(root: Path) -> dict[str, Any]:
    """The folder a chat works in, the way people read it (`~/Desktop/app`), and its git branch."""
    branch = _git(root, "symbolic-ref", "--short", "-q", "HEAD")
    if not branch:  # detached HEAD names a commit; a plain folder has no branch at all
        commit = _git(root, "rev-parse", "--short", "HEAD")
        branch = f"detached at {commit}" if commit else ""
    return {"path": config.display_path(str(root)), "abs": str(root), "name": root.name, "branch": branch}


__all__ = ["TYPES", "entry", "present", "shown_files", "page_bytes", "policy", "folder_badge", "CHAT_ID"]
