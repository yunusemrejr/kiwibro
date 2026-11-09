"""Task adapters: LoRA files that fit a base model, switched per request by the task.

A LoRA adapter is a small GGUF (``general.type = adapter``) trained on top of one base
model. The direct engine loads every adapter that fits the model it starts
(``--lora … --lora-init-without-apply``: loaded, scale 0) and each request then names
the scale of every adapter (llama-server's per-request ``lora`` field). Which adapter
is on follows the task the user is on: code, math, writing, summarize, translate or
research, read from the message itself. A follow-up without a signal of its own
("ok, do it") keeps the task of the message before it, so the adapter — and with it
the server's KV cache for the chat — only changes when the task really does.

What an adapter is for comes from a sidecar ``<file>.json`` (``{"tasks": ["code"],
"scale": 0.8}``; ``"*"`` = always on) or, without one, from words in its file name and
``general.name``. An adapter whose purpose is unknown is loaded but never switched on.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import gguf

TASKS = ("code", "math", "writing", "summarize", "translate", "research")
ALWAYS = "*"
MAX_ADAPTERS = 6  # per server: each one is held in memory next to the model

# What the user is asking for, from their own words (checked in TASKS order on a tie).
_TASK_PATTERNS: dict[str, re.Pattern[str]] = {
    "code": re.compile(
        r"```|\b(?:code|coding|program(?:ming)?|function|bug|debug|compil\w*|script|python|javascript|typescript|"
        r"rust|golang|java|sql|regex|refactor\w*|traceback|stack ?trace|exception|unit tests?|git|bash|shell|"
        r"html|css|json|api|class|algorithm|repo(?:sitory)?)\b", re.IGNORECASE),
    "math": re.compile(
        r"\b(?:math\w*|calculat\w*|equation|solve|integral|derivative|algebra|geometry|probabilit\w*|"
        r"statistic\w*|percent\w*|proof|prove|theorem|matrix|sum of|how many|how much)\b|\d\s*[-+*/^=]\s*\d",
        re.IGNORECASE),
    "writing": re.compile(
        r"\b(?:write|rewrite|draft|essay|story|poem|blog|article|email|letter|cover letter|speech|slogan|"
        r"proofread|rephrase|tone|caption|lyrics|novel)\b", re.IGNORECASE),
    "summarize": re.compile(r"\b(?:summar\w*|tl;?dr|key points|recap|condense|shorten|gist|outline)\b", re.IGNORECASE),
    "translate": re.compile(r"\b(?:translat\w*|in (?:english|turkish|german|french|spanish|arabic|japanese|chinese)|"
                            r"çevir\w*|türkçe\w*|ingilizce\w*)\b", re.IGNORECASE),
    "research": re.compile(
        r"\b(?:research|search|look up|find out|latest|news|sources?|cite|compare|review of|who is|what is|"
        r"when did|where is)\b", re.IGNORECASE),
}
# What an adapter was trained for, from its name.
_NAME_HINTS: dict[str, re.Pattern[str]] = {
    "code": re.compile(r"cod(?:e|er|ing)|program|sql|python|dev(?:el)?\b", re.IGNORECASE),
    "math": re.compile(r"math|reason|logic|arith", re.IGNORECASE),
    "writing": re.compile(r"writ|story|prose|creativ|essay|roleplay|novel", re.IGNORECASE),
    "summarize": re.compile(r"summar|tldr|digest", re.IGNORECASE),
    "translate": re.compile(r"translat|multiling", re.IGNORECASE),
    "research": re.compile(r"research|rag|search|retriev|\bqa\b", re.IGNORECASE),
}
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Adapter:
    path: str
    name: str
    arch: str
    base: str = ""  # the base model it was trained on, when the file says
    tasks: tuple[str, ...] = ()
    scale: float = 1.0

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "arch": self.arch, "base": self.base,
                "tasks": list(self.tasks), "scale": self.scale}


def classify(text: str) -> str:
    """The task a piece of user text asks for, or "" when it carries no signal."""
    best, best_hits = "", 0
    for task in TASKS:
        hits = len(_TASK_PATTERNS[task].findall(text or ""))
        if hits > best_hits:
            best, best_hits = task, hits
    return best


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multimodal parts
        return "\n".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
    return ""


def task_of(messages: list[dict[str, Any]]) -> str:
    """The task of a conversation: the newest user message that names one.

    Only the opening paragraph of the newest message is read — Kiwibro appends its own
    per-turn note (time, recalled memory) after it, and that must not pick the adapter."""
    newest = True
    for message in reversed(messages or []):
        if message.get("role") != "user":
            continue
        text = _text(message.get("content"))
        if newest:
            text, newest = text.split("\n\n", 1)[0], False
        task = classify(text[:4000])
        if task:
            return task
    return ""


def adapter_dirs(data_dir: str | Path | None = None, extra: list[Path] | None = None) -> list[Path]:
    """Folders searched for adapters: KIWIBRO_ADAPTERS_DIR, Kiwibro's own, then model folders."""
    candidates = [Path(raw.strip()).expanduser()
                  for raw in os.environ.get("KIWIBRO_ADAPTERS_DIR", "").split(os.pathsep) if raw.strip()]
    if data_dir:
        candidates.append(Path(data_dir).expanduser() / "adapters")
    candidates += list(extra or [])
    out: list[Path] = []
    for path in candidates:
        if path.is_dir() and path not in out:
            out.append(path)
    return out


def _sidecar(path: Path) -> dict[str, Any]:
    for candidate in (path.with_suffix(".json"), Path(str(path) + ".json")):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return {}


def read_adapter(path: Path) -> Adapter | None:
    """An :class:`Adapter` for a LoRA GGUF, or None for anything else (a model, a broken file)."""
    try:
        meta = gguf.read_metadata(path)
    except (OSError, gguf.GGUFError):
        return None
    if meta.get("general.type") != "adapter" or str(meta.get("adapter.type") or "lora").lower() != "lora":
        return None
    name = str(meta.get("general.name") or path.stem)
    base = str(meta.get("general.base_model.0.name") or meta.get("general.basename") or "")
    side = _sidecar(path)
    tasks = side.get("tasks")
    if isinstance(tasks, list):
        found = tuple(t for t in (str(t).lower() for t in tasks) if t in TASKS or t == ALWAYS)
    else:
        label = f"{path.stem} {name}"
        found = tuple(task for task in TASKS if _NAME_HINTS[task].search(label))
    try:
        scale = min(2.0, max(0.0, float(side.get("scale", 1.0))))
    except (TypeError, ValueError):
        scale = 1.0
    return Adapter(str(path), name, str(meta.get("general.architecture") or ""), base, found, scale)


def scan(directories: list[Path], depth: int = 3) -> list[Adapter]:
    """Every LoRA adapter under ``directories`` (GGUF headers only; tensors are never read)."""
    out: list[Adapter] = []
    seen: set[str] = set()

    def walk(folder: Path, left: int) -> None:
        try:
            entries = sorted(folder.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.is_dir() and left > 0 and not entry.name.startswith("."):
                walk(entry, left - 1)
            elif entry.suffix.lower() == ".gguf" and str(entry) not in seen:
                seen.add(str(entry))
                adapter = read_adapter(entry)
                if adapter is not None:
                    out.append(adapter)

    for directory in directories:
        walk(directory, depth)
    return out


def fits(adapter: Adapter, arch: str, *labels: str) -> bool:
    """Whether an adapter belongs on a model: same architecture, and when the adapter
    names its base model, every word of that name appears in the model's name or id."""
    if not adapter.arch or adapter.arch != arch:
        return False
    if not adapter.base:
        return True
    have = set(_WORD.findall(" ".join(labels).lower()))
    return all(word in have for word in _WORD.findall(adapter.base.lower()))


def request_scales(adapters: tuple[Adapter, ...], task: str) -> list[dict[str, Any]]:
    """llama-server's per-request ``lora`` list: every loaded adapter with its scale now."""
    return [{"id": index, "scale": adapter.scale if (task and task in adapter.tasks) or ALWAYS in adapter.tasks else 0.0}
            for index, adapter in enumerate(adapters)]
