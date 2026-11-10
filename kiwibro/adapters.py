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

Adapters adjust themselves. When a reply goes wrong with an adapter on and comes out
right with it off, that counts against the adapter (:class:`Health`): after
``TROUBLE_HALF`` such replies it runs at half its scale, after ``TROUBLE_OFF`` it stays
off. The count ages out after ``TROUBLE_DAYS``, so an adapter gets another chance.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from . import gguf, sense, terms

TASKS = ("code", "math", "writing", "summarize", "translate", "research")
ALWAYS = "*"
MAX_ADAPTERS = 6  # per server: each one is held in memory next to the model
TROUBLE_HALF = 2  # replies that went wrong with the adapter on and right with it off: half scale
TROUBLE_OFF = 4  # ... and from here the adapter stays off
TROUBLE_DAYS = 30  # how long such a reply counts against an adapter

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
# What each task is about, for a message that uses none of the words above ("why does my
# loop never end"). Kiwi Sense compares the message's content words with these.
_TASK_MEANINGS: dict[str, tuple[str, ...]] = {
    "code": ("program source code function error bug", "fix loop variable crash software",
             "write script command terminal"),
    "math": ("calculate number sum multiply divide", "solve equation fraction percentage",
             "arithmetic average total cost"),
    "writing": ("write paragraph text wording friendlier", "draft message note announcement",
                "improve sentence grammar style"),
    "summarize": ("short version main points brief", "summary overview key takeaways"),
    "translate": ("translate language German French Spanish", "say sentence another language meaning word"),
    "research": ("find information current facts online", "recent events news who what when"),
}
MEANING_MATCH = 0.5  # how close a message must be to a task's meaning ...
MEANING_LEAD = 0.15  # ... and how far ahead of the next task, to count as that task


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
    return best or _by_meaning(text or "")


def _by_meaning(text: str) -> str:
    """The task a message is about when it names none of the task words: the one clear
    winner by meaning, or "" (a wrong adapter is worse than none)."""
    gist = terms.gist(text[:400])
    reader = sense.shared() if gist else None
    if reader is None:
        return ""
    scored = sorted(((max(reader.closeness(gist, meaning) for meaning in _TASK_MEANINGS[task]), task)
                     for task in TASKS), reverse=True)
    (top, task), (second, _other) = scored[0], scored[1]
    return task if top >= MEANING_MATCH and top - second >= MEANING_LEAD else ""


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


def strength(trouble: int) -> float:
    """The share of its scale an adapter keeps after ``trouble`` replies it spoiled."""
    return 0.0 if trouble >= TROUBLE_OFF else 0.5 if trouble >= TROUBLE_HALF else 1.0


def request_scales(adapters: tuple[Adapter, ...], task: str, trouble: Mapping[str, int] | None = None,
                   off: bool = False) -> list[dict[str, Any]]:
    """llama-server's per-request ``lora`` list: every loaded adapter with its scale now.

    ``trouble`` (path -> spoiled replies) turns an adapter down or off; ``off`` switches
    all of them off for this request."""
    out = []
    for index, adapter in enumerate(adapters):
        wanted = not off and ((task and task in adapter.tasks) or ALWAYS in adapter.tasks)
        scale = adapter.scale * strength((trouble or {}).get(adapter.path, 0)) if wanted else 0.0
        out.append({"id": index, "scale": round(scale, 3)})
    return out


class Health:
    """Which adapters spoiled replies lately: ``{adapter path: [unix times]}`` in one small file."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._times: dict[str, list[float]] | None = None

    def _load(self) -> dict[str, list[float]]:
        if self._times is None:
            self._times = {}
            try:
                data = json.loads(self._path.read_text(encoding="utf-8")) if self._path else {}
            except (OSError, ValueError):
                data = {}
            if isinstance(data, dict):
                for key, times in data.items():
                    if isinstance(times, list):
                        self._times[str(key)] = [float(t) for t in times if isinstance(t, (int, float))]
        return self._times

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._times), encoding="utf-8")
        except OSError:  # the count still holds for this run
            pass

    def trouble(self, adapter_path: str) -> int:
        """Replies this adapter spoiled within the last TROUBLE_DAYS."""
        since = time.time() - TROUBLE_DAYS * 86400
        with self._lock:
            return sum(1 for t in self._load().get(adapter_path, ()) if t >= since)

    def counts(self) -> dict[str, int]:
        with self._lock:
            paths = list(self._load())
        return {path: count for path in paths if (count := self.trouble(path))}

    def note(self, adapter_path: str) -> int:
        """Count one more spoiled reply; returns the adapter's count now."""
        since = time.time() - TROUBLE_DAYS * 86400
        with self._lock:
            times = [t for t in self._load().get(adapter_path, ()) if t >= since]
            times = [*times, time.time()][-2 * TROUBLE_OFF:]
            self._times[adapter_path] = times
            self._save()
            return len(times)

    def reset(self, adapter_path: str | None = None) -> None:
        """Forget what counted against one adapter, or against all of them."""
        with self._lock:
            self._load()
            if adapter_path is None:
                self._times = {}
            else:
                self._times.pop(adapter_path, None)
            self._save()
