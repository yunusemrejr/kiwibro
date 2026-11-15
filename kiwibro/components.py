"""Kiwibro's component library: kits of starter files and recipes of proven steps.

The library is plain data under ``kiwibro/components/library``: one folder per
component holding ``component.json`` and, for a kit, a ``files/`` tree whose
contents and names may carry ``{{blank}}`` placeholders. This module reads that
data, ranks it for a search, writes it out for the model in a fixed shape, and
writes a kit's files into the workspace through the same sandbox and checkpoint
machinery as ``write_file``, so a path cannot leave the workspace and
``undo_changes`` takes the files back.

It also picks the component that fits a user's message (``suggest``), so the fitting
kit or recipe reaches the model in the turn note without a search of its own.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import checkpoints, config
from .sandbox import SandboxError, Workspace

log = logging.getLogger(__name__)

LIBRARY_ROOT = Path(__file__).parent / "components" / "library"
ACTIONS = ("search", "list", "show", "use")
KINDS = ("kit", "recipe")
KIT = "kit"
MAX_RESULTS = 6
_UNSET = "Nothing was written. Choose another folder with path, or move that file first."
_USE_HINT = 'To use one: component(action="use", name="<name>", path="<folder>", values={...})'
# A word in the name says most about what a component is, the title next; the
# summary and the "when" line are where a near miss usually turns up.
_NAME_WEIGHT = 5
_TITLE_WEIGHT = 3
_SUMMARY_WEIGHT = 2
_WHEN_WEIGHT = 1
_HOLDER = re.compile(r"\{\{([a-z_]+)\}\}")
_WORD = re.compile(r"[a-z0-9]+")


class ComponentError(ValueError):
    """A component call that cannot be answered as asked; its text is what the model reads."""


@dataclass(frozen=True)
class Component:
    """One library entry: a kit of starter files, or a recipe of proven steps."""

    name: str
    kind: str
    title: str
    summary: str
    when: str
    blanks: dict[str, dict[str, str]]
    next: str
    steps: tuple[str, ...]
    folder: Path
    match: tuple[str, ...] = ()  # phrases in a user's message that call for this component
    unless: tuple[str, ...] = ()  # phrases that rule it out: "3d printer" is not "3d scene"
    finished: bool = False  # the kit's files are the finished result: nothing to read, run or adapt

    @property
    def area(self) -> str:
        return self.name.split("/")[0]

    def file_paths(self) -> list[str]:
        """A kit's files relative to its ``files/`` folder, placeholders still in them."""
        root = self.folder / "files"
        if not root.is_dir():
            return []
        found = [
            item.relative_to(root).as_posix() for item in root.rglob("*")
            # running a kit's tests in place leaves bytecode behind; it is not part of the kit
            if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"
        ]
        return sorted(found, key=str.lower)

    def line(self) -> str:
        """The one-line form search and list print."""
        text = f"{self.name} ({self.kind}) - {self.summary}"
        return f"{text} Blanks: {', '.join(self.blanks)}" if self.blanks else text


class Library:
    """The component folders under one root, read once and kept in memory."""

    def __init__(self, root: str | Path = LIBRARY_ROOT) -> None:
        self.root = Path(root)
        self._components: list[Component] | None = None

    def all(self) -> list[Component]:
        if self._components is None:
            self._components = self._load()
        return self._components

    def areas(self) -> list[str]:
        return sorted({component.area for component in self.all()})

    def get(self, name: str) -> Component:
        wanted = (name or "").strip()
        for component in self.all():
            if component.name == wanted:
                return component
        raise ComponentError(_unknown_text(wanted, [component.name for component in self.all()]))

    def _load(self) -> list[Component]:
        components = []
        for path in self.root.glob("*/*/component.json"):
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            components.append(
                Component(
                    name=str(data["name"]),
                    kind=str(data["kind"]),
                    title=str(data.get("title") or ""),
                    summary=str(data.get("summary") or ""),
                    when=str(data.get("when") or ""),
                    blanks={str(key): dict(blank) for key, blank in (data.get("blanks") or {}).items()},
                    next=str(data.get("next") or ""),
                    steps=tuple(str(step) for step in data.get("steps") or ()),
                    folder=path.parent,
                    match=tuple(str(phrase) for phrase in data.get("match") or ()),
                    unless=tuple(str(phrase) for phrase in data.get("unless") or ()),
                    finished=data.get("finished") is True,
                )
            )
        return sorted(components, key=lambda component: component.name)


_DEFAULT: Library | None = None


def default_library() -> Library:
    """The library that ships with Kiwibro."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Library()
    return _DEFAULT


def _unknown_text(name: str, names: list[str]) -> str:
    close = difflib.get_close_matches(name, names, n=3, cutoff=0.6)
    if not close:  # nothing alike: the other entries of the same area still help
        area = name.split("/")[0]
        close = [other for other in names if other.split("/")[0] == area][:3]
    hint = f" Closest: {', '.join(close)}." if close else ""
    return f'there is no component named "{name}".{hint} component(action="list") shows all.'


# --- search -------------------------------------------------------------------------


def _stem(word: str) -> str:
    """Simple plurals count as one word: "pages" and "page" meet."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(text: str) -> set[str]:
    return {_stem(word) for word in _WORD.findall(str(text).lower())}


def _score(component: Component, wanted: set[str]) -> int:
    """How well a component answers the query's words; zero means it does not."""
    fields = (
        (_NAME_WEIGHT, _words(component.name)),
        (_TITLE_WEIGHT, _words(component.title)),
        (_SUMMARY_WEIGHT, _words(component.summary)),
        (_WHEN_WEIGHT, _words(component.when)),
    )
    return sum(weight for word in wanted for weight, words in fields if word in words)


def _use_hint(component: Component) -> str:
    if component.kind == KIT:
        return f'To use one: component(action="use", name="{component.name}", path="<folder>", values={{...}})'
    return f'To use one: component(action="use", name="{component.name}")'


def search(query: str, library: Library | None = None) -> str:
    """The best components for a query, one line each, then how to use one."""
    text = (query or "").strip()
    if not text:
        raise ComponentError("search needs query.")
    wanted = _words(text)
    found = []
    for component in (library or default_library()).all():
        score = _score(component, wanted)
        if score:
            found.append((score, component))
    if not found:
        return (f'No component matches "{text}". Try fewer or other words, '
                'or component(action="list") to see all.')
    found.sort(key=lambda hit: (-hit[0], hit[1].name))
    best = [component for _score_value, component in found[:MAX_RESULTS]]
    count = len(best)
    head = f'{count} component{"s" if count != 1 else ""} match{"es" if count == 1 else ""} "{text}":'
    return "\n".join([head, *(component.line() for component in best), _use_hint(best[0])])


def catalogue(query: str | None = None, library: Library | None = None) -> str:
    """Every component, one line each; a query that names an area narrows it to that area."""
    known = library or default_library()
    components = known.all()
    area = (query or "").strip().lower()
    if area in known.areas():
        components = [component for component in components if component.area == area]
    count = len(components)
    return "\n".join([f"{count} component{'s' if count != 1 else ''}:",
                      *(component.line() for component in components), _USE_HINT])


# --- suggest ------------------------------------------------------------------------

SUGGEST_HEADER = "## Ready-made part for this turn"
_MAX_MESSAGE = 600  # a pasted document names many things; none of them is the request
_MAX_GAP = 2  # "fix the login bug" still says "fix bug"
_FILLER = frozenset({"a", "an", "the", "my", "this", "that", "these", "those", "me", "please", "our", "your", "some"})
# A kit is files to start from, so it fits only a message that asks for something to be made.
_MAKES = re.compile(
    r"\b(?:create|make|build|write|generate|scaffold|set\s*up|start|add|implement|develop|code|draw|"
    r"need|want|give\s+me|new)\b",
    re.IGNORECASE,
)


def _sequence(text: str) -> list[str]:
    return [_stem(word) for word in _WORD.findall(str(text).lower()) if word not in _FILLER]


def _says(message: list[str], phrase: list[str]) -> int:
    """Where the message holds the phrase's words in order, at most ``_MAX_GAP`` words apart; -1 if nowhere."""
    if not phrase:
        return -1
    for start, word in enumerate(message):
        if word != phrase[0]:
            continue
        at = start
        for wanted in phrase[1:]:
            reach = message[at + 1:at + 2 + _MAX_GAP]
            if wanted not in reach:
                break
            at += 1 + reach.index(wanted)
        else:
            return start
    return -1


def suggest(message: str, library: Library | None = None) -> Component | None:
    """The one component a message calls for, or None when none or several fit equally.

    Each component lists the phrases that call for it (``match`` in its manifest) and, for
    the ones whose call is easy to say by accident, the phrases that rule it out (``unless``:
    a "3d printer" message is not a call for a 3D scene kit). Both are judged the same way.
    The longest phrase found wins, then the number of phrases found, then the one said first
    ("a web page with a contact form" is a page). A tie names nothing, because a wrong
    recipe in the turn note costs more than a missing one."""
    text = (message or "").strip()
    if not text or len(text) > _MAX_MESSAGE:
        return None
    said = _sequence(text)
    makes = bool(_MAKES.search(text))
    ranked = []
    for component in (library or default_library()).all():
        if component.kind == KIT and not makes:
            continue
        if any(_says(said, _sequence(phrase)) >= 0 for phrase in component.unless):
            continue
        hits = [(len(phrase), _says(said, phrase)) for phrase in map(_sequence, component.match)]
        hits = [(length, at) for length, at in hits if at >= 0]
        if hits:
            ranked.append(((max(length for length, _at in hits), len(hits), -min(at for _length, at in hits)), component))
    if not ranked:
        return None
    ranked.sort(key=lambda hit: hit[0], reverse=True)
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None
    return ranked[0][1]


def advice(component: Component) -> str:
    """What the turn note says about a suggested component: a kit's call, or a recipe's steps."""
    if component.kind == KIT:
        values = ", ".join(f'"{key}": "..."' for key in component.blanks)
        call = (
            f"{SUGGEST_HEADER}\n"
            f"A ready kit fits this request: {component.name} - {component.summary}\n"
            f'If the user wants it made, start with component(action="use", name="{component.name}", '
            f'path="<folder>", values={{{values}}}) and adapt the files it writes; do not write them from scratch.'
        )
        described = [f"- {key}: {blank['about']}" for key, blank in component.blanks.items() if blank.get("about")]
        if not described:
            return call
        # the blank's name alone cannot say what a value should look like: "scene" wants its lines
        return "\n".join([call, "What each value is:", *described])
    return "\n".join([
        SUGGEST_HEADER,
        f"A proven recipe fits this request ({component.name}). Follow its steps in order:",
        *(f"{number}. {step}" for number, step in enumerate(component.steps, 1)),
        f"Then: {component.next}",
    ])


# --- show ---------------------------------------------------------------------------


def show(name: str, library: Library | None = None) -> str:
    """One component in full: a kit's blanks and files, or a recipe's steps."""
    wanted = (name or "").strip()
    if not wanted:
        raise ComponentError("show needs name.")
    return _describe((library or default_library()).get(wanted))


def _describe(component: Component) -> str:
    head = f"{component.name} ({component.kind}) - {component.summary}"
    if component.kind == KIT:
        lines = [head, f"When: {component.when}"]
        if component.blanks:
            blanks = "; ".join(
                f'{key} (default "{blank.get("default", "")}"): {blank.get("about", "")}'
                for key, blank in component.blanks.items()
            )
            lines.append(f"Blanks: {blanks}")
        lines.append(f"Files: {', '.join(component.file_paths())}")
        # a model that has only looked at a kit tends to edit files that are not there yet
        values = ", ".join(f'"{key}": "..."' for key in component.blanks)
        lines.append(
            f'Nothing is written yet. To write these files: component(action="use", name="{component.name}", '
            f'path="<folder>", values={{{values}}})'
        )
        lines.append(f"After that: {component.next}")
        return "\n".join(lines)
    return "\n".join([head, "Steps:",
                      *(f"{number}. {step}" for number, step in enumerate(component.steps, 1)),
                      f"Then: {component.next}"])


# --- use ----------------------------------------------------------------------------


def _fill(text: str, values: dict[str, str]) -> str:
    return _HOLDER.sub(lambda match: values.get(match.group(1), match.group(0)), text)


def _folder(workspace: Workspace, path: str | None) -> Path:
    """The folder a kit goes into; a path outside the workspace is refused here, not later."""
    try:
        return workspace.resolve(path or "")
    except SandboxError as exc:
        raise ComponentError(str(exc)) from exc


def _blank_values(component: Component, values: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """The kit's blanks with the given values over the defaults, and the given keys it has no blank for.

    An extra key does not stop the kit: a small model that adds one would otherwise give up on a
    kit that fits. The result of ``use`` names what was left out."""
    filled = {key: str(blank.get("default", "")) for key, blank in component.blanks.items()}
    filled.update({key: str(value) for key, value in values.items() if key in component.blanks})
    return filled, [key for key in values if key not in component.blanks]


def _kit_plan(
    component: Component, values: dict[str, str], folder: Path, workspace: Workspace
) -> list[tuple[Path, Path, str]]:
    """Every file the kit writes: its source, the target in the workspace, and the target's text."""
    root = component.folder / "files"
    paths = component.file_paths()
    named = {key for path in paths for key in _HOLDER.findall(path)}
    for key in sorted(named):
        if "/" in values[key] or values[key].startswith("."):
            raise ComponentError(
                f'the value for "{key}" is used in a file name and cannot contain "/" or start with a dot.'
            )
    plan = []
    for path in paths:
        try:
            target = workspace.resolve(folder / _fill(path, values))
        except SandboxError as exc:
            raise ComponentError(str(exc)) from exc
        plan.append((root / path, target, _fill((root / path).read_text(encoding="utf-8"), values)))
    return plan


def _checkpoint(
    plan: list[tuple[Path, Path, str]], workspace: Workspace, conversation_id: str | None,
    data_dir: str | Path | None,
) -> None:
    """Undo cover for the files about to be written. A checkpoint never blocks the write."""
    if not conversation_id:
        return
    try:
        files = [(target, workspace.public(target)) for _source, target, _text in plan]
        checkpoints.store_for(Path(data_dir or config.data_dir())).snapshot(
            files, conversation_id=conversation_id, tool="component"
        )
    except Exception:
        log.warning("checkpoint before component failed", exc_info=True)


def use(
    name: str,
    values: dict[str, str] | None = None,
    *,
    workspace: Workspace,
    path: str | None = None,
    library: Library | None = None,
    conversation_id: str | None = None,
    data_dir: str | Path | None = None,
) -> str:
    """A recipe's steps, or a kit's files written into the workspace folder ``path``.

    Every target is checked before the first byte is written, so a kit is never
    left half-written and an existing file is never overwritten."""
    wanted = (name or "").strip()
    if not wanted:
        raise ComponentError("use needs name.")
    component = (library or default_library()).get(wanted)
    filled, extra = _blank_values(component, values or {})
    if component.kind != KIT:
        return _describe(component)
    folder = _folder(workspace, path)
    if folder.exists() and not folder.is_dir():
        raise ComponentError(f"{workspace.public(folder)} is a file, not a folder.")
    plan = _kit_plan(component, filled, folder, workspace)
    for _source, target, _text in plan:
        if target.exists():
            raise ComponentError(f"{workspace.public(target)} already exists. {_UNSET}")
    _checkpoint(plan, workspace, conversation_id, data_dir)
    written = []
    for source, target, text in plan:
        workspace.write_file(str(target), text)
        mode = source.stat().st_mode & 0o777
        try:
            os.chmod(target, mode)
        except OSError as exc:
            raise SandboxError(f"could not set the mode of {workspace.public(target)}: {exc.strerror or exc}") from exc
        written.append((workspace.public(target), len(text.splitlines())))
    where = "the workspace" if folder == workspace.root else workspace.public(folder)
    count = len(written)
    return "\n".join([
        f"Used {component.name} in {where}: {count} file{'s' if count != 1 else ''} written",
        *(f"{shown} ({number} lines)" for shown, number in written),
        *([_left_out(component, extra)] if extra else []),
        f"Next: {component.next}",
    ])


def _left_out(component: Component, extra: list[str]) -> str:
    names = ", ".join(f'"{key}"' for key in extra)
    known = ", ".join(component.blanks) or "none"
    return f"Left out: {names} ({component.name} has no such blank; its blanks: {known})."
