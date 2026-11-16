"""The loop around a page Kiwibro made: look at it, say what is wrong, let the model fix it, look again.

The turn code (agent.py) owns the model, the database and the stream; this module holds what is decided without
them: which pages of a turn are due for a look, which line Kiwibro says after a look, and how a ``/goal`` message
is read. Every line is a constant of prompts.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import pagecheck, prompts
from .sandbox import Workspace

LOOKS = 3  # looks in a normal turn: the first, and two after edits
GOAL_LOOKS = 8
GOAL_REVIEWS = 3  # end-of-turn reviews of work that is not a page, in a goal turn
STYLES = (".css", ".js", ".mjs")  # a changed one next to a known page makes that page due again
_GOAL = re.compile(r"/goal(?:\s+(.*))?", re.IGNORECASE | re.DOTALL)


def parse_goal(message: str) -> tuple[bool, str]:
    """Is this a ``/goal`` message, and what is the goal after it ("" for a bare ``/goal``)?"""
    found = _GOAL.fullmatch((message or "").strip())
    return (True, (found.group(1) or "").strip()) if found else (False, "")


@dataclass
class Page:
    """An HTML page the turn wrote or changed."""

    path: str  # as shown to the user
    folder: Path
    workspace: Workspace
    kit: str = ""  # the component that wrote it, if one did
    dirty: bool = True  # written or changed since the last look
    looks: int = 0
    findings: list[dict[str, str]] = field(default_factory=list)  # the problems of the latest look
    confirmed: list[str] = field(default_factory=list)  # what the latest look found to be right
    said: bool = False  # Kiwibro has said how the latest look ended ("It is good now", "I stopped"), so a goal line leaves it out


@dataclass
class Pages:
    """The pages of one turn, in the order they were written."""

    pages: dict[str, Page] = field(default_factory=dict)
    looks: int = 0

    def note_html(self, workspace: Workspace, target: Path, kit: str = "") -> None:
        shown = workspace.public(target)
        page = self.pages.get(shown)
        if page is None:
            self.pages[shown] = Page(shown, target.parent, workspace, kit)
        else:
            page.dirty = True

    def note_changed(self, target: Path) -> None:
        """A stylesheet or script was written: the pages that sit beside it, or above it, are due again."""
        for page in self.pages.values():
            if target.suffix.lower() in STYLES and (page.folder == target.parent or page.folder in target.parents):
                page.dirty = True

    def due(self) -> Page | None:
        return next((page for page in self.pages.values() if page.dirty), None)

    def unfinished(self) -> Page | None:
        """A page whose latest look found problems and that has not been edited since."""
        return next((page for page in self.pages.values() if not page.dirty and page.findings), None)


_SETTINGS = re.compile(r"<script([^>]*)>(.*?)</script>", re.IGNORECASE | re.DOTALL)


def settings_lines(page: Page) -> str:
    """The lines a kit page keeps its settings in (its plain-text script blocks), as they read now."""
    try:
        source = page.workspace.resolve(page.path).read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return ""
    lines = [
        line.strip() for attrs, body in _SETTINGS.findall(source)
        if re.search(r"text/plain|id=[\"']settings[\"']", attrs, re.IGNORECASE) for line in body.splitlines() if line.strip()
    ]
    return "\n".join(lines[:12])[:600]


def verdict(before: list[dict[str, str]] | None, now: list[dict[str, str]]) -> str:
    """What Kiwibro says after a look that found problems, by how they compare with the look before."""
    problems = "; ".join(item["short"] for item in now)
    if before is None:
        return prompts.LOOK_BAD.format(problems=problems)
    old, new = {item["short"] for item in before}, {item["short"] for item in now}
    if old == new:
        return prompts.LOOK_SAME.format(problems=problems)
    if len(now) > len(before):
        return prompts.LOOK_WORSE.format(problems=problems)
    if len(now) < len(before):
        fixed = "; ".join(item["fixed"] for item in before if item["short"] not in new)
        return prompts.LOOK_BETTER.format(fixed=fixed, problems=problems)
    return prompts.LOOK_BAD.format(problems=problems)


def good(look: dict[str, Any]) -> str:
    confirmed = look.get("confirmed") or []
    if confirmed:
        return prompts.LOOK_GOOD.format(confirmed=pagecheck.join_words(list(confirmed)))
    return prompts.LOOK_GOOD_PLAIN.format(file=look["page"])
