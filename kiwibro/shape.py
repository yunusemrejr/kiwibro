"""The form of a finished reply, put right in code.

A small model often has the content and loses the form: it says the same line three times, or
answers "as one paragraph" with a list. Both are fixed here without asking the model again;
nothing is added to what it wrote.
"""

from __future__ import annotations

import difflib
import re

REPEAT_MIN_CHARS = 20  # shorter lines ("Yes.", "}") may well come twice
_MARKER = re.compile(r"^\s*(?:[-*•–]|\d{1,2}[.)])\s+")
_HEADING = re.compile(r"^\s*(?:#{1,6}\s+.*|\*\*[^*]+\*\*:?|__[^_]+__:?)\s*$")
_ONE = frozenset({"one", "single", "1"})
_INTO = frozenset({"as", "in", "into", "to"})  # "in a short paragraph" asks for one; "what is a paragraph" asks about them


def without_repeats(reply: str) -> str:
    """The reply with every line it says again, word for word, kept once. Code blocks and
    table rows are left alone."""
    seen: set[str] = set()
    kept: list[str] = []
    fenced = False
    for line in reply.split("\n"):
        if line.strip().startswith("```"):
            fenced = not fenced
        key = " ".join(_MARKER.sub("", line).lower().split()).rstrip(".!")
        if fenced or line.lstrip().startswith(("|", "```")) or len(key) < REPEAT_MIN_CHARS:
            kept.append(line)
        elif key not in seen:
            seen.add(key)
            kept.append(line)
    return "\n".join(kept)


def wants_one_paragraph(message: str, asks: tuple[str, ...] = ()) -> bool:
    """The user asked for a single paragraph ("as one paragraph", "in a short paragraf")."""
    words = re.findall(r"[a-z0-9]+", " ".join((message or "", *asks)).lower())
    for at, word in enumerate(words):
        if len(word) >= 7 and not word.endswith("s") and difflib.SequenceMatcher(None, word, "paragraph").ratio() >= 0.8:
            before = words[max(0, at - 4):at]
            if _ONE.intersection(before) or ("a" in before and _INTO.intersection(before)):
                return True
    return False


def as_one_paragraph(reply: str) -> str:
    """A reply written as a list or in several blocks, run together as one paragraph: headings
    and lead-in lines go, list markers go, every item ends as a sentence. A reply with code in
    it, or one that is a single paragraph already, comes back unchanged."""
    lines = [line.strip() for line in reply.split("\n") if line.strip()]
    if "```" in reply or len(lines) < 2:
        return reply
    lines = [line for line in lines if not _HEADING.match(line)]
    parts: list[str] = []
    for index, line in enumerate(lines):
        if not _MARKER.match(line) and line.endswith(":") and index + 1 < len(lines) and _MARKER.match(lines[index + 1]):
            continue  # "Here are the stories:" only announces the list
        text = _MARKER.sub("", line).strip()
        if text and text[-1] not in ".!?…":
            text = text.rstrip(",;:") + "."
        if text:
            parts.append(text)
    return " ".join(parts) if parts else reply
