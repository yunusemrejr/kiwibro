"""Notes: longer pages (title, markdown body, tags) written by the user or by Kiwibro.

Memories hold short facts; notes hold whole pages. This module checks and shapes what
comes in and decides which notes the model should see; Database keeps them and indexes
them for full-text search.
"""

from __future__ import annotations

import re
from typing import Any

from . import sense, terms
from .db import Database

MAX_TITLE = 120
MAX_BODY = 20000
MAX_TAGS = 8
MAX_TAG_CHARS = 32
UNTITLED = "Untitled note"

_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’", "«": "»"}
_TAG_SPLIT_RE = re.compile(r"[,\s]+")
_TAG_DROP_RE = re.compile(r"[^\w-]+")


def _text(value: Any, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    return value


def _cut(text: str, limit: int) -> str:
    """``text`` shortened to ``limit`` characters (ellipsis included) at a word boundary."""
    if len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    head = text[: limit - 1]
    if text[limit - 1] != " " and " " in head:
        head = head.rsplit(" ", 1)[0]
    return head.rstrip() + "…"


def clean_title(text: Any) -> str:
    """One line, without markdown heading marks or wrapping quotes, at most MAX_TITLE characters."""
    line = " ".join(_text(text, "title").split()).lstrip("#").strip()
    if len(line) >= 2 and _QUOTE_PAIRS.get(line[0]) == line[-1]:
        line = line[1:-1].strip()
    return line[:MAX_TITLE].rstrip()


def derive_title(body: str) -> str:
    """A title for a note nobody named: its first non-empty line."""
    for line in (body or "").splitlines():
        title = clean_title(line)
        if title:
            return _cut(title, 60)
    return UNTITLED


def clean_body(text: Any) -> str:
    """Markdown kept as written: one line-ending style, no trailing spaces, no outer blank lines."""
    lines = _text(text, "note").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    body = "\n".join(line.rstrip() for line in lines).strip("\n")
    if not body.strip():
        raise ValueError("note is empty")
    if len(body) > MAX_BODY:
        raise ValueError(f"note is too long (max {MAX_BODY} characters)")
    return body


def clean_tags(value: Any) -> list[str]:
    """Lower-case tags without the leading '#', letters/digits/-/_ only, unique, at most MAX_TAGS.

    Accepts a list or a string with tags separated by commas or spaces."""
    if value is None:
        return []
    if isinstance(value, str):
        pieces = _TAG_SPLIT_RE.split(value)
    elif isinstance(value, (list, tuple)):
        pieces = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("tags must be text")
            pieces.extend(_TAG_SPLIT_RE.split(item))
    else:
        raise ValueError("tags must be a list or a comma separated string")
    tags: list[str] = []
    for piece in pieces:
        tag = _TAG_DROP_RE.sub("", piece.lower().lstrip("#"))[:MAX_TAG_CHARS].strip("-_")
        if tag and tag not in tags:
            tags.append(tag)
        if len(tags) == MAX_TAGS:
            break
    return tags


def _note_keys(note: dict[str, Any]) -> set[str]:
    return terms.key_terms(" ".join([note["title"], note["body"], *note["tags"]]))


class Notebook:
    """Writes, lookups and prompt selection for notes; storage stays in Database."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def save(
        self,
        body: Any,
        title: Any = None,
        tags: Any = None,
        *,
        origin: str = "you",
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Store a note. A note Kiwibro wrote is replaced by its next version with the same
        title, so a page it keeps up to date is edited in place instead of piling up as
        twins. A page the user wrote is never overwritten this way."""
        body = clean_body(body)
        title = clean_title(title) or derive_title(body)
        tags = clean_tags(tags)
        if origin == "kiwibro":
            existing = self.db.find_note_by_title(title)
            if existing is not None and existing["origin"] == "kiwibro":
                return self.db.update_note(existing["id"], body=body, tags=tags or existing["tags"]) or existing
        return self.db.create_note(
            title, body, tags, origin=origin, source_conversation_id=conversation_id,
        )

    def append(self, reference: str, text: Any) -> dict[str, Any] | None:
        """Add ``text`` after a blank line at the end of the note ``reference`` names."""
        note = self.find(reference)
        if note is None:
            return None
        body = clean_body(f"{note['body']}\n\n{clean_body(text)}")
        return self.db.update_note(note["id"], body=body)

    def find(self, reference: str) -> dict[str, Any] | None:
        """The note a user or the model means: by id, else exact title, else the best
        full-text hit whose title shares a meaningful word with the reference."""
        reference = (reference or "").strip()
        if not reference:
            return None
        note = self.db.get_note(reference) or self.db.find_note_by_title(reference)
        if note is not None:
            return note
        keys = terms.key_terms(reference)
        if not keys:
            return None
        for hit in self.db.search_notes(reference, limit=8):
            if keys & terms.key_terms(hit["title"]):
                return hit
        return None

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        return self.db.search_notes(query, limit=limit)

    def update(
        self,
        note_id: str,
        *,
        title: Any = None,
        body: Any = None,
        tags: Any = None,
        pinned: Any = None,
    ) -> dict[str, Any] | None:
        current = self.db.get_note(note_id)
        if current is None:
            return None
        fields: dict[str, Any] = {}
        if body is not None:
            fields["body"] = clean_body(body)
        if title is not None:
            fields["title"] = clean_title(title) or derive_title(fields.get("body", current["body"]))
        if tags is not None:
            fields["tags"] = clean_tags(tags)
        if pinned is not None:
            fields["pinned"] = bool(pinned)
        if not fields:
            return current
        return self.db.update_note(note_id, **fields)

    def delete(self, note_id: str) -> bool:
        return self.db.delete_note(note_id)

    def relevant(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        """Notes worth showing the model for this message: up to two pinned notes, the notes
        whose title, body or tags share a meaningful word with the message, then the notes
        that are about the same thing in other words."""
        recent = self.db.list_notes(limit=50)
        chosen = [note for note in recent if note["pinned"]][:2]
        seen = {note["id"] for note in chosen}
        keys = terms.key_terms(query)
        if keys:
            for hit in self._candidates(query, keys):
                if hit["id"] not in seen and keys & _note_keys(hit):
                    chosen.append(hit)
                    seen.add(hit["id"])
        if len(chosen) < limit and keys:
            others = [note for note in recent if note["id"] not in seen]
            chosen += self._meaning_matches(query, others, limit - len(chosen))
        return chosen[:limit]

    @staticmethod
    def _meaning_matches(query: str, notes: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        """Read by Kiwi Sense; nothing when it is unavailable."""
        reader = sense.shared()
        if reader is None or not notes:
            return []
        texts = [terms.gist(f"{note['title']} {' '.join(note['tags'])} {note['body'][:300]}") for note in notes]
        try:
            return [notes[index] for index, _score in reader.nearest(terms.gist(query), texts, limit=limit)]
        except Exception:  # pragma: no cover - defensive
            return []

    def _candidates(self, query: str, keys: set[str]) -> list[dict[str, Any]]:
        """Full-text hits for the query and for the words of any everyday topic it names."""
        pool: dict[str, dict[str, Any]] = {}
        for hit in self.db.search_notes(query, limit=12):
            pool.setdefault(hit["id"], hit)
        for group in terms.concept_groups(keys):
            for start in range(0, len(group), 12):  # an FTS query keeps 12 words
                for hit in self.db.search_notes(" ".join(group[start:start + 12]), limit=12):
                    pool.setdefault(hit["id"], hit)
        return list(pool.values())

    @staticmethod
    def render_block(notes: list[dict[str, Any]], max_chars: int = 900) -> str:
        """One line per note for the prompt, the whole block within ``max_chars``. Each note
        gets an equal share (at least 80 characters, so notes beyond that many are dropped)."""
        if not notes:
            return ""
        notes = notes[: max(1, (max_chars + 1) // 81)]
        share = (max_chars - (len(notes) - 1)) // len(notes)
        lines = []
        for note in notes:
            tags = f" [{', '.join(note['tags'])}]" if note["tags"] else ""
            label = _cut(f"{note['title']}{tags}", share // 2)
            excerpt = _cut(" ".join(note["body"].split()), share - len(f"- {label}: "))
            lines.append(f"- {label}: {excerpt}")
        return "\n".join(lines)
