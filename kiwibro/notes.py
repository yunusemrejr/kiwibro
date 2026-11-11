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


# A note the user dictates word for word ("save a note: buy oat milk", "jot down call mom"), as
# opposed to one they ask to have written ("make a note about the meeting"): only the first can be
# saved as typed.
_POLITE = (r"(?:(?:hey|ok(?:ay)?|please|pls|kiwi(?:bro)?|(?:can|could|would|will)\s+you|"
           r"i\s+(?:want|need)\s+you\s+to)\b[\s,]*)*")
_SAID = r"(?:\s*:\s*|\s+[\-–—]\s+|\s+(?:that|saying)\s+)"
# The first pattern that matches decides, so the bare "note down ..." forms come last.
_DICTATED = tuple(re.compile(rf"^\s*{_POLITE}{pattern}(?P<text>.+)$", re.I | re.S) for pattern in (
    r"(?:save|take|make|write|add|create|keep|store)\s+(?:down\s+)?(?:me\s+)?(?:a|an|this|the|the\s+following|a\s+new|a\s+quick)\s+"
    rf"note\b(?:\s+to\s+self)?{_SAID}",
    r"(?:new\s+)?note(?:\s+(?:down|to\s+self))?(?:\s*:\s*|\s+[\-–—]\s+)",
    r"note\s+(?:down\s+)?that\s+",
    rf"(?:remember|save|keep|store)\s+(?:this|that|it)\s+as\s+(?:a\s+(?:new\s+)?|new\s+)?note{_SAID}",
    rf"(?:make|take|save|create|keep)\s+(?:a\s+)?note\s+of\s+(?:this|that|it){_SAID}",
    rf"(?:add|put|save|write)\s+(?:this\s+)?(?:to|in|into)\s+(?:my\s+)?notes?{_SAID}",
    r"write\s+(?:(?:this|that|it)\s+)?down\s*[:\-–—]\s*",
    r"jot\s+(?:(?:this|that|it)\s+)?down(?:\s*[:\-–—]\s*|\s+)",
    r"note\s+(?:down|to\s+self)\s+",
))
# "make a note called Trip with flights on the 12th": the title ends at a colon, " with " or " that ".
_TITLED = re.compile(
    rf"^\s*{_POLITE}(?:(?:save|take|make|write|add|create|keep|store)\s+)?"
    r"(?:(?:a|an|the|new|a\s+new|a\s+quick)\s+)?note\s+(?:called|titled|named)\s+"
    r"(?P<title>.+?)(?:\s*:\s*|\s+[\-–—]\s+|\s+(?:with|that)\s+)(?P<text>.+)$",
    re.I | re.S,
)
# A polite question ("can you note down ...?") ends in a question mark that is not part of the note.
_ASKED = re.compile(rf"^\s*{_POLITE}(?:can|could|would|will)\s+you\b", re.I)
# "save that as a note": the note is what was just said.
_THAT = r"(?:that|this|it|the\s+(?:answer|above|last\s+(?:answer|reply|message)))"
_POINTED = tuple(re.compile(rf"^\s*{_POLITE}{pattern}[\s.!]*$", re.I) for pattern in (
    rf"(?:save|keep|store|add|put|write|remember)\s+{_THAT}\s+(?:down\s+)?(?:as|in|into|to)\s+(?:a\s+|my\s+)?(?:new\s+)?notes?(?:\s+(?:please|for\s+me))?",
    rf"(?:note|jot)\s+{_THAT}\s+down(?:\s+(?:please|for\s+me))?",
    rf"(?:make|take|save|create|keep)\s+(?:a\s+)?note\s+of\s+{_THAT}(?:\s+(?:please|for\s+me))?",
))


# A note the user asks to have written ("make a note about the meeting", "summarise this and
# save it as a note"): the model writes it. When it only writes and never saves, the reply is
# the note.
_WRITTEN = (
    re.compile(rf"^\s*{_POLITE}(?:make|write|create|draft|start|put\s+together|prepare)\s+(?:me\s+)?(?:a|an|some|the)?\s*"
               r"(?:short\s+|quick\s+|new\s+|brief\s+|small\s+)?notes?\s+(?:about|on|for|explaining|summari[sz]ing|covering|of|with)\s+"
               r"(?P<topic>.+)$", re.I | re.S),
    re.compile(rf"^\s*{_POLITE}(?P<topic>.+?)\s*,?\s+(?:and|then)\s+(?:save|keep|store|put|add)\s+(?:it|that|this|them)\s+"
               r"(?:as|in|into|to)\s+(?:a\s+|my\s+)?(?:new\s+)?notes?\b[\s.!]*$", re.I | re.S),
)


def written_request(message: str) -> str | None:
    """The topic of a note the user asks Kiwibro to write ("" when none can be read); None when
    the message asks for no such note."""
    text = (message or "").strip()
    if not text or len(text) > MAX_BODY or dictation(text) is not None:
        return None
    for pattern in _WRITTEN:
        found = pattern.match(text)
        if found:
            topic = re.sub(r"^(?:write|make|give|create)\s+(?:me\s+)?(?:a|an)?\s*", "", found.group("topic").strip(), flags=re.I)
            return clean_title(topic.rstrip(" .!?"))[:60].strip()
    return None


def dictation(message: str) -> tuple[str, str] | None:
    """(title, body) of the note a message dictates, word for word. The title is "" when none
    was given; the body is "" when the message points at what was just said ("save that as a
    note"); None when it does not ask for a note that code can save."""
    text = (message or "").strip()
    if not text or len(text) > MAX_BODY:
        return None
    if any(pattern.match(text) for pattern in _POINTED):
        return "", ""
    named = _TITLED.match(text)
    if named:
        title = clean_title(named.group("title"))
        body = _dictated_body(named.group("text"), text)
        if title and body:
            return title, body
    for pattern in _DICTATED:
        found = pattern.match(text)
        if found:
            body = _dictated_body(found.group("text"), text)
            return ("", body) if len(body.split()) >= 2 else None
    return None


def dictated(message: str) -> str | None:
    """The body of the note a message dictates, word for word; "" when it points at what was
    just said ("save that as a note"); None when it does not ask for a note that code can save."""
    found = dictation(message)
    return None if found is None else found[1]


def _dictated_body(text: str, message: str) -> str:
    """The words to save: wrapping quotes go, and so does the question mark of a polite question."""
    body = text.strip()
    if len(body) >= 2 and _QUOTE_PAIRS.get(body[0]) == body[-1]:
        body = body[1:-1].strip()
    if body.endswith("?") and _ASKED.match(message):
        body = body[:-1].rstrip()
    return body


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


def _bears(query: str, note: dict[str, Any]) -> bool:
    """False when the relate helper sees that a note does not bear on the message."""
    from . import readings

    score = readings.relates(query, f"{note['title']}. {note['body'][:400]}")
    return score is None or score >= readings.RELATE_FLOOR


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
        return [note for note in chosen if note["pinned"] or _bears(query, note)][:limit]

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
