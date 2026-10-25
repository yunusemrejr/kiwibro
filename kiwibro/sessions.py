"""Continuity across chats: what the other conversations were about, when the user asks.

Each chat's prompt holds only that chat. When a message points back in time ("like we
did yesterday", "continue where we left off", "what did you find last time"), the turn
note gets a short list of the most recent other chats — title, when, and a gist — plus
exact excerpts from them that match the message. Gists come from stored text (the
rolling summary, else the first real request and the last reply), never from a model
call, so this costs nothing on a CPU-only machine. Nothing is added otherwise, so normal
turns pay no prefill for it.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from . import terms
from .context import estimate_tokens

log = logging.getLogger(__name__)

RECENT_CHATS = 5
EXCERPTS = 3
GIST_CHARS = 220
EXCERPT_CHARS = 360
NOTE_MAX_TOKENS = 500

_REFERS_BACK = re.compile(
    r"\b(?:last\s+(?:time|session|chat|conversation|night|week)|yesterday|the\s+other\s+day"
    r"|earlier\s+(?:today|chat|conversation|session)|(?:previous|other|earlier|last|old)\s+(?:chat|conversation|session)s?"
    r"|we\s+(?:talked|spoke|discussed|chatted|worked|were\s+working|did|decided|agreed)\b"
    r"|you\s+(?:said|told\s+me|suggested|recommended|wrote|found|made|built|promised)\b"
    r"|(?:continue|pick\s+up|carry\s+on|resume)\s+(?:from\s+)?where\s+we\s+(?:left|stopped)"
    r"|where\s+we\s+left\s+off|remember\s+when|as\s+(?:we|i)\s+(?:discussed|mentioned|said)"
    r"|dün|geçen\s+(?:sefer|hafta)|kaldığımız\s+yerden)\b",
    re.IGNORECASE,
)


def refers_back(message: str) -> bool:
    """Does the message point at an earlier chat or an earlier day?"""
    return bool(_REFERS_BACK.search(message or ""))


def when(timestamp: str, now: datetime | None = None) -> str:
    """'today', 'yesterday', '3 days ago' or the date, from an ISO timestamp."""
    try:
        moment = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return ""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    days = (now.astimezone().date() - moment.astimezone().date()).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    return moment.astimezone().strftime("%Y-%m-%d")


_MARKUP = re.compile(r"(?m)^\s*(?:-{3,}|#{1,6}\s*)|\*\*|__|`+")


def _flat(text: str, limit: int) -> str:
    text = " ".join(_MARKUP.sub(" ", text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def gist(db: Any, conversation: dict[str, Any]) -> str:
    """What a chat was about, from stored text only."""
    summary = (conversation.get("summary") or "").strip()
    if summary:
        return _flat(summary, GIST_CHARS)
    messages = db.list_messages(conversation["id"], limit=40)
    request = next(
        (m for m in messages if m.get("role") == "user" and len(terms.content_words(m.get("content") or "")) >= 3),
        None,
    ) or next((m for m in messages if m.get("role") == "user"), None)
    replies = [m for m in db.recent_messages(conversation["id"], limit=6)
               if m.get("role") == "assistant" and (m.get("content") or "").strip()]
    parts = []
    if request:
        parts.append("asked: " + _flat(request.get("content") or "", GIST_CHARS // 2))
    if replies:
        parts.append("last reply: " + _flat(replies[-1].get("content") or "", GIST_CHARS // 2))
    return "; ".join(parts)


def recent_note(db: Any, conversation_id: str, message: str, *, max_tokens: int = NOTE_MAX_TOKENS) -> str:
    """The earlier-chats block for the turn note ('' unless the message refers back)."""
    if max_tokens < 60 or not refers_back(message):
        return ""
    try:
        chats = [c for c in db.recent_conversations(RECENT_CHATS + 1) if c["id"] != conversation_id][:RECENT_CHATS]
        if not chats:
            return ""
        titles = {c["id"]: c.get("title") or "Untitled" for c in chats}
        lines = ["## Your other recent chats (newest first)"]
        spent = estimate_tokens(lines[0])
        for chat in chats:
            line = f"- {titles[chat['id']]} ({when(chat.get('updated_at') or '')})"
            about = gist(db, chat)
            if about:
                line += f": {about}"
            cost = estimate_tokens(line)
            if spent + cost > max_tokens and len(lines) > 1:
                break
            lines.append(line)
            spent += cost
        excerpts = []
        seen: set[str] = set()
        words = terms.content_words(message)
        if words:
            hits = [h for h in db.search_messages(" ".join(words), limit=16)
                    if h.get("conversation_id") != conversation_id and h.get("role") in ("user", "assistant")]
            # Replies hold what was found or decided; the user's own question is the fallback.
            for hit in sorted(hits, key=lambda h: h.get("role") != "assistant"):
                text = _flat(hit.get("content") or "", EXCERPT_CHARS)
                if not text or text in seen:
                    continue
                seen.add(text)
                title = titles.get(hit.get("conversation_id"))
                if title is None:
                    other = db.get_conversation(hit.get("conversation_id") or "")
                    title = (other or {}).get("title") or "another chat"
                speaker = "you" if hit.get("role") == "assistant" else "user"
                line = f"- [{title}, {when(hit.get('created_at') or '')}, {speaker}] {text}"
                cost = estimate_tokens(line)
                if spent + cost > max_tokens:
                    break
                excerpts.append(line)
                spent += cost
                if len(excerpts) >= EXCERPTS:
                    break
        if excerpts:
            lines.append("## Matching messages from other chats (exact excerpts)")
            lines.extend(excerpts)
        return "\n".join(lines)
    except Exception:  # continuity is an enhancement, never a failure mode
        log.warning("earlier-chats note failed", exc_info=True)
        return ""
