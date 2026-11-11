"""What the Kiwi Sense helpers make of a user message, put to work for the turn.

The helpers (kiwibro/sense/helpers.py) are tiny trained deciders. They never write and
never call the language model. Here their readings become decisions made in code:

    language     a message that is not in English gets a fixed reply asking for English
    search       whether the answer needs the web (with the regex rules in plan.py)
    time         the time phrases of the message, turned into real dates
    demands      what the answer must honour (format, length, tools, audience)
    shift        whether the message follows up, stays on the topic or starts a new one
    keep         whether the message says something about the user worth remembering
    relates      whether a saved memory or note bears on the message

Every function returns a neutral answer when the helpers are missing, so Kiwibro works
the same without them.
"""

from __future__ import annotations

import calendar
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from . import sense

log = logging.getLogger(__name__)

# How sure a helper must be before code acts on it alone.
WEB_SURE = 0.6  # search first when the message needs live data or asks for a search
WEB_NOT_SURE = 0.9  # drop a search the word rules wanted when the helper is this sure it needs none
KEEP_SKIP = 0.7  # do not save "facts" from a hypothetical, quoted or passing message
NEW_TOPIC_SURE = 0.85  # the shift helper is the least sure of the five: only a clear change counts
RELATE_FLOOR = 0.02  # a saved item the relate helper is this sure does not bear on the message is left out
MAX_ASKS = 6

ENGLISH_ONLY = (
    "This doesn't look like English{named}. Kiwibro works in English only, so please write your message "
    "in English. (Quoting words in another language inside an English message is fine.)"
)


@dataclass
class TurnReading:
    language: Any = None
    reading: Any = None
    shift: tuple[str, float] | None = None
    ranges: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.reading is not None:
            out.update(self.reading.summary())
        if self.shift is not None:
            out["shift"] = self.shift[0]
            out.setdefault("sure", {})["shift"] = round(self.shift[1], 2)
        if self.ranges:
            out["ranges"] = [f"{phrase} = {span}" for phrase, span in self.ranges]
        return out


def helpers():
    reader = sense.shared()
    if reader is None:
        return None
    try:
        found = reader.helpers
    except Exception:  # pragma: no cover - defensive
        log.warning("Kiwi Sense helpers could not load", exc_info=True)
        return None
    return found if any(index is not None for index in found.tags.values()) else None


def language(message: str):
    """The language reading, or None when the helper is missing or the message is English."""
    found = helpers()
    if found is None:
        return None
    try:
        reading = found.language(message)
    except Exception:  # never break a turn
        log.warning("language check failed", exc_info=True)
        return None
    return None if reading.english else reading


def english_only_reply(reading) -> str:
    name = getattr(reading, "name", "") or ""
    return ENGLISH_ONLY.format(named=f" (it reads as {name})" if name else "")


def read(message: str, before: str = "") -> TurnReading:
    """Everything the helpers say about a message; ``before`` is the previous exchange."""
    out = TurnReading()
    found = helpers()
    if found is None:
        return out
    try:
        out.reading = found.read(message)
        if before.strip() and found.has("Hshf"):
            out.shift = found.shift(before, message)
        phrases = out.reading.times if out.reading is not None else []
        out.ranges = resolve_times(phrases or _time_words(message))
    except Exception:  # never break a turn
        log.warning("message reading failed", exc_info=True)
    return out


def relates(message: str, item: str) -> float | None:
    found = helpers()
    if found is None or not found.has("Hrel"):
        return None
    try:
        return found.relates(message, item)
    except Exception:  # pragma: no cover - defensive
        return None


# ---- decisions ----


def wants_search(turn_reading: TurnReading) -> bool:
    r = turn_reading.reading
    return r is not None and r.web in ("asked", "live") and r.sure.get("web", 0.0) >= WEB_SURE


def needs_no_search(turn_reading: TurnReading) -> bool:
    r = turn_reading.reading
    return r is not None and r.web in ("none", "own") and r.sure.get("web", 0.0) >= WEB_NOT_SURE


def keep_nothing(turn_reading: TurnReading) -> bool:
    """The message only seems to state facts about the user: hypothetical, quoted, about
    someone else, or a passing mood."""
    r = turn_reading.reading
    return r is not None and r.keep in ("unsure", "passing") and r.sure.get("keep", 0.0) >= KEEP_SKIP


def worth_keeping(turn_reading: TurnReading) -> bool:
    r = turn_reading.reading
    return r is not None and r.keep in ("fact", "preference")


def new_topic(turn_reading: TurnReading) -> bool:
    return turn_reading.shift is not None and turn_reading.shift[0] == "new" and turn_reading.shift[1] >= NEW_TOPIC_SURE


def follows_up(turn_reading: TurnReading) -> bool:
    return turn_reading.shift is not None and turn_reading.shift[0] == "followup"


def note(turn_reading: TurnReading) -> str:
    """Lines for the per-turn note: the demands to honour, the dates meant, a new topic."""
    lines: list[str] = []
    r = turn_reading.reading
    if r is not None and r.asks:
        asks = "; ".join(r.asks[:MAX_ASKS])
        lines.append(f"The answer must honour: {asks}.")
    if turn_reading.ranges:
        spans = "; ".join(f"{phrase} = {span}" for phrase, span in turn_reading.ranges)
        lines.append(f"Dates the message means: {spans}.")
    if new_topic(turn_reading):
        lines.append("This message starts a new topic: answer it on its own; the earlier talk is only background.")
    return "\n".join(lines)


# ---- time phrases -> dates ----

_MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): index for index, name in enumerate(calendar.month_abbr) if name})
_DAYS = {name.lower(): index for index, name in enumerate(calendar.day_name)}
_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10, "few": 3, "couple": 2, "several": 3}
_TIME_WORDS = re.compile(
    r"\b(?:today|tonight|yesterday|tomorrow|(?:this|last|next|past|coming)\s+(?:\d+\s+)?(?:week|month|year|weekend|"
    r"days?|weeks|months|years|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"(?:\d+|a|an|one|two|three|four|five|six|seven|ten|few|couple(?:\s+of)?)\s+(?:days?|weeks?|months?|years?)\s+(?:ago|from\s+now)|"
    r"in\s+(?:19|20)\d\d|since\s+(?:19|20)\d\d)\b",
    re.I,
)


def _time_words(message: str) -> list[str]:
    return [m.group(0) for m in _TIME_WORDS.finditer(message or "")][:4]


def _fmt(day: date) -> str:
    return f"{day.day} {calendar.month_name[day.month]} {day.year}"


def _span(start: date, end: date) -> str:
    return _fmt(start) if start == end else f"{_fmt(start)} to {_fmt(end)}"


def _count(word: str) -> int | None:
    word = word.lower()
    if word.isdigit():
        return int(word)
    return _NUMBERS.get(word)


def resolve_time(phrase: str, today: date | None = None) -> str:
    """A time phrase as real dates ('' when it names none Kiwibro can work out)."""
    today = today or datetime.now().astimezone().date()
    low = " ".join(re.sub(r"[^\w\s]", " ", phrase.lower()).split())
    words = low.split()
    if not words:
        return ""
    if low in ("today", "tonight", "this morning", "this evening", "this afternoon", "now", "right now"):
        return _fmt(today)
    if "yesterday" in words:
        return _fmt(today - timedelta(days=1))
    if "tomorrow" in words:
        return _fmt(today + timedelta(days=1))
    monday = today - timedelta(days=today.weekday())
    unit_at = next((i for i, w in enumerate(words) if w.rstrip("s") in ("day", "week", "month", "year", "weekend")), -1)
    if unit_at >= 0:
        unit = words[unit_at].rstrip("s")
        lead = words[unit_at - 1] if unit_at else ""
        number = _count(lead) if lead else None
        if "ago" in words and number:
            if unit == "day":
                return _fmt(today - timedelta(days=number))
            if unit == "week":
                return _span(today - timedelta(days=7 * number + 3), today - timedelta(days=7 * number - 3))
            if unit == "month":
                month = (today.month - 1 - number) % 12 + 1
                year = today.year + (today.month - 1 - number) // 12
                return f"{calendar.month_name[month]} {year}"
            if unit == "year":
                return str(today.year - number)
        if number and ("past" in words or "last" in words):
            days = {"day": 1, "week": 7, "month": 30, "year": 365}.get(unit, 0) * number
            return _span(today - timedelta(days=days), today) if days else ""
        if number and ("next" in words or "coming" in words or "from" in words):
            days = {"day": 1, "week": 7, "month": 30, "year": 365}.get(unit, 0) * number
            return _span(today, today + timedelta(days=days)) if days else ""
        shift = -1 if "last" in words or "past" in words else 1 if "next" in words or "coming" in words else 0
        if "this" in words or shift:
            if unit == "week":
                start = monday + timedelta(days=7 * shift)
                return _span(start, start + timedelta(days=6))
            if unit == "weekend":
                saturday = monday + timedelta(days=5 + 7 * shift)
                return _span(saturday, saturday + timedelta(days=1))
            if unit == "month":
                month = (today.month - 1 + shift) % 12 + 1
                year = today.year + (today.month - 1 + shift) // 12
                return f"{calendar.month_name[month]} {year}"
            if unit == "year":
                return str(today.year + shift)
            if unit == "day" and shift:
                return _fmt(today + timedelta(days=shift))
    for word in words:
        if word in _DAYS:
            target = _DAYS[word]
            if "last" in words:
                back = (today.weekday() - target) % 7 or 7
                return _fmt(today - timedelta(days=back))
            ahead = (target - today.weekday()) % 7
            if "next" in words and ahead == 0:
                ahead = 7
            return _fmt(today + timedelta(days=ahead))
    year = next((int(w) for w in words if re.fullmatch(r"(?:19|20)\d\d", w)), None)
    month = next((_MONTHS[w] for w in words if w in _MONTHS), None)
    if "since" in words and (year or month):
        start = date(year or today.year, month or 1, 1)
        if start > today:
            start = date(start.year - 1, start.month, 1)
        return _span(start, today)
    if month and year:
        return f"{calendar.month_name[month]} {year}"
    if month and ("last" in words or "next" in words or "in" in words or "this" in words):
        year = today.year - 1 if "last" in words and month >= today.month else today.year
        if "next" in words and month <= today.month:
            year = today.year + 1
        return f"{calendar.month_name[month]} {year}"
    return ""


def resolve_times(phrases: list[str], today: date | None = None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for phrase in phrases[:4]:
        span = resolve_time(phrase, today)
        plain = " ".join(phrase.split())
        # a year alone ("in 2019") needs no working out; only spell out what takes counting
        if span and span != plain and not re.fullmatch(r"(?:in\s+)?\d{4}", plain.lower()):
            out.append((plain, span))
    return out
