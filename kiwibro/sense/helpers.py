"""Kiwi Sense helpers: five tiny trained deciders that read every user message.

They never write text and never call the language model, so they work the same for
every model and cost a few milliseconds per message on the CPU:

    Hlng  which language a text is in (English only: anything else is turned back)
    Hmsg  what the message wants: intent, whether the web is needed, what time it is
          about, its topic, and whether it says something about the user worth keeping
    Htag  which words are time phrases, which are demands on the answer (format, length,
          tools, audience) and which name what the message is about
    Hshf  how a message relates to the exchange before it: follow-up, same topic, new topic
    Hrel  whether a saved memory or note bears on the message

Each is one section of static/sense/sense.bin, run by the Kiwi Sense core (core.c, or
its Python twin). Training: scripts/helpers/ (labels from a teacher model, trained on
Google Colab).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from . import MAX_CHARS, _ASCII_PUNCT, _CJK_RANGES, normalize, softmax

INTENTS = ["chat", "question", "task", "share", "remember", "note", "recall", "feedback", "continue", "meta"]
WEB = ["none", "own", "asked", "live", "lookup"]
TIME = ["none", "clock", "now", "recent", "past", "future"]
TOPICS = ["software", "computers", "ai", "science", "math", "health", "money", "work", "learning", "writing",
          "society", "entertainment", "sports", "places", "home", "people", "kiwibro", "none"]
KEEP = ["none", "fact", "preference", "passing", "unsure"]
GROUPS = (("intent", INTENTS), ("web", WEB), ("time", TIME), ("topic", TOPICS), ("keep", KEEP))
SHIFT = ["followup", "same", "new"]
TAGS = ["O", "B-TIME", "I-TIME", "B-ASK", "I-ASK", "B-KEY", "I-KEY"]

LANG_TAG, MSG_TAG, TOKEN_TAG, SHIFT_TAG, RELATE_TAG = "Hlng", "Hmsg", "Htag", "Hshf", "Hrel"

# ---- text prepared for the language head: like normalize(), but the accents stay ----

_CODE_RE = re.compile(r"```.*?(?:```|\Z)|`[^`\n]*`", re.DOTALL)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[\w.+-]+@[\w-]+\.[\w.]+\b|(?:~|\.{0,2})/[\w./-]+|\b\w+\.(?:py|js|ts|md|txt|json|html|css|c|h|rs|go|java|sh|yml|yaml|toml|csv|pdf|png|jpg)\b")
_QUOTED_RE = re.compile(r"\"[^\"\n]{1,200}\"|“[^”\n]{1,200}”|«[^»\n]{1,200}»|„[^“”\n]{1,200}[“”]|「[^」\n]{1,200}」|(?<![\w])'[^'\n]{1,120}'(?![\w])")
_SEGMENT_RE = re.compile(r"[.!?…。！？;:\n]+")
_CODE_LINE_RE = re.compile(r"^\s*(?:[{}()\[\];<>/#=*+-]|def |class |import |from |return |if |for |while |const |let |var |function |SELECT |\$ )")


def lang_text(text: str) -> str:
    """Lower case, punctuation split off, accents kept (they tell languages apart)."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFC", text[:MAX_CHARS].lower()):
        category = unicodedata.category(ch)
        if ch in "\t\n\r" or category == "Zs":
            out.append(" ")
            continue
        if category[0] == "C":
            continue
        code = ord(ch)
        if code in _ASCII_PUNCT or category[0] in "PS" or any(lo <= code <= hi for lo, hi in _CJK_RANGES):
            out.append(f" {ch} ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def judged_part(text: str) -> str:
    """The words the user wrote themselves: code, links, paths and quotations taken out."""
    text = _CODE_RE.sub(" ", text or "")
    text = "\n".join(line for line in text.splitlines() if not _CODE_LINE_RE.match(line))
    text = _URL_RE.sub(" ", text)
    return _QUOTED_RE.sub(" ", text)


def segments(text: str) -> list[str]:
    return [part.strip() for part in _SEGMENT_RE.split(text) if part.strip()]


def letters(text: str) -> tuple[int, int]:
    """(letters, letters outside the Latin script)."""
    total = other = 0
    for ch in text:
        if ch.isalpha():
            total += 1
            if ord(ch) > 0x24F and not 0x1E00 <= ord(ch) <= 0x1EFF:
                other += 1
    return total, other


def word_count(text: str) -> int:
    return sum(1 for word in lang_text(text).split() if any(ch.isalpha() for ch in word))


def distinct_words(text: str) -> int:
    """How much a text says about its language: "kiwi kiwi kiwi …" says no more than "kiwi"."""
    return len({word for word in lang_text(text).split() if any(ch.isalpha() for ch in word)})


# ---- readings ----


@dataclass
class Language:
    english: bool
    code: str = "en"
    name: str = "English"
    confidence: float = 1.0
    judged: bool = False


@dataclass
class Reading:
    """What the helpers make of one message."""

    intent: str = "question"
    web: str = "none"
    time: str = "none"
    topic: str = "none"
    keep: str = "none"
    sure: dict[str, float] = field(default_factory=dict)
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    times: list[str] = field(default_factory=list)
    asks: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {"intent": self.intent, "web": self.web, "time": self.time, "topic": self.topic,
                "keep": self.keep, "times": self.times, "asks": self.asks, "keys": self.keys,
                "sure": {name: round(value, 2) for name, value in self.sure.items()}}


def join_words(words: list[str]) -> str:
    out = ""
    for word in words:
        if out and not (len(word) == 1 and word in ",.;:!?)%'" or out.endswith(("(", "$", "#"))):
            out += " "
        out += word
    return out


def spans(words: list[str], tags: list[str]) -> dict[str, list[str]]:
    """BIO tags -> the phrases of each kind, in order, without repeats."""
    found: dict[str, list[str]] = {"TIME": [], "ASK": [], "KEY": []}
    current: list[str] = []
    kind = ""

    def close() -> None:
        if current and kind:
            phrase = join_words(current).strip(" ,.;:")
            if phrase and phrase not in found[kind]:
                found[kind].append(phrase)

    for word, tag in zip(words, tags):
        if tag.startswith("B-") or (tag.startswith("I-") and tag[2:] != kind):
            close()
            kind, current = tag[2:], [word]
        elif tag.startswith("I-"):
            current.append(word)
        else:
            close()
            kind, current = "", []
    close()
    return found


class Helpers:
    """The helper heads of one loaded Kiwi Sense data file."""

    def __init__(self, sense) -> None:
        self.sense = sense
        meta = sense.meta.get("helpers") or {}
        self.languages: list[str] = list(meta.get("languages") or [])
        self.language_names: dict[str, str] = dict(meta.get("language_names") or {})
        self.thresholds: dict[str, float] = dict(meta.get("thresholds") or {})
        self.tags = {tag: sense.head_index(tag) for tag in (LANG_TAG, MSG_TAG, TOKEN_TAG, SHIFT_TAG, RELATE_TAG)}

    def has(self, tag: str) -> bool:
        return self.tags.get(tag) is not None

    # -- language --

    def language_scores(self, text: str) -> dict[str, float]:
        words: list[str] = []
        for word in lang_text(text).split():
            if not words or word != words[-1]:  # a word repeated in a row says no more than once
                words.append(word)
        raw = self.sense.head_text(LANG_TAG, " ".join(words))
        if raw is None:
            return {}
        return dict(zip(self.languages, softmax(raw)))

    def language(self, text: str) -> Language:
        """Is the user writing English? Only the words they wrote count: code, links and
        quotations are left out, and English around foreign words is English."""
        if not self.has(LANG_TAG) or not self.languages:
            return Language(True)
        part = judged_part(text)
        total, other = letters(part)
        words = distinct_words(part)
        if total < 2:
            return Language(True)
        weight = 0.0
        english = 0.0
        votes: dict[str, float] = {}
        strong_english = False
        floor = self.thresholds.get("english_segment", 0.8)
        for segment in segments(part):
            n = distinct_words(segment)  # each segment weighs by its distinct words
            if not n:
                continue
            scores = self.language_scores(segment)
            p_en = scores.get("en", 0.0)
            if n >= 3 and p_en >= floor:
                strong_english = True
            weight += n
            english += n * p_en
            for code, p in scores.items():
                if code != "en":
                    votes[code] = votes.get(code, 0.0) + n * p
        if not weight:
            return Language(True)
        share = english / weight
        code, mass = max(votes.items(), key=lambda kv: kv[1]) if votes else ("xx", 0.0)
        top = mass / weight
        name = self.language_names.get(code, "")
        result = Language(True, "en", "English", share, True)
        script = other / total if total else 0.0
        if words <= 2 and script < 0.6:
            return result  # "ciao", "merci", a name, one word repeated: too little to judge in the Latin script
        if strong_english and share >= self.thresholds.get("english_mixed", 0.15):
            return result
        if share < self.thresholds.get("english_share", 0.3):
            sure = top >= self.thresholds.get("language_named", 0.6) and code != "xx"
            return Language(False, code if sure else "xx", name if sure and name else "", 1.0 - share, True)
        return result

    # -- message --

    def read(self, text: str) -> Reading | None:
        if not self.has(MSG_TAG):
            return None
        normal = normalize(text)
        raw = self.sense.head_text(MSG_TAG, normal)
        if raw is None:
            return None
        reading = Reading()
        at = 0
        for name, labels in GROUPS:
            probs = softmax(raw[at:at + len(labels)])
            at += len(labels)
            scores = dict(zip(labels, probs))
            best = max(scores, key=scores.get)
            setattr(reading, name, best)
            reading.sure[name] = scores[best]
            reading.scores[name] = scores
        if self.has(TOKEN_TAG):
            words = normal.split()
            rows = self.sense.head_tokens(TOKEN_TAG, normal) or []
            tags = [TAGS[max(range(len(row)), key=row.__getitem__)] for row in rows]
            found = spans(words[:len(tags)], tags)
            reading.times, reading.asks, reading.keys = found["TIME"], found["ASK"], found["KEY"]
        return reading

    # -- pairs --

    def shift(self, before: str, message: str) -> tuple[str, float] | None:
        raw = self.sense.head_pair(SHIFT_TAG, normalize(before), normalize(message))
        if raw is None:
            return None
        probs = softmax(raw)
        best = max(range(len(SHIFT)), key=probs.__getitem__)
        return SHIFT[best], probs[best]

    def relates(self, message: str, item: str) -> float | None:
        """Chance that a saved item bears on the message (None without the helper)."""
        raw = self.sense.head_pair(RELATE_TAG, normalize(item), normalize(message))
        if raw is None:
            return None
        return softmax(raw)[1]
