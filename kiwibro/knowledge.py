"""A read-only knowledge base Kiwibro looks into when the model would have to guess.

``knowledge/base/*.md`` ships with Kiwibro: one ``# Topic`` line per file, then entries made
of a ``## Title`` and a short body. They hold facts that do not go out of date (commands,
language rules, formulas, constants, countries, dates of history), and nothing writes to
them at run time. Code searches them, not the model: before a question is answered
(``note``) and when a web search comes back empty (``passages``), so the base helps
offline and when the web has nothing.

A match has to be strong, or the base stays silent. Most of what the question asks about
must be in the entry and its subject must be in the entry's title; a word the base has
never seen weighs most. A message about anything else finds nothing, so the turn is not
steered anywhere.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import guidance

log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent / "knowledge" / "base"
MAX_BYTES = 200 * 1024 * 1024  # the whole base stays under this, however much is added
NOTE_HEADER = "## From the knowledge base"
NOTE_LEAD = (
    "Kiwibro's own reference, kept on this computer. It matches the question: "
    "use it if it answers it, leave it out if it does not."
)
MAX_MESSAGE = 300  # a longer message is a task or a story, not a question to look up
MAX_ENTRIES = 2
COVERED = 0.7  # share of the question's weight an entry must hold
IN_TITLE = 0.4  # share that must be in its title: the entry is about it, not just mentioning it
NEAR_BEST = 0.9  # a second entry is given only when it scores this close to the best
COMMON = 25  # a word in more entries than this (or than 1 in 80) cannot ask for an entry alone
KNOWLEDGE_INSTEAD = (
    "No relevant web results were found. Kiwibro's own reference, kept on this computer, holds this "
    "on it (there is no link to cite):"
)

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    """a an the of in on at to for from by with about as into and or but not no is are was were be been am do
    does did has have had can could should would will shall may might must it its this that these those there
    here what which who whom whose when where why how i you he she we they me him her us them my your his our
    their if then than so such very just also any some all each every more most much many one ones please
    tell explain describe define definition mean means meaning called know give show example examples use
    used using work works get make like between difference different vs versus exactly actually really
    briefly short simple simply terms plain thing something stand stands speak spoken convert calculate
    compute""".split()
)
# The forms a question about a subject takes. "Which tab is open", "how many tests failed" and a bare
# "and the header?" ask about the session, so they are left alone.
_ASKS = re.compile(
    r"^\W*(?:please\s+)?(?:"
    r"what(?:'s|s|\s+(?:is|are|was|were|does|do|did))|"
    r"who(?:'s|\s+(?:is|was|were|invented|discovered|wrote|founded|built|painted|created))|"
    r"when\s+(?:did|was|were|is)|where\s+(?:is|are|was|were)|why\s+(?:is|are|was|were|do|does|did)|"
    r"how\s+(?:do|does|did|can|could|should|would|to|is|are|was|were|long|far|big|old|fast|tall|heavy|hot|cold)|"
    r"how\s+(?:many|much)\b(?=.*\b(?:in|per|there)\b)|"
    r"define|explain|describe)\b",
    re.IGNORECASE,
)
# "Can you tell me what a mutex is" is the question "what a mutex is".
_POLITE = re.compile(
    r"^\W*(?:please\s+)?(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:tell\s+me|explain|describe|define)|"
    r"do\s+you\s+know|tell\s+me|i\s+(?:want|need|would\s+like)\s+to\s+know)\b[\s,:]*",
    re.IGNORECASE,
)
# "Is the server running?", "can you make it bigger": about this session or a request, not a subject.
_YES_NO = re.compile(
    r"^\W*(?:is|are|am|was|were|does|do|did|can|could|will|would|should|shall|has|have|had)\b", re.IGNORECASE
)
# What changes with the day is the web's to answer; the base holds only what stays true.
_LIVE = re.compile(
    r"\b(?:latest|newest|recent(?:ly)?|today|tonight|tomorrow|yesterday|now|news|this\s+(?:week|month|year))\b",
    re.IGNORECASE,
)
# "What does this error mean?" points at something in the chat, and "what are you doing" at Kiwibro.
_POINTS = re.compile(r"\b(?:this|these|those|that|it|they|them|above|attached|you|your|yours)\b", re.IGNORECASE)
# "How much disk space do I have?" asks about this person's own things, which no reference knows.
# "How do I undo a commit?" asks how something is done.
_OWN = re.compile(
    r"\b(?:my|mine|our|ours|we)\b|"
    r"(?<!how do )(?<!how can )(?<!how could )(?<!how would )(?<!how should )\bi\b",
    re.IGNORECASE,
)
# "The downloads folder", "what version is installed": what is on this computer takes a look, not a reference.
_HERE = re.compile(
    r"\b(?:installed|running|downloads|laptop|"
    r"the\s+(?:\w+\s+)?(?:folder|directory|file|project|repo|app|page|script|code|logs?|screen))\b",
    re.IGNORECASE,
)
_THE = re.compile(r"\bthe\b", re.IGNORECASE)
_HEAD = re.compile(r"[(,:]")  # a title's name ends where its gloss begins: "Mutex (lock, mutual exclusion)"
# British endings are read as the American ones, so "quantisation" finds "quantization".
_ROMAN = {"ii": "2", "iii": "3", "iv": "4", "vi": "6", "vii": "7", "viii": "8", "ix": "9"}
_SPELLED = (("isation", "ization"), ("ise", "ize"), ("yse", "yze"), ("our", "or"), ("tre", "ter"))


@dataclass(frozen=True)
class Entry:
    topic: str
    title: str
    text: str
    title_words: frozenset[str]
    words: frozenset[str]  # title, body, topic and file name
    head: frozenset[str]  # the title before its gloss: what the entry is named


@dataclass(frozen=True)
class _Index:
    entries: tuple[Entry, ...]
    weight: dict[str, float]  # rarer words weigh more
    unseen: float
    common: float  # the weight of a word too common to ask for an entry alone
    holding: dict[str, tuple[int, ...]]  # word -> the entries that have it


def _stem(word: str) -> str:
    """Singular and plural, a verb's forms, with and without a final e, land on one form."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        word = _undoubled(word[:-3])
        word += "e" if word.endswith("c") else ""  # "forcing" is "force", never "forc"
    elif len(word) > 4 and word.endswith("ed") and not word.endswith("eed"):
        word = _undoubled(word[:-2])
        word += "e" if word.endswith("c") else ""
    elif len(word) > 3 and word.endswith("es"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    for british, american in _SPELLED:
        if len(word) > 4 and word.endswith(british):
            word = word[: -len(british)] + american
            break
    # "France" keeps its e, or it would be the currency "franc".
    return word[:-1] if len(word) > 3 and word.endswith("e") and not word.endswith("ce") else word


def _undoubled(word: str) -> str:
    """ "stopp" from "stopped" is "stop"."""
    return word[:-1] if len(word) > 2 and word[-1] == word[-2] and word[-1] not in "aeiouls" else word


def _words(text: str, reference: bool = False) -> list[str]:
    """The words that carry meaning. In the base's own text a lone "I" is a numeral ("World War I")."""
    plain = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(ch for ch in plain if not unicodedata.combining(ch))
    found = []
    for word in _WORD.findall(plain):
        if word in _ROMAN or (reference and word == "i"):
            found.append(_ROMAN.get(word, "1"))
            continue
        if word in _STOP or (len(word) == 1 and not word.isdigit()):
            continue
        stem = _stem(word)
        if stem not in _STOP:
            found.append(stem)
    return found


def _parse(text: str, name: str) -> list[Entry]:
    topic, title, body = name.replace("-", " "), "", []
    entries: list[Entry] = []

    def close() -> None:
        if not (title and body):
            return
        in_title = _words(title, reference=True)
        in_body = _words(" ".join(body), reference=True)
        everything = frozenset(in_title) | frozenset(in_body) | frozenset(_words(f"{topic} {name}"))
        head = frozenset(_words(_HEAD.split(title, maxsplit=1)[0], reference=True))
        entries.append(Entry(topic, title, " ".join(body), frozenset(in_title), everything, head))

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("## "):
            close()
            title, body = line[3:].strip(), []
        elif line.startswith("# "):
            topic = line[2:].strip()
        elif line and title:
            body.append(line)
    close()
    return entries


@lru_cache(maxsize=1)
def _index() -> _Index:
    entries: list[Entry] = []
    size = 0
    for path in sorted(BASE.glob("*.md")):
        size += path.stat().st_size
        if size > MAX_BYTES:
            log.error("knowledge base: over %d MB, %s and the files after it are left out", MAX_BYTES >> 20, path.name)
            break
        entries.extend(_parse(path.read_text(encoding="utf-8"), path.stem))
    holding: dict[str, list[int]] = {}
    for number, entry in enumerate(entries):
        for word in entry.words:
            holding.setdefault(word, []).append(number)
    count = len(entries)

    def weigh(held_by: int) -> float:
        return math.log(1 + (count - held_by + 0.5) / (held_by + 0.5))

    return _Index(
        tuple(entries),
        {word: weigh(len(numbers)) for word, numbers in holding.items()},
        weigh(0),
        weigh(max(COMMON, count // 80)),
        {word: tuple(numbers) for word, numbers in holding.items()},
    )


def entries() -> tuple[Entry, ...]:
    return _index().entries


def _asked(text: str) -> frozenset[str]:
    """The words a question is about. Figures alone ask for a sum, not for an entry."""
    words = _words(text)
    return frozenset(words) if any(not word.isdigit() for word in words) else frozenset()


def search(query: str, limit: int = MAX_ENTRIES) -> list[Entry]:
    """The entries that answer ``query``, best first; none when the base does not cover it."""
    index = _index()
    asked = _asked(query)
    if not asked or not index.entries:
        return []
    weight = {word: index.weight.get(word, index.unseen) for word in asked}
    total = sum(weight.values())
    rare = [word for word, value in weight.items() if value >= index.common]
    if not rare and len(asked) == 1:
        return []  # "time", "error": one common word names no subject
    scored: list[tuple[float, str, Entry]] = []
    for number in index.holding.get(rare[0] if rare else min(asked, key=lambda word: len(index.holding.get(word, ()))), ()):
        entry = index.entries[number]
        # One word fits many entries: only the one named exactly so. Common words only ("speed of light",
        # "git branch") have to be the entry's title.
        if entry.head != asked if len(asked) == 1 else not (rare or asked <= entry.title_words):
            continue
        if any(word not in entry.words for word in rare):
            continue  # everything specific the question names has to be in the entry
        held = sum(weight[word] for word in asked if word in entry.words)
        titled = sum(weight[word] for word in asked if word in entry.title_words)
        if held < COVERED * total or titled < IN_TITLE * total:
            continue
        named = sum(weight[word] for word in asked if word in entry.head)  # "Spain" before "Port of Spain"
        scored.append((held + titled + named, entry.title, entry))
    scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    if not scored:
        return []
    best = scored[0][0]
    return [entry for score, _, entry in scored[:limit] if score >= NEAR_BEST * best]


def _lines(found: list[Entry]) -> str:
    return "\n".join(f"- {entry.title}: {entry.text}" for entry in found)


def note(message: str) -> str:
    """The block for the turn's note: what the base holds on this question, or nothing."""
    text = (message or "").strip().replace("\u2019", "'")
    if not text or len(text) > MAX_MESSAGE:
        return ""
    asked = _POLITE.sub("", text, count=1)
    if asked == text and not _ASKS.search(text):
        return ""
    if _YES_NO.search(asked) or _LIVE.search(asked) or _POINTS.search(asked) or _OWN.search(asked):
        return ""
    if _HERE.search(asked) or (_THE.search(asked) and len(_asked(asked)) == 1):
        return ""  # "where is the bug": one named thing with "the" is something in this session
    if guidance._WEB_EXTRA.search(text) or guidance._ABOUT_USER.search(text):
        return ""
    found = search(asked)
    return f"{NOTE_HEADER}\n{NOTE_LEAD}\n{_lines(found)}" if found else ""


def passages(query: str, limit: int = 3) -> str:
    """What the base holds on a search query, for a web search that found nothing."""
    return _lines(search(query, limit))
