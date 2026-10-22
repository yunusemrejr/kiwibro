"""Deterministic facts about the user, read from one chat message.

The small local model often forgets to call its ``remember`` tool. ``extract``
catches the clearest first-person statements ("my name is X", "I live in X",
"remember that I use tabs") with fixed patterns, so they are saved anyway.
Precision matters more than recall: a question, a hypothetical, quoted text,
code, reported speech about someone else or an instruction yields nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Same values as kiwibro/config.py MEMORY_KINDS. Kept literal so this module imports nothing from the app.
MEMORY_KINDS = ("fact", "preference", "project", "person", "decision")

MIN_CONTENT = 3
MAX_CONTENT = 300
MAX_SENTENCE = 300  # longer sentences are ignored outright


@dataclass(frozen=True)
class Fact:
    """One durable fact about the user. ``content`` is a third-person sentence."""

    kind: str
    content: str
    slot: str | None = None  # single-valued: a later fact with the same slot replaces this one
    forget: bool = False  # explicit "forget ..." request; ``content`` is what to forget


SLOT_PREFIXES: dict[str, str] = {
    "name": "User's name is ",
    "location": "User lives in ",
    "hometown": "User is from ",
    "job": "User works as ",
    "employer": "User works at ",
    "os": "User's operating system is ",
    "editor": "User's editor is ",
    "reply_language": "User's preferred reply language is ",
    "native_language": "User's native language is ",
}
_FAVORITE_SLOT = "favorite:"


def slot_prefix(slot: str) -> str:
    """Exact start of every content stored under ``slot``, including the ``favorite:<thing>`` family."""
    if slot in SLOT_PREFIXES:
        return SLOT_PREFIXES[slot]
    if slot.startswith(_FAVORITE_SLOT):
        return f"User's favorite {slot[len(_FAVORITE_SLOT):]} is "
    raise ValueError(f"unknown slot: {slot!r}")


# --- sentence preparation -----------------------------------------------------

_FENCE_RE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# A sentence ends at . ! or ? followed by whitespace or the end ("Pop!_OS" and "3.5" stay whole), or at a newline.
_SPLIT_RE = re.compile(r"([.!?]+(?=\s|$)|\n)")
_FILLER_RE = re.compile(
    r"^(?:(?:btw|by the way|also|fyi|oh and|ok so|okay so|actually|anyway|well)\b[\s,]*|note:\s*)+", re.IGNORECASE
)

# --- sentences that are never facts -------------------------------------------

_HYPO_RE = re.compile(
    r"\b(?:if|unless|would|could|suppose|imagine|pretend|hypothetically|wish|what if|let's say)\b",
    re.IGNORECASE,
)
_INSTRUCTION_RE = re.compile(
    r"\b(?:act as|roleplay|role-play|role play|pretend to be)\b"
    r"|\b(?:write|compose|draft|create|generate)\b.{0,40}?"
    r"\b(?:story|poem|dialogue|script|scene|novel|song|character|essay)\b",
    re.IGNORECASE,
)
_PERSON_WORDS = (
    "friend|friends|brother|sister|mom|mum|mother|dad|father|wife|husband|boss|colleague|coworker|"
    "partner|son|daughter|kid|kids|child|children|roommate|girlfriend|boyfriend|manager|teacher|"
    "cousin|uncle|aunt|grandma|grandpa|sibling|team|client|customer|doctor|professor"
)
_OTHER_START_RE = re.compile(
    rf"^(?:my\s+(?:{_PERSON_WORDS})|he|she|they|someone|somebody|everyone|nobody|people)\b",
    re.IGNORECASE,
)
# Reported speech: "he said", "Ali says", "they told me" (but never "I told ...").
_SPEECH_RE = re.compile(
    r"\b(?:he|she|they|someone|somebody|everyone|nobody|people|my\s+\w+|(?!I\b)(?-i:[A-Z])\w*)\s+"
    r"(?:said|says|say|told|tells|tell|wrote|writes|claims?|thinks?|asked|asks|mentioned|insisted)\b",
    re.IGNORECASE,
)

# --- helpers --------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\S+")
_NAME_WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")
# Where a capitalised slot value, or a phrase, ends: punctuation or a clause word.
_CLIP_RE = re.compile(
    r"[,;:()!?]|\s(?:and|or|but|because|since|when|which|that|who|where|while|if|so|for|with|at|in|on"
    r"|from|about|as|over|than|instead|rather|to)(?=\s|$)",
    re.IGNORECASE,
)
_LIST_CONJ = frozenset({"and", "or", "&", "/", "+"})
# Capitalised words that cannot be a name or a place ("My name is Later" is not a name).
_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "i", "it", "this", "that", "not", "just", "also", "so", "now", "here", "there",
        "later", "today", "tomorrow", "again", "maybe", "really", "sorry", "ok", "okay", "please", "thanks",
        "fine", "good", "bad", "tired", "ready", "back", "up", "down", "sure", "yes", "no", "hi", "hello",
        "hey", "well", "wrong", "right", "and", "or", "but", "my", "me", "you", "we", "he", "she", "they",
    }
)
# Words that make a restated clause unsafe ("User uses it." says nothing; "User likes that." is wrong).
_REFERENT_WORDS = frozenset(
    {
        "i", "i'm", "i've", "i'll", "i'd", "me", "you", "you're", "your", "yours", "it", "it's", "its",
        "they", "them", "he", "she", "him", "her", "we", "us", "this", "that", "that's", "these", "those",
        "one", "something", "anything", "everything", "nothing",
    }
)
_OBJECT_BLOCK = _REFERENT_WORDS | frozenset(
    {
        "my", "mine", "idea", "answer", "response", "ones", "thing", "things", "stuff", "problem",
        "problems", "issue", "bit", "lot", "up", "to", "not",
    }
)
_EDGE_PUNCT = ".,;:!?\"()[]"
_MY_RE = re.compile(r"my\s+", re.IGNORECASE)
_MINE_RE = re.compile(r"\b(?:my|mine)\b", re.IGNORECASE)

# "I use", "I'm", "I am", "I've", "I'll" plus an optional adverb and negation, then the verb.
_SUBJECT_RE = re.compile(
    r"i(?P<aux>'m|'ve|'ll|\s+am)\s+"
    r"|i\s+(?:(?P<adv>always|usually|never|often|sometimes|really|also|still)\s+)?"
    r"(?:(?P<neg>do\s+not|don't)\s+)?(?P<verb>[a-z]+)\s+",
    re.IGNORECASE,
)
_AUX_VERB = {"'m": "is", "am": "is", "'ve": "has", "'ll": "will"}
_VERB3 = {
    "use": "uses", "like": "likes", "love": "loves", "hate": "hates", "dislike": "dislikes",
    "prefer": "prefers", "live": "lives", "work": "works", "have": "has", "want": "wants",
    "need": "needs", "run": "runs", "code": "codes", "speak": "speaks", "write": "writes",
    "know": "knows", "own": "owns", "drive": "drives", "study": "studies", "teach": "teaches",
    "play": "plays", "read": "reads", "enjoy": "enjoys", "build": "builds", "develop": "develops",
    "go": "goes", "watch": "watches", "eat": "eats", "drink": "drinks", "keep": "keeps",
    "start": "starts", "manage": "manages", "maintain": "maintains", "travel": "travels",
}
_IRREGULAR_PAST = frozenset(
    {
        "was", "were", "had", "did", "went", "got", "bought", "made", "took", "gave", "saw", "met", "ran",
        "wrote", "knew", "thought", "left", "built", "moved", "grew", "kept", "found", "lost", "won",
        "sold", "paid", "felt", "ate", "drank", "drove", "rode", "spoke", "taught", "began",
    }
)
_LANGUAGES = (
    "English", "Turkish", "German", "French", "Spanish", "Italian", "Portuguese", "Dutch", "Russian",
    "Arabic", "Chinese", "Japanese", "Korean", "Hindi", "Persian", "Polish", "Ukrainian", "Swedish",
)
_LANG_CANON = {lang.lower(): lang for lang in _LANGUAGES}
_LANG_ALT = "|".join(_LANGUAGES)
_OS_NAMES = {
    "ubuntu": "Ubuntu", "debian": "Debian", "fedora": "Fedora", "arch": "Arch", "arch linux": "Arch Linux",
    "linux mint": "Linux Mint", "pop!_os": "Pop!_OS", "nixos": "NixOS", "windows 10": "Windows 10",
    "windows 11": "Windows 11", "macos": "macOS", "mac": "macOS",
}
_EDITOR_NAMES = {
    "vim": "Vim", "neovim": "Neovim", "nvim": "Neovim", "emacs": "Emacs", "vs code": "VS Code",
    "vscode": "VS Code", "visual studio code": "VS Code", "sublime text": "Sublime Text",
    "jetbrains": "JetBrains", "pycharm": "PyCharm", "intellij": "IntelliJ", "zed": "Zed", "helix": "Helix",
    "nano": "nano",
}
_LIKE_FORMS = {"like": "likes", "love": "loves", "hate": "hates", "dislike": "dislikes"}
_PREF_WORDS_RE = re.compile(r"\b(?:prefer\w*|likes?|loves?|hates?|wants?|always|never|dislikes?)\b", re.IGNORECASE)
_START_WORDS = frozenset(
    {
        "i", "i'm", "i'd", "i've", "my", "call", "you", "please", "remember", "forget", "don't", "keep",
        "note", "stop", "always", "usually", "never", "answer", "reply", "respond", "talk", "speak",
    }
)
# What may follow a name: nothing, punctuation, or a conjunction ("Bob the builder" is a nickname).
_NAME_TAIL = frozenset({"and", "but", "so", "btw", "because", "though", "i"})


def _alternation(names: dict[str, str]) -> str:
    """Regex alternation over the keys, longest first, so "arch linux" wins over "arch"."""
    return "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))


def _norm(token: str) -> str:
    return token.lower().strip(_EDGE_PUNCT)


def _canon_key(text: str) -> str:
    return " ".join(text.lower().split())


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _has_block(text: str, block: frozenset[str] = _OBJECT_BLOCK) -> bool:
    return any(_norm(token) in block for token in text.split())


def _has_referent(text: str) -> bool:
    return _has_block(text, _REFERENT_WORDS)


def _tail_ok(rest: str) -> bool:
    """False when a captured name or place is only the first item of a list ("Turkey and Germany")."""
    words = rest.split()
    return not (
        len(words) >= 2 and words[0].lower() in _LIST_CONJ and words[1][:1].isupper()
        and words[1].split("'")[0] != "I"  # "… and I'm on Ubuntu" starts a new clause, not a list
    )


def _name_tail_ok(rest: str) -> bool:
    words = rest.split()
    return not words or words[0][0] in ",;(" or words[0].lower() in _NAME_TAIL


def _clip(text: str) -> str:
    """Text up to the first punctuation or clause word."""
    match = _CLIP_RE.search(text)
    return (text[: match.start()] if match else text).strip()


def _phrase(text: str, max_words: int) -> str | None:
    words = _clip(text).split()
    if not 1 <= len(words) <= max_words:
        return None
    return " ".join(words)


def _proper_run(text: str, max_words: int) -> tuple[list[str], str]:
    """Leading capitalised name-like words of ``text``, and the text after them. A comma ends the run."""
    words: list[str] = []
    rest = text
    for match in _TOKEN_RE.finditer(text):
        raw = match.group()
        word = raw.rstrip(",")
        if (
            len(words) == max_words
            or not _NAME_WORD_RE.fullmatch(word)
            or not word[0].isupper()
            or word.lower() in _STOP_WORDS
        ):
            break
        words.append(word)
        rest = text[match.end():]
        if word != raw:
            break
    return words, rest


def _is_about_other(text: str) -> bool:
    return bool(_OTHER_START_RE.match(text) or _SPEECH_RE.search(text))


def _verb_phrase(subject: re.Match[str]) -> str | None:
    verb = subject.group("verb").lower()
    adv = subject.group("adv")
    if subject.group("neg"):
        # "I don't use X" -> "User doesn't use X". Only plain present-tense verbs qualify.
        if adv or verb.endswith("ed") or verb in _IRREGULAR_PAST:
            return None
        return f"doesn't {verb}"
    if verb in _VERB3:
        form = _VERB3[verb]
    elif verb.endswith("ed") or verb in _IRREGULAR_PAST:
        form = verb
    else:
        return None
    return f"{adv.lower()} {form}" if adv else form


def _clause_content(clause: str) -> str | None:
    """Third-person body for something the user said: "User uses tabs", or "User noted: ..." for a
    neutral clause. None when the clause cannot be restated safely."""
    clause = clause.strip()
    words = clause.split()
    if len(words) < 2 or _norm(words[0]) == "to":
        return None
    subject = _SUBJECT_RE.match(clause)
    if subject is not None:
        tail = clause[subject.end():]
        if subject.group("aux"):
            head = _AUX_VERB[subject.group("aux").strip().lower()]
        else:
            head = _verb_phrase(subject)
            if head is None:
                return None
        body = f"User {head} "
    elif (my := _MY_RE.match(clause)) is not None:
        tail = clause[my.end():]
        body = "User's "
    else:
        if _has_referent(clause):
            return None
        return "User noted: " + _cap(_MINE_RE.sub("the user's", clause))
    if _has_referent(tail):
        return None
    return (body + _MINE_RE.sub("the user's", tail)).rstrip()


def _fact(kind: str, body: str | None, slot: str | None = None, forget: bool = False) -> Fact | None:
    if body is None:
        return None
    content = body.rstrip(" ,;:-") + "."
    if not MIN_CONTENT <= len(content) <= MAX_CONTENT:
        return None
    return Fact(kind, content, slot, forget)


# --- slots (single-valued facts) ----------------------------------------------

_NAME_RE = re.compile(
    r"(?:my name is|call me|i'?m called|i am called|you can call me)\s+(?P<rest>.+)", re.IGNORECASE
)
_LOCATION_RE = re.compile(
    r"(?:i live in|i'?m based in|i am based in|i'?m living in|i am living in)\s+(?P<rest>.+)", re.IGNORECASE
)
_HOMETOWN_RE = re.compile(r"(?:i'?m from|i am from|i come from)\s+(?P<rest>.+)", re.IGNORECASE)
_EMPLOYER_RE = re.compile(r"(?:i work at|i work for)\s+(?P<rest>.+)", re.IGNORECASE)
# Not "my job is a …": that is as often a judgement ("a nightmare") as a profession.
_JOB_RE = re.compile(r"i work as\s+(?P<art>an?)\s+(?P<x>.+)", re.IGNORECASE)
_EMPLOYED_RE = re.compile(r"(?:i'?m|i am) employed as\s+(?P<x>.+)", re.IGNORECASE)
_OS_RE = re.compile(
    r"(?:i'?m on|i am on|i use|i run)\s+(?P<name>" + _alternation(_OS_NAMES) + r")(?![\w-])", re.IGNORECASE
)
_EDITOR_RE = re.compile(
    r"(?:i use|i code in|my editor is)\s+(?P<name>" + _alternation(_EDITOR_NAMES) + r")(?![\w-])",
    re.IGNORECASE,
)
_REPLY_LANG_RE = re.compile(
    r"(?:please\s+)?(?:always\s+)?(?:answer|reply|respond|talk|speak)(?:\s+to\s+me)?\s+in\s+"
    r"(?P<lang>" + _LANG_ALT + r")(?![\w-])",
    re.IGNORECASE,
)
_NATIVE_RE = re.compile(r"my native language is\s+(?P<lang>" + _LANG_ALT + r")(?![\w-])", re.IGNORECASE)
_FAVORITE_RE = re.compile(
    r"my favou?rite\s+(?P<thing>[^\W\d_]+(?:\s+[^\W\d_]+)?)\s+(?:is|are)\s+(?P<val>.+)", re.IGNORECASE
)
_FAVORITE_BAD_WORDS = frozenset({"is", "are", "was", "were", "the", "a", "an", "my", "your", "our"})


def _name_fact(text: str) -> Fact | None:
    match = _NAME_RE.match(text)
    if match is None:
        return None
    words, rest = _proper_run(match.group("rest"), 3)
    if not words or not _tail_ok(rest) or not _name_tail_ok(rest):
        return None
    return _fact("fact", slot_prefix("name") + " ".join(words), "name")


def _location_fact(text: str) -> Fact | None:
    match = _LOCATION_RE.match(text)
    if match is None:
        return None
    words, rest = _proper_run(match.group("rest"), 4)
    if not words or not _tail_ok(rest):
        return None
    return _fact("fact", slot_prefix("location") + " ".join(words), "location")


def _hometown_fact(text: str) -> Fact | None:
    match = _HOMETOWN_RE.match(text)
    if match is None:
        return None
    words, rest = _proper_run(match.group("rest"), 4)
    if not words or not _tail_ok(rest):
        return None
    return _fact("fact", slot_prefix("hometown") + " ".join(words), "hometown")


def _employer_fact(text: str) -> Fact | None:
    match = _EMPLOYER_RE.match(text)
    if match is None:
        return None
    words, rest = _proper_run(match.group("rest"), 4)
    if not words or not _tail_ok(rest):
        return None
    return _fact("fact", slot_prefix("employer") + " ".join(words), "employer")


def _job_fact(text: str) -> Fact | None:
    match = _JOB_RE.match(text)
    if match is not None:
        # "I work as a teacher": the article is required.
        phrase = _phrase(match.group("x"), 4)
        if phrase is None or _has_block(phrase):
            return None
        return _fact("fact", slot_prefix("job") + f"{match.group('art').lower()} {phrase}", "job")
    match = _EMPLOYED_RE.match(text)
    if match is not None:
        phrase = _phrase(match.group("x"), 4)
        if phrase is None or _has_block(phrase):
            return None
        return _fact("fact", slot_prefix("job") + phrase, "job")
    return None


def _os_fact(text: str) -> Fact | None:
    match = _OS_RE.match(text)
    if match is None:
        return None
    return _fact("fact", slot_prefix("os") + _OS_NAMES[_canon_key(match.group("name"))], "os")


def _editor_fact(text: str) -> Fact | None:
    match = _EDITOR_RE.match(text)
    if match is None:
        return None
    return _fact("fact", slot_prefix("editor") + _EDITOR_NAMES[_canon_key(match.group("name"))], "editor")


def _reply_language_fact(text: str) -> Fact | None:
    match = _REPLY_LANG_RE.match(text)
    if match is None:
        return None
    lang = _LANG_CANON[match.group("lang").lower()]
    return _fact("preference", slot_prefix("reply_language") + lang, "reply_language")


def _native_language_fact(text: str) -> Fact | None:
    match = _NATIVE_RE.match(text)
    if match is None:
        return None
    lang = _LANG_CANON[match.group("lang").lower()]
    return _fact("fact", slot_prefix("native_language") + lang, "native_language")


def _favorite_fact(text: str) -> Fact | None:
    match = _FAVORITE_RE.match(text)
    if match is None:
        return None
    thing = _canon_key(match.group("thing"))
    if any(word in _FAVORITE_BAD_WORDS for word in thing.split()):
        return None
    value = _phrase(match.group("val"), 4)
    if value is None or _has_block(value):
        return None
    return _fact("preference", f"User's favorite {thing} is {value}", f"{_FAVORITE_SLOT}{thing}")


_SLOT_MATCHERS = (
    _name_fact,
    _location_fact,
    _hometown_fact,
    _job_fact,
    _employer_fact,
    _os_fact,
    _editor_fact,
    _reply_language_fact,
    _native_language_fact,
    _favorite_fact,
)


def _slot_fact(text: str) -> Fact | None:
    for matcher in _SLOT_MATCHERS:
        fact = matcher(text)
        if fact is not None:
            return fact
    return None


# --- explicit memory requests, preferences and projects -------------------------

_SAVE_RE = re.compile(
    r"(?:please\s+)?(?:remember(?!\s+(?:when|to|how|if|what|who|where|why)\b)(?:\s+that)?"
    r"|don'?t\s+forget(?:\s+that)?|keep\s+in\s+mind(?:\s+that)?|note\s+that)\s+(?P<x>.+)",
    re.IGNORECASE,
)
_FORGET_RE = re.compile(
    r"(?:please\s+)?(?:forget(?:\s+(?:that|about))?|stop\s+remembering)\s+(?P<x>.+)", re.IGNORECASE
)
_PREFER_RE = re.compile(r"i (?:really )?prefer\s+(?P<rest>.+)", re.IGNORECASE)
_PREFER_SPLIT_RE = re.compile(r"\s+(?:over|to|rather than|instead of)\s+", re.IGNORECASE)
_LIKE_RE = re.compile(r"i (?:really )?(?P<verb>like|love|hate|dislike)\s+(?P<rest>.+)", re.IGNORECASE)
_HABIT_RE = re.compile(r"i (?P<adv>always|usually|never) use\s+(?P<rest>.+)", re.IGNORECASE)
_PROJECT_RE = re.compile(
    r"(?:i'?m|i am) (?:working on|building|developing)\s+(?P<rest>.+)|my project is\s+(?P<rest2>.+)",
    re.IGNORECASE,
)


def _forget_fact(text: str) -> Fact | None:
    match = _FORGET_RE.match(text)
    if match is None or _is_about_other(match.group("x")):
        return None
    fact = _fact("fact", _clause_content(match.group("x")))
    return None if fact is None else Fact(fact.kind, fact.content, None, True)


def _save_fact(text: str) -> Fact | None:
    match = _SAVE_RE.match(text)
    if match is None:
        return None
    clause = match.group("x").strip()
    if _is_about_other(clause):
        return None
    slot = _slot_fact(clause)
    if slot is not None:
        return slot
    kind = "preference" if _PREF_WORDS_RE.search(clause) else "fact"
    return _fact(kind, _clause_content(clause))


def _preference_fact(text: str) -> Fact | None:
    match = _PREFER_RE.match(text)
    if match is not None:
        parts = _PREFER_SPLIT_RE.split(match.group("rest"), maxsplit=1)
        first = _phrase(parts[0], 5)
        if first is None:
            return None
        body = first
        if len(parts) == 2:
            second = _phrase(parts[1], 5)
            if second is None:
                return None
            body = f"{first} over {second}"
        if _has_block(body):
            return None
        return _fact("preference", f"User prefers {body}")
    match = _LIKE_RE.match(text)
    if match is not None:
        phrase = _phrase(match.group("rest"), 4)
        if phrase is None or _has_block(phrase):
            return None
        return _fact("preference", f"User {_LIKE_FORMS[match.group('verb').lower()]} {phrase}")
    match = _HABIT_RE.match(text)
    if match is not None:
        phrase = _phrase(match.group("rest"), 5)
        if phrase is None or _has_block(phrase):
            return None
        return _fact("preference", f"User {match.group('adv').lower()} uses {phrase}")
    return None


# A project description may go on ("a thesis about memory systems"); it ends at
# punctuation or a new clause, not at a preposition.
_PROJECT_CLIP_RE = re.compile(
    r"[,;:()!?]|\s(?:and|or|but|because|since|which|while|if|so|though|although)(?=\s|$)", re.IGNORECASE
)


def _project_fact(text: str) -> Fact | None:
    match = _PROJECT_RE.match(text)
    if match is None:
        return None
    rest = (match.group("rest") or match.group("rest2") or "").strip()
    cut = _PROJECT_CLIP_RE.search(rest)
    words = (rest[: cut.start()] if cut else rest).split()
    name = " ".join(words) if 1 <= len(words) <= 10 else None
    if name is None or _has_block(name):
        return None
    words = name.split()
    # A single word must be a capitalised name ("Kiwibro"), not "building apps".
    if len(words) == 1 and not (words[0][0].isupper() and _NAME_WORD_RE.fullmatch(words[0])):
        return None
    return _fact("project", f"User is working on {name}")


# "My name is Yunus and I'm on Ubuntu", "Actually I moved, I live in Istanbul now": each
# first-person clause is its own statement. Guards run on the whole sentence first, so the
# "If I won, I'd live in Paris" kind never reaches the clauses.
_CLAUSE_RE = re.compile(
    r"(?:\s*[,;]\s*|\s+)(?:and|but|so|plus)\s+(?=(?:i|i'm|i've|i am|my)\b)|\s*[,;]\s*(?=(?:i|i'm|i've|i am|my)\b)",
    re.IGNORECASE,
)
_COMMAND_RE = re.compile(r"^(?:please\s+)?(?:remember|forget|don't forget|keep in mind|note that|stop remembering)\b",
                         re.IGNORECASE)


def _clause_fact(clause: str) -> Fact | None:
    text = _FILLER_RE.sub("", clause, count=1).strip()
    # Every pattern is anchored at the start, so a clause must open with one of these words.
    first = text.split(None, 1)[0].lower() if text else ""
    if first not in _START_WORDS or _is_about_other(text):
        return None
    for matcher in (_forget_fact, _save_fact, _slot_fact, _preference_fact, _project_fact):
        fact = matcher(text)
        if fact is not None:
            return fact
    return None


def _sentence_facts(sentence: str) -> list[Fact]:
    text = _FILLER_RE.sub("", sentence, count=1).strip()
    if not text or _HYPO_RE.search(text) or _INSTRUCTION_RE.search(text) or _is_about_other(text):
        return []
    # "Forget that I live in Berlin and I use vim": the whole sentence is the command's scope.
    clauses = [text] if _COMMAND_RE.match(text) else _CLAUSE_RE.split(text)
    return [fact for fact in map(_clause_fact, clauses) if fact is not None]


def _clean(message: str) -> str:
    """Drop code, quoted lines and URLs. Removed spans become spaces so words do not merge."""
    text = message.replace("’", "'").replace("‘", "'")
    text = _FENCE_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    lines = ["" if line.lstrip().startswith(">") else _URL_RE.sub(" ", line) for line in text.split("\n")]
    return "\n".join(lines)


def _sentences(text: str):
    """Yield (sentence, is_question) pairs in order, skipping empty and over-long sentences."""
    parts = _SPLIT_RE.split(text)
    for index in range(0, len(parts), 2):
        sentence = parts[index].strip()
        separator = parts[index + 1] if index + 1 < len(parts) else ""
        if sentence and len(sentence) <= MAX_SENTENCE:
            yield sentence, "?" in separator


def extract(message: str, *, limit: int = 5) -> list[Fact]:
    """Facts the user stated about themselves in ``message``, in message order, at most ``limit``."""
    if not message or limit <= 0:
        return []
    facts: list[Fact] = []
    seen: set[tuple[str, bool]] = set()
    for sentence, is_question in _sentences(_clean(message)):
        if is_question:
            continue
        for fact in _sentence_facts(sentence):
            if (fact.content, fact.forget) in seen:
                continue
            seen.add((fact.content, fact.forget))
            facts.append(fact)
            if len(facts) >= limit:
                return facts
    return facts
