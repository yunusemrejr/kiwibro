"""A question about the user's own saved facts, checked against what is saved.

"Where do I live?" has its answer in memory ("User lives in Ankara"). A small model deep in a
session sometimes answers it with an older answer, or says it was never told. The fact's own
name or figure has to be in the reply: if it is not, the model is shown the fact once more, and
after that Kiwibro says the fact itself. Only questions that ask for the fact are checked;
"what should I cook tonight?" is the model's to answer.
"""

from __future__ import annotations

import re
from typing import Any

from . import terms

MAX_WORDS = 14
_ASKS_FOR_A_FACT = re.compile(
    r"^\W*(?:(?:and|so|ok|okay|hey|kiwibro|now|then|also)\W+)*(?:"
    r"what(?:'s|s|\s+is|\s+are|\s+was)\s+(?:the\s+name\s+of\s+)?my\b|"
    r"(?:where|when)\s+(?:do|did|am|was|is)\s+(?:i|my)\b|who\s+(?:is|are|was)\s+my\b|how\s+old\s+(?:am\s+i|is\s+my)\b|"
    r"which\s+\w+\s+(?:do|did|am)\s+i\b|what\s+(?:do|did)\s+i\s+(?:do|like|prefer|use|own|have|study|drive|work)\b|"
    r"(?:do|can)\s+you\s+(?:remember|know|recall)\b.*\b(?:my|i)\b|what\s+do\s+you\s+(?:remember|know)\s+about\s+m[ey]\b)",
    re.I,
)
_ABOUT_USER = re.compile(r"^\W*(?:the\s+)?user(?:(?P<owns>'s|’s)|\b)\s*", re.I)
_ATTRIBUTE = frozenset({"name", "call", "remember", "know", "recall", "tell"})  # they ask for the fact, they are not its subject
_BE = {"is": "are", "has": "have", "was": "were", "does": "do", "doesn't": "don't", "isn't": "aren't", "hasn't": "haven't"}
_WORD = re.compile(r"[^\W_][\w'’-]*")


def asks_for_a_fact(message: str) -> bool:
    """The question asks for something the user told Kiwibro about themselves."""
    text = (message or "").replace("’", "'").strip()
    return len(text.split()) <= MAX_WORDS and bool(_ASKS_FOR_A_FACT.match(text))


def _stems(text: str) -> set[str]:
    return {terms.stem(word) for word in terms.content_words(text)} - {"user"}


def answers(message: str, facts: list[str]) -> list[tuple[str, tuple[str, ...]]]:
    """The saved facts a question asks for, each with the words that are its answer: the names
    and figures it has and the question does not (all its other words, for a fact without any)."""
    asked = _stems(re.sub(r"\bwhats\b", "what is", message or "", flags=re.I))
    subject = asked - {terms.stem(word) for word in _ATTRIBUTE}
    found: list[tuple[str, tuple[str, ...]]] = []
    for fact in facts:
        if not asked or not _ABOUT_USER.match(fact) or not (subject or asked) <= _stems(fact):
            continue
        if not subject:  # "what's my name?" asks about the user: "User's name is Ana", not "User has a cat named Pamuk"
            first = terms.content_words(_ABOUT_USER.sub("", fact))[:1]
            if not first or terms.stem(first[0]) not in asked:
                continue
        new = [word for word in _WORD.findall(_ABOUT_USER.sub("", fact))
               if terms.stem(word.lower()) not in asked and word.lower() not in terms.STOPWORDS]
        named = [word for word in new if word[0].isupper() or word[0].isdigit()]
        if named or new:
            found.append((fact, tuple(named or new)))
    return found


def said(reply: str, found: list[tuple[str, tuple[str, ...]]]) -> bool:
    """The reply gives the answer word of one of the facts."""
    spoken = {word.lower().strip("'’") for word in _WORD.findall(reply or "")}
    spoken |= {terms.stem(word) for word in spoken}
    return any(word.lower() in spoken or terms.stem(word.lower()) in spoken for _fact, telling in found for word in telling)


def to_the_user(fact: str) -> str:
    """A saved fact as it is said to the user: "User lives in Ankara" -> "you live in Ankara"."""
    about = _ABOUT_USER.match(fact)
    rest = fact[about.end():].strip().rstrip(".") if about else fact.strip().rstrip(".")
    if not about:
        return rest
    if about.group("owns"):
        return f"your {rest}"
    verb, _, tail = rest.partition(" ")
    low = verb.lower()
    if low in _BE:
        verb = _BE[low]
    elif low.endswith("ies") and len(low) > 4:
        verb = low[:-3] + "y"
    elif low.endswith(("sses", "shes", "ches", "xes", "zes", "oes")):
        verb = low[:-2]
    elif low.endswith("s") and not low.endswith("ss"):
        verb = low[:-1]
    return f"you {verb} {tail}".strip()


def told(found: list[tuple[str, tuple[str, ...]]], newest: Any = None) -> str:
    """The answer in the saved fact's own words."""
    fact = found[0][0] if newest is None else newest
    return f"From what you told me: {to_the_user(fact)}."
