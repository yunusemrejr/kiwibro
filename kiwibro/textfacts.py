"""Exact answers about the letters and words of a text, worked out in code.

A language model reads tokens, not letters, so "how many r's are in strawberry" is
a guess for it and a count for code. What is found here joins the sums of
``compute.worked`` in the turn note, and the reply is checked against it the same way.
"""

from __future__ import annotations

import re

MAX_FACTS = 3
_MAX_MESSAGE = 600  # a pasted document is not a question about a word
_QUOTED = r"""["\u201c'\u2018]([^"\u201c\u201d\u2018\u2019']{1,200})["\u201d'\u2019]"""
_WORD = r"([A-Za-z\u00c0-\u024f]{2,40})"
_LETTER_IN = re.compile(
    r"\bhow\s+many\s+(?:times\s+(?:does|do|is)\s+)?(?:the\s+)?(?:letters?\s+)?"
    r"""["\u201c'\u2018]?([A-Za-z\u00c0-\u024f])["\u201d'\u2019]?(?:'?s)?\s+"""
    r"(?:(?:are|is)\s+)?(?:there\s+)?(?:appears?\s+|occurs?\s+)?in\s+(?:the\s+)?(?:word\s+|name\s+)?"
    rf"(?:{_QUOTED}|{_WORD})",
    re.IGNORECASE,
)
_COUNT_IN = re.compile(
    rf"\bhow\s+many\s+(words|characters|letters)\s+(?:(?:are|is)\s+)?(?:there\s+)?in\s+(?:{_QUOTED}|(?:the\s+word\s+){_WORD})",
    re.IGNORECASE,
)
_REVERSE = re.compile(
    rf"\breverse\s+(?:the\s+)?(?:(?:word|text|string|name)\s+(?:{_QUOTED}|{_WORD})|{_QUOTED})|"
    rf"\bspell\s+(?:the\s+word\s+)?(?:{_QUOTED}|{_WORD})\s+backwards?\b",
    re.IGNORECASE,
)


def worked(message: str) -> list[tuple[str, str]]:
    """The letter and word questions a message asks, as (what was asked, exact result)."""
    text = (message or "").replace("\u2019", "'")
    if len(text) > _MAX_MESSAGE:
        return []
    found: list[tuple[str, str]] = []
    for match in _LETTER_IN.finditer(text):
        letter, target = match.group(1), match.group(2) or match.group(3)
        found.append((f'the letter "{letter}" in "{target}"', str(target.casefold().count(letter.casefold()))))
    for match in _COUNT_IN.finditer(text):
        what, target = match.group(1).lower(), match.group(2) or match.group(3)
        if what == "words":
            count = len(target.split())
        elif what == "letters":
            count = sum(char.isalpha() for char in target)
        else:
            count = len(target)
        found.append((f'{what} in "{target}"', str(count)))
    for match in _REVERSE.finditer(text):
        target = next(group for group in match.groups() if group)
        found.append((f'"{target}" backwards', target[::-1]))
    return found[:MAX_FACTS]
