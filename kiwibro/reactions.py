"""How a message reacts to the answer before it.

"all wrong and old news", "those aren't today", "no. use web", "nah, there was something
about a woman business owner": none of these is a request of its own. Each turns the last
answer down and means "do what I asked again, properly". Read as a new request, the words
themselves get searched ("all wrong" finds a TV show) or the model defends what it wrote.

This module tells such a message apart and finds the request it is about, so Kiwibro can
do that request again: with a new and deeper search, with whatever the message adds, and
with the answer that was turned down set aside so it is not repeated.

Several weak signs are added up: the helper's reading of the intent (feedback), words that
complain about an answer, a "no" at the start, words that point back at the answer, a call
to try again or to use the web. A question with a subject of its own, or a long message,
counts against. Nothing here calls a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from . import plan, terms

REJECT = "reject"  # the answer was wrong, stale or off: the request before it is done again
REFINE = "refine"  # not what was meant: the message says what was ("no, the one about …")

SURE = 0.8  # the signs together must reach this
MAX_WORDS = 30  # a longer message says something of its own
ADDED_MIN = 2  # words a message must add for it to steer the search, not just restart it
QUERY_CHARS = 160

_WRONG = re.compile(
    r"\b(?:wrong|incorrect|inaccurate|untrue|false|mistaken|nonsense|rubbish|garbage|useless|fake|invented|"
    r"not\s+(?:true|right|correct|accurate|real|helpful|useful|relevant)|"
    r"(?:made|making|make|makes)\s+(?:that|this|it|things|stuff|them)\s+up|made[- ]up|hallucinat\w*|"
    r"not\s+what\s+i\s+(?:asked|meant|wanted|said|was\s+asking)|did(?:n'?t|\s+not)\s+ask|"
    r"(?:does|did|do)(?:n'?t|\s+not)\s+(?:answer|help|make\s+sense)|off[- ]topic|irrelevant|(?:way|far)\s+off|off\s+base|"
    r"guess(?:ing|ed|es)?|instead\s+of\s+(?:guessing|making))\b",
    re.I,
)
_STALE = re.compile(
    r"\b(?:(?:too|so|very|really|all|just|pretty)\s+old|old\s+(?:news|info|information|data|stuff|results?|stories|articles?|ones?)|"
    r"(?:is|are|'s|its|thats|was|were)\s+old|outdated|out\s+of\s+date|stale|expired|"
    r"(?:not|n'?t|arent|isnt|wasnt|werent)\s+(?:from\s+)?(?:today|recent|new|current|the\s+latest|up\s+to\s+date|"
    r"this\s+(?:week|year|month))|(?:days?|weeks?|months?|years?)\s+(?:old|ago)|from\s+last\s+(?:year|month|week))\b",
    re.I,
)
_AGAIN = re.compile(
    r"\b(?:again|redo|retry|once\s+more|one\s+more\s+time|properly|"
    r"(?:do|try|search|look|dig|check|read)\s+(?:it\s+|that\s+|this\s+)?(?:better|harder|deeper|more\s+carefully|carefully))\b", re.I)
_USE_WEB = re.compile(
    r"\b(?:use|check|try|via|with|from|on)\s+(?:the\s+)?(?:web|internet|net|google|search)\b|"
    r"\b(?:search|google|look)\s+(?:it\s+|that\s+|this\s+)?(?:up|again|online|properly|better|harder|deeper|the\s+web)\b|"
    r"^\W*(?:search|google)\W*$",
    re.I,
)
# A "no" that turns the answer down stands alone ("nope"), is set off ("no, it was 1912"), or goes
# on about the answer or what was meant ("no thats wrong", "nah there was something about …").
# "no wifi on the train", "no worries", "nah i'm good" are something else.
_DECLINES = (  # "no thanks", "nah its fine", "no, not needed": the offer is turned down, not the answer
    r"(?:thanks|thank\s+you|worries|problem|need|rush|never\s?mind|forget\s+it|leave\s+it|not\s+(?:needed|necessary)|"
    r"(?:i'?m|i\s+am|we'?re|it'?s|that'?s|thats|all)\s+(?:good|fine|ok|okay|alright|set|enough|all)|that\s+is\s+(?:fine|ok|okay|enough|all))\b"
)
_NO_LEAD = re.compile(
    rf"^\W*(?:still\s+)?(?:(?:no+|nope|nah)(?:\W*$|\s*[,.!;:-]+\s*(?!{_DECLINES})|"
    rf"\s+(?!{_DECLINES})(?=(?:that|this|these|those|it|its|thats|there|still|wrong|not|you|u|ur|your|the\s+(?:answer|one|other)|"
    r"i\s+(?:meant|mean|asked|said|wanted|was\s+(?:asking|talking)))\b))|"
    r"(?:wrong|incorrect|false)(?:\W*$|\s*[,.!;:-])|"
    r"not\s+(?=(?:the\s+(?:one|right|same|latest|answer)|what|today'?s?|from|recent|even|it|that|this)\b)|"
    r"(?:that|this|these|those|it)(?:'?s|\s+is|\s+are|\s+was|\s+were)?\s+(?:not|n'?t)\b|"
    r"(?:that|this|these|those|it)\s*(?:isn'?t|aren'?t|wasn'?t|weren'?t|isnt|arent|wasnt|werent)\b)",
    re.I,
)
# "I meant the long one", "I asked for a summary", "I wanted a Python example": said first, it
# is the whole point of the message.
_MEANT_LEAD = re.compile(
    r"^\W*(?:(?:but|well|actually|sorry)\W+)*i\s+(?:meant|asked\s+(?:for|about|you)|was\s+asking\s+(?:for|about)|wanted)\s+(?!to\b)",
    re.I,
)
# An order to search for what was asked before: "google it", "look this up online", "search for it".
_SEARCH_IT = re.compile(
    r"\b(?:search|google|look)\s+(?:(?:for|at|into)\s+)?(?:it|this|that)\b(?:\s+up\b)?|\blook\s+up\s+(?:it|this|that)\b", re.I)
# An order to do it over: "try again", "search again", "redo it".
_DO_OVER = re.compile(
    r"(?:^|[,.;!?]\s*)\W*(?:(?:pls|please|just|now|so)\s+)*(?:(?:try|do|search|look|check)\s+(?:it\s+|that\s+|this\s+)?"
    r"(?:again|once\s+more|properly|better|harder)|redo|retry)\b",
    re.I,
)
# "I meant the one about …", "there was something about …": the message says what was meant.
_MEANT = re.compile(
    r"\b(?:i\s+(?:meant|mean|was\s+(?:asking|talking)|am\s+(?:asking|talking)|asked)\b|there\s+(?:was|were|is)\s+(?:something|some|a|an|one)\b|"
    r"the\s+(?:one|ones|story|article|news|thing)\s+(?:about|on|with|where|that)\b|talking\s+about|looking\s+for|"
    r"(?:something|anything)\s+about|i\s+(?:want|wanted|need|needed|ask)\b)\b",
    re.I,
)
_POINTER = (
    r"(?:that|this|those|these|it|they|them|"
    r"(?:your|the|that|this)\s+(?:answer|reply|response|summary|list|results?|info|information|news|stories))"
)
_POINTS_BACK = re.compile(rf"\b{_POINTER}\b", re.I)
_POINTS_AT_START = re.compile(rf"^\W*(?:i\s+think\s+)?(?:you|u|ur|your|thats|its|{_POINTER})\b", re.I)
_YES_NO_ASKED = re.compile(r"\?\s*$")
# Words of a reaction that say how the answer failed or only fill the sentence; they are
# never what the user is after.
_ASIDE = frozenset(
    """no nope nah wrong incorrect false true right correct accurate old outdated stale news answer reply response
    something anything thing things stuff one ones mean meant talking said asked want wanted looking like also
    actually instead rather sad sadly lol haha ugh damn please pls again redo retry properly better try use web
    internet google search look check today recent new current latest really quite made make making invented fake
    real helpful useful relevant all still just only even about there was were been more last year years month
    months week weeks day days ago date""".split()
)


@dataclass(frozen=True)
class Reaction:
    kind: str  # REJECT or REFINE
    request: str  # the user's request that is done again
    added: str  # what this message adds to it ("" for a plain rejection)
    fresher: bool  # the complaint is about the answer's age: newer sources are wanted
    wants_web: bool  # the message says to use the web
    score: float
    asks: str = ""  # the question or order the message carries itself: that is answered, nothing is done again

    def note(self) -> str:
        """For the model: what the user is after, in one paragraph."""
        request = self.request.strip()[:400]
        if self.asks:  # the new message is the instruction; a wrong reading of it then costs nothing
            return (f"The user's new message may be correcting your last answer (they had asked: \"{request}\"). "
                    "Answer the new message as it is written. Do not repeat or defend the last answer.")
        how = "out of date" if self.fresher else "wrong or not what they asked for"
        lines = [f"The user says the last answer was {how}. Do not repeat or defend it, and do not apologise at length.",
                 f"Their request stands: \"{request}\"."]
        if self.added:
            lines.append(f"What they are after: {self.added}.")
        lines.append("Answer that request again, correctly this time.")
        return " ".join(lines)


_REWORK = re.compile(
    r"\b(?:summari[sz]e|summary|shorten|shorter|longer|translate|rewrite|rephrase|reword|simplify|explain|expand|"
    r"elaborate|list|bullet|combine|merge|format|turn|convert|make|put|sort|rank|compare)\b", re.I)


def reworks(message: str) -> bool:
    """The message asks to reshape what was just given ("summarize them as one paragraph"):
    it is answered from the conversation, not from a new search."""
    text = (message or "").strip()
    return len(text.split()) <= 14 and bool(_REWORK.search(text)) and bool(_POINTS_BACK.search(text))


_SIGNAL_WORDS = ("wrong", "again", "search", "incorrect", "outdated", "google", "false", "wanted", "meant", "asked")
_VOWELS = str.maketrans("", "", "aeiou")


def _untypo(text: str) -> str:
    """The message with slips in the words that carry a complaint put right: letters swapped
    ("agian", "wrnog", "serach") or vowels left out ("wrng", "srch"). Other words are left alone."""
    def fix(found: re.Match[str]) -> str:
        word = found.group(0).lower()
        if word in _SIGNAL_WORDS or len(word) < 4:
            return found.group(0)
        for right in _SIGNAL_WORDS:
            swapped = (len(word) == len(right) and word[0] == right[0] and sorted(word) == sorted(right)) or (
                len(word) > len(right) and re.sub(r"(.)\1+", r"\1", word) == re.sub(r"(.)\1+", r"\1", right))
            dropped = len(word) < len(right) and word[0] == right[0] and word.translate(_VOWELS) == right.translate(_VOWELS) \
                and len(word.translate(_VOWELS)) >= 3 and all(ch in right for ch in word)
            if swapped or dropped:
                return right
        return found.group(0)
    return re.sub(r"[A-Za-z]{4,}", fix, text)


def _feedback(turn_reading: Any) -> float:
    reading = getattr(turn_reading, "reading", None)
    scores = getattr(reading, "scores", None) or {}
    return float((scores.get("intent") or {}).get("feedback", 0.0))


def _own_subject(message: str) -> list[str]:
    """The words a message brings itself, once what complains and what points back is removed."""
    return [word for word in terms.content_words(_POINTS_BACK.sub(" ", message)) if word not in _ASIDE]


def score(message: str, turn_reading: Any = None, asked_yes_no: bool = False) -> float:
    """How sure it is that the message turns the last answer down (see SURE)."""
    text = _untypo((message or "").replace("\u2019", "'").strip())
    words = text.split()
    if not words or len(words) > MAX_WORDS or plan.is_small_talk(text):
        return 0.0
    wrong, stale = bool(_WRONG.search(text)), bool(_STALE.search(text))
    meant_first = bool(_MEANT_LEAD.match(text))
    lead = bool(_NO_LEAD.search(text)) or meant_first
    again = bool(_AGAIN.search(text))
    use_web = bool(_USE_WEB.search(text)) or plan.told_to_search(text)
    do_over = bool(_DO_OVER.search(text)) or bool(_SEARCH_IT.search(text))
    meant = bool(_MEANT.search(text))
    if asked_yes_no and lead and not (wrong or stale or meant_first or meant):
        return 0.0  # "no" to a question Kiwibro asked is an answer, not a complaint
    own = _own_subject(text)
    bare = not own  # nothing but the complaint: "not helpful", "search again", "this is outdated"
    total = _feedback(turn_reading)
    total += 0.5 * wrong + 0.4 * stale + 0.6 * lead + 0.3 * meant_first + 0.3 * again + 0.3 * use_web + 0.5 * do_over
    if (lead or wrong or stale) and meant:
        total += 0.3
    if bare and (again or use_web):
        total += 0.5
    elif bare and (wrong or stale):
        total += 0.3
    aimed = bool(_POINTS_AT_START.match(text))  # "these results are …", "that is …", "u made …"
    if (wrong or stale) and aimed:
        total += 0.4  # the answer is what the sentence is about
    if stale and use_web:
        total += 0.2  # "out of date, google it"
    if not (wrong or stale or lead or again or use_web):
        total -= 0.3  # the helper alone is not enough: "thanks, that helps" is feedback too
    if text.endswith("?") or plan._QUESTION.match(text):
        if not (lead or wrong):
            return 0.0  # a question asks; "how old is he?" is no complaint
        total -= 0.7 if own else 0.3  # "what is wrong with my code?" asks; it does not complain
    elif not (lead or aimed or do_over or (wrong and meant)) and len(own) >= 3:
        total -= 0.3  # a sentence about something of its own: "the test gives the wrong output"
    return total


def _is_reaction(message: dict[str, Any]) -> bool:
    meta = message.get("meta")
    if isinstance(meta, dict) and meta.get("reaction"):
        return True
    return score(str(message.get("content") or "")) >= SURE


def previous_request(earlier: list[dict[str, Any]]) -> tuple[int, str] | None:
    """(index, text) of the request a reaction is about: the latest user message that was
    answered and was not itself a reaction. ``earlier`` holds the messages before the
    reaction, oldest first."""
    answered = False
    for index in range(len(earlier) - 1, -1, -1):
        message = earlier[index]
        role, content = message.get("role"), str(message.get("content") or "").strip()
        if role == "assistant" and content:
            answered = True
        elif role == "user" and content and answered and not _is_reaction(message):
            return index, content
    return None


def read(message: str, turn_reading: Any, earlier: list[dict[str, Any]]) -> Reaction | None:
    """The reaction a message is, or None for a message that stands on its own."""
    found = previous_request(earlier)
    if found is None:
        return None
    _index, request = found
    last_reply = next((str(m.get("content") or "") for m in reversed(earlier)
                       if m.get("role") == "assistant" and str(m.get("content") or "").strip()), "")
    total = score(message, turn_reading, asked_yes_no=bool(_YES_NO_ASKED.search(last_reply.strip())))
    if total < SURE:
        return None
    text = _untypo(message.replace("’", "'"))
    asked = set(terms.content_words(request))
    added = [word for word in dict.fromkeys(_own_subject(text)) if word not in asked]
    kind = REFINE if len(added) >= ADDED_MIN else REJECT
    told = bool(_USE_WEB.search(text)) or plan.told_to_search(text)
    return Reaction(kind, request, " ".join(added) if kind == REFINE else "",
                    bool(_STALE.search(text)) and _about_the_answer(text), told, round(total, 2),
                    own_request(text) if kind == REFINE and not told else "")


_LEAD_OFF = re.compile(r"^\W*(?:still\s+)?(?:no+|nope|nah|wrong|incorrect|false)\b[\s,.!;:-]*", re.I)
_MEANT_OFF = re.compile(
    r"^(?:(?:but|well|actually|sorry)\W+)*i\s+(?:meant|mean|asked|was\s+asking|wanted\s+to\s+(?:know|ask))\b[\s,:]*", re.I)
_CLAUSES = re.compile(r"(?<=[,.;!?])\s+|\s+(?:and|but|so)\s+(?=(?:what|how|why|when|where|who|which|is|are|can|could|does|"
                      r"do|did|should|will|would)\b)", re.I)
_ASKS_ITSELF = re.compile(
    r"^(?:what|whats|how|why|when|where|who|which|is|are|was|were|can|could|does|do|did|should|will|would|tell|give|"
    r"show|find|write|make|list|explain|suggest|recommend|help|any)\b", re.I)


def own_request(message: str) -> str:
    """The question or order a message carries besides turning the answer down: "nope, mine is
    washed already, what then", "no, I meant what is it like in winter?". "" for a message that
    only complains or only says what was meant ("no, the one in Paris", "wrong, search again")."""
    rest = _MEANT_OFF.sub("", _LEAD_OFF.sub("", (message or "").strip()))
    clauses = [re.sub(r"^(?:and|but|so|also|then|btw|by\s+the\s+way)\b[\s,]*", "", clause.strip(), flags=re.I)
               for clause in _CLAUSES.split(rest)]
    clauses = [clause for clause in clauses if clause]
    for at, clause in enumerate(clauses):
        if clause.endswith("?") or _ASKS_ITSELF.match(clause):
            asked = " ".join(clauses[at:])
            return asked if _own_subject(asked) or at else ""  # "tell me the real one" asks for the same again
    return ""


def _about_the_answer(text: str) -> bool:
    """A word about age is said of the answer ("these are old", "outdated"), not of something of
    the user's own ("mine is old and washed", "I started two months ago")."""
    return not _own_subject(text) or bool(_POINTS_AT_START.match(_LEAD_OFF.sub("", text)))


def redo_query(reaction: Reaction) -> str:
    """The search that does the request again: its own subject, plus what the reaction adds."""
    query, _borrowed = plan.search_request(reaction.request)
    query = query or plan.query_of(reaction.request)
    if reaction.added:
        query = f"{query} {reaction.added}"
    return " ".join(query.split())[:QUERY_CHARS]
