"""Steering: notice when a reply goes wrong and say how to put it right.

Kiwi Sense (kiwibro/sense) reads the reply; this module checks what it reads against the
facts of the turn and writes the correction. It never asks the language model anything,
so it works the same for every model.

A sentence is only a problem when the facts say so:

    "I am ChatGPT"                  always (Kiwibro is not another assistant)
    "I am <the user's name>"        always (the model took the user's identity)
    "I can't browse the web"        when a web tool is available this turn
    "I've saved the file"           when no tool has run this turn
    "Let me search for that."       when the reply ends there without a tool call
    "The user is asking about …"    when it opens the reply (thinking out loud)
    "How can I help you today?"     when the user asked a real question
    the previous question, again    when the new question is about something else

Each verdict carries a short note for the model. The agent retries once with that note;
if the retry fails the same way, `repair` removes the offending sentences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from . import autofacts, compute
from . import sense as sense_module

OTHER_IDENTITY = "other_identity"
ADOPTED_USER = "adopted_user"
DENIAL = "denial"
UNBACKED_CLAIM = "unbacked_claim"
UNKEPT_PROMISE = "unkept_promise"
META_LEAK = "meta_leak"
OFF_TOPIC = "off_topic"
WRONG_RESULT = "wrong_result"
LOOP = "loop"

# What each kind is called where a person reads it (the inference panel, How it works).
LABELS = {
    OTHER_IDENTITY: "spoke as another assistant",
    ADOPTED_USER: "spoke as you",
    DENIAL: "said it could not use a tool it has",
    UNBACKED_CLAIM: "claimed an action that did not happen",
    UNKEPT_PROMISE: "announced a step without taking it",
    META_LEAK: "talked about its instructions",
    OFF_TOPIC: "answered something else",
    WRONG_RESULT: "gave a wrong number",
    LOOP: "went in circles",
}

EARLY_CHARS = 500  # only the opening of a reply is checked while it streams
RETRY_TEMPERATURE = 0.3  # a corrected attempt samples closer to the likeliest words
TRAIT_DAYS = 30  # a model's past corrections count for this long
LOOP_HABIT = 2  # collapses into repetition before anti-repeat sampling becomes that model's default
MAX_SENTENCE_WORDS = 60
# Off-topic checks (cosine of meaning vectors unless noted). Measured on hand-written pairs:
# empty offers of help score 0.61-0.85 against the empty direction, real answers at most 0.48
# unless the question itself was small talk (then they sit close to the question instead).
EMPTY_MIN = 0.6
EMPTY_QUESTION_MAX = 0.25
EMPTY_MAX_WORDS = 40
# A reply that answers an earlier question again scored 0.51-0.87 against that question and at
# least 0.47 lower against the new one; real answers to the new one scored higher against it.
STALE_EARLIER_MIN = 0.5
STALE_MARGIN = 0.4

_ENGLISH = frozenset(
    "i i'm i've i'll the a an is am are was to my me you your can can't cannot have has as by let "
    "will not do don't it this that of for and".split()
)
_OTHER_NAMES = (
    "chatgpt", "gpt-4", "gpt-3", "gpt4", "openai", "claude", "anthropic", "gemini", "bard", "google",
    "deepmind", "qwen", "tongyi", "alibaba", "llama", "meta ai", "mistral", "gemma", "phi", "microsoft",
    "copilot", "bing", "deepseek", "grok", "xai", "alexa", "siri", "granite", "ibm", "smollm", "ernie", "baidu",
)
_GREETING = r"(?:(?:hi|hello|hey|greetings|sure|okay|ok|well|yes)\b[\s,!.]*)*"
_SELF_INTRO = r"(?:i\s+am|i['’]m|my\s+name\s+is|this\s+is|you\s+can\s+call\s+me|call\s+me)\s+"

# (capability, words that name it in a denial, tools that provide it)
_CAPABILITIES: tuple[tuple[str, re.Pattern[str], tuple[str, ...]], ...] = (
    ("web", re.compile(
        r"brows|internet|online|\bweb|real[- ]time|current (?:info|event|data|price|news)|latest|live (?:data|info|score)|"
        r"\bnews\b|weather|search|\burls?\b|\blinks?\b|up[- ]to[- ]date|recent events|stock price", re.IGNORECASE),
     ("web_search", "multi_search", "web_research", "fetch_page", "browse_page")),
    ("files", re.compile(r"\bfiles?\b|folder|document|\bdisk\b|director|file system|attachment", re.IGNORECASE),
     ("read_file", "read_files", "list_files", "write_file", "study_file")),
    ("run", re.compile(r"\brun\b|execut|\bcode\b|commands?\b|programs?\b|terminal|script|install", re.IGNORECASE),
     ("run_command", "run_python", "multi_command", "run_javascript")),
    ("see", re.compile(r"image|picture|photo|screen", re.IGNORECASE), ("describe_image", "screenshot")),
)
_MEMORY_WORDS = re.compile(
    r"remember|memory|previous (?:conversation|chat)|past (?:conversation|chat)|earlier (?:conversation|chat)|"
    r"recall|between (?:chats|sessions|conversations)", re.IGNORECASE)
_CLOCK_WORDS = re.compile(r"\b(?:current |today's )?(?:time|date)\b|what day|\btoday\b", re.IGNORECASE)
# A claimed action counts only when it names something outside the chat.
_OUTSIDE_WORLD = re.compile(
    r"\bfiles?\b|folder|\bdisk\b|desktop|workspace|director|command|script|package|\bweb\b|online|website|"
    r"\bpage\b|e-?mail|reminder|memory|\bnotes?\b|clipboard|screenshot|server|\bchanges\b|patch|\w+\.[a-z]{1,5}\b",
    re.IGNORECASE)
_MEMORY_CLAIM = re.compile(r"memory|remember|noted|\bnotes?\b|stored your", re.IGNORECASE)
# "I've saved the note in your notes": a claim in so many words, whatever the probe makes of it.
_DID_IT = re.compile(
    r"\bI(?:['’]ve| have)?\s+(?:just\s+|now\s+)?(?:saved|added|stored|created|written|recorded|jotted|updated|"
    r"deleted|sent)\b[^.!?]{0,60}\b(?:notes?|files?|reminders?)\b",
    re.IGNORECASE,
)
_WAS_DONE = re.compile(
    r"\b(?:notes?|files?|reminders?)\b[^.!?]{0,30}\b(?:has been|have been|was|were|is now)\s+(?:successfully\s+)?"
    r"(?:saved|added|stored|created|recorded|updated|deleted|sent)\b",
    re.IGNORECASE,
)
# "I don't have a memory", "I can't remember earlier chats": the ability itself denied, in so many
# words. Not "I don't have your birthday yet", which may be true.
_NO_MEMORY = re.compile(
    r"\bI\s+(?:do\s+not|don['’]t)\s+have\s+(?:a\s+|any\s+|persistent\s+|long[- ]term\s+)*(?:memory|memories)\b|"
    r"\bI\s+(?:cannot|can['’]t|can\s+not|am\s+unable\s+to|am\s+not\s+able\s+to)\s+(?:remember|recall|retain)\b|"
    r"\bI\s+have\s+no\s+(?:memory|memories|recollection)\b",
    re.IGNORECASE,
)
_BEFORE_NOW = re.compile(r"\b(?:already|earlier|previously|last time)\b", re.IGNORECASE)
# A list under a line such as "What I can't do:" denies an ability in every item, without
# repeating the "I can't". Each item is read with those words put back.
_CANNOT_LEAD = re.compile(r"\b(?:can(?:['’]|no)t|can not|unable to|not able to|limitations?)\b", re.IGNORECASE)
_LIST_ITEM = re.compile(r"^\s{0,8}(?:[-*+•–]|\d+[.)])\s+(?=\S)")
_OWN_SUBJECT = re.compile(r"\W*(?:i|we|you|kiwibro)\b", re.IGNORECASE)
_EMPTY_ITEM = re.compile(r"(?m)^[ \t]*(?:[-*+•–]|\d+[.)])[ \t]*(?:\n|\Z)")
_CANNOT_FRAME = "I can't "
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class Facts:
    """What is true in this turn; the reply is judged against it."""

    question: str = ""
    user_names: tuple[str, ...] = ()
    tools: frozenset[str] = frozenset()  # tools the model may call now (empty: none)
    tools_ran: int = 0  # tool calls made so far in this turn
    memory_on: bool = True
    memory_saved: bool = False  # Kiwibro itself saved a fact from this message
    recent: tuple[str, ...] = ()  # earlier messages the reply may be continuing
    asked_before: tuple[str, ...] = ()  # what the user asked earlier in this chat (oldest first)
    small_talk: bool = False  # the message is a greeting or thanks
    model: str = ""
    note: str = ""  # Kiwibro's context note for this turn (the model must never read it out)
    learned: tuple[str, ...] = ()  # facts about the user that Kiwibro saved from this message
    told_only: bool = False  # the message states those facts and asks nothing
    worked: tuple[tuple[str, str], ...] = ()  # sums the message asks for, with their exact results


@dataclass(frozen=True)
class Verdict:
    kind: str
    note: str  # what the model is told before it tries again
    sentence: str = ""
    score: float = 1.0
    rewrite: str = ""  # the sentence put right by code; when set, no new attempt is needed


class Redirect(Exception):
    """Raised inside the stream to stop a reply whose opening already failed a check."""

    def __init__(self, verdict: Verdict) -> None:
        super().__init__(verdict.kind)
        self.verdict = verdict


def _reader() -> Any:
    return sense_module.shared()


def _threshold(reader: Any) -> float:
    return float((reader.meta.get("probe") or {}).get("threshold", 0.8))


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _looks_english(sentence: str) -> bool:
    """The probe was trained on English; other languages are left to the other checks."""
    return any(word in _ENGLISH for word in re.findall(r"[a-z']+", sentence.lower()))


def _probe(reader: Any, sentence: str) -> tuple[str, float]:
    if not _looks_english(sentence):
        return "fine", 0.0
    words = len(_words(sentence))
    if words < 2 or words > MAX_SENTENCE_WORDS:
        return "fine", 0.0
    scores = reader.probe(sentence)
    if not scores:
        return "fine", 0.0
    label = max(scores, key=scores.get)
    return label, scores[label]


def _cannot_lead(line: str) -> bool:
    """Whether a line introduces a list of things the assistant says it cannot do."""
    text = line.strip().strip("#*_ ").strip()
    return bool(text) and len(_words(text)) <= 12 and bool(_CANNOT_LEAD.search(text)) and (
        text.endswith(":") or line.lstrip().startswith(("#", "**")))


def _sentences(reply: str, limit: int = 12) -> list[tuple[str, bool]]:
    """The reply as (sentence, listed) pairs. ``listed`` marks an item of a "what I cannot
    do" list, written out as the sentence it stands for."""
    out: list[tuple[str, bool]] = []
    block: list[str] = []
    framed = False

    def flush() -> None:
        if block:
            out.extend((sentence, False) for sentence in sense_module.sentences("\n".join(block), limit))
            block.clear()

    for line in sense_module._FENCE_RE.sub(" ", reply or "").split("\n"):
        item = _LIST_ITEM.match(line)
        if item and framed and not _OWN_SUBJECT.match(line[item.end():]):
            flush()
            out.extend((sentence, True) for sentence in sense_module.sentences(_CANNOT_FRAME + line[item.end():], 1))
            continue
        if line.strip() and not item:
            framed = _cannot_lead(line)
        block.append(line)
    flush()
    return out[:limit]


def _names_in(text: str) -> set[str]:
    lowered = text.lower()
    return {name for name in _OTHER_NAMES if re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", lowered)}


def _adopted_user(sentence: str, facts: Facts) -> str:
    """The user's name when the sentence introduces the speaker by it."""
    for name in facts.user_names:
        first = name.split()[0] if name.split() else ""
        if len(first) < 2 or first.lower() == "kiwibro":
            continue
        if re.match(rf"^\W*{_GREETING}{_SELF_INTRO}{re.escape(first)}\b", sentence, re.IGNORECASE):
            return first
    return ""


_USER_SPEAKS = frozenset({"i", "i'm", "im", "i've", "my"})
PARROT_WORDS = 4  # this many of the user's own words in a row, starting at "I" or "my"
_QUOTING = re.compile(r"[\"“”«»]|:\s*(?:i|my)\b", re.IGNORECASE)  # the user's words, given as theirs
# Requests where the reply is rightly written in the user's voice (a bio, a translation, a fix).
_IN_USER_VOICE = re.compile(
    r"\b(?:write|draft|compose|translate|rewrite|rephrase|reword|correct|fix|proofread|repeat|say|email|"
    r"message|letter|bio|cv|resume|post|tweet|caption|speech|introduce|introduction)\b",
    re.IGNORECASE,
)


def _plain(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower().replace("\u2019", "'")).split()


# "What is my dog's name?" answered "My dog's name is Biscuit": things an assistant has none of.
# Thoughts are left out ("my understanding is…" may well be Kiwibro's own).
_OWN_THOUGHTS = frozenset(
    {
        "understanding", "opinion", "view", "guess", "take", "answer", "suggestion", "advice", "recommendation",
        "plan", "goal", "apologies", "apology", "point", "thought", "thoughts", "pleasure", "bad", "mistake",
        "aim", "job", "role", "purpose", "task", "best", "knowledge", "memory", "response", "reply",
    }
)
_ASKS_SELF = re.compile(r"\b(?:do|did|am|have|should|can|could|will|would)\s+i\s+(?P<verb>[a-z]+)", re.IGNORECASE)
_SELF_VERBS = frozenset(
    {"live", "work", "like", "love", "hate", "prefer", "use", "speak", "own", "drive", "study", "teach", "play",
     "enjoy", "eat", "drink"}
)
_IS_QUESTION = re.compile(r"\?\s*$|^\W*(?:what|where|who|when|which|how|do|did|am|is|are|was|have|should|can)\b",
                          re.IGNORECASE)


def _spoke_as_user(sentence: str, facts: Facts) -> bool:
    """The reply says, as its own, something that is the user's: what they just told Kiwibro about
    themselves ("…, I'm allergic to peanuts"), or the thing they asked about ("My dog's name is Biscuit")."""
    if _QUOTING.search(sentence) or _IN_USER_VOICE.search(facts.question):
        return False
    asked, words = _plain(facts.question), _plain(sentence)
    said = " " + " ".join(asked) + " "
    if facts.learned and any(
        word in _USER_SPEAKS and len(words) - start >= PARROT_WORDS
        and " " + " ".join(words[start:start + PARROT_WORDS]) + " " in said
        for start, word in enumerate(words)
    ):
        return True
    if len(words) < 3 or not _IS_QUESTION.search(facts.question.strip()):
        return False
    if words[0] == "my":
        return words[1].split("'")[0] not in _OWN_THOUGHTS and f" my {words[1]} " in said
    if words[0] == "i":
        return words[1] in _SELF_VERBS and any(m.group("verb").lower() == words[1] for m in _ASKS_SELF.finditer(facts.question))
    return False


_TO_YOU = (
    (r"\bI\s+am\b", "you are"), (r"\bI'm\b", "you're"), (r"\bI\s+was\b", "you were"), (r"\bI've\b", "you've"),
    (r"\bI'd\b", "you'd"), (r"\bI'll\b", "you'll"), (r"\bmyself\b", "yourself"), (r"\bmine\b", "yours"),
    (r"\bmy\b", "your"), (r"\bme\b", "you"), (r"\bI\b", "you"),
)


def _to_you(sentence: str) -> str:
    """The sentence said to the user instead of as the user: "My dog…" becomes "Your dog…"."""
    text = sentence.replace("\u2019", "'")
    for pattern, word in _TO_YOU:
        text = re.sub(
            pattern,
            lambda m, word=word: word.capitalize() if not text[: m.start()].strip(" \t\n*_>-#") else word,
            text, flags=re.IGNORECASE,
        )
    return text


def _identity(reader: Any, sentence: str, facts: Facts) -> Verdict | None:
    name = _adopted_user(sentence, facts)
    if name:
        return Verdict(
            ADOPTED_USER,
            f"You are Kiwibro, the assistant. {name} is the user you are talking to, not you. "
            f"Write your reply again as Kiwibro, speaking to {name}.",
            sentence,
        )
    if _spoke_as_user(sentence, facts):
        return Verdict(
            ADOPTED_USER,
            "That is about the user, not about you. You are Kiwibro: say it to the user with “you” and “your”.",
            sentence, rewrite=_to_you(sentence),
        )
    label, score = _probe(reader, sentence)
    if label != OTHER_IDENTITY or score < _threshold(reader) or "kiwibro" in sentence.lower():
        return None
    if _names_in(sentence) & _names_in(facts.question):  # the user brought that assistant up
        return None
    return Verdict(
        OTHER_IDENTITY,
        "You are Kiwibro, a local AI agent on the user's computer. The model underneath is only your "
        "engine: never introduce yourself as it or as its maker. Write your reply again as Kiwibro.",
        sentence, score,
    )


_HAS_MEMORY = (
    "That is not true here: you have a memory. What you know about the user is in your context "
    "note, and you can save new facts. Answer again without saying you cannot remember."
)


def _denial(reader: Any, sentence: str, facts: Facts, listed: bool = False) -> Verdict | None:
    label, score = _probe(reader, sentence)
    if label != DENIAL or score < _threshold(reader):
        if not (facts.memory_on and _NO_MEMORY.search(sentence)):
            return None
        return Verdict(DENIAL, _HAS_MEMORY, sentence, 1.0)
    if facts.tools and not facts.tools_ran:
        for _capability, pattern, tool_names in _CAPABILITIES:
            usable = [tool for tool in tool_names if tool in facts.tools]
            if usable and pattern.search(sentence):
                todo = ("Write your reply again and leave it out of the things you cannot do." if listed
                        else "Call the tool now instead of saying you cannot.")
                return Verdict(
                    DENIAL, f"That is not true here: you can do it with your `{usable[0]}` tool. {todo}",
                    sentence, score,
                )
    if facts.memory_on and _MEMORY_WORDS.search(sentence):
        return Verdict(DENIAL, _HAS_MEMORY, sentence, score)
    if _CLOCK_WORDS.search(sentence) and not _MEMORY_WORDS.search(sentence):
        return Verdict(
            DENIAL,
            "That is not true here: the current date and time are in your context note. Answer with them.",
            sentence, score,
        )
    return None


ECHO_MIN_CHARS = 25  # a shorter line can match the note by chance
_LINE_MARKS = re.compile(r"^[\s#>*+\-•–\d.)\[\]]+|[\s*_\]]+$")
_NOTE_HEADINGS = ("guidance for this turn", "kiwibro context for this turn")


def _bare(line: str) -> str:
    return _LINE_MARKS.sub("", line).lower()


def _echoed(reply: str, facts: Facts) -> list[str]:
    """Lines of the reply copied from Kiwibro's own context note."""
    if not facts.note:
        return []
    noted = {bare for bare in map(_bare, facts.note.splitlines()) if len(bare) >= ECHO_MIN_CHARS}
    echoed = []
    for line in reply.splitlines():
        bare = _bare(line)
        if bare and (bare in noted or bare.startswith(_NOTE_HEADINGS)):
            echoed.append(line)
    return echoed


def _echo(reply: str, facts: Facts) -> Verdict | None:
    lines = _echoed(reply, facts)
    if not lines:
        return None
    return Verdict(
        META_LEAK,
        "You copied Kiwibro's note for this turn into your reply. The user must not see it. "
        "Follow it silently: answer the user's message itself.",
        lines[0].strip(),
    )


# A reply that opens by describing the request ("The user wants to know…") is thinking out loud.
_NARRATES = re.compile(
    r"^\W*(?:(?:okay|ok|alright|so|hmm|well)\b[,.\s]*)?the user(?:'s|’s)?\s+(?:wants|asks|asked|is asking|has asked|"
    r"would like|needs|said|says|is requesting|requests|wrote|message|question|request)\b", re.IGNORECASE)
_ABOUT_USERS = re.compile(r"\busers?\b", re.IGNORECASE)


def _meta(reader: Any, sentence: str, facts: Facts | None = None) -> Verdict | None:
    label, score = _probe(reader, sentence)
    narrates = _NARRATES.match(sentence) is not None and not (facts and _ABOUT_USERS.search(facts.question))
    if not narrates and (label != META_LEAK or score < _threshold(reader)):
        return None
    return Verdict(
        META_LEAK,
        "Speak to the user directly. Do not describe their request, your instructions or your context "
        "note: just give the answer.",
        sentence, score,
    )


def early(text: str, facts: Facts, checked: int = 0) -> tuple[Verdict | None, int]:
    """Check the sentences of a streaming reply that are complete and not yet checked.

    Returns (verdict, how many sentences are now checked). Only problems that later text
    cannot undo are raised here, so a bad reply is cut off after its first sentence."""
    reader = _reader()
    if reader is None or len(text) > EARLY_CHARS:
        return None, checked
    echo = _echo(text.rsplit("\n", 1)[0], facts) if "\n" in text else None  # whole lines only
    if echo is not None:
        return echo, checked
    complete = sense_module.sentences(text)
    if complete and text.rstrip(" ")[-1:] not in ".!?…\n":
        complete = complete[:-1]  # the last one is still being written
    for index in range(checked, len(complete)):
        sentence = complete[index]
        verdict = _identity(reader, sentence, facts)
        if verdict is not None and verdict.rewrite:
            verdict = None  # code puts it right once the reply is whole: no reason to stop the stream
        verdict = verdict or _denial(reader, sentence, facts)
        if verdict is None and index < 2:
            verdict = _meta(reader, sentence, facts)
        if verdict is not None:
            return verdict, len(complete)
    return None, len(complete)


# Replies that offer help instead of giving it. Their average meaning is the "empty reply"
# direction; a reply close to it and far from the question said nothing.
_EMPTY_REPLIES = (
    "Hello! How can I help you today?",
    "Sure, I'd be happy to help you with that. Please let me know what you need.",
    "I'm here to assist you with any questions you may have.",
    "Of course! Just let me know how I can assist you.",
    "Thank you for your message. Feel free to ask me anything.",
    "Is there something I can help you with?",
    "What would you like to talk about today?",
    "Please provide more details so that I can help you better.",
)
_REFERS_BACK = re.compile(
    r"\b(?:that|this|it|its|again|more|same|above|previous|earlier|those|these|them|one|another|else|"
    r"instead|also|too|other|rest|continue|keep going|go on)\b", re.IGNORECASE)
_empty_direction: dict[int, list[float]] = {}


def _emptiness(reader: Any, reply: str) -> float:
    """How close a reply is to an empty offer of help (cosine, -1..1)."""
    direction = _empty_direction.get(id(reader))
    if direction is None:
        vectors = [vector for vector in (reader.embed(text) for text in _EMPTY_REPLIES) if vector]
        if not vectors:
            return 0.0
        mean = [sum(column) / len(vectors) for column in zip(*vectors)]
        norm = sum(value * value for value in mean) ** 0.5 or 1.0
        direction = _empty_direction[id(reader)] = [value / norm for value in mean]
    vector = reader.embed(reply)
    return sum(a * b for a, b in zip(vector, direction)) if vector else 0.0


def _off_topic(reader: Any, reply: str, facts: Facts) -> Verdict | None:
    """A reply that does not answer: an empty offer of help, or the previous question answered again.

    Distance between question and reply alone cannot tell (a joke is far from "tell me a
    joke"), so each check needs a second, independent sign."""
    question = facts.question.strip()
    body = sense_module.prose(reply)
    if facts.tools_ran or facts.small_talk or len(_words(question)) < 4 or len(_words(body)) < 6:
        return None
    if not _looks_english(question) or not _looks_english(body):
        return None
    close = reader.similarity(question, body)
    if close is None:
        return None  # the embeddings cannot read one side: no verdict
    shown = question if len(question) <= 300 else question[:300].rsplit(" ", 1)[0] + "…"
    note = f"Your reply did not answer what was asked. The question is: “{shown}” Answer exactly that, briefly."
    if len(_words(body)) <= EMPTY_MAX_WORDS and close < EMPTY_QUESTION_MAX:
        emptiness = _emptiness(reader, body)
        if emptiness >= EMPTY_MIN:
            return Verdict(OFF_TOPIC, note, "", emptiness)
    if len(_words(question)) >= 5 and not _REFERS_BACK.search(question):
        earlier = max((reader.similarity(old, body) or 0.0 for old in facts.asked_before[-3:]), default=0.0)
        if earlier >= STALE_EARLIER_MIN and earlier - close >= STALE_MARGIN:
            return Verdict(OFF_TOPIC, note, "", earlier)
    return None


def review(reply: str, facts: Facts, *, final: bool = True) -> Verdict | None:
    """Judge a finished reply. ``final`` is False when the model also called tools."""
    reader = _reader()
    if reader is None or not (reply or "").strip():
        return None
    echo = _echo(reply, facts)
    if echo is not None:
        return echo
    read = _sentences(reply)
    for index, (sentence, listed) in enumerate(read):
        verdict = _identity(reader, sentence, facts) or _denial(reader, sentence, facts, listed)
        if verdict is None and index < 2:
            verdict = _meta(reader, sentence, facts)
        if verdict is not None:
            return verdict
    if not final:
        return None
    parts = [sentence for sentence, listed in read if not listed]
    threshold = _threshold(reader)
    for sentence in parts:
        label, score = _probe(reader, sentence)
        claimed = (label == "action_claim" and score >= threshold) or (
            (_DID_IT.search(sentence) or _WAS_DONE.search(sentence)) is not None and not _BEFORE_NOW.search(sentence)
        )
        if not claimed or facts.tools_ran or not _OUTSIDE_WORLD.search(sentence):
            continue
        if facts.memory_saved and _MEMORY_CLAIM.search(sentence):
            continue  # true: Kiwibro saved it in code before the reply
        if facts.tools:
            note = (f"You wrote “{sentence.strip()}” but no tool has run in this turn, so nothing happened. "
                    "Call the tool that does it now. If you cannot, say plainly that it is not done.")
        else:
            note = (f"You wrote “{sentence.strip()}” but you have no tools in this turn, so nothing happened. "
                    "Do not claim it: tell the user how they can do it themselves.")
        return Verdict(UNBACKED_CLAIM, note, sentence, score)
    if parts and facts.tools:
        label, score = _probe(reader, parts[-1])
        if label == "action_promise" and score >= threshold:
            return Verdict(
                UNKEPT_PROMISE,
                "You said what you would do next but did not do it. Do it now with a tool call, then answer.",
                parts[-1], score,
            )
    return _wrong_result(reply, facts) or _unheard(reply, facts) or _off_topic(reader, reply, facts)


def _exact(worked: tuple[tuple[str, str], ...]) -> str:
    return "\n".join(f"{asked} = **{result}**" for asked, result in worked)


def _wrong_result(reply: str, facts: Facts) -> Verdict | None:
    """The message asked for a sum Kiwibro worked out exactly, and the reply gives another number or none."""
    missing = tuple((asked, result) for asked, result in facts.worked if not compute.gives(reply, result))
    if not missing:
        return None
    exact = "; ".join(f"{asked} = {result}" for asked, result in missing)
    return Verdict(
        WRONG_RESULT,
        f"Kiwibro worked it out exactly: {exact}. Your reply does not give that. Answer again with exactly "
        "this result.",
    )


_FACT_FILLER = frozenset({"user", "user's", "named", "name", "the", "and", "are", "has", "have", "from", "with"})


def acknowledgement(learned: tuple[str, ...]) -> str:
    """What Kiwibro says when the user only told it things: what it will remember, in a sentence."""
    said = [autofacts.to_you(fact) for fact in learned[:4]]
    listed = said[0] if len(said) == 1 else ", ".join(said[:-1]) + " and " + said[-1]
    return f"Got it, I'll remember that {listed}."


def _unheard(reply: str, facts: Facts) -> Verdict | None:
    """The user only told Kiwibro things about themselves, and the reply takes up none of it."""
    if not (facts.told_only and facts.learned) or facts.tools_ran:
        return None
    told = {word for fact in facts.learned for word in _plain(fact) if len(word) >= 3} - _FACT_FILLER
    if told & set(_plain(reply)):
        return None
    return Verdict(
        OFF_TOPIC, "The user told you facts about themselves. Say in one sentence what you will remember.",
        "", rewrite=acknowledgement(facts.learned),
    )


def repair(reply: str, verdict: Verdict, facts: Facts) -> str:
    """The reply without the sentences that fail in the way ``verdict`` names ("" if none is left).

    Used when a second attempt fails the same way: what remains is true, if shorter."""
    if verdict.kind == OFF_TOPIC:
        return verdict.rewrite or reply
    if verdict.kind == WRONG_RESULT:
        return _exact(facts.worked)  # the model could not say it: the result itself is the reply
    reader = _reader()
    if reader is None or verdict.kind not in (OTHER_IDENTITY, ADOPTED_USER, META_LEAK, DENIAL):
        return reply
    kept = reply
    if verdict.kind == META_LEAK:
        echoed = set(_echoed(reply, facts))
        kept = "\n".join(line for line in reply.splitlines() if line not in echoed)
    for index, (sentence, listed) in enumerate(_sentences(kept, limit=40)):
        if verdict.kind in (OTHER_IDENTITY, ADOPTED_USER):
            found = _identity(reader, sentence, facts)
        elif verdict.kind == DENIAL:
            found = _denial(reader, sentence, facts, listed)
        else:
            found = _meta(reader, sentence, facts) if index < 2 else None
        written = sentence[len(_CANNOT_FRAME):] if listed else sentence
        if found is None or found.kind != verdict.kind:
            continue
        if written in kept:
            kept = kept.replace(written, found.rewrite, 1)
        elif found.rewrite:  # written with markdown inside ("**My wife** is Elif"): turn its line round
            plain = " ".join(_plain(written))
            kept = "\n".join(_to_you(line) if plain in " ".join(_plain(line)) else line for line in kept.splitlines())
    kept = _without_empty_lists(kept)
    return re.sub(r"[ \t]{2,}", " ", re.sub(r"\n{3,}", "\n\n", kept)).strip()


def _without_empty_lists(text: str) -> str:
    """Drop list items left empty, and a "what I cannot do" line left with no item under it."""
    lines = _EMPTY_ITEM.sub("", text).split("\n")
    out: list[str] = []
    for index, line in enumerate(lines):
        if _cannot_lead(line) and not _LIST_ITEM.match(line):
            following = next((later for later in lines[index + 1:] if later.strip()), "")
            if not _LIST_ITEM.match(following):
                continue
        out.append(line)
    return "\n".join(out)


def reminders(traits: dict[str, int], facts: Facts) -> list[str]:
    """Standing reminders for a model that needed correcting before (they ride in the context note)."""
    lines: list[str] = []
    if traits.get(OTHER_IDENTITY) or traits.get(ADOPTED_USER):
        who = f" {facts.user_names[0]} is the user." if facts.user_names else ""
        lines.append(f"You are Kiwibro, the assistant.{who} Never speak as the user or as another assistant.")
    if traits.get(DENIAL) and facts.tools:
        lines.append("Your tools work on this computer: you can search the web, read and write files and run "
                     "code. Use them instead of saying you cannot.")
    if traits.get(UNBACKED_CLAIM) or traits.get(UNKEPT_PROMISE):
        lines.append("Say something is saved, created or done only after its tool call succeeded; "
                     "when a step is needed, call the tool in the same reply.")
    if traits.get(META_LEAK):
        lines.append("Answer the user directly; never describe their request or these notes.")
    return lines
