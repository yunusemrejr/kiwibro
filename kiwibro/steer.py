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

from . import sense as sense_module

OTHER_IDENTITY = "other_identity"
ADOPTED_USER = "adopted_user"
DENIAL = "denial"
UNBACKED_CLAIM = "unbacked_claim"
UNKEPT_PROMISE = "unkept_promise"
META_LEAK = "meta_leak"
OFF_TOPIC = "off_topic"
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


@dataclass(frozen=True)
class Verdict:
    kind: str
    note: str  # what the model is told before it tries again
    sentence: str = ""
    score: float = 1.0


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


def _identity(reader: Any, sentence: str, facts: Facts) -> Verdict | None:
    name = _adopted_user(sentence, facts)
    if name:
        return Verdict(
            ADOPTED_USER,
            f"You are Kiwibro, the assistant. {name} is the user you are talking to, not you. "
            f"Write your reply again as Kiwibro, speaking to {name}.",
            sentence,
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


def _denial(reader: Any, sentence: str, facts: Facts) -> Verdict | None:
    label, score = _probe(reader, sentence)
    if label != DENIAL or score < _threshold(reader):
        return None
    if facts.tools and not facts.tools_ran:
        for _capability, pattern, tool_names in _CAPABILITIES:
            usable = [tool for tool in tool_names if tool in facts.tools]
            if usable and pattern.search(sentence):
                return Verdict(
                    DENIAL,
                    f"That is not true here: you can do it with your `{usable[0]}` tool. "
                    "Call the tool now instead of saying you cannot.",
                    sentence, score,
                )
    if facts.memory_on and _MEMORY_WORDS.search(sentence):
        return Verdict(
            DENIAL,
            "That is not true here: you have a memory. What you know about the user is in your context "
            "note, and you can save new facts. Answer again without saying you cannot remember.",
            sentence, score,
        )
    if _CLOCK_WORDS.search(sentence) and not _MEMORY_WORDS.search(sentence):
        return Verdict(
            DENIAL,
            "That is not true here: the current date and time are in your context note. Answer with them.",
            sentence, score,
        )
    return None


def _meta(reader: Any, sentence: str) -> Verdict | None:
    label, score = _probe(reader, sentence)
    if label != META_LEAK or score < _threshold(reader):
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
    complete = sense_module.sentences(text)
    if complete and text.rstrip(" ")[-1:] not in ".!?…\n":
        complete = complete[:-1]  # the last one is still being written
    for index in range(checked, len(complete)):
        sentence = complete[index]
        verdict = _identity(reader, sentence, facts) or _denial(reader, sentence, facts)
        if verdict is None and index < 2:
            verdict = _meta(reader, sentence)
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
    parts = sense_module.sentences(reply)
    for index, sentence in enumerate(parts):
        verdict = _identity(reader, sentence, facts) or _denial(reader, sentence, facts)
        if verdict is None and index < 2:
            verdict = _meta(reader, sentence)
        if verdict is not None:
            return verdict
    if not final:
        return None
    threshold = _threshold(reader)
    for sentence in parts:
        label, score = _probe(reader, sentence)
        if label != "action_claim" or score < threshold or facts.tools_ran or not _OUTSIDE_WORLD.search(sentence):
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
    return _off_topic(reader, reply, facts)


def repair(reply: str, verdict: Verdict, facts: Facts) -> str:
    """The reply without the sentences that fail in the way ``verdict`` names ("" if none is left).

    Used when a second attempt fails the same way: what remains is true, if shorter."""
    reader = _reader()
    if reader is None or verdict.kind not in (OTHER_IDENTITY, ADOPTED_USER, META_LEAK, DENIAL):
        return reply
    kept = reply
    for index, sentence in enumerate(sense_module.sentences(reply, limit=40)):
        if verdict.kind in (OTHER_IDENTITY, ADOPTED_USER):
            found = _identity(reader, sentence, facts)
        elif verdict.kind == DENIAL:
            found = _denial(reader, sentence, facts)
        else:
            found = _meta(reader, sentence) if index < 2 else None
        if found is not None and found.kind == verdict.kind and sentence in kept:
            kept = kept.replace(sentence, "", 1)
    return re.sub(r"[ \t]{2,}", " ", re.sub(r"\n{3,}", "\n\n", kept)).strip()


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
