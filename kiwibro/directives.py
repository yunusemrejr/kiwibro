"""Directive ledger: the user's standing instructions, kept in view every turn.

Small models drift in long sessions. After compaction, many tool rounds or a
change of topic (browser work → coding → a new request) they forget a
constraint given twenty turns ago ("never touch prod", "use TypeScript"), or
keep following an instruction the user has since overridden. The ledger fixes
that cheaply:

* It is **derived** from the conversation's user messages: no LLM calls and
  nothing new to store. Messages are never deleted, so it survives restarts by
  construction, and it is updated incrementally (only new messages are read).
* Requests ("build…", "can you…") and constraints ("always…", "never…",
  "only…", "make sure…") are kept verbatim. Explicit course changes
  ("actually…", "instead…", "forget that…", "not anymore") supersede the
  earlier directives they share words with, or the previous goal when they
  share none. Unrelated new requests just add on.
* :func:`render_note` returns a short block for the per-turn context note
  (after the KV-cached prefix), only when it adds something beyond the newest
  message. :func:`review_prompt` turns the same ledger into a one-time
  end-of-turn checklist so the model verifies its work before finishing.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol

MAX_DIRECTIVES = 40
MAX_CLAUSES_PER_MESSAGE = 8
NOTE_CHAR_BUDGET = 700
REVIEW_MIN_ACTIONS = 3  # write/exec calls in a turn before the end-of-turn self-check
TEXT_CHARS = 110
_CACHE_SIZE = 64

_TRIVIAL = re.compile(
    r"^(?:ok(?:ay)?|k|thanks?(?: you)?|thx|great|nice|cool|yes|yep|yeah|no|nope|sure|continue|go on|"
    r"go ahead|keep going|proceed|done|good|perfect|hi|hello|hey|right|fine|alright)[\s.!]*$",
    re.I,
)
# Discourse openers stripped before matching a request ("Actually, use…", "Forget that, write…").
_LEAD = re.compile(
    r"^(?:(?:actually|instead|no|nope|ok(?:ay)?|wait|hmm+|well|also|and|so|then|now|please|pls|next|but|"
    r"alright|right|forget (?:it|that)|scratch that|never ?mind|on second thought|change of plans?)"
    r"[,.!:;]?\s+)+",
    re.I,
)
REQUEST = re.compile(
    r"^(?:(?:please|pls|now|then|also|and|so|ok(?:ay)?|next)[,\s]+)*"
    r"(?:(?:can|could|would|will) you\s+|i (?:want|need|would like|'d like)(?: you)? to\s+|"
    r"let'?s\s+|help me(?: to)?\s+|try to\s+|go\s+|you should\s+)?"
    r"(build|create|write|make|fix|find|research|add|remove|delete|update|change|implement|deploy|tell|give|put|"
    r"analy[sz]e|summari[sz]e|explain|check|run|install|design|generate|compare|search|collect|"
    r"gather|show|list|convert|refactor|test|review|translate|draft|plan|set up|setup|configure|"
    r"investigate|look|open|read|edit|rename|move|copy|download|upload|organi[sz]e|clean|improve|"
    r"optimi[sz]e|debug|scrape|monitor|calculate|compute|prepare|send|publish|finish|redo|undo|"
    r"rewrite|polish|switch|use|focus|start|stop|describe|extract|merge|split|save|export|import|"
    r"document|benchmark|verify|validate|support|integrate|replace|restructure|port|migrate)\b",
    re.I,
)
_CONSTRAINT = re.compile(
    r"\b(always|never|don'?t|do not|must(?: not)?|should(?:n'?t| not)?|make sure|ensure|only|avoid|"
    r"without|no more than|at (?:least|most)|keep (?:it|the|them|everything)|stay|prefer|"
    r"under no circumstances|no (?:emojis?|comments|external|third[- ]party|network|internet))\b",
    re.I,
)
_OVERRIDE = re.compile(
    r"\b(instead|actually|rather than|scratch that|forget (?:it|that|about|what)|never ?mind|"
    r"change of plans?|on second thought|no longer|not anymore|any ?more|cancel|stop doing|"
    r"ignore (?:what|my|the previous|that)|switch to|don'?t do that|change it to|replace it with)\b",
    re.I,
)
# Requests that are standing decisions rather than tasks ("use TypeScript", "switch to SQLite").
_CHOICE = re.compile(
    r"^(?:use|switch to|stick (?:with|to)|go with|prefer|choose|pick|keep using|target|write (?:it|everything) in|"
    r"do (?:it|everything) in|answer in|reply in|respond in)\b",
    re.I,
)
_STOPWORDS = frozenset(
    """a an the and or but if then else so to of in on at by for with from into onto about as is are
    was were be been being it its this that these those there here i me my we our you your he she they
    them their please pls can could would will should shall may might must do does did done not no
    yes just also too very really now then again some any all each every more most less use using
    make made get got let lets want need like it's i'm don't do instead actually rather than anymore
    any more never always only keep sure ensure avoid without stop start switch change forget that
    what which who how when where why""".split()
)
_CODE_FENCE = re.compile(r"```.*?```", re.S)
_SPLIT = re.compile(r"(?:\n+|(?<=[.!?;])\s+)")


@dataclass
class Directive:
    id: int
    turn: int  # 1-based user turn it came from
    kind: str  # goal | request | constraint
    text: str
    words: frozenset[str]
    status: str = "active"  # active | superseded
    superseded_at: int | None = None  # turn that overrode it


@dataclass
class Ledger:
    turns: int = 0  # user turns seen
    last_seq: int = 0  # newest message seq processed
    directives: list[Directive] = field(default_factory=list)

    def active(self, kind: str | None = None) -> list[Directive]:
        return [d for d in self.directives if d.status == "active" and (kind is None or d.kind == kind)]

    def goal(self) -> Directive | None:
        goals = self.active("goal")
        return goals[-1] if goals else None


class _MessageSource(Protocol):
    def list_messages(self, conversation_id: str, after_seq: int = 0, limit: int | None = None) -> list[dict[str, Any]]: ...

    def last_message_seq(self, conversation_id: str) -> int: ...


def keywords(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z0-9][a-z0-9+#._-]{1,}", text.lower()) if w not in _STOPWORDS)


def _clauses(message: str) -> list[str]:
    """The instruction-bearing sentences of a user message (code and pastes dropped)."""
    text = _CODE_FENCE.sub(" ", message)
    clauses: list[str] = []
    for part in _SPLIT.split(text):
        part = part.strip(" -*•\t>")
        if 3 <= len(part) and not _TRIVIAL.match(part) and part[:400] not in clauses:
            clauses.append(part[:400])
    if len(text) > 1500:  # a pasted document: only the framing around it carries instructions
        clauses = clauses[:2] + clauses[2:][-2:]
    return clauses[:MAX_CLAUSES_PER_MESSAGE]


def _shorten(text: str, limit: int = TEXT_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def apply_message(ledger: Ledger, message: str) -> None:
    """Fold one user message into the ledger (pure; deterministic)."""
    ledger.turns += 1
    turn = ledger.turns
    for clause in _clauses(message):
        core = _LEAD.sub("", clause)
        is_request = bool(REQUEST.match(core)) or (clause.endswith("?") and REQUEST.search(core) is not None)
        is_constraint = bool(_CONSTRAINT.search(clause)) and not clause.endswith("?")
        if not (is_request or is_constraint):
            continue
        words = keywords(clause)
        if _OVERRIDE.search(clause):
            hit = [d for d in ledger.active() if d.turn < turn and d.words & words]
            if not hit and is_request:  # "forget that, do X": the previous goal/requests go
                hit = [d for d in ledger.active() if d.kind in ("goal", "request") and d.turn < turn]
            for directive in hit:
                directive.status, directive.superseded_at = "superseded", turn
        if (is_constraint and not is_request) or _CHOICE.match(core):
            kind = "constraint"
        elif ledger.goal() is None:
            kind = "goal"
        else:
            kind = "request"
        ledger.directives.append(Directive(len(ledger.directives) + 1, turn, kind, clause, words))
    if len(ledger.directives) > MAX_DIRECTIVES:  # drop the oldest superseded first, then oldest requests
        removable = [d for d in ledger.directives if d.status != "active"] + [
            d for d in ledger.directives if d.kind == "request" and d.status == "active"
        ]
        for directive in removable[: len(ledger.directives) - MAX_DIRECTIVES]:
            ledger.directives.remove(directive)


def build_ledger(messages: list[str]) -> Ledger:
    ledger = Ledger()
    for message in messages:
        apply_message(ledger, message)
    return ledger


# --------------------------------------------------------------------------
# Incremental cache over the database


_CACHE: OrderedDict[str, Ledger] = OrderedDict()
_LOCK = threading.Lock()


def ledger_for(db: _MessageSource, conversation_id: str) -> Ledger:
    """The conversation's ledger, reading only messages added since the last call."""
    with _LOCK:
        ledger = _CACHE.pop(conversation_id, None)
    last = db.last_message_seq(conversation_id)
    if ledger is None or last < ledger.last_seq:  # first use, or messages were removed
        ledger = Ledger()
    for message in db.list_messages(conversation_id, after_seq=ledger.last_seq):
        if message.get("role") == "user":
            apply_message(ledger, str(message.get("content") or ""))
        ledger.last_seq = max(ledger.last_seq, int(message.get("seq") or 0))
    with _LOCK:
        _CACHE[conversation_id] = ledger
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return ledger


def forget(conversation_id: str) -> None:
    with _LOCK:
        _CACHE.pop(conversation_id, None)


# --------------------------------------------------------------------------
# Rendering


def render(ledger: Ledger) -> str:
    """Compact note: goal, current request, standing constraints, recent course changes."""
    if ledger.turns < 2:
        return ""  # the only instructions are in the newest message itself
    newest = ledger.turns
    goal = ledger.goal()
    requests = ledger.active("request")
    now = requests[-1] if requests and (goal is None or requests[-1].turn > goal.turn) else None
    constraints = ledger.active("constraint")
    changed = [d for d in ledger.directives if d.status == "superseded"][-2:]
    older = [d for d in (goal, *constraints) if d is not None and d.turn < newest]
    if not older and not changed:
        return ""  # nothing beyond what the newest message already says
    lines = ["Standing instructions from the user (newer wins on conflict; follow them unless the user changes them):"]
    if goal is not None:
        lines.append(f"- Goal (turn {goal.turn}): {_shorten(goal.text)}")
    if now is not None:
        lines.append(f"- Now (turn {now.turn}): {_shorten(now.text)}")
    for directive in constraints[-5:]:
        lines.append(f"- Keep (turn {directive.turn}): {_shorten(directive.text)}")
    for directive in changed:
        lines.append(f"- Dropped (turn {directive.turn}, changed at turn {directive.superseded_at}): {_shorten(directive.text, 70)}")
    note = "\n".join(lines)
    return note if len(note) <= NOTE_CHAR_BUDGET else note[: NOTE_CHAR_BUDGET - 1].rstrip() + "…"


def render_note(db: _MessageSource, conversation_id: str) -> str:
    """The per-turn context block ('' when it would add nothing)."""
    return render(ledger_for(db, conversation_id))


def review_prompt(db: _MessageSource, conversation_id: str) -> str:
    """A one-time end-of-turn self-check built from the ledger ('' if nothing to check)."""
    ledger = ledger_for(db, conversation_id)
    items = [d for d in (ledger.goal(), *ledger.active("request")[-1:], *ledger.active("constraint")[-5:]) if d]
    if not items:
        return ""
    unique = {d.id: d for d in items}.values()
    checklist = "\n".join(f"- {_shorten(d.text, 140)}" for d in unique)
    return (
        "Before you finish, check your work against the user's instructions:\n"
        f"{checklist}\n"
        "If something is missing or wrong, fix it now with your tools. If everything is done, "
        "reply with one short line confirming what you delivered. Do not repeat earlier output."
    )
