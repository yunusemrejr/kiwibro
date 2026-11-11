"""Agent loop: one worker thread per turn, SSE queue, tools, approvals, abort."""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterator

from . import (
    adapters, autofacts, checkpoints, compute, config, delegate, directives, guard, guidance, inference, grounding, library, lookfirst, plan, prompts, reactions, readings, recall, research, safety,
    sessions, shape, steer, terms, toolcall, tools, weather, workspaces,
)
from .context import (
    COMPACT_TRIGGER, ContextEngine, PreparedContext, build_llm_messages, combine_system_prompt, compute_budget,
    estimate_tokens,
)
from .db import Database
from .llm import ADAPTER_TASK, ADAPTERS_OFF, ChatResult, ContextLimitError, LLMError, LLMClient, TurnAborted
from .memory import MemoryEngine
from . import notes as notes_module
from .notes import Notebook
from .sandbox import LIST_MAX_DEPTH, SandboxError, Workspace
from .search import SearchEngine
from .tools import ToolContext, ToolResult

log = logging.getLogger(__name__)

NOTES_HEADER = "Notes the user kept that may bear on this message (use the note tool for the full text):"


# Shares of the message budget: the tool schemas may take at most this much of it (else
# a smaller set is sent), and one user message at most this much (else it is paged).
TOOLS_SHARE_MAX = 0.35
USER_MESSAGE_SHARE = 0.4
# Reply cap while a planned document is still unwritten (thinking + the whole file in one call).
WRITE_ROUND_MAX_TOKENS = 6144
WRITTEN_NOTE_MIN_CHARS = 40  # a reply shorter than this is a remark, not the note that was asked for


class BusyError(Exception):
    """A turn is already running for this conversation."""


class ConversationNotFound(Exception):
    """Unknown conversation id."""


_LEADING_NOISE = re.compile(
    r"^(?:(?:please|pls|kindly|hey|hi|hello|ok|okay|so|now|just)\b[\s,]+)+", re.IGNORECASE
)
_FILLER_PREFIX = re.compile(
    r"^(?:there\s+(?:has|have)\s+been|there\s+(?:is|are|was|were)|what(?:'s|\s+is)?\s+"
    r"(?:happened|happening|going\s+on)(?:\s+(?:yesterday|today|recently|lately|this\s+week"
    r"|last\s+(?:night|week|month)))?)\s+(?:a\s+|an\s+|the\s+)?",
    re.IGNORECASE,
)
_LEADING_PREPOSITION = re.compile(
    r"^(?:from|about|regarding|in|on|at|by|for|to|with)\s+", re.IGNORECASE
)
_REQUEST_PREFIX = re.compile(
    r"^(?:(?:can|could|would|will)\s+you(?:\s+please)?|i\s+(?:want|need)\s+you\s+to"
    r"|i'd\s+like\s+you\s+to|i\s+would\s+like\s+you\s+to"
    r"|(?:can|could)(?=\s+(?:search|find|look|write|make|create|help|explain|show|give|check|fix|build"
    r"|summari[sz]e)\b))\s+",
    re.IGNORECASE,
)
_TOOL_PREFIX = re.compile(
    r"^(?:please\s+)?(?:use|call|invoke|run)\s+(?:the\s+)?[\w.\-]+(?:\s+tool)?\s*(?:to\s+)?",
    re.IGNORECASE,
)
_SEARCH_PREFIX = re.compile(
    r"^(?:please\s+)?(?:search|look\s+up|find|google)\s+(?:the\s+web\s+)?(?:for\s+)?",
    re.IGNORECASE,
)
_REPLY_PREFIX = re.compile(
    r"^(?:please\s+)?(?:reply|respond|answer|say|output|tell\s+me|tell)\b"
    r"(?:\s+(?:with|only|just|exactly|back|me))*\s*:?\s*",
    re.IGNORECASE,
)
_TRAILING_STOPWORDS = {
    "a",
    "about",
    "into",
    "lately",
    "like",
    "recently",
    "an",
    "and",
    "as",
    "at",
    "but",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "so",
    "than",
    "that",
    "the",
    "then",
    "this",
    "to",
    "was",
    "were",
    "with",
}
_TITLE_BLACKLIST = (
    "the user",
    "user wants",
    "user asked",
    "user wants me",
    "chat title",
    "conversation title",
    "write a",
    "the assistant",
    "assistant should",
    "here is",
    "sure,",
    "based on the",
    "prompt",
    "titled",
    "i will",
)


def derive_title(message: str) -> str:
    """Deterministic, fast conversation title from the first user message."""
    text = " ".join((message or "").split()).strip()
    for _pass in range(4):
        before = text
        for pattern in (
            _LEADING_NOISE,
            _REQUEST_PREFIX,
            _TOOL_PREFIX,
            _REPLY_PREFIX,
            _SEARCH_PREFIX,
            _FILLER_PREFIX,
            _LEADING_PREPOSITION,
        ):
            text = pattern.sub("", text).strip()
        if text == before:
            break
    text = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    clause = re.split(r"\s*[,;]\s+", text, maxsplit=1)[0]
    if len(clause.split()) >= 2:
        text = clause
    text = text.strip("\"'\u201c\u201d\u2018\u2019`.,:;!?()[]{} ")
    if not text:
        return "New chat"
    words = text.split()[:5]
    while words and words[-1].lower() in _TRAILING_STOPWORDS:
        words.pop()
    title = " ".join(words).strip(" \"'\u201c\u201d\u2018\u2019`.,:;!?()[]{}")
    if len(title) > 40:
        title = title[:40].rstrip(" ,.;:-")
    if not title:
        return "New chat"
    return title[:1].upper() + title[1:]


_SMALL_TALK = frozenset(
    "good morning afternoon evening night day sup yo hola merhaba selam test testing whats up new chat howdy "
    "greetings hii hiii heyy there bye".split()
)


REPEAT_MIN_WORDS = 8  # a shorter text is a preface ("Let me look that up"), not an answer
REPEAT_SHARE = 0.6  # of its content words found again in the final reply


def _says_again(earlier: str, final: str) -> bool:
    """Whether ``final`` covers what ``earlier`` (written in the same turn) already said."""
    said, now = set(terms.content_words(earlier)), set(terms.content_words(final))
    return len(said) >= REPEAT_MIN_WORDS and len(said & now) >= REPEAT_SHARE * len(said)


def is_small_talk_title(title: str) -> bool:
    """A title that names no topic ("Hi", "How's it going", "Good morning")."""
    return not [word for word in terms.content_words(title) if word not in _SMALL_TALK]


def retitle_small_talk_chats(db: Database) -> int:
    """Give auto-titled chats that opened with small talk the title of their first real
    request (chats from before re-titling existed). Returns how many were renamed."""
    renamed = 0
    for conversation in db.list_conversations():
        if not conversation.get("title_auto") or not is_small_talk_title(conversation.get("title") or ""):
            continue
        request = next((
            m for m in db.list_messages(conversation["id"], limit=40)
            if m.get("role") == "user" and len(terms.content_words(m.get("content") or "")) >= 3
        ), None)
        title = derive_title(request["content"]) if request else ""
        if title and not is_small_talk_title(title):
            db.update_conversation(conversation["id"], title=title, touch=False)
            renamed += 1
    return renamed


def validate_llm_title(text: str | None, *, prompt_text: str = "") -> str | None:
    """Accept a model-written title only if it looks like a real title."""
    if not text:
        return None
    first = next((line for line in str(text).strip().splitlines() if line.strip()), "")
    first = " ".join(first.split()).strip("\"'\u201c\u201d\u2018\u2019` ")
    if first.lower().startswith("title:"):
        first = first[6:].strip()
    first = first.rstrip(".:;,!?")
    if not first or len(first) > 40 or ":" in first:
        return None
    if not any(char.isalpha() for char in first):
        return None
    words = first.split()
    if not 2 <= len(words) <= 6:
        return None
    lowered = first.lower()
    if any(bad in lowered for bad in _TITLE_BLACKLIST):
        return None
    if prompt_text and SequenceMatcher(None, lowered, prompt_text.lower()).ratio() >= 0.55:
        return None
    return first


@dataclass
class Approval:
    id: str
    conversation_id: str
    tool: str
    args: dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    decision: str | None = None


# Small models can loop on failing tool calls (e.g. guessing absolute paths) until the
# round limit: coach them after a few failures in a row, then refuse further calls so
# they answer. Refusing (not removing the tools) keeps the KV-cached prefix intact.
FAILURE_COACH_AT = 3
FAILURE_PAUSE_AT = 6


def _directives_note(db: Database, conversation_id: str) -> str:
    """The directive ledger's per-turn note ('' when it would add nothing)."""
    try:
        return directives.render_note(db, conversation_id)
    except Exception:  # the ledger must never break a turn
        log.warning("directive note failed", exc_info=True)
        return ""


# Always offered in text-protocol mode: deterministic helpers small models need most.
_TEXT_MODE_EXTRA = ("calculate", "doc_search", "doc_read")


def _text_mode_specs(settings: dict[str, Any]) -> list[tools.ToolSpec]:
    """Tools for the text protocol. Depends only on settings (never on the message),
    so the protocol block in message 0 stays byte-stable across turns."""
    profile = settings.get("tool_profile")
    if profile in (None, "auto"):
        profile = "minimal"
    names = [item["function"]["name"] for item in tools.tool_defs_for(profile=profile, settings=settings)]
    names += [
        name for name in _TEXT_MODE_EXTRA
        if name in tools.TOOL_SPECS and name not in names and tools.permission_mode(settings, name) != "deny"
    ]
    return [tools.TOOL_SPECS[name] for name in names]


@dataclass
class Turn:
    conversation_id: str
    message: str
    requested_model: str | None
    queue: queue.Queue = field(default_factory=queue.Queue)
    abort: threading.Event = field(default_factory=threading.Event)
    started_at: float = field(default_factory=time.monotonic)
    model: str | None = None
    user_message_id: str | None = None
    last_message_id: str | None = None
    partial_text: str = ""
    partial_reasoning: str = ""
    usage_prompt: int = 0
    usage_completion: int = 0
    usage_total: int = 0
    has_usage: bool = False
    approval_ids: list[str] = field(default_factory=list)
    finished: bool = False
    done_sent: bool = False
    tool_signatures: dict[str, int] = field(default_factory=dict)
    tool_calls_made: int = 0
    failure_streak: int = 0  # consecutive failed tool calls (loop breaker)
    stream_tail: str = ""  # last streamed text of this request (degeneration guard)
    loop_checked: int = 0  # stream_tail growth since the last (costlier) loop check
    tool_rounds: int = 0
    steering_notes: list[str] = field(default_factory=list)
    attachments: list[dict[str, Any]] = field(default_factory=list)
    steering_lock: threading.Lock = field(default_factory=threading.Lock)
    system_prompt: str = ""
    turn_context: str = ""
    timer: inference.RoundTimer | None = None
    thinking: bool | None = None
    inference: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)  # guard/safety turn state
    stream_filter: toolcall.StreamFilter | None = None
    page_limit: int = 8000  # tool outputs longer than this are paged into the library
    nudged: bool = False
    reserve: int = 0  # prompt tokens the tool schemas take (kept out of the message budget)
    reviewed: bool = False  # the one end-of-turn self-check has been offered
    pending_review: str = ""  # checklist nudge for the next round (transient)
    plan: plan.TurnPlan | None = None  # the request's steps, when it has several or names a place
    reply_text: str = ""  # everything the model said this turn (the plan checks answers against it)
    truncated_calls: bool = False  # this round's tool calls were cut off by max_tokens
    searches: list[tuple[frozenset[str], str]] = field(default_factory=list)  # (terms, query) this turn
    search_rounds: int = 0  # rounds that ran a web search
    plan_done_at: int | None = None  # search_rounds when every planned tool step was done
    seen_urls: set[str] = field(default_factory=set)  # search results already shown this turn
    think_chars: int = 0  # reasoning streamed by the current request
    think_budget: int = 0  # reasoning allowed before acting (0: no limit for this request)
    facts: steer.Facts | None = None  # what is true this turn; replies are checked against it
    steer_left: int = 1  # corrections Kiwi Sense may still ask for in this turn
    tools_ok: list[str] = field(default_factory=list)  # tools that ran and succeeded this turn
    interim: list[tuple[str, str]] = field(default_factory=list)  # (message id, text) written beside tool calls
    acted_instead: str = ""  # the tool Kiwibro called for a model that was asked to and did not
    bulletin: str = ""  # the news Kiwibro found, shown under the model's answer as its sources
    own_search: bool = False  # the search running now is Kiwibro's own: the best pages are opened and read
    learned: tuple[str, ...] = ()  # facts about the user saved from this message, before the reply
    questions: list[str] = field(default_factory=list)  # the separate questions of a message that only asks
    worked: tuple[tuple[str, str], ...] = ()  # sums the message asks for, worked out exactly before the reply
    reading: Any = None  # what the Kiwi Sense helpers make of the message (readings.TurnReading)
    before: str = ""  # the exchange before this message (for follow-ups and topic shifts)
    steered: list[str] = field(default_factory=list)  # kinds of correction made this turn
    adapter_task: str | None = None  # the task this turn's adapters follow, read once from the user's words
    adapters_tried: list[Any] = field(default_factory=list)  # on when a reply went wrong; off from then on
    adapters_judged: bool = False  # whether that reply was the adapters' doing is settled
    sense_checked: int = 0  # sentences of the streaming reply already read
    reaction: Any = None  # reactions.Reaction when the message turns the last answer down
    evidence: list[str] = field(default_factory=list)  # what this turn's tools returned; the answer is checked against it
    evidence_items: list[dict[str, Any]] = field(default_factory=list)  # search results, for showing them as found
    looked: dict[str, dict[str, Any]] = field(default_factory=dict)  # the figures of this turn's looks at the computer, by tool
    shown_before: str = ""  # the answer the user turned down (a redo lists what was not in it)
    must_mention: tuple[str, ...] = ()  # the subject a refined redo stays on
    redoing: bool = False  # this turn searches again for the request before (the user turned its answer down)
    recalled: list[dict[str, Any]] = field(default_factory=list)  # the saved memories this turn's context shows
    done_note: str = ""  # what the note tool did this turn, in a sentence ("Added to note “Shopping”: cheese.")
    ground_left: int = 1  # new attempts the source check may still ask for
    routes: dict[str, Any] = field(default_factory=dict)  # call id -> the user's folder a code-made look reads

    def emit(self, event: str, data: dict[str, Any]) -> None:
        self.queue.put((event, data))


class AgentManager:
    def __init__(
        self,
        db: Database,
        llm: LLMClient,
        context: ContextEngine,
        memory: MemoryEngine,
        search: SearchEngine,
        settings_provider: Callable[[], dict[str, Any]],
        notes: Notebook | None = None,
    ) -> None:
        self.db = db
        self.llm = llm
        self.context = context
        self.memory = memory
        self.search = search
        self.notes = notes
        self._settings = settings_provider
        self.data_dir = Path(db.path).resolve().parent  # checkpoints + document library live here
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._turns: dict[str, Turn] = {}
        self._thinking: dict[str, bool] = {}  # conversation id -> last turn's thinking decision
        self._approvals: dict[str, Approval] = {}
        self._approvals_guard = threading.Lock()
        # Background memory extraction: queued per conversation, run one at a time once no
        # reply has streamed for memory_idle_s, cancelled (and re-queued) when one starts.
        self._extracting: dict[str, threading.Event] = {}  # running: conversation -> cancel
        self._memory_queue: dict[str, str] = {}  # waiting: conversation -> model
        self._extracting_guard = threading.Lock()
        self._memory_wake = threading.Event()
        self._memory_worker: threading.Thread | None = None
        self._turn_activity_at = time.monotonic()
        self.memory_idle_s = config.MEMORY_IDLE_S
        self.readiness: Any = None  # readiness.Readiness, wired by create_app (load + KV warm-up)
        # "Allow in this chat" (automatic modes): tools whose risky steps the user approved for
        # the rest of a conversation. In memory only: a restart asks again.
        self._chat_allowed: dict[str, set[str]] = {}

    # --- lifecycle ---------------------------------------------------------

    def _lock_for(self, conversation_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(conversation_id, threading.Lock())

    def start_chat(
        self,
        conversation_id: str | None,
        message: str,
        model: str | None = None,
        *,
        attachments: list[dict[str, Any]] | None = None,
        private: bool = False,
    ) -> Turn:
        if conversation_id:
            conversation = self.db.get_conversation(conversation_id)
            if conversation is None:
                raise ConversationNotFound(conversation_id)
        else:
            conversation = self.db.create_conversation(
                title="New chat", model=model or (self._settings().get("model")), private=private
            )
            conversation_id = conversation["id"]
        lock = self._lock_for(conversation_id)
        if not lock.acquire(blocking=False):
            raise BusyError(conversation_id)
        self._turn_activity_at = time.monotonic()
        self._cancel_memory_extraction()  # a live reply gets the whole CPU
        turn = Turn(
            conversation_id=conversation_id, message=message, requested_model=model,
            attachments=list(attachments or []),
        )
        with self._locks_guard:
            self._turns[conversation_id] = turn
        thread = threading.Thread(
            target=self._run_turn,
            args=(turn,),
            name=f"kiwibro-turn-{conversation_id}",
            daemon=True,
        )
        thread.start()
        return turn

    def stop(self, conversation_id: str) -> bool:
        with self._locks_guard:
            turn = self._turns.get(conversation_id)
        if turn is None:
            return False
        turn.abort.set()
        return True

    def steer(self, conversation_id: str, message: str) -> bool:
        """Queue a mid-turn steering note; the worker applies it at the next round."""
        note = (message or "").strip()
        if not note:
            return False
        with self._locks_guard:
            turn = self._turns.get(conversation_id)
        if turn is None:
            return False
        with turn.steering_lock:
            turn.steering_notes.append(note[:4000])
        return True

    def _drain_steering(self, turn: Turn) -> None:
        """Persist queued steering notes as user messages the model will see."""
        with turn.steering_lock:
            notes, turn.steering_notes = turn.steering_notes, []
        for note in notes:
            self.db.add_message(
                turn.conversation_id,
                "user",
                f"[Steering note from the user, sent while you were working] {note}",
            )
            turn.emit("steering", {"text": note})

    def is_busy(self, conversation_id: str) -> bool:
        with self._locks_guard:
            return conversation_id in self._turns

    def active_turns(self) -> int:
        with self._locks_guard:
            return len(self._turns)

    def resolve_approval(self, approval_id: str, decision: str) -> str | None:
        if decision not in ("allow", "always", "deny"):
            return None
        with self._approvals_guard:
            approval = self._approvals.get(approval_id)
        if approval is None or approval.event.is_set():
            return None
        approval.decision = decision
        approval.event.set()
        return decision

    # --- streaming ---------------------------------------------------------

    @staticmethod
    def _frame(event: str, data: dict[str, Any]) -> str:
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        return f"event: {event}\ndata: {payload}\n\n"

    def event_stream(self, turn: Turn) -> Iterator[str]:
        """Yield SSE frames from the turn queue, with idle heartbeats."""
        pending: Any = ...
        try:
            while True:
                if pending is not ...:
                    item, pending = pending, ...
                else:
                    try:
                        item = turn.queue.get(timeout=config.HEARTBEAT_S)
                    except queue.Empty:
                        yield ": ping\n\n"
                        continue
                if item is None:
                    break
                event, data = item
                # Coalesce queued token deltas into one frame: fewer SSE
                # writes and frontend renders when the consumer lags.
                if event in ("delta", "reasoning"):
                    texts = [str(data.get("text") or "")]
                    while True:
                        try:
                            nxt = turn.queue.get_nowait()
                        except queue.Empty:
                            break
                        if nxt is None or nxt[0] != event:
                            pending = nxt
                            break
                        texts.append(str(nxt[1].get("text") or ""))
                    data = {**data, "text": "".join(texts)}
                yield self._frame(event, data)
        finally:
            if not turn.done_sent:
                turn.abort.set()

    # --- worker ------------------------------------------------------------

    def _run_turn(self, turn: Turn) -> None:
        try:
            self._execute(turn)
        except TurnAborted:
            self._finish_aborted(turn)
        except LLMError as exc:
            self._finish_error(turn, exc.kind, exc.message)
        except Exception:  # never leak a traceback into the stream
            if self.db.get_conversation(turn.conversation_id) is None:
                # The chat was deleted mid-turn; nothing left to report to.
                log.info("turn %s closed because the conversation was deleted", turn.conversation_id)
            else:
                log.exception("turn crashed for %s", turn.conversation_id)
                self._finish_error(turn, "internal", "Something went wrong while generating the reply.")
        finally:
            self._cleanup(turn)

    def _cleanup(self, turn: Turn) -> None:
        with self._approvals_guard:
            for approval_id in turn.approval_ids:
                self._approvals.pop(approval_id, None)
        with self._locks_guard:
            self._turns.pop(turn.conversation_id, None)
            lock = self._locks.get(turn.conversation_id)
        if lock is not None and lock.locked():
            try:
                lock.release()
            except RuntimeError:  # pragma: no cover - defensive
                pass
        # Lock objects are intentionally kept in memory (one per conversation ever
        # used). Dropping and recreating them could let two turns race on a chat.
        self._turn_activity_at = time.monotonic()
        turn.finished = True
        turn.done_sent = True
        turn.queue.put(None)

    def _execute(self, turn: Turn) -> None:
        settings = self._settings()
        conversation = self.db.get_conversation(turn.conversation_id)
        if conversation is None:
            raise ConversationNotFound(turn.conversation_id)
        user_message = self._unanswered_repeat(turn) or self.db.add_message(
            turn.conversation_id, "user", turn.message,
            meta={"attachments": turn.attachments} if turn.attachments else None,
        )
        turn.user_message_id = user_message["id"]
        # Content-safety state: taint from injected content and secret placeholders
        # carry over between turns of the same conversation.
        turn.safety = guard.begin_turn(turn.conversation_id, self._workspace(turn.conversation_id).root)
        turn.safety["user_message"] = turn.message  # lets safety tell user-asked folder switches apart
        self._raise_if_aborted(turn)

        try:
            model = self.llm.resolve_model(turn.requested_model)
        except LLMError as exc:
            self._emit_meta(turn, conversation, settings, model=None)
            raise exc
        turn.model = model
        if conversation["model"] != model:
            conversation = (
                self.db.update_conversation(turn.conversation_id, model=model, touch=False) or conversation
            )
        self._raise_if_aborted(turn)
        if self._english_only(turn, conversation, settings, model):
            return
        turn.reading = self._read_message(turn, settings)
        turn.reaction = self._read_reaction(turn, settings)

        model_info = self.llm.model_info(model)
        if turn.attachments:
            # meta is contractually the first SSE event: send it before looking at the
            # files (a vision helper can take a while); real context stats follow.
            self._emit_meta(turn, conversation, settings, model=model)
            self._see_attachments(turn, model, user_message)
        # KV-cache rule: message 0 stays byte-identical across turns so the server
        # reuses the cached prefix; the clock, retrieved memories and directives
        # ride on the newest user message (fixed for the whole turn).
        system_prompt = self._system_prompt(
            {**settings, "workspace": self._folder(turn.conversation_id)}, model=model, model_info=model_info
        )
        # One profile-sized set, identical for every turn and conversation: the tool list
        # renders ahead of all messages, so any change re-prefills them. Its tokens are
        # reserved in the budget so compaction starts before the window overflows.
        tool_list = self._tool_list(model, settings, model_info)
        turn.reserve = estimate_tokens(json.dumps(tool_list, ensure_ascii=False)) if tool_list else 0
        self._page_long_message(turn, user_message, model)
        turn.plan = self._plan_for(turn, settings)
        turn.questions = plan.questions(turn.message) if turn.plan is None else []
        turn.worked = self._worked(turn)
        captured = self._capture_memory(turn, settings)
        turn_context = self._turn_context(turn, settings)
        turn.system_prompt, turn.turn_context = system_prompt, turn_context
        prepared = self.context.prepare(
            turn.conversation_id, model, system_prompt, turn_context=turn_context, reserve_tokens=turn.reserve
        )
        # meta is contractually the first SSE event; context follows it.
        if turn.attachments:
            turn.emit("context", {**prepared.stats, "compacted": prepared.compacted or prepared.trimmed})
        else:
            self._emit_meta(turn, conversation, settings, model=model, prepared=prepared)
            self._emit_context(turn, prepared)
        if turn.plan is not None:  # after meta, which is always the first event
            turn.emit("plan", {"steps": turn.plan.statuses()})
        if any(captured.values()):
            turn.emit("memory", captured)
        self._raise_if_aborted(turn)
        turn.facts = self._facts(turn, settings, tool_list, model_info, captured, prepared)

        if self.readiness is not None:
            # A background warm-up of this very prefix is cheaper to wait for than to repeat.
            self.readiness.before_turn(
                model, turn.conversation_id, abort=turn.abort,
                on_wait=lambda: turn.emit("status", {"phase": "thinking", "label": "Getting the model ready"}),
            )
            self._raise_if_aborted(turn)
        model_info = model_info or {}
        turn.page_limit = library.page_limit(prepared.budget)
        # Models without native tool calling use the text protocol: tools are
        # described in message 0 and calls are parsed from the reply text.
        text_mode = bool(settings.get("agent_enabled", True)) and not model_info.get("tools", False)
        try:
            max_rounds = int(settings.get("max_tool_rounds") or config.MAX_TOOL_ROUNDS)
        except (TypeError, ValueError):
            max_rounds = config.MAX_TOOL_ROUNDS
        max_rounds = max(config.TOOL_ROUNDS_RANGE[0], min(max_rounds, config.TOOL_ROUNDS_RANGE[1]))

        looked = self._look_first(turn, settings)
        if self._search_first(turn, settings) or looked:
            self._raise_if_aborted(turn)
            if turn.reply_text.strip():  # answered from the results alone
                max_rounds = 0
                self._close_plan(turn, model)
            else:
                prepared = self.context.extend(prepared, turn.conversation_id, model, system_prompt,
                                               turn_context=turn_context)
                self._emit_context(turn, prepared)

        for round_index in range(max_rounds):
            self._raise_if_aborted(turn)
            if round_index > 0:
                self._drain_steering(turn)
                # Append only the new assistant/tool messages; extend() falls
                # back to a full prepare if the compaction trigger is crossed.
                prepared = self.context.extend(
                    prepared, turn.conversation_id, model, system_prompt, turn_context=turn_context
                )
                self._emit_context(turn, prepared)
            turn.emit(
                "status",
                {"phase": "thinking", "label": "Thinking" if round_index == 0 else "Working"},
            )
            turn.partial_text = ""
            turn.partial_reasoning = ""
            review, turn.pending_review = turn.pending_review, ""
            result = self._stream_with_recovery(
                turn, model, prepared, tool_list, round_index, text_mode=text_mode, nudge=review or None
            )
            if not turn.nudged and toolcall.is_empty_reply(result.content, result.tool_calls):
                # Small models sometimes stop with an empty reply (most often after tool
                # results); one transient nudge (never stored) gets them going.
                turn.nudged = True
                self._record_usage(turn, result, prepared)
                log.info("turn %s round %s: empty reply, nudging once", turn.conversation_id, round_index)
                result = self._stream_with_recovery(
                    turn, model, prepared, tool_list, round_index, text_mode=text_mode, nudge=toolcall.EMPTY_REPLY_NUDGE
                )
            log.info(
                "turn %s round %s: finish_reason=%s content_chars=%s tool_calls=%s",
                turn.conversation_id,
                round_index,
                result.finish_reason,
                len(result.content),
                len(result.tool_calls),
            )
            if result.tools_disabled:
                tool_list = None
            self._act_instead(turn, result)
            result = self._steer_reply(turn, model, prepared, tool_list, round_index, result, text_mode)
            result = self._from_memory(turn, model, prepared, round_index, result, text_mode)
            result = self._not_said_before(turn, model, prepared, round_index, result, text_mode)
            result = self._ground_reply(turn, model, prepared, tool_list, round_index, result, text_mode)
            self._shape_reply(turn, result)
            self._record_usage(turn, result, prepared)
            assistant_message = None
            if result.content or result.reasoning or result.tool_calls:
                assistant_message = self.db.add_message(
                    turn.conversation_id,
                    "assistant",
                    result.content,
                    reasoning=result.reasoning,
                    tool_calls=result.tool_calls or None,
                    model=model,
                    tokens=(result.usage or {}).get("completion_tokens"),
                )
                turn.last_message_id = assistant_message["id"]
                if result.tool_calls and result.content.strip():
                    turn.interim.append((assistant_message["id"], result.content))
            turn.partial_text = ""
            turn.partial_reasoning = ""
            if result.content:
                turn.reply_text += "\n" + result.content
            if not result.tool_calls:
                self._retract_repeats(turn, result.content)
                last_round = round_index >= max_rounds - 1
                if last_round:
                    turn.pending_review = ""
                elif turn.plan is not None:
                    # Checked in code (files on disk, searches made, topics answered); the generic
                    # self-check round only runs for steps code cannot check.
                    turn.pending_review = turn.plan.followup(turn.reply_text) or (
                        "" if turn.plan.checkable else self._review_nudge(turn, settings)
                    )
                else:
                    turn.pending_review = self._unanswered_nudge(turn) or self._review_nudge(turn, settings)
                if not turn.pending_review:
                    self._close_plan(turn, model)
                    break
                turn.emit("status", {"phase": "thinking", "label": "Finishing the request" if turn.plan else "Checking the work"})
                continue
            turn.tool_rounds = round_index + 1
            turn.truncated_calls = result.finish_reason == "length"
            self._handle_tool_calls(turn, result.tool_calls, settings)
            self._raise_if_aborted(turn)
            if (turn.done_note and turn.acted_instead == "note" and turn.plan is None
                    and turn.pending_review in (prompts.APPENDED_FOR_YOU, prompts.NOTED_FOR_YOU)):
                # Kiwibro kept the dictated note itself: it says so itself too, in the tool's own words.
                turn.pending_review = ""
                said = self.db.add_message(turn.conversation_id, "assistant", turn.done_note, model=model,
                                           meta={"done_for_you": True})
                turn.last_message_id = said["id"]
                turn.emit("delta", {"text": ("\n\n" if turn.reply_text.strip() else "") + turn.done_note})
                turn.reply_text += "\n" + turn.done_note
                turn.inference["said_by_kiwibro"] = "note"
                break
            if round_index == max_rounds - 1:
                note = "I reached the tool-use limit for this turn, so I stopped here."
                left = turn.plan.summary_line(turn.reply_text) if turn.plan is not None else ""
                note = f"{note} {left}" if left else note
                note_message = self.db.add_message(
                    turn.conversation_id, "assistant", note, model=model, meta={"truncated": True}
                )
                turn.last_message_id = note_message["id"]
                turn.emit("delta", {"text": "\n\n" + note})
                break

        if not turn.reply_text.strip():
            self._answer_anyway(turn, model, prepared, tool_list, text_mode)
        self._show_bulletin(turn, model)

        # Final answer phase. Title is deterministic-first so `done` stays fast; memory
        # extraction runs in a daemon thread after `done` (the stream closes immediately,
        # the UI refreshes its memory count from /api/bootstrap or /api/memories later).
        self._raise_if_aborted(turn)
        self._drain_steering(turn)
        self._maybe_title(turn)
        self._insert_usage(turn, model)
        self._start_memory_extraction(turn.conversation_id, model)
        turn.emit(
            "done",
            {
                "message_id": turn.last_message_id or turn.user_message_id,
                "aborted": False,
                "model": model,
                "usage": self._usage_payload(turn),
                "stats": {
                    "tool_calls": turn.tool_calls_made,
                    "tool_rounds": turn.tool_rounds,
                    "duration_ms": int((time.monotonic() - turn.started_at) * 1000),
                },
                "inference": turn.inference or None,
                **({"plan": turn.plan.statuses()} if turn.plan is not None else {}),
            },
        )

    def _english_only(self, turn: Turn, conversation: dict[str, Any], settings: dict[str, Any], model: str) -> bool:
        """A message that is not in English gets one fixed reply asking for English, written by
        code: no model is asked, so it is the same whatever model is loaded."""
        if not settings.get("english_only", True) or not settings.get("sense_enabled", True):
            return False
        found = readings.language(turn.message)
        if found is None:
            return False
        text = readings.english_only_reply(found)
        log.info("turn %s: not English (%s, %.2f); asked for English", turn.conversation_id, found.code,
                 found.confidence)
        self._emit_meta(turn, conversation, settings, model=model)
        reply = self.db.add_message(turn.conversation_id, "assistant", text, model=model,
                                    meta={"english_only": True, "language": found.code})
        turn.last_message_id = reply["id"]
        turn.reply_text = text
        turn.inference["english_only"] = found.code
        turn.emit("delta", {"text": text})
        turn.emit("done", {
            "message_id": reply["id"], "aborted": False, "model": model, "usage": self._usage_payload(turn),
            "stats": {"tool_calls": 0, "tool_rounds": 0,
                      "duration_ms": int((time.monotonic() - turn.started_at) * 1000)},
            "inference": turn.inference,
        })
        return True

    def _read_message(self, turn: Turn, settings: dict[str, Any]) -> readings.TurnReading:
        """The helpers' reading of the message, with the exchange before it for topic shifts."""
        if not settings.get("sense_enabled", True):
            return readings.TurnReading()
        before = ""
        try:
            earlier = [m for m in self.db.recent_messages(turn.conversation_id, limit=6)
                       if m.get("role") in ("user", "assistant") and m.get("id") != turn.user_message_id
                       and str(m.get("content") or "").strip()]
            users = [i for i, m in enumerate(earlier) if m["role"] == "user"]
            if users:
                after = [m for m in earlier[users[-1] + 1:] if m["role"] == "assistant"]
                reply = str(after[-1]["content"]) if after else ""
                before = f"{str(earlier[users[-1]]['content'])[:600]} {reply[:300]}".strip()
        except Exception:  # a convenience; never break a turn
            log.warning("turn %s: could not read the exchange before", turn.conversation_id, exc_info=True)
        found = readings.read(turn.message, before)
        turn.before = before
        summary = found.summary()
        if summary:
            turn.inference["reading"] = summary
        return found

    def _search_first(self, turn: Turn, settings: dict[str, Any]) -> bool:
        """The message asks for a search, or about something only a search knows (news, weather,
        prices, who won): Kiwibro runs it before the model writes a word. A model that answers
        first writes from what it remembers, and a streamed reply cannot be taken back, so the
        answer is written from the results instead, whatever the model."""
        if turn.facts is None or turn.facts.small_talk or "web_search" not in turn.facts.tools:
            return False
        # A reaction that carries a question of its own is answered as asked: nothing is done again,
        # so reading a normal message as a complaint by mistake starts no search and hides no answer.
        redo = turn.reaction if turn.reaction is not None and not turn.reaction.asks else None
        try:
            reading = turn.reading or readings.TurnReading()
            if redo is not None:
                query = reactions.redo_query(redo) if self._redo_on_web(turn, redo) else ""
                if query and redo.fresher and not re.search(r"\b(?:today|latest|current|now|this\s+(?:week|year))\b", query, re.I):
                    query = f"{query} latest"
            elif turn.reaction is None and reactions.score(turn.message, reading) >= reactions.SURE:
                query = ""  # a complaint with nothing before it: its own words are not a search
            else:
                query = self._own_search(turn, reading)
        except Exception:  # a convenience; never break a turn
            log.warning("turn %s: could not read the search the message asks for", turn.conversation_id, exc_info=True)
            return False
        if redo is None and weather.place_asked(turn.message):
            query = turn.message.strip()  # always looked up, as asked: the search reads the place from it
        if not query:
            return False
        if redo is not None:
            turn.redoing = True
            self._set_aside_rejected(turn)
        return self._run_search_first(turn, settings, query)

    def _redo_on_web(self, turn: Turn, reaction: Any) -> bool:
        """Whether doing the request again means searching: it asked for the web or for something
        only the web knows, the user now says to use the web, or the answer before came from a search."""
        if reaction.wants_web or reaction.fresher:
            return True
        if plan.search_request(reaction.request)[0] or research.asks_for_news(reaction.request):
            return True
        return bool(self._last_search(turn))

    def _last_search(self, turn: Turn) -> str:
        """The query of the web search the turn before this one ran ("" if it ran none)."""
        earlier = [m for m in self.db.recent_messages(turn.conversation_id, limit=14)
                   if m.get("id") != turn.user_message_id]
        users = [i for i, m in enumerate(earlier) if m.get("role") == "user"]
        for message in reversed(earlier[users[-1]:] if users else []):
            for call in message.get("tool_calls") or []:
                function = (call or {}).get("function") or {}
                if function.get("name") == "web_search":
                    try:
                        return str(json.loads(function.get("arguments") or "{}").get("query") or "")
                    except (TypeError, ValueError):
                        return ""
        return ""

    def _set_aside_rejected(self, turn: Turn) -> None:
        """The answer the user turned down is not shown to the model again: everything Kiwibro
        said in the turn before this one. A plain rejection also keeps what was shown, so the
        new search lists other stories first; a message that says what was meant keeps the
        subject of the request, so the new results stay on it."""
        shown: list[str] = []
        earlier = [m for m in self.db.recent_messages(turn.conversation_id, limit=14) if m.get("id") != turn.user_message_id]
        for message in reversed(earlier):
            if message.get("role") == "user":
                break
            text = str(message.get("content") or "").strip()
            if message.get("role") == "assistant" and text:
                self.db.update_message(message["id"], meta={**(message.get("meta") or {}), "rejected": True})
            if text:
                shown.append(text)
        reaction = turn.reaction
        if reaction is None:
            return
        if reaction.kind == reactions.REFINE:
            asked = research.news_request(plan.query_of(reaction.request) or reaction.request)
            subject = asked.topic if asked is not None and asked.topic else plan.query_of(reaction.request)
            turn.must_mention = tuple(word for word in terms.content_words(subject) if len(word) >= 3)[:6]
        else:
            turn.shown_before = "\n".join(shown)

    def _read_reaction(self, turn: Turn, settings: dict[str, Any]) -> Any:
        """Whether the message turns the last answer down (reactions.py), and the request it is about.
        The helpers' reading adds to the signs in the words; with the helpers off the words decide alone."""
        try:
            earlier = [m for m in self.db.recent_messages(turn.conversation_id, limit=14)
                       if m.get("role") in ("user", "assistant") and m.get("id") != turn.user_message_id]
            found = reactions.read(turn.message, turn.reading, earlier)
            if found is None:
                return None
            current = self.db.get_message(turn.user_message_id) or {}
            self.db.update_message(turn.user_message_id, meta={**(current.get("meta") or {}), "reaction": found.kind})
        except Exception:  # a convenience; never break a turn
            log.warning("turn %s: could not read the reaction", turn.conversation_id, exc_info=True)
            return None
        log.info("turn %s: the message turns the last answer down (%s, %.2f)", turn.conversation_id, found.kind, found.score)
        turn.inference["reaction"] = found.kind
        return found

    def _look_first(self, turn: Turn, settings: dict[str, Any]) -> bool:
        """A question about this computer or one of the user's folders: Kiwibro makes the
        read-only call first, so the answer is written from what is really there."""
        if turn.facts is None or turn.facts.small_talk or turn.plan is not None:
            return False
        if turn.reaction is not None and not turn.reaction.asks:
            return False
        try:
            earlier = [m for m in self.db.recent_messages(turn.conversation_id, limit=8)
                       if m.get("role") == "user" and m.get("id") != turn.user_message_id]
            before = str(earlier[-1].get("content") or "") if earlier else ""
            # Not only the tools listed for the model: every read-only look the user has not switched off.
            allowed = frozenset(name for name in ("system_info", "list_files", "git")
                                if name in tools.TOOL_SPECS and tools.permission_mode(settings, name) != "deny")
            looks = lookfirst.calls(turn.message, self._workspace(turn.conversation_id).root, allowed, before)
        except Exception:  # a convenience; never break a turn
            log.warning("turn %s: could not look first", turn.conversation_id, exc_info=True)
            return False
        if not looks:
            return False
        calls = []
        for look in looks:
            call = {"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                    "function": {"name": look.name, "arguments": json.dumps(look.args, ensure_ascii=False)}}
            if look.folder is not None:
                turn.routes[call["id"]] = plan.Route(Workspace(look.folder), dict(look.args), look.shown)
            calls.append(call)
        log.info("turn %s: looking first with %s", turn.conversation_id, [look.name for look in looks])
        message = self.db.add_message(turn.conversation_id, "assistant", "", tool_calls=calls, model=turn.model)
        turn.last_message_id = message["id"]
        turn.inference["looked_first"] = [look.name for look in looks]
        self._handle_tool_calls(turn, calls, settings)
        said = ""
        try:
            for look in looks:  # a folder inside the workspace is named by its path
                if look.name == "list_files" and "list_files" in turn.looked and not turn.looked["list_files"].get("shown"):
                    turn.looked["list_files"]["shown"] = look.shown or str(look.args.get("path") or "")
            if all(look.name in turn.looked for look in looks):
                said = lookfirst.answer(turn.message, turn.looked)
        except Exception:  # a convenience; the model answers from the result instead
            log.warning("turn %s: could not answer from the look", turn.conversation_id, exc_info=True)
        if said:
            log.info("turn %s: the look is the answer", turn.conversation_id)
            reply = self.db.add_message(turn.conversation_id, "assistant", said, model=turn.model, meta={"from_look": True})
            turn.last_message_id = reply["id"]
            turn.reply_text = said
            turn.inference["answered_from_look"] = True
            turn.emit("delta", {"text": said})
            return True
        turn.pending_review = prompts.LOOKED_FIRST
        return True

    def _own_search(self, turn: Turn, reading: Any) -> str:
        """The search the message itself asks for ("" for none)."""
        query, _borrowed = plan.search_request(turn.message)
        if query and readings.needs_no_search(reading) and not plan.told_to_search(turn.message):
            log.info("turn %s: the helpers see no need for the web; no search first", turn.conversation_id)
            query = ""
        elif (not query and readings.wants_search(reading) and not readings.follows_up(reading)
              and not readings.is_feedback(reading) and not readings.names_a_file_or_link(turn.message)):
            query = plan.query_of(turn.message)  # live data or an explicit ask the word rules missed
        query = research.with_time_words(query, turn.message) if query else ""
        if not query and research.asks_for_news(turn.message):  # "summarize today's ai news"
            query = turn.message.strip().rstrip("?.! ")
        if not query:
            query = self._follow_up_search(turn, reading)
        return query

    def _follow_up_search(self, turn: Turn, reading: Any) -> str:
        """"And who came second?" after an answer that came from a search: the same subject is
        searched again with the new question, instead of being answered from memory."""
        text = turn.message.strip()
        if len(text.split()) > 12 or reactions.reworks(text) or plan._OWN.search(text):
            return ""
        asks = text.endswith("?") or bool(plan._QUESTION.match(text))
        leads = bool(re.match(r"(?:and|what\s+about|how\s+about|also|but|then)\b", text, re.I))
        if not ((asks or leads) and (leads or readings.follows_up(reading) or readings.continues(reading))):
            return ""
        previous = self._last_search(turn)
        if not previous:
            return ""
        return " ".join(f"{research.broader(previous)} {plan.query_of(text) or text}".split())[:200]

    def _run_search_first(self, turn: Turn, settings: dict[str, Any], query: str) -> bool:
        call = {
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": "web_search", "arguments": json.dumps({"query": query}, ensure_ascii=False)},
        }
        log.info("turn %s: searching first for %s", turn.conversation_id, json.dumps(query, ensure_ascii=False)[:160])
        message = self.db.add_message(turn.conversation_id, "assistant", "", tool_calls=[call], model=turn.model)
        turn.last_message_id = message["id"]
        turn.acted_instead = "web_search"
        turn.inference["searched_first"] = True
        turn.own_search = True
        try:
            self._handle_tool_calls(turn, [call], settings)
        finally:
            turn.own_search = False
        more = turn.plan is not None and any(step.kind != "web" for step in turn.plan.remaining())
        found = [m for m in self.db.recent_messages(turn.conversation_id, limit=2) if m.get("role") == "tool"]
        news = bool(found) and str(found[-1].get("content") or "").startswith(research.NEWS_HEAD)
        bulletin = "\n".join(line for line in str(found[-1]["content"]).splitlines()
                             if not line.startswith("[Kiwibro")).strip() if news else ""  # notes for the model
        if news and not more and research.bulletin_is_the_answer(turn.message):
            log.info("turn %s: the news bulletin is the answer", turn.conversation_id)
            reply = self.db.add_message(turn.conversation_id, "assistant", bulletin, model=turn.model,
                                        meta={"news_bulletin": True})
            turn.last_message_id = reply["id"]
            turn.reply_text = bulletin
            turn.inference["news_bulletin"] = True
            turn.emit("delta", {"text": bulletin})
            return True
        told = str(found[-1].get("content") or "") if found else ""
        if told.startswith(research.WEATHER_HEAD) and not more and len(plan.questions(turn.message)) <= 1:
            said = told.split("\n\n", 1)[-1].strip()
            log.info("turn %s: the forecast data is the answer", turn.conversation_id)
            reply = self.db.add_message(turn.conversation_id, "assistant", said, model=turn.model, meta={"from_look": True})
            turn.last_message_id = reply["id"]
            turn.reply_text = said
            turn.inference["answered_from_look"] = True
            turn.emit("delta", {"text": said})
            return True
        if news and not more:
            turn.bulletin = bulletin
        then = prompts.SEARCHED_FIRST_THEN if more else ""
        if turn.redoing and turn.reaction is not None and not news:
            turn.pending_review = prompts.REDO_FIRST.format(request=turn.reaction.request.strip()[:300], then=then)
        else:
            turn.pending_review = (prompts.NEWS_FIRST if news else prompts.SEARCHED_FIRST).format(then=then)
        return True

    _PLACEHOLDER = re.compile(r"\[(?!Kiwibro)[A-Za-z][A-Za-z _/'-]{2,40}\](?!\()|\{\{[^{}\n]{1,40}\}\}|<(?:insert|your|placeholder)[^<>\n]{0,40}>", re.I)

    _ONLY_A_BRACKET = re.compile(r"\s*\[(?!Kiwibro)[^\[\]]{3,300}\]\s*")
    # "[One coherent paragraph summarizing the news from the results.]": a sentence in brackets
    # that describes the answer instead of giving it. No digits, quotes or code in it, and not a link.
    _BRACKET_SENTENCE = re.compile(r"\[(?!Kiwibro)([A-Za-z][^\[\]\n\d\"=`]{24,300})\](?!\()")
    _CODE = re.compile(r"```.*?(?:```|\Z)|`[^`\n]+`", re.S)
    BRACKET_SENTENCE_WORDS = 5

    def _blanks(self, text: str) -> list[str]:
        """The placeholders in a reply: a slot left to fill ("[Name]", "{{city}}"), or a bracketed
        line that says what should be written there."""
        prose = self._CODE.sub(" ", text or "")
        found = list(self._PLACEHOLDER.findall(prose))
        found += [inner[:60] for inner in self._BRACKET_SENTENCE.findall(prose)
                  if len(inner.split()) >= self.BRACKET_SENTENCE_WORDS]
        if not found and self._ONLY_A_BRACKET.fullmatch(text or ""):
            found = [text.strip()[:60]]  # the whole reply is a template line
        return list(dict.fromkeys(found))

    def _sources(self, turn: Turn) -> tuple[list[str], bool]:
        """What the answer must come from, and whether it may say nothing else. This turn's tool
        results; for "summarize them", what the turn before found and said."""
        if turn.evidence and (plan.WEB_TOOLS.intersection(turn.tools_ok) or turn.inference.get("looked_first")):
            return turn.evidence, bool(plan.WEB_TOOLS.intersection(turn.tools_ok))
        if turn.evidence:  # other tools ran: the model may still answer from what it knows
            return [], False
        if not reactions.reworks(turn.message):
            return [], False
        earlier = [m for m in self.db.recent_messages(turn.conversation_id, limit=10)
                   if m.get("id") != turn.user_message_id and m.get("role") in ("assistant", "tool")]
        texts = [str(m.get("content") or "") for m in earlier if str(m.get("content") or "").strip()]
        return (texts, True) if any(m.get("role") == "tool" for m in earlier) else ([], False)

    def _known(self, turn: Turn) -> list[str]:
        known = [turn.message, turn.turn_context, *(turn.facts.asked_before if turn.facts else ())]
        known += [f"{asked} {result}" for asked, result in turn.worked]
        if turn.reaction is not None:
            known.append(turn.reaction.request)
        return known

    REPEAT_MIN_CHARS = 20

    def _not_said_before(self, turn: Turn, model: str, prepared: PreparedContext, round_index: int,
                         result: ChatResult, text_mode: bool) -> ChatResult:
        """A small model deep in a session sometimes answers a new question with an old answer,
        word for word ("where do I live?" gets the cat's name it gave three turns ago). Such a
        reply is written once more, with the two questions set side by side."""
        def bare(words: str) -> str:  # "**Pamuk**." and "pamuk" are the same answer
            return " ".join(re.sub(r"[^\w\s]", "", words.lower()).split())

        text = bare(result.content or "")
        if result.tool_calls or len(text) < self.REPEAT_MIN_CHARS or turn.tools_ok or turn.abort.is_set():
            return result
        if turn.facts is not None and turn.facts.small_talk:
            return result  # two greetings may well get the same greeting back
        try:
            earlier = [m for m in self.db.recent_messages(turn.conversation_id, limit=24)
                       if m.get("id") != turn.user_message_id and m.get("role") in ("user", "assistant")]
            asked = ""
            for message in earlier:
                if message.get("role") == "user":
                    asked = " ".join(str(message.get("content") or "").split())
                elif bare(str(message.get("content") or "")) == text and asked and bare(asked) != bare(turn.message):
                    break
            else:
                return result
            log.info("turn %s: the reply repeats an answer to an earlier question; asking again", turn.conversation_id)
            turn.inference["repeated_answer"] = True
            self._record_usage(turn, result, prepared)
            turn.partial_text = ""
            turn.emit("rewrite", {"text": "", "reasoning": ""})
            retry = self._stream_with_recovery(
                turn, model, prepared, None, round_index, text_mode=text_mode,
                nudge=prompts.SAID_BEFORE.format(earlier=asked[:200], message=turn.message.strip()[:300]))
            if retry.content.strip() and not retry.tool_calls and bare(retry.content) != text:
                return retry
        except (TurnAborted, LLMError):
            raise
        except Exception:  # a check must never break a reply
            log.warning("turn %s: repeat check failed", turn.conversation_id, exc_info=True)
        return result

    def _from_memory(self, turn: Turn, model: str, prepared: PreparedContext, round_index: int,
                     result: ChatResult, text_mode: bool) -> ChatResult:
        """"Where do I live?" is answered by a saved fact. A reply without that fact's own name or
        figure gets the fact shown once more; if it is still missing, Kiwibro says the fact itself."""
        if result.tool_calls or turn.tools_ok or turn.abort.is_set() or not turn.recalled:
            return result
        try:
            if not recall.asks_for_a_fact(turn.message):
                return result
            newest = sorted(turn.recalled, key=lambda memory: str(memory.get("updated_at") or ""), reverse=True)
            found = recall.answers(turn.message, [str(memory.get("content") or "") for memory in newest])
            if not found or recall.said(result.content or "", found):
                return result
            log.info("turn %s: the reply leaves out the saved fact it was asked for; showing it again", turn.conversation_id)
            turn.inference["from_memory"] = "asked again"
            self._record_usage(turn, result, prepared)
            turn.partial_text = ""
            turn.emit("rewrite", {"text": "", "reasoning": ""})
            retry = self._stream_with_recovery(
                turn, model, prepared, None, round_index, text_mode=text_mode, cool=True,
                nudge=prompts.FROM_MEMORY.format(facts="\n".join(f"- {fact}" for fact, _telling in found[:3]),
                                                 message=turn.message.strip()[:300]))
            if retry.content.strip() and not retry.tool_calls and recall.said(retry.content, found):
                return retry
            turn.inference["from_memory"] = "said by Kiwibro"
            retry.content, retry.tool_calls = recall.told(found), []
            turn.partial_text = retry.content
            turn.emit("rewrite", {"text": retry.content, "reasoning": ""})
            return retry
        except (TurnAborted, LLMError):
            raise
        except Exception:  # a check must never break a reply
            log.warning("turn %s: memory check failed", turn.conversation_id, exc_info=True)
        return result

    def _ground_reply(
        self, turn: Turn, model: str, prepared: PreparedContext, tool_list: list[dict[str, Any]] | None,
        round_index: int, result: ChatResult, text_mode: bool,
    ) -> ChatResult:
        """A final answer written after tools ran is checked against what they returned. Names
        and figures found nowhere in the results get one new attempt; if they are still there,
        they are removed, and when too little is left the results are shown as found."""
        if result.tool_calls or not (result.content or "").strip() or turn.abort.is_set() or turn.facts is None:
            return result
        try:
            sources, strict = self._sources(turn)
            known = self._known(turn)
            # One text per result: a statement has to hold within the result it is about.
            separate = ([" ".join(str(entry.get(key) or "") for key in ("title", "source", "snippet", "passage", "url"))
                         for entry in turn.evidence_items] if turn.evidence_items and sources is turn.evidence
                        else grounding.results_in(sources)) if strict else []

            cited = " ".join(str(text or "") for text in (*sources, *known)).lower()

            def named(text: str) -> list[str]:  # "[Reuters]" after a story from Reuters is a reference, not a blank
                return [blank for blank in self._blanks(text) if blank.strip("[]{}<> ").lower() not in cited]

            def judge(text: str) -> tuple[list[Any], list[str]]:
                return (grounding.check(text, sources, known, separate) if sources else []), named(text)

            findings, blanks = judge(result.content)
            if not findings and not blanks:
                return result
            draft = result.content
            turn.inference["grounding"] = {"unsupported": grounding.missing(findings)[:8], "placeholders": blanks[:4]}
            log.info("turn %s: the answer says what no result has: %s %s", turn.conversation_id,
                     grounding.missing(findings)[:8], blanks[:4])
            if blanks and turn.facts.small_talk:  # "hi" answered with a template: there is no value to ask for
                result.content = turn.partial_text = plan.small_talk_reply(turn.message)
                turn.emit("rewrite", {"text": result.content, "reasoning": ""})
                return result
            if turn.ground_left > 0:
                turn.ground_left -= 1
                nudge = (prompts.PLACEHOLDER.format(items=", ".join(blanks[:4])) if blanks
                         else prompts.UNSUPPORTED.format(items=grounding.listed(findings)))
                self._record_usage(turn, result, prepared)
                turn.partial_text = ""
                turn.emit("rewrite", {"text": "", "reasoning": ""})
                turn.emit("status", {"phase": "thinking", "label": "Checking against the sources"})
                retry = self._stream_with_recovery(turn, model, prepared, None, round_index, text_mode=text_mode,
                                                   nudge=nudge, cool=True)
                if retry.content.strip() and not retry.tool_calls:
                    again, blanks_again = judge(retry.content)
                    if len(again) + len(blanks_again) <= len(findings) + len(blanks):
                        result, findings, blanks = retry, again, blanks_again
                if not findings and not blanks and not self._hollow(result.content, draft):
                    turn.inference["grounding"]["fixed"] = True
                    return result
            cleaned = result.content
            if blanks:
                # the lines with a placeholder go, and with them a line that only announced one
                kept = [line for line in cleaned.split("\n") if not named(line)]
                while kept and (not kept[-1].strip() or kept[-1].rstrip().endswith(":")):
                    kept.pop()
                cleaned = grounding.tidy(kept)
            if findings and strict:
                cleaned = grounding.strip(cleaned, grounding.check(cleaned, sources, known, separate)) if cleaned else ""
            if cleaned and self._hollow(cleaned, draft):
                cleaned = ""
            if not cleaned:  # turned down twice: the facts as the sources give them, or that it was not found
                cleaned = self._as_found(turn, sources) if (sources or turn.evidence) else self._plain_fallback(turn)
                turn.inference["grounding"]["shown_as_found"] = True
            if cleaned != result.content:
                result.content = cleaned
            turn.partial_text = result.content
            turn.emit("rewrite", {"text": result.content, "reasoning": ""})
        except (TurnAborted, LLMError):
            raise
        except Exception:  # a check must never break a reply
            log.warning("turn %s: source check failed", turn.conversation_id, exc_info=True)
        return result

    @staticmethod
    def _hollow(text: str, draft: str) -> bool:
        """Whether a second attempt kept the form and lost the substance: "The answer is." after a
        draft full of names says nothing, and is no more an answer than an empty reply."""
        def said(words: str) -> set[str]:
            return set(terms.content_words(words)) | set(re.findall(r"\d+", words))

        return len(said(text)) < 2 and len(said(draft)) >= 8

    def _plain_fallback(self, turn: Turn) -> str:
        """What Kiwibro says when the model's words could not be used and there are no sources to
        show: a plain reply to a greeting, what was just done, or that no answer came."""
        if turn.facts is not None and turn.facts.small_talk:
            return plan.small_talk_reply(turn.message)
        if turn.done_note:
            return turn.done_note
        return prompts.NO_ANSWER_WITH_TOOLS if turn.tool_calls_made else prompts.NO_ANSWER

    AS_FOUND_CHARS = 700

    def _as_found(self, turn: Turn, sources: list[str]) -> str:
        """What Kiwibro found, in the sources' own words: shown when no wording of the model's
        could be trusted. Never empty: with nothing to show it says that it was not found."""
        if turn.bulletin:
            return prompts.STORIES_AS_FOUND  # the bulletin itself follows
        if turn.evidence_items:
            shown = grounding.digest(turn.evidence_items)
            if shown:
                return shown
        if turn.evidence:  # a look at the computer: the tool's own summary and first lines
            newest = "\n".join(line for line in turn.evidence[-1].split("\n") if not line.startswith("[Kiwibro"))
            head = newest.strip()[: self.AS_FOUND_CHARS]
            if len(newest.strip()) > self.AS_FOUND_CHARS:
                head = head.rsplit("\n", 1)[0] if "\n" in head else head.rsplit(" ", 1)[0] + "…"
            if head:
                return f"{prompts.AS_FOUND}\n\n{head}"
        titles = grounding.headlines(sources)
        if titles:  # "summarize them": the stories by their own titles
            return f"{prompts.IN_THEIR_WORDS} " + "; ".join(titles) + "."
        return prompts.NOT_FOUND

    def _shape_reply(self, turn: Turn, result: ChatResult) -> None:
        """The final reply in the form that was asked for (shape.py): no line said twice, and one
        paragraph when the user asked for one."""
        if result.tool_calls or not (result.content or "").strip() or turn.abort.is_set():
            return
        try:
            shaped = shape.without_repeats(result.content)
            reading = getattr(turn.reading, "reading", None)
            if shape.wants_one_paragraph(turn.message, tuple(getattr(reading, "asks", None) or ())):
                shaped = shape.as_one_paragraph(shaped)
        except Exception:  # tidying up; never break a reply
            log.warning("turn %s: could not shape the reply", turn.conversation_id, exc_info=True)
            return
        if shaped != result.content:
            turn.inference["shaped"] = True
            result.content = turn.partial_text = shaped
            turn.emit("rewrite", {"text": shaped, "reasoning": ""})

    def _show_bulletin(self, turn: Turn, model: str) -> None:
        """The news the model was asked to summarise or explain, under its answer: whatever the
        model made of it, the user sees the real stories, dated and linked."""
        if not turn.bulletin or turn.abort.is_set():
            return
        as_found = bool((turn.inference.get("grounding") or {}).get("shown_as_found"))
        text = turn.bulletin if as_found else f"{prompts.NEWS_SOURCES}\n\n{turn.bulletin}"
        message = self.db.add_message(turn.conversation_id, "assistant", text, model=model, meta={"news_bulletin": True})
        turn.last_message_id = message["id"]
        turn.reply_text += "\n" + text
        turn.emit("delta", {"text": "\n\n" + text})

    def _act_instead(self, turn: Turn, result: ChatResult) -> None:
        """The user asked for something code can do exactly as asked (keep a dictated note,
        search the web) and the model replied without having done it, whatever else it
        called: Kiwibro adds the call itself, once per turn. A small model that only talks still gets the work done, and
        the answer comes from results instead of from what it happens to recall."""
        if result.tool_calls or turn.acted_instead or turn.abort.is_set():
            return
        if turn.facts is None or turn.facts.small_talk:
            return
        try:
            call = self._note_call(turn) or self._written_note_call(turn, result) or self._search_call(turn)
        except Exception:  # a convenience; never break a turn
            log.warning("turn %s: could not act for the model", turn.conversation_id, exc_info=True)
            return
        if call is None:
            return
        name, args, keep_reply, nudge = call
        if not name:  # nothing code can do, and the reply would only pretend: say what is true
            result.content = turn.partial_text = nudge
            turn.emit("rewrite", {"text": nudge, "reasoning": ""})
            return
        turn.acted_instead = name
        log.info("turn %s: the model made no call; Kiwibro calls %s %s", turn.conversation_id, name,
                 json.dumps(args, ensure_ascii=False)[:160])
        result.tool_calls = [{
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        }]
        result.finish_reason = "tool_calls"
        if not keep_reply:  # written without the work it promised or needed
            result.content = ""
            turn.partial_text = ""
            turn.emit("rewrite", {"text": "", "reasoning": ""})
        turn.inference["acted_instead"] = name
        turn.pending_review = nudge

    def _note_call(self, turn: Turn) -> tuple[str, dict[str, Any], bool, str] | None:
        """A `note` call for a note the user dictated, or for "save that as a note"."""
        if "note" not in turn.facts.tools or "note" in turn.tools_ok:
            return None
        added = re.match(r"\W*(?:please\s+)?(?:add|append|put)\s+(.+?)\s+(?:to|in|into|on)\s+(?:that|the|my|this)\s+"
                         r"(?:(.+?)\s+)?(?:note|list)\W*$", turn.message.strip(), re.I)
        if added:  # "add cheese to that note": the note meant is the named one, or the last one used
            title = (added.group(2) or "").strip() or self._last_note_title(turn)
            if not title:
                return None
            return "note", {"action": "append", "title": title, "content": added.group(1).strip()}, False, prompts.APPENDED_FOR_YOU
        found = notes_module.dictation(turn.message)
        if found is None:
            return None
        title, body = found
        if not body:  # it points at what was just said
            earlier = [m for m in self.db.recent_messages(turn.conversation_id, limit=8)
                       if m.get("id") != turn.user_message_id and m.get("role") == "assistant"
                       and str(m.get("content") or "").strip() and not (m.get("meta") or {}).get("plan_open")]
            if not earlier:
                return "", {}, True, prompts.NOTHING_TO_NOTE
            body = str(earlier[-1]["content"]).strip()
        return "note", {"action": "save", "content": body, **({"title": title} if title else {})}, False, prompts.NOTED_FOR_YOU

    def _last_note_title(self, turn: Turn) -> str:
        """The title of the note this chat last saved or read ("" if none)."""
        for message in reversed(self.db.recent_messages(turn.conversation_id, limit=30)):
            for call in message.get("tool_calls") or []:
                function = (call or {}).get("function") or {}
                if function.get("name") == "note":
                    try:
                        title = str(json.loads(function.get("arguments") or "{}").get("title") or "").strip()
                    except (TypeError, ValueError):
                        title = ""
                    if title:
                        return title
        return ""

    def _written_note_call(self, turn: Turn, result: ChatResult) -> tuple[str, dict[str, Any], bool, str] | None:
        """A `note` call saving the reply itself, when the user asked for a note to be written and
        the model wrote it without saving it."""
        if "note" not in turn.facts.tools or "note" in turn.tools_ok:
            return None
        topic = notes_module.written_request(turn.message)
        body = result.content.strip()
        if topic is None or len(body) < WRITTEN_NOTE_MIN_CHARS:
            return None
        args = {"action": "save", "content": body, **({"title": topic[:1].upper() + topic[1:]} if topic else {})}
        return "note", args, True, prompts.NOTED_FOR_YOU

    def _search_call(self, turn: Turn) -> tuple[str, dict[str, Any], bool, str] | None:
        """A `web_search` call for the search the message asks for."""
        if "web_search" not in turn.facts.tools or plan.WEB_TOOLS.intersection(turn.tools_ok):
            return None
        query, borrowed = plan.search_request(turn.message)
        if not query:
            return None
        more = turn.plan is not None and any(step.kind != "web" for step in turn.plan.remaining())
        nudge = prompts.SEARCHED_FOR_YOU.format(
            then=prompts.SEARCHED_THEN_GO_ON if more else prompts.SEARCHED_THEN_ANSWER)
        # A reply to the very question the search is for was written without looking.
        return "web_search", {"query": query}, not borrowed and turn.plan is not None, nudge

    def _plan_for(self, turn: Turn, settings: dict[str, Any]) -> plan.TurnPlan | None:
        """The request's step plan (None for chat, plain questions or when tools are off)."""
        if not settings.get("agent_enabled", True):
            return None
        try:
            built = plan.build(turn.message, self._workspace(turn.conversation_id).root)
        except Exception:  # the plan is guidance; it must never break a turn
            log.warning("plan failed", exc_info=True)
            return None
        if built is not None:
            log.info("turn %s plan: %s", turn.conversation_id, [(s.kind, s.short) for s in built.steps])
        return built

    def _close_plan(self, turn: Turn, model: str) -> None:
        """An honest closing line when the model stopped with steps still open."""
        if turn.plan is None:
            return
        turn.plan.settle(turn.reply_text)
        turn.emit("plan", {"steps": turn.plan.statuses()})
        left = turn.plan.summary_line(turn.reply_text)
        if not left:
            return
        log.info("turn %s ended with open steps: %s", turn.conversation_id, left)
        note_message = self.db.add_message(turn.conversation_id, "assistant", left, model=model, meta={"plan_open": True})
        turn.last_message_id = note_message["id"]
        turn.emit("delta", {"text": "\n\n" + left})

    @staticmethod
    def _worked(turn: Turn) -> tuple[tuple[str, str], ...]:
        """The sums in the message, done exactly in code: a small model gets them wrong with confidence."""
        try:
            return tuple(compute.worked(turn.message))
        except Exception:  # a convenience; never break a turn
            log.warning("turn %s: could not work out the sums", turn.conversation_id, exc_info=True)
            return ()

    def _retract_repeats(self, turn: Turn, final: str) -> None:
        """A small model often answers, calls one more tool, and answers again. What it wrote
        beside its tool calls is taken back when the final reply says it again: the user reads
        the answer once, and the chat's context carries it once."""
        kept: list[tuple[str, str]] = []
        for message_id, text in turn.interim:
            if not _says_again(text, final):
                kept.append((message_id, text))
                continue
            try:
                self.db.update_message(message_id, content="")
            except Exception:  # tidying up; never break a turn
                log.warning("turn %s: could not retract a repeated text", turn.conversation_id, exc_info=True)
                continue
            turn.emit("retract", {"text": text})
        turn.interim = kept

    @staticmethod
    def _unanswered_nudge(turn: Turn) -> str:
        """One push for a question the reply skipped ("who is my wife and what am I allergic to?"
        answered with the wife alone). Asked once: the questions are cleared."""
        asked, turn.questions = turn.questions, []
        if not asked or turn.abort.is_set():
            return ""
        missing = plan.unanswered(asked, turn.reply_text)
        if not missing:
            return ""
        log.info("turn %s: reply skipped %s", turn.conversation_id, missing)
        return prompts.UNANSWERED.format(questions="; ".join(f"“{question.strip(' ?')}?”" for question in missing))

    def _review_nudge(self, turn: Turn, settings: dict[str, Any]) -> str:
        """One end-of-turn self-check against the directive ledger, after real work.

        Small models often stop one step short; a single checklist round catches
        missed instructions cheaply (the prefix is KV-cached). Supervised turns
        skip it: a human is already reviewing every step."""
        if turn.reviewed or settings.get("autonomy", config.DEFAULT_AUTONOMY) == "supervised":
            return ""
        if turn.safety.get("counters", {}).get("actions", 0) < directives.REVIEW_MIN_ACTIONS:
            return ""
        turn.reviewed = True
        try:
            return directives.review_prompt(self.db, turn.conversation_id)
        except Exception:  # the ledger must never break a turn
            log.warning("review prompt failed", exc_info=True)
            return ""

    def _stream_with_recovery(
        self,
        turn: Turn,
        model: str,
        prepared: PreparedContext,
        tool_list: list[dict[str, Any]] | None,
        round_index: int = 0,
        *,
        text_mode: bool = False,
        nudge: str | None = None,
        cool: bool = False,
    ) -> ChatResult:
        settings = self._settings()
        temperature = float(settings.get("temperature", config.DEFAULT_TEMPERATURE))
        if cool:  # a corrected attempt: less chance, same prompt prefix
            temperature = min(temperature, steer.RETRY_TEMPERATURE)
        max_tokens = self._max_tokens(turn, prepared, settings)
        profile = inference.model_profile(model, prepared.model_info)
        budgets = inference.uses_budgets(profile, self.llm.direct_on())  # thinking never touches the prompt
        if turn.thinking is None:  # decided once per turn: with the server a flip re-prefills everything
            turn.thinking = inference.turn_reasoning(
                settings.get("reasoning"),
                turn.message,
                previous=self._thinking.get(turn.conversation_id),
                prompt_tokens=0 if budgets else prepared.used,
            )
            self._thinking[turn.conversation_id] = turn.thinking
        think = turn.thinking
        must_act = round_index > 0 or (turn.plan is not None and bool(turn.plan.remaining()))
        if budgets:
            extra = inference.budget_payload(profile, think=think, must_act=must_act)
        else:
            extra = inference.reasoning_payload(profile, think=think)
        prompt_extra = {k: v for k, v in extra.items() if k not in inference.BUDGET_KEYS}  # what renders
        attempts = 0
        collapsed = False  # the reply collapsed once; this request recovers it
        kept = ""  # the good start of a reply that began repeating itself (continued, not redone)
        overthought = False  # thought past the budget without acting; retried with thinking off
        correction = ""  # Kiwi Sense cut the reply off; the next attempt is told why (transient)
        if self._trait(model, steer.LOOP) >= steer.LOOP_HABIT:  # this model keeps circling: sample against it
            extra = {**extra, **inference.ANTI_REPEAT}
        while True:
            self._raise_if_aborted(turn)
            messages = toolcall.to_text_protocol(prepared.messages) if text_mode else prepared.messages
            if nudge:  # transient: steers this one request, never stored
                messages = [*messages, {"role": "user", "content": nudge}]
            request_extra = extra
            if collapsed:  # continue the good start without repeating it (also transient)
                recovery = ([{"role": "assistant", "content": kept}, {"role": "user", "content": prompts.DEGENERATE_CONTINUE}]
                            if kept else [{"role": "user", "content": prompts.DEGENERATE_NUDGE}])
                messages = [*messages, *recovery]
                request_extra = {**extra, **inference.ANTI_REPEAT}
            if correction:
                messages = [*messages, {"role": "user", "content": correction}]
            if overthought:  # act without thinking this once (also transient)
                messages = [*messages, {"role": "user", "content": prompts.ACT_NOW.format(step=self._next_step(turn))}]
                if budgets:  # the runtime ignored the budget: switch thinking off in the template too
                    request_extra = {k: v for k, v in request_extra.items() if k != "reasoning_budget_message"}
                    request_extra["thinking_budget_tokens"] = 0
                    if profile.reasoning_toggle:
                        request_extra["chat_template_kwargs"] = {"enable_thinking": False}
                else:
                    request_extra = {**request_extra, **inference.reasoning_payload(profile, think=False)}
            if budgets:  # llama.cpp ends the thought at its budget; this only catches a runtime that doesn't
                limit = 8 * max(250, int(request_extra.get("thinking_budget_tokens") or 0))
            else:
                no_thinking = request_extra.get("reasoning_effort") == "none"
                limit = inference.THINK_BUDGET_CHARS if think and must_act and not no_thinking else 0
            turn.think_chars, turn.think_budget = 0, limit
            request_extra = {**request_extra, **self._adapter_extra(turn, model, prepared)}
            reuse = inference.PREFIX_TRACKER.observe(model, messages, tool_list, prompt_extra)  # sampling is not prompt
            turn.timer = inference.RoundTimer()
            turn.stream_filter = toolcall.StreamFilter()
            turn.stream_tail, turn.loop_checked, turn.sense_checked = "", 0, 0
            try:
                result = self.llm.stream_chat(
                    model=model,
                    messages=messages,
                    tools=tool_list,
                    temperature=min(temperature, steer.RETRY_TEMPERATURE) if correction else temperature,
                    max_tokens=max_tokens,
                    on_delta=self._delta_callback(turn),
                    on_reasoning=self._reasoning_callback(turn),
                    on_rewrite=self._rewrite_callback(turn),
                    abort=turn.abort,
                    extra=request_extra or None,
                )
                turn.timer.finish()
                self._note_round(turn, round_index, think=think, reuse=reuse, result=result)
                self._rescue_tool_calls(turn, result, settings)
                if kept:  # the stored reply is what the user saw: the good start, then the rest
                    result.content = f"{kept}\n\n{result.content.lstrip()}" if result.content.strip() else kept
                return result
            except inference.Overthinking:
                turn.timer.finish()
                log.info("turn %s round %s: thought %s chars without acting; acting with thinking off",
                         turn.conversation_id, round_index, turn.think_chars)
                if overthought:  # thinking could not be switched off: end the round, the plan follows up
                    return ChatResult(content=turn.partial_text, finish_reason="overthinking")
                overthought = True
                turn.emit("status", {"phase": "thinking", "label": "Acting on it"})
            except steer.Redirect as redirect:
                # Cut off after the sentence that went wrong; the next attempt reuses the cached prefix.
                turn.timer.finish()
                trial = self._try_without_adapters(turn, model, prepared)
                self._note_steer(turn, redirect.verdict, model, trait=not trial)
                if not trial:  # the same request without its adapter comes first, uncorrected
                    self._judge_adapters(turn, spoiled=False)
                    correction = prompts.STEER_NOTE.format(note=redirect.verdict.note)
                turn.partial_text = ""
                turn.emit("rewrite", {"text": "", "reasoning": ""})
                turn.emit("status", {"phase": "thinking", "label": "Correcting course"})
            except inference.DegenerateOutput as collapse:
                self._note_trait(model, steer.LOOP)
                # Cut off (closing the stream stops the server generating). A reply that began
                # well and then repeated itself keeps its start and is continued with
                # anti-repetition sampling; a circling one is cleared and redone. Both reuse
                # the cached prefix. A second collapse ends the round with what is good.
                turn.timer.finish()
                clean = "" if collapse.circling else inference.trim_degenerate(turn.partial_text)
                log.info("turn %s round %s: output %s; cut off", turn.conversation_id, round_index,
                         "circled" if collapse.circling else "repeated itself")
                if collapsed:
                    turn.partial_text = clean
                    turn.emit("rewrite", {"text": clean, "reasoning": ""})
                    if not clean:
                        turn.partial_text = prompts.DEGENERATE_GAVE_UP
                        turn.emit("delta", {"text": prompts.DEGENERATE_GAVE_UP})
                    return ChatResult(content=turn.partial_text, finish_reason="degenerate")
                collapsed, kept = True, clean
                turn.partial_text = f"{clean}\n\n" if clean else ""
                turn.emit("rewrite", {"text": turn.partial_text, "reasoning": ""})
            except ContextLimitError:
                attempts += 1
                if attempts == 1 and self.context.recover_aggressive(turn.conversation_id, model):
                    log.info("context limit hit; summarised half the history and retrying")
                elif attempts <= 2 and self.context.hard_recover(turn.conversation_id):
                    log.info("context limit hit again; hard-trimmed history and retrying")
                else:
                    raise
                if turn.abort.is_set():
                    raise TurnAborted()
                prepared = self.context.prepare(
                    turn.conversation_id, model, turn.system_prompt, turn_context=turn.turn_context,
                    reserve_tokens=turn.reserve,
                )
                self._emit_context(turn, prepared)
                turn.partial_text = ""
                turn.partial_reasoning = ""

    @staticmethod
    def _max_tokens(turn: Turn, prepared: PreparedContext, settings: dict[str, Any]) -> int:
        """The reply cap for one request. While a document still has to be written, the
        cap is raised (within the window's free room) so thinking plus the file's text fit
        in one tool call; a call cut off mid-content cannot run and costs a whole round."""
        cap = int(settings.get("max_output_tokens", config.DEFAULT_MAX_OUTPUT_TOKENS))
        if turn.plan is None or not any(step.kind == "write" for step in turn.plan.remaining()):
            return cap
        room = prepared.context_length - prepared.used - turn.reserve - 256
        return max(cap, min(WRITE_ROUND_MAX_TOKENS, room))

    @staticmethod
    def _note_round(
        turn: Turn, round_index: int, *, think: bool, reuse: dict[str, int], result: ChatResult
    ) -> None:
        """Fold one round's inference telemetry into the turn's ``done.inference``."""
        timing = turn.timer.stats((result.usage or {}).get("completion_tokens")) if turn.timer else {}
        counted = result.timings or {}
        reuse = inference.server_reuse(counted) or reuse  # the direct engine counts exactly
        if counted.get("predicted_per_second") and counted.get("predicted_n", 0) >= 8:
            timing["tokens_per_s"] = round(counted["predicted_per_second"], 1)
        if round_index == 0:
            turn.inference.update(reuse, thinking=think)
            if "ttft_ms" in timing:
                turn.inference["ttft_ms"] = timing["ttft_ms"]
        if "tokens_per_s" in timing:
            turn.inference["tokens_per_s"] = timing["tokens_per_s"]
        if turn.model:  # measured speeds size background-call timeouts (memory extraction)
            exact = "prompt_ms" in counted
            inference.SPEEDS.note(
                turn.model,
                new_tokens=int(reuse.get("prompt_tokens_est", 0)) - int(reuse.get("reused_tokens_est", 0)),
                ttft_ms=round(counted["prompt_ms"]) if exact else timing.get("ttft_ms"),
                tokens_per_s=timing.get("tokens_per_s"),
            )
        log.info(
            "turn %s round %s: think=%s reuse=%s%% ttft=%sms %s tok/s",
            turn.conversation_id, round_index, think, reuse.get("reuse_percent"),
            timing.get("ttft_ms"), timing.get("tokens_per_s"),
        )

    @staticmethod
    def _rescue_tool_calls(turn: Turn, result: ChatResult, settings: dict[str, Any]) -> None:
        """Recover tool calls the model wrote as text; release stream text held back for nothing."""
        held_back, turn.stream_filter = turn.stream_filter, None
        if not result.tool_calls and settings.get("agent_enabled", True):
            calls, cleaned = toolcall.extract_calls(result.content, tools.TOOL_SPECS)
            if calls:
                log.info("turn %s: recovered %s tool call(s) written as text", turn.conversation_id, len(calls))
                result.tool_calls, result.content = calls, cleaned
        written = result.content
        result.content = toolcall.unwrap(written)  # "<output>400</output>" is the answer 400
        if held_back is not None:
            release = held_back.flush() + ("" if result.tool_calls else held_back.held)
            if held_back.opener and result.content != written:
                release = release.replace(held_back.closer, "").rstrip()
            elif held_back.opener and not result.tool_calls:
                # It only began like a wrapper ("<output> is an HTML element…"): show the tag after all.
                turn.partial_text = held_back.opener + turn.partial_text
                turn.emit("rewrite", {"text": turn.partial_text, "reasoning": ""})
            if release:
                turn.partial_text += release
                turn.emit("delta", {"text": release})

    # --- event helpers -----------------------------------------------------

    def _delta_callback(self, turn: Turn) -> Callable[[str], None]:
        def on_delta(text: str) -> None:
            if turn.abort.is_set():
                raise TurnAborted()
            if turn.timer is not None:
                turn.timer.mark_first()
            self._guard_stream(turn, text)
            # Tool-call markup is held back so users never see raw call syntax.
            visible = turn.stream_filter.feed(text) if turn.stream_filter is not None else text
            if visible:
                turn.partial_text += visible
                turn.emit("delta", {"text": visible})
                if turn.steer_left > 0 and turn.facts is not None and any(mark in visible for mark in ".!?\n"):
                    self._read_opening(turn)

        return on_delta

    @staticmethod
    def _read_opening(turn: Turn) -> None:
        """Kiwi Sense reads each finished sentence of the reply's opening; a wrong one stops the stream."""
        if len(turn.partial_text) > steer.EARLY_CHARS:
            return
        try:
            facts = replace(turn.facts, tools_ran=len(turn.tools_ok))
            verdict, turn.sense_checked = steer.early(turn.partial_text, facts, turn.sense_checked)
        except Exception:  # reading must never break a reply
            log.warning("turn %s: sense check failed", turn.conversation_id, exc_info=True)
            turn.steer_left = 0
            return
        if verdict is not None:
            turn.steer_left -= 1
            raise steer.Redirect(verdict)

    @staticmethod
    def _rewrite_callback(turn: Turn) -> Callable[[str, str], None]:
        """The model wrote reasoning as plain text: show it as reasoning, keep the answer."""

        def on_rewrite(text: str, reasoning: str) -> None:
            turn.partial_text = text
            turn.partial_reasoning += reasoning
            turn.emit("rewrite", {"text": text, "reasoning": reasoning})

        return on_rewrite

    @staticmethod
    def _guard_stream(turn: Turn, text: str) -> None:
        """Raise DegenerateOutput when the reply collapses into repetition or circles."""
        turn.stream_tail = (turn.stream_tail + text)[-3000:]
        turn.loop_checked += len(text)
        if inference.is_degenerate(turn.stream_tail):
            raise inference.DegenerateOutput()
        if turn.loop_checked >= 250:  # the phrase-recurrence check is costlier: run it in steps
            turn.loop_checked = 0
            if inference.is_looping(turn.stream_tail):
                raise inference.DegenerateOutput(circling=True)

    def _reasoning_callback(self, turn: Turn) -> Callable[[str], None]:
        def on_reasoning(text: str) -> None:
            if turn.abort.is_set():
                raise TurnAborted()
            if turn.timer is not None:
                turn.timer.mark_first()
            self._guard_stream(turn, text)  # thinking loops burn the budget too
            turn.think_chars += len(text)
            if turn.think_budget and turn.think_chars > turn.think_budget and not turn.partial_text.strip():
                raise inference.Overthinking()
            turn.partial_reasoning += text
            turn.emit("reasoning", {"text": text})

        return on_reasoning

    def _emit_context(self, turn: Turn, prepared: PreparedContext) -> None:
        if prepared.compacted or prepared.trimmed:
            turn.emit(
                "context",
                {
                    "used": prepared.used,
                    "budget": prepared.budget,
                    "percent": prepared.percent,
                    "compacted": True,
                },
            )

    def _emit_meta(
        self,
        turn: Turn,
        conversation: dict[str, Any],
        settings: dict[str, Any],
        *,
        model: str | None,
        prepared: PreparedContext | None = None,
    ) -> None:
        if prepared is not None:
            stats = prepared.stats
            title = conversation["title"]
        else:
            ctx = self.context.inventory(model)[0] if model else 32768
            budget = compute_budget(
                ctx, int(settings.get("max_output_tokens", config.DEFAULT_MAX_OUTPUT_TOKENS))
            )
            stats = {"used": 0, "budget": budget, "percent": 0}
            title = conversation["title"]
        turn.emit(
            "meta",
            {
                "conversation_id": turn.conversation_id,
                "user_message_id": turn.user_message_id,
                "model": model,
                "title": title,
                "context": stats,
            },
        )

    def _see_attachments(self, turn: Turn, model: str, user_message: dict[str, Any]) -> None:
        """Describe the user's attachments once, before the main model runs.

        Images and videos are looked at by the main model when it can see, else by
        the smallest installed vision model (loaded for this and unloaded after);
        audio and SVG get facts measured by code; documents keep just their path
        for the file and document tools. Stored in the message's meta, the result
        renders identically on every later turn.
        """
        workspace = self._workspace(turn.conversation_id)
        seen = []
        for item in turn.attachments:
            self._raise_if_aborted(turn)
            if item.get("kind") in ("image", "video", "audio", "svg") and "seen" not in item:
                turn.emit("status", {"phase": "seeing", "label": f"Looking at {item['name']}"})
                try:
                    item = {**item, "seen": delegate.see_attachment(
                        self.llm, workspace, item, question=turn.message, current_model=model
                    )}
                except (SandboxError, OSError) as exc:  # moved or unreadable since upload
                    log.warning("attachment %s unreadable: %s", item.get("path"), exc)
            seen.append(item)
        turn.attachments = seen
        meta = {**(user_message.get("meta") or {}), "attachments": seen}
        self.db.update_message(user_message["id"], meta=meta)

    def warm_prefix(
        self, model: str, conversation_id: str | None = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, dict[str, Any]] | None:
        """The leading messages, tools and server knobs of the next request to ``model``,
        rendered exactly as that turn will render them (readiness prefills them).

        Without a chat this is message 0 plus the tools, which every new chat shares. A
        chat's history is rendered as if a new user message followed it (older tool
        results already compact). None when the next turn compacts first anyway or the
        chat is too long to prefill ahead of time."""
        from . import readiness  # local: readiness imports routing, which imports this package's tools

        settings = self._settings()
        model_info = self.llm.model_info(model)
        system_prompt = self._system_prompt(
            {**settings, "workspace": self._folder(conversation_id)}, model=model, model_info=model_info
        )
        summary, history = "", []
        if conversation_id:
            conversation = self.db.get_conversation(conversation_id)
            if conversation is None:
                return None
            summary = conversation["summary"]
            history = self.db.list_messages(conversation_id, after_seq=conversation["summary_through_seq"])
        messages = build_llm_messages(
            combine_system_prompt(system_prompt, summary), [*history, {"role": "user", "content": "…"}]
        )
        if history:
            _ctx, budget, _info = self.context.budget_for(model)
            used = self.context.used_tokens(messages, model)
            if used > int(budget * COMPACT_TRIGGER) or used > readiness.warm_limit(model):
                return None
        info = model_info or {}
        tool_list = self._tool_list(model, settings, model_info)
        if settings.get("agent_enabled", True) and not info.get("tools", False):
            messages = toolcall.to_text_protocol(messages)
        profile = inference.model_profile(model, info)
        if inference.uses_budgets(profile, self.llm.direct_on()):  # every turn renders thinking on
            return messages, tool_list, dict(inference.TEMPLATE_THINKING) if profile.reasoning_toggle else {}
        previous = self._thinking.get(conversation_id) if conversation_id else None
        think = previous if previous is not None else inference.wants_reasoning(settings.get("reasoning"), "")
        extra = inference.reasoning_payload(profile, think=think)
        return messages, tool_list, extra

    def _tool_list(
        self, model: str, settings: dict[str, Any], model_info: dict[str, Any] | None
    ) -> list[dict[str, Any]] | None:
        """Native tool definitions for this model, sized to its window.

        The auto set is ~3.5k tokens. In a window where it would crowd out the
        conversation (a model that cannot be reloaded larger), the minimal set is
        sent instead; others still run when called by name."""
        info = model_info or {}
        if not settings.get("agent_enabled", True) or not info.get("tools", False):
            return None
        _ctx, budget, _info = self.context.budget_for(model)
        room = int(budget * TOOLS_SHARE_MAX)
        tool_list = tools.tool_defs_for(settings=settings)
        if estimate_tokens(json.dumps(tool_list, ensure_ascii=False)) <= room:
            return tool_list
        small = tools.tool_defs_for("minimal", settings=settings)
        log.info("model %s: window too small for the full tool set; using the minimal set", model)
        return small if estimate_tokens(json.dumps(small, ensure_ascii=False)) <= room else None

    def _page_long_message(self, turn: Turn, user_message: dict[str, Any], model: str) -> None:
        """A message too long for the model's window goes into the document library; the
        model reads its start, its end and how to read the rest (the chat shows it whole)."""
        _ctx, budget, _info = self.context.budget_for(model)
        limit = max(4000, int((budget - turn.reserve) * USER_MESSAGE_SHARE * 3.6))
        text = turn.message
        if len(text) <= limit or (user_message.get("meta") or {}).get("paged"):
            return
        try:
            stored = self._library().stash_text(text, title=derive_title(text) or "Long message", source="chat")
        except Exception:  # paging is an optimisation; the full message still goes out
            log.warning("could not store a long message", exc_info=True)
            return
        current = self.db.get_message(user_message["id"]) or user_message  # attachments may have added meta
        meta = {**(current.get("meta") or {}), "prompt_text": library.excerpt(text, stored, limit),
                "paged": {"handle": stored["handle"], "chars": len(text)}}
        self.db.update_message(user_message["id"], meta=meta)
        log.info("turn %s: %s-char message stored as %s", turn.conversation_id, len(text), stored["handle"])

    def _capture_memory(self, turn: Turn, settings: dict[str, Any]) -> dict[str, int]:
        """Facts the user just stated, saved by code before the reply (memory.capture)."""
        if not settings.get("memory_enabled", True):
            return {"added": 0, "updated": 0, "removed": 0}
        if turn.reading is not None and readings.keep_nothing(turn.reading):
            log.info("turn %s: nothing about the user to keep (%s)", turn.conversation_id, turn.reading.reading.keep)
            return {"added": 0, "updated": 0, "removed": 0}
        try:
            counts = self.memory.capture(turn.message, conversation_id=turn.conversation_id)
            if counts.get("added") or counts.get("updated"):
                turn.learned = tuple(fact.content for fact in autofacts.extract(turn.message) if not fact.forget)
            return counts
        except Exception:  # memory must never break a turn
            log.warning("turn %s: memory capture failed", turn.conversation_id, exc_info=True)
            return {"added": 0, "updated": 0, "removed": 0}

    def _turn_context(self, turn: Turn, settings: dict[str, Any]) -> str:
        """Volatile per-turn note: clock, retrieved memories, directive ledger (never message 0)."""
        reading = turn.reading or readings.TurnReading()
        # "and in Rust?" finds what the question before it was about
        lookup = f"{turn.before[:300]} {turn.message}" if readings.continues(reading) and turn.before else turn.message
        memory_block, turn.recalled = self.memory.build_block(
            lookup,
            enabled=bool(settings.get("memory_enabled", True)),
            # Budget-scaled cap: a fixed 800-token block would starve the
            # working window on small-context models.
            max_tokens=self.context.memory_cap_for(turn.model) if turn.model else 800,
        )
        plan_note = turn.plan.note() if turn.plan is not None else ""
        steps = turn.plan.steps if turn.plan is not None else []
        guidance_note = guidance.note(guidance.select(
            turn.message,
            tools=bool(settings.get("agent_enabled", True)),
            attachments=bool(turn.attachments),
            plan_kinds=[step.kind for step in steps],
            plan_steps=len(steps),
        ))
        learned = prompts.JUST_LEARNED.format(facts="\n".join(f"- {fact}" for fact in turn.learned)) if turn.learned else ""
        worked = prompts.WORKED_OUT.format(
            lines="\n".join(f"- {asked} = {result}" for asked, result in turn.worked)
        ) if turn.worked else ""
        several = prompts.SEVERAL_QUESTIONS.format(
            count=len(turn.questions), questions="\n".join(f"{i}. {q}" for i, q in enumerate(turn.questions, 1)),
        ) if turn.questions else ""
        blocks = [
            inference.time_line(), memory_block, self._notes_block(turn), self._earlier_chats_note(turn),
            _directives_note(self.db, turn.conversation_id), plan_note, several, worked, readings.note(reading),
            guidance_note, learned, turn.reaction.note() if turn.reaction is not None else "",
            self._standing_reminders(turn, settings, memory_block),
        ]
        body = "\n\n".join(block.strip() for block in blocks if block and block.strip())
        return f"{prompts.TURN_CONTEXT_HEADER}\n{body}"

    def _notes_block(self, turn: Turn) -> str:
        """The user's notes that bear on this message (never in a private chat, never for small talk)."""
        if self.notes is None or plan.is_small_talk(turn.message):
            return ""
        try:
            if self.memory.is_private(turn.conversation_id):
                return ""
            reading = turn.reading or readings.TurnReading()
            lookup = f"{turn.before[:300]} {turn.message}" if readings.continues(reading) and turn.before else turn.message
            found = self.notes.relevant(lookup, limit=2)
            text = turn.message.strip()
            if not found and plan._OWN.search(text) and (text.endswith("?") or plan._QUESTION.match(text)):
                # "what did I ask you to buy?" shares no word with "Groceries: milk, eggs": a question about
                # the user's own things is shown the note this chat last kept or read.
                title = self._last_note_title(turn)
                last = self.notes.find(title) if title else None
                found = [last] if last else []
            return f"{NOTES_HEADER}\n{Notebook.render_block(found, max_chars=600)}" if found else ""
        except Exception:  # notes must never break a turn
            log.warning("turn %s: notes lookup failed", turn.conversation_id, exc_info=True)
            return ""

    # --- Kiwi Sense: facts of the turn, corrections, what a model tends to get wrong ---

    def _user_names(self, settings: dict[str, Any], memory_block: str = "") -> tuple[str, ...]:
        names = [str(settings.get("user_name") or "").strip()]
        names += re.findall(r"(?:[Uu]ser(?:'s|’s)? name is|[Uu]ser is called|The user is)\s+([A-Z][\w'’-]+)", memory_block or "")
        return tuple(dict.fromkeys(name for name in names if name))

    def _facts(
        self, turn: Turn, settings: dict[str, Any], tool_list: list[dict[str, Any]] | None,
        model_info: dict[str, Any] | None, captured: dict[str, int], prepared: PreparedContext,
    ) -> steer.Facts | None:
        """What is true this turn, for judging the reply (None switches the checks off)."""
        if not settings.get("sense_enabled", True):
            return None
        if tool_list:
            names = {str((tool.get("function") or {}).get("name") or "") for tool in tool_list}
        elif settings.get("agent_enabled", True) and not (model_info or {}).get("tools"):
            names = {spec.name for spec in _text_mode_specs(settings)}
        else:
            names = set()
        recent: list[str] = []
        asked: list[str] = []
        for message in prepared.messages[1:-1][-6:]:
            content = message.get("content")
            if message.get("role") in ("user", "assistant") and isinstance(content, str) and content.strip():
                text = content.split(prompts.TURN_CONTEXT_HEADER, 1)[0].strip()[:1500]
                recent.append(text)
                if message.get("role") == "user":
                    asked.append(text)
        return steer.Facts(
            question=turn.message,
            user_names=self._user_names(settings, turn.turn_context),
            tools=frozenset(name for name in names if name),
            memory_on=bool(settings.get("memory_enabled", True)),
            memory_saved=any(captured.values()),
            recent=tuple(recent[-4:]),
            asked_before=tuple(asked),
            small_talk=plan.is_small_talk(turn.message),
            model=turn.model or "",
            note=turn.turn_context or "",
            learned=turn.learned,
            told_only=bool(turn.learned) and autofacts.stated_only(turn.message),
            worked=turn.worked,
        )

    def _trait(self, model: str | None, kind: str) -> int:
        """How often ``model`` needed this correction in the last weeks."""
        if not model:
            return 0
        try:
            return self._traits(model).get(kind, 0)
        except Exception:  # traits are advice; never break a turn
            return 0

    def _traits(self, model: str) -> dict[str, int]:
        since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - steer.TRAIT_DAYS * 86400))
        return self.db.traits(model, since=since)

    def _note_trait(self, model: str | None, kind: str) -> None:
        if not model:
            return
        try:
            self.db.note_trait(model, kind)
        except Exception:
            log.warning("could not record trait %s for %s", kind, model, exc_info=True)

    def _note_steer(self, turn: Turn, verdict: steer.Verdict, model: str, trait: bool = True) -> None:
        log.info("turn %s: sense verdict %s (%.2f) on %r", turn.conversation_id, verdict.kind, verdict.score,
                 verdict.sentence[:120])
        if verdict.kind not in turn.steered:
            turn.steered.append(verdict.kind)
        turn.inference["steered"] = list(turn.steered)
        if trait:  # not yet the model's habit while an adapter may be the cause
            self._note_trait(model, verdict.kind)

    # --- task adapters that adjust themselves (adapters.py) ---

    def _adapter_extra(self, turn: Turn, model: str, prepared: PreparedContext) -> dict[str, Any]:
        """How this turn's requests use the task adapters: one task for the whole turn, read
        from the user's own messages (Kiwibro's passing notes must not flip the adapter and
        with it the server's cache), and none at all once one of them is being tested."""
        if turn.adapters_tried:
            return {ADAPTERS_OFF: True}
        if turn.adapter_task is None:
            if not self.llm.has_adapters(model):
                return {}  # none loaded with this model (or it is still starting): nothing to pin
            turn.adapter_task = adapters.task_of(prepared.messages)
        return {ADAPTER_TASK: turn.adapter_task}

    def _try_without_adapters(self, turn: Turn, model: str, prepared: PreparedContext) -> bool:
        """A reply went wrong with a task adapter on: the same request runs once more without
        it, uncorrected, to see whether the adapter was the cause. False when none was on."""
        if turn.adapters_tried or turn.adapters_judged:
            return False
        turn.adapters_tried = self.llm.adapters_on(model, prepared.messages, turn.adapter_task)
        if not turn.adapters_tried:
            turn.adapters_judged = True
            return False
        turn.steer_left += 1  # a test, not one of the turn's corrections
        log.info("turn %s: trying without adapter %s", turn.conversation_id,
                 ", ".join(a.name for a in turn.adapters_tried))
        return True

    def _judge_adapters(self, turn: Turn, spoiled: bool) -> None:
        """Settle the test: right without the adapter counts against it (after a few it runs
        at half scale, then not at all); wrong either way is the model's own doing."""
        if not turn.adapters_tried or turn.adapters_judged:
            return
        turn.adapters_judged = True
        if not spoiled:
            return
        try:
            counts = self.llm.blame_adapters(turn.adapters_tried)
        except Exception:
            log.warning("turn %s: could not record adapter trouble", turn.conversation_id, exc_info=True)
            return
        turn.inference["adapters_turned_down"] = [
            {"name": name, "trouble": count, "strength": adapters.strength(count)} for name, count in counts.items()]
        log.info("turn %s: the reply was right without adapter %s", turn.conversation_id,
                 ", ".join(f"{name} ({count})" for name, count in counts.items()))

    def _standing_reminders(self, turn: Turn, settings: dict[str, Any], memory_block: str) -> str:
        """Reminders for a model that needed correcting before (volatile note, never message 0)."""
        if not turn.model or not settings.get("sense_enabled", True):
            return ""
        try:
            traits = self._traits(turn.model)
        except Exception:
            return ""
        if not traits:
            return ""
        tools_on = bool(settings.get("agent_enabled", True))
        facts = steer.Facts(user_names=self._user_names(settings, memory_block),
                            tools=frozenset({"web_search"}) if tools_on else frozenset())
        return "\n".join(f"- {line}" for line in steer.reminders(traits, facts))

    def _steer_reply(
        self, turn: Turn, model: str, prepared: PreparedContext, tool_list: list[dict[str, Any]] | None,
        round_index: int, result: ChatResult, text_mode: bool,
    ) -> ChatResult:
        """Check a finished reply against the turn's facts. One corrected attempt; if that
        fails the same way, the sentences that are wrong are removed."""
        if turn.facts is None or not (result.content or "").strip() or turn.abort.is_set():
            return result
        try:
            facts = replace(turn.facts, tools_ran=len(turn.tools_ok))
            verdict = steer.review(result.content, facts, final=not result.tool_calls)
        except Exception:  # reading must never break a reply
            log.warning("turn %s: sense review failed", turn.conversation_id, exc_info=True)
            return result
        if verdict is None:
            self._judge_adapters(turn, spoiled=True)  # a reply redone without its adapter came out right
            return result
        if verdict.kind == steer.UNKEPT_PROMISE and self._plan_will_chase(turn, result):
            return result  # the plan's own follow-up names the open steps, which is the better push
        if verdict.rewrite:  # code can put it right exactly: no second attempt, no wait
            self._note_steer(turn, verdict, model)
        while turn.steer_left > 0 and not verdict.rewrite:
            turn.steer_left -= 1
            # With a task adapter on, the first new attempt only drops the adapter; the
            # corrected attempt follows if that was not it.
            trial = self._try_without_adapters(turn, model, prepared)
            self._note_steer(turn, verdict, model, trait=not trial)
            if not trial:
                self._judge_adapters(turn, spoiled=False)
            self._record_usage(turn, result, prepared)
            turn.partial_text = ""
            turn.emit("rewrite", {"text": "", "reasoning": ""})
            turn.emit("status", {"phase": "thinking", "label": "Correcting course"})
            retry = self._stream_with_recovery(
                turn, model, prepared, tool_list, round_index, text_mode=text_mode,
                nudge=None if trial else prompts.STEER_NOTE.format(note=verdict.note), cool=not trial,
            )
            if retry.tool_calls or retry.content.strip():
                again = steer.review(retry.content, facts, final=not retry.tool_calls) if retry.content.strip() else None
                if again is None:
                    self._judge_adapters(turn, spoiled=True)
                if again is None or again.kind != verdict.kind:
                    return retry
                result, verdict = retry, again
            else:  # the new attempt came back empty: keep the one before, repaired below
                turn.partial_text = result.content
                turn.emit("rewrite", {"text": result.content, "reasoning": ""})
        cleaned = steer.repair(result.content, verdict, facts)
        if not cleaned and verdict.kind in (steer.OTHER_IDENTITY, steer.ADOPTED_USER):
            # Nothing true was left. What stands in is true: who Kiwibro is, or that the fact is saved.
            cleaned = prompts.REMEMBERED_LINE if verdict.kind == steer.ADOPTED_USER and facts.learned else prompts.IDENTITY_LINE
        if cleaned != result.content and (cleaned or verdict.kind == steer.META_LEAK):  # a copied note is never shown
            log.info("turn %s: removed what %s", turn.conversation_id, steer.LABELS.get(verdict.kind, verdict.kind))
            result.content = cleaned
            turn.partial_text = cleaned
            turn.emit("rewrite", {"text": cleaned, "reasoning": ""})
        return result

    @staticmethod
    def _plan_will_chase(turn: Turn, result: ChatResult) -> bool:
        """Whether the turn's plan will send its own follow-up for work left undone."""
        current = turn.plan
        if current is None or result.tool_calls or current.followups >= plan.MAX_FOLLOWUPS:
            return False
        try:
            return any(step.kind != "answer" for step, _why in current.unmet(f"{turn.reply_text}\n{result.content}"))
        except Exception:
            return False

    def _answer_anyway(
        self, turn: Turn, model: str, prepared: PreparedContext, tool_list: list[dict[str, Any]] | None,
        text_mode: bool,
    ) -> None:
        """The turn produced no words at all: ask once more for a plain answer, and if the
        model still says nothing, say so honestly instead of leaving an empty reply."""
        text = ""
        try:
            prepared = self.context.extend(
                prepared, turn.conversation_id, model, turn.system_prompt, turn_context=turn.turn_context
            )
            turn.emit("status", {"phase": "thinking", "label": "Putting the answer together"})
            result = self._stream_with_recovery(
                turn, model, prepared, tool_list, 1, text_mode=text_mode, nudge=prompts.ANSWER_NOW, cool=True
            )
            self._record_usage(turn, result, prepared)
            text = (result.content or "").strip()
        except (TurnAborted, LLMError):
            raise
        except Exception:
            log.warning("turn %s: could not force an answer", turn.conversation_id, exc_info=True)
        meta = None
        if not text:
            text = self._plain_fallback(turn)
            meta = {"fallback": True}
            turn.emit("rewrite", {"text": "", "reasoning": ""})
            turn.emit("delta", {"text": text})
        message = self.db.add_message(turn.conversation_id, "assistant", text, model=model, meta=meta)
        turn.last_message_id = message["id"]
        turn.reply_text = text

    def _earlier_chats_note(self, turn: Turn) -> str:
        """Other recent chats, only when the message refers back to one (sessions.py)."""
        if not sessions.refers_back(turn.message):
            return ""
        budget = self.context.budget_for(turn.model)[1] if turn.model else 4096
        return sessions.recent_note(
            self.db, turn.conversation_id, turn.message, max_tokens=min(sessions.NOTE_MAX_TOKENS, budget // 10)
        )

    @staticmethod
    def _system_prompt(
        settings: dict[str, Any],
        *,
        model: str | None = None,
        model_info: dict[str, Any] | None = None,
    ) -> str:
        """Message 0: everything here must be stable across turns (KV-cache prefix)."""
        name = str(settings.get("user_name") or "").strip()
        user_block = prompts.USER_BLOCK.format(name=name) if name else ""
        model_block = prompts.MODEL_BLOCK.format(model=model) if model else ""
        workspace = str(settings.get("workspace") or "").strip()
        workspace_block = prompts.WORKSPACE_BLOCK.format(workspace=config.display_path(workspace)) if workspace else ""
        if not settings.get("agent_enabled", True):
            tools_block = prompts.TOOLS_DISABLED_BLOCK
        elif model_info is not None and not model_info.get("tools"):
            tools_block = toolcall.protocol_block(_text_mode_specs(settings))
        else:
            manual = settings.get("autonomy", config.DEFAULT_AUTONOMY) == "supervised"
            permissions_line = prompts.PERMISSIONS_MANUAL if manual else prompts.PERMISSIONS_AUTOMATIC
            tools_block = prompts.TOOLS_BLOCK.replace("{permissions_line}", permissions_line) + tools.more_tools_line(settings)
        return prompts.SYSTEM_PROMPT.format(
            model_block=model_block,
            workspace_block=workspace_block,
            tools_block=tools_block,
            user_block=user_block,
        )

    def _record_usage(self, turn: Turn, result: ChatResult, prepared: PreparedContext | None = None) -> None:
        usage = result.usage
        if not usage:
            return
        if prepared is not None and turn.model:
            self.context.note_usage(turn.model, prepared.raw_sum, int(usage.get("prompt_tokens") or 0))
        turn.has_usage = True
        turn.usage_prompt += int(usage.get("prompt_tokens") or 0)
        turn.usage_completion += int(usage.get("completion_tokens") or 0)
        turn.usage_total += int(usage.get("total_tokens") or 0)

    @staticmethod
    def _usage_payload(turn: Turn) -> dict[str, int] | None:
        if not turn.has_usage:
            return None
        return {
            "prompt_tokens": turn.usage_prompt,
            "completion_tokens": turn.usage_completion,
            "total_tokens": turn.usage_total or (turn.usage_prompt + turn.usage_completion),
        }

    def _insert_usage(self, turn: Turn, model: str) -> None:
        if not turn.has_usage:
            return
        try:
            self.db.add_usage(
                turn.conversation_id, model, turn.usage_prompt, turn.usage_completion
            )
        except Exception:  # pragma: no cover - usage logging must not fail a turn
            log.warning("could not record usage", exc_info=True)

    # --- tools & approvals -------------------------------------------------

    def _handle_tool_call(self, turn: Turn, call: dict[str, Any], settings: dict[str, Any]) -> None:
        """Single-call convenience wrapper (see _handle_tool_calls)."""
        self._handle_tool_calls(turn, [call], settings)

    def _handle_tool_calls(
        self, turn: Turn, calls: list[dict[str, Any]], settings: dict[str, Any]
    ) -> None:
        """Validate, approve and run one round of tool calls.

        Approvals are always sequential. Independent read-only/web calls then
        run concurrently; writes/commands run one by one. Results are recorded
        in the original call order, and an identical call repeated 3+ times in
        a turn is refused (loop guard) instead of re-executed.
        """
        prepared_calls: list[dict[str, Any]] = []
        for index, call in enumerate(calls):
            self._raise_if_aborted(turn)
            # max_tokens can only cut off the last call of a reply.
            cut_off = turn.truncated_calls and index == len(calls) - 1
            prepared_calls.append(self._prepare_tool_call(turn, call, settings, cut_off=cut_off))
            self._raise_if_aborted(turn)
        runnable = [item for item in prepared_calls if item["result"] is None]
        searched = any(item["name"] in plan.SEARCH_TOOLS for item in runnable)
        turn.search_rounds += searched
        outcomes: dict[str, tuple[ToolResult, int]] = {}

        def run_one(item: dict[str, Any]) -> tuple[str, ToolResult, int]:
            started = time.monotonic()
            name, args = item["name"], item["args"]
            context = ToolContext(
                workspace=item["workspace"] or self._workspace(turn.conversation_id),
                memory=self.memory,
                search=self.search,
                conversation_id=turn.conversation_id,
                llm=self.llm,
                current_model=turn.model,
                notes=self.notes,
                shown_before=turn.shown_before,
                must_mention=turn.must_mention,
                deep=turn.own_search,
            )
            if tools.TOOL_SPECS[name].risk == "write":  # every file change stays undoable
                checkpoints.snapshot_call(self.data_dir, context.workspace, name, args, turn.conversation_id)
            elif item["decision"] is not None and item["decision"].targets:  # and what a command changes or removes
                checkpoints.snapshot_command(self.data_dir, context.workspace, name, item["decision"].targets,
                                             turn.conversation_id)
            with safety.applied(item["decision"], approved_by_user=item["approved"]):
                result = tools.execute_tool(name, guard.reveal(name, args, turn.safety), context)
            result = guard.postprocess(name, args, result, turn.safety)
            result = library.page_output(self._library(), name, result, turn.page_limit)
            return item["id"], result, int((time.monotonic() - started) * 1000)

        parallel_items = [item for item in runnable if tools.is_parallel_safe(item["name"])]
        serial_items = [item for item in runnable if not tools.is_parallel_safe(item["name"])]
        if len(parallel_items) > 1:
            phase, label = tools.phase_for(parallel_items[0]["name"])
            turn.emit("status", {"phase": phase, "label": label})
            workers = max(1, min(int(getattr(config, "MAX_PARALLEL_TOOLS", 4) or 4), len(parallel_items)))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kiwibro-tool") as pool:
                for call_id, result, duration in pool.map(run_one, parallel_items):
                    outcomes[call_id] = (result, duration)
        else:
            serial_items = parallel_items + serial_items
        for item in serial_items:
            self._raise_if_aborted(turn)
            phase, label = tools.phase_for(item["name"])
            turn.emit("status", {"phase": phase, "label": label})
            call_id, result, duration = run_one(item)
            outcomes[call_id] = (result, duration)
            self._raise_if_aborted(turn)
        finished: list[tuple[dict[str, Any], ToolResult, int]] = []
        for item in prepared_calls:
            if item["result"] is not None:
                result, duration = item["result"], 0
            else:
                result, duration = outcomes[item["id"]]
            turn.tool_calls_made += 1
            if result.ok:
                turn.tools_ok.append(item["name"])
                if item["name"] == "note" and item["args"].get("action") in ("save", "append"):
                    added = str(item["args"].get("content") or "").strip() if item["args"].get("action") == "append" else ""
                    turn.done_note = (f"{result.content.rstrip('.')}: {added}." if added and len(added) <= 200
                                      else result.content)
                if item["name"] in ("system_info", "list_files", "git") and isinstance(result.meta, dict):
                    turn.looked[item["name"]] = {**result.meta, "shown": item["location"] or "",
                                                 "depth": LIST_MAX_DEPTH}
                if item["name"] not in ("note", "memory") and (result.meta or {}).get("count") != 0:  # found something
                    turn.evidence.append(result.content)
                    found = (result.meta or {}).get("results")
                    if isinstance(found, list):
                        turn.evidence_items.extend(entry for entry in found if isinstance(entry, dict))
            if item["notes"]:  # show small models how their call was corrected
                result = replace(result, content=f"{result.content}\n[Kiwibro fixed this call: {'; '.join(item['notes'][:4])}]")
            if item["location"] and result.ok:  # ran in a folder the user named, outside the workspace
                result = replace(result, content=f"{result.content}\n[Location: {item['location']}]")
            if item["name"] in ("web_search", "multi_search") and result.ok:
                result = replace(result, content=research.drop_seen_snippets(result.content, turn.seen_urls))
            turn.failure_streak = 0 if result.ok else turn.failure_streak + 1
            if turn.failure_streak == FAILURE_COACH_AT:
                result = replace(result, content=f"{result.content}\n{prompts.FAILURE_COACH.format(count=FAILURE_COACH_AT)}")
            finished.append((item, result, duration))
        # What is next rides on the round's last result: stored with it, so the next
        # request's prefix matches what the server already has cached.
        progress = self._advance_plan(turn, finished)
        if searched and turn.search_rounds == plan.SEARCH_ROUNDS_SOFT:
            progress = "\n".join(filter(None, (progress, prompts.SEARCH_ENOUGH.format(count=turn.search_rounds))))
        for position, (item, result, duration) in enumerate(finished):
            if progress and position == len(finished) - 1:
                result = replace(result, content=f"{result.content}\n{progress}")
            self._record_tool_result(turn, item["id"], item["name"], result, duration)

    def _advance_plan(self, turn: Turn, finished: list[tuple[dict[str, Any], ToolResult, int]]) -> str:
        """Tick off the plan steps this round completed; the progress line ('' if none)."""
        if turn.plan is None:
            return ""
        advanced = False
        for item, result, _duration in finished:
            ok = result.ok and (result.meta or {}).get("count") != 0  # a search that found nothing
            written = None
            if ok and item["name"] in plan.WRITE_TOOLS:
                written = plan.written_path(item["name"], item["args"], item["workspace"] or self._workspace(turn.conversation_id))
            advanced = turn.plan.observe(item["name"], item["args"], ok, written) or advanced
        if not advanced:
            return ""
        if not turn.plan.remaining() and turn.plan_done_at is None:
            turn.plan_done_at = turn.search_rounds
        turn.emit("plan", {"steps": turn.plan.statuses()})
        return turn.plan.progress_line(turn.reply_text)

    def _anchor_searches(self, turn: Turn, item: dict[str, Any]) -> None:
        """Complete a search that points at the conversation ("how old is he") with what it is about."""
        if item["name"] not in ("web_search", "multi_search"):
            return
        try:
            earlier = [m for m in self.db.recent_messages(turn.conversation_id, limit=8)
                       if m.get("id") != turn.user_message_id and m.get("role") in ("user", "assistant")]
        except Exception:  # pragma: no cover - defensive
            return
        args = item["args"]
        if item["name"] == "web_search":
            fixed = research.anchor_query(args["query"], turn.message, earlier)
            changed = fixed != args["query"]
            item["args"] = {**args, "query": fixed}
        else:
            queries = [research.anchor_query(query, turn.message, earlier) for query in args["queries"]]
            changed = queries != args["queries"]
            fixed = "; ".join(queries)
            item["args"] = {**args, "queries": queries}
        if changed:
            item["notes"] = [*item["notes"], f"searched for \"{fixed}\" so the query names its subject"]

    def _prepare_tool_call(
        self, turn: Turn, call: dict[str, Any], settings: dict[str, Any], *, cut_off: bool = False
    ) -> dict[str, Any]:
        """Rescue, validate, safety-check and approve one call (no execution)."""
        rescued = toolcall.prepare_call(call, tools.TOOL_SPECS)
        name = rescued.name
        item: dict[str, Any] = {
            "id": rescued.id, "name": name, "args": rescued.args, "notes": rescued.notes,
            "result": None, "decision": None, "approved": False, "workspace": None, "location": "",
        }

        def refuse(content: str, preview: str) -> dict[str, Any]:
            item["result"] = ToolResult(ok=False, content=content, preview=preview)
            return item

        def refuse_shown(content: str, preview: str, risk: str) -> dict[str, Any]:  # refused before approval
            turn.emit("tool_call", {"id": item["id"], "name": name, "args": item["args"], "risk": risk,
                                    "requires_approval": False})
            return refuse(content, preview)

        if turn.failure_streak >= FAILURE_PAUSE_AT:  # stop a guessing loop; make the model answer
            return refuse(prompts.TOOLS_PAUSED, "Tools paused after repeated failures")
        spec = tools.TOOL_SPECS.get(name)
        try:
            if rescued.error or spec is None:
                raise ValueError(rescued.error or f"unknown tool: {name}")
            item["args"] = tools.validate_args(name, rescued.args)
            self._anchor_searches(turn, item)
        except ValueError as exc:
            # Small models fix calls far better when shown the exact valid usage.
            message = toolcall.correction(spec, str(exc)) if spec is not None else f"Error: {exc}"
            return refuse_shown(message, message.splitlines()[0], "read")
        if cut_off and spec.risk in ("write", "exec"):  # repaired JSON would save a truncated file
            return refuse_shown(prompts.CUT_OFF_CALL, "Call cut off by the reply length limit", spec.risk)
        pacing = self._pacing_refusal(turn, name, item["args"])
        if pacing:
            return refuse_shown(pacing, pacing.split(":")[0], spec.risk)
        workspace, state = self._workspace(turn.conversation_id), turn.safety
        route = turn.routes.get(item["id"]) or (
            turn.plan.route(name, item["args"], workspace) if turn.plan is not None else None)
        if route is not None:  # a file or folder the user named outside the workspace
            workspace, item["args"], item["location"] = route.workspace, route.args, route.shown
            turn.safety.setdefault("counters", {"writes": 0, "deletes": 0})  # shared by the copy below
            state = {**turn.safety, "workspace": str(workspace.root)}
        elif name in plan.ROUTED_TOOLS and isinstance(item["args"].get("path"), str):
            try:  # outside the workspace and not named by the user: refuse now, never ask for a call that cannot run
                workspace.resolve(item["args"]["path"])
            except SandboxError as exc:
                return refuse_shown(f"Error: {exc}. {prompts.OUTSIDE_HINT}", str(exc), spec.risk)
        item["workspace"] = workspace
        decision = safety.decide(
            self._settings(), name, item["args"], workspace=workspace, turn_state=state, risk=spec.risk
        )
        requires = decision.mode == "ask"
        # "Allow in this chat" covers the same kind of step only, and never one that can destroy data.
        allowed_here = requires and decision.can_always and decision.allow_key in self._chat_allowed.get(
            turn.conversation_id, ()
        )
        if allowed_here:
            requires = False
            decision = replace(decision, mode="auto", reasons=(*decision.reasons, "allowed for this chat"))
        # The user sees where a routed call really goes, not its folder-relative path.
        shown_args = {**item["args"], "path": item["location"]} if item["location"] else item["args"]
        turn.emit(
            "tool_call",
            {
                "id": item["id"],
                "name": name,
                "args": shown_args,
                "risk": spec.risk,
                "requires_approval": requires,
                **decision.payload(),
            },
        )
        item["decision"] = decision
        if decision.mode == "deny":
            preview = "Tool disabled" if decision.source == "permission" else "Blocked by a safety check"
            return refuse(decision.denial_text(), preview)
        signature = tools.tool_call_signature(name, item["args"])
        seen = turn.tool_signatures.get(signature, 0)
        turn.tool_signatures[signature] = seen + 1
        if seen >= 2:
            return refuse(
                "Error: this exact tool call was already made twice this turn with the "
                "same arguments. Use the earlier result, change the arguments, or move on.",
                "Duplicate tool call stopped",
            )
        if requires:
            allow_key = decision.allow_key if decision.can_always else None
            if not self._request_approval(turn, name, spec.risk, shown_args, decision.payload(), allow_key=allow_key):
                return refuse("The user denied permission for this tool call. Do not retry it.", "Denied by user")
            item["approved"] = True
        elif allowed_here:
            item["approved"] = True  # the user's in-chat approval covers it (runs as an approved call)
        return item

    @staticmethod
    def _next_step(turn: Turn) -> str:
        """The plan's next open step as an instruction, for the act-now nudge."""
        left = turn.plan.remaining() if turn.plan is not None else []
        if left:
            return f"{left[0].index}. {left[0].short} → {left[0].hint()}."
        return "make the tool call you need, or answer the user directly."

    @staticmethod
    def _pacing_refusal(turn: Turn, name: str, args: dict[str, Any]) -> str:
        """Why a call should not run: a greeting needs no tools, and searching has a budget
        (near-repeats refused; after the plan's tool steps, one more round). '' to run it."""
        if plan.is_small_talk(turn.message):
            return prompts.SMALL_TALK_NO_TOOLS
        if plan.is_about_kiwibro(turn.message):
            return prompts.ABOUT_YOU_NO_TOOLS
        if name not in plan.SEARCH_TOOLS:
            return ""
        limit = plan.SEARCH_ROUNDS_MAX
        if turn.plan_done_at is not None:
            limit = min(limit, turn.plan_done_at + plan.SEARCH_ROUNDS_AFTER_PLAN)
        if turn.search_rounds >= limit:
            return prompts.SEARCH_SPENT
        terms = plan.search_terms(name, args)
        earlier = plan.repeated(terms, turn.searches)
        if earlier is not None:
            return prompts.SEARCH_REPEATED.format(query=json.dumps(earlier[:120], ensure_ascii=False))
        query = " | ".join(map(str, args.get("queries") or [])) if name == "multi_search" else str(args.get("query") or "")
        turn.searches.append((terms, query))
        return ""

    def _folder(self, conversation_id: str | None = None) -> str:
        """The conversation's own folder when it has one, else the Settings workspace."""
        bound = workspaces.folder_for(self.db, conversation_id)
        return bound or str(self._settings().get("workspace") or config.default_workspace())

    def _workspace(self, conversation_id: str | None = None) -> Workspace:
        return Workspace(self._folder(conversation_id))

    def _library(self) -> library.Library:
        return library.get_library(self.data_dir / "library.db")

    def _record_tool_result(
        self, turn: Turn, call_id: str, name: str, result: ToolResult, duration_ms: int
    ) -> None:
        self.db.add_message(
            turn.conversation_id,
            "tool",
            tools.shape_content(result.content or "", 8000),
            tool_call_id=call_id,
            tool_name=name,
            meta={"ok": result.ok, "duration_ms": duration_ms},
        )
        turn.emit(
            "tool_result",
            {
                "id": call_id,
                "name": name,
                "ok": result.ok,
                "duration_ms": duration_ms,
                "preview": (result.preview or result.content or "")[:400],
            },
        )

    def _request_approval(
        self,
        turn: Turn,
        name: str,
        risk: str,
        args: dict[str, Any],
        safety_info: dict[str, Any] | None = None,
        *,
        allow_key: str | None = "",
    ) -> bool:
        """Ask the user. ``allow_key`` is what "allow in this chat" would remember (None: the
        step can destroy data, so it is offered once only and the choice is never kept)."""
        approval_id = f"ap_{uuid.uuid4().hex}"
        approval = Approval(
            id=approval_id,
            conversation_id=turn.conversation_id,
            tool=name,
            args=args,
        )
        with self._approvals_guard:
            self._approvals[approval_id] = approval
        turn.approval_ids.append(approval_id)
        manual = self._settings().get("autonomy", config.DEFAULT_AUTONOMY) == "supervised"
        turn.emit(
            "approval",
            {
                "approval_id": approval_id,
                "tool": name,
                "args": args,
                "summary": tools.summary_for(name, args),
                "risk": risk,
                "timeout_s": int(config.APPROVAL_TIMEOUT_S),
                # What "always" means: Manual mode sets the tool's permission to auto; the
                # automatic modes allow this tool's risky steps for the rest of the chat.
                "always_scope": "tool" if manual else "chat",
                **(safety_info or {}),
            },
        )
        deadline = time.monotonic() + config.APPROVAL_TIMEOUT_S
        while not approval.event.wait(0.25):
            if turn.abort.is_set():
                with self._approvals_guard:
                    self._approvals.pop(approval_id, None)
                raise TurnAborted()
            if time.monotonic() >= deadline:
                break
        decision = approval.decision or "deny"
        with self._approvals_guard:
            self._approvals.pop(approval_id, None)
        if decision == "always" and allow_key is not None:
            if manual:
                try:
                    self.db.set_permission(name, "auto")
                except ValueError:  # pragma: no cover - defensive
                    log.warning("could not persist always permission for %s", name)
            else:
                self._chat_allowed.setdefault(turn.conversation_id, set()).add(allow_key or name)
        return decision in ("allow", "always")

    # --- finish paths ------------------------------------------------------

    def _raise_if_aborted(self, turn: Turn) -> None:
        if turn.abort.is_set():
            raise TurnAborted()

    def _finish_aborted(self, turn: Turn) -> None:
        try:
            if turn.partial_text or turn.partial_reasoning:
                message = self.db.add_message(
                    turn.conversation_id,
                    "assistant",
                    turn.partial_text,
                    reasoning=turn.partial_reasoning,
                    model=turn.model,
                    meta={"aborted": True},
                )
                turn.last_message_id = message["id"]
            self._insert_usage(turn, turn.model or "")
            turn.emit(
                "done",
                {
                    "message_id": turn.last_message_id or turn.user_message_id,
                    "aborted": True,
                    "model": turn.model,
                    "usage": self._usage_payload(turn),
                },
            )
        except Exception:  # pragma: no cover - best effort
            log.exception("failed to finish aborted turn")

    def _finish_error(self, turn: Turn, code: str, message: str) -> None:
        try:
            if turn.partial_text or turn.partial_reasoning:
                saved = self.db.add_message(
                    turn.conversation_id,
                    "assistant",
                    turn.partial_text,
                    reasoning=turn.partial_reasoning,
                    model=turn.model,
                    meta={"error": code},
                )
                turn.last_message_id = saved["id"]
            self._insert_usage(turn, turn.model or "")
            recoverable = code in ("connection", "timeout", "stream", "server", "no_model")
            turn.emit(
                "error",
                {"message": message, "code": code, "recoverable": recoverable},
            )
        except Exception:  # pragma: no cover - best effort
            log.exception("failed to finish error turn")

    # --- title -------------------------------------------------------------

    def _unanswered_repeat(self, turn: Turn) -> dict[str, Any] | None:
        """The chat's last message when it is this same user message, still unanswered.

        Retry (and re-sending after a failed or stopped turn) repeats the text; storing it
        again showed the model the request twice in a row and searched it twice."""
        last = self.db.recent_messages(turn.conversation_id, limit=1)
        if not last or last[0].get("role") != "user":
            return None
        message = last[0]
        if " ".join((message.get("content") or "").split()) != " ".join(turn.message.split()):
            return None
        stored = (message.get("meta") or {}).get("attachments") or []
        if [{k: v for k, v in item.items() if k != "seen"} for item in stored] != (turn.attachments or []):
            return None
        turn.attachments = stored  # already looked at: _see_attachments keeps the descriptions
        return message

    def _maybe_title(self, turn: Turn) -> None:
        """Deterministic title (no extra LLM call): from the first user message, and again
        from the first real request when the chat opened with small talk ("Hi")."""
        conversation = self.db.get_conversation(turn.conversation_id)
        if conversation is None or not conversation.get("title_auto"):
            return
        current = conversation.get("title") or "New chat"
        if current != "New chat":
            if not is_small_talk_title(current) or len(terms.content_words(turn.message)) < 3:
                return
        title = derive_title(turn.message)
        if not title or title == current or (current != "New chat" and is_small_talk_title(title)):
            return
        updated = self.db.update_conversation(turn.conversation_id, title=title, touch=False)
        if updated:
            turn.emit("title", {"id": turn.conversation_id, "title": title})

    def _start_memory_extraction(self, conversation_id: str, model: str | None) -> None:
        """Queue memory extraction for this chat. It runs in the background once no reply
        has streamed for ``memory_idle_s``: on a CPU-only machine an extraction running
        beside the next reply slowed its first token from 2.8 s to 8.4 s (measured), and
        could store a fact that reply had just changed."""
        if not model:
            return
        with self._extracting_guard:
            self._memory_queue[conversation_id] = model
            if self._memory_worker is None or not self._memory_worker.is_alive():
                self._memory_worker = threading.Thread(
                    target=self._memory_loop, name="kiwibro-memory", daemon=True
                )
                self._memory_worker.start()
        self._memory_wake.set()

    def _cancel_memory_extraction(self) -> None:
        with self._extracting_guard:
            for cancel in self._extracting.values():
                cancel.set()

    def _wait_until_idle(self) -> None:
        """Until no reply streams, none did for ``memory_idle_s`` and no warm-up runs."""
        while True:
            quiet = time.monotonic() - self._turn_activity_at
            warming = self.readiness is not None and self.readiness.status().get("phase") in ("loading", "warming")
            busy = bool(self.active_turns()) or warming
            if not busy and quiet >= self.memory_idle_s:
                return
            time.sleep(0.25 if busy else max(0.05, min(0.5, self.memory_idle_s - quiet)))

    def _memory_loop(self) -> None:
        while True:
            self._memory_wake.wait(30.0)
            self._memory_wake.clear()
            while True:
                self._wait_until_idle()
                with self._extracting_guard:
                    if not self._memory_queue:
                        break
                    conversation_id = next(iter(self._memory_queue))
                    model = self._memory_queue.pop(conversation_id)
                    cancel = self._extracting[conversation_id] = threading.Event()
                self._memory_extraction_worker(conversation_id, model, cancel)

    def _memory_extraction_worker(self, conversation_id: str, model: str, cancel: threading.Event) -> None:
        try:
            counts = self.memory.extract_and_apply(conversation_id, model, abort=cancel)
            if counts.get("cancelled"):
                with self._extracting_guard:  # a reply started: try again once it is over
                    self._memory_queue.setdefault(conversation_id, model)
                log.info("memory extraction for %s paused for a reply", conversation_id)
            else:
                log.info("memory extraction for %s: %s", conversation_id, counts)
        except Exception:  # never fail anything because of memory extraction
            log.warning("memory extraction worker failed", exc_info=True)
        finally:
            with self._extracting_guard:
                self._extracting.pop(conversation_id, None)
