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

from . import checkpoints, config, delegate, directives, guard, inference, library, prompts, safety, toolcall, tools, workspaces
from .context import ContextEngine, PreparedContext, compute_budget
from .db import Database
from .llm import ChatResult, ContextLimitError, LMStudioError, LMStudioClient, TurnAborted
from .memory import MemoryEngine
from .sandbox import SandboxError, Workspace
from .search import SearchEngine
from .tools import ToolContext, ToolResult

log = logging.getLogger(__name__)


class BusyError(Exception):
    """A turn is already running for this conversation."""


class ConversationNotFound(Exception):
    """Unknown conversation id."""


_LEADING_NOISE = re.compile(
    r"^(?:(?:please|pls|kindly|hey|hi|hello|ok|okay|so|now|just)\b[\s,]+)+", re.IGNORECASE
)
_LEADING_PREPOSITION = re.compile(
    r"^(?:from|about|regarding|in|on|at|by|for|to)\s+", re.IGNORECASE
)
_REQUEST_PREFIX = re.compile(
    r"^(?:(?:can|could|would|will)\s+you(?:\s+please)?|i\s+(?:want|need)\s+you\s+to"
    r"|i'd\s+like\s+you\s+to|i\s+would\s+like\s+you\s+to)\s+",
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
            _LEADING_PREPOSITION,
        ):
            text = pattern.sub("", text).strip()
        if text == before:
            break
    text = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
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
    reviewed: bool = False  # the one end-of-turn self-check has been offered
    pending_review: str = ""  # checklist nudge for the next round (transient)

    def emit(self, event: str, data: dict[str, Any]) -> None:
        self.queue.put((event, data))


class AgentManager:
    def __init__(
        self,
        db: Database,
        llm: LMStudioClient,
        context: ContextEngine,
        memory: MemoryEngine,
        search: SearchEngine,
        settings_provider: Callable[[], dict[str, Any]],
    ) -> None:
        self.db = db
        self.llm = llm
        self.context = context
        self.memory = memory
        self.search = search
        self._settings = settings_provider
        self.data_dir = Path(db.path).resolve().parent  # checkpoints + document library live here
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._turns: dict[str, Turn] = {}
        self._thinking: dict[str, bool] = {}  # conversation id -> last turn's thinking decision
        self._approvals: dict[str, Approval] = {}
        self._approvals_guard = threading.Lock()

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
    ) -> Turn:
        if conversation_id:
            conversation = self.db.get_conversation(conversation_id)
            if conversation is None:
                raise ConversationNotFound(conversation_id)
        else:
            conversation = self.db.create_conversation(
                title="New chat", model=model or (self._settings().get("model"))
            )
            conversation_id = conversation["id"]
        lock = self._lock_for(conversation_id)
        if not lock.acquire(blocking=False):
            raise BusyError(conversation_id)
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
        except LMStudioError as exc:
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
        turn.finished = True
        turn.done_sent = True
        turn.queue.put(None)

    def _execute(self, turn: Turn) -> None:
        settings = self._settings()
        conversation = self.db.get_conversation(turn.conversation_id)
        if conversation is None:
            raise ConversationNotFound(turn.conversation_id)
        user_message = self.db.add_message(
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
        except LMStudioError as exc:
            self._emit_meta(turn, conversation, settings, model=None)
            raise exc
        turn.model = model
        if conversation["model"] != model:
            conversation = (
                self.db.update_conversation(turn.conversation_id, model=model, touch=False) or conversation
            )
        self._raise_if_aborted(turn)

        model_info = self.llm.model_info(model)
        if turn.attachments:
            # meta is contractually the first SSE event: send it before looking at the
            # files (a vision helper can take a while); real context stats follow.
            self._emit_meta(turn, conversation, settings, model=model)
            self._see_attachments(turn, model, user_message)
        # KV-cache rule: message 0 stays byte-identical across turns so LM Studio
        # reuses the cached prefix; the clock, retrieved memories and directives
        # ride on the newest user message (fixed for the whole turn).
        system_prompt = self._system_prompt(
            {**settings, "workspace": self._folder(turn.conversation_id)}, model=model, model_info=model_info
        )
        turn_context = self._turn_context(turn, settings)
        turn.system_prompt, turn.turn_context = system_prompt, turn_context
        prepared = self.context.prepare(
            turn.conversation_id, model, system_prompt, turn_context=turn_context
        )
        # meta is contractually the first SSE event; context follows it.
        if turn.attachments:
            turn.emit("context", {**prepared.stats, "compacted": prepared.compacted or prepared.trimmed})
        else:
            self._emit_meta(turn, conversation, settings, model=model, prepared=prepared)
            self._emit_context(turn, prepared)
        self._raise_if_aborted(turn)

        model_info = prepared.model_info or model_info or {}
        turn.page_limit = library.page_limit(prepared.budget)
        tool_list: list[dict[str, Any]] | None = None
        # Models without native tool calling use the text protocol: tools are
        # described in message 0 and calls are parsed from the reply text.
        text_mode = bool(settings.get("agent_enabled", True)) and not model_info.get("tools", False)
        if settings.get("agent_enabled", True) and model_info.get("tools", False):
            # One profile-sized set, identical for every turn and conversation: the
            # tool list renders ahead of all messages, so any change re-prefills them.
            tool_list = tools.tool_defs_for(settings=settings)
        try:
            max_rounds = int(settings.get("max_tool_rounds") or config.MAX_TOOL_ROUNDS)
        except (TypeError, ValueError):
            max_rounds = config.MAX_TOOL_ROUNDS
        max_rounds = max(config.TOOL_ROUNDS_RANGE[0], min(max_rounds, config.TOOL_ROUNDS_RANGE[1]))

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
            if round_index > 0 and not turn.nudged and toolcall.is_empty_reply(result.content, result.tool_calls):
                # Small models sometimes stop with an empty reply after tool
                # results; one transient nudge (never stored) gets them going.
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
            turn.partial_text = ""
            turn.partial_reasoning = ""
            if not result.tool_calls:
                turn.pending_review = self._review_nudge(turn, settings) if round_index < max_rounds - 1 else ""
                if not turn.pending_review:
                    break
                turn.emit("status", {"phase": "thinking", "label": "Checking the work"})
                continue
            turn.tool_rounds = round_index + 1
            self._handle_tool_calls(turn, result.tool_calls, settings)
            self._raise_if_aborted(turn)
            if round_index == max_rounds - 1:
                note = "I reached the tool-use limit for this turn, so I stopped here."
                note_message = self.db.add_message(
                    turn.conversation_id, "assistant", note, model=model, meta={"truncated": True}
                )
                turn.last_message_id = note_message["id"]
                turn.emit("delta", {"text": "\n\n" + note})
                break

        # Final answer phase. Title is deterministic-first so `done` stays fast; memory
        # extraction runs in a daemon thread after `done` (the stream closes immediately,
        # the UI refreshes its memory count from /api/bootstrap or /api/memories later).
        self._raise_if_aborted(turn)
        self._drain_steering(turn)
        if conversation["title_auto"] and conversation["title"] == "New chat":
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
            },
        )

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
    ) -> ChatResult:
        settings = self._settings()
        temperature = float(settings.get("temperature", config.DEFAULT_TEMPERATURE))
        max_tokens = int(settings.get("max_output_tokens", config.DEFAULT_MAX_OUTPUT_TOKENS))
        if turn.thinking is None:  # decided once per turn: flipping mid-turn re-prefills everything
            turn.thinking = inference.turn_reasoning(
                settings.get("reasoning"),
                turn.message,
                previous=self._thinking.get(turn.conversation_id),
                prompt_tokens=prepared.used,
            )
            self._thinking[turn.conversation_id] = turn.thinking
        think = turn.thinking
        extra = inference.reasoning_payload(inference.model_profile(model, prepared.model_info), think=think)
        attempts = 0
        while True:
            self._raise_if_aborted(turn)
            messages = toolcall.to_text_protocol(prepared.messages) if text_mode else prepared.messages
            if nudge:  # transient: steers this one request, never stored
                messages = [*messages, {"role": "user", "content": nudge}]
            reuse = inference.PREFIX_TRACKER.observe(model, messages, tool_list, extra)
            turn.timer = inference.RoundTimer()
            turn.stream_filter = toolcall.StreamFilter()
            try:
                result = self.llm.stream_chat(
                    model=model,
                    messages=messages,
                    tools=tool_list,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_delta=self._delta_callback(turn),
                    on_reasoning=self._reasoning_callback(turn),
                    on_rewrite=self._rewrite_callback(turn),
                    abort=turn.abort,
                    extra=extra or None,
                )
                turn.timer.finish()
                self._note_round(turn, round_index, think=think, reuse=reuse, result=result)
                self._rescue_tool_calls(turn, result, settings)
                return result
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
                    turn.conversation_id, model, turn.system_prompt, turn_context=turn.turn_context
                )
                self._emit_context(turn, prepared)
                turn.partial_text = ""
                turn.partial_reasoning = ""

    @staticmethod
    def _note_round(
        turn: Turn, round_index: int, *, think: bool, reuse: dict[str, int], result: ChatResult
    ) -> None:
        """Fold one round's inference telemetry into the turn's ``done.inference``."""
        timing = turn.timer.stats((result.usage or {}).get("completion_tokens")) if turn.timer else {}
        if round_index == 0:
            turn.inference.update(reuse, thinking=think)
            if "ttft_ms" in timing:
                turn.inference["ttft_ms"] = timing["ttft_ms"]
        if "tokens_per_s" in timing:
            turn.inference["tokens_per_s"] = timing["tokens_per_s"]
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
        if held_back is not None:
            release = held_back.flush() + ("" if result.tool_calls else held_back.held)
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
            # Tool-call markup is held back so users never see raw call syntax.
            visible = turn.stream_filter.feed(text) if turn.stream_filter is not None else text
            if visible:
                turn.partial_text += visible
                turn.emit("delta", {"text": visible})

        return on_delta

    @staticmethod
    def _rewrite_callback(turn: Turn) -> Callable[[str, str], None]:
        """The model wrote reasoning as plain text: show it as reasoning, keep the answer."""

        def on_rewrite(text: str, reasoning: str) -> None:
            turn.partial_text = text
            turn.partial_reasoning += reasoning
            turn.emit("rewrite", {"text": text, "reasoning": reasoning})

        return on_rewrite

    def _reasoning_callback(self, turn: Turn) -> Callable[[str], None]:
        def on_reasoning(text: str) -> None:
            if turn.abort.is_set():
                raise TurnAborted()
            if turn.timer is not None:
                turn.timer.mark_first()
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
            if item.get("kind") in ("image", "video", "audio", "svg"):
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

    def _turn_context(self, turn: Turn, settings: dict[str, Any]) -> str:
        """Volatile per-turn note: clock, retrieved memories, directive ledger (never message 0)."""
        memory_block, _memories = self.memory.build_block(
            turn.message,
            enabled=bool(settings.get("memory_enabled", True)),
            # Budget-scaled cap: a fixed 800-token block would starve the
            # working window on small-context models.
            max_tokens=self.context.memory_cap_for(turn.model) if turn.model else 800,
        )
        blocks = [inference.time_line(), memory_block, _directives_note(self.db, turn.conversation_id)]
        body = "\n\n".join(block.strip() for block in blocks if block and block.strip())
        return f"{prompts.TURN_CONTEXT_HEADER}\n{body}"

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
        workspace_block = prompts.WORKSPACE_BLOCK.format(workspace=workspace) if workspace else ""
        if not settings.get("agent_enabled", True):
            tools_block = prompts.TOOLS_DISABLED_BLOCK
        elif model_info is not None and not model_info.get("tools"):
            tools_block = toolcall.protocol_block(_text_mode_specs(settings))
        else:
            tools_block = prompts.TOOLS_BLOCK + tools.more_tools_line(settings)
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
        for call in calls:
            self._raise_if_aborted(turn)
            prepared_calls.append(self._prepare_tool_call(turn, call, settings))
            self._raise_if_aborted(turn)
        runnable = [item for item in prepared_calls if item["result"] is None]
        outcomes: dict[str, tuple[ToolResult, int]] = {}

        def run_one(item: dict[str, Any]) -> tuple[str, ToolResult, int]:
            started = time.monotonic()
            name, args = item["name"], item["args"]
            context = ToolContext(
                workspace=self._workspace(turn.conversation_id),
                memory=self.memory,
                search=self.search,
                conversation_id=turn.conversation_id,
                llm=self.llm,
                current_model=turn.model,
            )
            if tools.TOOL_SPECS[name].risk == "write":  # every file change stays undoable
                checkpoints.snapshot_call(self.data_dir, context.workspace, name, args, turn.conversation_id)
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
        for item in prepared_calls:
            if item["result"] is not None:
                result, duration = item["result"], 0
            else:
                result, duration = outcomes[item["id"]]
            turn.tool_calls_made += 1
            if item["notes"]:  # show small models how their call was corrected
                result = replace(result, content=f"{result.content}\n[Kiwibro fixed this call: {'; '.join(item['notes'][:4])}]")
            turn.failure_streak = 0 if result.ok else turn.failure_streak + 1
            if turn.failure_streak == FAILURE_COACH_AT:
                result = replace(result, content=f"{result.content}\n{prompts.FAILURE_COACH.format(count=FAILURE_COACH_AT)}")
            self._record_tool_result(turn, item["id"], item["name"], result, duration)

    def _prepare_tool_call(
        self, turn: Turn, call: dict[str, Any], settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Rescue, validate, safety-check and approve one call (no execution)."""
        rescued = toolcall.prepare_call(call, tools.TOOL_SPECS)
        name = rescued.name
        item: dict[str, Any] = {
            "id": rescued.id, "name": name, "args": rescued.args, "notes": rescued.notes,
            "result": None, "decision": None, "approved": False,
        }

        def refuse(content: str, preview: str) -> dict[str, Any]:
            item["result"] = ToolResult(ok=False, content=content, preview=preview)
            return item

        if turn.failure_streak >= FAILURE_PAUSE_AT:  # stop a guessing loop; make the model answer
            return refuse(prompts.TOOLS_PAUSED, "Tools paused after repeated failures")
        spec = tools.TOOL_SPECS.get(name)
        try:
            if rescued.error or spec is None:
                raise ValueError(rescued.error or f"unknown tool: {name}")
            item["args"] = tools.validate_args(name, rescued.args)
        except ValueError as exc:
            turn.emit(
                "tool_call",
                {"id": item["id"], "name": name, "args": item["args"], "risk": "read", "requires_approval": False},
            )
            # Small models fix calls far better when shown the exact valid usage.
            message = toolcall.correction(spec, str(exc)) if spec is not None else f"Error: {exc}"
            return refuse(message, message.splitlines()[0])
        decision = safety.decide(
            self._settings(), name, item["args"], workspace=self._workspace(turn.conversation_id), turn_state=turn.safety, risk=spec.risk
        )
        requires = decision.mode == "ask"
        turn.emit(
            "tool_call",
            {
                "id": item["id"],
                "name": name,
                "args": item["args"],
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
            if not self._request_approval(turn, name, spec.risk, item["args"], decision.payload()):
                return refuse("The user denied permission for this tool call. Do not retry it.", "Denied by user")
            item["approved"] = True
        return item

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
    ) -> bool:
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
        turn.emit(
            "approval",
            {
                "approval_id": approval_id,
                "tool": name,
                "args": args,
                "summary": tools.summary_for(name, args),
                "risk": risk,
                "timeout_s": int(config.APPROVAL_TIMEOUT_S),
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
        if decision == "always":
            try:
                self.db.set_permission(name, "auto")
            except ValueError:  # pragma: no cover - defensive
                log.warning("could not persist always permission for %s", name)
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

    def _maybe_title(self, turn: Turn) -> None:
        """Deterministic title derived from the first user message (no extra LLM call)."""
        title = derive_title(turn.message)
        if not title:
            return
        updated = self.db.update_conversation(turn.conversation_id, title=title, touch=False)
        if updated:
            turn.emit("title", {"id": turn.conversation_id, "title": title})

    def _start_memory_extraction(self, conversation_id: str, model: str | None) -> None:
        if not model:
            return
        thread = threading.Thread(
            target=self._memory_extraction_worker,
            args=(conversation_id, model),
            name=f"kiwibro-memory-{conversation_id}",
            daemon=True,
        )
        thread.start()

    def _memory_extraction_worker(self, conversation_id: str, model: str) -> None:
        try:
            counts = self.memory.extract_and_apply(conversation_id, model)
            log.info("memory extraction for %s: %s", conversation_id, counts)
        except Exception:  # never fail anything because of memory extraction
            log.warning("memory extraction worker failed", exc_info=True)
