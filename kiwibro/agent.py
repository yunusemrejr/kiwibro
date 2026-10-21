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
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Iterator

from . import config, prompts, tools

try:
    from . import inference
except ImportError:  # inference lane not landed yet: degrade, never break imports
    import types as _types
    from datetime import datetime as _datetime

    class _NoOpTimer:
        def mark_first(self) -> None: pass
        def finish(self) -> None: pass
        def stats(self, _completion_tokens: Any = None) -> dict[str, Any]: return {}

    class _NoOpTracker:
        def observe(self, *_args: Any, **_kw: Any) -> dict[str, Any]: return {}

    inference = _types.SimpleNamespace(  # type: ignore[assignment]
        RoundTimer=_NoOpTimer,
        PREFIX_TRACKER=_NoOpTracker(),
        wants_reasoning=lambda *_a, **_k: True,
        model_profile=lambda *_a, **_k: None,
        reasoning_payload=lambda *_a, **_k: {},
        time_line=lambda now=None: "Local time: "
        + (now or _datetime.now()).astimezone().strftime("%A, %d %B %Y, %H:%M %Z"),
    )
    logging.getLogger(__name__).warning(
        "kiwibro.inference unavailable; reasoning control + telemetry disabled"
    )
from .context import ContextEngine, PreparedContext, compute_budget
from .db import Database
from .llm import ChatResult, ContextLimitError, LMStudioError, LMStudioClient, TurnAborted
from .memory import MemoryEngine
from .sandbox import Workspace
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


def _directives_note(conversation_id: str) -> str:
    """The directive ledger's per-turn note ('' until ``directives`` is installed or when quiet)."""
    try:
        from . import directives
    except ImportError:
        return ""
    try:
        return directives.render_note(conversation_id) or ""
    except Exception:  # the ledger must never break a turn
        log.warning("directive note failed", exc_info=True)
        return ""


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
    tool_rounds: int = 0
    writes_made: int = 0
    steering_notes: list[str] = field(default_factory=list)
    steering_lock: threading.Lock = field(default_factory=threading.Lock)
    system_prompt: str = ""
    turn_context: str = ""
    timer: inference.RoundTimer | None = None
    inference: dict[str, Any] = field(default_factory=dict)

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
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._turns: dict[str, Turn] = {}
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
        turn = Turn(conversation_id=conversation_id, message=message, requested_model=model)
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
        user_message = self.db.add_message(turn.conversation_id, "user", turn.message)
        turn.user_message_id = user_message["id"]
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
        # KV-cache rule: message 0 stays byte-identical across turns so LM Studio
        # reuses the cached prefix; the clock, retrieved memories and directives
        # ride on the newest user message (fixed for the whole turn).
        system_prompt = self._system_prompt(settings, model=model, model_info=model_info)
        turn_context = self._turn_context(turn, settings)
        turn.system_prompt, turn.turn_context = system_prompt, turn_context
        prepared = self.context.prepare(
            turn.conversation_id, model, system_prompt, turn_context=turn_context
        )
        # meta is contractually the first SSE event; context follows it.
        self._emit_meta(turn, conversation, settings, model=model, prepared=prepared)
        self._emit_context(turn, prepared)
        self._raise_if_aborted(turn)

        model_info = prepared.model_info or model_info or {}
        tool_list: list[dict[str, Any]] | None = None
        if settings.get("agent_enabled", True) and model_info.get("tools", False):
            # Task-shaped subset (tool_profile) keeps small models effective;
            # fixed for the whole turn so the prompt prefix stays stable.
            tool_list = tools.tool_defs_for(settings=settings, message=turn.message)
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
            result = self._stream_with_recovery(turn, model, prepared, tool_list, round_index)
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
                break
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

    def _stream_with_recovery(
        self,
        turn: Turn,
        model: str,
        prepared: PreparedContext,
        tool_list: list[dict[str, Any]] | None,
        round_index: int = 0,
    ) -> ChatResult:
        settings = self._settings()
        temperature = float(settings.get("temperature", config.DEFAULT_TEMPERATURE))
        max_tokens = int(settings.get("max_output_tokens", config.DEFAULT_MAX_OUTPUT_TOKENS))
        think = inference.wants_reasoning(settings.get("reasoning"), turn.message, round_index=round_index)
        extra = inference.reasoning_payload(inference.model_profile(model, prepared.model_info), think=think)
        attempts = 0
        while True:
            self._raise_if_aborted(turn)
            reuse = inference.PREFIX_TRACKER.observe(model, prepared.messages, tool_list)
            turn.timer = inference.RoundTimer()
            try:
                result = self.llm.stream_chat(
                    model=model,
                    messages=prepared.messages,
                    tools=tool_list,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_delta=self._delta_callback(turn),
                    on_reasoning=self._reasoning_callback(turn),
                    abort=turn.abort,
                    extra=extra or None,
                )
                turn.timer.finish()
                self._note_round(turn, round_index, think=think, reuse=reuse, result=result)
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

    # --- event helpers -----------------------------------------------------

    def _delta_callback(self, turn: Turn) -> Callable[[str], None]:
        def on_delta(text: str) -> None:
            if turn.abort.is_set():
                raise TurnAborted()
            if turn.timer is not None:
                turn.timer.mark_first()
            turn.partial_text += text
            turn.emit("delta", {"text": text})

        return on_delta

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

    def _turn_context(self, turn: Turn, settings: dict[str, Any]) -> str:
        """Volatile per-turn note: clock, retrieved memories, directive ledger (never message 0)."""
        memory_block, _memories = self.memory.build_block(
            turn.message,
            enabled=bool(settings.get("memory_enabled", True)),
            # Budget-scaled cap: a fixed 800-token block would starve the
            # working window on small-context models.
            max_tokens=self.context.memory_cap_for(turn.model) if turn.model else 800,
        )
        blocks = [inference.time_line(), memory_block, _directives_note(turn.conversation_id)]
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
            tools_block = prompts.TOOLS_UNSUPPORTED_BLOCK
        else:
            tools_block = prompts.TOOLS_BLOCK
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
            workspace = Workspace(self._settings().get("workspace") or config.default_workspace())
            context = ToolContext(
                workspace=workspace,
                memory=self.memory,
                search=self.search,
                conversation_id=turn.conversation_id,
                llm=self.llm,
                current_model=turn.model,
            )
            result = tools.execute_tool(item["name"], item["args"], context)
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
                spec = tools.TOOL_SPECS.get(item["name"])
                if spec is not None and spec.risk in ("write", "exec"):
                    turn.writes_made += 1
            turn.tool_calls_made += 1
            self._record_tool_result(turn, item["id"], item["name"], result, duration)

    def _prepare_tool_call(
        self, turn: Turn, call: dict[str, Any], settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Parse, validate, permission-check and approve one call (no execution)."""
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:8]}")
        raw_arguments = function.get("arguments")
        try:
            parsed = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            if not isinstance(parsed, dict):
                raise ValueError
            args: dict[str, Any] = parsed
        except ValueError:
            args = {}
        try:
            spec = tools.get_spec(name)
            validated = tools.validate_args(name, args)
        except ValueError as exc:
            turn.emit(
                "tool_call",
                {"id": call_id, "name": name, "args": args, "risk": "read", "requires_approval": False},
            )
            result = ToolResult(ok=False, content=f"Error: {exc}", preview=f"Error: {exc}")
            return {"id": call_id, "name": name, "args": args, "result": result}
        mode = tools.permission_mode(self._settings(), name)
        requires = mode == "ask"
        turn.emit(
            "tool_call",
            {
                "id": call_id,
                "name": name,
                "args": validated,
                "risk": spec.risk,
                "requires_approval": requires,
            },
        )
        if mode == "deny":
            result = ToolResult(
                ok=False,
                content="The user has disabled this tool. Do not retry it.",
                preview="Tool disabled",
            )
            return {"id": call_id, "name": name, "args": validated, "result": result}
        if spec.risk in ("write", "exec") and turn.writes_made >= config.MAX_WRITES_PER_TURN:
            result = ToolResult(
                ok=False,
                content=(
                    f"Error: safety limit reached — {config.MAX_WRITES_PER_TURN} write/command "
                    "actions already ran this turn. Summarise progress and let the user "
                    "continue in a new message instead of making more changes."
                ),
                preview="Write limit reached for this turn",
            )
            return {"id": call_id, "name": name, "args": validated, "result": result}
        signature = tools.tool_call_signature(name, validated)
        seen = turn.tool_signatures.get(signature, 0)
        turn.tool_signatures[signature] = seen + 1
        if seen >= 2:
            result = ToolResult(
                ok=False,
                content=(
                    "Error: this exact tool call was already made twice this turn with the "
                    "same arguments. Use the earlier result, change the arguments, or move on."
                ),
                preview="Duplicate tool call stopped",
            )
            return {"id": call_id, "name": name, "args": validated, "result": result}
        if requires and not self._request_approval(turn, name, spec.risk, validated):
            result = ToolResult(
                ok=False,
                content="The user denied permission for this tool call. Do not retry it.",
                preview="Denied by user",
            )
            return {"id": call_id, "name": name, "args": validated, "result": result}
        return {"id": call_id, "name": name, "args": validated, "result": None}

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
