"""Token estimation, context budget and rolling-summary compaction."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable

from . import config, prompts
from .db import Database
from .llm import LMStudioClient, LMStudioError

log = logging.getLogger(__name__)

MESSAGE_OVERHEAD = 4
SAFETY_FACTOR = 1.1
TOOL_RESULT_CAP = 4000
COMPACT_TRIGGER = 0.75
MAX_COMPACTION_PASSES = 3
MIN_KEEP_MESSAGES = 4
SUMMARY_MAX_TOKENS = 600
SUMMARY_TRANSCRIPT_CAP = 14000


def estimate_tokens(text: str | None) -> int:
    """Cheap tokenizer-free estimate (~3.6 ASCII chars/token, ~0.9 tokens/UTF-8 char).

    C-speed counting only: ``str.isascii`` fast path, otherwise one
    ``encode('ascii', 'ignore')`` pass instead of a per-character Python loop.
    """
    if not text:
        return 0
    if text.isascii():
        return max(1, math.ceil(len(text) / 3.6))
    ascii_chars = len(text.encode("ascii", "ignore"))
    non_ascii = len(text) - ascii_chars
    return max(1, math.ceil(ascii_chars / 3.6 + non_ascii * 0.9))


def _tool_call_tokens(tool_calls: Any) -> int:
    if not tool_calls:
        return 0
    try:
        encoded = json.dumps(tool_calls, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return 0
    return estimate_tokens(encoded)


def message_tokens(message: dict[str, Any]) -> int:
    content = message.get("content")
    if isinstance(content, str):
        tokens = estimate_tokens(content)
    elif isinstance(content, list):
        tokens = estimate_tokens(json.dumps(content, ensure_ascii=False))
    else:
        tokens = 0
    tokens += _tool_call_tokens(message.get("tool_calls"))
    tokens += estimate_tokens(message.get("name") or "")
    return tokens + MESSAGE_OVERHEAD


def raw_message_sum(messages: list[dict[str, Any]]) -> int:
    """Sum of per-message token estimates before safety/calibration factors."""
    return sum(message_tokens(message) for message in messages)


def estimate_messages(messages: list[dict[str, Any]], factor: float = 1.0) -> int:
    return math.ceil(raw_message_sum(messages) * SAFETY_FACTOR * factor)


def context_length(model_info: dict[str, Any] | None, fallback: int = 32768) -> int:
    if model_info:
        try:
            value = int(model_info.get("context") or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return fallback


def compute_budget(context: int, max_output_tokens: int) -> int:
    reserve = min(int(max_output_tokens), max(512, context // 8))
    return max(1024, context - reserve - 512)


def context_stats(used: int, budget: int) -> dict[str, Any]:
    budget = max(1, budget)
    percent = max(0, min(100, int(round(used * 100 / budget))))
    return {"used": int(used), "budget": int(budget), "percent": percent}


# Tool results decay with age: the newest outputs stay detailed, older ones
# shrink hard (head+tail) instead of being re-sent in full on every turn.
TOOL_RESULT_CAP_RECENT = TOOL_RESULT_CAP
TOOL_RESULT_CAP_MID = 1200
TOOL_RESULT_CAP_OLD = 300
TOOL_RECENT_FULL = 2
TOOL_MID_COUNT = 4


def capped_tool_content(content: str, cap: int = TOOL_RESULT_CAP) -> str:
    if len(content) <= cap:
        return content
    omitted = len(content) - cap
    if cap >= 1500:
        return content[:cap] + f"\n… [truncated {omitted} chars]"
    marker = f"\n… [truncated {omitted} chars] …\n"
    head = max(40, (cap - len(marker)) * 2 // 3)
    tail = max(20, cap - len(marker) - head)
    return content[:head] + marker + content[-tail:]


def to_llm_message(message: dict[str, Any], tool_cap: int = TOOL_RESULT_CAP) -> dict[str, Any] | None:
    role = message.get("role")
    if role not in ("user", "assistant", "tool"):
        return None
    content = message.get("content") or ""
    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.get("tool_call_id") or "",
            "content": capped_tool_content(content, tool_cap),
        }
    tool_calls = message.get("tool_calls")
    if role == "assistant":
        if not content and not tool_calls:
            return None
        result: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            normalised = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, (dict, list)):
                    arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
                if not isinstance(arguments, str) or not arguments.strip():
                    arguments = "{}"
                normalised.append(
                    {
                        "id": call.get("id") or "call_0",
                        "type": call.get("type") or "function",
                        "function": {"name": function.get("name") or "", "arguments": arguments},
                    }
                )
            if normalised:
                result["tool_calls"] = normalised
        return result
    return {"role": "user", "content": content}


def tool_cap_for_age(age_from_newest: int) -> int:
    """Character cap tiers by age (kept as the shared tier table).

    The build path keys tiers on *user-turn* age (see ``build_llm_messages``)
    so a rendered tool result changes tier at most once; this helper remains
    for callers that count raw tool-result age.
    """
    if age_from_newest < TOOL_RECENT_FULL:
        return TOOL_RESULT_CAP_RECENT
    if age_from_newest < TOOL_RECENT_FULL + TOOL_MID_COUNT:
        return TOOL_RESULT_CAP_MID
    return TOOL_RESULT_CAP_OLD


def build_llm_messages(system_prompt: str, db_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert DB messages to LLM messages with prefix-stable tool caps.

    KV-cache rule: text rendered for older turns must not change when a new
    turn arrives, or the model server re-prefills the whole history. Tool
    caps therefore key on user-turn age with a single transition — results
    from the newest user turn stay full-size, everything older is compact
    forever (exact content remains available via episodic recall).
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    last_user = max(
        (i for i, m in enumerate(db_messages) if m.get("role") == "user"), default=-1
    )
    for index, db_message in enumerate(db_messages):
        cap = TOOL_RESULT_CAP_RECENT if index > last_user else TOOL_RESULT_CAP_OLD
        converted = to_llm_message(db_message, cap)
        if converted is not None:
            messages.append(converted)
    return messages


def place_turn_context(messages: list[dict[str, Any]], note: str) -> list[dict[str, Any]]:
    """Append volatile per-turn context to the newest user message only.

    Volatile text (time, recalled excerpts, anchors) must never enter the
    system prompt or older messages — that would invalidate the server's
    KV prefix for the whole conversation. Returns a new list; the input
    messages are not mutated.
    """
    note = (note or "").strip()
    if not note:
        return messages
    result = list(messages)
    for index in range(len(result) - 1, -1, -1):
        if result[index].get("role") == "user":
            merged = dict(result[index])
            merged["content"] = f"{merged.get('content') or ''}\n\n{note}"
            result[index] = merged
            return result
    return result


def format_transcript(db_messages: list[dict[str, Any]], cap: int = SUMMARY_TRANSCRIPT_CAP) -> str:
    """Compact transcript for summarisation representing the WHOLE segment.

    The per-message cap shrinks adaptively until every message fits, so the
    oldest part of a large segment is summarised (truncated) rather than
    silently dropped from the summariser's input.
    """
    per_message = 1500
    while True:
        lines: list[str] = []
        for message in db_messages:
            role = message.get("role")
            content = (message.get("content") or "").strip()
            if role == "tool":
                label = f"Tool ({message.get('tool_name') or 'result'})"
                content = capped_tool_content(content, min(700, per_message))
            elif role == "assistant":
                label = "Assistant"
                calls = message.get("tool_calls") or []
                if calls:
                    names = ", ".join(
                        str((call.get("function") or {}).get("name") or "tool")
                        for call in calls
                        if isinstance(call, dict)
                    )
                    content = (content + f"\n[used tools: {names}]").strip()
                content = content[:per_message]
            else:
                label = "User"
                content = content[:per_message]
            if content:
                lines.append(f"{label}: {content}")
        joined = "\n".join(lines)
        if len(joined) <= cap or per_message <= 120:
            return joined[:cap]
        per_message //= 2


def combine_system_prompt(system_prompt: str, summary: str) -> str:
    """Append the rolling conversation summary so compacted history is never lost."""
    summary = (summary or "").strip()
    if not summary:
        return system_prompt
    return f"{system_prompt.rstrip()}\n\n## Earlier conversation summary\n{summary}"


# --- small-context adaptation (v2) -------------------------------------------
#
# Every prompt share and every utility (summariser) call is derived from the
# model's own context length. Fixed sizes are what broke v1 on 4k models:
# the summary call alone needed ~4k tokens regardless of the model.

RECALL_MAX_TOKENS = 1200
RECALL_EXCERPT_CHARS = 500
ANCHOR_MAX_CHARS = 280


def utility_limits(ctx_len: int) -> tuple[int, int]:
    """(max input tokens, max output tokens) for a summariser call on this model."""
    ctx = max(1024, int(ctx_len or 32768))
    output = min(SUMMARY_MAX_TOKENS, max(150, ctx // 16))
    # Leave room for the system prompt + previous summary + the output itself.
    input_tokens = min(5000, max(700, ctx // 3))
    return input_tokens, output


def recall_token_cap(budget: int) -> int:
    """Tokens available for recalled exact excerpts within a prompt budget."""
    return min(RECALL_MAX_TOKENS, max(0, budget // 8))


def memory_token_cap(budget: int) -> int:
    """Tokens available for the durable-memory block within a prompt budget."""
    return min(800, max(150, budget // 10))


def _db_message_est(message: dict[str, Any]) -> int:
    """Token estimate for a DB message, preferring the cached ``est_tokens``."""
    est = message.get("est_tokens")
    if isinstance(est, int) and est > 0:
        return est
    return estimate_tokens(message.get("content") or "") + MESSAGE_OVERHEAD


def _chunk_messages(
    segment: list[dict[str, Any]], max_tokens: int
) -> list[list[dict[str, Any]]]:
    """Split a message segment into chunks each fitting ``max_tokens``."""
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    spent = 0
    for message in segment:
        cost = _db_message_est(message)
        if current and spent + cost > max_tokens:
            chunks.append(current)
            current, spent = [], 0
        current.append(message)
        spent += cost
    if current:
        chunks.append(current)
    return chunks


def _clean_leading_boundary(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop orphan leading tool results so the history starts at a valid turn boundary."""
    start = 0
    while start < len(messages) and messages[start].get("role") == "tool":
        start += 1
    return messages[start:]


def _clean_summary_segment(segment: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove trailing half-finished tool exchanges from a segment to summarise."""
    segment = list(segment)
    while segment:
        last = segment[-1]
        if last.get("role") == "tool" or (last.get("role") == "assistant" and last.get("tool_calls")):
            segment.pop()
        else:
            break
    return segment


@dataclass
class PreparedContext:
    messages: list[dict[str, Any]]
    used: int
    budget: int
    percent: int
    compacted: bool = False
    trimmed: bool = False
    context_length: int = 32768
    model_info: dict[str, Any] | None = None
    summary: str = ""
    raw_sum: int = 0
    last_seq: int = 0
    trigger: int = 0
    recalled: int = 0
    anchored: bool = False

    @property
    def stats(self) -> dict[str, Any]:
        return {"used": self.used, "budget": self.budget, "percent": self.percent}


class ContextEngine:
    def __init__(
        self,
        db: Database,
        llm: LMStudioClient,
        settings_provider: Callable[[], dict[str, Any]],
    ) -> None:
        self.db = db
        self.llm = llm
        self._settings = settings_provider
        self._calibration: dict[str, float] = {}

    # --- calibration ---------------------------------------------------------

    def calibration_factor(self, model: str) -> float:
        return self._calibration.get(model, 1.0)

    def note_usage(self, model: str, raw_sum: int, actual_prompt_tokens: int) -> None:
        """Learn estimator bias per model from server-reported prompt tokens (EMA)."""
        if raw_sum < 200 or not actual_prompt_tokens:
            return
        estimated = raw_sum * SAFETY_FACTOR * self.calibration_factor(model)
        if estimated <= 0:
            return
        ratio = min(1.6, max(0.6, actual_prompt_tokens / estimated))
        old = self.calibration_factor(model)
        self._calibration[model] = min(1.4, max(0.75, old * (0.7 + 0.3 * ratio)))

    def _used(self, raw: int, model: str) -> int:
        return math.ceil(raw * SAFETY_FACTOR * self.calibration_factor(model))

    # --- sizing ------------------------------------------------------------

    def inventory(self, model: str) -> tuple[int, dict[str, Any] | None]:
        info = self.llm.model_info(model)
        return context_length(info), info

    def budget_for(self, model: str) -> tuple[int, int, dict[str, Any] | None]:
        settings = self._settings()
        ctx, info = self.inventory(model)
        return ctx, compute_budget(ctx, int(settings.get("max_output_tokens") or config.DEFAULT_MAX_OUTPUT_TOKENS)), info

    def estimate_conversation(
        self,
        conversation_id: str,
        model: str,
        *,
        conversation: dict[str, Any] | None = None,
        db_messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Read-only context usage for display (no compaction side effects).

        Callers that already loaded the conversation/messages (e.g. the API
        detail endpoint) can pass them to avoid duplicate queries.
        """
        conversation = conversation or self.db.get_conversation(conversation_id)
        if conversation is None:
            return {"used": 0, "budget": 1024, "percent": 0}
        if db_messages is None:
            db_messages = self.db.list_messages(
                conversation_id, after_seq=conversation["summary_through_seq"]
            )
        else:
            db_messages = [
                m for m in db_messages if m["seq"] > conversation["summary_through_seq"]
            ]
        system = combine_system_prompt("", conversation["summary"])
        used = estimate_messages(
            build_llm_messages(system, db_messages), self.calibration_factor(model)
        )
        _ctx, budget, _info = self.budget_for(model)
        return context_stats(used, budget)

    # --- preparation -------------------------------------------------------

    def prepare(
        self,
        conversation_id: str,
        model: str,
        system_prompt: str,
        *,
        allow_compaction: bool = True,
        turn_context: str = "",
    ) -> PreparedContext:
        """Build the prompt for one turn.

        ``turn_context`` is volatile per-turn text from the caller (time,
        directives, …); it is merged with this engine's recalled excerpts
        and task anchor and appended to the newest user message only, so
        the stable prefix (system + older turns) stays byte-identical
        between turns for KV-cache reuse.
        """
        settings = self._settings()
        conversation = self.db.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"unknown conversation {conversation_id}")
        ctx_len, model_info = self.inventory(model)
        budget = compute_budget(ctx_len, int(settings.get("max_output_tokens") or config.DEFAULT_MAX_OUTPUT_TOKENS))
        trigger = int(budget * COMPACT_TRIGGER)
        history = self.db.list_messages(conversation_id, after_seq=conversation["summary_through_seq"])
        messages = build_llm_messages(
            combine_system_prompt(system_prompt, conversation["summary"]), history
        )
        raw = raw_message_sum(messages)
        compacted = False
        trimmed = False

        if allow_compaction:
            passes = 0
            while self._used(raw, model) > trigger and passes < MAX_COMPACTION_PASSES:
                segment = self._target_segment(history, messages, raw, trigger, model)
                if not segment:
                    break
                summary = self.summarise(
                    conversation["summary"], segment, model, conversation_id=conversation_id
                )
                if not summary:
                    break
                conversation = self.db.update_conversation(
                    conversation_id,
                    summary=summary,
                    summary_through_seq=segment[-1]["seq"],
                    touch=False,
                ) or conversation
                history = self.db.list_messages(conversation_id, after_seq=conversation["summary_through_seq"])
                messages = build_llm_messages(
                    combine_system_prompt(system_prompt, conversation["summary"]), history
                )
                raw = raw_message_sum(messages)
                compacted = True
                passes += 1

            if self._used(raw, model) > trigger:
                trimmed_history = self._drop_old_tool_results(history)
                candidate = build_llm_messages(
                    combine_system_prompt(system_prompt, conversation["summary"]), trimmed_history
                )
                candidate_raw = raw_message_sum(candidate)
                if candidate_raw < raw:
                    history = trimmed_history
                    messages = candidate
                    raw = candidate_raw
                    trimmed = True

            while self._used(raw, model) > budget and len(history) > MIN_KEEP_MESSAGES:
                keep = max(MIN_KEEP_MESSAGES, len(history) // 2)
                history = self._truncate(history, keep)
                messages = build_llm_messages(
                    combine_system_prompt(system_prompt, conversation["summary"]), history
                )
                raw = raw_message_sum(messages)
                trimmed = True

        # Volatile per-turn note: caller context + recalled exact excerpts +
        # task anchor, appended to the newest user message only (KV rule).
        recalled = 0
        anchored = False
        note_parts = [turn_context.strip()] if turn_context.strip() else []
        if conversation["summary_through_seq"] > 0:
            cap = recall_token_cap(budget)
            room = budget - self._used(raw, model)
            note, recalled, anchored = self._recall_note(
                conversation, history, min(cap, max(0, room))
            )
            if note:
                note_parts.append(note)
        if note_parts:
            messages = place_turn_context(messages, "\n\n".join(note_parts))
            raw = raw_message_sum(messages)

        used = self._used(raw, model)
        stats = context_stats(used, budget)
        return PreparedContext(
            messages=messages,
            used=stats["used"],
            budget=stats["budget"],
            percent=stats["percent"],
            compacted=compacted,
            trimmed=trimmed,
            context_length=ctx_len,
            model_info=model_info,
            summary=conversation["summary"],
            raw_sum=raw,
            last_seq=history[-1]["seq"] if history else conversation["summary_through_seq"],
            trigger=trigger,
            recalled=recalled,
            anchored=anchored,
        )

    def _recall_note(
        self,
        conversation: dict[str, Any],
        history: list[dict[str, Any]],
        token_cap: int,
    ) -> tuple[str, int, bool]:
        """Exact older excerpts + original-task anchor for compacted chats.

        Returns (note_text, excerpt_count, anchored). Everything here is
        budget-capped and best-effort: recall failures degrade to the
        summary alone, never to a failed turn. Invariant: the note never
        pushes the prompt over budget (callers pass remaining room).
        """
        if token_cap < 80 or not history:
            return "", 0, False
        window_start = history[0]["seq"]
        lines: list[str] = []
        spent = 0
        anchored = False
        try:
            first_rows = self.db.list_messages(conversation["id"], after_seq=0, limit=12)
            first_user = next((m for m in first_rows if m.get("role") == "user"), None)
            if first_user and first_user["seq"] < window_start:
                anchor = " ".join((first_user.get("content") or "").split())[:ANCHOR_MAX_CHARS]
                if anchor:
                    block = f"## Original task (anchor)\n{anchor}"
                    spent += estimate_tokens(block)
                    lines.append(block)
                    anchored = True
            queries = [
                (m.get("content") or "") for m in reversed(history) if m.get("role") == "user"
            ][:2]
            query = " ".join(q for q in queries if q.strip())[:600]
            excerpts: list[str] = []
            if query.strip():
                hits = self.db.search_messages(
                    query, conversation_id=conversation["id"], before_seq=window_start, limit=6
                )
                for hit in hits:
                    text = " ".join((hit.get("content") or "").split())[:RECALL_EXCERPT_CHARS]
                    if not text:
                        continue
                    label = "User" if hit.get("role") == "user" else "Assistant" if hit.get("role") == "assistant" else "Tool"
                    line = f"- [{label}, earlier] {text}"
                    cost = estimate_tokens(line)
                    if spent + cost > token_cap and excerpts:
                        break
                    excerpts.append(line)
                    spent += cost
            if excerpts:
                header = "## Recalled from earlier in this conversation (exact excerpts)"
                lines.append(header + "\n" + "\n".join(excerpts))
            return "\n\n".join(lines), len(excerpts), anchored
        except Exception:  # recall is an enhancement, never a failure mode
            log.warning("episodic recall note failed", exc_info=True)
            return "", 0, False

    def memory_cap_for(self, model: str) -> int:
        """Durable-memory prompt cap scaled to this model's budget."""
        _ctx, budget, _info = self.budget_for(model)
        return memory_token_cap(budget)

    def _target_segment(
        self,
        history: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        raw: int,
        trigger: int,
        model: str,
    ) -> list[dict[str, Any]]:
        """Oldest prefix whose summarisation brings usage to ~60% of trigger.

        Summarising only what is needed keeps more recent history verbatim
        instead of always folding everything except the last few messages,
        while the 60% target (hysteresis) keeps compaction infrequent.
        """
        if len(history) <= MIN_KEEP_MESSAGES:
            return []
        candidates = history[:-MIN_KEEP_MESSAGES]
        # Hysteresis: compact down to ~60% of the trigger, not just under it.
        # Compaction rewrites the prompt prefix (full re-prefill on the
        # server), so it should happen rarely and buy many turns of headroom.
        excess = self._used(raw, model) - int(trigger * 0.6) + SUMMARY_MAX_TOKENS
        # messages[0] is the system prompt; the rest align with converted history.
        body = messages[1:] if messages and messages[0].get("role") == "system" else messages
        factor = SAFETY_FACTOR * self.calibration_factor(model)
        saved = 0.0
        count = 0
        for converted in body:
            if count >= len(candidates) or saved >= excess:
                break
            saved += message_tokens(converted) * factor
            count += 1
        segment = _clean_summary_segment(candidates[: max(count, 1)])
        return segment

    def extend(
        self,
        prepared: PreparedContext,
        conversation_id: str,
        model: str,
        system_prompt: str,
        *,
        turn_context: str = "",
    ) -> PreparedContext:
        """Append only new DB messages to a prepared context (agent tool rounds).

        Falls back to a full ``prepare`` when the appended estimate crosses
        the compaction trigger or the summary boundary moved underneath us.
        """
        conversation = self.db.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"unknown conversation {conversation_id}")
        if conversation["summary"] != prepared.summary or turn_context.strip():
            # A fresh turn note belongs to a full prepare (it rewrites the
            # newest user message); tool rounds within a turn pass none.
            return self.prepare(conversation_id, model, system_prompt, turn_context=turn_context)
        new_db = self.db.list_messages(conversation_id, after_seq=prepared.last_seq)
        if not new_db:
            return prepared
        if any(m["seq"] <= conversation["summary_through_seq"] for m in new_db):
            return self.prepare(conversation_id, model, system_prompt)
        appended = build_llm_messages("", new_db)
        raw = prepared.raw_sum + raw_message_sum(appended)
        used = self._used(raw, model)
        if prepared.trigger and used > prepared.trigger:
            return self.prepare(conversation_id, model, system_prompt)
        stats = context_stats(used, prepared.budget)
        return PreparedContext(
            messages=prepared.messages + appended,
            used=stats["used"],
            budget=stats["budget"],
            percent=stats["percent"],
            compacted=False,
            trimmed=False,
            context_length=prepared.context_length,
            model_info=prepared.model_info,
            summary=prepared.summary,
            raw_sum=raw,
            last_seq=new_db[-1]["seq"],
            trigger=prepared.trigger,
            recalled=prepared.recalled,
            anchored=prepared.anchored,
        )

    @staticmethod
    def _drop_old_tool_results(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tool_positions = [i for i, message in enumerate(history) if message.get("role") == "tool"]
        drop = set(tool_positions[:-2]) if len(tool_positions) > 2 else set()
        result = []
        for index, message in enumerate(history):
            if index in drop:
                item = dict(message)
                item["content"] = "[earlier tool output omitted]"
                result.append(item)
            else:
                result.append(message)
        return result

    @staticmethod
    def _truncate(history: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
        kept = _clean_leading_boundary(history[-keep:])
        # Never start with an assistant tool-call message whose tool results were cut off.
        while kept and kept[0].get("role") == "assistant" and kept[0].get("tool_calls"):
            kept = _clean_leading_boundary(kept[1:])
        return kept

    # --- summarisation & recovery -----------------------------------------

    def _summary_call(self, previous: str, transcript: str, model: str, max_out: int) -> str | None:
        user_prompt = prompts.SUMMARY_USER.format(
            previous=previous or "(none)", transcript=transcript, max_tokens=max_out
        )
        try:
            completion = self.llm.complete(
                model=model,
                messages=[
                    {"role": "system", "content": prompts.SUMMARY_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_out,
                temperature=0.0,
            )
        except LMStudioError as exc:
            log.warning("summary call failed: %s", exc.message)
            return None
        return (completion.text or "").strip() or None

    def summarise(
        self,
        previous_summary: str,
        segment: list[dict[str, Any]],
        model: str,
        *,
        conversation_id: str | None = None,
    ) -> str | None:
        """Summarise a segment in chunks that always fit the model's window.

        Each chunk becomes a durable ``summary_segments`` row (when the
        conversation is known), so the fine-grained record survives even
        after the rolling digest is reduced again. A single-chunk segment
        keeps the exact v1 prompt shape (previous + transcript).
        Invariant: no summariser call ever exceeds the model's context —
        input and output sizes come from ``utility_limits(ctx_len)``.
        """
        if not segment:
            return None
        ctx_len, _info = self.inventory(model)
        in_tokens, out_tokens = utility_limits(ctx_len)
        chunks = _chunk_messages(segment, in_tokens)
        char_cap = max(1200, in_tokens * 3)

        def run_chunk(chunk: list[dict[str, Any]], previous: str) -> str | None:
            transcript = format_transcript(chunk, cap=char_cap)
            if not transcript.strip():
                return None
            text = self._summary_call(previous, transcript, model, out_tokens)
            if text and conversation_id:
                try:
                    self.db.add_summary_segment(
                        conversation_id, chunk[0]["seq"], chunk[-1]["seq"], text,
                        est_tokens=estimate_tokens(text),
                    )
                except Exception:  # segment bookkeeping must never fail compaction
                    log.warning("could not store summary segment", exc_info=True)
            return text

        if len(chunks) == 1:
            text = run_chunk(chunks[0], previous_summary)
            return text[:4000] if text else None
        partials: list[str] = []
        for chunk in chunks:
            text = run_chunk(chunk, "")
            if not text:
                return None  # all-or-nothing: never advance past unsummarised history
            partials.append(text)
        merged = self._reduce_summaries(partials, previous_summary, model, in_tokens, out_tokens)
        if not merged:
            # The segment rows are durable; a concatenated digest still beats loss.
            merged = "\n\n".join(partials)
        return merged[:4000]

    def _reduce_summaries(
        self, partials: list[str], previous: str, model: str, in_tokens: int, out_tokens: int
    ) -> str | None:
        """Merge partial summaries in batches that each fit the model.

        A flat join of many partials would itself overflow a small window
        (the v1 bug one level up), so reduction is a bounded tree: group
        consecutive partials under the input cap, summarise each group,
        repeat until one text remains. The caller's ``previous`` summary
        only enters the final merge.
        """
        level = list(partials)
        for _round in range(6):
            joined = "\n\n".join(level)
            if estimate_tokens(joined) + estimate_tokens(previous) <= in_tokens:
                return self._summary_call(previous, joined, model, out_tokens)
            groups: list[list[str]] = []
            group: list[str] = []
            spent = 0
            for part in level:
                cost = estimate_tokens(part)
                if group and spent + cost > in_tokens:
                    groups.append(group)
                    group, spent = [], 0
                group.append(part)
                spent += cost
            if group:
                groups.append(group)
            if len(groups) == len(level):  # single oversized partials: cannot shrink further
                return self._summary_call(previous, joined[: in_tokens * 3], model, out_tokens)
            reduced: list[str] = []
            for batch in groups:
                text = self._summary_call("", "\n\n".join(batch), model, out_tokens)
                if not text:
                    return None
                reduced.append(text)
            level = reduced
        return self._summary_call(previous, "\n\n".join(level)[: in_tokens * 3], model, out_tokens)

    def summarize_long_text(self, text: str, model: str) -> str | None:
        """Map-reduce summary of one long text (tool output, document, page).

        Splits on paragraph boundaries into model-sized chunks, summarises
        each, then reduces. Returns None when the model cannot summarise —
        callers fall back to head+tail truncation, never to silent loss of
        the only copy (the full text stays wherever the caller stored it).
        """
        if not (text or "").strip():
            return None
        ctx_len, _info = self.inventory(model)
        in_tokens, out_tokens = utility_limits(ctx_len)
        char_cap = max(1200, in_tokens * 3)
        pieces: list[str] = []
        current = ""
        for paragraph in text.split("\n\n"):
            if current and len(current) + len(paragraph) > char_cap:
                pieces.append(current)
                current = ""
            current = f"{current}\n\n{paragraph}".strip()
            while len(current) > char_cap:  # one oversized paragraph
                pieces.append(current[:char_cap])
                current = current[char_cap:]
        if current:
            pieces.append(current)
        partials = []
        for piece in pieces:
            summary = self._summary_call("", piece, model, out_tokens)
            if not summary:
                return None
            partials.append(summary)
        if len(partials) == 1:
            return partials[0][:4000]
        merged = self._reduce_summaries(partials, "", model, in_tokens, out_tokens)
        return (merged or "\n\n".join(partials))[:4000]

    def recover_aggressive(self, conversation_id: str, model: str) -> bool:
        """Summarise roughly half of the remaining history (context-limit recovery)."""
        conversation = self.db.get_conversation(conversation_id)
        if conversation is None:
            return False
        history = self.db.list_messages(conversation_id, after_seq=conversation["summary_through_seq"])
        if len(history) <= 2:
            return False
        keep = max(2, len(history) // 2)
        segment = _clean_summary_segment(history[:-keep])
        if not segment:
            return False
        summary = self.summarise(
            conversation["summary"], segment, model, conversation_id=conversation_id
        )
        if not summary:
            return False
        self.db.update_conversation(
            conversation_id,
            summary=summary,
            summary_through_seq=segment[-1]["seq"],
            touch=False,
        )
        return True

    def hard_recover(self, conversation_id: str, keep: int = 6) -> bool:
        """Last-resort truncation: keep only the newest few messages."""
        conversation = self.db.get_conversation(conversation_id)
        if conversation is None:
            return False
        history = self.db.list_messages(conversation_id, after_seq=conversation["summary_through_seq"])
        if len(history) <= keep:
            return False
        kept = self._truncate(history, keep)
        if not kept:
            return False
        boundary_seq = kept[0]["seq"] - 1
        if boundary_seq <= conversation["summary_through_seq"]:
            return False
        previous = conversation["summary"].strip()
        note = "[Earlier conversation was trimmed to fit the model's context window.]"
        summary = (previous + "\n\n" + note).strip() if previous else note
        self.db.update_conversation(
            conversation_id,
            summary=summary,
            summary_through_seq=boundary_seq,
            touch=False,
        )
        return True
