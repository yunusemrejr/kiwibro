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
    """Character cap for a tool result given how many tool results are newer."""
    if age_from_newest < TOOL_RECENT_FULL:
        return TOOL_RESULT_CAP_RECENT
    if age_from_newest < TOOL_RECENT_FULL + TOOL_MID_COUNT:
        return TOOL_RESULT_CAP_MID
    return TOOL_RESULT_CAP_OLD


def build_llm_messages(system_prompt: str, db_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    tool_total = sum(1 for m in db_messages if m.get("role") == "tool")
    tool_seen = 0
    for db_message in db_messages:
        cap = TOOL_RESULT_CAP
        if db_message.get("role") == "tool":
            cap = tool_cap_for_age(tool_total - 1 - tool_seen)
            tool_seen += 1
        converted = to_llm_message(db_message, cap)
        if converted is not None:
            messages.append(converted)
    return messages


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
    ) -> PreparedContext:
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
                summary = self.summarise(conversation["summary"], segment, model)
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
        )

    def _target_segment(
        self,
        history: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        raw: int,
        trigger: int,
        model: str,
    ) -> list[dict[str, Any]]:
        """Oldest prefix whose summarisation brings usage under ~92% of trigger.

        Summarising only what is needed keeps more recent history verbatim
        instead of always folding everything except the last few messages.
        """
        if len(history) <= MIN_KEEP_MESSAGES:
            return []
        candidates = history[:-MIN_KEEP_MESSAGES]
        excess = self._used(raw, model) - int(trigger * 0.92) + SUMMARY_MAX_TOKENS
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
    ) -> PreparedContext:
        """Append only new DB messages to a prepared context (agent tool rounds).

        Falls back to a full ``prepare`` when the appended estimate crosses
        the compaction trigger or the summary boundary moved underneath us.
        """
        conversation = self.db.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"unknown conversation {conversation_id}")
        if conversation["summary"] != prepared.summary:
            return self.prepare(conversation_id, model, system_prompt)
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

    def summarise(self, previous_summary: str, segment: list[dict[str, Any]], model: str) -> str | None:
        transcript = format_transcript(segment)
        if not transcript.strip():
            return None
        user_prompt = prompts.SUMMARY_USER.format(
            previous=previous_summary or "(none)",
            transcript=transcript,
            max_tokens=SUMMARY_MAX_TOKENS,
        )
        try:
            completion = self.llm.complete(
                model=model,
                messages=[
                    {"role": "system", "content": prompts.SUMMARY_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=SUMMARY_MAX_TOKENS,
                temperature=0.0,
            )
        except LMStudioError as exc:
            log.warning("summary call failed: %s", exc.message)
            return None
        text = (completion.text or "").strip()
        if not text:
            return None
        return text[:4000]

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
        summary = self.summarise(conversation["summary"], segment, model)
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
