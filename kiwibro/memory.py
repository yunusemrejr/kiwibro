"""Durable memory: extraction, deduplication, retrieval, rendering, user CRUD."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import Any, Callable

from . import config, prompts
from .context import estimate_tokens
from .db import Database
from .llm import LMStudioClient, LMStudioError, parse_json_loose

log = logging.getLogger(__name__)

DEDUPE_SEQ_RATIO = 0.86
DEDUPE_JACCARD = 0.72
MEMORY_PROMPT_TOKEN_CAP = 800
MAX_OPS_PER_TURN = 3
MIN_CONTENT = 3
MAX_CONTENT = 300
EXTRACT_TRANSCRIPT_CHARS = 9000

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    lowered = (text or "").lower().strip()
    lowered = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _tokens(normalized: str) -> set[str]:
    return set(_WORD_RE.findall(normalized or ""))


def jaccard(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def similarity(a: str, b: str) -> float:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def is_similar(a: str, b: str) -> bool:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if jaccard(na, nb) >= DEDUPE_JACCARD:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= DEDUPE_SEQ_RATIO


def clean_content(text: Any) -> str:
    content = re.sub(r"\s+", " ", str(text or "")).strip()
    return content


def validate_content(text: Any) -> str:
    content = clean_content(text)
    if len(content) < MIN_CONTENT:
        raise ValueError("memory content is too short")
    if len(content) > MAX_CONTENT:
        content = content[:MAX_CONTENT].rstrip()
    return content


def validate_kind(kind: Any) -> str:
    value = str(kind or "fact").strip().lower()
    if value not in config.MEMORY_KINDS:
        raise ValueError(f"unknown memory kind: {kind}")
    return value


def parse_ops(raw: Any) -> list[dict[str, Any]]:
    """Parse extraction output defensively into a list of operation dicts."""
    payload = parse_json_loose(raw) if isinstance(raw, str) else raw
    if isinstance(payload, dict):
        payload = payload.get("ops") or payload.get("operations") or payload.get("memories")
        if isinstance(payload, dict):
            payload = [payload]
    if not isinstance(payload, list):
        return []
    ops: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op") or item.get("operation") or "").strip().lower()
        if op not in ("add", "update", "delete"):
            continue
        entry: dict[str, Any] = {"op": op}
        if item.get("id"):
            entry["id"] = str(item["id"]).strip()
        if item.get("content") is not None:
            entry["content"] = item["content"]
        if item.get("kind") is not None:
            entry["kind"] = item["kind"]
        ops.append(entry)
    return ops


class MemoryEngine:
    def __init__(
        self,
        db: Database,
        llm: LMStudioClient,
        settings_provider: Callable[[], dict[str, Any]],
    ) -> None:
        self.db = db
        self.llm = llm
        self._settings = settings_provider

    # --- retrieval ---------------------------------------------------------

    def retrieve(self, query: str, limit: int = 12) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []

        def add(memories: list[dict[str, Any]], cap: int) -> None:
            for memory in memories:
                if len(result) >= limit or len(result) >= cap:
                    return
                if memory["id"] in seen:
                    continue
                seen.add(memory["id"])
                result.append(memory)

        add(self.db.pinned_memories(5), limit)
        if (query or "").strip():
            add(self.db.search_memories(query, 8), limit)
        add(self.db.recently_used_memories(5), limit)
        if result:
            try:
                self.db.mark_memories_used([memory["id"] for memory in result])
            except Exception:  # pragma: no cover - defensive
                log.warning("could not mark memories as used", exc_info=True)
        return result

    @staticmethod
    def render_block(memories: list[dict[str, Any]], max_tokens: int = MEMORY_PROMPT_TOKEN_CAP) -> str:
        lines: list[str] = []
        used = 0
        for memory in memories:
            line = f"- {memory['content']}"
            cost = estimate_tokens(line)
            if lines and used + cost > max_tokens:
                break
            lines.append(line)
            used += cost
        return "\n".join(lines)

    def build_block(self, query: str, enabled: bool = True) -> tuple[str, list[dict[str, Any]]]:
        if not enabled:
            return "", []
        memories = self.retrieve(query)
        bullets = self.render_block(memories)
        if not bullets:
            return "", []
        return prompts.MEMORY_BLOCK.format(bullets=bullets), memories

    # --- user-facing operations -------------------------------------------

    def find_similar(self, content: str, existing: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        candidates = existing if existing is not None else self.db.all_memories()
        for memory in candidates:
            if is_similar(content, memory["content"]):
                return memory
        return None

    def remember(
        self,
        content: Any,
        kind: Any = "fact",
        *,
        source_conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Insert a memory, merging into a near-duplicate when one exists."""
        cleaned = validate_content(content)
        kind_name = validate_kind(kind)
        existing = self.db.all_memories()
        match = self.find_similar(cleaned, existing)
        if match:
            if len(cleaned) > len(match["content"]):
                updated = self.db.update_memory(
                    match["id"],
                    content=cleaned,
                    normalized=normalize(cleaned),
                    kind=kind_name,
                )
                return updated or match
            return self.db.update_memory(match["id"]) or match
        memory = self.db.create_memory(
            kind_name,
            cleaned,
            normalize(cleaned),
            source_conversation_id=source_conversation_id,
        )
        self.db.prune_memories(config.MEMORY_CAP)
        return memory

    def update(
        self,
        memory_id: str,
        *,
        content: Any = None,
        kind: Any = None,
        pinned: Any = None,
    ) -> dict[str, Any] | None:
        fields: dict[str, Any] = {}
        if content is not None:
            cleaned = validate_content(content)
            fields["content"] = cleaned
            fields["normalized"] = normalize(cleaned)
        if kind is not None:
            fields["kind"] = validate_kind(kind)
        if pinned is not None:
            fields["pinned"] = bool(pinned)
        return self.db.update_memory(memory_id, **fields)

    def delete(self, memory_id: str) -> bool:
        return self.db.delete_memory(memory_id)

    def delete_all(self) -> int:
        return self.db.delete_all_memories()

    def count(self) -> int:
        return self.db.count_memories()

    # --- extraction --------------------------------------------------------

    def extract_and_apply(self, conversation_id: str, model: str | None) -> dict[str, int]:
        """Synchronous post-turn extraction. Never raises, bounded by the 12s cap."""
        counts = {"added": 0, "updated": 0, "removed": 0}
        settings = self._settings()
        if not settings.get("memory_enabled", True) or not model:
            return counts
        try:
            raw = self._bounded_extract(conversation_id, model)
            if raw is None:
                return counts
            ops = parse_ops(raw)[:MAX_OPS_PER_TURN]
            if not ops:
                return counts
            return self._apply_ops(ops)
        except Exception:  # extraction must never fail the turn
            log.warning("memory extraction failed for %s", conversation_id, exc_info=True)
            return counts

    def _bounded_extract(self, conversation_id: str, model: str) -> Any:
        messages = self._extraction_messages(conversation_id)
        if not messages:
            return None
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kiwibro-mem")
        try:
            future = executor.submit(
                self.llm.complete_json,
                model=model,
                messages=messages,
                max_tokens=400,
                temperature=0.0,
                read_timeout=int(config.MEMORY_EXTRACTION_TIMEOUT_S),
                retries=0,
            )
            return future.result(timeout=config.MEMORY_EXTRACTION_TIMEOUT_S + 1.0)
        except TimeoutError:
            log.info("memory extraction timed out")
            return None
        except LMStudioError as exc:
            log.debug("memory extraction LLM error: %s", exc.message)
            return None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _extraction_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        recent = self.db.recent_messages(conversation_id, 8)
        lines: list[str] = []
        budget = EXTRACT_TRANSCRIPT_CHARS
        for message in reversed(recent):
            role = message.get("role")
            if role not in ("user", "assistant"):
                continue
            content = (message.get("content") or "").strip()
            if not content:
                continue
            line = f"{'User' if role == 'user' else 'Assistant'}: {content[:1200]}"
            if len(line) > budget and lines:
                break
            lines.append(line)
            budget -= len(line) + 1
        if not lines:
            return []
        lines.reverse()
        existing = self.db.all_memories(40)
        existing_lines = [
            f"{memory['id']} | {memory['kind']} | {memory['content'][:160]}" for memory in existing
        ]
        user_prompt = prompts.MEMORY_USER.format(
            existing="\n".join(existing_lines) if existing_lines else "(none)",
            transcript="\n\n".join(lines),
        )
        return [
            {"role": "system", "content": prompts.MEMORY_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

    def _apply_ops(self, ops: list[dict[str, Any]]) -> dict[str, int]:
        counts = {"added": 0, "updated": 0, "removed": 0}
        existing = self.db.all_memories()
        for op in ops:
            try:
                if op["op"] == "add":
                    content = clean_content(op.get("content"))
                    if not (MIN_CONTENT <= len(content) <= MAX_CONTENT):
                        continue
                    kind = validate_kind(op.get("kind"))
                    duplicate = self.find_similar(content, existing)
                    if duplicate:
                        replacement = content if len(content) > len(duplicate["content"]) else duplicate["content"]
                        self.db.update_memory(
                            duplicate["id"],
                            content=replacement,
                            normalized=normalize(replacement),
                            kind=kind,
                        )
                        counts["updated"] += 1
                    else:
                        created = self.db.create_memory(kind, content, normalize(content))
                        existing.append(created)
                        counts["added"] += 1
                elif op["op"] == "update":
                    memory_id = op.get("id")
                    memory = self.db.get_memory(memory_id) if memory_id else None
                    if not memory:
                        continue
                    content = clean_content(op.get("content")) or memory["content"]
                    if not (MIN_CONTENT <= len(content) <= MAX_CONTENT):
                        continue
                    kind = validate_kind(op.get("kind") or memory["kind"])
                    self.db.update_memory(
                        memory_id,
                        content=content,
                        normalized=normalize(content),
                        kind=kind,
                    )
                    counts["updated"] += 1
                elif op["op"] == "delete":
                    memory_id = op.get("id")
                    if memory_id and self.db.delete_memory(memory_id):
                        existing = [m for m in existing if m["id"] != memory_id]
                        counts["removed"] += 1
            except (ValueError, KeyError):
                continue
        excess = self.db.count_memories() - config.MEMORY_CAP
        if excess > 0:
            counts["removed"] += self.db.prune_memories(config.MEMORY_CAP)
        return counts
