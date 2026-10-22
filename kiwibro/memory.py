"""Durable memory: extraction, deduplication, retrieval, rendering, user CRUD."""

from __future__ import annotations

import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import Any, Callable

from . import config, inference, prompts
from .context import estimate_tokens
from .db import Database
from .llm import LMStudioClient, LMStudioError, parse_json_loose

log = logging.getLogger(__name__)

DEDUPE_SEQ_RATIO = 0.86
DEDUPE_JACCARD = 0.72
MEMORY_PROMPT_TOKEN_CAP = 800
MAX_OPS_PER_TURN = 5
MIN_CONTENT = 3
MAX_CONTENT = 300
EXTRACT_TRANSCRIPT_CHARS = 9000
EXTRACT_MAX_TOKENS = 400
EXTRACT_EXISTING_CHARS = 4000
EXTRACT_EXISTING_LIMIT = 32
EXTRACT_WINDOW_MESSAGES = 12
# A window whose extraction fails this many times in a row (timeout, error, non-JSON
# reply) is skipped, so one bad window cannot block every later message forever.
EXTRACT_MAX_ATTEMPTS = 2
EXTRACT_LINE_CHARS = 1200

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


def _similar_normalized(na: str, tokens_a: set[str], nb: str, tokens_b: set[str]) -> bool:
    """Similarity on pre-normalised text/token sets (hot dedupe path).

    Semantically identical to ``is_similar`` on the raw strings, but avoids
    re-normalising stored memories on every comparison. Cheap bounds run
    first so ``SequenceMatcher`` only executes when it can still pass:
    its ratio is at most ``2*min(len)/(len_a+len_b)``.
    """
    if not na or not nb:
        return False
    if na == nb:
        return True
    if tokens_a and tokens_b:
        overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        if overlap >= DEDUPE_JACCARD:
            return True
    shorter, longer = min(len(na), len(nb)), max(len(na), len(nb))
    if 2 * shorter / (shorter + longer) < DEDUPE_SEQ_RATIO:
        return False
    return SequenceMatcher(None, na, nb).ratio() >= DEDUPE_SEQ_RATIO


def is_similar(a: str, b: str) -> bool:
    na, nb = normalize(a), normalize(b)
    return _similar_normalized(na, _tokens(na), nb, _tokens(nb))


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


_FIRST_PERSON_RE = re.compile(r"^(i|i'm|i've|i'd|i'll|my|me|we|our)\b", re.IGNORECASE)


def third_person(content: str) -> str:
    """Model-written memories sometimes quote the user ("I use Neovim"); later prompts
    would read that "I" as the assistant, so attribute it explicitly."""
    return f"User said: {content}" if _FIRST_PERSON_RE.match(content) else content


def validate_kind(kind: Any) -> str:
    value = str(kind or "fact").strip().lower()
    if value not in config.MEMORY_KINDS:
        raise ValueError(f"unknown memory kind: {kind}")
    return value


def lenient_kind(kind: Any) -> str | None:
    """A model-written kind: None when absent, "fact" when it is not a known kind."""
    if not kind:
        return None
    value = str(kind).strip().lower()
    return value if value in config.MEMORY_KINDS else "fact"


def parse_ops(raw: Any) -> list[dict[str, Any]]:
    """Parse extraction output defensively into a list of operation dicts."""
    payload = parse_json_loose(raw) if isinstance(raw, str) else raw
    if isinstance(payload, dict):
        # A bare operation (some models drop the {"ops": [...]} wrapper) still counts.
        payload = [payload] if "op" in payload else (
            payload.get("ops") or payload.get("operations") or payload.get("memories")
        )
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
        content = next(
            (item[key] for key in ("content", "memory", "text", "fact") if item.get(key) is not None), None
        )
        if content is not None:
            entry["content"] = content
        if item.get("kind") is not None:
            entry["kind"] = item["kind"]
        ops.append(entry)
    return ops


# Framing words a memory adds around the fact itself ("User prefers …", "User's name is …").
_FRAMING_WORDS = frozenset(
    """user users the and for with from that this has have had is are was his her their its
    name named called prefers prefer likes like dislikes loves hates wants uses works lives
    lived living currently also very really strongly""".split()
)


def _folded_words(text: str) -> set[str]:
    """Lower-cased words with accents removed ("İzmir'de" -> {"izmir", "de"})."""
    decomposed = unicodedata.normalize("NFKD", (text or "").casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return _tokens(normalize(plain))


def grounded(content: str, transcript: str) -> bool:
    """True when the conversation mentions enough of a memory's key words.

    Small models copy the prompt's examples or invent facts ("User prefers dark themes"
    from a chat about cats); a fact the conversation never mentions is not saved.
    Words match on their first five letters ("developer"/"development"), and a third
    of the key words is enough, so a memory written in English about a Turkish chat
    still passes on its names, places and numbers.
    """
    terms = {word for word in _folded_words(content) if len(word) > 2 and word not in _FRAMING_WORDS}
    if not terms:
        return True
    seen = {word[:5] for word in _folded_words(transcript)}
    hits = sum(1 for term in terms if term[:5] in seen)
    return hits * 3 >= len(terms)


def _transcript_cost(message: dict[str, Any]) -> int:
    """Characters the extraction transcript spends on a message (0 when it is skipped)."""
    if message.get("role") not in ("user", "assistant"):
        return 0
    content = (message.get("content") or "").strip()
    if not content:
        return 0
    return len("Assistant: ") + min(len(content), EXTRACT_LINE_CHARS) + 1


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
        self._failed_windows: dict[str, tuple[int, int]] = {}  # conversation -> (through_seq, failures)

    # --- retrieval ---------------------------------------------------------

    def _search_pool(self, query: str, per_query_limit: int = 24) -> list[dict[str, Any]]:
        """Candidate memories for a query: full-query FTS + per-token passes.

        The FTS expression is an OR over query tokens, so one common word
        ("subject", "user") can fill the whole result window with recent
        noise and push the true match out. A single-token pass per
        discriminative term guarantees memories containing that exact term
        enter the pool; coverage ranking then orders the merged pool.
        """
        seen: set[str] = set()
        pool: list[dict[str, Any]] = []

        def add(memories: list[dict[str, Any]]) -> None:
            for memory in memories:
                if memory["id"] not in seen:
                    seen.add(memory["id"])
                    pool.append(memory)

        add(self.db.search_memories(query, per_query_limit))
        tokens = [t for t in _tokens(normalize(query)) if len(t) >= 2]
        for token in tokens[:4]:
            add(self.db.search_memories(token, 12))
        return pool

    def retrieve(self, query: str, limit: int = 12) -> list[dict[str, Any]]:
        """Gather candidates (pinned + search + recently used), then rank them.

        Pinned memories always lead. The rest are ordered by query-token
        overlap, then by most recent use/update — so the token-capped prompt
        block spends its budget on the most relevant memories first instead
        of whichever happened to be updated last.
        """
        seen: set[str] = set()
        candidates: list[dict[str, Any]] = []

        def add(memories: list[dict[str, Any]]) -> None:
            for memory in memories:
                if memory["id"] in seen:
                    continue
                seen.add(memory["id"])
                candidates.append(memory)

        # Fetch a wider pool than the limit so ranking has a choice. (The
        # FTS query is an OR over query tokens, so common words match many
        # rows; coverage scoring below separates true matches from noise.)
        add(self.db.pinned_memories(8))
        if (query or "").strip():
            add(self._search_pool(query))
        add(self.db.recently_used_memories(8))
        result = self._rank(candidates, query)[:limit]
        if result:
            try:
                self.db.mark_memories_used([memory["id"] for memory in result])
            except Exception:  # pragma: no cover - defensive
                log.warning("could not mark memories as used", exc_info=True)
        return result

    @staticmethod
    def _rank(memories: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
        """Rank by pinned, then query-token coverage, Jaccard, recency.

        Coverage (fraction of *query* tokens present) leads because the FTS
        pool is an OR-match: a memory matching every query term must beat a
        newer one matching a single common word.
        """
        query_tokens = _tokens(normalize(query or ""))

        def score(memory: dict[str, Any]) -> tuple[int, float, float, int, str]:
            tokens = _tokens(memory.get("normalized") or normalize(memory.get("content") or ""))
            if query_tokens and tokens:
                matched = len(query_tokens & tokens)
                coverage = matched / len(query_tokens)
                jaccard_score = matched / len(query_tokens | tokens)
            else:
                coverage = jaccard_score = 0.0
            uses = min(int(memory.get("use_count") or 0), 50)
            recency = memory.get("last_used_at") or memory.get("updated_at") or ""
            return (1 if memory.get("pinned") else 0, coverage, jaccard_score, uses, recency)

        return sorted(memories, key=score, reverse=True)

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

    def build_block(
        self,
        query: str,
        enabled: bool = True,
        max_tokens: int = MEMORY_PROMPT_TOKEN_CAP,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Render the prompt memory block within ``max_tokens``.

        The default (800) preserves the v1 behaviour; the context engine
        passes a budget-scaled cap so small-context models are not starved
        by a fixed-size memory block.
        """
        if not enabled:
            return "", []
        memories = self.retrieve(query)
        # The block header costs tokens too; on small budgets it is a real
        # share, so charge it against the cap (keeping a floor so at least
        # the top-ranked memory survives).
        overhead = estimate_tokens(prompts.MEMORY_BLOCK.format(bullets=""))
        bullets = self.render_block(memories, max_tokens=max(30, max_tokens - overhead))
        if not bullets:
            return "", []
        return prompts.MEMORY_BLOCK.format(bullets=bullets), memories

    def recall(
        self,
        query: str,
        conversation_id: str | None = None,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        """Unified recall for tools/UI: durable memories + conversation episodes.

        Returns ranked items shaped ``{"source": "memory"|"episode", ...}``.
        Episodes come from ``Database.search_messages`` when available
        (schema v2); on older databases only memories are returned. This is
        the single recall engine — prompt injection and the ``recall`` tool
        both call it so behaviour cannot drift apart.
        """
        items: list[dict[str, Any]] = []
        memories = self._rank(self._search_pool(query, per_query_limit=max(limit * 4, 32)), query)
        for memory in memories:
            items.append(
                {
                    "source": "memory",
                    "id": memory["id"],
                    "kind": memory["kind"],
                    "content": memory["content"],
                    "pinned": memory["pinned"],
                    "updated_at": memory["updated_at"],
                }
            )
        search_messages = getattr(self.db, "search_messages", None)
        if callable(search_messages):
            try:
                episodes = search_messages(query, conversation_id=conversation_id, limit=max(limit, 8))
            except Exception:  # pragma: no cover - recall must never fail a turn/tool
                log.warning("episodic search failed", exc_info=True)
                episodes = []
            for message in episodes or []:
                items.append(
                    {
                        "source": "episode",
                        "id": message["id"],
                        "conversation_id": message.get("conversation_id"),
                        "seq": message.get("seq"),
                        "role": message.get("role"),
                        "content": message.get("content") or "",
                        "created_at": message.get("created_at"),
                    }
                )
        # Durable facts lead, but reserve up to half the slots for exact
        # episode wording when episodes exist; backfill from either side.
        mem_items = [item for item in items if item["source"] == "memory"]
        ep_items = [item for item in items if item["source"] == "episode"]
        ep_reserve = min(len(ep_items), limit // 2) if ep_items else 0
        mem_taken = mem_items[: limit - ep_reserve]
        merged = list(mem_taken)
        merged.extend(ep_items[: limit - len(merged)])
        merged.extend(mem_items[len(mem_taken):][: limit - len(merged)])
        return merged[:limit]

    # --- user-facing operations -------------------------------------------

    def find_similar(self, content: str, existing: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        normalised = normalize(content)
        tokens = _tokens(normalised)
        if existing is not None:
            candidates = existing
        else:
            # A near-duplicate must share most of its tokens (Jaccard path) or
            # most of its characters (sequence path), so an FTS OR-query over
            # the content is a safe prefilter. Fall back to a full scan only
            # when FTS finds nothing at all (e.g. tokeniser edge cases).
            candidates = self.db.search_memories(content, 50)
            if not candidates:
                candidates = self.db.all_memories()
        for memory in candidates:
            stored = memory.get("normalized") or normalize(memory.get("content") or "")
            if _similar_normalized(normalised, tokens, stored, _tokens(stored)):
                return memory
        return None

    def remember(
        self,
        content: Any,
        kind: Any = None,
        *,
        source_conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Insert a memory, merging into a near-duplicate when one exists.

        Without an explicit ``kind`` a merge keeps the existing memory's kind. Memories
        saved from a conversation (the remember tool) are model-written, so first-person
        wording is attributed to the user.
        """
        cleaned = validate_content(content)
        if source_conversation_id is not None:
            cleaned = validate_content(third_person(cleaned))
        kind_name = validate_kind(kind) if kind else None
        match = self.find_similar(cleaned)
        if match:
            if len(cleaned) > len(match["content"]):
                updated = self.db.update_memory(
                    match["id"],
                    content=cleaned,
                    normalized=normalize(cleaned),
                    kind=kind_name or match["kind"],
                )
                return updated or match
            return self.db.update_memory(match["id"]) or match
        memory = self.db.create_memory(
            kind_name or "fact",
            cleaned,
            normalize(cleaned),
            source_conversation_id=source_conversation_id,
        )
        self.db.prune_memories(config.MEMORY_CAP, keep=[memory["id"]])
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

    def _pending_window(self, conversation_id: str) -> tuple[list[dict[str, Any]], int]:
        """Oldest un-extracted messages (bounded) + the seq they cover.

        Watermark contract: ``extracted_through_seq`` advances only after a
        successful extraction response, so a timed-out or failed attempt is
        retried on a later turn instead of silently skipping those facts
        forever (the v1 last-8-messages window lost them). Pre-v2 chats
        start near the tail — v1 already extracted turn-by-turn as they
        happened — rather than re-reading ancient history.
        """
        conversation = self.db.get_conversation(conversation_id)
        if conversation is None:
            return [], 0
        watermark = int(conversation.get("extracted_through_seq") or 0)
        last_seq = self.db.last_message_seq(conversation_id)
        if watermark <= 0 and last_seq > EXTRACT_WINDOW_MESSAGES:
            watermark = last_seq - EXTRACT_WINDOW_MESSAGES
        pending = self.db.list_messages(
            conversation_id, after_seq=watermark, limit=EXTRACT_WINDOW_MESSAGES * 2
        )
        window: list[dict[str, Any]] = []
        spent = 0
        for message in pending:
            # Charge exactly what the transcript builder spends on this message, and stop
            # *before* the budget is exceeded: the builder drops whatever does not fit,
            # and a dropped message would be skipped for good once the watermark advances.
            cost = _transcript_cost(message)
            if window and spent + cost > EXTRACT_TRANSCRIPT_CHARS:
                break
            window.append(message)
            spent += cost
            if len(window) >= EXTRACT_WINDOW_MESSAGES:
                break
        through = window[-1]["seq"] if window else watermark
        return window, through

    def _advance_watermark(self, conversation_id: str, through_seq: int) -> None:
        try:
            current = self.db.get_conversation(conversation_id)
            if current and through_seq > int(current.get("extracted_through_seq") or 0):
                self.db.update_conversation(
                    conversation_id, extracted_through_seq=through_seq, touch=False
                )
        except Exception:  # watermark bookkeeping must never fail a turn
            log.warning("could not advance extraction watermark", exc_info=True)

    def extract_and_apply(self, conversation_id: str, model: str | None) -> dict[str, int]:
        """Synchronous post-turn extraction. Never raises; bounded by a speed-sized budget."""
        counts = {"added": 0, "updated": 0, "removed": 0}
        settings = self._settings()
        if not settings.get("memory_enabled", True) or not model:
            return counts
        try:
            window, through_seq = self._pending_window(conversation_id)
            if not window:
                return counts
            has_user_text = any(
                m.get("role") == "user" and len((m.get("content") or "").strip()) >= 3
                for m in window
            )
            if not has_user_text:
                # Nothing learnable (tool chatter only): advance without an LLM call.
                self._advance_watermark(conversation_id, through_seq)
                return counts
            raw = self._bounded_extract(conversation_id, model, window=window)
            if raw is None:
                last_seq, failures = self._failed_windows.get(conversation_id, (through_seq, 0))
                failures = failures + 1 if last_seq == through_seq else 1
                if failures < EXTRACT_MAX_ATTEMPTS:
                    self._failed_windows[conversation_id] = (through_seq, failures)
                    return counts  # watermark unchanged: retried next turn
                log.info("memory extraction skipped a window after %d failed attempts", failures)
                self._failed_windows.pop(conversation_id, None)
                self._advance_watermark(conversation_id, through_seq)
                return counts
            self._failed_windows.pop(conversation_id, None)
            self._advance_watermark(conversation_id, through_seq)
            ops = parse_ops(raw)[:MAX_OPS_PER_TURN]
            if not ops:
                return counts
            transcript = "\n".join(str(message.get("content") or "") for message in window)
            return self._apply_ops(ops, transcript)
        except Exception:  # extraction must never fail the turn
            log.warning("memory extraction failed for %s", conversation_id, exc_info=True)
            return counts

    def _bounded_extract(
        self, conversation_id: str, model: str, window: list[dict[str, Any]] | None = None
    ) -> Any:
        messages = self._extraction_messages(conversation_id, window=window)
        if not messages:
            return None
        # Sized from this model's measured speeds: a fixed short timeout means a CPU-only
        # machine never finishes an extraction (the watermark then retries forever).
        budget = inference.SPEEDS.budget_s(
            model,
            prompt_tokens=sum(estimate_tokens(str(m.get("content") or "")) for m in messages),
            max_tokens=EXTRACT_MAX_TOKENS,
            floor=config.MEMORY_EXTRACTION_TIMEOUT_S,
        )
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kiwibro-mem")
        try:
            future = executor.submit(
                self.llm.complete_json,
                model=model,
                messages=messages,
                max_tokens=EXTRACT_MAX_TOKENS,
                temperature=0.0,
                read_timeout=int(budget),
                retries=0,
            )
            return future.result(timeout=budget + 1.0)
        except TimeoutError:
            log.info("memory extraction timed out")
            return None
        except LMStudioError as exc:
            log.debug("memory extraction LLM error: %s", exc.message)
            return None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _relevant_existing_memories(self, transcript_lines: list[str]) -> list[dict[str, Any]]:
        """Memories the extractor might update/delete for this transcript.

        Pinned first, then FTS matches against the transcript text, then
        recently used — deduped and capped. Sending only relevant memories
        (instead of the latest 40) both improves dedupe decisions and keeps
        the extraction prompt viable on small-context models.
        """
        seen: set[str] = set()
        picked: list[dict[str, Any]] = []

        def add(memories: list[dict[str, Any]]) -> None:
            for memory in memories:
                if len(picked) >= EXTRACT_EXISTING_LIMIT or memory["id"] in seen:
                    continue
                seen.add(memory["id"])
                picked.append(memory)

        add(self.db.pinned_memories(8))
        query = " ".join(transcript_lines)[:600]
        if query.strip():
            add(self.db.search_memories(query, 20))
        add(self.db.recently_used_memories(8))
        if len(picked) < EXTRACT_EXISTING_LIMIT:
            add(self.db.all_memories(EXTRACT_EXISTING_LIMIT))
        return picked

    def _extraction_messages(
        self, conversation_id: str, window: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        recent = window if window is not None else self._pending_window(conversation_id)[0]
        lines: list[str] = []
        budget = EXTRACT_TRANSCRIPT_CHARS
        for message in reversed(recent):
            role = message.get("role")
            if role not in ("user", "assistant"):
                continue
            content = (message.get("content") or "").strip()
            if not content:
                continue
            line = f"{'User' if role == 'user' else 'Assistant'}: {content[:EXTRACT_LINE_CHARS]}"
            if len(line) > budget and lines:
                break
            lines.append(line)
            budget -= len(line) + 1
        if not lines:
            return []
        lines.reverse()
        existing = self._relevant_existing_memories(lines)
        existing_lines: list[str] = []
        spent = 0
        for memory in existing:
            line = f"{memory['id']} | {memory['kind']} | {memory['content'][:160]}"
            # Keep the utility prompt small enough for small-context models:
            # the dedupe decision needs the *relevant* memories, not all 40
            # most recent ones (~2k tokens before the transcript even starts).
            if spent + len(line) > EXTRACT_EXISTING_CHARS and existing_lines:
                break
            existing_lines.append(line)
            spent += len(line) + 1
        user_prompt = prompts.MEMORY_USER.format(
            existing="\n".join(existing_lines) if existing_lines else "(none)",
            transcript="\n\n".join(lines),
        )
        return [
            {"role": "system", "content": prompts.MEMORY_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

    def _apply_ops(self, ops: list[dict[str, Any]], transcript: str | None = None) -> dict[str, int]:
        """Apply extraction ops; with ``transcript``, new facts must be grounded in it."""
        counts = {"added": 0, "updated": 0, "removed": 0}
        added_ids: list[str] = []
        existing = self.db.all_memories()
        for op in ops:
            if op["op"] == "update" and not (op.get("id") and self.db.get_memory(op["id"])):
                # Small models "update" invented ids (mem_123) when nothing exists yet;
                # the content is still a new fact.
                op = {**op, "op": "add"}
            try:
                if op["op"] == "add":
                    content = validate_content(third_person(clean_content(op.get("content"))))
                    if len(content.split()) < 2:
                        continue  # a bare name ("Pamuk") is not a self-contained fact
                    if transcript is not None and not grounded(content, transcript):
                        log.info("memory extraction dropped an ungrounded fact: %s", content[:80])
                        continue
                    kind = lenient_kind(op.get("kind"))
                    duplicate = self.find_similar(content, existing)
                    if duplicate:
                        replacement = content if len(content) > len(duplicate["content"]) else duplicate["content"]
                        self.db.update_memory(
                            duplicate["id"],
                            content=replacement,
                            normalized=normalize(replacement),
                            kind=kind or duplicate["kind"],
                        )
                        counts["updated"] += 1
                    else:
                        created = self.db.create_memory(kind or "fact", content, normalize(content))
                        existing.append(created)
                        added_ids.append(created["id"])
                        counts["added"] += 1
                elif op["op"] == "update":
                    memory_id = op.get("id")
                    memory = self.db.get_memory(memory_id) if memory_id else None
                    if not memory:
                        continue
                    content = validate_content(op.get("content") or memory["content"])
                    kind = lenient_kind(op.get("kind")) or memory["kind"]
                    self.db.update_memory(
                        memory_id,
                        content=content,
                        normalized=normalize(content),
                        kind=kind,
                    )
                    counts["updated"] += 1
                elif op["op"] == "delete":
                    memory_id = op.get("id")
                    memory = self.db.get_memory(memory_id) if memory_id else None
                    if memory is None or memory.get("pinned"):
                        continue  # pinned memories are the user's; only the user removes them
                    if self.db.delete_memory(memory_id):
                        existing = [m for m in existing if m["id"] != memory_id]
                        counts["removed"] += 1
            except (ValueError, KeyError):
                continue
        excess = self.db.count_memories() - config.MEMORY_CAP
        if excess > 0:
            counts["removed"] += self.db.prune_memories(config.MEMORY_CAP, keep=added_ids)
        return counts
