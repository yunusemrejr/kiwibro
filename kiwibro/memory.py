"""Durable memory: extraction, deduplication, retrieval, rendering, user CRUD."""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import Any, Callable

from . import autofacts, config, inference, prompts, terms
from .context import estimate_tokens
from .db import Database
from .llm import LMStudioClient, LMStudioError, TurnAborted, parse_json_loose

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
# An extraction is a fresh ~3k-token prompt plus up to 400 tokens of output: a minute or
# two of every core on a laptop CPU (prefill ~54 tok/s, decode ~7 tok/s with a 4B model).
# So it runs when the user said something about themselves, their work or a lasting
# preference, and otherwise only once a full window has gathered (a catch-all for facts
# the signal misses). Questions and plain requests ("write a function…") are skipped.
_MEMORABLE_RE = re.compile(
    r"\b(?:i|i'm|im|i've|i'd|i'll|my|mine|myself|we|we're|we've|we'll|our|ours|us)\b"
    r"|\b(?:remember|forget|prefer\w*|always|never|from now on|going forward|call me|don't|do not)\b",
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]*")
TARGET_MIN_SCORE = 0.5  # share of a reference's key words a memory must contain to be "the one"
SINGLE_FACT_TERMS = 4  # more key words than this: a fact that bundles several (kept whole)
# Facts about the assistant are not memories about the user (small models save "Kiwibro is…").
_ABOUT_ASSISTANT_RE = re.compile(r"^(?:kiwibro|the assistant|assistant|you are|you're|this ai)\b", re.IGNORECASE)

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def worth_extracting(window: list[dict[str, Any]]) -> bool:
    """True when a user message in ``window`` states something memorable: a sentence (not
    a question) in the first person, or a memory or standing-preference word. Code and
    links are ignored."""
    for message in window:
        if message.get("role") != "user":
            continue
        text = autofacts._URL_RE.sub(" ", autofacts._FENCE_RE.sub(" ", str(message.get("content") or "")))
        for sentence in _SENTENCE_RE.findall(autofacts._INLINE_CODE_RE.sub(" ", text)):
            if not sentence.rstrip().endswith("?") and _MEMORABLE_RE.search(sentence.replace("’", "'")):
                return True
    return False


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


# Single-valued facts every turn benefits from, kept in the canonical wording autofacts writes.
PROFILE_SLOTS = ("name", "reply_language")
_PROFILE_PREFIXES = tuple(autofacts.slot_prefix(slot) for slot in PROFILE_SLOTS)
# The user describing themselves in their own words in the Memory sheet ("I am Yunus, a
# developer from Istanbul…"). Model-written memories never start this way (third_person),
# and such a note answers "who am I?" or "what's my name?" without sharing a word with it.
_SELF_DESCRIPTION_RE = re.compile(r"^(?:i\s+am|i['\u2019]m|my\s+name\s+is|call\s+me)\b", re.IGNORECASE)
SELF_DESCRIPTIONS = 3


def is_profile(memory: dict[str, Any]) -> bool:
    content = str(memory.get("content") or "")
    return content.startswith(_PROFILE_PREFIXES) or bool(_SELF_DESCRIPTION_RE.match(content))


def _memory_keys(memory: dict[str, Any]) -> set[str]:
    """Key terms of a memory (terms.key_terms): what relevance and ranking compare."""
    return terms.key_terms(memory.get("content") or "")


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


def related(a: str, b: str) -> bool:
    """Whether two statements are about the same thing of the same subject: the user
    themself ("User lives in Istanbul" / "User moved to Ankara") or the same person or
    thing of theirs ("User's sister …"). They must share a content word or a concept
    (where someone lives, what they work at), or read alike."""
    if _subject(a) != _subject(b):
        return False

    def words(text: str) -> set[str]:
        found = {w for w in _folded_words(text) if len(w) > 2 and w not in _RELATED_STOP}
        return found | {_CONCEPTS[w] for w in found if w in _CONCEPTS}

    return bool(words(a) & words(b)) or similarity(a, b) >= DEDUPE_SEQ_RATIO


def _subject(text: str) -> str:
    """Who a memory is about: "user" for the user themself, "user s sister" for something of
    theirs (normalize turns "User's" into "user s"), else its first word."""
    words = normalize(text).split()
    if words[:2] == ["user", "s"] or words[:1] == ["users"]:
        rest = words[2:] if words[:2] == ["user", "s"] else words[1:]
        return "user s " + (rest[0] if rest else "")
    return words[0] if words else ""


# Too common to tie two memories together (unlike the framing verbs, which do: "lives").
_RELATED_STOP = frozenset("user users the and for with from that this has have had are was his her their its".split())
# Words that state the same kind of fact differently: "moved to Ankara" replaces "lives in Istanbul".
_CONCEPTS = {
    **dict.fromkeys(
        "live lives living lived moved move moving relocated based resides reside located city hometown".split(),
        "~residence",
    ),
    **dict.fromkeys(
        "work works working worked job employed employer company occupation profession".split(), "~work"
    ),
}


def _stated_at(content: str, window: list[dict[str, Any]]) -> str | None:
    """When the user last said this: the newest window message that grounds it."""
    return next(
        (m.get("created_at") for m in reversed(window)
         if m.get("role") == "user" and grounded(content, str(m.get("content") or ""))),
        None,
    )


def _superseded(
    content: str, stated: str | None, existing: list[dict[str, Any]], *,
    batch: list[tuple[str, str | None]], fresh: set[str],
) -> bool:
    """A learned fact is older news when something said later changed it.

    * a related memory that existed before this extraction changed after the message
      that stated the fact (measured: extraction of "I live in Istanbul" finished after
      the next turn saved "User moved to Ankara", and stored both);
    * a related fact in this same batch was stated later ("I live in Istanbul" … "I moved
      to Ankara") — unless this one also carries other facts, which would be lost
      (measured: a 4B model wrote "User is Yunus, lives in Istanbul and writes Python").
    Memories this batch just created (``fresh``) are never "newer": they are its own facts.
    """
    if not stated:
        return False
    if any(
        memory["id"] not in fresh and (memory.get("updated_at") or "") > stated
        and related(memory["content"], content) and not is_similar(memory["content"], content)
        for memory in existing
    ):
        return True
    single = len({w for w in _folded_words(content) if len(w) > 2 and w not in _FRAMING_WORDS}) <= SINGLE_FACT_TERMS
    return single and any(
        later and later > stated and related(other, content) and not is_similar(other, content)
        for other, later in batch
    )


def _relay(source: Any, target: Any) -> None:
    """Set ``target`` once ``source`` is set (ends when either is)."""
    while not target.wait(0.2):
        if source.is_set():
            target.set()
            return


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

        # Function words in an OR query only add noise; search the content words, longest
        # (most specific) first, plus the words of any everyday topic the query names.
        words = terms.content_words(query)
        add(self.db.search_memories(" ".join(words) if words else query, per_query_limit))
        for token in sorted(words, key=len, reverse=True)[:4]:
            add(self.db.search_memories(token, 12))
        for group in terms.concept_groups(terms.key_terms(query)):
            for start in range(0, len(group), 12):  # an FTS query keeps 12 words
                add(self.db.search_memories(" ".join(group[start:start + 12]), 12))
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
        add(self._profile())
        if (query or "").strip():
            add(self._search_pool(query))
        add(self.db.recently_used_memories(8))
        # Every injected line costs prefill on a small model: beyond the pinned memories and
        # the profile, only memories that share a key term (not a function word) with the
        # message ride along.
        query_keys = terms.key_terms(query or "")
        relevant = [m for m in candidates if query_keys & _memory_keys(m)]
        relevant_ids = {m["id"] for m in relevant}
        kept = [m for m in candidates if m.get("pinned") or is_profile(m) or m["id"] in relevant_ids]
        result = self._rank(kept, query)[:limit]
        # Only a memory that matched counts as used; counting every injected one made the
        # same memories win forever (use_count feeds the ranking).
        used = [m["id"] for m in result if m["id"] in relevant_ids]
        if used:
            try:
                self.db.mark_memories_used(used)
            except Exception:  # pragma: no cover - defensive
                log.warning("could not mark memories as used", exc_info=True)
        return result

    def _profile(self) -> list[dict[str, Any]]:
        """Who the user is and how to reply: useful on every turn (PROFILE_SLOTS, plus the
        user's own self-descriptions)."""
        try:
            memories = self.db.all_memories()
        except Exception:  # pragma: no cover - defensive
            return []
        slots = [m for m in memories if str(m.get("content") or "").startswith(_PROFILE_PREFIXES)]
        own = [m for m in memories if _SELF_DESCRIPTION_RE.match(str(m.get("content") or ""))]
        return slots[: len(PROFILE_SLOTS)] + own[:SELF_DESCRIPTIONS]

    @staticmethod
    def _rank(memories: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
        """Rank by pinned, then key-term coverage, Jaccard, recency.

        Coverage (fraction of *query* tokens present) leads because the FTS
        pool is an OR-match: a memory matching every query term must beat a
        newer one matching a single common word.
        """
        query_tokens = terms.key_terms(query or "")

        def score(memory: dict[str, Any]) -> tuple[int, int, float, float, int, str]:
            tokens = _memory_keys(memory)
            if query_tokens and tokens:
                matched = len(query_tokens & tokens)
                coverage = matched / len(query_tokens)
                jaccard_score = matched / len(query_tokens | tokens)
            else:
                coverage = jaccard_score = 0.0
            uses = min(int(memory.get("use_count") or 0), 50)
            recency = memory.get("last_used_at") or memory.get("updated_at") or ""
            return (1 if memory.get("pinned") else 0, 1 if is_profile(memory) else 0, coverage, jaccard_score, uses,
                    recency)

        return sorted(memories, key=score, reverse=True)

    @staticmethod
    def render_block(memories: list[dict[str, Any]], max_tokens: int = MEMORY_PROMPT_TOKEN_CAP) -> str:
        lines: list[str] = []
        used = 0
        for memory in memories:
            # The user's own first-person notes read as the user's words, never as the assistant's.
            line = f"- {third_person(memory['content'])}"
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
                episodes = list(search_messages(query, conversation_id=conversation_id, limit=max(limit, 8) * 2) or [])
                if conversation_id:
                    # Messages still verbatim in the window are already in view: returning them
                    # spends a small model's context on text it has. Tool results stay
                    # recallable, since the window shows older ones capped.
                    conversation = self.db.get_conversation(conversation_id)
                    if conversation is not None:
                        in_view_from = int(conversation.get("summary_through_seq") or 0) + 1
                        episodes = [m for m in episodes
                                    if int(m.get("seq") or 0) < in_view_from or m.get("role") == "tool"]
                episodes = episodes[: max(limit, 8)]
                if conversation_id:
                    # Then the user's other chats ("what did we decide in the other chat?").
                    episodes += [
                        m for m in search_messages(query, limit=max(limit, 8)) or []
                        if m.get("conversation_id") != conversation_id and m.get("role") != "tool"
                        and not self.is_private(m.get("conversation_id"))
                    ][: max(2, limit // 2)]
            except Exception:  # pragma: no cover - recall must never fail a turn/tool
                log.warning("episodic search failed", exc_info=True)
                episodes = []
            titles: dict[str, str] = {}
            for message in episodes:
                item = {
                    "source": "episode",
                    "id": message["id"],
                    "conversation_id": message.get("conversation_id"),
                    "seq": message.get("seq"),
                    "role": message.get("role"),
                    "content": message.get("content") or "",
                    "created_at": message.get("created_at"),
                }
                other = message.get("conversation_id")
                if conversation_id and other and other != conversation_id:
                    if other not in titles:
                        titles[other] = (self.db.get_conversation(other) or {}).get("title") or "another chat"
                    item["chat"] = titles[other]
                items.append(item)
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
        origin: str | None = None,
    ) -> dict[str, Any]:
        """Insert a memory, merging into a near-duplicate when one exists.

        Without an explicit ``kind`` a merge keeps the existing memory's kind. Memories
        saved from a conversation (the remember tool) are model-written, so first-person
        wording is attributed to the user. ``origin`` defaults to ``chat`` for those and
        ``you`` for memories added in the Memory sheet; a memory the user adds again
        becomes theirs.
        """
        cleaned = validate_content(content)
        if source_conversation_id is not None:
            cleaned = validate_content(third_person(cleaned))
        origin = origin or ("chat" if source_conversation_id is not None else "you")
        kind_name = validate_kind(kind) if kind else None
        match = self.find_similar(cleaned)
        if match:
            claim = {"origin": "you"} if origin == "you" and match.get("origin") != "you" else {}
            if len(cleaned) > len(match["content"]):
                updated = self.db.update_memory(
                    match["id"],
                    content=cleaned,
                    normalized=normalize(cleaned),
                    kind=kind_name or match["kind"],
                    **claim,
                )
                return updated or match
            return self.db.update_memory(match["id"], **claim) or match
        memory = self.db.create_memory(
            kind_name or "fact",
            cleaned,
            normalize(cleaned),
            source_conversation_id=source_conversation_id,
            origin=origin,
        )
        self.db.prune_memories(config.MEMORY_CAP, keep=[memory["id"]])
        return memory

    def capture(self, message: str, *, conversation_id: str | None = None) -> dict[str, int]:
        """Save what the user just said about themselves, before the reply (no model call).

        A small local model often never calls ``remember``, and LLM extraction runs only
        once the user is idle, so the facts autofacts reads with high-precision patterns
        are stored right away and are in memory for the very next turn. A single-valued
        fact (name, location, editor, reply language…) replaces the one already held for
        its slot: the later statement wins. Returns ``{added, updated, removed}``.
        """
        counts = {"added": 0, "updated": 0, "removed": 0}
        if self.is_private(conversation_id):
            return counts
        for fact in autofacts.extract(message):
            try:
                if fact.forget:
                    _memory, outcome = self.forget(fact.content)
                    counts["removed"] += outcome == "deleted"
                    continue
                replaced = self._replace_slot(fact) if fact.slot else None
                if replaced is not None:
                    counts["updated"] += replaced
                    continue
                before = self.db.count_memories()
                self.remember(fact.content, fact.kind, source_conversation_id=conversation_id, origin="learned")
                counts["added" if self.db.count_memories() > before else "updated"] += 1
            except ValueError:
                continue
        return counts

    def is_private(self, conversation_id: str | None) -> bool:
        """A private chat: nothing from it is saved to memory or shown to other chats."""
        if not conversation_id:
            return False
        return bool((self.db.get_conversation(conversation_id) or {}).get("private"))

    def _replace_slot(self, fact: autofacts.Fact) -> bool | None:
        """Point the slot's existing memory at the new value: True when it changed, False when
        it already said that, None when nothing is held for the slot yet."""
        prefix = autofacts.slot_prefix(fact.slot or "")
        held = [m for m in self.db.all_memories() if str(m.get("content") or "").startswith(prefix)]
        if not held:
            return None
        keep, extra = held[0], held[1:]
        changed = normalize(keep["content"]) != normalize(fact.content)
        if changed:
            self.db.update_memory(keep["id"], content=fact.content, normalized=normalize(fact.content), kind=fact.kind)
        for memory in extra:  # one value per slot
            if not memory.get("pinned"):
                self.db.delete_memory(memory["id"])
        return changed

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
            fields["origin"] = "you"  # edited by the user: theirs now, never auto-deleted
        if kind is not None:
            fields["kind"] = validate_kind(kind)
        if pinned is not None:
            fields["pinned"] = bool(pinned)
        return self.db.update_memory(memory_id, **fields)

    def delete(self, memory_id: str) -> bool:
        return self.db.delete_memory(memory_id)

    # --- model-driven edits (the remember tool) ------------------------------

    @staticmethod
    def about_assistant(content: str) -> bool:
        return bool(_ABOUT_ASSISTANT_RE.match((content or "").strip()))

    def find_target(self, reference: Any) -> dict[str, Any] | None:
        """The stored memory a model's reference means: an id, or words from it
        ("lives in Istanbul"). None unless one memory holds most of its key words."""
        text = str(reference or "").strip()
        if not text:
            return None
        if re.fullmatch(r"mem_[0-9a-f]{6,}", text):
            return self.db.get_memory(text)
        terms = {t for t in _tokens(normalize(text)) if len(t) > 2 and t not in _FRAMING_WORDS}
        terms = terms or _tokens(normalize(text))
        best, best_score = None, 0.0
        for memory in self._search_pool(text):
            words = _tokens(memory.get("normalized") or normalize(memory.get("content") or ""))
            # Key words decide ("tea" never means "jazz", however alike the sentences
            # read: "User likes tea" / "User likes jazz" are 0.83 similar); only a
            # near-duplicate counts on wording alone.
            score = len(terms & words) / len(terms) if terms else 0.0
            if is_similar(text, memory.get("content") or ""):
                score = 1.0
            if score > best_score:
                best, best_score = memory, score
        return best if best_score >= TARGET_MIN_SCORE else None

    def revise(
        self, reference: Any, content: Any, kind: Any = None, *, source_conversation_id: str | None = None
    ) -> tuple[dict[str, Any], bool]:
        """Replace an outdated memory with ``content`` (a fact changed or was corrected).
        Returns ``(memory, replaced)``; with no matching memory the fact is saved as new."""
        cleaned = validate_content(third_person(validate_content(content)))
        match = self.find_target(reference) if str(reference or "").strip() else None
        if match is None:
            return self.remember(cleaned, kind, source_conversation_id=source_conversation_id), False
        updated = self.db.update_memory(
            match["id"], content=cleaned, normalized=normalize(cleaned),
            kind=validate_kind(kind) if kind else match["kind"],
        )
        return updated or match, True

    def forget(self, reference: Any) -> tuple[dict[str, Any] | None, str]:
        """Delete the memory ``reference`` means. Returns ``(memory, outcome)`` with outcome
        ``deleted``, ``pinned`` (kept: only the user unpins) or ``missing``."""
        match = self.find_target(reference)
        if match is None:
            return None, "missing"
        if match.get("pinned"):
            return match, "pinned"
        self.db.delete_memory(match["id"])
        return match, "deleted"

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

    def extract_and_apply(
        self, conversation_id: str, model: str | None, *, abort: Any = None
    ) -> dict[str, int]:
        """Synchronous post-turn extraction. Never raises; bounded by a speed-sized budget.

        ``abort`` (a threading.Event) cancels the model call; the result then has
        ``cancelled: 1`` and the watermark is untouched, so nothing is skipped."""
        counts = {"added": 0, "updated": 0, "removed": 0}
        settings = self._settings()
        if not settings.get("memory_enabled", True) or not model:
            return counts
        try:
            if self.is_private(conversation_id):
                # Never read; the watermark moves on so making the chat public later
                # does not learn from what was said while it was private.
                self._advance_watermark(conversation_id, self.db.last_message_seq(conversation_id))
                return counts
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
            full = len(window) >= EXTRACT_WINDOW_MESSAGES or \
                sum(_transcript_cost(m) for m in window) >= EXTRACT_TRANSCRIPT_CHARS * 3 // 4
            if not full and not worth_extracting(window):
                # Questions and requests only: no model call yet. The watermark stays, so a
                # later turn extracts these messages together with whatever it adds.
                return {**counts, "deferred": 1}
            raw, outcome = self._bounded_extract(conversation_id, model, window=window, abort=abort)
            if outcome == "cancelled":
                return {**counts, "cancelled": 1}
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
            return self._apply_ops(ops, transcript, conversation_id=conversation_id, window=window)
        except Exception:  # extraction must never fail the turn
            log.warning("memory extraction failed for %s", conversation_id, exc_info=True)
            return counts

    def _bounded_extract(
        self, conversation_id: str, model: str, window: list[dict[str, Any]] | None = None,
        *, abort: Any = None,
    ) -> tuple[Any, str]:
        """(parsed JSON or None, outcome): outcome is ``ok``, ``failed`` or ``cancelled``."""
        messages = self._extraction_messages(conversation_id, window=window)
        if not messages:
            return None, "failed"
        # Sized from this model's measured speeds: a fixed short timeout means a CPU-only
        # machine never finishes an extraction (the watermark then retries forever).
        budget = inference.SPEEDS.budget_s(
            model,
            prompt_tokens=sum(estimate_tokens(str(m.get("content") or "")) for m in messages),
            max_tokens=EXTRACT_MAX_TOKENS,
            floor=config.MEMORY_EXTRACTION_TIMEOUT_S,
        )
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kiwibro-mem")
        stop = threading.Event()  # set on timeout too, so LM Studio stops working on it
        watcher = None
        if abort is not None:
            watcher = threading.Thread(target=_relay, args=(abort, stop), daemon=True)
            watcher.start()
        try:
            try:
                future = executor.submit(
                    self.llm.complete_json,
                    model=model,
                    messages=messages,
                    max_tokens=EXTRACT_MAX_TOKENS,
                    temperature=0.0,
                    read_timeout=int(budget),
                    retries=0,
                    **({"abort": stop} if abort is not None else {}),
                )
            except RuntimeError:  # the app is shutting down: not this window's failure
                return None, "cancelled"
            raw = future.result(timeout=budget + 1.0)
            return raw, "ok" if raw is not None else "failed"
        except TurnAborted:
            return None, "cancelled" if abort is not None and abort.is_set() else "failed"
        except TimeoutError:
            log.info("memory extraction timed out")
            return None, "failed"
        except LMStudioError as exc:
            log.debug("memory extraction LLM error: %s", exc.message)
            return None, "failed"
        finally:
            stop.set()
            executor.shutdown(wait=False, cancel_futures=True)

    def _relevant_existing_memories(self, transcript_lines: list[str]) -> list[dict[str, Any]]:
        """Memories the extractor might update or delete for this transcript: pinned ones,
        the profile, and those sharing a key term with what the user said.

        Duplicates of new facts are caught in code (``find_similar``), so unrelated
        memories stay out: each one cost prefill (up to ~1k tokens per extraction) and
        invited "updates" of the wrong memory, a mistake small models made live.
        """
        said = " ".join(line[len("User: "):] for line in transcript_lines if line.startswith("User: "))
        keys = terms.key_terms(said)
        seen: set[str] = set()
        picked: list[dict[str, Any]] = []

        def add(memories: list[dict[str, Any]], *, related_only: bool = False) -> None:
            for memory in memories:
                if len(picked) >= EXTRACT_EXISTING_LIMIT or memory["id"] in seen:
                    continue
                if related_only and not keys & _memory_keys(memory):
                    continue
                seen.add(memory["id"])
                picked.append(memory)

        add(self.db.pinned_memories(8))
        add(self._profile())
        if keys:
            add(self._rank(self._search_pool(said[:2000]), said), related_only=True)
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

    def _apply_ops(
        self, ops: list[dict[str, Any]], transcript: str | None = None, *, conversation_id: str | None = None,
        window: list[dict[str, Any]] | None = None,
    ) -> dict[str, int]:
        """Apply extraction ops; with ``transcript``, new facts must be grounded in it, and
        with the ``window`` of messages a fact a newer memory already changed is not added."""
        counts = {"added": 0, "updated": 0, "removed": 0}
        added_ids: list[str] = []
        existing = self.db.all_memories()
        fresh: set[str] = set()  # memories this batch created or rewrote
        # What this batch says, and when the user said it (for "a later statement wins").
        batch = [
            (text, _stated_at(text, window) if window else None)
            for text in (str(op.get("content") or "").strip() for op in ops)
            if text
        ]
        for op in ops:
            target = self.db.get_memory(op["id"]) if op["op"] == "update" and op.get("id") else None
            if op["op"] == "update" and (
                target is None or (op.get("content") and not related(target["content"], str(op["content"])))
            ):
                # Small models "update" invented ids (mem_123), or an unrelated memory they
                # happened to see (measured: "User lives in Istanbul…" overwrote a memory
                # about something else). The content is still a new fact.
                op = {**op, "op": "add"}
            try:
                if op["op"] == "add":
                    content = validate_content(third_person(clean_content(op.get("content"))))
                    if len(content.split()) < 2:
                        continue  # a bare name ("Pamuk") is not a self-contained fact
                    if transcript is not None and not grounded(content, transcript):
                        log.info("memory extraction dropped an ungrounded fact: %s", content[:80])
                        continue
                    if self.about_assistant(content):
                        continue
                    stated = _stated_at(content, window) if window else None
                    if _superseded(content, stated, existing, batch=batch, fresh=fresh):
                        log.info("memory extraction skipped a fact something newer changed: %s", content[:80])
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
                        fresh.add(duplicate["id"])
                        counts["updated"] += 1
                    else:
                        created = self.db.create_memory(
                            kind or "fact", content, normalize(content),
                            source_conversation_id=conversation_id, origin="learned",
                        )
                        existing.append(created)
                        added_ids.append(created["id"])
                        fresh.add(created["id"])
                        counts["added"] += 1
                elif op["op"] == "update":
                    memory_id = op.get("id")
                    memory = self.db.get_memory(memory_id) if memory_id else None
                    if not memory:
                        continue
                    content = validate_content(op.get("content") or memory["content"])
                    if self.about_assistant(content) or (
                        transcript is not None and content != memory["content"] and not grounded(content, transcript)
                    ):
                        continue
                    kind = lenient_kind(op.get("kind")) or memory["kind"]
                    self.db.update_memory(
                        memory_id,
                        content=content,
                        normalized=normalize(content),
                        kind=kind,
                    )
                    fresh.add(memory_id)
                    counts["updated"] += 1
                elif op["op"] == "delete":
                    memory_id = op.get("id")
                    memory = self.db.get_memory(memory_id) if memory_id else None
                    if memory is None or memory.get("pinned") or memory.get("origin") == "you":
                        continue  # pinned and user-added memories are the user's; only they remove them
                    if self.db.delete_memory(memory_id):
                        existing = [m for m in existing if m["id"] != memory_id]
                        counts["removed"] += 1
            except (ValueError, KeyError):
                continue
        excess = self.db.count_memories() - config.MEMORY_CAP
        if excess > 0:
            counts["removed"] += self.db.prune_memories(config.MEMORY_CAP, keep=added_ids)
        return counts
