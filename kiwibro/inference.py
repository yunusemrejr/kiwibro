"""Local-inference engine: how Kiwibro spends a laptop's compute.

On CPU-only machines prompt prefill dominates latency. Measured on a Ryzen 5
7530U with a 4B model in LM Studio 0.4.26: a 4,052-token prompt took 124 s
cold and 1.3 s when LM Studio could reuse its KV cache for the prefix; moving
one changing line (the clock) to the front of the prompt cost the full 135 s
again. Thinking is the other big cost: "say hi" spent 200 reasoning tokens
(28 s) with default thinking and 2 s with ``reasoning_effort: "none"``.

So this module owns three small, pure policies:

* **Prefix stability.** Message 0 and every message before the newest user
  turn must stay byte-identical between turns; whatever changes per turn
  (``time_line``, retrieved memories, recalled episodes, the directive ledger)
  rides on the newest user message via ``context.place_turn_context``.
  ``PrefixTracker`` measures how well that holds.
* **Reasoning control** (``turn_reasoning`` → ``reasoning_payload``): one
  decision per turn, changed between turns only while a re-prefill is cheap.
* **Model profiles** (``model_profile``): size, family, tier and what a model
  can do, merged from LM Studio metadata and the model id.

Plus per-round telemetry (``RoundTimer``). Nothing here performs I/O;
``PrefixTracker`` only keeps request digests.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping

from . import capabilities

# --- model profiles ------------------------------------------------------------

TIERS = (("tiny", 1.5), ("small", 5.0), ("medium", 15.0))  # by active params, billions
_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmbt])?", re.IGNORECASE)
_MOE_RE = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)b-a(\d+(?:\.\d+)?)b(?![a-z0-9])", re.IGNORECASE)
_ID_PARAMS_RE = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)([bm])(?![a-z0-9])", re.IGNORECASE)
_UNIT = {"k": 1e-6, "m": 1e-3, "b": 1.0, "t": 1e3}


@dataclass(frozen=True)
class ModelProfile:
    """What Kiwibro knows about one installed model (server facts first, id parsing second)."""

    id: str
    family: str
    params_b: float | None
    active_params_b: float | None
    tier: str
    tools: bool
    vision: bool
    reasoning: tuple[str, ...]
    reasoning_default: str | None
    context: int
    max_context: int
    quant_bits: float | None
    size_bytes: int | None
    loaded: bool

    @property
    def is_small(self) -> bool:
        if self.tier in ("tiny", "small"):
            return True
        return self.tier == "unknown" and (self.quant_bits or 99) <= 3

    @property
    def reasoning_toggle(self) -> bool:
        """True when LM Studio lets requests switch thinking off for this model."""
        return "off" in self.reasoning

    @property
    def cost(self) -> float:
        """Resident-size ranking key (bytes); unknown sizes rank last."""
        if self.size_bytes:
            return float(self.size_bytes)
        if self.params_b:
            return self.params_b * 1e9 * (self.quant_bits or 4.5) / 8
        return float("inf")


def parse_params(text: Any) -> float | None:
    """'4.4B' -> 4.4, '773M' -> 0.773, '27B' -> 27.0 (LM Studio ``params_string``)."""
    if not isinstance(text, str):
        return None
    match = _SIZE_RE.match(text)
    if not match:
        return None
    value = float(match.group(1)) * _UNIT[(match.group(2) or "b").lower()]
    return value if 0.01 <= value <= 5000 else None


def params_from_id(model_id: str) -> tuple[float | None, float | None]:
    """(total, active) parameters in billions parsed from a model id."""
    moe = _MOE_RE.search(model_id or "")
    if moe:
        return float(moe.group(1)), float(moe.group(2))
    best: float | None = None
    for match in _ID_PARAMS_RE.finditer(model_id or ""):
        value = float(match.group(1)) * (1e-3 if match.group(2).lower() == "m" else 1.0)
        if 0.01 <= value <= 5000 and (best is None or value > best):
            best = value
    return best, best


def _tier(active: float | None) -> str:
    if active is None:
        return "unknown"
    for name, limit in TIERS:
        if active < limit:
            return name
    return "large"


def _family(info: Mapping[str, Any], model_id: str) -> str:
    arch = str(info.get("arch") or info.get("architecture") or "").lower()
    match = re.match(r"[a-z]+", arch)
    if match:
        return match.group(0)
    name = model_id.rsplit("/", 1)[-1].lower()
    match = re.match(r"[a-z]+", name)
    return match.group(0) if match else "unknown"


def _reasoning(info: Mapping[str, Any]) -> tuple[tuple[str, ...], str | None]:
    raw = info.get("reasoning")
    caps = info.get("capabilities")
    if raw is None and isinstance(caps, Mapping):
        raw = caps.get("reasoning")
    if isinstance(raw, Mapping):
        options = raw.get("allowed_options") or ()
        default = raw.get("default")
    else:
        options, default = raw or (), info.get("reasoning_default")
    if not isinstance(options, (list, tuple)):
        return (), None
    return tuple(str(item) for item in options), (str(default) if default else None)


def _flag(info: Mapping[str, Any], *names: str) -> bool:
    caps = info.get("capabilities")
    for name in names:
        if info.get(name):
            return True
        if isinstance(caps, Mapping) and caps.get(name):
            return True
        if isinstance(caps, (list, tuple)) and name in caps:
            return True
    return False


def _int_or(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def model_profile(model_id: str, model_info: Mapping[str, Any] | None = None) -> ModelProfile:
    """Profile a model from Kiwibro's model record, a raw LM Studio entry, or just its id."""
    info: Mapping[str, Any] = model_info or {}
    model_id = str(model_id or info.get("id") or info.get("key") or "")
    total = parse_params(info.get("params") or info.get("params_string"))
    if total is None and isinstance(info.get("params_b"), (int, float)):
        total = float(info["params_b"])
    id_total, id_active = params_from_id(model_id)
    if total is None:
        total = id_total
    active = id_active if id_active is not None and id_total == total else total
    quant = info.get("quantization")
    if isinstance(quant, Mapping):
        bits = quant.get("bits_per_weight")
        quant_bits = float(bits) if isinstance(bits, (int, float)) else capabilities.quant_bits(quant.get("name"))
    else:
        quant_bits = capabilities.quant_bits(str(quant or ""))
    reasoning, reasoning_default = _reasoning(info)
    loaded = info.get("state") == "loaded" or bool(info.get("loaded_instances"))
    max_context = _int_or(info.get("max_context") or info.get("max_context_length"), 0)
    instances = info.get("loaded_instances") or [{}]
    loaded_config = (instances[0] or {}).get("config") or {} if isinstance(instances, list) else {}
    context = _int_or(
        info.get("context") or info.get("loaded_context_length") or loaded_config.get("context_length"),
        max_context or 32768,
    )
    size = info.get("size_bytes")
    return ModelProfile(
        id=model_id,
        family=_family(info, model_id),
        params_b=total,
        active_params_b=active,
        tier=_tier(active),
        tools=_flag(info, "tools", "tool_use", "trained_for_tool_use"),
        vision=info.get("type") == "vlm" or _flag(info, "vision"),
        reasoning=reasoning,
        reasoning_default=reasoning_default,
        context=context,
        max_context=max_context or context,
        quant_bits=quant_bits,
        size_bytes=int(size) if isinstance(size, (int, float)) and size > 0 else None,
        loaded=loaded,
    )


# --- reasoning control -----------------------------------------------------------

_THINK_HINTS = re.compile(
    r"\b(why|how (?:do|does|can|should|would)|explain|prove|derive|plan|design|debug|fix|refactor|"
    r"implement|optimi[sz]e|analy[sz]e|compare|calculate|solve|step[- ]by[- ]step|architecture|"
    r"algorithm|research|evaluate|trade-?offs?|investigate|diagnose)\b",
    re.IGNORECASE,
)
# "17*23", "2^10", "5 - 3" — but not dates or version strings such as 2026-10-22.
_ARITHMETIC = re.compile(r"\d\s*[+*/^%×÷]\s*\d|\d\s+-\s+\d")


# Chat templates render the thinking switch near the top of the prompt, so flipping
# it re-prefills the whole conversation (measured on CPU: a ~91% matching prompt
# took 90 s, the same as cold). Auto mode only changes its mind while that is cheap.
SWITCH_PREFILL_TOKENS = 1500


def wants_reasoning(mode: str | None, message: str = "") -> bool:
    """Whether a request deserves thinking: ``on``/``off`` are absolute; ``auto`` thinks for
    non-trivial requests (long, code, maths, analysis words) and answers small talk directly."""
    if mode == "on":
        return True
    if mode == "off":
        return False
    text = (message or "").strip()
    return len(text) >= 280 or "```" in text or bool(_THINK_HINTS.search(text) or _ARITHMETIC.search(text))


def turn_reasoning(mode: str | None, message: str, *, previous: bool | None, prompt_tokens: int) -> bool:
    """The thinking decision for a whole turn (every round of it uses the same one).

    In ``auto`` mode a change from the conversation's previous decision is only
    made while the prompt is short enough that the re-prefill it causes is cheap.
    """
    wanted = wants_reasoning(mode, message)
    if mode in ("on", "off") or previous is None or wanted == previous:
        return wanted
    return wanted if prompt_tokens <= SWITCH_PREFILL_TOKENS else previous


# Utility and helper calls never think where the model allows it. Sent even when LM
# Studio lists no reasoning switch (harmless there). Some models cannot be stopped:
# qwen3.8-27b reasons on every request whatever the flag or a /no_think prompt says,
# so helper calls keep enough max_tokens for reasoning plus the answer.
NO_THINKING = {"reasoning_effort": "none"}


def reasoning_payload(profile: ModelProfile, *, think: bool) -> dict[str, Any]:
    """LM Studio request fields for a thinking decision; empty when the model has no toggle."""
    if not profile.reasoning_toggle:
        return {}
    if think:
        return {"reasoning_effort": "medium"} if profile.reasoning_default == "off" else {}
    return {"reasoning_effort": "none"}


# --- per-turn context ---------------------------------------------------------------


def time_line(now: datetime | None = None) -> str:
    """The clock for the per-turn note (never message 0: it changes every minute)."""
    moment = (now or datetime.now()).astimezone()
    return "Local time: " + moment.strftime("%A, %d %B %Y, %H:%M %Z")


# --- prefix reuse telemetry --------------------------------------------------------


def _digest(message: Mapping[str, Any], salt: str = "") -> str:
    encoded = json.dumps(message, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.blake2b((salt + encoded).encode("utf-8"), digest_size=12).hexdigest()


def _default_tokens(message: Mapping[str, Any]) -> int:
    from .context import message_tokens  # local import: context depends on this module's helpers

    return message_tokens(dict(message))


class PrefixTracker:
    """Estimates how much of each request LM Studio can serve from its KV cache.

    LM Studio keeps one cache per parallel slot (4 by default; verified that a
    background request does not evict a chat's warm prefix) and reuses the
    longest matching token prefix. This tracks message-level digests of the
    last few requests per model and reports the longest shared leading run, in
    estimated tokens. It is an estimate: the server's tokens are not visible.
    """

    def __init__(self, slots: int = 4, token_fn: Callable[[Mapping[str, Any]], int] | None = None) -> None:
        self._slots = max(1, slots)
        self._token_fn = token_fn or _default_tokens
        self._recent: dict[str, deque[list[str]]] = {}
        self._lock = threading.Lock()

    def observe(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, int]:
        # Templates render the tool list and switches such as reasoning into the
        # system block, so they belong to message 0.
        salt = _digest({"tools": tools, "options": dict(options or {})}) if tools or options else ""
        digests = [_digest(message, salt if index == 0 else "") for index, message in enumerate(messages)]
        tokens = [self._token_fn(message) for message in messages]
        with self._lock:
            history = self._recent.setdefault(model, deque(maxlen=self._slots))
            shared, match_at = 0, None
            for position, previous in enumerate(history):
                run = 0
                for mine, theirs in zip(digests, previous):
                    if mine != theirs:
                        break
                    run += 1
                if run > shared:
                    shared, match_at = run, position
            if match_at is not None and shared == len(history[match_at]):
                del history[match_at]  # same conversation grew: replace its slot
            history.append(digests)
        total = sum(tokens)
        reused = sum(tokens[:shared])
        return {
            "prompt_tokens_est": total,
            "reused_tokens_est": reused,
            "reuse_percent": round(100 * reused / total) if total else 0,
        }

    def forget(self, model: str | None = None) -> None:
        with self._lock:
            if model is None:
                self._recent.clear()
            else:
                self._recent.pop(model, None)


# --- degeneration guard -----------------------------------------------------------
#
# Heavily quantized small models sometimes collapse mid-reply into filler or a loop
# ("...", blank lines, the same phrase over and over) and burn the whole output budget:
# minutes of a laptop CPU for nothing. The agent watches the streamed tail with
# `is_degenerate`, cuts the stream off and retries once with a corrective nudge.

_DEGENERATE_WINDOW = 300


class DegenerateOutput(Exception):
    """Raised from a stream callback when the reply has collapsed into repetition.

    ``circling`` marks a deliberation loop (nothing worth keeping); otherwise the reply
    started well and then repeated itself, and its good start is kept and continued."""

    def __init__(self, circling: bool = False) -> None:
        super().__init__("circling" if circling else "repeating")
        self.circling = circling


class Overthinking(Exception):
    """Raised from a stream callback when a reply that should act keeps thinking instead."""


# Reasoning a request may stream before it acts, when acting is what is left (a planned
# tool step, or a round after tool results). 30 acting rounds measured live thought a
# median 440 characters (p90 1,640, max 3,540); a 4B model once drafted the same file in
# its thinking for 20 minutes (29,500 characters) and never made the call. Past the
# budget the request is retried with thinking off: one re-prefill (the switch renders at
# the top of the template: 47 s for 1.6k tokens, 190 s for a 6k-token chat) instead of
# an unbounded think.
THINK_BUDGET_CHARS = 4000


# Anti-repetition sampling for the one request that continues a collapsed reply (LM Studio
# honours both; sampling does not touch the prompt, so the KV-cached prefix still applies).
ANTI_REPEAT = {"repeat_penalty": 1.15, "frequency_penalty": 0.3}
LOOP_MAX_BLOCK = 3  # lines in the repeating unit ("**The fix**" + its bullet is two)
_FENCE = re.compile(r"^\s*(?:```|~~~)")
_LINE_MARKER = re.compile(r"^(?:[#>*\-+•]+|\d{1,3}[.)])\s*")


def _line_key(line: str) -> str:
    """A line as a loop sees it: markdown markers, emphasis, case and end punctuation ignored."""
    text = _LINE_MARKER.sub("", re.sub(r"[*_`]+", "", line.strip().lower()))
    return " ".join(text.split()).rstrip(" .;:,!")


def loop_start(text: str, *, repeats: int = 3) -> int | None:
    """Offset in ``text`` where it began repeating a block of up to three lines, keeping
    the block's first copy; None when it does not end in such a loop.

    Small models collapse into "**The fix**\\n- The fix is to update PHP…" over and over,
    sometimes with small wording changes, which exact-period checks miss. The tail counts
    as a loop when its last lines use at most three distinct lines, at least six of them,
    one at least ``repeats`` times. Lists, FAQs and code (fenced blocks are skipped) never
    fit: their lines differ. Trimming text already known to have collapsed uses
    ``repeats=2`` (the stream it was detected on ran one chunk ahead of it)."""
    lines: list[tuple[int, str]] = []
    offset, fenced = 0, False
    complete = text[: text.rfind("\n") + 1]  # a line still streaming is not a line yet
    for raw in complete.splitlines(keepends=True):
        if _FENCE.match(raw):
            fenced = not fenced
        elif not fenced and sum(ch.isalnum() for ch in raw) >= 3:
            lines.append((offset, _line_key(raw)))
        offset += len(raw)
    distinct: set[str] = set()
    start = len(lines)
    for index in range(len(lines) - 1, -1, -1):
        key = lines[index][1]
        if key not in distinct and len(distinct) == LOOP_MAX_BLOCK:
            break
        distinct.add(key)
        start = index
    tail = [key for _offset, key in lines[start:]]
    if len(tail) < 2 * repeats or max(tail.count(key) for key in distinct) < repeats:
        return None
    if sum(ch.isalnum() for key in distinct for ch in key) < 12:  # "ok"/"}" noise, not a loop
        return None
    seen: set[str] = set()
    for line_offset, key in lines[start:]:
        if key in seen:
            return line_offset
        seen.add(key)
    return None


def is_degenerate(tail: str) -> bool:
    """True when the streamed tail is near content-free, one unit repeated end to end, the
    same sentence line after line, or a short block of lines repeated (:func:`loop_start`).
    Thresholds need several repetitions, so ordinary prose and code (even nested closing
    braces) never trip it."""
    if loop_start(tail[-3000:]) is not None:
        return True
    window = tail[-_DEGENERATE_WINDOW:]
    if len(window) < _DEGENERATE_WINDOW:
        return False
    if sum(ch.isalnum() for ch in window) < 0.04 * len(window):
        return True
    long = tail[-2 * _DEGENERATE_WINDOW:]
    for period in range(1, 151 if len(long) == 2 * _DEGENERATE_WINDOW else 41):
        source = window if period <= 40 else long  # longer units must repeat 4+ times
        unit = source[-period:]
        if unit.strip() and (unit * (len(source) // period + 1))[-len(source):] == source:
            return True
    lines = [line.strip() for line in tail[-3000:].splitlines() if sum(ch.isalnum() for ch in line) >= 3]
    return len(lines) >= 12 and len(set(lines[-12:])) == 1


_HEDGES = re.compile(
    r"\b(?:no|wait|hmm+|actually|maybe|perhaps|confus\w*|ambiguous|re-?read|reconsider|i think|"
    r"let me think|i'?m stuck|or not)\b|\?",
    re.I,
)


def is_looping(tail: str) -> bool:
    """True when a reply circles: the same phrases keep coming back while it hedges.

    Catches deliberation loops that never repeat verbatim ("so the content is 'kiwi\\n'?
    No…", seen live on a 4B model for 7,600 characters). Calibrated on real output: such
    loops reach ≥0.15 five-word-phrase recurrence at 5–8 hedges per 100 words, while
    documentation scores ≈0 recurrence and ≤3 hedges, and templated lists have no hedges.
    Code blocks are ignored (code repeats legitimately)."""
    text = re.sub(r"```.*?```", " ", tail[-3000:], flags=re.S)
    words = re.findall(r"\w+", text.lower())
    if len(words) < 300:
        return False
    half = len(words) // 2
    earlier = {tuple(words[i : i + 5]) for i in range(half - 4)}
    recent = [tuple(words[i : i + 5]) for i in range(half, len(words) - 4)]
    recurrence = sum(shingle in earlier for shingle in recent) / max(1, len(recent))
    return recurrence >= 0.15 and 100 * len(_HEDGES.findall(text)) / len(words) >= 4.0


def trim_degenerate(text: str) -> str:
    """The reply without its collapsed tail (trailing filler or repeated paragraphs); a
    repeated block of lines keeps only its first copy."""
    start = loop_start(text, repeats=2)
    if start is not None:
        text = text[:start]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    while paragraphs and (
        sum(ch.isalnum() for ch in paragraphs[-1]) < 0.3 * len(paragraphs[-1].strip())
        or (len(paragraphs) > 1 and paragraphs[-1].strip() == paragraphs[-2].strip())
        or is_degenerate(paragraphs[-1])
    ):
        paragraphs.pop()
    return "\n\n".join(p.strip() for p in paragraphs)


class SpeedBook:
    """Measured prefill and decode speed per model (moving average of streamed rounds).

    Background calls need timeouts that fit the machine: a fixed 8 s is plenty on
    a GPU and hopeless on a laptop CPU (~30 tok/s prefill, 4–8 tok/s decode), where
    memory extraction then silently never finishes. Until a model has been
    measured, pessimistic CPU defaults apply.
    """

    DEFAULT_PREFILL_TPS = 30.0
    DEFAULT_DECODE_TPS = 4.0
    _ALPHA = 0.3

    def __init__(self) -> None:
        self._prefill: dict[str, float] = {}
        self._decode: dict[str, float] = {}
        self._lock = threading.Lock()

    def note(self, model: str, *, new_tokens: int, ttft_ms: int | None, tokens_per_s: float | None) -> None:
        """Fold one round in: `new_tokens` were prefilled (not served from cache) before the first token."""
        with self._lock:
            if ttft_ms and new_tokens >= 50:
                self._update(self._prefill, model, new_tokens / max(ttft_ms / 1000.0, 0.05))
            if tokens_per_s:
                self._update(self._decode, model, float(tokens_per_s))

    def _update(self, book: dict[str, float], model: str, value: float) -> None:
        book[model] = value if model not in book else (1 - self._ALPHA) * book[model] + self._ALPHA * value

    def budget_s(self, model: str, *, prompt_tokens: int, max_tokens: int,
                 floor: float = 8.0, ceiling: float = 180.0) -> float:
        """Seconds a fresh (uncached) request of this size may reasonably take, with headroom."""
        with self._lock:
            prefill = self._prefill.get(model, self.DEFAULT_PREFILL_TPS)
            decode = self._decode.get(model, self.DEFAULT_DECODE_TPS)
        estimate = prompt_tokens / prefill + max_tokens / decode
        return max(floor, min(ceiling, 1.25 * estimate + 2.0))


SPEEDS = SpeedBook()


PREFIX_TRACKER = PrefixTracker()


# --- per-round timing ----------------------------------------------------------------


class RoundTimer:
    """Time-to-first-token and decode speed for one streamed round (monotonic clock)."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        import time

        self._clock = clock or time.monotonic
        self.started = self._clock()
        self.first: float | None = None
        self.ended: float | None = None

    def mark_first(self) -> None:
        if self.first is None:
            self.first = self._clock()

    def finish(self) -> None:
        self.ended = self._clock()

    def stats(self, completion_tokens: int | None) -> dict[str, Any]:
        ended = self.ended if self.ended is not None else self._clock()
        out: dict[str, Any] = {"total_ms": round((ended - self.started) * 1000)}
        if self.first is not None:
            out["ttft_ms"] = round((self.first - self.started) * 1000)
            decode_s = ended - self.first
            if completion_tokens and decode_s > 0.05:
                out["tokens_per_s"] = round(completion_tokens / decode_s, 1)
        return out
