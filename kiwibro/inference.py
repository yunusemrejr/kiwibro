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
