"""Capability routing: the stable seam for temporary helper models.

Kiwibro's main model is often a 1–4B text-only model. When a task needs a
capability it lacks (vision, audio, long context, …), tools call
:func:`run_helper` instead of failing: the smallest adequate *installed*
model answers once, is stopped again if we loaded it, and the main model
continues. If nothing installed fits, callers get a typed error and solve
the problem with code/tools instead.

Layering (one policy, no duplication): capability *facts* come from
:mod:`kiwibro.capabilities` and HTTP/serving primitives from
:class:`~kiwibro.llm.LMStudioClient` (``catalog``, ``complete_vision``,
``embed``, ``load_model``/``unload_model``). This module only ranks and
orchestrates, duck-typing those primitives so it keeps working (JIT-only)
on older clients. Everything is best-effort and bounded: a helper call
never unloads the main model or a model that was already loaded.

Lifecycle (``_lease``): the current model is used directly when it already
has the capability; otherwise a helper is loaded only if free RAM allows it
next to the main model, concurrent calls share one load (reference count),
and it is unloaded when the last of them finishes. Helper calls never think
(``reasoning_effort: "none"`` when the model supports the toggle) and carry
a JIT ``ttl`` so a crash cannot leave a helper resident for long.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from . import media
from .inference import model_profile, reasoning_payload
from .llm import LMStudioError

log = logging.getLogger(__name__)

NEEDS = ("vision", "audio", "reasoning", "coding", "long_context", "fast", "embeddings", "tools")

_REASONING_HINTS = ("r1", "reason", "think", "deepseek-r", "qwq", "o1", "o3", "magistral")
_CODING_HINTS = ("coder", "code", "starcoder", "codellama", "devstral")
_LONG_CONTEXT_MIN = 65536
HELPER_CONTEXT = 8192
HELPER_TTL_S = 600
RAM_HEADROOM = 1.25  # weights + KV cache at HELPER_CONTEXT + runtime overhead


def _annotate(model: dict[str, Any]) -> dict[str, Any]:
    """Capability-annotated copy (delegates to capabilities when present)."""
    try:
        from . import capabilities as caps

        return caps.infer_capabilities(model)
    except Exception:  # pragma: no cover - capabilities is in-repo; stay robust
        out = dict(model)
        out.setdefault("vision", str(out.get("type") or "").lower() == "vlm")
        out.setdefault("audio", False)
        out.setdefault("embeddings", str(out.get("type") or "").lower() == "embeddings")
        out.setdefault("tool_use", bool(out.get("tools")))
        out.setdefault("chat", str(out.get("type") or "llm").lower() in ("llm", "vlm"))
        return out


def _size_key(model: dict[str, Any]) -> tuple:
    try:
        from . import capabilities as caps

        return caps.size_key(model)
    except Exception:  # pragma: no cover - fallback ordering
        return (1e9, 0.0, str(model.get("id") or ""))


def _context_of(model: dict[str, Any]) -> int:
    for key in ("context", "max_context", "loaded_context_length", "max_context_length"):
        try:
            value = int(model.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return 0


def capabilities_of(model: dict[str, Any]) -> list[str]:
    """Capability names for one model record (server facts + soft hints).

    Soft needs (``reasoning``/``coding``/``fast``) are name/size heuristics —
    useful for ranking helpers, never treated as guarantees.
    """
    info = _annotate(model)
    caps: list[str] = []
    if info.get("chat"):
        caps.append("chat")
    if info.get("vision"):
        caps.append("vision")
    if info.get("audio"):
        caps.append("audio")
    if info.get("embeddings"):
        caps.append("embeddings")
    if info.get("tool_use") or info.get("tools"):
        caps.append("tools")
    model_id = str(info.get("id") or "").lower()
    if any(hint in model_id for hint in _REASONING_HINTS):
        caps.append("reasoning")
    if any(hint in model_id for hint in _CODING_HINTS):
        caps.append("coding")
    if _context_of(info) >= _LONG_CONTEXT_MIN:
        caps.append("long_context")
    params = info.get("params_b")
    memory_gb = info.get("est_memory_gb")
    if (isinstance(params, (int, float)) and 0 < params <= 4) or (
        isinstance(memory_gb, (int, float)) and 0 < memory_gb <= 3
    ):
        caps.append("fast")
    return caps


def _satisfies(info: dict[str, Any], need: str) -> bool:
    if need == "vision":
        return bool(info.get("vision"))
    if need == "audio":
        return bool(info.get("audio"))
    if need == "embeddings":
        return bool(info.get("embeddings"))
    if need == "tools":
        return bool(info.get("tool_use") or info.get("tools"))
    if need == "long_context":
        return bool(info.get("chat")) and _context_of(info) >= _LONG_CONTEXT_MIN
    if need in ("reasoning", "coding", "fast"):
        return bool(info.get("chat"))  # soft needs rank, not filter
    return False


def pick_helper(
    models: list[dict[str, Any]], need: str, current: str | None = None
) -> dict[str, Any] | None:
    """Smallest adequate installed model for ``need`` (loaded first).

    ``current`` (the main model id) is never returned: a helper exists
    precisely because the main model is not the right tool. Returns an
    annotated copy, or None when nothing installed satisfies the need.
    """
    if need not in NEEDS:
        raise ValueError(f"unknown capability need: {need}")
    annotated = [_annotate(m) for m in models if isinstance(m, dict) and m.get("id")]
    pool = [m for m in annotated if m["id"] != current and _satisfies(m, need)]
    if need in ("reasoning", "coding"):
        hinted = [m for m in pool if need in capabilities_of(m)]
        pool = hinted or pool
    if not pool:
        return None
    loaded_first = lambda m: 0 if m.get("state") == "loaded" else 1  # noqa: E731
    if need == "long_context":
        pool.sort(key=lambda m: (loaded_first(m), -_context_of(m), _size_key(m)))
    else:
        pool.sort(key=lambda m: (loaded_first(m), _size_key(m)))
    return pool[0]


def _catalog(llm: Any) -> list[dict[str, Any]]:
    catalog = getattr(llm, "catalog", None)
    if callable(catalog):
        try:
            return list(catalog())
        except Exception:
            log.warning("model catalog failed; falling back to chat list", exc_info=True)
    return list(llm.list_models())


def _image_parts(messages: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Extract (prompt text, image data URIs) from OpenAI-style messages."""
    prompt = ""
    uris: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and message.get("role") == "user":
            prompt = content
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    prompt = part["text"]
                elif part.get("type") == "image_url":
                    url = str((part.get("image_url") or {}).get("url") or "")
                    if url.startswith("data:image/"):
                        uris.append(url)
    return prompt, uris


def available_memory_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


MEMORY_PROBE: Callable[[], int | None] = available_memory_bytes
_STATE_LOCK = threading.Lock()
_LEASES: dict[str, int] = {}  # helper id -> active calls
_LOADED: set[str] = set()  # helpers this process loaded and must unload
_LOAD_LOCKS: dict[str, threading.Lock] = {}


def _check_memory(helper: dict[str, Any]) -> None:
    needed = model_profile(str(helper["id"]), helper).cost
    if needed == float("inf"):
        gb = helper.get("est_memory_gb")
        needed = gb * 1e9 if isinstance(gb, (int, float)) else needed
    free = MEMORY_PROBE()
    if free is not None and needed != float("inf") and needed * RAM_HEADROOM > free:
        raise LMStudioError(
            "no_memory",
            f"{helper['id']} needs about {needed * RAM_HEADROOM / 1e9:.1f} GB but only "
            f"{free / 1e9:.1f} GB is free, so it was not loaded next to the main model.",
        )


@contextmanager
def _lease(llm: Any, helper: dict[str, Any]) -> Iterator[bool]:
    """Hold ``helper`` loaded for one call; yields True when this process loaded it."""
    model_id = str(helper["id"])
    with _STATE_LOCK:
        _LEASES[model_id] = _LEASES.get(model_id, 0) + 1
        load_lock = _LOAD_LOCKS.setdefault(model_id, threading.Lock())
    try:
        with load_lock:  # concurrent calls on one cold helper load it once
            with _STATE_LOCK:
                needs_load = helper.get("state") != "loaded" and model_id not in _LOADED
            if needs_load:
                _check_memory(helper)
                load_model = getattr(llm, "load_model", None)
                loaded = False
                if callable(load_model):
                    try:
                        loaded = bool(load_model(model_id, context_length=HELPER_CONTEXT))
                    except Exception:  # JIT loading still applies on the call itself
                        log.debug("best-effort load failed for %s", model_id, exc_info=True)
                with _STATE_LOCK:
                    # JIT-loaded helpers are tracked too, so they are unloaded after use.
                    _LOADED.add(model_id)
                if loaded:
                    log.info("loaded helper %s", model_id)
        with _STATE_LOCK:
            temporary = model_id in _LOADED
        yield temporary
    finally:
        with _STATE_LOCK:
            remaining = _LEASES.get(model_id, 1) - 1
            if remaining > 0:
                _LEASES[model_id] = remaining
                release = False
            else:
                _LEASES.pop(model_id, None)
                release = model_id in _LOADED
                _LOADED.discard(model_id)
        unload_model = getattr(llm, "unload_model", None)
        if release and callable(unload_model):
            try:
                unload_model(model_id)
            except Exception:  # best-effort: the request ttl is the backstop
                log.debug("best-effort unload failed for %s", model_id, exc_info=True)


def temporary_models() -> list[str]:
    with _STATE_LOCK:
        return sorted(_LOADED)


def run_helper(
    llm: Any,
    need: str,
    messages: list[dict[str, Any]],
    *,
    current_model: str | None = None,
    max_tokens: int = 600,
) -> dict[str, Any]:
    """Run one bounded helper-model call; returns ``{"model", "text", ...}``.

    Lifecycle contract: if the helper was not loaded, it is best-effort
    loaded (JIT on most LM Studio builds) and best-effort unloaded after
    the call; a previously loaded model and the main model are never
    unloaded. Raises :class:`LMStudioError` (kind ``no_model``) when no
    installed model satisfies ``need`` — callers then fall back to
    code/tools. For ``need="embeddings"`` the result also carries
    ``embeddings`` vectors and ``text`` is empty.
    """
    models = _catalog(llm)
    current = next((m for m in models if m.get("id") == current_model), None)
    if current is not None and _satisfies(_annotate(current), need) and need not in ("fast", "long_context"):
        helper = _annotate(current)  # the main model already can: no swap, no load
    else:
        helper = pick_helper(models, need, current=current_model)
    if helper is None:
        raise LMStudioError(
            "no_model", f"No installed LM Studio model provides '{need}'."
        )
    model_id = str(helper["id"])
    with _lease(llm, helper) as temporary:
        extra = reasoning_payload(model_profile(model_id, helper), think=False)
        if temporary:
            extra["ttl"] = HELPER_TTL_S
        if need == "embeddings":
            embed = getattr(llm, "embed", None)
            if not callable(embed):
                raise LMStudioError("no_model", "Embeddings are not available from this client.")
            texts = [
                str(m.get("content") or "")
                for m in messages
                if isinstance(m.get("content"), str) and m.get("content")
            ]
            vectors = embed(model=model_id, texts=texts)
            return {"model": model_id, "text": "", "embeddings": vectors, "temporary": temporary}
        if need == "vision":
            prompt, uris = _image_parts(messages)
            complete_vision = getattr(llm, "complete_vision", None)
            if uris and callable(complete_vision):
                completion = complete_vision(
                    model=model_id, prompt=prompt or "Describe this image.",
                    image_data_uris=media.shrink_data_urls(uris), max_tokens=max_tokens,
                    temperature=0.0, extra=extra or None,
                )
                return {"model": model_id, "text": (completion.text or "").strip(), "temporary": temporary}
            messages = media.shrink_image_parts(messages)
        completion = llm.complete(
            model=model_id, messages=messages, max_tokens=max_tokens,
            temperature=0.0, retries=0, extra=extra or None,
        )
        return {"model": model_id, "text": (completion.text or "").strip(), "temporary": temporary}


# --- main-model context window --------------------------------------------------------

CONTEXT_STEPS = (8192, 16384, 32768, 65536, 131072, 262144)


def context_choices(model: dict[str, Any]) -> list[int]:
    """Larger windows worth offering for a loaded model (≤ its maximum, at most three)."""
    current = int((model.get("loaded_config") or {}).get("context_length") or model.get("context") or 0)
    ceiling = int(model.get("max_context") or 0)
    return [size for size in CONTEXT_STEPS if current < size <= ceiling][:3]


def reload_with_context(llm: Any, model: dict[str, Any], context_length: int) -> str:
    """Reload a loaded model with another context window; returns the new instance id.

    LM Studio can't resize a loaded model, so this unloads and loads again. If
    the bigger window does not fit (LM Studio's load guardrails refuse it), the
    previous window is restored so the user is never left without a model.
    """
    model_id = str(model["id"])
    key = str(model.get("key") or model_id)
    previous = int((model.get("loaded_config") or {}).get("context_length") or model.get("context") or 0)
    if not llm.unload_model(model_id):
        raise LMStudioError("load", f"LM Studio could not unload {model_id}.")
    if llm.load_model(key, context_length):
        return llm.loaded_instance(key) or model_id
    if previous and llm.load_model(key, previous):
        raise LMStudioError(
            "no_memory",
            f"{model_id} does not fit with a {context_length // 1024}K context; kept {previous // 1024}K.",
        )
    raise LMStudioError("load", f"LM Studio could not load {model_id} again. Load it in LM Studio.")
