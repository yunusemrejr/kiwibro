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
"""

from __future__ import annotations

import logging
from typing import Any

from .llm import LMStudioError

log = logging.getLogger(__name__)

NEEDS = ("vision", "audio", "reasoning", "coding", "long_context", "fast", "embeddings", "tools")

_REASONING_HINTS = ("r1", "reason", "think", "deepseek-r", "qwq", "o1", "o3", "magistral")
_CODING_HINTS = ("coder", "code", "starcoder", "codellama", "devstral")
_LONG_CONTEXT_MIN = 65536


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
    helper = pick_helper(models, need, current=current_model)
    if helper is None:
        raise LMStudioError(
            "no_model", f"No installed LM Studio model provides '{need}'."
        )
    model_id = str(helper["id"])
    was_loaded = helper.get("state") == "loaded"
    load_model = getattr(llm, "load_model", None)
    unload_model = getattr(llm, "unload_model", None)
    if not was_loaded and callable(load_model):
        try:
            load_model(model_id)
        except Exception:  # JIT loading still applies on the call itself
            log.debug("best-effort load failed for %s", model_id, exc_info=True)
    try:
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
            return {"model": model_id, "text": "", "embeddings": vectors}
        if need == "vision":
            prompt, uris = _image_parts(messages)
            complete_vision = getattr(llm, "complete_vision", None)
            if uris and callable(complete_vision):
                completion = complete_vision(
                    model=model_id, prompt=prompt or "Describe this image.",
                    image_data_uris=uris, max_tokens=max_tokens, temperature=0.0,
                )
                return {"model": model_id, "text": (completion.text or "").strip()}
        completion = llm.complete(
            model=model_id, messages=messages, max_tokens=max_tokens,
            temperature=0.0, retries=0,
        )
        return {"model": model_id, "text": (completion.text or "").strip()}
    finally:
        if not was_loaded and model_id != current_model and callable(unload_model):
            try:
                unload_model(model_id)
            except Exception:  # best-effort: LM Studio idle-eviction is the backstop
                log.debug("best-effort unload failed for %s", model_id, exc_info=True)
