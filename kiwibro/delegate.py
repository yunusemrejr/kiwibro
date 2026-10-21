"""Capability-aware delegation to other local LM Studio models.

The selected (main) model stays in charge of the conversation. When a task
needs a capability it lacks — vision, say — the harness can hand a small,
bounded side-question to the *smallest adequate* installed model that has it,
then continue with the main model. Model picking and JIT load/unload live in
:mod:`kiwibro.routing` when available; this module keeps a conservative
fallback picker so delegation works either way.

Also home to ``study_text``: deterministic map-reduce over long content —
split into overlapping chunks, ask the helper about each, merge the findings —
so small-context models can still reason about big files.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any

from .llm import LMStudioClient, LMStudioError
from .sandbox import SandboxError, Workspace

log = logging.getLogger(__name__)

NEEDS = ("vision", "audio", "reasoning", "coding", "long_context", "fast", "general")
MAX_IMAGE_BYTES = 8 * 1024 * 1024
STUDY_CHUNK_CHARS = 3000
STUDY_OVERLAP_CHARS = 200
STUDY_MAX_CHUNKS = 8

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")
_CODING_HINTS = ("coder", "code", "starcoder", "deepseek-coder", "qwen2.5-coder", "codellama")
_REASONING_HINTS = ("deepseek-r1", "qwq", "reason", "think", "-r1", "o1")
_AUDIO_HINTS = ("whisper", "audio", "omni")


class DelegateError(Exception):
    """No suitable helper model, or the helper call failed (user-facing)."""


def model_size_hint(model: dict[str, Any]) -> float:
    """Rough parameter size in billions parsed from the model id/name (inf if unknown)."""
    for text in (str(model.get("id") or ""), str(model.get("name") or "")):
        match = _SIZE_RE.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:  # pragma: no cover - defensive
                continue
    return float("inf")


def capabilities_of(model: dict[str, Any]) -> list[str]:
    """Capability labels for one normalised LM Studio model entry."""
    caps = {str(cap).lower() for cap in (model.get("capabilities") or [])}
    labels: list[str] = []
    if model.get("vision") or model.get("type") == "vlm" or "vision" in caps:
        labels.append("vision")
    ident = f"{model.get('id') or ''} {model.get('name') or ''}".lower()
    if any(hint in ident for hint in _AUDIO_HINTS) or "audio" in caps:
        labels.append("audio")
    if model.get("tools"):
        labels.append("tools")
    if any(hint in ident for hint in _CODING_HINTS):
        labels.append("coding")
    if any(hint in ident for hint in _REASONING_HINTS):
        labels.append("reasoning")
    try:
        if int(model.get("max_context") or model.get("context") or 0) >= 65536:
            labels.append("long_context")
    except (TypeError, ValueError):  # pragma: no cover - defensive
        pass
    return labels


def _adequate(model: dict[str, Any], need: str) -> bool:
    if need == "general":
        return True
    if need == "fast":
        return True  # smallest wins by sort order
    return need in capabilities_of(model)


def pick_helper(
    models: list[dict[str, Any]],
    need: str,
    *,
    current: str | None = None,
) -> dict[str, Any] | None:
    """Smallest adequate model for ``need`` (loaded first, then size, then context).

    The current main model is skipped for capability needs it does not meet —
    callers check the main model themselves when they prefer it.
    """
    need = need if need in NEEDS else "general"
    candidates = [model for model in models if _adequate(model, need)]
    if not candidates:
        return None

    def sort_key(model: dict[str, Any]) -> tuple[Any, ...]:
        loaded = 0 if model.get("state") == "loaded" else 1
        not_current = 0 if current and model.get("id") != current else 1
        return (loaded, not_current, model_size_hint(model), int(model.get("context") or 0))

    if need == "long_context":
        candidates.sort(key=lambda m: (0 if m.get("state") == "loaded" else 1, -int(m.get("max_context") or 0)))
        return candidates[0]
    candidates.sort(key=sort_key)
    return candidates[0]


def list_model_capabilities(llm: LMStudioClient) -> list[dict[str, Any]]:
    """All local chat models with capability labels (for the model_info tool)."""
    ok, models, error = llm.models_status()
    if not ok:
        raise DelegateError(error or "LM Studio is not reachable.")
    if not models:
        raise DelegateError("No chat models are installed in LM Studio.")
    return [
        {
            "id": model["id"],
            "name": model.get("name") or model["id"],
            "state": model.get("state"),
            "context": model.get("context"),
            "capabilities": capabilities_of(model),
            "size_hint_b": (None if model_size_hint(model) == float("inf") else model_size_hint(model)),
        }
        for model in models
    ]


def _routing_run_helper():
    """routing.run_helper when the routing module provides it, else None."""
    try:
        from . import routing  # type: ignore[attr-defined]
    except Exception:  # module not present yet — fallback picker applies
        return None
    helper = getattr(routing, "run_helper", None)
    return helper if callable(helper) else None


def run_helper(
    llm: LMStudioClient,
    need: str,
    messages: list[dict[str, Any]],
    *,
    current_model: str | None = None,
    model: str | None = None,
    max_tokens: int = 600,
    read_timeout: int = 120,
) -> dict[str, Any]:
    """Ask a helper model one bounded question; returns {model, text, usage}.

    Prefers the current model when it already meets the need (no swap cost);
    otherwise picks the smallest adequate installed model. Load/unload of a
    just-in-time helper is owned by :mod:`kiwibro.routing` when present;
    without it, LM Studio's own just-in-time loading applies and nothing is
    unloaded (the user's server settings govern eviction).
    """
    need = need if need in NEEDS else "general"
    routed = _routing_run_helper()
    if routed is not None and not model:
        try:
            result = routed(
                llm, need, messages,
                current_model=current_model, max_tokens=max_tokens,
            )
            if isinstance(result, dict) and result.get("text") is not None:
                return {
                    "model": result.get("model") or current_model or "",
                    "text": str(result["text"]),
                    "usage": result.get("usage"),
                    "need": need,
                }
        except Exception:  # routing is an optimisation; fall back gracefully
            log.warning("routing.run_helper failed; using fallback picker", exc_info=True)
    ok, models, error = llm.models_status()
    if not ok or not models:
        raise DelegateError(error or "No chat models are installed in LM Studio.")
    chosen: dict[str, Any] | None = None
    if model:
        chosen = next((item for item in models if item["id"] == model), None)
        if chosen is None:
            raise DelegateError(f"Model not found in LM Studio: {model}")
    else:
        current_entry = next((item for item in models if item["id"] == current_model), None)
        if current_entry is not None and _adequate(current_entry, need):
            chosen = current_entry
        else:
            chosen = pick_helper(models, need, current=current_model)
        if chosen is None:
            raise DelegateError(
                f"No installed LM Studio model provides “{need}”. "
                "Solve it another way (code execution, files, web) or tell the user "
                f"which kind of model to install for {need}."
            )
    try:
        completion = llm.complete(
            model=chosen["id"],
            messages=messages,
            max_tokens=max(64, min(int(max_tokens or 600), 2048)),
            temperature=0.2,
            read_timeout=read_timeout,
            retries=0,
        )
    except LMStudioError as exc:
        raise DelegateError(f"Helper model {chosen['id']} failed: {exc.message}") from exc
    text = (completion.text or "").strip()
    if not text:
        raise DelegateError(f"Helper model {chosen['id']} returned an empty answer.")
    return {"model": chosen["id"], "text": text, "usage": completion.usage, "need": need}


def image_data_url(workspace: Workspace, path: str) -> tuple[str, str]:
    """Read a workspace image as a data URL for vision models. (url, public path)"""
    resolved = workspace.resolve(path, must_exist=True)
    if not resolved.is_file():
        raise SandboxError(f"not a file: {workspace.public(resolved)}")
    mime, _encoding = mimetypes.guess_type(resolved.name)
    if not mime or not mime.startswith("image/"):
        raise DelegateError(f"not an image file: {workspace.public(resolved)}")
    try:
        size = resolved.stat().st_size
        if size > MAX_IMAGE_BYTES:
            raise DelegateError(f"image is too large ({size} bytes; limit {MAX_IMAGE_BYTES})")
        payload = resolved.read_bytes()
    except OSError as exc:
        raise DelegateError(f"could not read image: {exc.strerror or exc}") from exc
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}", workspace.public(resolved)


def describe_image(
    llm: LMStudioClient,
    workspace: Workspace,
    path: str,
    question: str = "",
    *,
    current_model: str | None = None,
) -> dict[str, Any]:
    """Describe/answer about a workspace image with the smallest vision model."""
    data_url, public = image_data_url(workspace, path)
    prompt = (question or "").strip() or (
        "Describe this image in detail: what it shows, any text in it, and anything unusual."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    result = run_helper(llm, "vision", messages, current_model=current_model, max_tokens=800)
    return {**result, "path": public, "question": prompt}


# --- study_text: map-reduce over long content -------------------------------------


def chunk_text(
    text: str,
    *,
    size: int = STUDY_CHUNK_CHARS,
    overlap: int = STUDY_OVERLAP_CHARS,
    max_chunks: int = STUDY_MAX_CHUNKS,
) -> list[str]:
    """Split long text into overlapping chunks on paragraph/line boundaries."""
    text = (text or "").strip()
    if len(text) <= size:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    while start < len(text) and len(chunks) < max_chunks:
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end))
            if boundary > start + size // 2:
                end = boundary
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def study_text(
    llm: LMStudioClient,
    text: str,
    question: str,
    *,
    current_model: str | None = None,
    source: str = "text",
) -> dict[str, Any]:
    """Answer a question about long text via chunk → per-chunk notes → merge."""
    question = (question or "").strip()
    if not question:
        raise DelegateError("a question is required to study a file")
    chunks = chunk_text(text)
    if not chunks:
        raise DelegateError(f"{source} is empty")
    truncated = len(text) > STUDY_CHUNK_CHARS * STUDY_MAX_CHUNKS
    if len(chunks) == 1:
        messages = [
            {"role": "system", "content": "Answer the question using only the provided text. Be specific and quote key details."},
            {"role": "user", "content": f"Text from {source}:\n\n{chunks[0]}\n\nQuestion: {question}"},
        ]
        result = run_helper(llm, "fast", messages, current_model=current_model, max_tokens=700)
        return {**result, "chunks": 1, "source": source}
    notes: list[str] = []
    helper_model = ""
    for index, chunk in enumerate(chunks, start=1):
        messages = [
            {"role": "system", "content": "Extract only facts relevant to the question, as short bullet points. If nothing is relevant, reply exactly: (nothing relevant)"},
            {"role": "user", "content": f"Part {index}/{len(chunks)} of {source}:\n\n{chunk}\n\nQuestion: {question}"},
        ]
        result = run_helper(llm, "fast", messages, current_model=current_model, max_tokens=400)
        helper_model = result["model"]
        note = result["text"].strip()
        if note and note != "(nothing relevant)":
            notes.append(f"From part {index}:\n{note}")
    if not notes:
        return {
            "model": helper_model,
            "text": f"No part of {source} seemed relevant to the question.",
            "chunks": len(chunks),
            "source": source,
            "truncated": truncated,
        }
    joined = "\n\n".join(notes)
    merge_messages = [
        {"role": "system", "content": "Combine the notes into one clear answer. Keep concrete details; drop repetition."},
        {"role": "user", "content": f"Question: {question}\n\nNotes from {source}:\n{joined[:8000]}"},
    ]
    merged = run_helper(llm, "fast", merge_messages, current_model=current_model, max_tokens=800)
    return {
        **merged,
        "chunks": len(chunks),
        "source": source,
        "truncated": truncated,
        "notes": notes,
    }


def study_file(
    llm: LMStudioClient,
    workspace: Workspace,
    path: str,
    question: str,
    *,
    current_model: str | None = None,
) -> dict[str, Any]:
    resolved = workspace.resolve(path, must_exist=True)
    if not resolved.is_file():
        raise SandboxError(f"not a file: {workspace.public(resolved)}")
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise DelegateError(f"could not read file: {exc.strerror or exc}") from exc
    return study_text(
        llm, text, question,
        current_model=current_model, source=workspace.public(resolved),
    )

