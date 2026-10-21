"""Capability-aware delegation to other local LM Studio models (agent-facing).

The selected (main) model stays in charge of the conversation. When a task
needs a capability it lacks — vision, say — a small, bounded side-question goes
to the *smallest adequate* installed model that has it, then the main model
continues. Choosing that model and loading/unloading it is
:mod:`kiwibro.routing`'s job (one policy); this module turns it into tool
results and actionable errors, and adds media helpers (``describe_image``,
``see_attachment``).

Also home to ``study_text``: deterministic map-reduce over long content —
split into overlapping chunks, ask the helper about each, merge the findings —
so small-context models can still reason about big files.
"""

from __future__ import annotations

import logging
import mimetypes
import re
from pathlib import Path
from typing import Any

from . import media, routing
from .inference import model_profile
from .llm import LMStudioClient, LMStudioError
from .routing import capabilities_of, pick_helper  # noqa: F401 - the delegation API
from .sandbox import SandboxError, Workspace

log = logging.getLogger(__name__)

NEEDS = ("vision", "audio", "reasoning", "coding", "long_context", "fast", "general")
MAX_IMAGE_BYTES = 8 * 1024 * 1024  # after downscaling, per image sent to a model
STUDY_CHUNK_CHARS = 3000
STUDY_OVERLAP_CHARS = 200
STUDY_MAX_CHUNKS = 8
STUDY_READ_CAP = 2_000_000  # extracted characters considered for one question


class DelegateError(Exception):
    """No suitable helper model, or the helper call failed (user-facing)."""


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
            "size_hint_b": model_profile(model["id"], model).params_b,
        }
        for model in models
    ]


def run_helper(
    llm: LMStudioClient,
    need: str,
    messages: list[dict[str, Any]],
    *,
    current_model: str | None = None,
    model: str | None = None,
    max_tokens: int = 600,
) -> dict[str, Any]:
    """Ask a helper model one bounded question → ``{model, text, usage, temporary, need}``.

    ``routing.run_helper`` keeps the main model when it already meets the need
    and otherwise leases the smallest adequate one; ``model`` names it explicitly.
    """
    need = need if need in NEEDS else "general"
    try:
        result = routing.run_helper(
            llm, need, messages, current_model=current_model, model=model,
            max_tokens=max(64, min(int(max_tokens or 600), 2048)),
        )
    except LMStudioError as exc:
        if exc.kind in ("no_model", "no_memory"):
            raise DelegateError(
                f"{exc.message} Solve it another way (code execution, files, web) or tell "
                f"the user which kind of model to install for {need}."
            ) from exc
        raise DelegateError(f"The helper model failed: {exc.message}") from exc
    if not result.get("text"):
        raise DelegateError(f"Helper model {result.get('model')} returned an empty answer.")
    return {**result, "need": need}


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
        if size > media.MAX_ATTACHMENT_BYTES:
            raise DelegateError(f"image is too large ({size} bytes; limit {media.MAX_ATTACHMENT_BYTES})")
        payload = resolved.read_bytes()
    except OSError as exc:
        raise DelegateError(f"could not read image: {exc.strerror or exc}") from exc
    url = media.image_data_url(payload, mime)  # downscaled: vision tokens scale with pixels
    if len(url) > MAX_IMAGE_BYTES * 4 // 3:
        raise DelegateError("image is too large to send to a model, even after downscaling")
    return url, workspace.public(resolved)


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


_SEE_PROMPT = (
    "{focus}Describe this {what} for an assistant that cannot see it: what it shows, "
    "all readable text and numbers, and anything relevant to the request. Be factual and complete."
)


def see_attachment(
    llm: LMStudioClient,
    workspace: Workspace,
    attachment: dict[str, Any],
    *,
    question: str = "",
    current_model: str | None = None,
) -> dict[str, Any]:
    """What the user's attachment contains, as text → ``{"model": str | None, "text": str}``.

    Images (and video keyframes) go to the main model when it can see, else to the
    smallest installed vision model; without one — or for audio and other files —
    the answer is deterministic facts measured by code (``model`` is then None).
    """
    resolved = workspace.resolve(attachment["path"], must_exist=True)
    kind = attachment.get("kind") or media.media_kind(resolved)
    focus = f'The user wrote: "{question.strip()[:400]}". ' if question.strip() else ""
    try:
        if kind == "image":
            prompt = _SEE_PROMPT.format(focus=focus, what="image")
            result = describe_image(llm, workspace, attachment["path"], prompt, current_model=current_model)
            return {"model": result["model"], "text": result["text"]}
        if kind == "video":
            frames = media.video_frames(resolved, count=4)
            if frames:
                parts: list[dict[str, Any]] = [
                    {"type": "text", "text": _SEE_PROMPT.format(
                        focus=focus, what=f"video (these are {len(frames)} evenly spaced frames, in order)"
                    )},
                    *({"type": "image_url", "image_url": {"url": media.image_data_url(frame)}} for frame in frames),
                ]
                result = run_helper(
                    llm, "vision", [{"role": "user", "content": parts}],
                    current_model=current_model, max_tokens=800,
                )
                return {"model": result["model"], "text": result["text"]}
    except (DelegateError, LMStudioError, SandboxError) as exc:
        log.info("no model could look at %s (%s); using measured facts", resolved.name, exc)
    return {"model": None, "text": media.facts_text(media.media_facts(resolved))}


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


def document_text(path: Path) -> str:
    """A file's text. PDF, Office, e-book and HTML files go through the library's extractors."""
    from . import library

    try:
        sections = library.extract(path)
    except ValueError:  # not a document type the library knows: read it as plain text
        return path.read_text(encoding="utf-8", errors="replace")[:STUDY_READ_CAP]
    parts: list[str] = []
    size = 0
    for _heading, _location, text in sections:
        parts.append(text)
        size += len(text)
        if size >= STUDY_READ_CAP:
            break
    return "\n\n".join(parts)


def focus_text(text: str, question: str) -> tuple[str, int, int]:
    """Keep the parts most relevant to the question when text exceeds the study budget.

    Returns (text, parts kept, parts total); kept parts stay in document order, so
    a long report is studied where it answers the question, not just at its start.
    """
    from .library import STOPWORDS

    chunks = chunk_text(text, max_chunks=STUDY_READ_CAP // STUDY_CHUNK_CHARS + 1)
    budget = STUDY_MAX_CHUNKS - 1  # leaves room for re-chunking overlap in study_text
    if len(chunks) <= budget:
        return text, len(chunks), len(chunks)
    terms = {word for word in re.findall(r"[^\W_]{3,}", question.lower()) if word not in STOPWORDS}

    def score(index: int) -> int:
        lowered = chunks[index].lower()
        return sum(lowered.count(term) for term in terms)

    keep = sorted(sorted(range(len(chunks)), key=score, reverse=True)[:budget])
    return "\n\n".join(chunks[index] for index in keep), len(keep), len(chunks)


def study_file(
    llm: LMStudioClient,
    workspace: Workspace,
    path: str,
    question: str,
    *,
    current_model: str | None = None,
) -> dict[str, Any]:
    resolved = workspace.resolve(path, must_exist=True)
    source = workspace.public(resolved)
    if not resolved.is_file():
        raise SandboxError(f"not a file: {source}")
    try:
        text = document_text(resolved)
    except Exception as exc:  # unreadable or corrupt document: a clear error, not a crash
        raise DelegateError(f"could not read {source}: {exc}") from exc
    focused, kept, total = focus_text(text, question)
    result = study_text(llm, focused, question, current_model=current_model, source=source)
    if kept < total:
        result["focused"] = f"{kept} of {total}"
    return result

