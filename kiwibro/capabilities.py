"""Model capability inference and smallest-capable-model selection.

Pure functions over LM Studio model records (raw ``/api/v0/models`` entries or
Kiwibro's normalised shape). Low-level facts only — capability annotation and
size ranking. Selection *policy* (which helper to run) lives in ``routing.py``;
this module deliberately has no picker, so there is exactly one policy owner.

Evidence discipline: server-reported fields are authoritative; name
heuristics only fill gaps and are tagged via ``*_source`` so callers can see
how much to trust them.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

CAPABILITIES = ("chat", "vision", "audio", "embeddings", "tool_use")

_PARAMS_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*b(?![a-z])", re.IGNORECASE)
_VISION_NAME_HINTS = (
    "llava", "-vl", "_vl", "vl-", "vision", "moondream", "minicpm-v", "internvl",
    "cogvlm", "qvq", "pixtral", "smolvlm", "gemma-3", "llama-3.2-vision",
)
_AUDIO_NAME_HINTS = ("whisper", "audio", "voxtral", "parakeet", "sensevoice", "qwen2-audio", "omni")
_QUANT_BITS = (
    ("Q2", 2.6), ("IQ2", 2.3), ("Q3", 3.4), ("IQ3", 3.0), ("Q4", 4.5), ("IQ4", 4.2),
    ("Q5", 5.5), ("Q6", 6.6), ("Q8", 8.5), ("F16", 16.0), ("BF16", 16.0), ("F32", 32.0),
)


def parse_params_b(model_id: str) -> float | None:
    """Parameter count in billions parsed from a model id (``qwen3.8-2b`` -> 2.0)."""
    best: float | None = None
    for match in _PARAMS_RE.finditer(model_id or ""):
        try:
            value = float(match.group(1))
        except ValueError:  # pragma: no cover - regex guarantees numeric
            continue
        if 0.05 <= value <= 1000 and (best is None or value > best):
            best = value
    return best


def quant_bits(quantization: str | None) -> float | None:
    text = (quantization or "").upper()
    for prefix, bits in _QUANT_BITS:
        if text.startswith(prefix):
            return bits
    return None


def est_memory_gb(model: dict[str, Any]) -> float | None:
    """Rough resident size (weights only) or None when size is unknown."""
    params = model.get("params_b")
    if params is None:
        params = parse_params_b(str(model.get("id") or ""))
    if not params:
        return None
    bits = quant_bits(str(model.get("quantization") or "")) or 4.5
    return round(params * bits / 8 * 1.12, 2)


def _name_hint(model_id: str, hints: tuple[str, ...]) -> bool:
    lowered = (model_id or "").lower()
    return any(hint in lowered for hint in hints)


def infer_capabilities(model: dict[str, Any]) -> dict[str, Any]:
    """Annotate a model record with capabilities + size rank (returns a copy).

    Accepts raw LM Studio entries (``type``, ``capabilities``, ``state``) and
    Kiwibro-normalised ones (``tools``, ``vision`` may already exist).
    """
    out = dict(model)
    model_id = str(out.get("id") or "")
    mtype = str(out.get("type") or "llm").lower()
    listed = out.get("capabilities")
    listed = [str(c).lower() for c in listed] if isinstance(listed, list) else []

    out["params_b"] = out.get("params_b") or parse_params_b(model_id)
    out["est_memory_gb"] = est_memory_gb(out)
    out["chat"] = mtype in ("llm", "vlm")
    out["embeddings"] = mtype == "embeddings" or "embeddings" in listed
    out["tool_use"] = bool(out.get("tools")) or "tool_use" in listed

    if mtype == "vlm" or "vision" in listed:
        out["vision"], out["vision_source"] = True, "server"
    elif _name_hint(model_id, _VISION_NAME_HINTS):
        out["vision"], out["vision_source"] = True, "name-heuristic"
    else:
        out["vision"], out["vision_source"] = False, "server"

    if "audio" in listed:
        out["audio"], out["audio_source"] = True, "server"
    elif _name_hint(model_id, _AUDIO_NAME_HINTS):
        out["audio"], out["audio_source"] = True, "name-heuristic"
    else:
        out["audio"], out["audio_source"] = False, "server"
    return out


def size_key(model: dict[str, Any]) -> tuple[float, float, str]:
    """Sort key ranking models smallest-first (unknown size sorts last)."""
    gb = model.get("est_memory_gb")
    params = model.get("params_b") or 0.0
    return (gb if gb is not None else 1e9, params, str(model.get("id") or ""))


def catalog(models: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [infer_capabilities(m) for m in models]


def catalog_summary(models: Iterable[dict[str, Any]], max_lines: int = 14) -> str:
    """Token-cheap one-line-per-model catalog for small-model system prompts."""
    lines: list[str] = []
    for model in catalog(models):
        caps = [c for c in CAPABILITIES if model.get(c)]
        size = f"~{model['est_memory_gb']}GB" if model.get("est_memory_gb") else "size?"
        state = "loaded" if model.get("state") == "loaded" else "idle"
        lines.append(f"- {model['id']} [{','.join(caps)}] {size} {state}")
        if len(lines) >= max_lines:
            lines.append("- …")
            break
    return "\n".join(lines)
