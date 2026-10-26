"""Read the metadata header of a GGUF model file (no tensors, no dependencies).

The direct engine needs a few facts LM Studio would otherwise report: the
architecture, the trained context length, the parameter label, the file type and
the chat template (whether it takes tools and a thinking switch). They sit in
the key/value section at the start of the file; the big arrays there (the
tokenizer vocabulary, ~150k strings) are skipped by seeking, never decoded.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any, BinaryIO

MAGIC = b"GGUF"
MAX_KV = 4096  # a real header has a few dozen keys; more means a corrupt file
MAX_STRING = 4 * 1024 * 1024  # chat templates run to tens of KB; vocab entries are tiny

# GGUF value types
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = range(13)
_SCALARS: dict[int, str] = {
    _UINT8: "<B", _INT8: "<b", _UINT16: "<H", _INT16: "<h", _UINT32: "<I", _INT32: "<i",
    _FLOAT32: "<f", _BOOL: "<?", _UINT64: "<Q", _INT64: "<q", _FLOAT64: "<d",
}

# general.file_type -> quantisation name (llama.cpp's LLAMA_FTYPE_* values)
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
    11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
    17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
    23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4",
}


class GGUFError(ValueError):
    pass


def _read(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise GGUFError("truncated GGUF header")
    return data


def _scalar(handle: BinaryIO, kind: int) -> Any:
    fmt = _SCALARS[kind]
    return struct.unpack(fmt, _read(handle, struct.calcsize(fmt)))[0]


def _string(handle: BinaryIO, keep: bool = True) -> str | None:
    (length,) = struct.unpack("<Q", _read(handle, 8))
    if length > MAX_STRING:
        raise GGUFError("implausible string length in GGUF header")
    if not keep:
        handle.seek(length, 1)
        return None
    return _read(handle, length).decode("utf-8", errors="replace")


def _value(handle: BinaryIO, kind: int, keep: bool) -> Any:
    if kind in _SCALARS:
        value = _scalar(handle, kind)
        return value if keep else None
    if kind == _STRING:
        return _string(handle, keep)
    if kind == _ARRAY:
        item_kind = _scalar(handle, _UINT32)
        (count,) = struct.unpack("<Q", _read(handle, 8))
        if item_kind in _SCALARS:  # fixed size: skip in one seek
            handle.seek(struct.calcsize(_SCALARS[item_kind]) * count, 1)
            return count if keep else None
        if item_kind == _STRING:
            for _ in range(count):
                _string(handle, keep=False)
            return count if keep else None
        raise GGUFError("nested GGUF arrays are not supported")
    raise GGUFError(f"unknown GGUF value type {kind}")


def read_metadata(path: str | Path) -> dict[str, Any]:
    """The header's key/values; arrays come back as their element count."""
    with open(path, "rb") as handle:
        if _read(handle, 4) != MAGIC:
            raise GGUFError("not a GGUF file")
        version = _scalar(handle, _UINT32)
        if version < 2:
            raise GGUFError("GGUF v1 files are not supported")
        _scalar(handle, _UINT64)  # tensor count: the tensors are never read
        count = _scalar(handle, _UINT64)
        if count > MAX_KV:
            raise GGUFError("implausible GGUF key count")
        meta: dict[str, Any] = {"gguf.version": version}
        for _ in range(count):
            key = _string(handle) or ""
            kind = _scalar(handle, _UINT32)
            meta[key] = _value(handle, kind, keep=True)
        if handle.tell() > os.fstat(handle.fileno()).st_size:
            raise GGUFError("truncated GGUF header")  # a skip (seek) ran past the end
        return meta


def summarize(meta: dict[str, Any]) -> dict[str, Any]:
    """The facts the engine uses, from :func:`read_metadata` output."""
    arch = str(meta.get("general.architecture") or "")
    template = meta.get("tokenizer.chat_template")
    template = template if isinstance(template, str) else ""
    context = meta.get(f"{arch}.context_length")
    file_type = meta.get("general.file_type")
    lowered = template.lower()
    return {
        "arch": arch,
        "name": str(meta.get("general.name") or ""),
        "size_label": str(meta.get("general.size_label") or ""),
        "context_length": int(context) if isinstance(context, int) and context > 0 else None,
        "quantization": FILE_TYPES.get(file_type, "") if isinstance(file_type, int) else "",
        "has_template": bool(template),
        "tools": "tools" in lowered or "tool_call" in lowered,
        # Thinks at all (a <think> block), and whether a request can switch it off.
        "thinking": any(marker in lowered for marker in ("enable_thinking", "<think>", "reasoning_effort")),
        "thinking_toggle": "enable_thinking" in lowered or "reasoning_effort" in lowered,
        # Encoders (embeddings) have pooling and no chat template.
        "embeddings": meta.get(f"{arch}.pooling_type") is not None and not template,
        "projector": arch == "clip" or "clip.vision.image_size" in meta,
    }
