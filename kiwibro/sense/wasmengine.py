"""The Kiwi Sense core run as WebAssembly on the server (needs the `wasmtime` package)."""

from __future__ import annotations

import struct
from pathlib import Path


class WasmHead:
    def __init__(self, tag: str, kind: int, classes: int) -> None:
        self.tag, self.kind, self.classes = tag, kind, classes


class WasmEngine:
    name = "wasm"

    def __init__(self, wasm_path: Path, blob: bytes) -> None:
        import wasmtime  # optional dependency; the caller falls back to PyEngine without it

        self._store = wasmtime.Store()
        module = wasmtime.Module.from_file(self._store.engine, str(wasm_path))
        exports = wasmtime.Instance(self._store, module, []).exports(self._store)
        self._memory = exports["memory"]
        self._fn = {
            name: exports[name]
            for name in (
                "ks_alloc", "ks_reset", "ks_init", "ks_dims", "ks_classes", "ks_embed",
                "ks_words", "ks_similarity", "ks_probe", "ks_repeat", "ks_lexsim",
                "ks_heads", "ks_head_tag", "ks_head_kind", "ks_head_classes",
                "ks_head_text", "ks_head_pair", "ks_head_tokens",
            )
        }
        at = self._put(blob)
        status = self._call("ks_init", at, len(blob))
        if status != 0:
            raise ValueError(f"Kiwi Sense data file was rejected (code {status})")
        self.dims = self._call("ks_dims")
        self.classes = self._call("ks_classes")
        self.heads = [
            WasmHead(struct.pack("<I", self._call("ks_head_tag", i)).decode("ascii", "replace"),
                     self._call("ks_head_kind", i), self._call("ks_head_classes", i))
            for i in range(self._call("ks_heads"))
        ]
        self.last_words = 0

    def _call(self, name: str, *args):
        return self._fn[name](self._store, *args)

    def _put(self, data: bytes) -> int:
        at = self._call("ks_alloc", max(1, len(data)))
        if not at:
            raise MemoryError("Kiwi Sense ran out of memory")
        self._memory.write(self._store, data, at)
        return at

    def _doubles(self, at: int, count: int) -> list[float]:
        return list(struct.unpack(f"<{count}d", bytes(self._memory.read(self._store, at, at + 8 * count))))

    def embed(self, text: bytes) -> tuple[list[float], int]:
        try:
            at = self._put(text)
            out = self._call("ks_alloc", 8 * self.dims)
            known = self._call("ks_embed", at, len(text), out)
            self.last_words = self._call("ks_words")
            return self._doubles(out, self.dims), known
        finally:
            self._call("ks_reset")

    def similarity(self, a: bytes, b: bytes) -> float:
        try:
            return self._call("ks_similarity", self._put(a), len(a), self._put(b), len(b))
        finally:
            self._call("ks_reset")

    def probe(self, text: bytes) -> list[float] | None:
        if not self.classes:
            return None
        try:
            at = self._put(text)
            out = self._call("ks_alloc", 8 * self.classes)
            self._call("ks_probe", at, len(text), out)
            return self._doubles(out, self.classes)
        finally:
            self._call("ks_reset")

    def repeat(self, text: bytes) -> float:
        try:
            return self._call("ks_repeat", self._put(text), len(text))
        finally:
            self._call("ks_reset")

    def lexsim(self, a: bytes, b: bytes) -> float:
        try:
            return self._call("ks_lexsim", self._put(a), len(a), self._put(b), len(b))
        finally:
            self._call("ks_reset")

    def head_text(self, index: int, text: bytes) -> list[float] | None:
        classes = self.heads[index].classes
        try:
            at = self._put(text)
            out = self._call("ks_alloc", 8 * classes)
            if self._call("ks_head_text", index, at, len(text), out) != 0:
                return None
            return self._doubles(out, classes)
        finally:
            self._call("ks_reset")

    def head_pair(self, index: int, a: bytes, b: bytes) -> list[float] | None:
        classes = self.heads[index].classes
        try:
            at_a, at_b = self._put(a), self._put(b)
            out = self._call("ks_alloc", 8 * classes)
            if self._call("ks_head_pair", index, at_a, len(a), at_b, len(b), out) != 0:
                return None
            return self._doubles(out, classes)
        finally:
            self._call("ks_reset")

    def head_tokens(self, index: int, text: bytes) -> list[list[float]] | None:
        classes = self.heads[index].classes
        try:
            at = self._put(text)
            out = self._call("ks_alloc", 8 * classes * 128)
            count = self._call("ks_head_tokens", index, at, len(text), out)
            if count < 0:
                return None
            flat = self._doubles(out, classes * count) if count else []
            return [flat[k * classes:(k + 1) * classes] for k in range(count)]
        finally:
            self._call("ks_reset")
