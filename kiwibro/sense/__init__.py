"""Kiwi Sense: a small reading model that watches the language model.

It runs beside whichever model is loaded and never asks that model anything, so it
works the same for every model. It answers four questions about text, in well under a
millisecond each:

    how close in meaning are these two texts?     (static word-piece embeddings)
    what kind of sentence is this?                (a small trained probe)
    is this text going round in circles?          (repeated four-word runs)
    do these two texts share wording?             (byte trigrams, any language)

The core is WebAssembly (core.c -> static/sense/sense.wasm). With the `wasmtime`
package installed the server runs that file; otherwise an identical Python twin runs.
The browser always runs the WebAssembly file.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Any

from . import blob as blobfile
from .pyengine import PyEngine

DATA_DIR = Path(__file__).resolve().parents[2] / "static" / "sense"
MAX_CHARS = 4000
CACHE_SIZE = 1024
# Cosine at or above which two texts are about the same thing. Measured on content words
# (terms.gist): a question and the note that answers it score 0.43 and up, unrelated pairs
# 0.40 or less.
MEANING_MATCH = 0.42

_ASCII_PUNCT = frozenset(range(33, 48)) | frozenset(range(58, 65)) | frozenset(range(91, 97)) | frozenset(range(123, 127))
_CJK_RANGES = (
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF), (0xF900, 0xFAFF), (0x2F800, 0x2FA1F),
)
_FENCE_RE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MARKUP_RE = re.compile(r"(?m)^\s{0,3}(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s?)|[*_~]{1,3}")
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


def normalize(text: str) -> str:
    """Lower case, accents removed, every word and punctuation mark separated by one space.

    This is the text form the core expects (it matches the tokenizer the embeddings
    were made with)."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFD", text[:MAX_CHARS].lower()):
        category = unicodedata.category(ch)
        if category == "Mn":
            continue
        if ch in "\t\n\r" or category == "Zs":
            out.append(" ")
            continue
        if category[0] == "C":
            continue
        code = ord(ch)
        if code in _ASCII_PUNCT or category[0] == "P" or any(lo <= code <= hi for lo, hi in _CJK_RANGES):
            out.append(f" {ch} ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def prose(text: str) -> str:
    """A reply without code blocks, inline code or markdown marks: the part that is talk."""
    text = _FENCE_RE.sub(" ", text or "")
    text = _INLINE_CODE_RE.sub(" ", text)
    text = _LINK_RE.sub(r"\1", text)
    return _MARKUP_RE.sub("", text)


def sentences(text: str, limit: int = 12) -> list[str]:
    """The first ``limit`` sentences of the prose in ``text``."""
    parts = [part.strip() for part in _SENTENCE_RE.split(prose(text))]
    return [part for part in parts if part][:limit]


def softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    weights = [math.exp(score - top) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


class Sense:
    """Thread-safe front door to the core. One instance serves the whole app."""

    def __init__(self, directory: Path | None = None, engine: str | None = None) -> None:
        directory = Path(directory or DATA_DIR)
        blob = (directory / "sense.bin").read_bytes()
        sections = blobfile.unpack(blob)
        self.meta: dict[str, Any] = json.loads(bytes(sections["META"])) if "META" in sections else {}
        self.labels: list[str] = list(self.meta.get("classes") or [])
        self._lock = threading.Lock()
        self._vectors: OrderedDict[str, tuple[tuple[float, ...], float] | None] = OrderedDict()
        wanted = (engine or os.environ.get("KIWIBRO_SENSE_ENGINE") or "auto").lower()
        self._engine: Any = None
        if wanted in ("auto", "wasm"):
            try:
                from .wasmengine import WasmEngine

                self._engine = WasmEngine(directory / "sense.wasm", blob)
            except Exception:
                if wanted == "wasm":
                    raise
        if self._engine is None:
            self._engine = PyEngine(blob)

    @property
    def engine(self) -> str:
        """"wasm" or "python": what is running the core in this process."""
        return self._engine.name

    # ---- meaning ----

    def _vector(self, text: str) -> tuple[tuple[float, ...], float] | None:
        """(unit vector, share of words recognised) for a text, or None when no word is known."""
        key = normalize(text)
        with self._lock:
            if key in self._vectors:
                self._vectors.move_to_end(key)
                return self._vectors[key]
            vector, known = self._engine.embed(key.encode("utf-8"))
            words = self._engine.last_words
            entry = (tuple(vector), known / words) if known else None
            self._vectors[key] = entry
            if len(self._vectors) > CACHE_SIZE:
                self._vectors.popitem(last=False)
            return entry

    def embed(self, text: str) -> tuple[float, ...] | None:
        entry = self._vector(text)
        return entry[0] if entry else None

    def coverage(self, text: str) -> float:
        """Share of the text's words the embeddings know (low for languages they were not made for)."""
        entry = self._vector(text)
        return entry[1] if entry else 0.0

    def similarity(self, a: str, b: str) -> float | None:
        """Cosine of the two meanings (-1..1); None when the embeddings cannot read one of them."""
        first, second = self._vector(a), self._vector(b)
        if first is None or second is None or min(first[1], second[1]) < 0.5:
            return None
        return sum(x * y for x, y in zip(first[0], second[0]))

    def lexical(self, a: str, b: str) -> float:
        """Shared wording (0..1), whatever the language."""
        with self._lock:
            return self._engine.lexsim(normalize(a).encode("utf-8"), normalize(b).encode("utf-8"))

    def closeness(self, a: str, b: str) -> float:
        """One number for ranking: meaning when it can be read, wording otherwise."""
        meaning = self.similarity(a, b)
        wording = self.lexical(a, b)
        return wording if meaning is None else max(meaning, 0.6 * wording)

    def nearest(self, query: str, texts: list[str], limit: int = 3,
                floor: float = MEANING_MATCH) -> list[tuple[int, float]]:
        """The texts that mean something close to the query, in other words or the same:
        (index, cosine) at or above ``floor``, closest first. Empty when the embeddings
        cannot read the query (another language, no known word)."""
        asked = self._vector(query)
        if asked is None or asked[1] < 0.5:
            return []
        hits: list[tuple[int, float]] = []
        for index, text in enumerate(texts):
            entry = self._vector(text)
            if entry is None or entry[1] < 0.5:
                continue
            score = sum(x * y for x, y in zip(asked[0], entry[0]))
            if score >= floor:
                hits.append((index, score))
        hits.sort(key=lambda item: -item[1])
        return hits[:limit]

    def rank(self, query: str, texts: list[str]) -> list[tuple[int, float]]:
        """(index, closeness) for every text, closest first."""
        scored = [(index, self.closeness(query, text)) for index, text in enumerate(texts)]
        return sorted(scored, key=lambda item: -item[1])

    # ---- behaviour ----

    def probe(self, sentence: str) -> dict[str, float]:
        """Probability of each behaviour class for one sentence ({} without a trained probe)."""
        key = normalize(sentence)
        if not key or not self.labels:
            return {}
        with self._lock:
            scores = self._engine.probe(key.encode("utf-8"))
        if not scores:
            return {}
        return dict(zip(self.labels, softmax(scores)))

    def repeat(self, text: str) -> float:
        """Share of four-word runs that already appeared earlier in the text (0..1)."""
        with self._lock:
            return self._engine.repeat(normalize(text).encode("utf-8"))


_shared: Sense | None = None
_shared_lock = threading.Lock()
_shared_failed = False


def shared() -> Sense | None:
    """The app-wide instance, loaded on first use; None when the data file is missing or damaged."""
    global _shared, _shared_failed
    if _shared is None and not _shared_failed:
        with _shared_lock:
            if _shared is None and not _shared_failed:
                try:
                    _shared = Sense()
                except Exception:
                    _shared_failed = True
    return _shared
