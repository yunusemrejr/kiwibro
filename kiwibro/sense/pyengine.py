"""The Kiwi Sense core in plain Python: the twin of core.c, bit for bit.

Used when no WebAssembly runtime is installed. Every step keeps the same order of
operations as the C code, so both give exactly the same numbers (tests compare them).
"""

from __future__ import annotations

import math
import struct

from . import blob as blobfile
from .blob import FNV_PRIME, FNV_SEED, MASK, fnv

MAX_WORDS = 256
MAX_PIECES = 100
REPEAT_WORDS = 1024
LEX_BINS = 1024


def mix(a: int, b: int) -> int:
    h = a ^ ((b + 0x9E3779B9 + ((a << 6) & MASK) + (a >> 2)) & MASK)
    h = (h * FNV_PRIME) & MASK
    return h ^ (h >> 15)


class PyEngine:
    name = "python"

    def __init__(self, blob: bytes) -> None:
        sections = blobfile.unpack(blob)
        if "VOCB" not in sections or "EMBD" not in sections:
            raise ValueError("Kiwi Sense data file has no vocabulary or embeddings")
        self._vocab = {token: index for index, token in enumerate(blobfile.vocab_tokens(sections["VOCB"]))}
        embd = sections["EMBD"]
        count, dims = struct.unpack_from("<II", embd, 0)
        if count != len(self._vocab):
            raise ValueError("Kiwi Sense vocabulary and embeddings disagree")
        self.dims = dims
        self._scale = struct.unpack_from(f"<{count}H", embd, 8)
        self._rows = embd[8 + 2 * count:8 + 2 * count + count * dims].cast("b")
        self.classes = 0
        if "PROB" in sections:
            self._load_probe(sections["PROB"])
        self.last_words = 0

    def _load_probe(self, prob: memoryview) -> None:
        buckets, pdim, edim, hidden, classes = struct.unpack_from("<5I", prob, 0)
        if edim != self.dims:
            raise ValueError("Kiwi Sense probe was trained for other embeddings")
        (scale,) = struct.unpack_from("<f", prob, 20)
        inputs = edim + pdim
        at = 24
        self._ptable = prob[at:at + buckets * pdim].cast("b")
        at += buckets * pdim
        w1 = struct.unpack_from(f"<{hidden * inputs}f", prob, at)
        at += 4 * hidden * inputs
        self._b1 = struct.unpack_from(f"<{hidden}f", prob, at)
        at += 4 * hidden
        w2 = struct.unpack_from(f"<{classes * hidden}f", prob, at)
        at += 4 * classes * hidden
        self._b2 = struct.unpack_from(f"<{classes}f", prob, at)
        self._w1 = [w1[j * inputs:(j + 1) * inputs] for j in range(hidden)]
        self._w2 = [w2[c * hidden:(c + 1) * hidden] for c in range(classes)]
        self._buckets, self._pdim, self._pscale = buckets, pdim, scale
        self.classes = classes

    # ---- word pieces ----

    def _wordpiece(self, word: bytes) -> list[int]:
        chars = sum(1 for byte in word if byte & 0xC0 != 0x80)
        if not chars or chars > 100:
            return []
        ids: list[int] = []
        start, n = 0, len(word)
        while start < n:
            end, found = n, -1
            while end > start:
                piece = word[start:end]
                found = self._vocab.get(b"##" + piece if start else piece, -1)
                if found >= 0:
                    break
                end -= 1
                while end > start and word[end] & 0xC0 == 0x80:
                    end -= 1
            if found < 0 or len(ids) >= MAX_PIECES:
                return []
            ids.append(found)
            start = end
        return ids

    # ---- meaning ----

    def embed(self, text: bytes) -> tuple[list[float], int]:
        """(unit vector, number of recognised words); the vector is all zeros when none is."""
        dims, rows, scales = self.dims, self._rows, self._scale
        total = [0] * dims
        words = known = 0
        for word in text.split(b" "):
            if not word:
                continue
            words += 1
            ids = self._wordpiece(word)
            if not ids:
                continue
            known += 1
            for token in ids:
                scale = scales[token]
                if not scale:
                    continue
                row = rows[token * dims:(token + 1) * dims]
                for d in range(dims):
                    total[d] += row[d] * scale
        self.last_words = words
        norm = 0.0
        for value in total:
            as_float = float(value)
            norm += as_float * as_float
        if norm > 0.0:
            norm = math.sqrt(norm)
            return [float(value) / norm for value in total], known
        return [0.0] * dims, known

    def similarity(self, a: bytes, b: bytes) -> float:
        va, _ = self.embed(a)
        vb, _ = self.embed(b)
        dot = 0.0
        for d in range(self.dims):
            dot += va[d] * vb[d]
        return dot

    # ---- behaviour probe ----

    def probe(self, text: bytes) -> list[float] | None:
        """Raw class scores (logits), or None when the data file carries no probe."""
        if not self.classes:
            return None
        vector, _ = self.embed(text)
        hashes = [fnv(word) for word in text.split(b" ") if word][:MAX_WORDS]
        n = len(hashes)
        features: list[int] = []
        for k in range(n):
            features.append(mix(1, hashes[k]))
            if k + 1 < n:
                features.append(mix(2, mix(hashes[k], hashes[k + 1])))
            if k + 2 < n:
                features.append(mix(3, mix(mix(hashes[k], hashes[k + 1]), hashes[k + 2])))
        if n:
            features.append(mix(4, hashes[0]))
            features.append(mix(6, hashes[n - 1]))
        if n > 1:
            features.append(mix(5, mix(hashes[0], hashes[1])))
            features.append(mix(7, mix(hashes[n - 2], hashes[n - 1])))
        pdim, table = self._pdim, self._ptable
        sums = [0] * pdim
        for h in features:
            at = (h % self._buckets) * pdim
            row = table[at:at + pdim]
            for d in range(pdim):
                sums[d] += row[d]
        count = len(features)
        inputs = vector + [sums[d] * self._pscale / count if count else 0.0 for d in range(pdim)]
        mid = []
        for j, weights in enumerate(self._w1):
            total = self._b1[j]
            for d, x in enumerate(inputs):
                total += weights[d] * x
            mid.append(total if total > 0.0 else 0.0)
        out = []
        for c, weights in enumerate(self._w2):
            total = self._b2[c]
            for j, x in enumerate(mid):
                total += weights[j] * x
            out.append(total)
        return out

    def feature_buckets(self, text: bytes, buckets: int) -> list[int]:
        """The probe's hashed n-gram buckets for a text (training uses this)."""
        hashes = [fnv(word) for word in text.split(b" ") if word][:MAX_WORDS]
        n = len(hashes)
        out: list[int] = []
        for k in range(n):
            out.append(mix(1, hashes[k]) % buckets)
            if k + 1 < n:
                out.append(mix(2, mix(hashes[k], hashes[k + 1])) % buckets)
            if k + 2 < n:
                out.append(mix(3, mix(mix(hashes[k], hashes[k + 1]), hashes[k + 2])) % buckets)
        if n:
            out.append(mix(4, hashes[0]) % buckets)
            out.append(mix(6, hashes[n - 1]) % buckets)
        if n > 1:
            out.append(mix(5, mix(hashes[0], hashes[1])) % buckets)
            out.append(mix(7, mix(hashes[n - 2], hashes[n - 1])) % buckets)
        return out

    # ---- loops ----

    def repeat(self, text: bytes) -> float:
        hashes = [fnv(word) for word in text.split(b" ") if word][-REPEAT_WORDS:]
        n = len(hashes)
        if n < 12:
            return 0.0
        seen: set[int] = set()
        repeated, runs = 0, n - 3
        for k in range(runs):
            h = mix(mix(mix(hashes[k], hashes[k + 1]), hashes[k + 2]), hashes[k + 3]) | 1
            if h in seen:
                repeated += 1
            else:
                seen.add(h)
        return repeated / runs

    # ---- wording overlap ----

    @staticmethod
    def _trigrams(text: bytes) -> list[int]:
        bins = [0] * LEX_BINS
        for i in range(len(text) - 2):
            h = FNV_SEED
            for byte in text[i:i + 3]:
                h = ((h ^ byte) * FNV_PRIME) & MASK
            bins[h & (LEX_BINS - 1)] += 1
        return bins

    def lexsim(self, a: bytes, b: bytes) -> float:
        bins_a, bins_b = self._trigrams(a), self._trigrams(b)
        dot = na = nb = 0
        for x, y in zip(bins_a, bins_b):
            dot += x * y
            na += x * x
            nb += y * y
        if not na or not nb:
            return 0.0
        return float(dot) / (math.sqrt(float(na)) * math.sqrt(float(nb)))
