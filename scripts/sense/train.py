#!/usr/bin/env python3
"""Train the Kiwi Sense behaviour probe and write it into static/sense/sense.bin.

    python3 scripts/sense/train.py            # train, report, write
    python3 scripts/sense/train.py --dry-run  # train and report only

Run build_blob.py first (the probe reads the embeddings in sense.bin).

The probe is a small network. Its input is the sentence's meaning vector (frozen static
embeddings) joined with the average of learned vectors for the sentence's hashed word
n-grams (so word order counts: "I am ChatGPT" is not "ChatGPT is"). One hidden layer,
one score per class. Everything is trained in float, the n-gram table is then rounded to
int8 and the two layers are tuned once more on the rounded table.

The reported scores come from the finished data file, run through the same Python core
the app uses.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus  # noqa: E402
from kiwibro import sense  # noqa: E402
from kiwibro.sense import blob as blobfile  # noqa: E402
from kiwibro.sense.pyengine import PyEngine  # noqa: E402

OUT = ROOT / "static" / "sense" / "sense.bin"
BUCKETS = 8192
PDIM = 16
HIDDEN = 32
SEED = 11


def featurize(engine: PyEngine, rows: list[tuple[str, int]]):
    vectors, buckets, labels = [], [], []
    for sentence, label in rows:
        text = sense.normalize(sentence).encode("utf-8")
        vector, _ = engine.embed(text)
        vectors.append(vector)
        buckets.append(np.array(engine.feature_buckets(text, BUCKETS), dtype=np.int64))
        labels.append(label)
    return np.array(vectors), buckets, np.array(labels)


def bag(table: np.ndarray, buckets: list[np.ndarray], rng=None, drop: float = 0.0) -> tuple[np.ndarray, list[np.ndarray]]:
    out = np.zeros((len(buckets), table.shape[1]))
    kept = []
    for i, ids in enumerate(buckets):
        if rng is not None and drop and len(ids) > 4:
            ids = ids[rng.random(len(ids)) >= drop]
        kept.append(ids)
        if len(ids):
            out[i] = table[ids].mean(axis=0)
    return out, kept


class Adam:
    def __init__(self, params: list[np.ndarray], rate: float) -> None:
        self.params, self.rate, self.step = params, rate, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def update(self, grads: list[np.ndarray]) -> None:
        self.step += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            p -= self.rate * (m / (1 - 0.9 ** self.step)) / (np.sqrt(v / (1 - 0.999 ** self.step)) + 1e-8)


def train(vectors, buckets, labels, classes: int):
    rng = np.random.default_rng(SEED)
    n, edim = vectors.shape
    table = rng.normal(0, 0.1, (BUCKETS, PDIM))
    w1 = rng.normal(0, np.sqrt(2 / (edim + PDIM)), (HIDDEN, edim + PDIM))
    b1 = np.zeros(HIDDEN)
    w2 = rng.normal(0, np.sqrt(1 / HIDDEN), (classes, HIDDEN))
    b2 = np.zeros(classes)
    targets = np.full((n, classes), 0.02 / (classes - 1))
    targets[np.arange(n), labels] = 0.98

    def run(epochs: int, rate: float, tune_table: bool) -> None:
        params = [w1, b1, w2, b2] + ([table] if tune_table else [])
        adam = Adam(params, rate)
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, 64):
                idx = order[start:start + 64]
                pooled, kept = bag(table, [buckets[i] for i in idx], rng, 0.2 if tune_table else 0.0)
                x = np.concatenate([vectors[idx], pooled], axis=1)
                pre = x @ w1.T + b1
                mid = np.maximum(pre, 0)
                logits = mid @ w2.T + b2
                logits -= logits.max(axis=1, keepdims=True)
                probs = np.exp(logits)
                probs /= probs.sum(axis=1, keepdims=True)
                d_logits = (probs - targets[idx]) / len(idx)
                g_w2 = d_logits.T @ mid + 1e-4 * w2
                g_b2 = d_logits.sum(axis=0)
                d_mid = (d_logits @ w2) * (pre > 0)
                g_w1 = d_mid.T @ x + 1e-4 * w1
                g_b1 = d_mid.sum(axis=0)
                grads = [g_w1, g_b1, g_w2, g_b2]
                if tune_table:
                    d_pooled = d_mid @ w1[:, edim:]
                    g_table = np.zeros_like(table)
                    for row, ids in zip(d_pooled, kept):
                        if len(ids):
                            np.add.at(g_table, ids, row / len(ids))
                    grads.append(g_table)
                adam.update(grads)

    run(30, 0.01, True)
    scale = float(np.float32(np.abs(table).max() / 127))
    table[:] = np.clip(np.rint(table / scale), -127, 127) * scale
    run(8, 0.002, False)
    return np.rint(table / scale).astype(np.int8), scale, w1, b1, w2, b2


def probe_section(table, scale, w1, b1, w2, b2, edim: int) -> bytes:
    return b"".join((
        struct.pack("<5If", BUCKETS, PDIM, edim, HIDDEN, w2.shape[0], scale),
        table.tobytes(),
        w1.astype("<f4").tobytes(), b1.astype("<f4").tobytes(),
        w2.astype("<f4").tobytes(), b2.astype("<f4").tobytes(),
    ))


def score(engine: PyEngine, rows: list[tuple[str, int]], threshold: float) -> dict:
    """Accuracy, plus how the app will see it: a class counts only at or above the threshold."""
    right = flagged_fine = fine = caught = bad = 0
    misses = []
    for sentence, label in rows:
        probs = sense.softmax(engine.probe(sense.normalize(sentence).encode("utf-8")))
        best = max(range(len(probs)), key=probs.__getitem__)
        right += best == label
        flagged = best != corpus.FINE and probs[best] >= threshold
        if label == corpus.FINE:
            fine += 1
            flagged_fine += flagged
        else:
            bad += 1
            caught += flagged and best == label
        if best != label:
            misses.append((sentence, corpus.CLASSES[label], corpus.CLASSES[best], round(probs[best], 2)))
    return {
        "sentences": len(rows), "accuracy": round(right / len(rows), 4),
        "fine_flagged": flagged_fine, "fine_total": fine, "caught": caught, "bad_total": bad, "misses": misses,
    }


def main() -> None:
    dry = "--dry-run" in sys.argv
    sections = {tag: bytes(data) for tag, data in blobfile.unpack(OUT.read_bytes()).items()}
    sections.pop("PROB", None)
    engine = PyEngine(blobfile.pack(sections))
    rows = corpus.build()
    cut = len(rows) // 10
    held, training = rows[:cut], rows[cut:]
    vectors, buckets, labels = featurize(engine, training)
    print(f"training on {len(training)} sentences, {len(held)} set aside, {len(corpus.HELD_OUT)} handwritten")
    table, scale, w1, b1, w2, b2 = train(vectors, buckets, labels, len(corpus.CLASSES))
    meta = json.loads(sections.get("META", b"{}"))
    threshold = 0.8
    sections["PROB"] = probe_section(table, scale, w1, b1, w2, b2, engine.dims)
    sections.pop("META", None)
    final = PyEngine(blobfile.pack(sections))
    generated = score(final, held, threshold)
    handwritten = score(final, corpus.HELD_OUT, threshold)
    for name, result in (("generated, set aside", generated), ("handwritten", handwritten)):
        print(f"{name}: accuracy {result['accuracy']}, flagged fine {result['fine_flagged']}/{result['fine_total']}, "
              f"caught {result['caught']}/{result['bad_total']} at p>={threshold}")
    for miss in handwritten["misses"]:
        print("   miss:", miss)
    meta["classes"] = corpus.CLASSES
    meta["probe"] = {
        "buckets": BUCKETS, "ngram_dims": PDIM, "hidden": HIDDEN, "threshold": threshold,
        "trained_on": len(training), "language": "en",
        "handwritten_test": {key: handwritten[key] for key in ("sentences", "accuracy", "fine_flagged", "fine_total", "caught", "bad_total")},
    }
    sections["META"] = json.dumps(meta, sort_keys=True).encode("utf-8")
    if not dry:
        OUT.write_bytes(blobfile.pack(sections))
        print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
