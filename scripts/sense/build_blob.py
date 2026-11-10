#!/usr/bin/env python3
"""Build static/sense/sense.bin's vocabulary and embeddings from a model2vec model.

    python3 -I scripts/sense/build_blob.py <folder with model.safetensors and vocab.txt>

The folder is a download of https://huggingface.co/minishlab/potion-base-2M (MIT licence),
revision 389b9f64be5aa4ae7a6bc6fe95ef20ce485ae5da; model.safetensors must have the SHA-256
below. An existing probe (PROB) in sense.bin is kept; train.py writes that part.

Each row is stored as int8 with one 16-bit scale: value ~ int8 / 127 * scale / 65535 * top,
where top is the largest absolute value in the table. Kiwi Sense only ever normalises sums
of rows, so top itself is not stored.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load_blob_module():
    """kiwibro/sense/blob.py on its own, without importing the app around it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("kiwi_sense_blob", ROOT / "kiwibro" / "sense" / "blob.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


blobfile = _load_blob_module()

OUT = ROOT / "static" / "sense" / "sense.bin"
SOURCE = "minishlab/potion-base-2M"
REVISION = "389b9f64be5aa4ae7a6bc6fe95ef20ce485ae5da"
SHA256 = "f95ffde02ad06f63ae38eb9d400038cd5ccaf8411ec3cb650c6025113f96cbb8"


def read_embeddings(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise SystemExit(f"{path} is not the expected file (SHA-256 differs)")
    (head_len,) = struct.unpack_from("<Q", raw, 0)
    head = json.loads(raw[8:8 + head_len])["embeddings"]
    if head["dtype"] != "F32":
        raise SystemExit("expected float32 embeddings")
    start, end = head["data_offsets"]
    data = np.frombuffer(raw, dtype="<f4", count=(end - start) // 4, offset=8 + head_len + start)
    return data.reshape(head["shape"]).astype(np.float64)


def embedding_section(table: np.ndarray) -> bytes:
    peak = np.abs(table).max(axis=1)
    top = float(peak.max())
    scale = np.rint(peak / top * 65535).astype(np.int64)
    safe = np.where(peak > 0, peak, 1.0)
    rows = np.rint(table / safe[:, None] * 127).astype(np.int8)
    return b"".join((
        struct.pack("<II", table.shape[0], table.shape[1]),
        scale.astype("<u2").tobytes(),
        rows.tobytes(),
    ))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    folder = Path(sys.argv[1])
    table = read_embeddings(folder / "model.safetensors")
    tokens = (folder / "vocab.txt").read_text(encoding="utf-8").split("\n")
    if tokens and tokens[-1] == "":
        tokens.pop()
    if len(tokens) != table.shape[0]:
        raise SystemExit(f"{len(tokens)} tokens for {table.shape[0]} rows")
    sections = {"VOCB": blobfile.vocab_section(tokens), "EMBD": embedding_section(table)}
    meta = {
        "embeddings": {"source": SOURCE, "revision": REVISION, "licence": "MIT",
                       "tokens": table.shape[0], "dims": table.shape[1]},
    }
    if OUT.exists():
        old = blobfile.unpack(OUT.read_bytes())
        if "PROB" in old:
            sections["PROB"] = bytes(old["PROB"])
        if "META" in old:
            previous = json.loads(bytes(old["META"]))
            meta = {**previous, **meta}
    sections["META"] = json.dumps(meta, sort_keys=True).encode("utf-8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(blobfile.pack(sections))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes): {table.shape[0]} tokens x {table.shape[1]} dims")


if __name__ == "__main__":
    main()
