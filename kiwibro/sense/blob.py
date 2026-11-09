"""The Kiwi Sense data file: named sections behind a small table of contents.

    "KSNS" | version u32 | section count u32 | (tag[4], offset u32, length u32) per section

All numbers are little-endian. Sections:

    VOCB  word-piece vocabulary as an open-addressing hash table
    EMBD  one int8 row and one u16 scale per vocabulary entry
    PROB  hashed n-gram table and the two layers of the behaviour probe
    META  JSON: class names, where the embeddings came from, thresholds
"""

from __future__ import annotations

import struct

MAGIC = b"KSNS"
VERSION = 1
FNV_SEED = 2166136261
FNV_PRIME = 16777619
MASK = 0xFFFFFFFF


def fnv(data: bytes, seed: int = FNV_SEED) -> int:
    h = seed
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & MASK
    return h


def pack(sections: dict[str, bytes]) -> bytes:
    """Sections in the given order, each starting on an 8-byte boundary."""
    head = 12 + 12 * len(sections)
    table, body, offset = [], [], head
    for tag, data in sections.items():
        raw = tag.encode("ascii")
        if len(raw) != 4:
            raise ValueError(f"section tag must be four ASCII characters: {tag!r}")
        pad = -offset % 8
        body.append(b"\0" * pad)
        offset += pad
        table.append(raw + struct.pack("<II", offset, len(data)))
        body.append(bytes(data))
        offset += len(data)
    return MAGIC + struct.pack("<II", VERSION, len(sections)) + b"".join(table) + b"".join(body)


def unpack(blob: bytes) -> dict[str, memoryview]:
    view = memoryview(blob)
    if len(view) < 12 or bytes(view[:4]) != MAGIC:
        raise ValueError("not a Kiwi Sense data file")
    version, count = struct.unpack_from("<II", view, 4)
    if version != VERSION:
        raise ValueError(f"unsupported Kiwi Sense data version {version}")
    if count > 16 or 12 + 12 * count > len(view):
        raise ValueError("damaged Kiwi Sense data file")
    sections: dict[str, memoryview] = {}
    for index in range(count):
        at = 12 + 12 * index
        tag = bytes(view[at:at + 4]).decode("ascii", "replace")
        offset, length = struct.unpack_from("<II", view, at + 4)
        if offset + length > len(view):
            raise ValueError("damaged Kiwi Sense data file")
        sections[tag] = view[offset:offset + length]
    return sections


def vocab_section(tokens: list[str]) -> bytes:
    """Hash table: slot = 1 + token index (0 = empty), linear probing, FNV-1a of the UTF-8 bytes."""
    encoded = [token.encode("utf-8") for token in tokens]
    if len(set(encoded)) != len(encoded):
        raise ValueError("vocabulary has duplicate tokens")
    size = 1
    while size < 2 * len(encoded):
        size *= 2
    slots = [0] * size
    for index, raw in enumerate(encoded):
        at = fnv(raw) & (size - 1)
        while slots[at]:
            at = (at + 1) & (size - 1)
        slots[at] = index + 1
    offsets, total = [0], 0
    for raw in encoded:
        total += len(raw)
        offsets.append(total)
    return b"".join((
        struct.pack("<III", size, len(encoded), total),
        struct.pack(f"<{size}I", *slots),
        struct.pack(f"<{len(offsets)}I", *offsets),
        b"".join(encoded),
    ))


def vocab_tokens(section: memoryview) -> list[bytes]:
    size, count, pool_len = struct.unpack_from("<III", section, 0)
    offsets_at = 12 + 4 * size
    offsets = struct.unpack_from(f"<{count + 1}I", section, offsets_at)
    pool = bytes(section[offsets_at + 4 * (count + 1):offsets_at + 4 * (count + 1) + pool_len])
    return [pool[offsets[i]:offsets[i + 1]] for i in range(count)]
