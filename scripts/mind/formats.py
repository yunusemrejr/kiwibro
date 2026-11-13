"""File formats of Kiwi Mind and the code that writes them.

Three binary files, all little-endian. This module is the writer and a plain reader of each; the C engine
(kiwibro/mind/engine/kiwimind.c) is the other reader. It needs numpy and runs on training machines and in
tests, never inside Kiwibro.

TRUNK (.fkm, version 1 or 2) -- the shared network, a Qwen3-style decoder with full attention.
    200-byte header, then one 72-byte record per tensor, then the tensor data, each tensor 64-byte aligned.
    Header: u32 magic "FKM1" (0x314D4B46), version, header_bytes (200 + 72 * n_tensors), n_tensors, vocab,
    dim, n_layers, n_heads, n_kv_heads, head_dim, ffn_dim, window (0 = full attention), max_ctx, bos_id,
    eos_id, flags (0); then f32 rope_theta, norm_eps; then 64 bytes layer_global (1 per layer); then
    u32 recall_layer, recall_bits, 14 reserved (all 0 here).
    Record: 40 bytes name (NUL padded), u32 type (0 f32, 2 q8, 3 q4), rows, cols, pad, u64 offset, u64 bytes.
    Order: embedding, then per layer attn_norm, q_norm, k_norm, q, k, v, o, ffn_norm, gate, up, down,
    then the final norm. Matrices are [out, in] row-major. Q8 data: rows*cols int8 codes, then one fp16
    scale per 32 values of a row. Q4 data: rows*cols/2 bytes, then the same scales; in each group of 32
    values byte j holds value j in its low nibble and value j+16 in its high nibble, stored as code + 8
    (code in -8..7). A q4 embedding makes the file version 2, as in the Flyingkiwi converter.

SKILL (.kms) -- low-rank deltas for the upper layers; y = W x + scale * B (A x), A [r, in], B [out, r].
    Header, 96 bytes:  0 magic "KMS1" | 4 u32 version (1) | 8 name, 64 bytes UTF-8, NUL padded |
    72 u32 first_layer | 76 u32 n_entries | 80 u32 n_layers (of the trunk it was made for) | 84..95 zero.
    Then n_entries records of 32 bytes: u32 layer | u32 matrix (0 q, 1 k, 2 v, 3 o, 4 gate, 5 up, 6 down) |
    u32 rank (1..128) | f32 scale (= alpha / rank) | u64 a_offset | u64 b_offset.
    Offsets are from the start of the file, 64-byte aligned; A is rank*in fp16, B is out*rank fp16, both
    row-major. Every layer is first_layer..n_layers-1, each (layer, matrix) at most once, ranks may differ.

HEAD (.kmh) -- the control head: out = W2 relu(W1 rmsnorm(x) + b1) + b2 on one hidden state.
    Header, 64 bytes: 0 magic "KMH1" | 4 u32 version (1) | 8 u32 layer (the state at the input of this
    layer is read; 0xFFFFFFFF = the first layer of the skills) | 12 u32 in_dim | 16 u32 hidden | 20 u32 n_out |
    24 f32 eps | 28 u32 json_offset | 32 u32 json_length | 36..63 zero.
    Then f32 W1 [hidden, in_dim], b1 [hidden], W2 [n_out, hidden], b2 [n_out] from byte 64, and at
    json_offset (4-byte aligned, the end of the file) a UTF-8 JSON object {"names": [one class name per output]}.
    rmsnorm(x) = x / sqrt(mean(x^2) + eps), without a weight.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

MAGIC_TRUNK = 0x314D4B46
MAGIC_SKILL = b"KMS1"
MAGIC_HEAD = b"KMH1"
F32, Q8, Q4 = 0, 2, 3
GROUP = 32
HEADER = struct.Struct("<16I2f64B16I")
TENSOR = struct.Struct("<40s4I2Q")
SKILL_HEADER = struct.Struct("<4sI64s3I12x")
SKILL_ENTRY = struct.Struct("<3If2Q")
HEAD_HEADER = struct.Struct("<4s3I2If2I28x")
assert HEADER.size == 200 and TENSOR.size == 72 and SKILL_HEADER.size == 96 and SKILL_ENTRY.size == 32
assert HEAD_HEADER.size == 64
MATRICES = ["q", "k", "v", "o", "gate", "up", "down"]
MAX_RANK = 128
LAYER_PARTS = ["input_layernorm", "self_attn.q_norm", "self_attn.k_norm", "self_attn.q_proj", "self_attn.k_proj",
               "self_attn.v_proj", "self_attn.o_proj", "post_attention_layernorm", "mlp.gate_proj", "mlp.up_proj",
               "mlp.down_proj"]
MATRIX_PARTS = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
                "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
CONFIG_KEYS = ["vocab", "dim", "n_layers", "n_heads", "n_kv_heads", "head_dim", "ffn_dim", "rope_theta", "norm_eps",
               "bos_id", "eos_id"]
QWEN3_0_6B = dict(vocab=151936, dim=1024, n_layers=28, n_heads=16, n_kv_heads=8, head_dim=128, ffn_dim=3072,
                  rope_theta=1e6, norm_eps=1e-6, bos_id=151643, eos_id=151645)
TINY = dict(vocab=300, dim=64, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=32, ffn_dim=128, rope_theta=1e6,
            norm_eps=1e-6, bos_id=297, eos_id=299)


def matrix_shape(config: dict, matrix: str) -> tuple[int, int]:
    """(out, in) of one of the seven linear layers."""
    d, f, q, kv = config["dim"], config["ffn_dim"], config["n_heads"] * config["head_dim"], \
        config["n_kv_heads"] * config["head_dim"]
    return {"q": (q, d), "k": (kv, d), "v": (kv, d), "o": (d, q), "gate": (f, d), "up": (f, d), "down": (d, f)}[matrix]


def config_from_hf(cfg: dict) -> dict:
    """The trunk config of a Hugging Face Qwen3 config.json."""
    eos = cfg["eos_token_id"][0] if isinstance(cfg["eos_token_id"], list) else cfg["eos_token_id"]
    return dict(vocab=cfg["vocab_size"], dim=cfg["hidden_size"], n_layers=cfg["num_hidden_layers"],
                n_heads=cfg["num_attention_heads"], n_kv_heads=cfg["num_key_value_heads"],
                head_dim=cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
                ffn_dim=cfg["intermediate_size"], rope_theta=float(cfg.get("rope_theta", 1e6)),
                norm_eps=float(cfg.get("rms_norm_eps", 1e-6)), bos_id=cfg.get("bos_token_id") or 0, eos_id=eos)


# ---------------------------------------------------------------- quantisation

def quantize_q4(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Codes (int8, -8..7) and fp16 scales (max|w| / 7 per group of 32), as the Flyingkiwi converter does."""
    rows, cols = w.shape
    g = w.astype(np.float32).reshape(rows, cols // GROUP, GROUP)
    s = np.maximum(np.abs(g).max(-1) / np.float32(7.0), np.float32(2.0 ** -14)).astype(np.float16)
    q = np.clip(np.round(g / s.astype(np.float32)[..., None]), -8, 7).astype(np.int8)
    return q.reshape(rows, cols), s


def quantize_q8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = w.shape
    g = w.astype(np.float32).reshape(rows, cols // GROUP, GROUP)
    s = np.maximum(np.abs(g).max(-1) / np.float32(127.0), np.float32(2.0 ** -14)).astype(np.float16)
    q = np.clip(np.round(g / s.astype(np.float32)[..., None]), -127, 127).astype(np.int8)
    return q.reshape(rows, cols), s


def dequantize(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    rows, cols = codes.shape
    g = codes.astype(np.float32).reshape(rows, cols // GROUP, GROUP) * scales.astype(np.float32)[..., None]
    return g.reshape(rows, cols)


def pack_q4(codes: np.ndarray, scales: np.ndarray) -> bytes:
    rows, cols = codes.shape
    g = (codes.astype(np.int16) + 8).astype(np.uint8).reshape(rows, cols // GROUP, GROUP)
    return (g[..., :16] | (g[..., 16:] << 4)).tobytes() + scales.astype("<f2").tobytes()


def pack_q8(codes: np.ndarray, scales: np.ndarray) -> bytes:
    return codes.astype(np.int8).tobytes() + scales.astype("<f2").tobytes()


def unpack(type_: int, rows: int, cols: int, blob: bytes) -> np.ndarray:
    """A tensor's data as float32 (what the engine computes with, before its own activation rounding)."""
    if type_ == F32:
        return np.frombuffer(blob, "<f4").reshape(rows, cols).astype(np.float32)
    ng = cols // GROUP
    n_codes = rows * cols // (2 if type_ == Q4 else 1)
    scales = np.frombuffer(blob, "<f2", count=rows * ng, offset=n_codes).reshape(rows, ng)
    if type_ == Q8:
        return dequantize(np.frombuffer(blob, np.int8, count=n_codes).reshape(rows, cols), scales)
    packed = np.frombuffer(blob, np.uint8, count=n_codes).reshape(rows, ng, 16)
    q = np.concatenate([(packed & 15).astype(np.int8) - 8, (packed >> 4).astype(np.int8) - 8], axis=-1)
    return dequantize(q.reshape(rows, cols), scales)


# ---------------------------------------------------------------- trunk

Blob = Callable[[], bytes]


def _trunk_specs(config: dict, emb: tuple[int, Blob], layer: Callable[[int, str], tuple[int, int, int, Blob]],
                 final_norm: Blob) -> list[tuple[str, int, int, int, Blob]]:
    d, v = config["dim"], config["vocab"]
    specs = [("embed", emb[0], v, d, emb[1])]
    for i in range(config["n_layers"]):
        for part in LAYER_PARTS:
            t, r, c, blob = layer(i, part)
            specs.append((f"model.layers.{i}.{part}", t, r, c, blob))
    specs.append(("model.norm", F32, 1, d, final_norm))
    return specs


def _write_trunk_file(path: Path, config: dict, specs: list, max_ctx: int, version: int) -> None:
    header_bytes = HEADER.size + TENSOR.size * len(specs)
    offset = (header_bytes + 63) // 64 * 64
    records, offsets = [], []
    for name, typ, rows, cols, _ in specs:
        size = _tensor_bytes(typ, rows, cols)
        records.append(TENSOR.pack(name.encode()[:39], typ, rows, cols, 0, offset, size))
        offsets.append(offset)
        offset = (offset + size + 63) // 64 * 64
    layers = config["n_layers"]
    header = HEADER.pack(MAGIC_TRUNK, version, header_bytes, len(specs), config["vocab"], config["dim"], layers,
                         config["n_heads"], config["n_kv_heads"], config["head_dim"], config["ffn_dim"], 0, max_ctx,
                         config["bos_id"], config["eos_id"], 0, float(config["rope_theta"]),
                         float(config["norm_eps"]), *([1] * layers + [0] * (64 - layers)), 0, 0, *([0] * 14))
    with open(path, "wb") as fh:
        fh.write(header)
        for record in records:
            fh.write(record)
        for (name, typ, rows, cols, blob), at in zip(specs, offsets):
            fh.write(b"\0" * (at - fh.tell()))
            data = blob()
            if len(data) != _tensor_bytes(typ, rows, cols):
                raise ValueError(f"{name}: {len(data)} bytes, expected {_tensor_bytes(typ, rows, cols)}")
            fh.write(data)


def _tensor_bytes(typ: int, rows: int, cols: int) -> int:
    if typ == F32:
        return rows * cols * 4
    if cols % GROUP:
        raise ValueError(f"quantised tensors need cols divisible by {GROUP}, got {cols}")
    return rows * cols // (2 if typ == Q4 else 1) + rows * cols // GROUP * 2


def write_trunk(path: Path, config: dict, weights: dict[str, np.ndarray], mode: str = "q4", emb: str = "q8",
                max_ctx: int = 4096, keep8: Iterable[str] = ()) -> None:
    """Quantise float weights (Hugging Face names, [out, in] matrices) and write a trunk.

    mode: q4 | q8 | f32 for the seven linear layers per layer; keep8 names linear tensors that stay q8 in q4 mode;
    emb: q8 | q4 | f32 for the tied embedding and output matrix."""
    keep8 = set(keep8)
    pack = {Q4: lambda w: pack_q4(*quantize_q4(w)), Q8: lambda w: pack_q8(*quantize_q8(w)),
            F32: lambda w: w.astype("<f4").tobytes()}
    emb_type = {"q8": Q8, "q4": Q4, "f32": F32}[emb]
    linear_type = {"q4": Q4, "q8": Q8, "f32": F32}[mode]

    def layer(i: int, part: str):
        name = f"model.layers.{i}.{part}"
        w = weights[f"{name}.weight"]
        if not part.endswith("proj"):
            return F32, 1, w.size, lambda: w.astype("<f4").tobytes()
        typ = Q8 if name in keep8 and mode == "q4" else linear_type
        return typ, w.shape[0], w.shape[1], lambda: pack[typ](w)

    table = weights["model.embed_tokens.weight"][:config["vocab"]]
    specs = _trunk_specs(config, (emb_type, lambda: pack[emb_type](table)), layer,
                         lambda: weights["model.norm.weight"].astype("<f4").tobytes())
    _write_trunk_file(Path(path), config, specs, max_ctx, 2 if emb_type == Q4 else 1)


def write_random_trunk(path: Path, config: dict, mode: str = "q4", emb: str = "q8", seed: int = 0,
                       weight_std: float = 0.08, emb_std: float = 0.6, max_ctx: int = 4096) -> None:
    """A trunk of random codes and scales, streamed tensor by tensor (a full-size one never sits in memory).

    The weights are meaningless, but the logits have a usable spread, so the file exercises every code path
    of the engine and gives honest timings. weight_std and emb_std are the standard deviations of the
    dequantised matrices."""
    rng = np.random.default_rng(seed)
    emb_type = {"q8": Q8, "q4": Q4, "f32": F32}[emb]
    linear_type = {"q4": Q4, "q8": Q8, "f32": F32}[mode]

    def random_blob(typ: int, rows: int, cols: int, std: float) -> bytes:
        if typ == F32:
            return (rng.standard_normal((rows, cols), dtype=np.float32) * std).astype("<f4").tobytes()
        spread = 4.6 if typ == Q4 else 73.3  # standard deviation of uniform codes
        scales = (std / spread * rng.uniform(0.6, 1.4, (rows, cols // GROUP))).astype("<f2").tobytes()
        if typ == Q8:
            codes = rng.integers(-127, 128, (rows, cols), dtype=np.int8).tobytes()
        else:
            codes = rng.integers(0, 256, rows * cols // 2, dtype=np.uint8).tobytes()
        return codes + scales

    def norm(n: int, jitter: float = 0.1) -> bytes:
        return (1.0 + jitter * rng.standard_normal(n)).astype("<f4").tobytes()

    def layer(i: int, part: str):
        hd, d = config["head_dim"], config["dim"]
        if part in ("input_layernorm", "post_attention_layernorm"):
            return F32, 1, d, lambda: norm(d)
        if part in ("self_attn.q_norm", "self_attn.k_norm"):
            return F32, 1, hd, lambda: norm(hd)
        matrix = next(m for m, p in MATRIX_PARTS.items() if p == part)
        rows, cols = matrix_shape(config, matrix)
        return linear_type, rows, cols, lambda: random_blob(linear_type, rows, cols, weight_std)

    specs = _trunk_specs(config, (emb_type, lambda: random_blob(emb_type, config["vocab"], config["dim"], emb_std)),
                         layer, lambda: norm(config["dim"]))
    _write_trunk_file(Path(path), config, specs, max_ctx, 2 if emb_type == Q4 else 1)


def read_trunk(path: Path) -> tuple[dict, dict[str, np.ndarray], dict[str, int]]:
    """(config, tensors as float32 keyed by tensor name, tensor types). Matrices are the dequantised values."""
    data = Path(path).read_bytes()
    h = HEADER.unpack_from(data, 0)
    if h[0] != MAGIC_TRUNK or h[1] not in (1, 2):
        raise ValueError(f"{path}: not a trunk file")
    config = dict(zip(CONFIG_KEYS[:7], h[4:11]))
    config.update(rope_theta=h[16], norm_eps=h[17], bos_id=h[13], eos_id=h[14], max_ctx=h[12], window=h[11])
    tensors, types = {}, {}
    for i in range(h[3]):
        name, typ, rows, cols, _, offset, size = TENSOR.unpack_from(data, HEADER.size + i * TENSOR.size)
        name = name.split(b"\0")[0].decode()
        tensors[name] = unpack(typ, rows, cols, data[offset:offset + size])
        types[name] = typ
    return config, tensors, types


# ---------------------------------------------------------------- skills

def _name_bytes(name: str) -> bytes:
    raw = name.encode()
    if not raw or len(raw) > 64:
        raise ValueError("a skill name is 1..64 bytes")
    return raw


def write_skill(path: Path, name: str, first_layer: int, n_layers: int, entries: list[dict]) -> None:
    """entries: dicts with layer, matrix (q|k|v|o|gate|up|down), a [rank, in], b [out, rank], scale (alpha / rank)."""
    seen = set()
    for e in entries:
        key = (e["layer"], e["matrix"])
        if key in seen or not first_layer <= e["layer"] < n_layers or e["matrix"] not in MATRICES:
            raise ValueError(f"bad or repeated skill entry {key}")
        if not 1 <= e["a"].shape[0] <= MAX_RANK or e["b"].shape[1] != e["a"].shape[0]:
            raise ValueError(f"bad rank in skill entry {key}")
        seen.add(key)
    offset = (SKILL_HEADER.size + SKILL_ENTRY.size * len(entries) + 63) // 64 * 64
    records, blobs = [], []
    for e in entries:
        a, b = e["a"].astype("<f2").tobytes(), e["b"].astype("<f2").tobytes()
        a_at = offset
        b_at = (a_at + len(a) + 63) // 64 * 64
        offset = (b_at + len(b) + 63) // 64 * 64
        records.append(SKILL_ENTRY.pack(e["layer"], MATRICES.index(e["matrix"]), e["a"].shape[0], float(e["scale"]),
                                        a_at, b_at))
        blobs += [(a_at, a), (b_at, b)]
    with open(path, "wb") as fh:
        fh.write(SKILL_HEADER.pack(MAGIC_SKILL, 1, _name_bytes(name), first_layer, len(entries), n_layers))
        for record in records:
            fh.write(record)
        for at, blob in blobs:
            fh.write(b"\0" * (at - fh.tell()))
            fh.write(blob)
        fh.write(b"\0" * (offset - fh.tell()))


def read_skill(path: Path) -> dict:
    data = Path(path).read_bytes()
    magic, version, name, first_layer, n_entries, n_layers = SKILL_HEADER.unpack_from(data, 0)
    if magic != MAGIC_SKILL or version != 1:
        raise ValueError(f"{path}: not a skill file")
    entries = []
    for i in range(n_entries):
        layer, matrix, rank, scale, a_at, b_at = SKILL_ENTRY.unpack_from(data, SKILL_HEADER.size + i * SKILL_ENTRY.size)
        entries.append(dict(layer=layer, matrix=MATRICES[matrix], rank=rank, scale=scale, a_at=a_at, b_at=b_at))
    return dict(name=name.rstrip(b"\0").decode(), first_layer=first_layer, n_layers=n_layers, entries=entries,
                raw=data)


def skill_deltas(skill: dict, config: dict) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray, float]]:
    """{(layer, matrix): (A float32 [r, in], B float32 [out, r], scale)} from fp16 storage."""
    out = {}
    for e in skill["entries"]:
        rows, cols = matrix_shape(config, e["matrix"])
        r = e["rank"]
        a = np.frombuffer(skill["raw"], "<f2", count=r * cols, offset=e["a_at"]).reshape(r, cols).astype(np.float32)
        b = np.frombuffer(skill["raw"], "<f2", count=rows * r, offset=e["b_at"]).reshape(rows, r).astype(np.float32)
        out[(e["layer"], e["matrix"])] = (a, b, e["scale"])
    return out


def random_skill_entries(config: dict, first_layer: int, rank: int | dict[str, int], matrices: Iterable[str] = MATRICES,
                         seed: int = 0, strength: float = 0.05, alpha: float = 32.0) -> list[dict]:
    """Random low-rank deltas for layers first_layer..n_layers-1. rank is one number or per matrix."""
    rng = np.random.default_rng(seed)
    entries = []
    for layer in range(first_layer, config["n_layers"]):
        for matrix in matrices:
            r = rank[matrix] if isinstance(rank, dict) else rank
            rows, cols = matrix_shape(config, matrix)
            a = (rng.standard_normal((r, cols)) / np.sqrt(cols)).astype(np.float16)
            b = (rng.standard_normal((rows, r)) * strength).astype(np.float16)
            entries.append(dict(layer=layer, matrix=matrix, a=a, b=b, scale=alpha / r))
    return entries


# ---------------------------------------------------------------- head

def write_head(path: Path, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray, names: list[str],
               layer: int | None = None, eps: float = 1e-6) -> None:
    """layer None reads at the first layer of the skills (the split point)."""
    hidden, in_dim = w1.shape
    n_out = w2.shape[0]
    if b1.shape != (hidden,) or w2.shape[1] != hidden or b2.shape != (n_out,) or len(names) != n_out:
        raise ValueError("head shapes do not fit together")
    floats = b"".join(a.astype("<f4").tobytes() for a in (w1, b1, w2, b2))
    blob = json.dumps({"names": names}).encode()
    json_at = HEAD_HEADER.size + len(floats)
    with open(path, "wb") as fh:
        fh.write(HEAD_HEADER.pack(MAGIC_HEAD, 1, 0xFFFFFFFF if layer is None else layer, in_dim, hidden, n_out, eps,
                                  json_at, len(blob)))
        fh.write(floats)
        fh.write(blob)


def read_head(path: Path) -> dict:
    data = Path(path).read_bytes()
    magic, version, layer, in_dim, hidden, n_out, eps, json_at, json_len = HEAD_HEADER.unpack_from(data, 0)
    if magic != MAGIC_HEAD or version != 1:
        raise ValueError(f"{path}: not a head file")
    floats = np.frombuffer(data, "<f4", count=hidden * in_dim + hidden + n_out * hidden + n_out, offset=HEAD_HEADER.size)
    parts, at = [], 0
    for shape in ((hidden, in_dim), (hidden,), (n_out, hidden), (n_out,)):
        n = int(np.prod(shape))
        parts.append(floats[at:at + n].reshape(shape))
        at += n
    names = json.loads(data[json_at:json_at + json_len])["names"]
    return dict(layer=None if layer == 0xFFFFFFFF else layer, eps=eps, w1=parts[0], b1=parts[1], w2=parts[2],
                b2=parts[3], names=names)


def random_head(config: dict, names: list[str], hidden: int = 32, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    d = config["dim"]
    return ((rng.standard_normal((hidden, d)) / np.sqrt(d)).astype(np.float32), rng.standard_normal(hidden).astype(np.float32) * 0.1,
            (rng.standard_normal((len(names), hidden)) / np.sqrt(hidden)).astype(np.float32),
            rng.standard_normal(len(names)).astype(np.float32) * 0.1)


# ---------------------------------------------------------------- a tiny tokenizer

QWEN_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
              r"\s+(?!\S)|\s+")
TINY_TEXT = ("the quick brown fox jumps over the lazy dog. hello world! what is the weather today? "
             "the user asks about the weather in paris and the tool returns the answer. "
             "def add(a, b): return a + b  print(\"hello\") {\"name\": \"kiwi\", \"value\": 42} "
             "https://example.com/path?q=1 is the answer to the question")


def _byte_table() -> dict[int, str]:
    """GPT-2's byte -> printable character table, in the order Qwen numbers its 256 byte tokens."""
    keep = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    extra = [b for b in range(256) if b not in keep]
    return {b: chr(b) for b in keep} | {b: chr(256 + i) for i, b in enumerate(extra)}


def tiny_tokenizer(vocab_size: int = 300, specials: tuple[str, ...] = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")) -> dict:
    """A byte-level BPE tokenizer.json in Qwen's format: 256 byte tokens, learned merges, then the specials
    (<|endoftext|>, <|im_start|>, <|im_end|> get the last three ids)."""
    to_char = _byte_table()
    vocab = {c: i for i, c in enumerate(to_char.values())}
    import re
    words: dict[tuple, int] = {}
    for w in re.findall(r"\s?\w+|\s?[^\w\s]+|\s+", TINY_TEXT * 3):
        key = tuple(to_char[b] for b in w.encode())
        words[key] = words.get(key, 0) + 1
    merges = []
    while len(vocab) < vocab_size - len(specials):
        pairs: dict[tuple, int] = {}
        for word, n in words.items():
            for pair in zip(word, word[1:]):
                pairs[pair] = pairs.get(pair, 0) + n
        if not pairs:
            raise ValueError("the training text is too short for that vocabulary")
        best = max(sorted(pairs), key=pairs.get)
        merges.append(list(best))
        vocab["".join(best)] = len(vocab)
        merged = {}
        for word, n in words.items():
            out, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                    out.append(word[i] + word[i + 1])
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            merged[tuple(out)] = merged.get(tuple(out), 0) + n
        words = merged
    added = [dict(id=len(vocab) + i, content=s, single_word=False, lstrip=False, rstrip=False, normalized=False,
                  special=True) for i, s in enumerate(specials)]
    byte_level = dict(type="ByteLevel", add_prefix_space=False, trim_offsets=False, use_regex=False)
    return dict(version="1.0", truncation=None, padding=None, added_tokens=added, normalizer=dict(type="NFC"),
                pre_tokenizer=dict(type="Sequence", pretokenizers=[
                    dict(type="Split", pattern=dict(Regex=QWEN_SPLIT), behavior="Isolated", invert=False), byte_level]),
                post_processor=byte_level, decoder=byte_level,
                model=dict(type="BPE", dropout=None, unk_token=None, continuing_subword_prefix="", end_of_word_suffix="",
                           fuse_unk=False, byte_fallback=False, ignore_merges=False, vocab=vocab, merges=merges))


def tiny_encode(tokenizer: dict, text: str) -> list[int]:
    """Reference encoder for the tiny tokenizer: an independent, deliberately plain implementation (ranked pair
    merging, one merge at a time) that the tokenizer in the host is tested against. ASCII text only: the split
    pattern here spells \\p{L} and \\p{N} as ASCII classes, which is all the tiny vocabulary can tell apart."""
    import re
    to_char = _byte_table()
    vocab = tokenizer["model"]["vocab"]
    rank = {tuple(pair): i for i, pair in enumerate(tokenizer["model"]["merges"])}
    specials = {t["content"]: t["id"] for t in tokenizer["added_tokens"]}
    split = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]| ?[^\sA-Za-z0-9]+[\r\n]*|\s*[\r\n]+|"
             r"\s+(?!\S)|\s+")
    ids: list[int] = []
    for part in re.split("(" + "|".join(re.escape(s) for s in specials) + ")", text):
        if part in specials:
            ids.append(specials[part])
            continue
        for piece in re.findall(split, part):
            symbols = [to_char[b] for b in piece.encode()]
            while len(symbols) > 1:
                pairs = [(rank.get((a, b), None), i) for i, (a, b) in enumerate(zip(symbols, symbols[1:]))]
                pairs = [p for p in pairs if p[0] is not None]
                if not pairs:
                    break
                _, at = min(pairs)
                symbols[at:at + 2] = [symbols[at] + symbols[at + 1]]
            ids.extend(vocab[s] for s in symbols)
    return ids


# ---------------------------------------------------------------- command line

def write_tiny_set(out: Path, seed: int = 1, first_layer: int = 2) -> dict:
    """A tiny trunk, two skills (different ranks and matrix subsets), a head and a tokenizer in `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_random_trunk(out / "tiny.fkm", TINY, mode="q4", emb="q8", seed=seed, max_ctx=512)
    write_random_trunk(out / "tiny_q8.fkm", TINY, mode="q8", emb="q4", seed=seed + 1, max_ctx=512)
    skills = {
        "alpha": (random_skill_entries(TINY, first_layer, 4, MATRICES, seed=seed + 10)),
        "beta": (random_skill_entries(TINY, first_layer, {"q": 2, "k": 3, "v": 4, "o": 5, "gate": 6, "up": 7, "down": 8},
                                      ["q", "v", "o", "gate", "down"], seed=seed + 11)),
    }
    for name, entries in skills.items():
        write_skill(out / f"{name}.kms", name, first_layer, TINY["n_layers"], entries)
    names = ["none", "alpha", "beta"]
    write_head(out / "head.kmh", *random_head(TINY, names, seed=seed + 20), names)
    write_head(out / "head_l1.kmh", *random_head(TINY, names, hidden=16, seed=seed + 21), names, layer=1)
    (out / "tokenizer.json").write_text(json.dumps(tiny_tokenizer(TINY["vocab"])))
    return dict(first_layer=first_layer, config=TINY)


def main(argv: list[str]) -> int:
    usage = "usage: formats.py tiny OUTDIR | random-qwen OUTDIR [--mode q4|q8] [--emb q4|q8] [--rank 16] [--skill-first 18]"
    if len(argv) < 3 or argv[1] not in ("tiny", "random-qwen"):
        print(usage, file=sys.stderr)
        return 2
    out = Path(argv[2])
    if argv[1] == "tiny":
        print(json.dumps(write_tiny_set(out)))
        return 0
    opts = dict(zip(argv[3::2], argv[4::2]))
    out.mkdir(parents=True, exist_ok=True)
    write_random_trunk(out / "qwen-shaped.fkm", QWEN3_0_6B, mode=opts.get("--mode", "q4"), emb=opts.get("--emb", "q4"))
    first = int(opts.get("--skill-first", 18))
    write_skill(out / "random.kms", "random", first, QWEN3_0_6B["n_layers"],
                random_skill_entries(QWEN3_0_6B, first, int(opts.get("--rank", 16))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
