#!/usr/bin/env python3
"""Stream a narrower BF16 Gemma 4 E4B checkpoint without loading a model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import tempfile

import numpy as np

MODEL = "google/gemma-4-E4B-it"
REVISION = "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c"
INDEX = "model.safetensors.index.json"
REPORT = "ffn-compaction.json"
MLP = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight")
CHUNK = 8 * 1024 * 1024
ROWS = 128


@dataclass(frozen=True)
class Tensor:
    shard: Path
    offset: int
    size: int
    info: dict


def read_shard(path: Path) -> tuple[dict | None, dict[str, Tensor]]:
    with path.open("rb") as reader:
        prefix = reader.read(8)
        if len(prefix) != 8 or not 0 < (size := struct.unpack("<Q", prefix)[0]) <= 100_000_000:
            raise ValueError(f"Invalid safetensors header length in {path.name}")
        encoded = reader.read(size)
        if len(encoded) != size:
            raise ValueError(f"Truncated safetensors header in {path.name}")
        header = json.loads(encoded)
        start = reader.tell()
        payload = reader.seek(0, 2) - start
    metadata = header.pop("__metadata__", None)
    tensors = {}
    end = 0
    for name, item in sorted(header.items(), key=lambda pair: pair[1]["data_offsets"]):
        left, right = item["data_offsets"]
        if left != end or not left <= right <= payload:
            raise ValueError(f"Invalid payload offsets for {name}")
        tensors[name] = Tensor(path, start + left, right - left, item)
        end = right
    if end != payload:
        raise ValueError(f"Unindexed bytes in {path.name}")
    return metadata, tensors


def blocks(reader, tensor: Tensor):
    reader.seek(tensor.offset)
    rows, cols = tensor.info["shape"]
    for left in range(0, rows, ROWS):
        count = min(ROWS, rows - left)
        data = reader.read(2 * count * cols)
        if len(data) != 2 * count * cols:
            raise ValueError(f"Truncated BF16 tensor in {tensor.shard.name}")
        yield left, np.frombuffer(data, dtype="<u2").reshape(count, cols)


def weight_norms(tensor: Tensor, *, columns: bool) -> np.ndarray:
    shape = tensor.info["shape"]
    norms = np.zeros(shape[1 if columns else 0], dtype=np.float64)
    with tensor.shard.open("rb") as reader:
        for left, bits in blocks(reader, tensor):
            values = (bits.astype(np.uint32) << 16).view(np.float32)
            if columns:
                norms += np.einsum("ij,ij->j", values, values, dtype=np.float64)
            else:
                norms[left:left + len(bits)] = np.einsum("ij,ij->i", values, values, dtype=np.float64)
    return norms


def copy_tensor(reader, writer, tensor: Tensor) -> None:
    reader.seek(tensor.offset)
    remaining = tensor.size
    while remaining:
        block = reader.read(min(remaining, CHUNK))
        if not block:
            raise ValueError(f"Truncated tensor in {tensor.shard.name}")
        writer.write(block)
        remaining -= len(block)


def write_shard(path, metadata, tensors, slices, width):
    header = {} if metadata is None else {"__metadata__": metadata}
    end = 0
    for name, tensor in tensors.items():
        shape = list(tensor.info["shape"])
        size = tensor.size
        if name in slices:
            axis, _ = slices[name]
            shape[axis] = width
            size = 2 * shape[0] * shape[1]
        header[name] = {**tensor.info, "shape": shape, "data_offsets": [end, end + size]}
        end += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with next(iter(tensors.values())).shard.open("rb") as reader, path.open("xb") as writer:
        writer.write(struct.pack("<Q", len(encoded)))
        writer.write(encoded)
        for name, tensor in tensors.items():
            if name not in slices:
                copy_tensor(reader, writer, tensor)
                continue
            axis, channels = slices[name]
            for left, array in blocks(reader, tensor):
                if axis == 0:
                    start, stop = np.searchsorted(channels, [left, left + len(array)])
                    selected = array[channels[start:stop] - left]
                else:
                    selected = array[:, channels]
                writer.write(selected.tobytes())
        if writer.tell() != 8 + len(encoded) + end:
            raise ValueError(f"Rewritten payload length mismatch in {path.name}")
    return end


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def export(source: Path, output: Path, *, scores: Path | None, width: int, expected_shape=(42, 2560, 10240)):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if source.parent.name == "snapshots" and source.name != REVISION:
        raise ValueError("The Hugging Face snapshot path does not match the pinned revision")
    config_bytes = (source / "config.json").read_bytes()
    config = json.loads(config_bytes)
    text = config["text_config"]
    layers, hidden, before = (text[key] for key in ("num_hidden_layers", "hidden_size", "intermediate_size"))
    if (
        config.get("architectures") != ["Gemma4ForConditionalGeneration"]
        or (layers, hidden, before) != expected_shape
        or text.get("enable_moe_block", False)
        or text.get("use_double_wide_mlp", False)
        or not 0 < width < before or width % 256
    ):
        raise ValueError("Expected the pinned dense E4B shape and a smaller K-quant-aligned width")

    index_path = source / INDEX
    index_bytes = index_path.read_bytes() if index_path.is_file() else None
    index = json.loads(index_bytes) if index_bytes is not None else {}
    weight_map = index.get("weight_map")
    names = sorted(set(weight_map.values())) if weight_map is not None else ["model.safetensors"]
    shards, inferred, mlp = {}, {}, {}
    for name in names:
        if Path(name).name != name or Path(name).suffix != ".safetensors":
            raise ValueError(f"Invalid shard name {name!r}")
        metadata, tensors = read_shard(source / name)
        if not tensors:
            raise ValueError(f"Empty checkpoint shard {name}")
        shards[name] = (metadata, tensors)
        for key, tensor in tensors.items():
            if key in inferred:
                raise ValueError(f"Duplicate tensor {key}")
            inferred[key] = name
            if match := MLP.fullmatch(key):
                layer, projection = int(match[1]), match[2]
                shape = [hidden, before] if projection == "down" else [before, hidden]
                if tensor.info["dtype"] != "BF16" or tensor.info["shape"] != shape or tensor.size != 2 * hidden * before:
                    raise ValueError(f"Unexpected BF16 shape or size of {key}")
                mlp[layer, projection] = (key, tensor)
    if weight_map is not None and weight_map != inferred:
        raise ValueError("Shard headers disagree with the checkpoint tensor index")
    if set(mlp) != {(i, p) for i in range(layers) for p in ("gate", "up", "down")}:
        raise ValueError("Missing or unexpected dense gate, up or down tensors")

    calibration = None if scores is None else np.load(scores, mmap_mode="r", allow_pickle=False)
    if calibration is not None and (calibration.shape != (layers, before) or calibration.dtype.kind not in "fiu"):
        raise ValueError(f"Expected a numeric .npy score matrix with shape {(layers, before)}")
    slices, selected_by_layer = {}, {}
    for layer in range(layers):
        gate, up, down = (mlp[layer, part] for part in ("gate", "up", "down"))
        salience = (weight_norms(gate[1], columns=False) * weight_norms(up[1], columns=False)
                    * weight_norms(down[1], columns=True)) if calibration is None else calibration[layer].astype(np.float64)
        if not np.isfinite(salience).all() or np.any(salience < 0) or salience.max() <= 0:
            raise ValueError(f"Layer {layer} scores must be finite, nonnegative and nonzero")
        selected = np.sort(np.argsort(-salience, kind="stable")[:width])
        selected_by_layer[str(layer)] = selected.tolist()
        for axis, pair in ((0, gate), (0, up), (1, down)):
            slices[pair[0]] = (axis, selected)

    report = {
        "base_model": MODEL, "base_revision": REVISION,
        "source_revision": "verified snapshot path" if source.parent.name == "snapshots" else "operator-supplied pin",
        "width_before": before, "width_after": width, "layers": layers, "hidden_size": hidden,
        "selection": "weight_l2_product_fallback" if scores is None else "activation_scores",
        "activation_scores_sha256": None if scores is None else hashlib.sha256(scores.read_bytes()).hexdigest(),
        "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_index_sha256": None if index_bytes is None else hashlib.sha256(index_bytes).hexdigest(),
        "removed_weights": 3 * layers * hidden * (before - width),
        "kept_original_channels": selected_by_layer,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".e4b-compact-", dir=output.parent) as temporary:
        staging = Path(temporary) / "checkpoint"
        staging.mkdir()
        count = 0
        for name, (metadata, tensors) in shards.items():
            count += write_shard(staging / name, metadata, tensors, slices, width)
        for path in source.iterdir():
            if (path.is_file() and not path.name.startswith(".") and path.suffix != ".safetensors"
                    and path.name not in {"config.json", INDEX, REPORT}):
                shutil.copyfile(path, staging / path.name)
        text["intermediate_size"] = width
        index["weight_map"] = inferred
        index.setdefault("metadata", {})["total_size"] = count
        write_json(staging / "config.json", config)
        write_json(staging / INDEX, index)
        write_json(staging / REPORT, report)
        staging.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-revision", required=True, choices=[REVISION])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--activation-scores", type=Path, metavar="SCORES.npy")
    mode.add_argument("--weight-salience", action="store_true", help="Uncalibrated L2 fallback for comparison only")
    args = parser.parse_args()
    report = export(args.source, args.output, scores=args.activation_scores, width=8192)
    print(f"Wrote {args.output} ({report['removed_weights']:,} weights removed, {report['selection']}).")


if __name__ == "__main__":
    main()
