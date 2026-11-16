#!/usr/bin/env python3
"""Export a narrower BF16 Gemma 4 MoE using static, coupled channel salience."""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np

from scripts.mind.prune_gemma4 import BASE_REVISION, EXPERT, INDEX, MAX_METADATA, _read_header, _transfer

GATE_UP = "experts.gate_up_proj"
DOWN = "experts.down_proj"
CHANNEL_WEIGHTS = {GATE_UP, DOWN}
BF16_BYTES = 2
K_BLOCK_SIZE = 256


@dataclass(frozen=True)
class TensorLocation:
    path: Path
    offset: int


def _prepare_shard(
    source: Path,
    target: Path,
    keys: set[str],
    expected_shapes: dict[str, list[int]],
    compact_shapes: dict[str, list[int]],
) -> tuple[int, dict[str, tuple[TensorLocation, TensorLocation]]]:
    locations = {}
    with source.open("rb") as reader:
        original, source_start, source_size = _read_header(reader)
        metadata = original.pop("__metadata__", None)
        if set(original) != keys:
            raise ValueError(f"Tensor keys and index disagree in {source.name}.")
        header = {"__metadata__": metadata} if metadata is not None else {}
        source_end, target_end = 0, 0
        ordered = sorted(original.items(), key=lambda pair: pair[1]["data_offsets"])
        for name, item in ordered:
            start, end = item["data_offsets"]
            if start != source_end or not start <= end <= source_size:
                raise ValueError(f"Noncontiguous tensor payload in {source.name}.")
            source_end = end
            shape, length = item["shape"], end - start
            if name in compact_shapes:
                if (
                    item["dtype"] != "BF16"
                    or shape != expected_shapes[name]
                    or length != math.prod(shape) * BF16_BYTES
                ):
                    raise ValueError(f"Unexpected BF16 expert shape or size for {name}.")
                shape = compact_shapes[name]
                length = math.prod(shape) * BF16_BYTES
            header[name] = {**item, "shape": shape, "data_offsets": [target_end, target_end + length]}
            target_end += length
        if source_end != source_size:
            raise ValueError(f"Unindexed bytes in {source.name}.")
        if not keys.intersection(compact_shapes):
            with target.open("xb") as writer:
                _transfer(reader, writer, 0, source_start + source_size)
            return target_end, locations

        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        with target.open("xb") as writer:
            writer.write(struct.pack("<Q", len(encoded)))
            writer.write(encoded)
            target_start = writer.tell()
            writer.truncate(target_start + target_end)
            for name, item in ordered:
                start, end = item["data_offsets"]
                offset = target_start + header[name]["data_offsets"][0]
                if name in compact_shapes:
                    locations[name] = (TensorLocation(source, source_start + start), TensorLocation(target, offset))
                else:
                    writer.seek(offset)
                    _transfer(reader, writer, source_start + start, end - start)
    return target_end, locations


def _read_matrix(reader, position, rows, columns):
    reader.seek(position)
    count = rows * columns
    data = reader.read(count * BF16_BYTES)
    if len(data) != count * BF16_BYTES:
        raise ValueError("Truncated BF16 expert matrix.")
    return np.frombuffer(data, dtype="<u2").reshape(rows, columns)


def _squared_norms(bf16: np.ndarray, *, columns: bool = False) -> np.ndarray:
    bits = bf16.astype(np.uint32)
    np.left_shift(bits, 16, out=bits)
    values = bits.view(np.float32)
    expression = "ij,ij->j" if columns else "ij,ij->i"
    return np.einsum(expression, values, values, dtype=np.float64)


def _score(gate: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    # Approximate the contribution of a channel using its three weight norms.
    score = _squared_norms(gate) * _squared_norms(up) * _squared_norms(down, columns=True)
    if not np.isfinite(score).all():
        raise ValueError("Nonfinite BF16 expert weight or channel salience.")
    return score


def _compact_layer(gate_files, down_files, *, experts: int, hidden: int, before: int, after: int):
    gate_source, gate_target = gate_files
    down_source, down_target = down_files
    gate_stride = 2 * before * hidden * BF16_BYTES
    down_stride = before * hidden * BF16_BYTES
    masks = []
    coverage = []
    with ExitStack() as stack:
        sources = {
            path: stack.enter_context(path.open("rb")) for path in {gate_source.path, down_source.path}
        }
        targets = {
            path: stack.enter_context(path.open("r+b")) for path in {gate_target.path, down_target.path}
        }
        for expert in range(experts):
            gate_up = _read_matrix(
                sources[gate_source.path], gate_source.offset + expert * gate_stride, 2 * before, hidden,
            )
            down = _read_matrix(
                sources[down_source.path], down_source.offset + expert * down_stride, hidden, before,
            )
            gate, up = gate_up[:before], gate_up[before:]
            score = _score(gate, up, down)
            selected = np.sort(np.argsort(-score, kind="stable")[:after])

            writer = targets[gate_target.path]
            writer.seek(gate_target.offset + expert * 2 * after * hidden * BF16_BYTES)
            writer.write(gate[selected].tobytes())
            writer.write(up[selected].tobytes())
            writer = targets[down_target.path]
            writer.seek(down_target.offset + expert * after * hidden * BF16_BYTES)
            writer.write(down[:, selected].tobytes())

            mask = np.zeros(before, dtype=np.uint8)
            mask[selected] = 1
            masks.append(base64.b64encode(np.packbits(mask, bitorder="little")).decode("ascii"))
            salience = score.sum()
            coverage.append(float(score[selected].sum() / salience) if salience else 1.0)
    return {"kept_channels": masks, "min_salience_share": min(coverage), "mean_salience_share": sum(coverage) / experts}


def export(source: Path, output: Path, *, width: int = 512) -> dict:
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((source / "config.json").read_text())
    architecture = json.loads((source / "architecture.json").read_text())
    if config.get("architectures") != ["Gemma4ForConditionalGeneration"]:
        raise ValueError("Expected the Gemma 4 multimodal Transformers architecture.")
    if architecture.get("base_revision") != BASE_REVISION:
        raise ValueError("Expected KiwiBro's pinned Gemma 4 base revision.")
    text = config["text_config"]
    before = text["moe_intermediate_size"]
    if text.get("expert_intermediate_size", before) != before:
        raise ValueError("Conflicting Gemma expert width fields.")
    experts = text["num_experts"]
    hidden = text["hidden_size"]
    layers = text["num_hidden_layers"]
    if not text["enable_moe_block"] or not 0 < width < before or width % K_BLOCK_SIZE:
        raise ValueError("Expected a smaller positive MoE width divisible by 256.")
    if layers <= 0 or hidden <= 0 or not 1 <= text["top_k_experts"] <= experts:
        raise ValueError("Invalid Gemma 4 expert architecture.")

    index = json.loads((source / INDEX).read_text())
    expected_shards = defaultdict(set)
    layer_weights = defaultdict(dict)
    for name, shard in index["weight_map"].items():
        if Path(shard).name != shard or Path(shard).suffix != ".safetensors":
            raise ValueError(f"Invalid shard name {shard!r}.")
        expected_shards[shard].add(name)
        match = EXPERT.fullmatch(name)
        if not match:
            continue
        layer, tensor = int(match[1]), match[2]
        if tensor.startswith("experts.") and tensor not in CHANNEL_WEIGHTS:
            raise ValueError(f"Expected a BF16 checkpoint without auxiliary expert tensors, found {name}.")
        if tensor in CHANNEL_WEIGHTS:
            layer_weights[layer][tensor] = name
    if set(layer_weights) != set(range(layers)) or any(set(pair) != CHANNEL_WEIGHTS for pair in layer_weights.values()):
        raise ValueError("Missing fused gate/up or down expert matrices.")

    old_shapes, new_shapes = {}, {}
    for pair in layer_weights.values():
        old_shapes[pair[GATE_UP]] = [experts, 2 * before, hidden]
        old_shapes[pair[DOWN]] = [experts, hidden, before]
        new_shapes[pair[GATE_UP]] = [experts, 2 * width, hidden]
        new_shapes[pair[DOWN]] = [experts, hidden, width]
    source_hashes = {}
    for name in ("config.json", INDEX, "architecture.json", "training.json"):
        path = source / name
        if path.is_file():
            source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    text["moe_intermediate_size"] = width
    if "expert_intermediate_size" in text:
        text["expert_intermediate_size"] = width
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gemma-channels-", dir=output.parent) as folder:
        stage = Path(folder) / "weights"
        stage.mkdir()
        total, locations = 0, {}
        for shard, keys in expected_shards.items():
            size, locations_by_name = _prepare_shard(source / shard, stage / shard, keys, old_shapes, new_shapes)
            total += size
            locations.update(locations_by_name)
        selections = {}
        for layer, pair in sorted(layer_weights.items()):
            selections[layer] = _compact_layer(
                locations[pair[GATE_UP]], locations[pair[DOWN]],
                experts=experts, hidden=hidden, before=before, after=width,
            )
        report = {
            **architecture,
            "weight_bytes": total,
            "channel_compaction": {
                "method": "product of squared L2 norms of the gate row, up row, and down column",
                "activation_calibration": None,
                "source_metadata_sha256": source_hashes,
                "width_before": before,
                "width_after": width,
                "experts": experts,
                "top_k_experts": text["top_k_experts"],
                "layers": selections,
                "kept_channels_encoding": "base64, little-endian bit order, one mask per expert in checkpoint order",
            },
        }
        index.setdefault("metadata", {})["total_size"] = total
        generated = {"config.json": config, INDEX: index, "architecture.json": report}
        for item in source.iterdir():
            if (
                item.name not in generated and item.is_file() and item.stat().st_size < MAX_METADATA
                and (item.suffix in {".json", ".jinja", ".txt", ".model"} or item.name == "LICENSE")
            ):
                shutil.copyfile(item, stage / item.name)
        for name, data in generated.items():
            (stage / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()
    report = export(args.source, args.output, width=args.width)
    compaction = report["channel_compaction"]
    print(json.dumps({
        "width_before": compaction["width_before"],
        "width_after": compaction["width_after"],
        "weight_bytes": report["weight_bytes"],
        "method": compaction["method"],
    }, indent=2))


if __name__ == "__main__":
    main()
