#!/usr/bin/env python3
"""Reduce the Gemma 4 A4B expert banks using per-workflow router calibration.

Copies the selected expert rows and their router parameters together while
preserving every shared, attention, vision and embedding tensor byte-for-byte.
The resulting checkpoint uses the existing Gemma 4 architecture identifier and
its smaller expert and top-k settings.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import BinaryIO

BASE = "google/gemma-4-26B-A4B-it"
BASE_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
INDEX = "model.safetensors.index.json"
EXPERT = re.compile(r"^model\.language_model\.layers\.(\d+)\.(experts\..+|router\..+)$")
SLICED = {"experts.gate_up_proj", "experts.down_proj", "router.proj.weight", "router.per_expert_scale"}
MAX_METADATA = 100_000_000
BUFFER = 4 << 20


def choose(counts: dict[str, list[float]], num_experts: int, retain: int) -> tuple[list[int], dict[str, float]]:
    """Rank by the best workflow share, with a mean-share tiebreak."""
    distributions = {}
    for family, values in counts.items():
        if len(values) != num_experts or any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"Invalid router usage for {family!r}.")
        total = sum(values)
        if total <= 0:
            raise ValueError(f"No router observations for {family!r}.")
        distributions[family] = [value / total for value in values]
    if not distributions:
        raise ValueError("Missing workflow router measurements.")

    def rank(index: int) -> tuple[float, float, int]:
        shares = [values[index] for values in distributions.values()]
        return max(shares), sum(shares), -index

    selected = sorted(sorted(range(num_experts), key=rank, reverse=True)[:retain])
    coverage = {family: sum(values[index] for index in selected) for family, values in distributions.items()}
    return selected, coverage


def _read_header(handle: BinaryIO) -> tuple[dict, int, int]:
    prefix = handle.read(8)
    if len(prefix) != 8:
        raise ValueError("Missing safetensors header.")
    length = struct.unpack("<Q", prefix)[0]
    if length > MAX_METADATA:
        raise ValueError("Oversized safetensors header.")
    data = handle.read(length)
    if len(data) != length:
        raise ValueError("Truncated safetensors header.")
    start = 8 + length
    return json.loads(data), start, handle.seek(0, os.SEEK_END) - start


def _transfer(source: BinaryIO, target: BinaryIO, offset: int, size: int) -> None:
    source.seek(offset)
    while size:
        block = source.read(min(size, BUFFER))
        if not block:
            raise ValueError("Truncated tensor payload.")
        target.write(block)
        size -= len(block)


def _rewrite_shard(source: Path, target: Path, keys: set[str],
                   rows: dict[str, list[int]], num_experts: int) -> int:
    with source.open("rb") as reader:
        original, payload, size = _read_header(reader)
        meta = original.pop("__metadata__", None)
        if set(original) != keys:
            raise ValueError(f"Tensor keys and index disagree in {source.name}.")
        if not rows:
            # The unchanged shard (e.g. vision or embeddings) needs no copy inside this calibration run.
            # Use a streamed copy so the checkpoint also works after the reference is deleted.
            with target.open("xb") as writer:
                _transfer(reader, writer, 0, payload + size)
            return size
        header = {"__metadata__": meta} if meta is not None else {}
        ranges = []
        written = 0
        for name, item in sorted(original.items(), key=lambda pair: pair[1]["data_offsets"][0]):
            start, end = item["data_offsets"]
            if not 0 <= start <= end <= size:
                raise ValueError(f"Tensor lies outside payload in {source.name}.")
            shape = item["shape"]
            length = end - start
            selected = rows.get(name)
            if selected is not None:
                if item["dtype"] != "BF16" or not shape or shape[0] != num_experts:
                    raise ValueError(f"Unexpected dtype or expert axis for {name}.")
                stride = 2 * math.prod(shape[1:])
                if length != stride * num_experts:
                    raise ValueError(f"Unexpected tensor size for {name}.")
                ranges.extend((payload + start + index * stride, stride) for index in selected)
                shape = [len(selected), *shape[1:]]
                length = stride * len(selected)
            else:
                ranges.append((payload + start, length))
            header[name] = {**item, "shape": shape, "data_offsets": [written, written + length]}
            written += length
        data = json.dumps(header, separators=(",", ":")).encode()
        data += b" " * (-len(data) % 8)
        with target.open("xb") as writer:
            writer.write(struct.pack("<Q", len(data)))
            writer.write(data)
            for offset, length in ranges:
                _transfer(reader, writer, offset, length)
    return written


def export(source: Path, output: Path, profile: dict, *, retain: int, top_k: int) -> dict:
    config = json.loads((source / "config.json").read_text())
    if config.get("architectures") != ["Gemma4ForConditionalGeneration"]:
        raise ValueError("Expected the Gemma 4 multimodal Transformers architecture.")
    text = config["text_config"]
    expert_count = text["num_experts"]
    original_k = text["top_k_experts"]
    if not text["enable_moe_block"] or not 1 <= top_k <= original_k <= retain < expert_count:
        raise ValueError("Expected a reduced expert bank and no increase in router top-k.")
    if profile.get("base_revision") != BASE_REVISION:
        raise ValueError("Calibration was built for a different checkpoint revision.")
    index = json.loads((source / INDEX).read_text())
    weight_map = index["weight_map"]
    expected: dict[str, set[str]] = defaultdict(set)
    rows: dict[str, dict[str, list[int]]] = defaultdict(dict)
    layer_keys: dict[int, set[str]] = defaultdict(set)
    picks = {}
    coverage = {}
    for name, shard in weight_map.items():
        if Path(shard).name != shard or Path(shard).suffix != ".safetensors":
            raise ValueError(f"Invalid shard name {shard!r}.")
        expected[shard].add(name)
        match = EXPERT.fullmatch(name)
        if match is None or match[2] == "router.scale":
            continue
        layer, axis = int(match[1]), match[2]
        if axis not in SLICED:
            raise ValueError(f"Unknown expert tensor {name!r}.")
        if layer not in picks:
            picks[layer], coverage[layer] = choose(profile["layers"][str(layer)], expert_count, retain)
        rows[shard][name] = picks[layer]
        layer_keys[layer].add(axis)
    if set(layer_keys) != set(range(text["num_hidden_layers"])) or any(keys != SLICED for keys in layer_keys.values()):
        raise ValueError("Incomplete Gemma 4 expert/router tensor families.")
    text["num_experts"], text["top_k_experts"] = retain, top_k
    report = {
        "base_model": BASE, "base_revision": BASE_REVISION,
        "experts_before": expert_count, "experts_after": retain,
        "top_k_before": original_k, "top_k_after": top_k,
        "kept_original_experts": picks, "retained_router_mass": coverage,
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".gemma-experts-") as folder:
        stage = Path(folder) / "weights"
        stage.mkdir()
        total = sum(
            _rewrite_shard(source / shard, stage / shard, keys, rows[shard], expert_count)
            for shard, keys in expected.items()
        )
        for item in source.iterdir():
            if (
                item.is_file()
                and item.stat().st_size < MAX_METADATA
                and (item.suffix in {".json", ".jinja", ".txt", ".model"} or item.name == "LICENSE")
                and item.name not in {"config.json", INDEX}
            ):
                (stage / item.name).write_bytes(item.read_bytes())
        report["weight_bytes"] = total
        for name, data in (
            ("config.json", config),
            (INDEX, {"metadata": {"total_size": total}, "weight_map": weight_map}),
            ("architecture.json", report),
        ):
            (stage / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.rename(stage, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--retain", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output, json.loads(args.profile.read_text()),
                            retain=args.retain, top_k=args.top_k), indent=2))


if __name__ == "__main__":
    main()
