#!/usr/bin/env python3
"""Measure Gemma 4 router usage on training-only KiwiBro trajectories in Colab."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import file_digest
import json
from pathlib import Path
import random
import re

from scripts.mind.prune_gemma4 import BASE, BASE_REVISION

ROUTER = re.compile(r"model\.language_model\.layers\.(\d+)\.router\Z")


def rows(path: Path, per_family: int, seed: int) -> list[dict]:
    pools = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            pools[row["family"]].append(row)
    rng = random.Random(seed)
    return [row for name in sorted(pools) for row in rng.sample(pools[name], min(per_family, len(pools[name])))]


def messages_for_processor(row: dict, corpus: Path) -> list[dict]:
    from PIL import Image

    root = corpus.resolve()
    messages = []
    for message in row["messages"]:
        content = message.get("content")
        if isinstance(content, list):
            content = []
            for part in message["content"]:
                if part.get("type") != "image_url":
                    content.append(part)
                    continue
                blob = (root / part["image_url"]["url"]).resolve()
                if not blob.is_relative_to(root) or not blob.is_file():
                    raise ValueError(f"Image outside corpus or missing in {row['id']}.")
                with Image.open(blob) as image:
                    content.append({"type": "image", "image": image.convert("RGB")})
        messages.append({**message, "content": content})
    return messages


def load(base: str, revision: str | None, device: str, gpu_budget: str | None = None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(base, revision=revision)
    limits = {}
    if gpu_budget:
        from accelerate.utils import get_max_memory

        memory = get_max_memory()
        memory[0] = gpu_budget
        limits["max_memory"] = memory
    model = AutoModelForImageTextToText.from_pretrained(
        base, revision=revision, device_map=device, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        **limits,
    ).eval()
    return model, processor


def collect(args: argparse.Namespace) -> dict:
    import torch

    base = str(args.checkpoint) if args.checkpoint else BASE
    model, processor = load(base, None if args.checkpoint else BASE_REVISION, args.device, args.gpu_budget)
    cfg = model.config.text_config
    if cfg.num_experts != 128 or cfg.top_k_experts != 8:
        raise ValueError("Calibration needs the original 128-expert, top-8 checkpoint.")
    cases = rows(args.corpus, args.per_family, args.seed)
    if not cases:
        raise ValueError("No training trajectories for router calibration.")
    counts = defaultdict(dict)
    family = None
    handles = []

    def record(layer: int):
        def observe(_module, _inputs, output):
            probabilities, _scaled, selected = output
            weights = probabilities.gather(-1, selected)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-10)
            totals = torch.bincount(selected.reshape(-1), weights.reshape(-1), minlength=cfg.num_experts).cpu()
            current = counts[layer].get(family)
            counts[layer][family] = totals if current is None else current + totals
        return observe

    for name, module in model.named_modules():
        match = ROUTER.fullmatch(name)
        if match:
            handles.append(module.register_forward_hook(record(int(match[1]))))
    if len(handles) != cfg.num_hidden_layers:
        raise RuntimeError(f"Found {len(handles)} routers for {cfg.num_hidden_layers} configured layers.")

    token_count = 0
    processed = defaultdict(int)
    embedding_device = model.get_input_embeddings().weight.device
    try:
        with torch.inference_mode():
            for i, row in enumerate(cases, 1):
                tokenized = processor.apply_chat_template(
                    messages_for_processor(row, args.corpus.parent),
                    tools=row["tools"], tokenize=True, add_generation_prompt=False,
                    return_dict=True, return_tensors="pt",
                )
                size = tokenized["input_ids"].shape[-1]
                if size > args.max_tokens:
                    print(f"skip {row['id']} ({size} tokens)", flush=True)
                    continue
                family = row["family"]
                model(**tokenized.to(embedding_device), logits_to_keep=1, use_cache=False)
                token_count += size
                processed[family] += 1
                print(f"{i}/{len(cases)} {row['id']} {family} {size} tokens", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    families = {row["family"] for row in cases}
    if set(processed) != families or any(set(counts[layer]) != families for layer in range(cfg.num_hidden_layers)):
        raise RuntimeError("Every workflow family must exercise each router. Review the token budget and corpus.")
    with args.corpus.open("rb") as handle:
        corpus_hash = file_digest(handle, "sha256").hexdigest()
    return {
        "base": BASE, "base_revision": BASE_REVISION, "corpus_sha256": corpus_hash,
        "source": "training trajectories only", "seed": args.seed, "tokens": token_count,
        "cases_by_family": dict(processed),
        "layers": {str(layer): {name: values.tolist() for name, values in groups.items()}
                   for layer, groups in counts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Local copy of the pinned original checkpoint.")
    parser.add_argument("--device", default="auto", help="Accelerate device map, e.g. auto, cuda or cpu.")
    parser.add_argument("--gpu-budget", help="Leave VRAM for CPU-offloaded layers and activations, e.g. 18GiB.")
    parser.add_argument("--per-family", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=4071)
    args = parser.parse_args()
    args.output.write_text(json.dumps(collect(args), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
