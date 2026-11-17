#!/usr/bin/env python3
"""Calibrate and recover the pinned dense E4B on a single Colab L4."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import random
import re
import shutil
import tempfile

from scripts.mind.compact_e4b import INDEX, MODEL, REPORT, REVISION
from scripts.mind.calibrate_gemma4 import messages_for_processor, rows
from scripts.mind.fit_gemma4 import _encoded, _loss, _supervision_template

GIB = 1 << 30
LAYERS, HIDDEN, ORIGINAL, COMPACT = 42, 2560, 10240, 8192
MODES = ("text", "tool", "image")
TARGETS = re.compile(
    r"model\.language_model\.layers\.(\d+)\.(self_attn\.(?:q_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@contextmanager
def new_directory(path):
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".e4b-", dir=path.parent) as tmp:
        stage = Path(tmp) / "output"
        stage.mkdir()
        yield stage
        stage.rename(path)


def mode(row):
    if row["images"]:
        return "image"
    return "tool" if any(msg.get("tool_calls") for msg in row["messages"]) else "text"


def cases(args):
    # Fit input comes from explicit base and supplemental training shards. Probes carry oracle fields.
    if not args.corpus.name.endswith("train.jsonl"):
        raise ValueError("Pass a training shard or training mixture ending in train.jsonl.")
    selected = rows(args.corpus, args.per_family, args.seed)
    groups = {kind: [] for kind in MODES}
    for row in selected:
        if "expected" in row or "episode" in row or row["messages"][-1]["role"] != "assistant":
            raise ValueError(f"Expected a complete training trajectory in {row['id']}.")
        groups[mode(row)].append(row)
    if len({row["id"] for row in selected}) != len(selected) or not all(groups.values()):
        raise ValueError("Expected unique training IDs and coverage of text, tool, and image episodes.")
    rng = random.Random(args.seed)
    for group in groups.values():
        rng.shuffle(group)
    return [row for batch in zip_longest(*groups.values()) for row in batch if row is not None]


def memory(torch, gib):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Use a BF16-capable Colab GPU such as L4 or A100.")
    free, total = torch.cuda.mem_get_info(0)
    limit = int(gib * GIB)
    if limit + GIB > free:
        raise RuntimeError(f"Need {gib + 1:.2f} GiB free at startup, found {free / GIB:.2f} GiB.")
    torch.cuda.set_per_process_memory_fraction(limit / total, 0)
    torch.cuda.reset_peak_memory_stats(0)
    info = {"device": torch.cuda.get_device_name(0), "device_total_bytes": total,
            "free_before_bytes": free, "allocator_limit_bytes": limit}
    print(json.dumps(info), flush=True)
    return info


def peak(torch, initial):
    torch.cuda.synchronize(0)
    return {**initial, "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}


def load(checkpoint, expected_width, *, device="cuda"):
    import torch
    from transformers import AutoModelForImageTextToText

    config = json.loads((checkpoint / "config.json").read_text())
    text = config["text_config"]
    if (config["architectures"] != ["Gemma4ForConditionalGeneration"]
            or (text["num_hidden_layers"], text["hidden_size"], text["intermediate_size"])
            != (LAYERS, HIDDEN, expected_width)
            or text.get("enable_moe_block") or text.get("use_double_wide_mlp")):
        raise ValueError(f"Expected pinned E4B dense width {expected_width}.")
    return AutoModelForImageTextToText.from_pretrained(
        checkpoint, device_map={"": 0 if device == "cuda" else "cpu"},
        dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True,
        attn_implementation={"": "sdpa", "audio_config": "eager"},
    )


def calibrated(args):
    import numpy as np
    import torch
    from transformers import AutoProcessor

    source = args.checkpoint.resolve(strict=True)
    if source.name != REVISION or source.parent.name != "snapshots":
        raise ValueError(f"Use the Hugging Face snapshot path pinned to {REVISION}.")
    data = cases(args)
    before = memory(torch, args.vram_gib)
    teacher = load(source, ORIGINAL).eval()
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    scores = torch.zeros((len(MODES), LAYERS, ORIGINAL), dtype=torch.float32, device="cuda")
    seen, tokens = Counter(), Counter()
    handles, downs = [], []
    bucket = -1
    length = 0

    def observe(layer):
        def record(_module, inputs):
            values = inputs[0].reshape(-1, ORIGINAL)
            if values.shape[0] != length:
                raise RuntimeError(f"Expected all {length} input tokens at dense layer {layer}.")
            for batch in values.split(128):
                scores[bucket, layer].add_(batch.float().square().sum(dim=0))
        return record

    try:
        for i, layer in enumerate(teacher.model.language_model.layers):
            down = layer.mlp.down_proj
            if down.weight.shape != (HIDDEN, ORIGINAL) or down.weight.dtype != torch.bfloat16:
                raise ValueError(f"Unexpected BF16 down projection in layer {i}")
            downs.append(down)
            handles.append(down.register_forward_pre_hook(observe(i)))
        with torch.inference_mode():
            for row in data:
                kind = mode(row)
                if seen[kind] == args.per_mode:
                    continue
                bucket = MODES.index(kind)
                encoded = processor.apply_chat_template(
                    messages_for_processor(row, args.corpus.parent), tools=row["tools"],
                    enable_thinking=row.get("thinking", False), tokenize=True,
                    add_generation_prompt=False, return_dict=True, return_tensors="pt",
                )
                length = encoded["input_ids"].shape[-1]
                if length > args.max_tokens:
                    continue
                # A small positive logits_to_keep can also slice E4B's KV-shared layers.
                # The inner model skips vocabulary projection while visiting every token.
                teacher.model(**encoded.to("cuda", torch.bfloat16), use_cache=False)
                seen[kind] += 1
                tokens[kind] += length
                print(f"{kind} {seen[kind]}/{args.per_mode}: {row['id']} {length} tokens", flush=True)
                if all(seen[k] == args.per_mode for k in MODES):
                    break
            if not all(seen[k] == args.per_mode for k in MODES):
                raise RuntimeError(f"Insufficient eligible trajectories by modality: {dict(seen)}")
            for kind in MODES:
                scores[MODES.index(kind)].div_(tokens[kind])
            matrix = scores.mean(dim=0)
            for i, down in enumerate(downs):
                norm2 = torch.zeros(ORIGINAL, dtype=torch.float32, device="cuda")
                for batch in down.weight.split(128, dim=0):
                    norm2.add_(batch.float().square().sum(dim=0))
                matrix[i].mul_(norm2)
            matrix = matrix.cpu().numpy()
    finally:
        for hook in handles:
            hook.remove()
    if matrix.shape != (LAYERS, ORIGINAL) or not np.isfinite(matrix).all() or (matrix.min(axis=1) < 0).any() or (matrix.max(axis=1) <= 0).any():
        raise ValueError("Every dense layer needs finite, nonnegative, nonzero activation scores.")
    with new_directory(args.output) as stage:
        path = stage / "scores.npy"
        np.save(path, matrix, allow_pickle=False)
        write_json(stage / "calibration.json", {
            "base_model": MODEL, "base_revision": REVISION, "corpus_sha256": digest(args.corpus),
            "scores_sha256": digest(path), "shape": list(matrix.shape), "seed": args.seed,
            "estimator": "mean over modalities of E[down_input^2] times squared down column L2",
            "episodes_by_modality": dict(seen), "tokens_by_modality": dict(tokens),
            "max_tokens": args.max_tokens, "gpu_memory": peak(torch, before),
        })


def compact_identity(path):
    compaction = path / REPORT
    info = json.loads(compaction.read_text())
    if (info["base_model"], info["base_revision"], info["layers"], info["hidden_size"],
        info["width_before"], info["width_after"], info["selection"]) != (
            MODEL, REVISION, LAYERS, HIDDEN, ORIGINAL, COMPACT, "activation_scores"
        ) or not re.fullmatch(r"[0-9a-f]{64}", info.get("activation_scores_sha256") or ""):
        raise ValueError("Expected the activation-scored 10240 to 8192 pinned E4B compaction.")
    return {name: digest(path / name) for name in (REPORT, "config.json", INDEX)}


def fit(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

    identity = compact_identity(args.checkpoint)
    data = cases(args)
    before = memory(torch, args.vram_gib)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    backbone = load(args.checkpoint, COMPACT)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    template = _supervision_template(processor.chat_template)
    target_names = []
    found = set()
    for name, module in backbone.named_modules():
        if (match := TARGETS.fullmatch(name)) and isinstance(module, torch.nn.Linear):
            target_names.append(name)
            found.add((int(match[1]), match[2]))
    required = {(i, name) for i in range(LAYERS) for name in
                ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
    required.update((i, "self_attn.v_proj") for i in range(LAYERS - 18))
    if not required <= found:
        raise ValueError(f"Missing required pinned E4B projections: {sorted(required - found)}.")
    model = get_peft_model(backbone, LoraConfig(
        r=8, lora_alpha=16, target_modules=target_names, lora_dropout=0.04, task_type="CAUSAL_LM",
    ))
    backbone.config.use_cache = False
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone.enable_input_require_grads()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, fused=True)
    targets = []
    for row in data:
        indices = [i for i, message in enumerate(row["messages"]) if message["role"] == "assistant"]
        rng.shuffle(indices)
        targets.append(indices)
    steps, supervised = 0, 0
    losses = []
    by_family, by_mode = Counter(), Counter()
    for target_set in zip_longest(*targets):
        for row, target in zip(data, target_set, strict=True):
            if target is None:
                continue
            encoded, start = _encoded(processor, template, messages_for_processor(row, args.corpus.parent),
                                      row["tools"], target, thinking=row.get("thinking", False))
            length = encoded["input_ids"].shape[-1]
            if length > args.max_tokens:
                continue
            loss = _loss(backbone, encoded.to("cuda", torch.bfloat16), start)
            value = loss.item()
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite assistant loss in {row['id']}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(value)
            steps += 1
            supervised += length - start
            by_family[row["family"]] += 1
            by_mode[mode(row)] += 1
            print(f"{steps}/{args.steps} {row['id']} {length - start} supervised tokens loss={value:.4f}", flush=True)
            if steps == args.steps:
                break
        if steps == args.steps:
            break
    if steps != args.steps or (args.steps >= len(MODES) and not all(by_mode[kind] for kind in MODES)):
        raise RuntimeError(f"Insufficient eligible assistant targets: {steps} steps, modalities {dict(by_mode)}.")
    if not all(torch.isfinite(p).all().item() for p in trainable):
        raise ValueError("Nonfinite trained adapter parameter.")
    with new_directory(args.output) as stage:
        model.save_pretrained(stage, safe_serialization=True)
        write_json(stage / "training.json", {
            "base_model": MODEL, "base_revision": REVISION, "compact_identity": identity,
            "corpus_sha256": digest(args.corpus), "adapter_sha256": digest(stage / "adapter_model.safetensors"),
            "mode": "bf16-dense-lora", "rank": 8, "alpha": 16, "steps": steps, "seed": args.seed,
            "learning_rate": args.lr, "max_tokens": args.max_tokens, "supervised_tokens": supervised,
            "trainable_parameters": sum(p.numel() for p in trainable), "workflow_steps": dict(by_family),
            "modality_steps": dict(by_mode), "first_loss": losses[0], "last_loss": losses[-1],
            "gpu_memory": peak(torch, before),
        })


def merge(args):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor

    identity = compact_identity(args.checkpoint)
    fit_report = args.adapter / "training.json"
    training = json.loads(fit_report.read_text())
    if (training["compact_identity"] != identity or training["base_revision"] != REVISION
            or training["mode"] != "bf16-dense-lora"
            or training["adapter_sha256"] != digest(args.adapter / "adapter_model.safetensors")):
        raise ValueError("Adapter does not match this sliced E4B checkpoint.")
    before = memory(torch, args.vram_gib) if args.device == "cuda" else None
    model = load(args.checkpoint, COMPACT, device=args.device).eval()
    model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True).merge_and_unload(safe_merge=True)
    model.config.use_cache = True
    with new_directory(args.output) as stage:
        model.save_pretrained(stage, max_shard_size="2GB", safe_serialization=True)
        AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True).save_pretrained(stage)
        shutil.copyfile(args.checkpoint / REPORT, stage / REPORT)
        shutil.copyfile(fit_report, stage / fit_report.name)
        write_json(stage / "merge.json", {"base_revision": REVISION, "device": args.device,
                                        "gpu_memory": peak(torch, before) if before else None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, action in (("calibrate", calibrated), ("fit", fit), ("merge", merge)):
        command = commands.add_parser(name)
        command.add_argument("checkpoint", type=Path)
        if name == "merge":
            command.add_argument("adapter", type=Path)
            command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        else:
            command.add_argument("corpus", type=Path)
            command.add_argument("--per-family", type=int, default=6 if name == "calibrate" else 24)
            command.add_argument("--seed", type=int, default=4072)
            command.add_argument("--max-tokens", type=int, default=1536 if name == "calibrate" else 1792)
            command.add_argument("--dry-run", action="store_true", help="Check episode and modality selection without weights.")
        command.add_argument("output", type=Path)
        command.add_argument("--vram-gib", type=float, default=20)
        command.set_defaults(action=action)
    commands.choices["calibrate"].add_argument("--per-mode", type=int, default=8)
    commands.choices["fit"].add_argument("--steps", type=int, default=112)
    commands.choices["fit"].add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()
    if not 16 <= args.vram_gib <= 24:
        parser.error("--vram-gib must be between 16 and 24")
    if args.command != "merge" and (args.per_family < 1 or not 128 <= args.max_tokens <= 4096):
        parser.error("Use a positive per-family sample and 128–4096 tokens")
    if args.command == "calibrate" and not 1 <= args.per_mode <= 100:
        parser.error("Use 1–100 calibration episodes per modality")
    if args.command == "fit" and (not 1 <= args.steps <= 128 or not 0 < args.lr <= 1e-3):
        parser.error("Use 1–128 steps and a learning rate in (0, 0.001]")
    if getattr(args, "dry_run", False):
        selected = cases(args)
        print(json.dumps({"episodes": len(selected), "modalities": dict(Counter(map(mode, selected))),
                          "families": dict(Counter(row["family"] for row in selected)),
                          "corpus_sha256": digest(args.corpus)}, indent=2))
    else:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(args.output)
        args.action(args)


if __name__ == "__main__":
    main()
