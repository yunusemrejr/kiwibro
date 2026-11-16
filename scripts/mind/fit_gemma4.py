#!/usr/bin/env python3
"""Recover and specialize a depth-preserving compact Gemma 4 checkpoint on Colab.

Use training-only trajectories from agent_corpus.py. Each step supervises one
complete assistant message with its exact tool and image history. The full mode
updates weights in backward via factored Adafactor state on a single GPU.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
import random
import time

from scripts.mind.calibrate_gemma4 import messages_for_processor, rows
from scripts.mind.prune_gemma4 import BASE_REVISION

PROJECTION_CHUNK = 16


def _supervision_template(template: str) -> str:
    # For a new model turn, the canonical non-thinking generation prompt ends
    # with an empty thought channel. Include it before the final training
    # answer as well. Prior messages and continuations keep their normal form.
    start_turn = r"{{- '<|turn>' + role + '\n' }}"
    if template.count(start_turn) != 1:
        raise ValueError("Unexpected Gemma model-turn template.")
    return template.replace(start_turn, start_turn + r"""
        {%- if loop.last and role == 'model' and not enable_thinking
            and not (message.get('reasoning') or message.get('reasoning_content')) -%}
            {{- '<|channel>thought\n<channel|>' -}}
        {%- endif -%}
    """)


def _encoded(processor, template: str, messages: list[dict], tools: list[dict], target: int, *, thinking: bool = False):
    params = dict(chat_template=template, tools=tools, enable_thinking=thinking,
                  tokenize=True, return_dict=True, return_tensors="pt")
    prefix = processor.apply_chat_template(messages[:target], add_generation_prompt=True, **params)
    complete = processor.apply_chat_template(messages[:target + 1], add_generation_prompt=False, **params)
    input_ids = complete["input_ids"][0]
    prefix_tokens = prefix["input_ids"][0]
    start = len(prefix_tokens)
    while start and (start >= len(input_ids) or not prefix_tokens[:start].equal(input_ids[:start])):
        start -= 1
        if len(prefix_tokens) - start > 2:
            raise ValueError("Generation prompt is not the prefix of the Gemma assistant turn.")
    if start < 1 or start >= len(input_ids):
        raise ValueError("Assistant turn has no supervised tokens.")
    return complete, start


def _loss(model, encoded, start: int):
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint

    tokens = encoded["input_ids"]
    positions = torch.arange(start - 1, tokens.shape[1] - 1, device=tokens.device)
    expected = tokens[:, start:][0]
    # Keep Gemma's multimodal forward pass and selective final-layer evaluation.
    # A single vocabulary projection for a long answer can be several GiB, so
    # recompute only small groups of logits on backward.
    hidden = model.model(**encoded, logits_to_keep=positions, use_cache=False, return_dict=True).last_hidden_state
    if hidden.shape[1] == tokens.shape[1]:
        hidden = hidden.index_select(1, positions)
    if hidden.shape[1] != expected.numel():
        raise RuntimeError("Gemma returned an unexpected number of supervised states.")
    softcap = model.config.get_text_config().final_logit_softcapping

    def project(states, labels):
        logits = model.lm_head(states)
        if softcap is not None:
            logits = (logits / softcap).tanh() * softcap
        return F.cross_entropy(logits.float(), labels, reduction="sum")

    losses = [
        checkpoint(project, states, labels, use_reentrant=False)
        for states, labels in zip(hidden[0].split(PROJECTION_CHUNK), expected.split(PROJECTION_CHUNK), strict=True)
    ]
    return torch.stack(losses).sum() / expected.numel()


def fit(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A Colab CUDA accelerator with BF16 is required.")
    if args.steps <= 0 or args.max_tokens < 128:
        raise ValueError("Training requires positive steps and a usable token budget.")
    if args.output.exists():
        raise FileExistsError(args.output)
    corpus = rows(args.corpus, args.per_family, args.seed)
    if not corpus:
        raise ValueError("No training rows.")
    for row in corpus:
        thinking = row.get("thinking", False)
        assistant = [message for message in row["messages"] if message["role"] == "assistant"]
        if type(thinking) is not bool or (thinking and any(
            not isinstance(message.get("reasoning"), str) or not message["reasoning"].strip()
            for message in assistant
        )):
            raise ValueError(f"Invalid reasoning supervision in {row['id']}.")
    architecture_path = args.checkpoint / "architecture.json"
    architecture = json.loads(architecture_path.read_text())
    if architecture.get("base_revision") != BASE_REVISION:
        raise ValueError("Missing or mismatched compact architecture calibration.")
    initial_steps = 0
    checkpoint_report = args.checkpoint / "training.json"
    if checkpoint_report.is_file():
        baked = json.loads(checkpoint_report.read_text())
        initial_steps = baked.get("total_steps", baked.get("steps"))
        if baked.get("base_revision") != BASE_REVISION or type(initial_steps) is not int or initial_steps < 1:
            raise ValueError("The checkpoint's existing training report is invalid.")
    if args.init_adapter:
        prior = json.loads((args.init_adapter / "training.json").read_text())
        prior_steps = prior.get("total_steps", prior.get("steps"))
        if (
            args.mode != "l4-lora" or prior.get("mode") != args.mode
            or prior.get("base_revision") != BASE_REVISION
            or type(prior_steps) is not int or prior_steps <= initial_steps
            or (args.init_adapter / "architecture.json").read_bytes() != architecture_path.read_bytes()
        ):
            raise ValueError("The initial adapter must come from the same compact L4 checkpoint.")
        initial_steps = prior_steps
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    supervision_template = _supervision_template(processor.chat_template)
    packed_training = args.mode == "l4-lora"
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint, device_map="cpu" if packed_training else "cuda",
        dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    backbone = model
    packed_bytes = 0
    if packed_training:
        from scripts.mind.gemma4_expert_training import pack_experts_for_l4

        packed_bytes = pack_experts_for_l4(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.mode != "full":
        from peft import LoraConfig, PeftModel, get_peft_model

        if args.init_adapter:
            model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(
                r=16, lora_alpha=32,
                target_modules=r"model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|v_proj|o_proj)|router\.proj)",
                lora_dropout=0.04, task_type="CAUSAL_LM",
            ))
        model.enable_input_require_grads()

    trainable = [p for p in model.parameters() if p.requires_grad]
    parameter_count = sum(p.numel() for p in trainable)
    optimizer = None
    hooks = []
    if args.mode == "full":
        # Each expert bank's gradient is released after accumulation. Trying to
        # hold every BF16 gradient at once would exceed the accelerator budget.
        def install(parameter):
            stepper = torch.optim.Adafactor([parameter], lr=args.lr, foreach=False)

            def step(ready):
                stepper.step()
                ready.grad = None

            return parameter.register_post_accumulate_grad_hook(step)
        hooks.extend(install(parameter) for parameter in trainable)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, fused=True)

    model.train()
    coverage = Counter()
    reasoning_steps = 0
    losses = []
    seen = set()
    began = time.monotonic()
    for _ in range(args.epochs):
        rng.shuffle(corpus)
        for row in corpus:
            thinking = row.get("thinking", False)
            messages = messages_for_processor(row, args.corpus.parent)
            targets = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
            rng.shuffle(targets)
            for target in targets:
                if (row["id"], target) in seen:
                    continue
                encoded, start = _encoded(processor, supervision_template, messages, row["tools"], target,
                                          thinking=thinking)
                length = encoded["input_ids"].shape[-1]
                if length > args.max_tokens:
                    continue
                seen.add((row["id"], target))
                encoded = encoded.to("cuda", torch.bfloat16)
                loss = _loss(backbone, encoded, start)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at {row['id']} step {len(losses)}.")
                loss.backward()
                if optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(loss.item())
                coverage[row["family"]] += 1
                reasoning_steps += thinking
                print(f"{len(losses)}/{args.steps} {row['id']} {length - start} supervised tokens "
                      f"loss={losses[-1]:.4f}", flush=True)
                break  # diversify across tasks before taking a second target from a row
            if len(losses) >= args.steps:
                break
        if len(losses) >= args.steps:
            break
    if len(losses) < args.steps:
        raise RuntimeError("Too few eligible assistant turns for the requested steps and token budget.")

    for handle in hooks:
        handle.remove()
    if args.mode == "lora":
        model = model.merge_and_unload()
    model.config.use_cache = True
    args.output.mkdir(parents=True)
    model.save_pretrained(args.output, max_shard_size="4GB")
    processor.save_pretrained(args.output)
    (args.output / "architecture.json").write_bytes(architecture_path.read_bytes())
    with args.corpus.open("rb") as handle:
        digest = file_digest(handle, "sha256").hexdigest()
    report = {
        "base_revision": BASE_REVISION, "mode": args.mode, "parameters_tuned": parameter_count,
        "seed": args.seed, "learning_rate": args.lr, "steps": len(losses), "max_tokens": args.max_tokens,
        "initial_training_steps": initial_steps, "total_steps": initial_steps + len(losses),
        "reasoning_steps": reasoning_steps,
        "projection_chunk_tokens": PROJECTION_CHUNK,
        "corpus_sha256": digest, "workflow_steps": dict(coverage),
        "device": torch.cuda.get_device_name(), "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "frozen_fp8_expert_bytes": packed_bytes,
        "training_seconds": round(time.monotonic() - began, 2),
        "first_20_loss": sum(losses[:20]) / len(losses[:20]),
        "last_20_loss": sum(losses[-20:]) / len(losses[-20:]),
    }
    (args.output / "training.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=["full", "lora", "l4-lora"], default="full")
    parser.add_argument("--init-adapter", type=Path, help="Start an L4 fit from a prior adapter using fresh optimizer state.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per-family", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=4072)
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
