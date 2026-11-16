#!/usr/bin/env python3
"""Train Kiwi Mind 3.0 on a GPU machine (written for Google Colab): Spark X2.6 4B as the trunk, nine activated
LoRA layers on top of it, and a vision input, shipped as an IQ2_M GGUF pack that llama.cpp's server runs.

Stages, each a sub-command (gen is scripts/mind/train.py's, which writes the conversations):

  judge     the teacher labels real first messages HEAVY or LIGHT (the rubric of scripts/mind/heavy.py)
  trunk     full fine-tune of every Spark weight on the conversations, rendered with kiwibro/mind/render3.py
  gguf      f16 GGUF, importance matrix, IQ2_M, and the IQ2_M weights read back into torch
  vision-features  CLIP ViT-L/14 patch features (layer 22 of 24, as llama.cpp reads them) of the 16,000 COCO pictures
  vision-a  a new projector from CLIP ViT-L/14 patch features to the trunk's width, trained on COCO captions
  adapters  the route layer and the skill layers on the frozen IQ2_M weights, as activated LoRA
  vision-b  the see layer and the projector together on LLaVA-Instruct; writes see.pt and vision.gguf (the mmproj)
  eval      held-out numbers: route and tool-call accuracy, loss per skill with and without the layers, heavy_at
  export    the pack: trunk, route, skills, vision.gguf, manifest.json
  parity    llama.cpp's server against torch on held-out prompts

Layers (blocks SPLIT.. of the trunk): ``route`` is switched on at ``<think>`` and the skill's layer at
``</think>``; the server keeps what earlier tokens computed, so every piece of a prompt is trained under
exactly the layer that computes it there (see ``adapter_route``).
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str, *candidates: Path):
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location(name, candidate)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
    raise SystemExit(f"{name}: none of {[str(c) for c in candidates]} exists")


T = _load("kiwimind_train", HERE / "train.py")  # 2.x script: conversation builders, batching, schedule
R = _load("kiwimind_render3", HERE / "render3.py", HERE.parent.parent / "kiwibro" / "mind" / "render3.py")
log = T.log

WORK, BASE, TEACHER, SEED, MAX_TOKENS = T.WORK, T.BASE, T.TEACHER, T.SEED, T.MAX_TOKENS
SKILLS = R.SKILLS  # chat web code terminal computer tools write see
TEXT_SKILLS = SKILLS[:-1]
ADAPTERS = ("route", *SKILLS)  # adapter id: 0 route, 1.. the skills
SPLIT, RANK, ALPHA, B_RATE = 24, 16, 32.0, 4.0
WEIGHTS = T.WEIGHTS  # a token that is only shown, a word of an answer, a token of a tool call, a route word or first token
FIRST_TOKENS = T.FIRST_TOKENS
# the five matrices of a block that map one to one to a GGUF tensor: name there -> (holder, attribute) here
MATRICES = {"attn_qkv": ("self_attn", "q_k_v_proj"), "attn_output": ("self_attn", "out_proj"),
            "ffn_gate": ("mlp", "gate_proj"), "ffn_up": ("mlp", "up_proj"), "ffn_down": ("mlp", "down_proj")}
THINK_ID, THINK_END_ID, EOS_ID, BOS_ID, PAD_ID = 3, 4, 1, 0, 2
N_IMAGE = 256  # positions one picture takes
CALL_END_ID = 130984  # </tool_call>
# llama.cpp starts an activated LoRA this many tokens before its invocation token (measured by the parity stage)
ALORA_BACK = int(os.environ.get("KIWIMIND_ALORA_BACK", "1"))  # the offset measured on Colab (see alora_offset.py)
NEWEST_UNLAYERED = 0.7  # a reply request re-reads the newest turn's route piece with no layer on (measured on the laptop)
ALORA_SEEN = 0.7  # a layer is trained under the measured offset this often; the other two of 0, 1, 2 share the rest
ASKS_HEAVY = "judged.jsonl"

def kill_teacher() -> None:
    """Stop what is left of the teacher (the generator and vLLM's engine process), so training has the whole GPU.
    Read from /proc and killed one by one: a shell pkill -f would also hit the shell that runs it."""
    mine = {os.getpid(), os.getppid()}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in mine:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").lower()
        except OSError:
            continue
        if "vllm" in command or "enginecore" in command or "train.py gen" in command:
            log(f"killing {entry.name}: {command[:80]}")
            os.kill(int(entry.name), 9)
    time.sleep(5)


# ------------------------------------------------------------------ the heavy rubric

RUBRIC = (
    "You label requests for an AI agent whose own model is small: about 4 billion parameters, run at 2-bit precision on a "
    "laptop. A much larger model is available as a helper. Would the small model likely fail on the request below where the "
    "larger model does visibly better (several steps of reasoning, a maths word problem, code beyond a few lines, a long or "
    "structured text, an in-depth comparison or explanation)? Answer heavy for those. Greetings, a short fact, a one-line "
    "sum, a few lines of verse and anything one tool call carries out are light, and so is every request you are unsure "
    "about. Answer with JSON only: {\"heavy\": true or false}.")


def stage_judge(args) -> None:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    out = Path(args.out)
    kill_teacher()
    asks = T.load_asks(Path(args.asks))
    random.Random(SEED + 3).shuffle(asks)
    asks = asks[:24 if args.smoke else args.judge_n]
    schema = {"type": "object", "properties": {"heavy": {"type": "boolean"}}, "required": ["heavy"],
              "additionalProperties": False}
    llm = LLM(model=TEACHER, max_model_len=4096, gpu_memory_utilization=args.gpu, max_num_seqs=256, seed=SEED)
    params = SamplingParams(temperature=0.0, max_tokens=20, structured_outputs=StructuredOutputsParams(json=schema))
    talks = [[{"role": "system", "content": RUBRIC}, {"role": "user", "content": f"Request: {ask}"}] for ask in asks]
    outputs = llm.chat(talks, params, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
    kept = []
    for ask, output in zip(asks, outputs):
        try:
            kept.append({"ask": ask, "heavy": bool(json.loads(output.outputs[0].text)["heavy"])})
        except (ValueError, KeyError, TypeError):
            log(f"judge: unreadable label for {ask[:60]!r}: {output.outputs[0].text!r}")
    out.mkdir(parents=True, exist_ok=True)
    (out / ("judged-smoke.jsonl" if args.smoke else ASKS_HEAVY)).write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in kept) + "\n")
    log(f"judge: {len(kept)} of {len(asks)} labelled, {sum(item['heavy'] for item in kept)} heavy")


# ------------------------------------------------------------------ rows: one conversation as the trainer reads it

def judged_side(ask: str) -> str:
    return "held" if int(hashlib.sha256(ask.encode()).hexdigest()[:8], 16) % 10 == 0 else "train"


def tokenize_pieces(tokenizer, pieces: list[tuple[str, str]]):
    """The token ids of a prompt given in pieces, the piece each token belongs to, its place in the piece and whether
    it stands for an image. Every piece starts at a special token, so no token spans two of them. The marker
    <__media__> is where llama.cpp puts the embeddings of one picture: N_IMAGE positions here."""
    ids, owner, place, media = [], [], [], []
    for number, (_, body) in enumerate(pieces):
        mine = []
        for at, part in enumerate(body.split(R.MEDIA)):
            if at:
                mine += [(PAD_ID, True)] * N_IMAGE
            if part:
                mine += [(i, False) for i in tokenizer(part, add_special_tokens=False)["input_ids"]]
        ids += [i for i, _ in mine]
        media += [m for _, m in mine]
        owner += [number] * len(mine)
        place += range(len(mine))
    return ids, owner, place, media


def reply_routes(record: dict) -> tuple[list[tuple[str, bool, bool]], list[int], list[bool]]:
    """The route of every assistant message of a record (skill, heavy, act), the message index, and whether it is
    the first reply to its user message.
    Heavy only on the first reply to a user message, from the record's flag; later steps of a tool loop keep
    the skill and are light."""
    messages = record["messages"]
    turns = record.get("turns") or [{"at": next(i for i, m in enumerate(messages) if m["role"] == "user"),
                                     "skill": record["skill"], "hard": record["hard"]}]
    routes, answers, seen, firsts = [], [], set(), []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        turn = max((n for n, item in enumerate(turns) if item["at"] < index), default=0)
        first = turn not in seen
        seen.add(turn)
        firsts.append(first)
        routes.append((turns[turn]["skill"], bool(turns[turn].get("hard")) and first,
                       R.acts(R.assistant_text(message))))
        answers.append(index)
    return routes, answers, firsts


def encode_record(tokenizer, record: dict, catalog: dict) -> dict:
    """A conversation as arrays over its tokens: ids; kind (0 read, 1 route, 2 reply); tturn (the assistant turn a
    route, reply or closing EOS token belongs to, -1 elsewhere); code (how much the token counts when it is
    learned: an index into WEIGHTS)."""
    import torch

    messages, skip = record["messages"], set(record.get("skip") or [])
    routes, answers, firsts = reply_routes(record)
    tools = None if record.get("notools") else T.shown_tools(record, catalog)
    pieces = R.segments(messages, tools, routes=[R.route_text(*route) for route in routes], add_generation_prompt=False)
    ids, owner, places, media = tokenize_pieces(tokenizer, pieces)
    n = len(ids)
    kind, tturn, code = [0] * n, [-1] * n, [0] * n
    turn_of_piece, turn = {}, -1
    for number, (name, _) in enumerate(pieces):
        if name == "route":
            turn += 1
        if name != "read":
            turn_of_piece[number] = turn
    think_at, end_at, eos_at, trainable = [-1] * (turn + 1), [-1] * (turn + 1), [-1] * (turn + 1), [True] * (turn + 1)
    counted = {}
    for place, (piece, begin) in enumerate(zip(owner, places)):
        name = pieces[piece][0]
        if name == "read":
            if begin == 0 and piece > 0 and pieces[piece - 1][0] == "reply" and ids[place] == EOS_ID:
                a = turn_of_piece[piece - 1]  # the EOS that closes a reply: the skill learns to write it
                tturn[place], eos_at[a] = a, place
                if answers[a] not in skip:
                    code[place] = 2 if messages[answers[a]].get("tool_calls") else 1
            continue
        a = turn_of_piece[piece]
        kind[place], tturn[place] = (1 if name == "route" else 2), a
        trainable[a] = answers[a] not in skip
        if begin == 0:  # <think> or </think>: an invocation token, written by the runtime and never learned
            (think_at if name == "route" else end_at)[a] = place
        elif trainable[a]:
            if name == "route":
                code[place] = 3 if record.get("learn_route", True) else 0
            else:
                counted[a] = counted.get(a, 0) + 1
                code[place] = 3 if counted[a] <= FIRST_TOKENS else (2 if messages[answers[a]].get("tool_calls") else 1)
    if min(think_at + end_at + eos_at) < 0:
        raise ValueError(f"record {record['id']}: a turn lacks its route, reply or closing token")
    return {"id": record["id"], "ids": torch.tensor(ids, dtype=torch.int32), "kind": torch.tensor(kind, dtype=torch.uint8),
            "tturn": torch.tensor(tturn, dtype=torch.int8), "code": torch.tensor(code, dtype=torch.uint8),
            "turns": turn + 1, "tskill": [SKILLS.index(route[0]) for route in routes], "routes": [R.route_text(*r) for r in routes],
            "first": firsts, "act": [route[2] for route in routes], "heavy": [route[1] for route in routes],
            "think_at": think_at, "end_at": end_at, "eos_at": eos_at, "trainable": trainable,
            "tool": [(messages[i].get("tool_calls") or [{"function": {"name": ""}}])[0]["function"]["name"] for i in answers],
            "split": record["split"], "family": record.get("kind") if record.get("kind") in ("session", "repair", "real") else "single",
            "judged": False, "media": torch.tensor(media) if any(media) else None, "image": record.get("image")}


def encode_judged(tokenizer, ask: dict, skill: str | None, catalog: dict, rng: random.Random) -> dict:
    """A first message and its route up to the weight word: the skill word when the skill is known, the weight word
    from the teacher's label."""
    import torch

    seed = T.make_real_seed(ask["ask"], rng)
    messages = T.opening(seed) + [{"role": "user", "content": T.with_note(
        ask["ask"], seed["time"], reading=T.reading_lines(ask["ask"], seed["time"], rng), reminders=T.pick_reminders(rng))}]
    record = {"id": hashlib.sha256(ask["ask"].encode()).hexdigest()[:16], "messages": messages}
    pieces = R.segments(messages, T.shown_tools(record, catalog), add_generation_prompt=True)
    shown = skill or "chat"
    weight = "heavy" if ask["heavy"] else "light"
    pieces[-1] = ("route", f"{R.THINK}{shown} {weight}")
    ids, owner, places, _ = tokenize_pieces(tokenizer, pieces)
    n, last = len(ids), len(pieces) - 1
    kind, code = [0] * n, [0] * n
    for place, (piece, begin) in enumerate(zip(owner, places)):
        if piece == last:  # <think>, the skill word, the weight word
            kind[place] = 1
            if begin == 2 or (begin == 1 and skill):
                code[place] = 3
    return {"id": record["id"], "ids": torch.tensor(ids, dtype=torch.int32), "kind": torch.tensor(kind, dtype=torch.uint8),
            "tturn": torch.tensor([0 if k else -1 for k in kind], dtype=torch.int8), "code": torch.tensor(code, dtype=torch.uint8),
            "turns": 1, "tskill": [SKILLS.index(shown)], "routes": [f"{shown} {weight}"], "think_at": [kind.index(1)],
            "end_at": [n - 1], "eos_at": [n - 1], "trainable": [True], "tool": [""], "split": ask["split"],
            "family": "judged", "judged": True, "media": None, "image": None, "first": [True], "act": [False], "heavy": [bool(ask["heavy"])],
            "skill_known": bool(skill)}


def load_data(args, tokenizer) -> list[dict]:
    import torch

    out = Path(args.out)
    cache = out / "encoded3.pt"
    if cache.exists() and not args.fresh:
        return torch.load(cache)
    catalog = T.load_catalog(Path(args.tools))
    rows, dropped, known, relabelled = [], 0, {}, 0
    judged = [json.loads(line) for line in (out / ASKS_HEAVY).read_text().splitlines() if line.strip()] if (out / ASKS_HEAVY).exists() else []
    labels = {item["ask"]: item["heavy"] for item in judged}
    for line in Path(args.data).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") == "real":
            first = next(m for m in record["messages"] if m["role"] == "user")["content"]
            ask = first.split("\n\n" + T.NOTE_HEAD)[0].strip()
            known[ask] = (record["turns"][0]["skill"], record["split"])
            if ask in labels:  # the judge's word on a first message beats the generator's guess
                relabelled += record["turns"][0].get("hard") != labels[ask]
                record["turns"][0]["hard"] = record["hard"] = labels[ask]
        row = encode_record(tokenizer, record, catalog)
        if len(row["ids"]) > MAX_TOKENS or not int(row["code"].sum()):
            dropped += 1
            continue
        rows.append(row)
    conversations = len(rows)
    if judged:
        rng = random.Random(SEED + 5)
        for item in judged:
            skill, side = known.get(item["ask"], (None, judged_side(item["ask"])))
            row = encode_judged(tokenizer, {**item, "split": side}, skill, catalog, rng)
            if len(row["ids"]) <= MAX_TOKENS:
                rows.append(row)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, cache)
    lengths = sorted(len(row["ids"]) for row in rows)
    log(f"data: {conversations} conversations ({dropped} dropped) and {len(rows) - conversations} judged first messages "
        f"({sum(bool(row['heavy'][0]) for row in rows)} heavy first replies); tokens median {lengths[len(lengths) // 2]}, max {lengths[-1]}, "
        f"total {sum(lengths)}; held {sum(row['split'] == 'held' for row in rows)}; "
        f"assistant turns {sum(row['turns'] for row in rows if not row['judged'])}; first replies relabelled by the judge {relabelled}")
    return rows


# ------------------------------------------------------------------ what each position runs under

def adapter_route(row: dict, regime: str, rng: random.Random, back: int | None = None, unlayered: float = 0.0) -> dict | None:
    """The part of a row a pass reads, and under which layer each position runs.

    plain: no layers (the trunk alone), everything is learned.
    warm:  a running chat that keeps its cache: every route piece runs under ``route``, every reply piece under that
           turn's skill, every read piece under nothing; each token is learned where the position that predicts it
           ran under its own turn's layer, which is every route word and reply token and the closing EOS.
    cold:  the cache was lost: only the newest turn has layers, so the pass ends at that turn's EOS and learns just it.
    The invocation tokens <think> and </think> run under the layer they start and are never learned.
    unlayered: the chance that the newest turn's route piece is read under no layer, as a reply request reads it; the
           older turns keep the regime's mix."""
    import torch

    n, active = len(row["ids"]), list(range(row["turns"]))
    if regime == "cold":
        live = [a for a in active if row["trainable"][a]]
        if not live:
            return None
        active = [rng.choice(live)]
        n = row["eos_at"][active[0]] + 1
    tturn, kind = row["tturn"][:n].long(), row["kind"][:n].long()
    alive = torch.isin(tturn, torch.tensor(active)) if regime != "plain" else tturn >= 0
    route = torch.full((n,), -1, dtype=torch.long)
    if regime != "plain":
        skills = torch.tensor(row["tskill"])[tturn.clamp_min(0)] + 1
        route = torch.where(alive & (kind == 1), 0, torch.where(alive & (kind == 2), skills, -1))
        if back is None:
            back = ALORA_BACK if rng.random() < ALORA_SEEN else rng.choice([k for k in (0, 1, 2) if k != ALORA_BACK])
        bare = rng.random() < unlayered  # the newest turn's route piece under no layer
        if bare:
            route[(tturn == max(active)) & (kind == 1)] = -1
        for a in active:  # a build may start the layer up to two tokens before its invocation token
            if not (bare and a == max(active)):
                route[row["think_at"][a] - back:row["think_at"][a]] = 0
            if not row["judged"] and back:
                route[row["end_at"][a] - back:row["end_at"][a]] = int(skills[row["end_at"][a]])
    return {"ids": row["ids"][:n], "weight": torch.tensor(WEIGHTS)[row["code"][:n].long()] * alive, "route": route,
            "tturn": tturn, "kind": kind, "tskill": row["tskill"], "regime": regime, "first": row["first"],
            "family": row["family"], "heavy": row["heavy"]}


def collate(views: list[dict], indexes: list[int], pad: int, device) -> dict:
    import torch

    width = max(len(views[i]["ids"]) for i in indexes)
    ids = torch.full((len(indexes), width), pad, dtype=torch.long)
    weight = torch.zeros((len(indexes), width))
    route = torch.full((len(indexes), width), -1, dtype=torch.long)
    kind = torch.zeros((len(indexes), width), dtype=torch.long)
    owner = torch.full((len(indexes), width), -1, dtype=torch.long)  # skill of the turn each token belongs to
    for place, index in enumerate(indexes):
        view, size = views[index], len(views[index]["ids"])
        ids[place, :size], weight[place, :size], route[place, :size] = view["ids"].long(), view["weight"], view["route"]
        kind[place, :size] = view["kind"]
        owner[place, :size] = torch.where(view["tturn"] >= 0, torch.tensor(view["tskill"])[view["tturn"].clamp_min(0)], -1)
    return {"ids": ids.to(device), "weight": weight.to(device), "route": route.to(device), "kind": kind.to(device),
            "owner": owner.to(device)}


# ------------------------------------------------------------------ the model

def patch_attention(model) -> None:
    """The model's own attention is written out with a full score matrix; the fused kernel computes the same."""
    import torch
    import torch.nn.functional as F

    module = sys.modules[type(model).__module__]

    def attention(owner, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **_):
        key, value = module.repeat_kv(key, owner.num_key_value_groups), module.repeat_kv(value, owner.num_key_value_groups)
        mask = attention_mask[:, :, :, :key.shape[-2]] if attention_mask is not None else None
        if mask is not None and mask.dtype == torch.bool:  # a padded row with nothing to read would turn to NaN
            mask = mask | ~mask.any(-1, keepdim=True)
        causal = mask is None and query.shape[2] == key.shape[2] and query.shape[2] > 1
        return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, is_causal=causal, scale=scaling), None

    module.eager_attention_forward = attention
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):  # the model's code predates transformers 5's names
        module.__dict__[name] = (lambda original: lambda *, input_embeds=None, cache_position=None, **rest:
                                 original(inputs_embeds=input_embeds, **rest))(getattr(module, name))


def load_model(path: str = BASE, dtype=None):
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    cls = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], path)
    cls._tied_weights_keys = {"lm_head.weight": "model.embedding.weight"}  # the model's own list form breaks transformers 5
    model = cls.from_pretrained(path, config=config, dtype=dtype or torch.float32)
    patch_attention(model)
    model.config.use_cache = False
    return tokenizer, model.cuda()


def run_blocks(model, ids, first_grad: int = 0, checkpoint: bool = True, embeds=None):
    """The hidden states after the last norm, for right-padded ids. Blocks below first_grad run without gradient.
    embeds: the input vectors when some positions are pictures (else the embeddings of ids)."""
    import torch
    from torch.utils.checkpoint import checkpoint as remember

    inner, config = model.model, model.config
    module = sys.modules[type(model).__module__]
    length = ids.shape[1]
    positions = torch.arange(length, device=ids.device)
    ropes = {kind: module.compute_rope_cos_sin(positions, config.head_dim, config.get_rope_theta(kind),
                                               config.get_partial_rotary_factor(kind), ids.device)
             for kind in set(config.layer_types)}
    window = (positions[:, None] - positions[None, :])
    swa = ((window >= 0) & (window < config.sliding_window))[None, None]
    hidden = embeds if embeds is not None else inner.embedding(ids).float()
    for number, layer in enumerate(inner.layers):
        mask = swa if layer.layer_type == "sliding_attention" else None
        if number < first_grad:
            with torch.no_grad():
                hidden = layer(hidden, position_embeddings=ropes[layer.layer_type], attention_mask=mask)
        elif checkpoint:
            hidden = remember(layer, hidden, ropes[layer.layer_type], mask, use_reentrant=False)
        else:
            hidden = layer(hidden, position_embeddings=ropes[layer.layer_type], attention_mask=mask)
    return inner.norm(hidden)


def token_losses(model, batch: dict, first_grad: int = 0, checkpoint: bool = True, reduce: bool = True):
    """Cross-entropy of every token the batch teaches. reduce: the weighted mean; else the per-token losses and a
    boolean map of which tokens they are."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as remember

    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = run_blocks(model, batch["ids"], first_grad, checkpoint, batch.get("embeds"))
        table = model.model.embedding.weight.to(torch.bfloat16)
        wanted = batch["weight"][:, 1:] > 0
        states, targets = hidden[:, :-1][wanted], batch["ids"][:, 1:][wanted]

        def part(rows, tokens):
            return F.cross_entropy(F.linear(rows, table).float(), tokens, reduction="none")

        losses = torch.cat([remember(part, states[at:at + 2048], targets[at:at + 2048], use_reentrant=False)
                            for at in range(0, len(targets), 2048)]) if len(targets) else states.new_zeros(0).float()
    if not reduce:
        return losses, wanted
    weight = batch["weight"][:, 1:][wanted]
    return (losses * weight).sum() / weight.sum().clamp_min(1e-6)


def held_loss(model, views: list[dict], pad: int, budget: int, first_grad: int = 0, modules: dict | None = None) -> dict:
    """Plain cross-entropy over the learned tokens of held views: all, per skill (reply tokens) and the route words."""
    import torch

    total = torch.zeros(len(SKILLS) + 1, dtype=torch.float64)  # the last: route words
    count = torch.zeros(len(SKILLS) + 1, dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        for indexes in T.batches(views, budget, None):
            batch = collate(views, indexes, pad, "cuda")
            if modules:
                set_route(modules, batch["route"])
            losses, wanted = token_losses(model, batch, first_grad, checkpoint=False, reduce=False)
            owner = torch.where(batch["kind"][:, 1:][wanted] == 1, len(SKILLS), batch["owner"][:, 1:][wanted]).cpu()
            total.index_add_(0, owner, losses.double().cpu())
            count.index_add_(0, owner, torch.ones(len(owner), dtype=torch.float64))
    model.train()
    names = [*SKILLS, "route"]
    result = {name: float(total[i] / count[i]) for i, name in enumerate(names) if count[i]}
    result["all"] = float(total.sum() / count.sum().clamp_min(1))
    return result


def views_of(rows: list[dict], regime: str, rng: random.Random, back: int | None = None) -> list[dict]:
    return [view for row in rows for view in [adapter_route(row, regime, rng, back)] if view]


# ------------------------------------------------------------------ the layers

def make_adapted():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Adapted(nn.Module):
        """A frozen linear layer and one low-rank change per layer: y = W x + (alpha / r) B_i (A_i x) at every position
        whose layer id is i (route -1: none). Like the activated LoRA of llama.cpp, whose adapters hold A and B for
        the tensor of the same name."""

        def __init__(self, linear, count: int):
            super().__init__()
            self.linear = linear
            out_dim, in_dim = linear.weight.shape
            self.a = nn.Parameter(torch.empty(count, RANK, in_dim))
            self.b = nn.Parameter(torch.zeros(count, out_dim, RANK))
            for index in range(count):
                nn.init.kaiming_uniform_(self.a.data[index], a=math.sqrt(5))
            self.route, self.tail, self.present = None, None, []

        @property
        def weight(self):
            return self.linear.weight

        def forward(self, x):
            y = self.linear(x)
            route = self.route
            if route is None:
                return y
            if route.shape[1] != x.shape[1]:  # while a reply is written: one new position per sequence
                route = self.tail[:, None].expand(x.shape[0], x.shape[1])
            for index in self.present:
                pick = route == index
                change = F.linear(F.linear(x[pick], self.a[index].to(x.dtype)), self.b[index].to(x.dtype)) * (ALPHA / RANK)
                y = y.index_put((pick,), change.to(y.dtype), accumulate=True)
            return y

    return Adapted


def attach_adapters(model, count: int = len(ADAPTERS)) -> dict:
    Adapted = make_adapted()
    modules = {}
    for number, layer in enumerate(model.model.layers):
        if number < SPLIT:
            continue
        for name, (holder, attribute) in MATRICES.items():
            module = Adapted(getattr(getattr(layer, holder), attribute), count)
            setattr(getattr(layer, holder), attribute, module.cuda())
            modules[(number, name)] = module
    return modules


def set_route(modules: dict, route, tail=None) -> None:
    """Which layer each position of the next pass runs under (None: none anywhere); tail: the layer of the positions
    a sequence adds while a reply is written."""
    present = sorted(set(route.unique().tolist()) - {-1}) if route is not None else []
    if tail is not None and route is not None:
        present = sorted(set(present) | set(tail.unique().tolist()))
    for module in modules.values():
        module.route, module.tail, module.present = route, tail, present


def adapter_state(modules: dict, index: int) -> dict:
    """One layer's weights by GGUF tensor name: A as (r, in), B as (out, r)."""
    state = {}
    for (number, name), module in modules.items():
        state[f"blk.{number}.{name}.weight"] = (module.a[index].detach().float().cpu(), module.b[index].detach().float().cpu())
    return state


def schedule(step: int, steps: int, peak: float) -> float:
    return T.schedule(step, steps, peak)


# ------------------------------------------------------------------ stage: trunk

def save_hf(state: dict, folder: Path) -> None:
    """The trunk as a Hugging Face folder llama.cpp's converter reads: the base's config, tokenizer and code, one safetensors."""
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    base = Path(BASE)
    names = set(json.loads((base / "model.safetensors.index.json").read_text())["weight_map"])
    state = {key: value for key, value in state.items() if key in names}
    if set(state) != names:
        raise ValueError(f"weights to save differ from the base's: missing {sorted(names - set(state))[:3]}")
    for path in base.iterdir():
        if path.is_file() and not path.name.startswith(("model-", "model.safetensors", ".")):
            shutil.copy(path, folder / path.name)
    save_file({key: value.contiguous() for key, value in state.items()}, folder / "model.safetensors")
    (folder / "config.json").write_text((base / "config.json").read_text())


def stage_trunk(args) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out)
    kill_teacher()
    tokenizer, model = load_model()
    rows = load_data(args, tokenizer)
    rng = random.Random(SEED)
    train = views_of([row for row in rows if row["split"] == "train" or args.all_rows], "plain", rng)
    held = views_of([row for row in rows if row["split"] == "held"], "plain", rng)
    if args.smoke:
        train, held = train[:48], held[:16]
    start = held_loss(model, held, PAD_ID, args.budget)
    log(f"trunk: held loss of the base before training {start['all']:.4f} (route words {start.get('route', float('nan')):.4f})")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    plan = [b for _ in range(args.epochs) for b in T.batches(train, args.budget, rng)]
    if args.smoke:
        plan = plan[:4]
    log(f"trunk: {len(train)} sequences, {sum(len(v['ids']) for v in train)} tokens, {len(plan)} steps")
    model.train()
    began, seen, step, mark = time.time(), 0, 0, 0.0
    every = max(1, len(plan) // max(1, args.checks))
    best_loss, best_weights, best_step = start["all"], None, 0
    while step < len(plan):
        for group in optimizer.param_groups:
            group["lr"] = schedule(step, len(plan), args.lr)
        loss = token_losses(model, collate(train, plan[step], PAD_ID, "cuda"))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        seen += sum(len(train[i]["ids"]) for i in plan[step])
        if step % 20 == 0 or step == len(plan) - 1:
            log(f"trunk: step {step}/{len(plan)} loss {float(loss.detach()):.4f} {seen / (time.time() - began):.0f} tok/s")
        if (step + 1) % every == 0 or step == len(plan) - 1:
            now = held_loss(model, held, PAD_ID, args.budget)
            log(f"trunk: held loss {now['all']:.4f} (route words {now.get('route', float('nan')):.4f}) after step {step + 1}")
            if now["all"] < best_loss:
                best_loss, best_step = now["all"], step + 1
                best_weights = [p.detach().cpu().clone() for p in parameters]
        plan, mark = T.fit_plan(plan, step, mark, args.trunk_minutes * 60, "trunk")
        step += 1
    if best_weights is not None:
        with torch.no_grad():
            for parameter, kept in zip(parameters, best_weights):
                parameter.copy_(kept)
        del best_weights
    log(f"trunk: kept the weights of step {best_step or 'start'}, held loss {best_loss:.4f}; {time.time() - began:.0f} s")
    final = held_loss(model, held, PAD_ID, args.budget)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    save_hf(state, out / ("trunk-hf-smoke" if args.smoke else "trunk-hf"))
    if args.smoke:  # the save is where a long run would be lost, so the smoke run goes through it too
        shutil.rmtree(out / "trunk-hf-smoke")
        log("trunk: smoke run passed, saving included")
        return
    (out / "trunk.json").write_text(json.dumps({"held_loss_base": start, "held_loss_trunk": final, "steps": len(plan),
                                                "kept_step": best_step, "sequences": len(train), "seconds": time.time() - began}, indent=1))
    log(f"trunk: saved to {out / 'trunk-hf'}")

# ------------------------------------------------------------------ llama.cpp and the GGUF files

LLAMA = Path(os.environ.get("LLAMA_CPP", "/content/llama.cpp"))
GGUF_PARTS = {"attn_norm": "input_layernorm", "attn_qkv": "self_attn.q_k_v_proj", "attn_gate": "self_attn.g_proj",
              "attn_output": "self_attn.out_proj", "ffn_norm": "post_attention_layernorm", "ffn_gate": "mlp.gate_proj",
              "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
ROUTE_IDS = {"chat": 40539, "web": 10851, "code": 8012, "terminal": 48382, "computer": 116718, "tools": 37788, "write": 20192,
             "see": 8299, "light": 4077, "heavy": 13091, "say": 2775, "act": 5320}


def run_tool(command: list, logfile: Path) -> None:
    with open(logfile, "w") as handle:
        done = subprocess.run([str(part) for part in command], stdout=handle, stderr=subprocess.STDOUT)
    if done.returncode:
        tail = "\n".join(logfile.read_text(errors="replace").splitlines()[-15:])
        raise SystemExit(f"{Path(str(command[0])).name} failed with {done.returncode}; the end of {logfile}:\n{tail}")


def hf_name(name: str) -> str:
    """The Hugging Face name of a GGUF tensor of this architecture."""
    if name == "token_embd.weight":
        return "model.embedding.weight"
    if name == "output_norm.weight":
        return "model.norm.weight"
    found = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", name)
    if not found or found.group(2) not in GGUF_PARTS:
        raise ValueError(f"no place for the GGUF tensor {name}")
    return f"model.layers.{found.group(1)}.{GGUF_PARTS[found.group(2)]}.weight"


def source_weights(folder: Path):
    """name -> loader for the safetensors of a Hugging Face folder."""
    from safetensors import safe_open

    where = {}
    for path in sorted(folder.glob("*.safetensors")):
        with safe_open(path, "pt") as handle:
            where.update({key: path for key in handle.keys()})

    def get(name: str):
        with safe_open(where[name], "pt") as handle:
            return handle.get_tensor(name)

    return where, get


def read_quantized(path: Path, source: Path) -> tuple[dict, dict]:
    """Every tensor of a GGUF read back to fp32 under its Hugging Face name; each is checked against the weights the GGUF
    was made from (right shape, and an error that only a correctly placed 2-bit tensor can have)."""
    sys.path.insert(0, str(LLAMA / "gguf-py"))
    import gguf
    import numpy as np
    import torch

    where, get = source_weights(source)
    state, errors = {}, {}
    for tensor in gguf.GGUFReader(str(path)).tensors:
        name = hf_name(tensor.name)
        value = torch.from_numpy(np.ascontiguousarray(gguf.quants.dequantize(tensor.data, tensor.tensor_type))).float()
        wanted = get(name).float()
        if value.shape != wanted.shape:
            raise ValueError(f"{tensor.name}: {tuple(value.shape)} in the GGUF, {tuple(wanted.shape)} for {name}")
        error = float((value - wanted).norm() / wanted.norm())
        if not error < (0.9 if value.ndim > 1 else 0.02):
            raise ValueError(f"{tensor.name} reads back {error:.3f} away from {name}: placed wrongly?")
        state[name] = value
        errors.setdefault(tensor.tensor_type.name, []).append(error)
    if set(state) != set(where):
        raise ValueError(f"GGUF and source differ: {sorted(set(where) - set(state))[:3]} {sorted(set(state) - set(where))[:3]}")
    return state, {kind: {"tensors": len(v), "mean": sum(v) / len(v), "max": max(v)} for kind, v in errors.items()}


def load_deq(model, path: Path) -> None:
    from safetensors.torch import load_file

    missing, unexpected = model.load_state_dict(load_file(str(path)), strict=False)
    if unexpected or [key for key in missing if not key.startswith("lm_head")]:
        raise ValueError(f"the IQ2_M weights do not fit the model: missing {missing[:3]}, unexpected {unexpected[:3]}")


def imatrix_text(args, size: int) -> str:
    """Our own training conversations as the model reads them, until size characters."""
    catalog = T.load_catalog(Path(args.tools))
    records = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    records = [record for record in records if record["split"] == "train"]
    random.Random(SEED + 9).shuffle(records)
    texts, total = [], 0
    for record in records:
        routes, _, _ = reply_routes(record)
        tools = None if record.get("notools") else T.shown_tools(record, catalog)
        pieces = R.segments(record["messages"], tools, routes=[R.route_text(*route) for route in routes], add_generation_prompt=False)
        texts.append("".join(body for _, body in pieces).replace(R.MEDIA, ""))
        total += len(texts[-1])
        if total >= size:
            break
    log(f"imatrix text: {len(texts)} conversations, {total} characters")
    return "".join(texts)


def stage_gguf(args) -> None:
    out = Path(args.out)
    work = out / ("gguf-smoke" if args.smoke else "gguf")
    work.mkdir(parents=True, exist_ok=True)
    source = Path(BASE) if args.smoke else out / "trunk-hf"
    f16, quant, dat = work / "trunk-f16.gguf", work / "trunk-iq2m.gguf", work / "imatrix.dat"
    began = time.time()
    run_tool([sys.executable, LLAMA / "convert_hf_to_gguf.py", source, "--outfile", f16, "--outtype", "f16"], work / "convert.log")
    log(f"gguf: f16 {f16.stat().st_size / 1e9:.1f} GB after {time.time() - began:.0f} s")
    chunks, ctx = (6 if args.smoke else args.imatrix_chunks), 1024
    (work / "imatrix.txt").write_text(imatrix_text(args, chunks * ctx * 4))
    run_tool([LLAMA / "build/bin/llama-imatrix", "-m", f16, "-f", work / "imatrix.txt", "-o", dat, "-c", ctx, "--chunks", chunks,
              "-ngl", 0 if args.cpu else 99, "--parse-special", "--no-ppl"], work / "imatrix.log")
    log(f"gguf: importance matrix over {chunks} chunks of {ctx} tokens after {time.time() - began:.0f} s")
    run_tool([LLAMA / "build/bin/llama-quantize", "--imatrix", dat, f16, quant, "IQ2_M"], work / "quantize.log")
    log(f"gguf: IQ2_M {quant.stat().st_size / 1e9:.2f} GB after {time.time() - began:.0f} s")
    from safetensors.torch import save_file

    state, errors = read_quantized(quant, source)
    save_file(state, work / "deq.safetensors")
    (work / "gguf.json").write_text(json.dumps({"source": str(source), "bytes": quant.stat().st_size, "imatrix_chunks": chunks,
                                                "relative_error": errors, "seconds": time.time() - began}, indent=1))
    log(f"gguf: read back {len(state)} tensors, all placed; relative error by type {json.dumps(errors)}")
    f16.unlink()  # 8 GB nobody reads again


# ------------------------------------------------------------------ the layers: training

def load_adapters(modules: dict, states: dict) -> None:
    """states: adapter name -> {GGUF tensor name: (A, B)}."""
    for index, name in enumerate(ADAPTERS):
        for (number, matrix), module in modules.items():
            a, b = states[name][f"blk.{number}.{matrix}.weight"]
            module.a.data[index], module.b.data[index] = a.cuda(), b.cuda()


def stage_adapters(args) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out)
    tokenizer, model = load_model()
    load_deq(model, out / ("gguf-smoke" if args.smoke else "gguf") / "deq.safetensors")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = attach_adapters(model)
    rows = load_data(args, tokenizer)
    rng = random.Random(SEED + 11)
    held = views_of([row for row in rows if row["split"] == "held"], "warm", rng, back=ALORA_BACK)
    views = []
    for _ in range(args.epochs):
        for row in rows:
            if row["split"] == "train" or args.all_rows:
                view = adapter_route(row, "warm" if row["judged"] or rng.random() < 0.5 else "cold", rng, unlayered=NEWEST_UNLAYERED)
                if view:
                    views.append(view)
    if args.smoke:
        views, held = views[:24], held[:8]
    plan = T.batches(views, args.budget, rng)
    if args.smoke:
        plan = plan[:3]
    log(f"adapters: {len(views)} sequences, {len(plan)} steps, offset {ALORA_BACK} with probability {ALORA_SEEN}")
    groups = [{"params": [m.a for m in modules.values()], "rate": 1.0}, {"params": [m.b for m in modules.values()], "rate": B_RATE}]
    optimizer = torch.optim.AdamW(groups, lr=args.adapter_lr, betas=(0.9, 0.95), weight_decay=0.0)
    names = {name: i for i, name in enumerate(ADAPTERS)}
    best = {name: (float("inf"), None) for name in ADAPTERS}
    every = max(1, len(plan) // max(1, args.checks))
    began, step, mark = time.time(), 0, 0.0
    while step < len(plan):
        for group in optimizer.param_groups:
            group["lr"] = schedule(step, len(plan), args.adapter_lr) * group["rate"]
        batch = collate(views, plan[step], PAD_ID, "cuda")
        set_route(modules, batch["route"])
        loss = token_losses(model, batch, SPLIT, checkpoint=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([m for g in groups for m in g["params"]], 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == len(plan) - 1:
            log(f"adapters: step {step}/{len(plan)} loss {float(loss.detach()):.4f}")
        if (step + 1) % every == 0 or step == len(plan) - 1:
            now = held_loss(model, held, PAD_ID, args.budget, SPLIT, modules)
            log(f"adapters: held loss {json.dumps({k: round(v, 4) for k, v in now.items()})} after step {step + 1}")
            for name, index in names.items():
                if name in now and now[name] < best[name][0]:
                    best[name] = (now[name], [(m.a[index].detach().cpu().clone(), m.b[index].detach().cpu().clone()) for m in modules.values()])
        plan, mark = T.fit_plan(plan, step, mark, args.adapter_minutes * 60, "adapters")
        step += 1
    with torch.no_grad():
        for name, (_, kept) in best.items():
            if kept is not None:
                for module, (a, b) in zip(modules.values(), kept):
                    module.a.data[names[name]], module.b.data[names[name]] = a.cuda(), b.cuda()
    final = held_loss(model, held, PAD_ID, args.budget, SPLIT, modules)
    log(f"adapters: kept the best state of each layer; held loss {json.dumps({k: round(v, 4) for k, v in final.items()})}")
    torch.save({name: adapter_state(modules, index) for name, index in names.items()}, out / ("adapters-smoke.pt" if args.smoke else "adapters.pt"))
    if args.smoke:
        return
    (out / "adapters.json").write_text(json.dumps({"held_loss": final, "best": {k: v[0] for k, v in best.items()}, "steps": len(plan),
                                                   "seconds": time.time() - began, "sequences": len(views), "offset": ALORA_BACK}, indent=1))
    log(f"adapters: saved to {out / 'adapters.pt'}")


# ------------------------------------------------------------------ the numbers

def route_scores(model, modules, views: list[dict], pad: int, budget: int) -> dict:
    """Teacher-forced reading of every route a held view holds: the skill word among the eight, the weight word among two,
    say or act among two. The probability of heavy at the weight word is kept for first replies to real asks."""
    import torch
    import torch.nn.functional as F

    skills = torch.tensor([ROUTE_IDS[name] for name in SKILLS]).cuda()
    weigh = torch.tensor([ROUTE_IDS["light"], ROUTE_IDS["heavy"]]).cuda()
    acts = torch.tensor([ROUTE_IDS["say"], ROUTE_IDS["act"]]).cuda()
    tally, pairs = {"skill": [0, 0], "weight": [0, 0], "act": [0, 0]}, []
    set_route(modules, None) if modules else None
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for indexes in T.batches(views, budget, None):
            batch = collate(views, indexes, pad, "cuda")
            if modules:
                set_route(modules, batch["route"])
            hidden = run_blocks(model, batch["ids"], 0, False)
            table = model.model.embedding.weight.to(torch.bfloat16)
            for place, index in enumerate(indexes):
                view = views[index]
                for turn in sorted(set(view["tturn"][view["kind"] == 1].tolist())):
                    at = int(((view["kind"] == 1) & (view["tturn"] == turn)).nonzero()[0])
                    for step, (word, options) in enumerate((("skill", skills), ("weight", weigh), ("act", acts))):
                        target = at + step + 1
                        if target >= len(view["ids"]) or not float(view["weight"][target]) > 0:
                            continue
                        logits = F.linear(hidden[place, at + step], table).float()[options]
                        gold = int(view["ids"][target])
                        tally[word][0] += int(options[int(logits.argmax())]) == gold
                        tally[word][1] += 1
                        if word == "weight" and view["first"][turn] and view["family"] in ("judged", "real"):
                            pairs.append((float(logits.softmax(-1)[1]), gold == ROUTE_IDS["heavy"]))
    model.train()
    return {"accuracy": {k: v[0] / max(1, v[1]) for k, v in tally.items()}, "counts": {k: v[1] for k, v in tally.items()}, "heavy_pairs": pairs}


def heavy_threshold(pairs: list) -> dict:
    """The cut on P(heavy) with the best F1 for heavy against the labels, and the cut that lets 3% of light asks through."""
    best, strict = None, None
    for percent in range(1, 100):
        cut = percent / 100
        tp = sum(p >= cut and g for p, g in pairs)
        fp = sum(p >= cut and not g for p, g in pairs)
        fn = sum(p < cut and g for p, g in pairs)
        tn = len(pairs) - tp - fp - fn
        found = {"heavy_at": cut, "f1": 2 * tp / max(1, 2 * tp + fp + fn), "accuracy": (tp + tn) / max(1, len(pairs)),
                 "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "false_heavy": fp / max(1, fp + tn)}
        if best is None or found["f1"] > best["f1"]:
            best = found
        if strict is None and found["false_heavy"] <= 0.03:
            strict = found
    return {**best, "pairs": len(pairs), "heavy": sum(g for _, g in pairs), "at_3_percent_false_heavy": strict}


def tool_scores3(model, tokenizer, modules, rows: list[dict], catalog: dict, limit: int, size: int = 8) -> dict:
    """Greedy first reply of held turns whose gold reply is a tool call, after the route the conversation holds (the app
    opens the reply with the tool-call tag): is the named tool right, is the call JSON with the required arguments."""
    import torch

    opened = tokenizer("<tool_call>\n", add_special_tokens=False)["input_ids"]
    rng, cases = random.Random(SEED + 13), []
    for row in rows:
        if row["judged"] or row["split"] != "held":
            continue
        view = adapter_route(row, "warm", rng, back=ALORA_BACK)
        for turn in range(row["turns"]):
            if row["trainable"][turn] and row["tool"][turn] and row["act"][turn]:
                cut = row["end_at"][turn] + 1
                cases.append({"ids": row["ids"][:cut].long().tolist() + opened, "route": view["route"][:cut].tolist() + [row["tskill"][turn] + 1] * len(opened),
                              "skill": row["tskill"][turn] + 1, "gold": row["tool"][turn]})
    rng.shuffle(cases)
    cases = cases[:limit]
    named = valid = 0
    model.eval()
    for begin in range(0, len(cases), size):
        part = cases[begin:begin + size]
        width = max(len(c["ids"]) for c in part)
        ids = torch.full((len(part), width), PAD_ID, dtype=torch.long)
        attention, route = torch.zeros_like(ids), torch.full_like(ids, -1)
        for place, case in enumerate(part):
            n = len(case["ids"])
            ids[place, width - n:], attention[place, width - n:], route[place, width - n:] = torch.tensor(case["ids"]), 1, torch.tensor(case["route"])
        if modules:
            set_route(modules, route.cuda(), torch.tensor([c["skill"] for c in part]).cuda())
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            made = model.generate(input_ids=ids.cuda(), attention_mask=attention.cuda(), max_new_tokens=96, do_sample=False,
                                  eos_token_id=[EOS_ID, CALL_END_ID], pad_token_id=PAD_ID, use_cache=True)
        for case, new in zip(part, made[:, width:]):
            text = "<tool_call>\n" + tokenizer.decode(new, skip_special_tokens=False).replace("<｜end▁of▁sentence｜>", "")
            found = re.search(r'"name"\s*:\s*"([^"]+)"', text)
            name = found.group(1) if found else ""
            named += name == case["gold"]
            try:
                call = json.loads(text.split("<tool_call>")[1].split("</tool_call>")[0])
                required = (catalog.get(call["name"], {}).get("function", {}).get("parameters") or {}).get("required") or []
                valid += call["name"] in catalog and all(key in call.get("arguments", {}) for key in required)
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                pass
    model.train()
    return {"cases": len(cases), "named": named / max(1, len(cases)), "valid": valid / max(1, len(cases))}


def stage_eval(args) -> None:
    import torch

    out = Path(args.out)
    tokenizer, model = load_model()
    load_deq(model, out / ("gguf-smoke" if args.smoke else "gguf") / "deq.safetensors")
    rows = load_data(args, tokenizer)
    catalog = T.load_catalog(Path(args.tools))
    rng = random.Random(SEED + 17)
    held_rows = [row for row in rows if row["split"] == "held"]
    for name, expected in ((n, ROUTE_IDS[n]) for n in SKILLS):  # the word ids this script and the app rely on
        got = tokenizer(f"{R.THINK}{name} light say", add_special_tokens=False)["input_ids"]
        if got != [THINK_ID, expected, ROUTE_IDS["light"], ROUTE_IDS["say"]]:
            raise ValueError(f"route word {name}: tokens {got}")
    result = {}
    plain, warm = views_of(held_rows, "plain", rng), views_of(held_rows, "warm", rng, back=ALORA_BACK)
    if args.smoke:
        plain, warm, held_rows = plain[:16], warm[:16], held_rows[:16]
    result["quantized"] = held_loss(model, plain, PAD_ID, args.budget)
    modules = attach_adapters(model)
    states = torch.load(out / ("adapters-smoke.pt" if args.smoke else "adapters.pt"))
    load_adapters(modules, states)
    result["layered"] = held_loss(model, warm, PAD_ID, args.budget, SPLIT, modules)
    set_route(modules, None)
    result["layers_off"] = held_loss(model, warm, PAD_ID, args.budget, SPLIT, None)
    routed = route_scores(model, modules, warm, PAD_ID, args.budget)
    pairs = routed.pop("heavy_pairs")
    result["route"] = routed
    result["heavy"] = heavy_threshold(pairs) if pairs else None
    result["tools_layered"] = tool_scores3(model, tokenizer, modules, rows, catalog, 8 if args.smoke else args.eval_cases)
    log(f"eval: layered {json.dumps(result['layered'])}\neval: route {json.dumps(routed)}\neval: heavy {json.dumps(result['heavy'])}\neval: tools {result['tools_layered']}")
    del model, modules
    torch.cuda.empty_cache()
    if not args.smoke and (out / "trunk-hf").exists():  # the fine-tuned trunk in full precision, no layers: the ceiling
        tokenizer, model = load_model(str(out / "trunk-hf"))
        result["tools_trained"] = tool_scores3(model, tokenizer, None, rows, catalog, args.eval_cases)
        result["route_trained"] = {k: v for k, v in route_scores(model, None, plain, PAD_ID, args.budget).items() if k != "heavy_pairs"}
        log(f"eval: trained trunk tools {result['tools_trained']} route {result['route_trained']}")
    trunk = json.loads((out / "trunk.json").read_text()) if (out / "trunk.json").exists() else {}
    result["base"], result["trained"] = trunk.get("held_loss_base"), trunk.get("held_loss_trunk")
    (out / ("eval-smoke.json" if args.smoke else "eval.json")).write_text(json.dumps(result, indent=1))
    log(f"eval: written to {out / 'eval.json'}")


# ------------------------------------------------------------------ the pack

def write_lora_gguf(path: Path, tensors: list, alpha: float, invocation: list[int]) -> None:
    """An activated LoRA as llama.cpp reads it: keys, then for every tensor <name>.lora_a (dims [in, r]) and <name>.lora_b
    (dims [r, out]) in f32, 32-byte aligned (the format the lead's alora2_test.py proved against llama-server)."""
    import struct

    import numpy as np

    def text(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    keys = [text("general.type") + struct.pack("<I", 8) + text("adapter"), text("general.architecture") + struct.pack("<I", 8) + text("spark2_5"),
            text("adapter.type") + struct.pack("<I", 8) + text("lora"), text("adapter.lora.alpha") + struct.pack("<If", 6, float(alpha)),
            text("adapter.alora.invocation_tokens") + struct.pack("<IIQ", 9, 4, len(invocation)) + struct.pack(f"<{len(invocation)}I", *invocation)]
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(keys)) + b"".join(keys)
    infos, blobs, offset = b"", [], 0
    for name, array in tensors:
        dims = tuple(reversed(array.shape))
        infos += text(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", 0, offset)
        blob = np.ascontiguousarray(array, dtype=np.float32).tobytes()
        blob += b"\0" * (-len(blob) % 32)
        blobs.append(blob)
        offset += len(blob)
    body = head + infos
    body += b"\0" * (-len(body) % 32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body + b"".join(blobs))


def beside(pack: Path, stem: str, ext: str) -> Path:
    """A file next to a pack: the default pack folder keeps the plain names, another folder gets its own."""
    return pack.parent / (f"{stem}{ext}" if pack.name == "pack" else f"{stem}-{pack.name}{ext}")


def parity_cases(rows: list[dict], count: int = 16) -> list[tuple[dict, int]]:
    """(row, turn) of held text conversations: half first turns, half later turns of a running conversation."""
    rng = random.Random(SEED + 19)
    pool = [row for row in rows if row["split"] == "held" and not row["judged"] and row["media"] is None]
    later = [(row, turn) for row in pool for turn in range(1, row["turns"]) if row["trainable"][turn]]
    first = [(row, 0) for row in pool if row["trainable"][0]]
    if len(later) < count // 2 or len(first) < count // 2:
        raise ValueError(f"parity needs {count // 2} first and {count // 2} later turns, found {len(first)} and {len(later)}")
    return rng.sample(first, count // 2) + rng.sample(later, count // 2)


def parity_prompts(model, modules, cases: list) -> list[dict]:
    """The prompts a request holds and the token the torch model names next, under the rule llama.cpp applies to a prompt it has
    no cache for: no layer before the invocation token minus ALORA_BACK, the requested layer from there on."""
    import torch
    import torch.nn.functional as F

    made = []
    model.eval()
    for row, turn in cases:
        for layer, cut in (("route", row["think_at"][turn] + 1), (SKILLS[row["tskill"][turn]], row["end_at"][turn] + 1)):
            ids = row["ids"][:cut].long()
            index = ADAPTERS.index(layer)
            route = torch.full((1, cut), -1, dtype=torch.long)
            route[0, cut - 1 - ALORA_BACK:] = index
            set_route(modules, route.cuda())
            with torch.no_grad():
                hidden = run_blocks(model, ids[None].cuda(), 0, False)
                logits = F.linear(hidden[0, -1], model.model.embedding.weight).float()
            top = logits.topk(2)
            case = {"id": row["id"], "turn": turn, "layer": layer, "multi_turn": turn > 0, "ids": ids.tolist(), "next": int(top.indices[0]),
                    "next_text": "", "margin": float(top.values[0] - top.values[1])}
            if layer == "route":
                words = torch.tensor([ROUTE_IDS[name] for name in SKILLS]).cuda()
                case["next_route"] = int(words[logits[words].argmax()])
            made.append(case)
    return made


def stage_export(args) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out)
    pack = Path(args.pack) if args.pack else out / "pack"
    if pack.exists():
        shutil.rmtree(pack)
    (pack / "skills").mkdir(parents=True)
    states = torch.load(out / "adapters.pt")
    vision, see_file = out / "vision" / "vision.gguf", out / "vision" / "see.pt"
    with_vision = vision.exists() and see_file.exists() and not args.no_vision
    if with_vision:
        seen = torch.load(see_file)
        states["see"] = seen["see"]  # the layer the vision stage trained, in place of the text run's untouched one
    skills = list(SKILLS) if with_vision else list(TEXT_SKILLS)
    shutil.copy(out / "gguf" / "trunk-iq2m.gguf", pack / "kiwimind-trunk.gguf")
    for name in ("route", *skills):
        tensors = [(f"{tensor}.{side}", (array.numpy())) for tensor, pair in states[name].items() for side, array in zip(("lora_a", "lora_b"), pair)]
        write_lora_gguf(pack / ("route.gguf" if name == "route" else f"skills/{name}.gguf"), tensors, ALPHA, [THINK_ID] if name == "route" else [THINK_END_ID])
    if with_vision:
        shutil.copy(vision, pack / "vision.gguf")
    evaluation = json.loads((out / "eval.json").read_text())
    trunk = json.loads((out / "trunk.json").read_text())
    rows = torch.load(out / "encoded3.pt")
    heavy = evaluation["heavy"]
    manifest = {"name": "kiwimind", "version": "3.0.0", "engine": "llama", "base": "XHToken/Spark-X2.6-4B", "quant": "IQ2_M", "split": SPLIT,
                "rank": RANK, "ctx": 16384, "skills": skills, "route": {"file": "route.gguf", "heavy_at": heavy["heavy_at"]}}
    if with_vision:
        manifest["vision"] = {"file": "vision.gguf", "tokens": N_IMAGE}
    manifest["lessons"] = {"trunk": sum(r["split"] == "train" for r in rows), "skills": sum(r["split"] == "train" for r in rows),
                           "held": sum(r["split"] == "held" for r in rows)}
    manifest["report"] = {"held_loss": {"base": (evaluation["base"] or {}).get("all"), "trained": (evaluation["trained"] or {}).get("all"),
                                        "quantized": evaluation["quantized"]["all"], "layered": evaluation["layered"]["all"]},
                          "steps": trunk["steps"], "minutes": round(trunk["seconds"] / 60, 1),
                          "route_accuracy": evaluation["route"]["accuracy"], "tool_call_accuracy": evaluation["tools_layered"]["named"],
                          "tool_call_valid": evaluation["tools_layered"]["valid"], "heavy_f1": heavy["f1"], "heavy_accuracy": heavy["accuracy"]}
    manifest["files"] = {str(p.relative_to(pack)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in sorted(pack.rglob("*")) if p.is_file()}
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"export: pack at {pack}: {sum(v['bytes'] for v in manifest['files'].values()) / 1e9:.2f} GB in {len(manifest['files'])} files, skills {skills}")
    tokenizer, model = load_model()
    load_deq(model, out / "gguf" / "deq.safetensors")
    modules = attach_adapters(model)
    load_adapters(modules, states)
    cases = parity_cases(rows)
    prompts = parity_prompts(model, modules, cases)
    for case in prompts:
        case["next_text"] = tokenizer.decode([case["next"]])
    if with_vision:
        projector = make_projector(model.model.embedding.weight.shape[1]).cuda()
        projector.load_state_dict(seen["projector"])
        prompts += picture_parity(model, modules, projector, tokenizer, out, args.tools)
        log(f"export: {sum('images' in c for c in prompts)} picture cases")
    beside(pack, "parity", ".json").write_text(json.dumps({"offset": ALORA_BACK, "cases": prompts}))
    log(f"export: parity.json with {len(prompts)} prompts ({sum(c['multi_turn'] for c in prompts)} from later turns) next to the pack")


def stage_parity(args) -> None:
    """llama.cpp's server with the pack against the next tokens torch named (parity.json)."""
    import urllib.request

    pack = Path(args.pack) if args.pack else Path(args.out) / "pack"
    manifest = json.loads((pack / "manifest.json").read_text())
    cases = json.loads(beside(pack, "parity", ".json").read_text())["cases"]
    names = ["route", *manifest["skills"]]
    files = [pack / ("route.gguf" if n == "route" else f"skills/{n}.gguf") for n in names]
    command = [LLAMA / "build/bin/llama-server", "-m", pack / "kiwimind-trunk.gguf", "-c", 8192, "-ngl", 999, "--port", args.port, "--host", "127.0.0.1",
               "--special", "--parallel", 1, "--ctx-checkpoints", 8, "--no-cache-idle-slots", "--no-webui", "--lora-init-without-apply"]
    for file in files:
        command += ["--lora", file]
    if "vision" in manifest:
        command += ["--mmproj", pack / manifest["vision"]["file"]]
    logfile = open(beside(pack, "parity-server", ".log"), "w")
    server = subprocess.Popen([str(c) for c in command], stdout=logfile, stderr=subprocess.STDOUT)

    def call(path, body=None):
        request = urllib.request.Request(f"http://127.0.0.1:{args.port}{path}", data=json.dumps(body).encode() if body is not None else None,
                                         headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(request, timeout=900).read())

    try:
        for _ in range(600):
            if server.poll() is not None:
                raise SystemExit(f"llama-server stopped; see {beside(pack, 'parity-server', '.log')}")
            try:
                if call("/health").get("status") == "ok":
                    break
            except OSError:
                time.sleep(1)
        grammar = R.route_grammar()
        marker = call("/props").get("media_marker") or R.MEDIA  # newer servers name the picture marker themselves
        report, bad = [], []
        for case in cases:
            lora = [{"id": i, "scale": 1.0 if n == case["layer"] else 0.0} for i, n in enumerate(names)]
            prompt = ({"prompt_string": case["prompt_string"].replace(R.MEDIA, marker), "multimodal_data": case["images"]}
                      if "images" in case else case["ids"])
            body = {"prompt": prompt, "n_predict": 1, "temperature": 0, "cache_prompt": False, "n_probs": 5, "lora": lora}
            free = call("/completion", body)["completion_probabilities"][0]
            row = {"id": case["id"], "layer": case["layer"], "multi_turn": case["multi_turn"], "margin": case["margin"], "torch": case["next"],
                   "llama": free["id"], "agree": free["id"] == case["next"]}
            if case["layer"] == "route":
                held = call("/completion", {**body, "grammar": grammar})["completion_probabilities"][0]
                row.update({"torch_route": case["next_route"], "llama_route": held["id"], "agree_route": held["id"] == case["next_route"]})
            report.append(row)
            if not row["agree"] or not row.get("agree_route", True):
                bad.append(row)
        sure = [b for b in bad if b["margin"] >= 0.5]
        summary = {"cases": len(report), "agree": sum(r["agree"] for r in report), "route_agree": sum(r.get("agree_route", True) for r in report),
                   "disagree": bad, "disagree_with_margin_over_half": len(sure)}
        beside(pack, "parity-result", ".json").write_text(json.dumps({"summary": summary, "cases": report}, indent=1))
        log(f"parity: {summary['agree']} of {len(report)} first tokens agree, route with grammar {summary['route_agree']}; {len(sure)} disagreements with a clear margin")
        if sure:
            raise SystemExit(f"parity failed on {len(sure)} cases with a margin over 0.5: {json.dumps(sure[:3])}")
    finally:
        server.terminate()
        server.wait(timeout=60)
        logfile.close()


# ------------------------------------------------------------------ vision: pictures

VIS = Path(os.environ.get("KIWIMIND_VIS", "/content/vis"))
CLIP_LAYER = 22  # llama.cpp's graph runs 22 of CLIP's 24 blocks and reads the output of the 22nd: hidden_states[22] (measured)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
CANVAS = (122, 116, 104)  # the grey llama.cpp pads a picture with
HELD_PICTURES = 200
SEE_ASKS = ("What is in this picture?", "Describe this image.", "What do you see here?", "Describe what is in the picture.",
            "Tell me what this photo shows.", "What is happening in this image?")


def clip_pixels(path):
    """A picture as llama.cpp's fixed-size preprocessor makes it: scaled to fit 224 x 224 (Pillow bicubic), pasted in the
    middle of a grey canvas, then (x / 255 - mean) / std. Shape (3, 224, 224)."""
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = min(np.float32(224) / np.float32(width), np.float32(224) / np.float32(height))
    new = (min(int(np.ceil(np.float32(width) * scale)), 224), min(int(np.ceil(np.float32(height) * scale)), 224))
    canvas = Image.new("RGB", (224, 224), CANVAS)
    canvas.paste(image.resize(new, Image.BICUBIC), ((224 - new[0]) // 2, (224 - new[1]) // 2))
    pixels = (np.asarray(canvas, dtype=np.float32) / 255.0 - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    return pixels.transpose(2, 0, 1)


def stage_vision_features(args) -> None:
    """CLIP ViT-L/14's patch features of every picture, from the layer llama.cpp reads, without the CLS token: (n, 256, 1024) in fp16."""
    import numpy as np
    import torch
    from transformers import CLIPVisionModel

    work = Path(args.out) / ("vision-smoke" if args.smoke else "vision")
    work.mkdir(parents=True, exist_ok=True)
    names = json.loads((VIS / "images.json").read_text())
    names = names[:64] if args.smoke else names
    model = CLIPVisionModel.from_pretrained(str(VIS / "clip"), dtype=torch.float16).cuda().eval()
    features = torch.empty(len(names), N_IMAGE, 1024, dtype=torch.float16)
    began = time.time()
    for at in range(0, len(names), 64):
        pixels = torch.from_numpy(np.stack([clip_pixels(VIS / "images" / name) for name in names[at:at + 64]])).half().cuda()
        with torch.no_grad():
            features[at:at + len(pixels)] = model(pixel_values=pixels, output_hidden_states=True).hidden_states[CLIP_LAYER][:, 1:].cpu()
        if (at // 64) % 25 == 0:
            log(f"vision-features: {at + len(pixels)} of {len(names)} after {time.time() - began:.0f} s")
    torch.save({"names": names, "features": features}, work / "clip-features.pt")
    log(f"vision-features: {len(names)} pictures, {tuple(features.shape)}, layer {CLIP_LAYER}, {time.time() - began:.0f} s")


def make_projector(width: int):
    """mm.0 Linear(1024, width), GELU (the tanh form ggml uses), mm.2 Linear(width, width): llama.cpp's mlp projector."""
    import torch.nn as nn

    return nn.Sequential(nn.Linear(1024, width), nn.GELU(approximate="tanh"), nn.Linear(width, width))


def picture_record(kind: str, number: int, name: str, exchanges: list[tuple[str, str]], split: str) -> dict:
    """A conversation about one picture as a record encode_record reads: the picture comes with the first message
    (which carries the note the app adds); every reply is routed ``see light say``."""
    rng = random.Random(f"{kind}-{name}")
    seed = T.make_real_seed(exchanges[0][0], rng)
    first = T.with_note(exchanges[0][0], seed["time"], reading=T.reading_lines(exchanges[0][0], seed["time"], rng), reminders=T.pick_reminders(rng))
    messages = T.opening(seed) + [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": first}]},
                                  {"role": "assistant", "content": exchanges[0][1]}]
    for ask, answer in exchanges[1:]:
        messages += [{"role": "user", "content": ask}, {"role": "assistant", "content": answer}]
    return {"id": f"{kind}-{name}", "messages": messages, "split": split, "skill": "see", "hard": False, "learn_route": False,
            "turns": [{"at": at, "skill": "see", "hard": False} for at, m in enumerate(messages) if m["role"] == "user"],
            "notools": kind == "caption", "image": number, "kind": "picture"}


def picture_data(args, tokenizer, work: Path, kind: str, limit: int):
    """(train rows, held rows, features) of the pictures: kind caption (COCO captions, short prompt) or llava (LLaVA-Instruct
    conversations, the full prompt). The same HELD_PICTURES pictures are held out of both."""
    import torch

    loaded = torch.load(work / "clip-features.pt", mmap=True)
    names, features = loaded["names"], loaded["features"]
    number = {name: i for i, name in enumerate(names)}
    talks = {}
    for item in json.loads((VIS / "llava_instruct_150k.json").read_text()):
        if item["image"] in number and item["image"] not in talks:
            turns = [part["value"].replace("<image>", "").strip() for part in item["conversations"]]
            talks[item["image"]] = list(zip(turns[0::2], turns[1::2]))[:4]
    held_names = sorted(talks, key=lambda n: hashlib.sha256(n.encode()).hexdigest())[:min(HELD_PICTURES, len(talks) // 4)]
    held_set = set(held_names)
    catalog = T.load_catalog(Path(args.tools))
    if kind == "caption":
        found = {}
        for note in json.loads((VIS / "annotations" / "captions_train2017.json").read_text())["annotations"]:
            found.setdefault(f"{note['image_id']:012d}.jpg", []).append(note["caption"].strip())

        def exchanges(name):
            rng = random.Random(f"ask-{name}")
            return [(rng.choice(SEE_ASKS), rng.choice(found[name]))]
        pool = [n for n in names if n in found]
        held_pool = [n for n in held_names if n in found][:100]
    else:
        def exchanges(name):
            return talks[name]
        pool, held_pool = [n for n in names if n in talks], held_names
    train_names = [n for n in pool if n not in held_set]
    random.Random(SEED + 31).shuffle(train_names)
    rows = {"train": [], "held": []}
    for split, chosen in (("train", train_names[:limit]), ("held", held_pool)):
        for name in chosen:
            row = encode_record(tokenizer, picture_record(kind, number[name], name, exchanges(name), split), catalog)
            if len(row["ids"]) <= MAX_TOKENS:
                rows[split].append(row)
    log(f"vision data ({kind}): {len(rows['train'])} train and {len(rows['held'])} held conversations over {len(names)} pictures "
        f"({len(held_set)} held out); tokens median {sorted(len(r['ids']) for r in rows['train'])[len(rows['train']) // 2]}")
    return rows["train"], rows["held"], features


def picture_views(rows: list[dict], regime: str, rng: random.Random, back: int | None = None, unlayered: float = 0.0) -> list[dict]:
    views = []
    for row in rows:
        view = adapter_route(row, regime, rng, back, unlayered)
        if view:
            view["media"], view["image"] = row["media"][:len(view["ids"])], row["image"]
            views.append(view)
    return views


def picture_batch(model, projector, views: list[dict], indexes: list[int], features, pad: int = PAD_ID) -> dict:
    """collate, plus the input vectors: the trunk's embeddings, and at every picture position the projector's output."""
    batch = collate(views, indexes, pad, "cuda")
    embeds = model.model.embedding(batch["ids"]).float()
    for place, index in enumerate(indexes):
        mask = views[index]["media"]
        embeds[place, :len(mask)][mask.cuda()] = projector(features[views[index]["image"]].cuda().float()).reshape(-1, embeds.shape[-1])
    batch["embeds"] = embeds
    return batch


def picture_held(model, modules, projector, views: list[dict], features, budget: int) -> float:
    import torch

    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for indexes in T.batches(views, budget, None):
            batch = picture_batch(model, projector, views, indexes, features)
            if modules:
                set_route(modules, batch["route"])
            losses, _ = token_losses(model, batch, 0, checkpoint=False, reduce=False)
            total, count = total + float(losses.double().sum()), count + len(losses)
    model.train()
    return total / max(1, count)


def train_pictures(args, model, modules, projector, views, held, features, minutes: float, rate: float, name: str, see: int | None):
    """Projector (and, with see, that layer's A and B) trained on views against the frozen trunk; keeps the best state by held loss.
    The loss reaches the projector through every block, so the blocks keep their activations' gradients (recomputed, to save memory)."""
    import torch

    rng = random.Random(SEED + 41)
    plan = T.batches(views, args.picture_budget, rng)
    if args.smoke:
        plan = plan[:3]
    layer = [m for m in modules.values()] if see is not None else []
    groups = [{"params": list(projector.parameters()), "rate": 1.0}]
    if layer:
        groups += [{"params": [m.a for m in layer], "rate": 1.0}, {"params": [m.b for m in layer], "rate": B_RATE}]
    optimizer = torch.optim.AdamW(groups, lr=rate, betas=(0.9, 0.95), weight_decay=0.0)
    every = max(1, len(plan) // max(1, args.checks))
    first = picture_held(model, modules if layer else None, projector, held, features, args.picture_budget)
    log(f"{name}: held loss before {first:.4f}; {len(views)} sequences, {len(plan)} steps planned for {minutes} minutes")

    def keep():
        return ({k: v.detach().cpu().clone() for k, v in projector.state_dict().items()},
                [(m.a[see].detach().cpu().clone(), m.b[see].detach().cpu().clone()) for m in layer] if layer else None)

    best, step, mark, began = (first, keep()), 0, 0.0, time.time()
    while step < len(plan):
        for group in optimizer.param_groups:
            group["lr"] = schedule(step, len(plan), rate) * group["rate"]
        batch = picture_batch(model, projector, views, plan[step], features)
        if layer:
            set_route(modules, batch["route"])
        loss = token_losses(model, batch, 0, checkpoint=True)
        loss.backward()
        for m in layer:  # only the see layer learns here: route and the skills stay as they were
            m.a.grad[:see], m.b.grad[:see] = 0, 0
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == len(plan) - 1:
            log(f"{name}: step {step}/{len(plan)} loss {float(loss.detach()):.4f} after {time.time() - began:.0f} s")
        if (step + 1) % every == 0 or step == len(plan) - 1:
            now = picture_held(model, modules if layer else None, projector, held, features, args.picture_budget)
            log(f"{name}: held loss {now:.4f} after step {step + 1}")
            if now < best[0]:
                best = (now, keep())
        plan, mark = T.fit_plan(plan, step, mark, minutes * 60, name)
        step += 1
    state, kept = best[1]
    projector.load_state_dict(state)
    with torch.no_grad():
        for m, (a, b) in zip(layer, kept or []):
            m.a.data[see], m.b.data[see] = a.cuda(), b.cuda()
    log(f"{name}: kept the best state; held loss {first:.4f} before, {best[0]:.4f} after ({step} steps, {time.time() - began:.0f} s)")
    return {"held_before": first, "held_after": best[0], "steps": step, "seconds": time.time() - began, "sequences": len(views), "held": len(held)}


def vision_setup(args, with_adapters: bool):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out)
    work = out / ("vision-smoke" if args.smoke else "vision")
    tokenizer, model = load_model()
    load_deq(model, out / ("gguf-smoke" if args.smoke else "gguf") / "deq.safetensors")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = attach_adapters(model) if with_adapters else None
    projector = make_projector(model.model.embedding.weight.shape[1]).cuda()
    return out, work, tokenizer, model, modules, projector


def stage_vision_a(args) -> None:
    """The projector alone, on COCO captions, against the frozen IQ2_M trunk."""
    out, work, tokenizer, model, modules, projector = vision_setup(args, False)
    rows, held, features = picture_data(args, tokenizer, work, "caption", 64 if args.smoke else args.pictures)
    rng = random.Random(SEED + 43)
    result = train_pictures(args, model, modules, projector, picture_views(rows, "plain", rng), picture_views(held, "plain", rng), features,
                            args.vision_a_minutes, args.projector_lr, "vision-a", None)
    import torch

    torch.save(projector.state_dict(), work / "projector.pt")
    (work / "vision-a.json").write_text(json.dumps(result, indent=1))
    log(f"vision-a: projector saved to {work / 'projector.pt'}")


def export_projector(projector, pt: Path) -> None:
    import torch

    state = projector.state_dict()
    torch.save({f"mm.{i}.{k}": state[f"{i}.{k}"].detach().float().cpu() for i in (0, 2) for k in ("weight", "bias")}, pt)


def stage_vision_b(args) -> None:
    """The see layer and the projector together on LLaVA-Instruct; writes see.pt and vision.gguf (an mmproj llama-server loads)."""
    import torch

    out, work, tokenizer, model, modules, projector = vision_setup(args, True)
    load_adapters(modules, torch.load(out / ("adapters-smoke.pt" if args.smoke else "adapters.pt")))
    projector.load_state_dict(torch.load(work / "projector.pt"))
    see = ADAPTERS.index("see")
    rows, held, features = picture_data(args, tokenizer, work, "llava", 24 if args.smoke else args.pictures // 2)
    rng = random.Random(SEED + 47)
    views = picture_views(rows, "warm", rng, unlayered=NEWEST_UNLAYERED)
    views += picture_views(rows, "cold", rng, unlayered=NEWEST_UNLAYERED)
    random.Random(SEED + 49).shuffle(views)
    result = train_pictures(args, model, modules, projector, views, picture_views(held, "warm", rng, back=ALORA_BACK), features,
                            args.vision_b_minutes, args.adapter_lr, "vision-b", see)
    torch.save({"see": adapter_state(modules, see), "projector": projector.state_dict()}, work / "see.pt")
    pt, folder = work / "projector-llama.pt", work / "mmproj"
    export_projector(projector, pt)
    os.environ["PYTHONPATH"] = f"{LLAMA / 'gguf-py'}:{os.environ.get('PYTHONPATH', '')}"  # the converter imports gguf
    run_tool([sys.executable, LLAMA / "tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py", "-m", VIS / "clip",
              "--llava-projector", pt, "-o", folder], work / "mmproj.log")
    shutil.move(folder / "mmproj-model-f16.gguf", work / "vision.gguf")
    result["vision_bytes"] = (work / "vision.gguf").stat().st_size
    (work / "vision-b.json").write_text(json.dumps(result, indent=1))
    log(f"vision-b: see.pt and vision.gguf ({result['vision_bytes'] / 1e6:.0f} MB) written to {work}")


def picture_parity(model, modules, projector, tokenizer, out: Path, tools: str, count: int = 4) -> list[dict]:
    """Held LLaVA conversations as parity cases: the prompt string with the picture marker, the picture as base64, and the token
    torch names after </think> under the see layer (the layer starts ALORA_BACK tokens earlier, as llama.cpp does on a prompt it has no cache for)."""
    import base64

    import torch
    import torch.nn.functional as F

    loaded = torch.load(out / "vision" / "clip-features.pt", mmap=True)
    features, number = loaded["features"], {n: i for i, n in enumerate(loaded["names"])}
    talks = {}
    for item in json.loads((VIS / "llava_instruct_150k.json").read_text()):
        if item["image"] in number and item["image"] not in talks:
            turns = [part["value"].replace("<image>", "").strip() for part in item["conversations"]]
            talks[item["image"]] = list(zip(turns[0::2], turns[1::2]))[:4]
    held = sorted(talks, key=lambda n: hashlib.sha256(n.encode()).hexdigest())[:count]
    catalog = T.load_catalog(Path(tools))
    made, see = [], ADAPTERS.index("see")
    model.eval()
    for name in held:
        record = picture_record("llava", number[name], name, talks[name], "held")
        row = encode_record(tokenizer, record, catalog)
        cut = row["end_at"][0] + 1
        pieces = R.segments(record["messages"], T.shown_tools(record, catalog), routes=row["routes"], add_generation_prompt=False)
        prompt = ""
        for kind, body in pieces:
            prompt += body
            if kind == "route":
                break
        prompt += "</think>"
        route = torch.full((1, cut), -1, dtype=torch.long)
        route[0, cut - 1 - ALORA_BACK:] = see
        view = {"ids": row["ids"][:cut], "media": row["media"][:cut], "image": row["image"], "route": route[0],
                "weight": torch.zeros(cut), "kind": row["kind"][:cut], "tturn": row["tturn"][:cut].long(), "tskill": row["tskill"]}
        batch = picture_batch(model, projector, [view], [0], features)
        set_route(modules, route.cuda())
        with torch.no_grad():
            hidden = run_blocks(model, batch["ids"], 0, False, batch["embeds"])
            top = F.linear(hidden[0, -1], model.model.embedding.weight).float().topk(2)
        made.append({"id": row["id"], "turn": 0, "layer": "see", "multi_turn": False, "prompt_string": prompt,
                     "images": [base64.b64encode((VIS / "images" / name).read_bytes()).decode()],
                     "next": int(top.indices[0]), "next_text": tokenizer.decode([int(top.indices[0])]), "margin": float(top.values[0] - top.values[1])})
    return made


# STAGES-END


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["judge", "trunk", "gguf", "vision-features", "vision-a", "adapters", "vision-b", "eval", "export", "parity"])
    parser.add_argument("--data", default=str(WORK / "data" / "conv.jsonl"))
    parser.add_argument("--out", default=str(WORK / "out3"))
    parser.add_argument("--tools", default=str(WORK / "tools_full.json"))
    parser.add_argument("--asks", default=str(WORK / "asks.json"))
    parser.add_argument("--smoke", action="store_true", help="a tiny slice of the stage, to find mistakes early")
    parser.add_argument("--fresh", action="store_true", help="encode the conversations again")
    parser.add_argument("--gpu", type=float, default=0.85)
    parser.add_argument("--judge-n", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--all-rows", action="store_true", help="trunk: train on the held-out sixteenth too")
    parser.add_argument("--checks", type=int, default=6, help="trunk: look at the held-out loss this many times and keep the best weights")
    parser.add_argument("--trunk-minutes", type=float, default=0, help="cut the trunk plan to this time; 0 = no cut")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--adapter-lr", type=float, default=2e-4, help="adapters: A matrices; B matrices get B_RATE times as much")
    parser.add_argument("--adapter-minutes", type=float, default=20, help="adapters: cut the plan to this time")
    parser.add_argument("--imatrix-chunks", type=int, default=400)
    parser.add_argument("--cpu", action="store_true", help="gguf: run llama-imatrix on the CPU")
    parser.add_argument("--eval-cases", type=int, default=120, help="eval: held-out tool-call turns to generate")
    parser.add_argument("--pack", default="", help="export, parity: the pack folder (default: <out>/pack)")
    parser.add_argument("--no-vision", action="store_true", help="export: leave the vision file and the see layer out")
    parser.add_argument("--port", type=int, default=8795)
    parser.add_argument("--pictures", type=int, default=8000, help="vision-a: caption conversations to encode (vision-b: half as many LLaVA ones)")
    parser.add_argument("--vision-a-minutes", type=float, default=8)
    parser.add_argument("--vision-b-minutes", type=float, default=12)
    parser.add_argument("--projector-lr", type=float, default=1e-3, help="vision-a: the projector alone; vision-b uses --adapter-lr")
    parser.add_argument("--picture-budget", type=int, default=16000, help="tokens per batch of the vision stages (gradient runs through every block)")
    parser.add_argument("--budget", type=int, default=24000, help="tokens per batch")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    stages = {name[6:].replace("_", "-"): function for name, function in globals().items() if name.startswith("stage_")}
    stages[args.stage](args)


if __name__ == "__main__":
    main()
