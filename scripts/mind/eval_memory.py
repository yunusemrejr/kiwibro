"""Evaluate Kiwi Mind's durable-memory helpers on independent synthetic chats.

From the checkout root, run:
    python3 -m scripts.mind.eval_memory \
      --model stock examples /path/to/Qwen3-0.6B-Q8_0.gguf \
      --model trained short /path/to/kiwibro-memory-expert.gguf \
      --output /tmp/kiwibro-memory-eval.json

Requires the app's usual Python dependencies and llama-server, selected with its
normal discovery or --server. All memories and server logs are temporary. Use
the same prompt mode and quantization when isolating the effect of fine-tuning.

The corpus is frozen synthetic English. Raw scoring matches each expected
fact to one model line using labeled relation and value patterns, and scores
the kind separately. Persisted scoring runs the real app's parser and memory
write checks in a disposable database. The eligible subset applies the app's
existing preflight filter. Raw lines and memory changes stay in the report
for review of paraphrases and of unsupported details within a matching line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from kiwibro import config, engine, memory
from kiwibro.db import Database
from kiwibro.llm import LLMClient
from kiwibro.mind.little import CTX, THREADS, Little, LittleError
from kiwibro.mind.llama import LlamaRuntime

CORPUS = Path(__file__).with_name("memory_eval.json")
LINE = re.compile(r"(" + "|".join(config.MEMORY_KINDS) + r")\s*\|\s*(User\b.+)")
WARMUP = [
    {"role": "user", "content": "Read this harmless sentence and answer with the word none."},
]


def digest(path):
    with path.open("rb") as data:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: data.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def parse_output(raw):
    if raw == "none":
        return [], True
    lines = [LINE.fullmatch(line.strip()) for line in raw.splitlines() if line.strip()]
    return [line.groups() for line in lines if line], bool(lines) and all(lines)


def score(predictions, expected):
    """Match at most one prediction to each expected atom, preferring a correct kind."""
    owners = {}
    matches = [
        sorted(
            (index for index, (_, _, terms) in enumerate(expected)
             if all(re.search(term, text, flags=re.IGNORECASE) for term in terms)),
            key=lambda index: kind != expected[index][0],
        )
        for kind, text in predictions
    ]

    def assign(prediction, seen):
        for index in matches[prediction]:
            if index not in seen:
                seen.add(index)
                if index not in owners or assign(owners[index], seen):
                    owners[index] = prediction
                    return

    for prediction in range(len(predictions)):
        assign(prediction, set())
    owned = set(owners.values())
    return {
        "correct": len(owners),
        "kind_correct": sum(predictions[prediction][0] == expected[index][0]
                            for index, prediction in owners.items()),
        "missing": [atom[1] for index, atom in enumerate(expected) if index not in owners],
        "unexpected": [line for index, line in enumerate(predictions) if index not in owned],
    }


def run_case(helper, manager, db, case, mode):
    db.delete_all_memories()
    known = [
        db.create_memory(kind, content, memory.normalize(content), pinned=True, origin="you")
        for kind, content in case.get("known", [])
    ]
    user_messages = [
        {"role": "user", "content": content, "seq": index + 1, "created_at": config.now_iso()}
        for index, content in enumerate(case["user"])
    ]
    parts = manager._extraction_parts("", user_messages)
    messages = manager._little_extraction_messages(parts)
    if mode == "short":
        messages = [messages[0], messages[-1]]
    if {item["id"] for item in known} != {item["id"] for item in parts[1]}:
        raise ValueError(f"{case['id']}: expected known facts missing from the prompt")

    row = {
        "id": case["id"], "category": case["category"],
        "expected": [atom[1] for atom in case["expected"]],
        "eligible": memory.worth_extracting(user_messages),
    }
    started = time.perf_counter()
    try:
        completion = helper.complete(messages, max_tokens=memory.EXTRACT_MAX_TOKENS, temperature=0.0)
    except LittleError as exc:
        missing = score([], case["expected"])
        return {**row, "error": str(exc), "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "raw": missing, "persisted": missing}
    latency = (time.perf_counter() - started) * 1000
    text = completion.text
    raw, valid = parse_output(text)

    source = "\n".join(parts[0])
    ops = memory.parse_ops(memory.lines_to_ops(text, source))[:memory.MAX_OPS_PER_TURN]
    counts = manager._apply_ops(
        ops, "\n".join(case["user"]), window=user_messages,
    )
    baseline = {item["id"]: (item["kind"], item["content"]) for item in known}
    persisted = [
        (item["kind"], item["content"]) for item in db.all_memories()
        if baseline.get(item["id"]) != (item["kind"], item["content"])
    ]

    return {
        **row, "latency_ms": round(latency, 2),
        "finish_reason": completion.finish_reason, "usage": completion.usage, "valid": valid,
        "text": text,
        "raw": score(raw, case["expected"]),
        "persisted": score(persisted, case["expected"]),
        "saved": persisted, "writes": counts,
    }


def summarize(rows, stage):
    measured = [row for row in rows if stage != "eligible" or row["eligible"]]
    grading = "persisted" if stage == "eligible" else stage
    scores = [row[grading] for row in measured]
    correct = sum(result["correct"] for result in scores)
    predicted = correct + sum(len(result["unexpected"]) for result in scores)
    expected = sum(len(row["expected"]) for row in measured)
    negative = [row for row in measured if not row["expected"]]
    return {
        "cases": len(measured), "correct": correct, "predicted": predicted, "expected": expected,
        "precision": round(correct / predicted, 4) if predicted else None,
        "recall": round(correct / expected, 4) if expected else None,
        "correct_kind": sum(result["kind_correct"] for result in scores),
        "exact_cases": sum(
            "error" not in row and row["valid"] and row["finish_reason"] == "stop"
            and not row[grading]["missing"] and not row[grading]["unexpected"]
            for row in measured
        ),
        "negative_cases": len(negative),
        "false_accept_cases": sum(bool(row[grading]["unexpected"]) for row in negative),
    }


def evaluate(label, model_path, mode, cases, runtime, folder):
    client = LLMClient(dict, data_dir=folder)
    db = Database(folder / "memory.db")
    db.initialize()
    manager = memory.MemoryEngine(db, client, dict)
    helper = Little(folder, lambda: runtime, model_file=lambda: model_path, model_name=label)
    print(f"{label}: warming {model_path.name}", flush=True)
    try:
        started = time.perf_counter()
        helper.complete(WARMUP, max_tokens=12)
        warmup_ms = (time.perf_counter() - started) * 1000
        rows = []
        for case in cases:
            rows.append(run_case(helper, manager, db, case, mode))
            if len(rows) % 10 == 0:
                print(f"{label}: {len(rows)}/{len(cases)} cases", flush=True)
    except LittleError as exc:
        log = helper.log_path
        raise SystemExit(f"{label}: {exc}\n{log.read_text(errors='replace')[-2000:] if log.exists() else ''}") from exc
    finally:
        helper.stop()

    successful = [row for row in rows if "error" not in row]
    times = sorted(row["latency_ms"] for row in successful)
    categories = set(row["category"] for row in rows)
    report = {
        "model": model_path.name, "model_sha256": digest(model_path), "prompt": mode,
        "total_cases": len(rows), "request_errors": len(rows) - len(successful),
        "invalid_outputs": sum(not row["valid"] for row in successful),
        "finish_reasons": Counter(row["finish_reason"] for row in successful),
        "initial_warmup_ms": round(warmup_ms, 2),
        "median_ms": round(statistics.median(times), 2) if times else None,
        "p95_ms": times[math.ceil(0.95 * len(times)) - 1] if times else None,
        "raw": summarize(rows, "raw"),
        "persisted": summarize(rows, "persisted"),
        "eligible": summarize(rows, "eligible"),
        "by_category": {
            name: {stage: summarize([row for row in rows if row["category"] == name], stage)
                   for stage in ("raw", "persisted")}
            for name in sorted(categories)
        },
        "rows": rows,
    }
    print(json.dumps({key: value for key, value in report.items() if key not in ("rows", "by_category")},
                     ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", metavar=("LABEL", "PROMPT", "GGUF"), nargs=3, action="append", required=True,
                        help="PROMPT is examples or short. Repeat to compare models.")
    parser.add_argument("--server", type=Path, help="Optional path to llama-server.")
    parser.add_argument("--runtime-data", type=Path, default=Path("~/.local/share/kiwibro").expanduser())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.server:
        native = engine.Runtime(str(args.server.expanduser().resolve()), "", "unknown", "custom")
        runtime = LlamaRuntime(native.path, engine._flags_of(native), False, engine._runtime_env(native))
    else:
        runtime = LLMClient(dict, data_dir=args.runtime_data)._mind_runtime()
    if runtime is None:
        parser.error("llama-server was not found; supply --server")

    models = {}
    for label, mode, path in args.model:
        source = Path(path).expanduser().resolve()
        if label in models or mode not in ("examples", "short") or not source.is_file():
            parser.error(f"invalid model specification: {label} {mode} {path}")
        models[label] = (mode, source)
    cases = json.loads(CORPUS.read_text(encoding="utf-8"))
    results = {
        "corpus_sha256": digest(CORPUS), "context": CTX, "cpu_threads": THREADS,
        "python": platform.python_version(), "kernel": platform.release(),
        "llama_server": Path(runtime.path).parent.name + "/" + Path(runtime.path).name,
        "llama_server_sha256": digest(Path(runtime.path)), "models": {},
    }
    for label, (mode, path) in models.items():
        with TemporaryDirectory(prefix="kiwibro-memory-eval-") as tmp:
            results["models"][label] = evaluate(label, path, mode, cases, runtime, Path(tmp))
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
