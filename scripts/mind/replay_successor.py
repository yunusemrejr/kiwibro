#!/usr/bin/env python3
"""Score frozen KiwiBro next-action probes through the deployed native chat endpoint."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import re

from PIL import Image
import requests

from scripts.mind.agent_corpus import score
from scripts.mind.bench_successor import SAMPLING_SEED, Case, complete, image_part, is_loopback


def wire_messages(row: dict, root: Path) -> list[dict]:
    messages = deepcopy(row["messages"])
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") != "image_url":
                    parts.append(part)
                    continue
                file = (root / part["image_url"]["url"]).resolve()
                if not file.is_relative_to(root) or not file.is_file():
                    raise ValueError(f"Image from {row['id']} lies outside the fixture folder.")
                with Image.open(file) as image:
                    parts.append(image_part(image.convert("RGB")))
            message["content"] = parts
        for call in message.get("tool_calls") or []:
            function = call["function"]
            if isinstance(function.get("arguments"), dict):
                function["arguments"] = json.dumps(function["arguments"], ensure_ascii=False)
            call.setdefault("type", "function")
    return messages


def sample(rows: list[dict], per_family: int, seed: int) -> list[dict]:
    if not per_family:
        return rows
    choices = defaultdict(list)
    for row in rows:
        choices[(row["family"], bool(row["expected"]["calls"]))].append(row)
    rng = random.Random(seed)
    selected = [row for key in sorted(choices)
                for row in rng.sample(choices[key], min(per_family, len(choices[key])))]
    rng.shuffle(selected)
    return selected


def run(args: argparse.Namespace) -> dict:
    url = args.endpoint.rstrip("/").removesuffix("/v1")
    if not is_loopback(url):
        raise ValueError("Replay uses a local llama.cpp-compatible HTTP endpoint.")
    raw = args.probes.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Expected a nonempty fixture set with unique probe IDs.")
    cases = sample(rows, args.per_family, args.seed)
    records = []
    totals = defaultdict(Counter)
    model = args.model
    with requests.Session() as client, args.output.open("x", encoding="utf-8") as destination:
        if key := os.environ.get(args.key_env):
            client.headers["Authorization"] = f"Bearer {key}"
        if model == "auto":
            response = client.get(f"{url}/v1/models", timeout=(5, 20))
            response.raise_for_status()
            model = response.json()["data"][0]["id"]
        metadata = {"model": model, "probes_sha256": hashlib.sha256(raw).hexdigest(),
                    "seed": args.seed, "per_family_and_action_kind": args.per_family,
                    "thinking": args.thinking, "count": len(cases)}
        if args.artifact_sha256:
            metadata["binding"] = {
                "artifact_sha256": args.artifact_sha256, "runtime_id": args.runtime_id,
                "sampling_seed": SAMPLING_SEED, "temperature": 0, "cache_prompt": True,
                "thinking": args.thinking, "max_tokens": args.max_tokens, "timeout": args.timeout,
            }
        destination.write(json.dumps(metadata) + "\n")
        for i, row in enumerate(cases, 1):
            case = Case(row["id"], row["family"], wire_messages(row, args.probes.parent.resolve()),
                        expected="", tools=row["tools"])
            result = complete(client, url, model, case, True, args.pid, args)
            actual = {"role": "assistant", "content": result["answer"], "tool_calls": [
                {"function": {"name": name, "arguments": arguments}} for name, arguments in result["calls"]
            ]}
            expected = row["expected"]
            passed = result["error"] is None and result["finish_reason"] != "length" and score(expected, actual)
            counts = totals[row["family"]]
            counts["total"] += 1
            counts["passed"] += passed
            counts["tool_actions" if expected["calls"] else "answers"] += 1
            record = {**result, "passed": passed, "episode": row["episode"], "expected": expected}
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            destination.flush()
            records.append(record)
            print(f"{i}/{len(cases)} {row['id']} {'pass' if passed else 'miss'}", flush=True)
    return {"model": model, "passed": sum(r["passed"] for r in records), "count": len(records),
            "families": {name: dict(counts) for name, counts in sorted(totals.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--key-env", default="KIWIBRO_BENCH_API_KEY")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--per-family", type=int, default=0, help="Limit per workflow and action kind; zero runs all probes.")
    parser.add_argument("--seed", type=int, default=4073)
    parser.add_argument("--thinking", choices=("off", "on", "default"), default="off")
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--artifact-sha256", help="SHA-256 of the model artifact served for a bound replay.")
    parser.add_argument("--runtime-id", help="One stable ID for the server build, template and launch settings.")
    args = parser.parse_args()
    if args.per_family < 0 or min(args.max_tokens, args.timeout) <= 0:
        parser.error("invalid case or request budget")
    if (args.artifact_sha256 or args.runtime_id) and (
        not re.fullmatch(r"[a-f0-9]{64}", args.artifact_sha256 or "") or not (args.runtime_id or "").strip()
    ):
        parser.error("bound replay requires both a lowercase artifact SHA-256 and a runtime ID")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
