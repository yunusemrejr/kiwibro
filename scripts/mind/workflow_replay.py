#!/usr/bin/env python3
"""Bind recovery fitting and paired holdout replay to a specific successor."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import tempfile

from kiwibro import specialists
from scripts.mind import agent_corpus, calibration_traces as traces
from scripts.mind.train_workflows import digest, scoped

LABELS = {
    "missing_path": "location", "stale_browser": "render", "blocked": "boundary",
    "dependency": "dependency", "untrusted": "none", "success": "none",
}
HINTS = {
    "location": "Use an available read-only listing or search to locate the intended item before retrying the read.",
    "render": "Refresh the visible page or inspect the rendered state before relying on an old locator or partial DOM.",
    "boundary": "Respect the denied action. Use a permitted read-only fallback if it can answer the request, "
                "or state the blocker.",
    "dependency": "Verify which tools or interpreters are available. Pick a supported alternative within the request.",
}


def observation(message: dict) -> str:
    # Match the input used for the authored recovery data.
    return f"{message['name']}: {message['content'][:1050]}"


def frozen_probes(path: Path, split: str) -> list[dict]:
    from kiwibro.tools import tool_defs

    rows = agent_corpus.read_jsonl(path)
    catalog = {tool["function"]["name"]: tool for tool in tool_defs()}
    expected = [traces.probe(kind, split, name, index, catalog)
                for kind in traces.LABELS for index, name in enumerate(traces.NAMES[split])]
    if rows != expected:
        raise ValueError(f"{split}: expected the complete, unchanged frozen recovery probes")
    return rows


def recover_label(record: dict) -> str:
    """A complete miss with a recoverable observation is a candidate for advice."""
    return "none" if record["passed"] else LABELS[record["oracle"]["kind"]]


def report_rows(probes: Path, replay: Path) -> tuple[dict, list[dict]]:
    with replay.open(encoding="utf-8") as stream:
        header = json.loads(stream.readline())
    binding = header.get("binding") or {}
    artifact = binding.get("artifact_sha256", "")
    if (
        set(binding) != {"artifact_sha256", "runtime_id", "sampling_seed", "temperature", "cache_prompt",
                         "thinking", "max_tokens", "timeout"}
        or not re.fullmatch(r"[a-f0-9]{64}", artifact)
        or not binding["runtime_id"]
        or binding["thinking"] != header["thinking"]
        or header["per_family_and_action_kind"] != 0
    ):
        raise ValueError(f"{replay.name}: expected a complete artifact/runtime-bound replay")
    summary, rows = traces.join(probes, replay, artifact)
    if len(rows) != len(agent_corpus.read_jsonl(probes)):
        raise ValueError(f"{replay.name}: replay is missing probe IDs")
    return summary, rows


def model_profile(summary: dict) -> dict:
    return {"name": summary["model"], **summary["binding"]}


def prepare_observations(probes: Path, replays: Path, authored: dict) -> dict:
    """Recheck words, outcomes, and split membership before exporting fit inputs."""
    known = {
        agent_corpus.normalize(text): f"authored/{split}"
        for split, rows in authored.items() for text, _ in rows
    }
    splits = {}
    provenance = {"folds": {}}
    for split in traces.NAMES:
        path = probes / f"{split}.jsonl"
        cases = frozen_probes(path, split)
        observations = {case["id"]: observation(case["messages"][-1]) for case in cases}
        for text in observations.values():
            key = agent_corpus.normalize(text)
            previous = known.setdefault(key, split)
            if previous != split:
                raise ValueError(f"Duplicate recovery input across {previous} and {split}")
        if split == "test":
            provenance["test_probes_sha256"] = digest(path.read_bytes())
            continue
        summary, rows = report_rows(path, replays / path.name)
        if provenance.setdefault("model", model_profile(summary)) != model_profile(summary):
            raise ValueError("Recovery splits use different artifacts, runtimes or sampling settings")
        data = [(observations[row["id"]], recover_label(row), row["oracle"]["observation_id"])
                for row in rows if row["outcome"] != "incomplete_generation"]
        if all(label == "none" for _, label, _ in data):
            raise ValueError(f"{split}: no completed successor misses with a recovery target")
        provenance["folds"][split] = {
            key: summary[key] for key in ("probes_sha256", "report_sha256", "count", "outcomes")
        }
        provenance["folds"][split]["targets"] = dict(Counter(label for _, label, _ in data))
        splits[split] = data
    return {"provenance": provenance, "splits": splits}


def hint_probes(pack: Path, probes: Path, destination: Path) -> dict:
    raw = pack.read_bytes()
    payload = json.loads(gzip.decompress(raw))
    calibration = payload["report"]["successor_recovery"]
    if payload["origin"] != "google-colab" or calibration["test_probes_sha256"] != digest(probes.read_bytes()):
        raise ValueError("Expected a Colab candidate tied to this frozen test set")
    head = payload["heads"]["recovery"]
    rows = frozen_probes(probes, "test")
    selected = {}
    for row in rows:
        text = observation(row["messages"][-1])
        label = scoped("recovery", specialists.classify(head, specialists.features(text)), text)
        if label in HINTS:
            row["messages"][0]["content"] += f"\n\nKiwiBro recovery guidance: {HINTS[label]}"
            selected[row["id"]] = label
    agent_corpus.jsonl(destination, rows)
    return {
        "model": calibration["model"], "pack_sha256": digest(raw),
        "test_probes_sha256": calibration["test_probes_sha256"],
        "hinted_probes_sha256": digest(destination.read_bytes()), "selected": selected,
    }


def compare(pack: Path, probes: Path, baseline: Path, hinted: Path, assisted: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="kiwi-recovery-pair-") as temp:
        plan = hint_probes(pack, probes, Path(temp) / "hinted.jsonl")
    sides = [report_rows(path, replay) for path, replay in ((probes, baseline), (hinted, assisted))]
    case_lists = []
    for (summary, rows), key in zip(sides, ("test_probes_sha256", "hinted_probes_sha256")):
        if summary["split"] != "test" or summary["probes_sha256"] != plan[key]:
            raise ValueError("Paired replay does not match the frozen test or candidate hint plan")
        if model_profile(summary) != plan["model"]:
            raise ValueError("Paired replay does not match the candidate's model and runtime profile")
        case_lists.append({row["id"]: row for row in rows})
    unassisted, with_hints = case_lists
    if unassisted.keys() != with_hints.keys() or not plan["selected"].keys() <= unassisted.keys():
        raise ValueError("Expected identical cases in both replay arms")

    outcomes = Counter()
    per_kind: dict[str, Counter] = {}
    changes = []
    for identifier, before in unassisted.items():
        after = with_hints[identifier]
        if before["oracle"] != after["oracle"] or before["expected"] != after["expected"]:
            raise ValueError(f"{identifier}: oracle changed between replay arms")
        advice = plan["selected"].get(identifier, "none")
        kind = before["oracle"]["kind"]
        metrics = Counter({"cases": 1, "hints": int(advice != "none")})
        if "incomplete_generation" in (before["outcome"], after["outcome"]):
            metrics["incomplete_pair"] = 1
        else:
            wanted = recover_label(before)
            metrics["baseline_passed"] = int(before["passed"])
            metrics["assisted_passed"] = int(after["passed"])
            metrics["correct_hint_for_miss"] = int(advice == wanted != "none")
            # A hint may be safe and nevertheless redundant or off target.
            metrics["hint_on_control"] = int(LABELS[kind] == "none" and advice != "none")
            metrics["hint_on_solved_failure"] = int(LABELS[kind] != "none" and before["passed"] and advice != "none")
            metrics["wrong_hint_for_miss"] = int(wanted != "none" and advice not in ("none", wanted))
            metrics["miss_without_hint"] = int(wanted != "none" and advice == "none")
            if before["passed"] != after["passed"]:
                change = "gain" if after["passed"] else "regression"
                metrics[change if advice != "none" else f"no_hint_{change}"] = 1
                changes.append({"id": identifier, "result": change, "hint": advice, "kind": kind})
        outcomes.update(metrics)
        per_kind.setdefault(kind, Counter()).update(metrics)
    return {
        "model": plan["model"], "pack_sha256": plan["pack_sha256"],
        "replay_sha256": {"baseline": sides[0][0]["report_sha256"], "assisted": sides[1][0]["report_sha256"]},
        "outcomes": dict(outcomes), "per_kind": per_kind, "changes": changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    hint = commands.add_parser("hint")
    hint.add_argument("pack", type=Path)
    hint.add_argument("probes", type=Path)
    hint.add_argument("output", type=Path)
    paired = commands.add_parser("compare")
    paired.add_argument("pack", type=Path)
    paired.add_argument("probes", type=Path)
    paired.add_argument("baseline", type=Path)
    paired.add_argument("hinted", type=Path)
    paired.add_argument("assisted", type=Path)
    args = parser.parse_args()
    if args.action == "hint":
        result = hint_probes(args.pack, args.probes, args.output)
    else:
        result = compare(args.pack, args.probes, args.baseline, args.hinted, args.assisted)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
