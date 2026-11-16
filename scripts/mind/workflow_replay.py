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
    names = traces.BLIND_NAMES if split == "blind" else traces.NAMES[split]
    expected = [traces.probe(kind, split, name, index, catalog)
                for kind in traces.LABELS for index, name in enumerate(names)]
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


def object_digest(payload: dict) -> str:
    return digest(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def qualified_hints(per_hint: dict) -> list[str]:
    return sorted(label for label in HINTS if (
        (stats := per_hint.get(label, {})).get("gain", 0) > 0
        and not any(stats.get(metric, 0) for metric in ("regression", "incomplete_pair", "hint_on_control"))
    ))


def hint_probes(pack: Path, probes: Path, destination: Path, *, split: str = "test",
                policy: dict | None = None) -> dict:
    raw = pack.read_bytes()
    payload = json.loads(gzip.decompress(raw))
    calibration = payload["report"]["successor_recovery"]
    if payload["origin"] != "google-colab":
        raise ValueError("Expected a Colab recovery candidate")
    pack_sha = digest(raw)
    enabled = set(HINTS)
    if policy is not None:
        if split != "test":
            raise ValueError("A frozen calibration policy selects the fresh blind split")
        split = "blind"
        fold = calibration["folds"]["calibrate"]
        if (
            policy["version"] != 1 or policy["model"] != calibration["model"] or policy["pack_sha256"] != pack_sha
            or policy["hint_sha256"] != object_digest(HINTS)
            or policy["calibration_probes_sha256"] != fold["probes_sha256"]
            or policy["calibration_replay_sha256"]["baseline"] != fold["report_sha256"]
            or policy["labels"] != qualified_hints(policy["per_hint"])
        ):
            raise ValueError("Frozen calibration policy does not match the candidate or qualified labels")
        enabled = set(policy["labels"])
        expected_probe_sha = policy["blind_probes_sha256"]
    elif split == "calibrate":
        expected_probe_sha = calibration["folds"]["calibrate"]["probes_sha256"]
    elif split == "test":
        expected_probe_sha = calibration["test_probes_sha256"]
    else:
        raise ValueError("Fresh blind probes require a frozen calibration policy")
    if expected_probe_sha != digest(probes.read_bytes()):
        raise ValueError(f"Expected a Colab candidate tied to these frozen {split} probes")
    head = payload["heads"]["recovery"]
    rows = frozen_probes(probes, split)
    selected = {}
    for row in rows:
        text = observation(row["messages"][-1])
        label = scoped("recovery", specialists.classify(head, specialists.features(text)), text)
        if label in enabled:
            row["messages"][0]["content"] += f"\n\nKiwiBro recovery guidance: {HINTS[label]}"
            selected[row["id"]] = label
    agent_corpus.jsonl(destination, rows)
    source = ({"test_probes_sha256": expected_probe_sha} if split == "test"
              else {"split": split, "probes_sha256": expected_probe_sha})
    return {
        "model": calibration["model"], "pack_sha256": pack_sha,
        **source,
        "hinted_probes_sha256": digest(destination.read_bytes()), "selected": selected,
    }


def compare(pack: Path, probes: Path, baseline: Path, hinted: Path, assisted: Path, *,
            split: str = "test", policy: dict | None = None) -> dict:
    with tempfile.TemporaryDirectory(prefix="kiwi-recovery-pair-") as temp:
        plan = hint_probes(pack, probes, Path(temp) / "hinted.jsonl", split=split, policy=policy)
    split = plan.get("split", "test")
    probe_sha = plan.get("probes_sha256", plan.get("test_probes_sha256"))
    sides = [report_rows(path, replay) for path, replay in ((probes, baseline), (hinted, assisted))]
    case_lists = []
    for (summary, rows), sha in zip(sides, (probe_sha, plan["hinted_probes_sha256"]), strict=True):
        if summary["split"] != split or summary["probes_sha256"] != sha:
            raise ValueError(f"Paired replay does not match the frozen {split} probes or candidate hint plan")
        if model_profile(summary) != plan["model"]:
            raise ValueError("Paired replay does not match the candidate's model and runtime profile")
        case_lists.append({row["id"]: row for row in rows})
    unassisted, with_hints = case_lists
    if unassisted.keys() != with_hints.keys() or not plan["selected"].keys() <= unassisted.keys():
        raise ValueError("Expected identical cases in both replay arms")

    outcomes = Counter()
    per_kind: dict[str, Counter] = {}
    per_hint: dict[str, Counter] = {}
    changes = []
    for identifier, before in unassisted.items():
        after = with_hints[identifier]
        if before["oracle"] != after["oracle"] or before["expected"] != after["expected"]:
            raise ValueError(f"{identifier}: oracle changed between replay arms")
        advice = plan["selected"].get(identifier, "none")
        kind = before["oracle"]["kind"]
        metrics = Counter({"cases": 1, "hints": int(advice != "none")})
        if advice != "none" and LABELS[kind] == "none":
            # A hint on a control remains relevant even if generation fails.
            per_hint.setdefault(advice, Counter())["hint_on_control"] += 1
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
        if advice != "none":
            per_hint.setdefault(advice, Counter()).update({
                key: metrics[key] for key in ("hints", "gain", "regression", "incomplete_pair", "hint_on_solved_failure")
            })
    report = {
        "model": plan["model"], "pack_sha256": plan["pack_sha256"],
        "replay_sha256": {"baseline": sides[0][0]["report_sha256"], "assisted": sides[1][0]["report_sha256"]},
        "outcomes": dict(outcomes), "per_kind": per_kind, "changes": changes,
    }
    if split != "test":
        report.update(split=split, probes_sha256=probe_sha, per_hint=per_hint)
    if policy is not None:
        report.update(policy_sha256=object_digest(policy), frozen_labels=policy["labels"])
    return report


def freeze(pack: Path, probes: Path, baseline: Path, hinted: Path, assisted: Path, blind: Path) -> dict:
    """Choose hint labels on calibration only, before running the fresh blind pair."""
    report = compare(pack, probes, baseline, hinted, assisted, split="calibrate")
    payload = json.loads(gzip.decompress(pack.read_bytes()))
    if report["replay_sha256"]["baseline"] != payload["report"]["successor_recovery"]["folds"]["calibrate"]["report_sha256"]:
        raise ValueError("Use the exact calibration baseline recorded by the Colab fit")
    frozen_probes(blind, "blind")
    return {
        "version": 1, "model": report["model"], "pack_sha256": report["pack_sha256"],
        "hint_sha256": object_digest(HINTS), "calibration_probes_sha256": report["probes_sha256"],
        "calibration_replay_sha256": report["replay_sha256"], "blind_probes_sha256": digest(blind.read_bytes()),
        "labels": qualified_hints(report["per_hint"]), "per_hint": report["per_hint"],
        "calibration_outcomes": report["outcomes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    hint = commands.add_parser("hint")
    hint.add_argument("pack", type=Path)
    hint.add_argument("probes", type=Path)
    hint.add_argument("output", type=Path)
    paired = commands.add_parser("compare")
    choose = commands.add_parser("freeze")
    for command in (paired, choose):
        for name in ("pack", "probes", "baseline", "hinted", "assisted"):
            command.add_argument(name, type=Path)
    choose.add_argument("blind", type=Path)
    for command in (hint, paired):
        mode = command.add_mutually_exclusive_group()
        mode.add_argument("--split", choices=("test", "calibrate"), default="test")
        mode.add_argument("--policy", type=Path, help="Frozen calibration policy for the fresh blind probes.")
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze(args.pack, args.probes, args.baseline, args.hinted, args.assisted, args.blind)
    else:
        policy = json.loads(args.policy.read_text(encoding="utf-8")) if args.policy is not None else None
        if args.action == "hint":
            result = hint_probes(args.pack, args.probes, args.output, split=args.split, policy=policy)
        else:
            result = compare(args.pack, args.probes, args.baseline, args.hinted, args.assisted,
                             split=args.split, policy=policy)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
