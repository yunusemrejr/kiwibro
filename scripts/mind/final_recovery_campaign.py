#!/usr/bin/env python3
"""Stage and replay the recovery head for a selected 300-step Gemma export.

Run with python -m scripts.mind.final_recovery_campaign. Replay subcommands
attach to a running loopback endpoint. Fit-job writes a Colab CPU job file.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import requests

from scripts.mind import calibration_traces, replay_successor, train_workflows, workflow_replay
from scripts.mind.bench_successor import SAMPLING_SEED, is_loopback
from scripts.mind.prune_gemma4 import BASE_REVISION

EXPORTS = {
    "standard": ("kiwi-mind-gemma4-mixed-q4_k_m.gguf", "supplement-checksum.txt", "compact-mixed-24k"),
    "down": ("kiwi-mind-gemma4-mixed-q4-down.gguf", "supplement-down-checksum.txt", "compact-mixed-down-q4-24k"),
}
TRAINING_REPORT = "supplement-merged/training.json"
FIT_SPLITS = ("train", "calibrate")


def read_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def write_json(path: Path, payload: dict) -> None:
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".campaign-") as temp:
        partial = Path(temp) / path.name
        partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.link(partial, path)


def stage(exports: Path, work: Path, variant: str, runtime_id: str, model: str | None) -> dict:
    if work.exists() or work.is_symlink():
        raise FileExistsError(work)
    filename, sidecar, alias = EXPORTS[variant]
    training_bytes = (exports / TRAINING_REPORT).read_bytes()
    training = json.loads(training_bytes)
    if (training["base_revision"] != BASE_REVISION or training["total_steps"] != 300
            or training["mode"] != "l4-lora" or training["reasoning_steps"] < 1):
        raise ValueError("Expected the merged Gemma checkpoint from the 300-step mixed-reasoning campaign")
    checksum = (exports / sidecar).read_bytes()
    record = re.fullmatch(r"([a-f0-9]{64}) [ *]" + re.escape(filename), checksum.decode().strip())
    if not record:
        raise ValueError(f"Expected a single SHA-256 record for {filename} in {sidecar}")
    artifact = (exports / filename).resolve(strict=True)
    before = artifact.stat()
    with artifact.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError(f"Expected GGUF weights in {artifact}")
        stream.seek(0)
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    after = artifact.stat()
    if sha != record[1] or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Export SHA-256 mismatch for {filename}")
    if not runtime_id.strip():
        raise ValueError("Record the serving binary, projector and launch profile in the runtime ID")

    work.mkdir(parents=True)
    probes = calibration_traces.build(work / "probes", fresh_blind=True)
    campaign = {
        "variant": variant, "artifact": str(artifact),
        "training": {**training, "report_sha256": train_workflows.digest(training_bytes)},
        "export_checksum_sha256": train_workflows.digest(checksum),
        "model": {"name": model or alias, "artifact_sha256": sha, "runtime_id": runtime_id,
                  "sampling_seed": SAMPLING_SEED, "temperature": 0, "cache_prompt": True,
                  "thinking": "off", "max_tokens": 320, "timeout": 160.0},
        "probes": probes["splits"],
    }
    write_json(work / "campaign.json", campaign)
    return campaign


def campaign_at(work: Path) -> dict:
    campaign = read_json(work / "campaign.json")
    for split, info in campaign["probes"].items():
        if train_workflows.digest((work / "probes" / f"{split}.jsonl").read_bytes()) != info["sha256"]:
            raise ValueError(f"The campaign's frozen {split} probes changed")
    return campaign


def checked_replay(probes: Path, report: Path, campaign: dict) -> dict:
    summary, _ = workflow_replay.report_rows(probes, report)
    if workflow_replay.model_profile(summary) != campaign["model"]:
        raise ValueError(f"{report.name} was collected on a different artifact or serving profile")
    return summary


def replay(probes: Path, report: Path, campaign: dict, endpoint: str, pid: int | None) -> dict:
    if report.exists():
        return checked_replay(probes, report, campaign)
    url = endpoint.rstrip("/").removesuffix("/v1")
    if not is_loopback(url):
        raise ValueError("Recovery replay attaches to a loopback endpoint")
    with requests.Session() as session:
        session.trust_env = False
        if key := os.environ.get("KIWIBRO_BENCH_API_KEY"):
            session.headers["Authorization"] = f"Bearer {key}"
        listing = session.get(f"{url}/v1/models", timeout=(5, 15))
        listing.raise_for_status()
        if campaign["model"]["name"] not in {entry["id"] for entry in listing.json()["data"]}:
            raise ValueError("The loopback endpoint does not advertise the selected export's model alias")
    profile = dict(campaign["model"])
    profile["model"] = profile.pop("name")
    report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=report.parent, prefix=".recovery-") as temp:
        partial = Path(temp) / report.name
        replay_successor.run(argparse.Namespace(
            **profile, probes=probes, output=partial, endpoint=url, pid=pid, seed=4073,
            per_family=0, key_env="KIWIBRO_BENCH_API_KEY",
        ))
        summary = checked_replay(probes, partial, campaign)
        try:
            os.link(partial, report)
        except FileExistsError:
            summary = checked_replay(probes, report, campaign)
    return summary


def fit_job(work: Path, corpus: Path) -> dict:
    campaign = campaign_at(work)
    summaries = {split: checked_replay(work / "probes" / f"{split}.jsonl",
                                       work / "baseline" / f"{split}.jsonl", campaign) for split in FIT_SPLITS}
    train_workflows.prepare(corpus, work / "fit.py", work / "probes", work / "baseline")
    return {"fit_job": str(work / "fit.py"), "model": campaign["model"], "baseline": summaries}


def paired(work: Path, candidate: Path, endpoint: str, pid: int | None) -> dict:
    campaign = campaign_at(work)
    probes = work / "probes"
    baseline = work / "baseline"
    fit = json.loads(gzip.decompress(candidate.read_bytes()))["report"]["successor_recovery"]
    cal_probes = probes / "calibrate.jsonl"
    cal_baseline = baseline / "calibrate.jsonl"
    recorded = checked_replay(cal_probes, cal_baseline, campaign)
    if fit["model"] != campaign["model"] or fit["folds"]["calibrate"]["report_sha256"] != recorded["report_sha256"]:
        raise ValueError("Expected a Colab head fitted on this export's exact calibration baseline")

    cal_hinted = work / "calibrate-hinted.jsonl"
    cal_assisted = work / "calibrate-assisted.jsonl"
    policy_path = work / "recovery-policy.json"
    blind = probes / "blind.jsonl"
    cal_plan = workflow_replay.hint_probes(candidate, cal_probes, cal_hinted, split="calibrate")
    if not cal_plan["selected"]:
        return {"model": campaign["model"], "labels": [], "state": "no-calibration-interventions"}
    replay(cal_hinted, cal_assisted, campaign, endpoint, pid)
    policy = workflow_replay.freeze(candidate, cal_probes, cal_baseline, cal_hinted, cal_assisted, blind)
    if not policy_path.exists():
        write_json(policy_path, policy)
    elif read_json(policy_path) != policy:
        raise ValueError("Existing policy differs from the campaign's paired calibration evidence")

    blind_hinted = work / "blind-hinted.jsonl"
    plan = workflow_replay.hint_probes(candidate, blind, blind_hinted, policy=policy)
    if not plan["selected"]:
        return {"model": campaign["model"], "policy": str(policy_path), "labels": policy["labels"],
                "blind_hints": 0, "state": "abstain" if not policy["labels"] else "no-blind-interventions"}
    blind_baseline, blind_assisted = baseline / "blind.jsonl", work / "blind-assisted.jsonl"
    replay(blind, blind_baseline, campaign, endpoint, pid)
    replay(blind_hinted, blind_assisted, campaign, endpoint, pid)
    report = workflow_replay.compare(candidate, blind, blind_baseline, blind_hinted, blind_assisted, policy=policy)
    destination = work / "paired-report.json"
    if not destination.exists():
        write_json(destination, report)
    elif read_json(destination) != report:
        raise ValueError("Existing blind report differs from the frozen campaign")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    start = commands.add_parser("stage")
    start.add_argument("exports", type=Path)
    start.add_argument("work", type=Path)
    start.add_argument("--variant", choices=EXPORTS, required=True)
    start.add_argument("--runtime-id", required=True)
    start.add_argument("--model")
    baseline = commands.add_parser("baseline")
    fit = commands.add_parser("fit-job")
    assisted = commands.add_parser("paired")
    for command in (baseline, fit, assisted):
        command.add_argument("work", type=Path)
    fit.add_argument("corpus", type=Path)
    assisted.add_argument("candidate", type=Path)
    for command in (baseline, assisted):
        command.add_argument("--endpoint", default="http://127.0.0.1:18835")
        command.add_argument("--pid", type=int)
    args = parser.parse_args()
    if args.action == "stage":
        result = stage(args.exports, args.work, args.variant, args.runtime_id, args.model)
    elif args.action == "baseline":
        campaign = campaign_at(args.work)
        result = {split: replay(args.work / "probes" / f"{split}.jsonl", args.work / "baseline" / f"{split}.jsonl",
                                campaign, args.endpoint, args.pid) for split in FIT_SPLITS}
    elif args.action == "fit-job":
        result = fit_job(args.work, args.corpus)
    else:
        result = paired(args.work, args.candidate, args.endpoint, args.pid)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
