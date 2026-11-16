#!/usr/bin/env python3
"""Train optional workflow helpers on Colab CPU using validated KiwiBro episodes.

    python scripts/mind/train_workflows.py prepare CORPUS_DIR JOB.py
    colab --config SESSION.json run -s kiwi-workflows --timeout 180 JOB.py > LOG
    python scripts/mind/train_workflows.py collect LOG CANDIDATE.json.gz

The exported pack is an evaluation candidate. The installed specialist pack
uses a different contract. Only train.jsonl goes into feature fitting. Authored
class-specific calibration and test prompts are separate from that corpus.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import sys
import textwrap
import zipfile
from io import BytesIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CURRICULUM = "workflow_calibration.txt"
HEADS = {
    "workflow": {"repository", "terminal", "artifact", "delegate", "math", "none"},
    "recovery": {"render", "location", "boundary", "dependency", "none"},
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def first_user(row: dict) -> str:
    user = next(message["content"] for message in row["messages"] if message["role"] == "user")
    return user if isinstance(user, str) else " ".join(part["text"] for part in user if part["type"] == "text")


def derived(rows: list[dict]) -> dict:
    grouped: dict[str, list[tuple[str, str]]] = {name: [] for name in HEADS}
    families = {
        "repository": "repository", "native_code": "repository", "architecture": "repository",
        "environment": "terminal", "artifact": "artifact", "delegation": "delegate", "math_physics": "math",
    }
    failures = {
        "browser-js": ("browser", "id='app'", "render"),
        "recovery-denied": ("run_command", "Denied", "boundary"),
        "recovery-table": ("analyze_table", "File not found", "location"),
        "environment-tools": ("run_command", "command not found", "dependency"),
    }
    for row in rows:
        episode = re.sub(r"-\d+$", "", row["id"])
        kind = "terminal" if episode == "desktop-terminal" else families.get(row["family"], "none")
        grouped["workflow"].append((first_user(row), kind))
        bad_tool, text, recovery = failures.get(episode, ("", "", "none"))
        for message in row["messages"]:
            if message["role"] == "tool":
                tool, result = message["name"], message["content"]
                label = recovery if bad_tool == tool and text in result else "none"
                grouped["recovery"].append((f"{tool}: {result[:1050]}", label))
    return grouped


def dataset(corpus: bytes, additions: bytes) -> dict:
    train = derived([json.loads(line) for line in corpus.splitlines() if line])
    found = {name: {"train": rows, "calibrate": [], "test": []} for name, rows in train.items()}
    active = None
    for line in additions.decode().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            active = tuple(line[1:-1].split())
            continue
        head, split, label = active
        if label not in HEADS[head]:
            raise ValueError(f"{head}: unknown label {label}")
        if re.fullmatch(r"regression(?:-\d+)?", split):
            found[head].setdefault(split, [])
        found[head][split].append((line, label))
    for head, splits in found.items():
        used = {}
        for split, rows in splits.items():
            unique = {}
            for text, label in rows:
                key = " ".join(text.casefold().split())
                if key in used and used[key] != (split, label):
                    raise ValueError(f"Mismatched or leaked {head} prompt: {text}")
                unique[key] = text, label
                used[key] = split, label
            splits[split] = list(unique.values())
            if {label for _, label in unique.values()} != HEADS[head]:
                raise ValueError(f"{head}/{split} lacks labels")
    return found


RULES = {
    "workflow": {
        "repository": r"\brepository\b|\bgit diff\b|\bpatch\b|\bcompile\b|\bapi contract\b",
        "terminal": r"\bterminal\b|\bexit code\b|\bcommand -v\b|\bsystem info\b",
        "artifact": r"\b(?:3d|html|webgl|render)\b.{0,45}\b(?:page|browser|animation|cube|scene|mobile)\b",
        "delegate": r"\b(?:delegate|helper model|ask only)\b",
        "math": r"\b(?:calculate|acceleration|newtons?|equation|maths?|denominator)\b",
    },
    "recovery": {
        "render": r"\b(?:js shell|javascript required|spinner|stale tab|hydration|rendered view)\b|id=['\"]app",
        "location": r"\b(?:no such file|file not found|path does not exist|missing directory)\b",
        "boundary": r"\b(?:denied|permission required|read-only|approval needed)\b",
        "dependency": r"\b(?:not installed|command not found|no model|cuda unavailable|unavailable)\b",
    },
}


def baseline(head: str, text: str) -> str:
    return next((label for label, pattern in RULES[head].items() if re.search(pattern, text, re.I)), "none")


DELEGATION_REQUEST = re.compile(
    r"\A\s*(?:(?:please|can you|could you)\s+)?(?:ask|use|send|have|delegate|call|invoke|run|check|get|"
    r"i want (?:one|a)|i need (?:one|a))\b", re.I,
)
SUCCESSFUL_OBSERVATION = re.compile(
    r"\b(?:exited (?:normally|with (?:status|code) 0)|exit code 0|completed successfully|"
    r"validation (?:succeeded|passed)|file opened|file (?:saved|was added)|permission was approved|"
    r"returned (?:the answer|done)|rendered successfully)\b", re.I,
)


def scoped(head: str, decision: str, text: str) -> str:
    if head == "workflow" and decision == "delegate" and not DELEGATION_REQUEST.search(text):
        return "none"
    if head == "recovery" and SUCCESSFUL_OBSERVATION.search(text):
        return "none"
    return decision


def train() -> None:
    if not (os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION")):
        raise RuntimeError("Run this training command in Google Colab")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import numpy as np
    import sklearn
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    import specialists as runtime
    from train_specialists import choose_thresholds, score

    corpus = Path("train.jsonl").read_bytes()
    additions = Path(CURRICULUM).read_bytes()
    splits = dataset(corpus, additions)
    pack = {"version": 1, "origin": "google-colab", "scale": runtime.WEIGHT_SCALE, "heads": {}}
    report = {
        "kind": "workflow-and-recovery-candidate", "origin": pack["origin"],
        "release": os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"),
        "python": sys.version.split()[0], "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "sha256": {name: digest(Path(name).read_bytes())
                   for name in ("train.jsonl", CURRICULUM, "specialists.py", "train_workflows.py")},
        "heads": {},
    }
    for name, data in splits.items():
        samples, labels = zip(*data["train"])
        vectorizer = CountVectorizer(analyzer=runtime.features, binary=True, min_df=2, max_features=5000,
                                     dtype=np.float64)
        vectors = vectorizer.fit_transform(samples)
        with threadpool_limits(limits=1):
            fit = LogisticRegression(C=0.12, max_iter=500, random_state=11, class_weight="balanced").fit(vectors, labels)
        weights = np.rint(fit.coef_.T * runtime.WEIGHT_SCALE).astype(int)
        head = {
            "labels": fit.classes_.tolist(), "advice_labels": sorted(HEADS[name] - {"none"}),
            "bias": np.rint(fit.intercept_ * runtime.WEIGHT_SCALE).astype(int).tolist(),
            "weights": {key: values.tolist() for key, values in zip(vectorizer.get_feature_names_out(), weights)
                        if values.any()},
        }
        calibration, labels = zip(*data["calibrate"])
        head["thresholds"] = choose_thresholds(labels, [runtime.prediction(head, runtime.features(text)) for text in calibration])
        def evaluate(rows):
            tests, labels = zip(*rows)
            raw = [runtime.classify(head, runtime.features(text)) for text in tests]
            calls = [scoped(name, choice, text) for choice, text in zip(raw, tests)]
            return {
                "baseline": score(labels, [baseline(name, text) for text in tests]),
                "quantized_ungated": score(labels, raw),
                "candidate": score(labels, calls),
                "misses": [(text, expected, actual) for text, expected, actual in zip(tests, labels, calls)
                           if expected != actual],
            }
        report["heads"][name] = {
            "split_sizes": {key: len(rows) for key, rows in data.items()},
            "thresholds": list(head["thresholds"]), "feature_count": len(head["weights"]),
            **evaluate(data["test"]),
        }
        report["heads"][name]["regression"] = {key: evaluate(rows) for key, rows in data.items()
                                               if key.startswith("regression")}
        pack["heads"][name] = head
    pack["report"] = report
    payload = gzip.compress(json.dumps(pack, sort_keys=True, separators=(",", ":")).encode(), mtime=0)
    print("KIWI_WORKFLOW_HASH " + digest(payload))
    for chunk in textwrap.wrap(base64.b64encode(payload).decode("ascii"), 3000):
        print("KIWI_WORKFLOW_DATA " + chunk)


def prepare(corpus: Path, destination: Path) -> None:
    source = {
        "train.jsonl": corpus / "train.jsonl", "train_workflows.py": Path(__file__),
        CURRICULUM: HERE / CURRICULUM,
        "specialists.py": ROOT / "kiwibro" / "specialists.py",
        "train_specialists.py": ROOT / "scripts" / "helpers" / "train_specialists.py",
    }
    dataset(source["train.jsonl"].read_bytes(), source[CURRICULUM].read_bytes())
    bundle = BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in source.items():
            archive.writestr(name, path.read_bytes())
    payload = base64.b64encode(bundle.getvalue()).decode("ascii")
    destination.write_text(textwrap.dedent(f"""\
        import base64, io, os, runpy, sys, tempfile, zipfile
        with tempfile.TemporaryDirectory(prefix="kiwi-workflows-") as directory:
            with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))) as archive:
                archive.extractall(directory)
            os.chdir(directory)
            sys.path.insert(0, directory)
            sys.argv = ['train_workflows.py', '_train']
            runpy.run_path("train_workflows.py", run_name="__main__")
        """), encoding="utf-8")


def collect(log: Path, output: Path) -> dict:
    lines = log.read_text(encoding="utf-8").splitlines()
    hashes = [line.split(" ", 1)[1] for line in lines if line.startswith("KIWI_WORKFLOW_HASH ")]
    payload = base64.b64decode("".join(line.split(" ", 1)[1] for line in lines
                                       if line.startswith("KIWI_WORKFLOW_DATA ")), validate=True)
    if hashes != [digest(payload)]:
        raise ValueError("Colab artifact digest mismatch")
    pack = json.loads(gzip.decompress(payload))
    if pack["origin"] != "google-colab" or set(pack["heads"]) != set(HEADS):
        raise ValueError("Unexpected Colab artifact")
    output.write_bytes(payload)
    return {"pack_sha256": hashes[0], "bytes": len(payload), **pack["report"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    bundle = commands.add_parser("prepare")
    bundle.add_argument("corpus", type=Path)
    bundle.add_argument("output", type=Path)
    result = commands.add_parser("collect")
    result.add_argument("log", type=Path)
    result.add_argument("output", type=Path)
    commands.add_parser("_train")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.corpus, args.output)
    elif args.action == "collect":
        print(json.dumps(collect(args.log, args.output), indent=2))
    else:
        train()


if __name__ == "__main__":
    main()
