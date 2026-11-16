#!/usr/bin/env python3
"""Train and export the lightweight workflow advisers in Google Colab.

    work="$(mktemp -d)"
    python3 scripts/helpers/train_specialists.py prepare "$work/job.py"
    colab run -s kiwi-specialists --timeout 180 "$work/job.py" > "$work/colab.log"
    python3 scripts/helpers/train_specialists.py collect "$work/colab.log"

Training receives the checked-in synthetic corpus and source for feature parity.
The CPU job fits the heads, tunes abstention on a dedicated calibration split,
then scores the quantized runtime weights once on the held-out test split.
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
from collections import Counter
from io import BytesIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS_NAME = "specialists_corpus.txt"
ROOT = HERE.parents[1]
RUNTIME = ROOT / "kiwibro" / "specialists.py"
TARGET = RUNTIME.with_name("specialists-v1.json.gz")
HEADS = {
    "browser": {"lookup", "path", "state", "none"},
    "visual": {"scene", "mechanism", "reveal", "repair", "none"},
    "code": {"diagnose", "tests", "review", "none"},
    "data": {"aggregate", "reconcile", "evidence", "none"},
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_corpus(text: str) -> dict:
    dataset = {head: {split: [] for split in ("train", "calibrate", "test")} for head in HEADS}
    seen = {head: set() for head in HEADS}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            head, split, label = line.removeprefix("[").removesuffix("]").split()
            if label not in HEADS[head] or (
                split not in dataset[head] and not re.fullmatch(r"regression(?:-\d+)?", split)
            ):
                raise ValueError(f"Unknown corpus section {line}")
            dataset[head].setdefault(split, [])
            continue
        identity = " ".join(line.casefold().split())
        if identity in seen[head]:
            raise ValueError(f"Repeated example in {head}: {identity}")
        seen[head].add(identity)
        dataset[head][split].append((line, label))
    for head, splits in dataset.items():
        for split, rows in splits.items():
            if {label for _, label in rows} != HEADS[head]:
                raise ValueError(f"Missing class in {head} {split}")
    return dataset


def baseline(head: str, text: str) -> str:
    """A lexical workflow router, separate from the app's component selector."""
    ask = text.casefold()
    rules = {
        "browser": (
            ("state", r"stale|active tab|current tab|re-?inspect|rendered|redirect|spinner|placeholder|sekme"),
            ("path", r"https?://|follow .{0,36}link|open .{0,42}link|navigate .{0,40}(page|site)|bağlantı"),
            ("lookup", r"official|search the web|look up|find .{0,24}source|from .{0,24}web|resmi"),
        ),
        "visual": (
            ("repair", r"(fix|debug|repair|patch).{0,55}(3d|three\.js|webgl|scene|animation)"
             r"|(3d|webgl|scene).{0,60}(fix|debug|repair|patch)|mevcut.{0,40}(3b|webgl|sahne)"),
            ("reveal", r"exploded|disassembl|teardown|pulling apart|come apart|patlatılmış|parçalarına ayr"),
            ("mechanism", r"mechanical|mechanism|cutaway|piston|hinge|dişli|mekanik|kesit"),
            ("scene", r"three\.js|webgl|3d.{0,28}(scene|animat|page|world)|"
             r"(scene|animat|page).{0,28}3d|üç boyutlu"),
        ),
        "code": (
            ("review", r"\breview\b.{0,60}\b(?:diff|patch|pr|commit|branch|changes)|\bcode review\b"),
            ("tests", r"\b(?:write|add|expand|improve)\b.{0,40}\b(?:tests?|coverage|assertions?)\b"),
            ("diagnose", r"\b(?:crash|failing|traceback|regression|exception|segfault|debug)\b"),
        ),
        "data": (
            ("reconcile", r"\b(?:join|merge|reconcile|match)\b.{0,40}\b(?:csv|sheets?|tables?|exports?|rows?)\b"),
            ("evidence", r"\b(?:cite|quote|find)\b.{0,45}\b(?:pdfs?|documents?|manuals?|contracts?)\b"),
            ("aggregate", r"\b(?:calculate|compute|sum|average|aggregate|group by)\b.{0,55}"
             r"\b(?:csv|spreadsheet|column|data|rows|tables?)\b"),
        ),
    }
    return next((label for label, pattern in rules[head] if re.search(pattern, ask)), "none")


def score(expected: list[str], predicted: list[str]) -> dict:
    counts = Counter(expected)
    correct = Counter(wanted for wanted, found in zip(expected, predicted) if wanted == found)
    advice = sum(label != "none" for label in predicted)
    correct_advice = sum(correct.values()) - correct["none"]
    return {
        "requests": len(expected),
        "exact": sum(correct.values()),
        "advice": advice,
        "correct_advice": correct_advice,
        "wrong_advice": advice - correct_advice,
        "false_activation": sum(
            wanted == "none" and found != "none" for wanted, found in zip(expected, predicted)
        ),
        "positive_requests": len(expected) - counts["none"],
        "correct_by_class": {label: correct[label] for label in sorted(counts)},
        "count_by_class": dict(sorted(counts.items())),
    }


def choose_thresholds(expected: list[str], predictions: list[tuple[str, float, float]]) -> tuple[float, float]:
    choices = []
    for minimum in (0.32, 0.4, 0.48, 0.56, 0.64, 0.72, 0.8, 0.86, 0.9):
        for separation in (0.07, 0.14, 0.23, 0.33, 0.43, 0.53):
            decisions = [
                label if probability >= minimum and lead >= separation else "none"
                for label, probability, lead in predictions
            ]
            result = score(expected, decisions)
            choices.append((
                result["correct_advice"] - 5 * result["wrong_advice"] - 2 * result["false_activation"],
                -result["wrong_advice"], result["correct_advice"], minimum, separation,
            ))
    return max(choices)[-2:]


def qualifies(candidate: dict, reference: dict) -> bool:
    """Ship advice only if the frozen test split supports its usefulness."""
    return (
        candidate["false_activation"] == 0
        and candidate["wrong_advice"] <= 1
        and candidate["correct_advice"] >= candidate["positive_requests"] * 0.50
        and candidate["exact"] > reference["exact"]
        and all(
            candidate["correct_by_class"][label] >= count * 0.25
            for label, count in candidate["count_by_class"].items() if label != "none"
        )
    )


def train() -> None:
    if not os.environ.get("COLAB_RELEASE_TAG") and not os.environ.get("COLAB_BACKEND_VERSION"):
        raise RuntimeError("This job is configured to train in a Google Colab runtime")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import numpy as np
    import sklearn
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    import specialists as runtime

    raw_corpus = (HERE / CORPUS_NAME).read_bytes()
    dataset = read_corpus(raw_corpus.decode())
    pack = {"version": 1, "origin": "google-colab", "scale": runtime.WEIGHT_SCALE, "heads": {}}
    report = {
        "origin": "google-colab",
        "release": os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"),
        "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "corpus_sha256": digest(raw_corpus),
        "runtime_sha256": digest(Path(runtime.__file__).read_bytes()),
        "trainer_sha256": digest(Path(__file__).read_bytes()),
        "heads": {},
    }
    for head, splits in dataset.items():
        input_rows, gold = zip(*splits["train"])
        vectorizer = CountVectorizer(
            analyzer=runtime.features, binary=True, min_df=2, max_features=5500, dtype=np.float64,
        )
        vectors = vectorizer.fit_transform(input_rows)
        with threadpool_limits(limits=1):
            clf = LogisticRegression(C=0.08, max_iter=500, random_state=11, class_weight="balanced")
            clf.fit(vectors, gold)
        weights = np.rint(clf.coef_.T * runtime.WEIGHT_SCALE).astype(int)
        allowed = list(runtime.HINTS[head])
        model = {
            "labels": list(clf.classes_),
            "advice_labels": allowed,
            "bias": np.rint(clf.intercept_ * runtime.WEIGHT_SCALE).astype(int).tolist(),
            "weights": {
                word: values.tolist()
                for word, values in zip(vectorizer.get_feature_names_out(), weights) if values.any()
            },
        }
        def eligible(label):
            return label if label in allowed else "none"

        def calibration(text):
            label, probability, lead = runtime.prediction(model, runtime.features(text))
            return (eligible(label), probability, lead) if runtime.admissible(head, text) else ("none", 1, 1)

        cal_text, cal_gold = zip(*splits["calibrate"])
        model["thresholds"] = choose_thresholds(
            [eligible(label) for label in cal_gold], [calibration(text) for text in cal_text],
        )
        test_text, test_gold = zip(*splits["test"])
        test_gold = [eligible(label) for label in test_gold]
        raw = [eligible(runtime.prediction(model, runtime.features(text))[0]) for text in test_text]
        def accepted(texts):
            return [
                runtime.classify(model, runtime.features(text)) if runtime.admissible(head, text) else "none"
                for text in texts
            ]

        actual = accepted(test_text)
        reference = score(test_gold, [eligible(baseline(head, text)) for text in test_text])
        candidate = score(test_gold, actual)
        passed = qualifies(candidate, reference)
        regression = {}
        for split, rows in splits.items():
            if split.startswith("regression"):
                old_text, old_gold = zip(*rows)
                old_gold = [eligible(label) for label in old_gold]
                regression[split] = score(old_gold, accepted(old_text))
                old_reference = score(old_gold, [eligible(baseline(head, text)) for text in old_text])
                passed = passed and qualifies(regression[split], old_reference)
        report["heads"][head] = {
            "split_sizes": {split: len(rows) for split, rows in splits.items()},
            "feature_count": len(model["weights"]),
            "advice_labels": allowed,
            "thresholds": model["thresholds"],
            "baseline": reference,
            "raw_top_choice": score(test_gold, raw),
            "accepted": candidate,
            "passed": passed,
            "regression": regression,
        }
        mismatches = [
            f"{wanted} -> {found}: {text}" for text, wanted, found in zip(test_text, test_gold, actual)
            if wanted != found
        ]
        print(f"{head}: {'passed' if passed else 'failed'}")
        for mismatch in mismatches:
            print(f"  {mismatch}")
        if passed:
            pack["heads"][head] = model
    pack["report"] = report
    payload = gzip.compress(json.dumps(pack, separators=(",", ":"), sort_keys=True).encode(), mtime=0)
    print("KIWIBRO_REPORT " + json.dumps(report, sort_keys=True))
    print("KIWIBRO_SHA256 " + digest(payload))
    for chunk in textwrap.wrap(base64.b64encode(payload).decode("ascii"), 3072):
        print("KIWIBRO_BUNDLE " + chunk)


def prepare(path: Path) -> None:
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as source:
        for part in (Path(__file__), HERE / CORPUS_NAME, RUNTIME):
            source.writestr(part.name, part.read_bytes())
    payload = base64.b64encode(archive.getvalue()).decode("ascii")
    path.write_text(textwrap.dedent(f"""\
        import base64
        import io
        import runpy
        import sys
        import tempfile
        import zipfile
        with tempfile.TemporaryDirectory(prefix="kiwi-specialists-") as job:
            with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))) as archive:
                archive.extractall(job)
            sys.path.insert(0, job)
            sys.argv = ["train_specialists.py", "_train"]
            runpy.run_path(job + "/train_specialists.py", run_name="__main__")
        """), encoding="utf-8")


def collect(log: Path, out: Path) -> None:
    lines = log.read_text(encoding="utf-8").splitlines()
    hashes = [line.split(" ", 1)[1] for line in lines if line.startswith("KIWIBRO_SHA256 ")]
    chunks = [line.split(" ", 1)[1] for line in lines if line.startswith("KIWIBRO_BUNDLE ")]
    payload = base64.b64decode("".join(chunks), validate=True)
    if hashes != [digest(payload)]:
        raise ValueError("Colab output is incomplete or its digest does not match")
    pack = json.loads(gzip.decompress(payload))
    report = pack["report"]
    if pack["origin"] != "google-colab" or not pack["heads"]:
        raise ValueError("No qualifying Colab-trained specialist weights")
    for field, part in (
        ("corpus_sha256", HERE / CORPUS_NAME),
        ("runtime_sha256", RUNTIME),
        ("trainer_sha256", Path(__file__)),
    ):
        if report[field] != digest(part.read_bytes()):
            raise ValueError(f"Colab export differs from source: {part.name}")
    for head in pack["heads"]:
        stats = report["heads"][head]
        if not stats["passed"] or not qualifies(stats["accepted"], stats["baseline"]):
            raise ValueError(f"Head {head} did not pass its evaluation gate")
    out.write_bytes(payload)
    print(json.dumps(report, indent=2))
    print(f"Wrote {len(payload):,} bytes to {out}; sha256 {hashes[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare").add_argument("path", type=Path)
    reader = commands.add_parser("collect")
    reader.add_argument("log", type=Path)
    reader.add_argument("--out", type=Path, default=TARGET)
    commands.add_parser("_train")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.path)
    elif args.command == "collect":
        collect(args.log, args.out)
    else:
        train()


if __name__ == "__main__":
    main()
