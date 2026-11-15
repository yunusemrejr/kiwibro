#!/usr/bin/env python3
"""The control head's "needs a bigger model" output, trained on the engine's own states.

The head decides per message whether the words of the answer go to the helper model the user selected. This
tool trains that one output apart from the rest, on states read from the running engine (4-bit trunk, real
prompt), so what it learns is what it will see:

    states  asks the host for the state the head reads, once per labelled request (alone, and as the second
            message of a short chat), and writes them to a .npz file
    train   fits a small unit on those states and writes a head whose `hard` output is that unit. The hidden
            layer grows by the unit's size; the other outputs read none of the new units, so they keep their
            values. With --pack the head goes into that pack folder and the manifest follows it (size, sha256,
            the measured figures, the version)

Run both after the training script's export step: the head it writes is the one to pass as --head.

Labels: one JSON object per line, {"ask": "...", "heavy": true|false, "source": "..."}; replies for the short
chats: {"ask": "...", "reply": "..."}. Needs numpy only.

The labels of the 2.0.1 pack came from a larger model that saw each request alone and was asked: would a 0.6B
model likely fail here where a 4B model does visibly better (several steps of reasoning, a maths word problem,
code beyond a few lines, a long or structured text, an in-depth comparison or explanation)? Greetings, a short
fact, a one-line sum, a few lines of verse and anything one tool call carries out are light, and so is every
request it was unsure about. The requests were 6,000 first messages from the public data sets the training
script's `load_asks` reads (2,700 from OpenAssistant oasst2, 1,800 from Dolly, 1,500 from MASSIVE); the replies
for the short chats were one or two sentences the same larger model wrote for light requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import formats  # noqa: E402

HARD = "hard"
NOTE_HEADER = "[Kiwibro context for this turn, added automatically, not written by the user]"
NEW_TOPIC = "This message starts a new topic: answer it on its own; the earlier talk is only background."
ZONES = ("+03", "UTC", "CET", "EST", "+0530")


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _side(ask: str) -> str:
    """Which part of the data a request belongs to, the same in every run and for every prompt built from it."""
    slot = int(hashlib.sha256(ask.encode()).hexdigest()[:8], 16) % 100
    return "train" if slot < 70 else "val" if slot < 85 else "test"


# --- states -------------------------------------------------------------------------------------------


def _turn_note(rng: random.Random, ask: str, second: bool) -> str:
    """The note Kiwibro adds under the newest message: the clock always, the reading's lines sometimes."""
    moment = datetime(2026, 1, 1) + timedelta(minutes=rng.randrange(2 * 365 * 24 * 60))
    blocks = [f"{NOTE_HEADER}\nLocal time: {moment:%A}, {moment.day} {moment:%B %Y, %H:%M} {rng.choice(ZONES)}"]
    reading = []
    words = sorted({word.strip(".,?!\"'()").lower() for word in ask.split()}, key=len, reverse=True)
    if words and words[0] and rng.random() < 0.3:
        reading.append(f"The answer must honour: {' '.join(sorted(words[:rng.choice((1, 2))]))}.")
    if second and rng.random() < 0.5:
        reading.append(NEW_TOPIC)
    if reading:
        blocks.append("\n".join(reading))
    return "\n\n".join(blocks)


def _prompt(prefix: str, rng: random.Random, ask: str, before: tuple[str, str] | None) -> str:
    earlier = (f"<|im_start|>user\n{before[0]}<|im_end|>\n<|im_start|>assistant\n{before[1]}<|im_end|>\n"
               if before else "")
    return (f"{prefix}{earlier}<|im_start|>user\n{ask}\n\n{_turn_note(rng, ask, before is not None)}<|im_end|>\n"
            "<|im_start|>assistant\n")


def states(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    labels = _lines(args.labels)
    replies = [(row["ask"], row["reply"]) for row in _lines(args.replies)] if args.replies else []
    prefixes = [path.read_text(encoding="utf-8") for path in args.prefix]
    rng.shuffle(labels)
    labels = labels[:args.limit] if args.limit else labels
    samples = [(row, None) for row in labels]
    if replies:  # the same question after a short exchange: the head runs on every message, not only the first
        samples += [(row, rng.choice(replies)) for row in rng.sample(labels, int(len(labels) * args.second))]
    rng.shuffle(samples)

    def control(prompt: str, slot: str) -> dict:
        request = urllib.request.Request(
            args.host.rstrip("/") + "/control", data=json.dumps({"prompt": prompt, "slot": slot, "state": True}).encode(),
            headers={"Authorization": f"Bearer {args.key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)

    x, rows, names = [], [], None

    def save() -> None:  # also on the way: a run that is stopped keeps what it has read
        part = args.out.with_name(args.out.name + ".part.npz")
        np.savez_compressed(
            part, x=np.stack(x), heavy=np.array([r[1] for r in rows]), second=np.array([r[3] for r in rows]),
            ask=np.array([r[0] for r in rows]), source=np.array([r[2] for r in rows]),
            logits=np.array([r[4] for r in rows], dtype=np.float32), names=np.array(names))
        part.replace(args.out)

    for index, (row, before) in enumerate(samples):
        which = index % len(prefixes)
        prompt = _prompt(prefixes[which], rng, row["ask"], before)  # built for every sample: the draws stay the same
        if args.only and _side(row["ask"]) != args.only:
            continue
        got = control(prompt, f"p{which}")
        names = got["names"]
        x.append(np.asarray(got["state"], dtype=np.float32))
        rows.append((row["ask"], bool(row["heavy"]), row.get("source", ""), before is not None, got["logits"]))
        if len(rows) % 100 == 0:
            save()
            print(f"{len(rows)} states, request {index + 1} of {len(samples)}", flush=True)
    if not rows:
        raise SystemExit("no request to read: check --labels, --limit and --only")
    save()
    print(f"wrote {len(rows)} states to {args.out}")
    return 0


# --- train --------------------------------------------------------------------------------------------


def _normed(x: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt((x * x).mean(axis=1, keepdims=True) + eps)


def _fit(x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, hidden: int, seed: int,
         epochs: int = 80, rate: float = 1e-3, decay: float = 2e-2, drop: float = 0.15) -> tuple[np.ndarray, ...]:
    """A two-layer unit, logistic loss with the classes weighted evenly, Adam with weight decay and dropped
    inputs; the weights that rank the held-out validation requests best are the ones kept."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 1 / np.sqrt(x.shape[1]), (hidden, x.shape[1])).astype(np.float32)
    b1 = np.zeros(hidden, np.float32)
    w2 = rng.normal(0, 1 / np.sqrt(hidden), hidden).astype(np.float32)
    b2 = np.zeros(1, np.float32)
    params = [w1, b1, w2, b2]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    weight = np.where(y, 0.5 / max(1, y.sum()), 0.5 / max(1, (~y).sum())).astype(np.float32) * len(y)
    best, kept, step = -1.0, None, 0
    for _epoch in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 128):
            batch = order[start:start + 128]
            inputs = x[batch] * (rng.random((len(batch), x.shape[1])) >= drop) / (1 - drop)
            mid = np.maximum(0, inputs @ w1.T + b1)
            error = (1 / (1 + np.exp(-(mid @ w2 + b2[0]))) - y[batch]) * weight[batch] / len(batch)
            back = np.outer(error, w2) * (mid > 0)
            grads = [back.T @ inputs, back.sum(0), mid.T @ error, np.array([error.sum()], np.float32)]
            step += 1
            for p, g, m, v in zip(params, grads, first, second):
                m += 0.1 * (g - m)
                v += 0.001 * (g * g - v)
                p -= rate * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
                if p.ndim > 1 or p is w2:
                    p -= rate * decay * p
        score = _average_precision(_score(x_val, w1, b1, w2, b2), y_val)
        if score > best:
            best, kept = score, [p.copy() for p in params]
    return (*kept, best)


def _score(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    return np.maximum(0, x @ w1.T + b1) @ w2 + b2[0]


def _average_precision(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(-score)
    hits = np.cumsum(y[order])
    return float((hits / np.arange(1, len(y) + 1))[y[order]].mean()) if y.any() else 0.0


def _rates(said: np.ndarray, y: np.ndarray) -> dict:
    right = int((said & y).sum())
    return {"n": int(len(y)), "heavy": int(y.sum()), "said": int(said.sum()),
            "precision": round(right / max(1, int(said.sum())), 3), "recall": round(right / max(1, int(y.sum())), 3),
            "light_named": round(int((said & ~y).sum()) / max(1, int((~y).sum())), 3)}


def _mark(score: np.ndarray, y: np.ndarray, light: float) -> float:
    """The score above which the unit says "heavy": it names at most that share of the light requests.

    A share of the light requests, not of the named ones that are right: it means the same whatever part of the
    requests is heavy, which differs between data sets and from real use."""
    quiet = np.sort(score[~y])[::-1]  # the light requests, highest score first
    if not len(quiet):
        raise SystemExit("the validation side holds no light request to set the threshold on")
    allowed = min(int(light * len(quiet)), len(quiet) - 1)
    return float((quiet[allowed - 1] + quiet[allowed]) / 2) if allowed else float(quiet[0])


def _into_manifest(pack: Path, report: dict, version: str | None) -> None:
    """The pack's manifest follows its new head: the size and sha256 the app checks, the figures, the version."""
    path = pack / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    head = pack / "head.kmh"
    manifest["files"]["head.kmh"] = {"bytes": head.stat().st_size, "sha256": hashlib.sha256(head.read_bytes()).hexdigest()}
    kept = {key: value for key, value in (manifest.get("head_report") or {}).items() if not key.startswith("hard_")}
    manifest["head_report"] = {**kept, "hard_precision": report["test"]["precision"],
                               "hard_recall": report["test"]["recall"], "hard_light_named": report["test"]["light_named"],
                               "hard_share": report["heavy_share"], "hard_unit": report}
    if version:
        manifest["version"] = version
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def train(args: argparse.Namespace) -> int:
    if args.pack and not (args.pack / "manifest.json").is_file():
        raise SystemExit(f"{args.pack} holds no manifest.json: --pack takes the folder of a finished pack")
    out = args.pack / "head.kmh" if args.pack else args.out
    data = np.load(args.states)
    old = formats.read_head(args.head)
    names = list(old["names"])
    if HARD not in names:
        raise SystemExit(f"{args.head} has no output named {HARD!r}")
    x, y = _normed(data["x"].astype(np.float32), old["eps"]), data["heavy"].astype(bool)
    side = np.array([_side(str(ask)) for ask in data["ask"]])
    parts = {name: side == name for name in ("train", "val", "test")}
    w1, b1, w2, b2, ranked = _fit(x[parts["train"]], y[parts["train"]], x[parts["val"]], y[parts["val"]],
                                  args.hidden, args.seed)
    b2 = b2 - _mark(_score(x[parts["val"]], w1, b1, w2, b2), y[parts["val"]], args.light)
    said = _score(x, w1, b1, w2, b2) > 0
    before = data["logits"][:, names.index(HARD)] > 0  # the head the states were read with
    test = parts["test"]
    report = {
        "states": int(len(y)), "heavy_share": round(float(y.mean()), 3), "hidden": args.hidden, "seed": args.seed,
        "validation_average_precision": round(ranked, 3), "light_named_asked": args.light,
        "test": _rates(said[test], y[test]),
        "test_first_message": _rates(said[test & ~data["second"]], y[test & ~data["second"]]),
        "test_later_message": _rates(said[test & data["second"]], y[test & data["second"]]),
        "test_by_source": {str(source): _rates(said[test & (data["source"] == source)], y[test & (data["source"] == source)])
                           for source in sorted(set(data["source"].tolist()))},
        "test_before": _rates(before[test], y[test]),
    }
    print(json.dumps(report, indent=1))
    if out:
        k, wide = names.index(HARD), old["w1"].shape[0]
        new_w2 = np.zeros((len(names), wide + args.hidden), np.float32)
        new_w2[:, :wide] = old["w2"]
        new_w2[k] = 0
        new_w2[k, wide:] = w2
        new_b2 = old["b2"].astype(np.float32).copy()
        new_b2[k] = b2[0]
        formats.write_head(out, np.vstack([old["w1"], w1]), np.concatenate([old["b1"], b1]), new_w2, new_b2,
                           names, layer=old["layer"], eps=old["eps"])
        print(f"wrote {out}")
    if args.pack:
        _into_manifest(args.pack, report, args.version)
    if args.report:
        args.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("states", help="read the head's input from the running host for labelled requests")
    s.add_argument("--host", required=True, help="the host's address, as http://127.0.0.1:PORT")
    s.add_argument("--key", required=True, help="the secret the host was started with (KIWIMIND_KEY)")
    s.add_argument("--prefix", type=Path, action="append", required=True,
                   help="a file holding the system turn exactly as the app renders it; give several to vary it")
    s.add_argument("--labels", type=Path, required=True)
    s.add_argument("--replies", type=Path)
    s.add_argument("--second", type=float, default=0.35, help="share of the requests also read as a second message")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--only", choices=("train", "val", "test"),
                   help="read one side of the data only, with the prompts a full run gives it (to measure another pack)")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(run=states)
    t = sub.add_parser("train", help="fit the unit and write a head whose `hard` output is that unit")
    t.add_argument("--states", type=Path, required=True)
    t.add_argument("--head", type=Path, required=True, help="the head to keep every other output of")
    t.add_argument("--hidden", type=int, default=32)
    t.add_argument("--light", type=float, default=0.03,
                   help="share of the light validation requests it may name heavy; the threshold follows from it")
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--out", type=Path, help="where to write the new head")
    t.add_argument("--pack", type=Path, help="or: a pack folder, to replace its head.kmh and update its manifest")
    t.add_argument("--version", help="with --pack: the pack's new version")
    t.add_argument("--report", type=Path)
    t.set_defaults(run=train)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
