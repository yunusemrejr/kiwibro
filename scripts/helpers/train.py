#!/usr/bin/env python3
"""Train the Kiwi Sense helpers and write them into static/sense/sense.bin.

Runs on a GPU machine (it was run on Google Colab), never inside Kiwibro:

    python train.py langdata   # language-identification data from open datasets
    python train.py lang msg tag shift rel
    python train.py export     # write the heads into $KS_REPO/static/sense/sense.bin
    python train.py check      # run the written file through the Python twin of the core

The features come from the Python twin itself (kiwibro/sense/pyengine.py), so what the
network learns from is exactly what the core computes at run time. Tables are trained
already rounded to int8. Needs torch, numpy, datasets; labelled data from label.py.

Language data (all open): papluca/language-identification, OpenAssistant/oasst2,
mteb/amazon_massive_intent (MASSIVE, CC BY 4.0), allenai/WildChat-1M (ODC-BY).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import struct
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("KS_REPO", ".")).resolve()
DATA = Path(os.environ.get("KS_DATA", "data")).resolve()
OUT = Path(os.environ.get("KS_OUT", "heads")).resolve()
sys.path.insert(0, os.environ.get("KS_CODE", str(REPO)))

from kiwibro.sense import blob as blobfile, normalize  # noqa: E402
from kiwibro.sense.helpers import (  # noqa: E402
    GROUPS, SHIFT, TAGS, judged_part, lang_text, segments,
)
from kiwibro.sense.pyengine import (  # noqa: E402
    F_CHARS, F_EMB, F_WORDS, F_WORDS_A, KIND_PAIR, KIND_TEXT, KIND_TOKENS, MAX_TOKENS, MAX_WORDS,
    PAIR_SCALARS, PyEngine, char_features, token_features, word_features,
)
from kiwibro.sense.blob import fnv  # noqa: E402

SEED = 11
BLOB = REPO / "static" / "sense" / "sense.bin"

SPECS = {
    "Hlng": dict(kind=KIND_TEXT, flags=F_WORDS | F_CHARS, buckets=32768, dim=24, hidden=64, cap=900),
    "Hmsg": dict(kind=KIND_TEXT, flags=F_EMB | F_WORDS, buckets=16384, dim=32, hidden=96, cap=600),
    "Htag": dict(kind=KIND_TOKENS, flags=F_EMB | F_WORDS, buckets=16384, dim=24, hidden=64, cap=11),
    "Hshf": dict(kind=KIND_PAIR, flags=F_EMB | F_WORDS, buckets=8192, dim=16, hidden=48, cap=600),
    "Hrel": dict(kind=KIND_PAIR, flags=F_EMB | F_WORDS | F_WORDS_A, buckets=8192, dim=16, hidden=48, cap=600),
}

LANG_NAMES = {
    "en": "English", "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "az": "Azerbaijani", "bg": "Bulgarian",
    "bn": "Bengali", "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German", "el": "Greek",
    "es": "Spanish", "eu": "Basque", "fa": "Persian", "fi": "Finnish", "fr": "French", "gl": "Galician",
    "he": "Hebrew", "hi": "Hindi", "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian", "is": "Icelandic",
    "it": "Italian", "ja": "Japanese", "jv": "Javanese", "ka": "Georgian", "km": "Khmer", "kn": "Kannada",
    "ko": "Korean", "lv": "Latvian", "ml": "Malayalam", "mn": "Mongolian", "ms": "Malay", "my": "Burmese",
    "nb": "Norwegian", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sl": "Slovenian", "sq": "Albanian", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu",
    "th": "Thai", "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese",
    "zh": "Chinese", "xx": "",
}
WILD_LANGS = {"Chinese": "zh", "Russian": "ru", "Arabic": "ar", "Japanese": "ja", "Korean": "ko",
              "Persian": "fa", "Ukrainian": "uk", "Greek": "el", "Hindi": "hi", "Thai": "th", "Hebrew": "he",
              "Bulgarian": "bg"}
TECH = ["python", "docker", "api", "laptop", "email", "excel", "github", "linux", "chatgpt", "iphone", "wifi",
        "javascript", "sql", "react", "windows", "server", "bug", "login", "online", "ok", "cloud", "app"]


def log(*parts) -> None:
    print(*parts, flush=True)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def deaccent(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def non_latin_share(text: str) -> float:
    total = other = 0
    for ch in text:
        if ch.isalpha():
            total += 1
            if ord(ch) > 0x24F and not 0x1E00 <= ord(ch) <= 0x1EFF:
                other += 1
    return other / total if total else 0.0


# ---------------------------------------------------------------- language data

def langdata() -> None:
    from datasets import load_dataset

    rng = random.Random(SEED)
    by_lang: dict[str, list[str]] = defaultdict(list)

    def add(code: str, text: str, source: str) -> None:
        code = {"pt-BR": "pt", "zh-CN": "zh", "zh-TW": "zh", "no": "nb"}.get(code, code).split("-")[0]
        for seg in segments(judged_part(text or "")):
            words = seg.split()
            if not words or not any(ch.isalpha() for ch in seg):
                continue
            for start in range(0, len(words), 40):
                by_lang[code].append(" ".join(words[start:start + 40]))

    log("papluca …")
    for split in ("train", "validation"):
        for row in load_dataset("papluca/language-identification", split=split):
            add(row["labels"], row["text"], "papluca")
    log("oasst2 …")
    for row in load_dataset("OpenAssistant/oasst2", split="train"):
        if not row.get("deleted"):
            add(row["lang"], row["text"][:1500], "oasst")
    log("massive …")
    for code in ["af", "am", "ar", "az", "bn", "cy", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi",
                 "hu", "hy", "id", "is", "it", "ja", "jv", "ka", "km", "kn", "ko", "lv", "ml", "mn", "ms", "my",
                 "nb", "nl", "pl", "pt", "ro", "ru", "sl", "sq", "sv", "sw", "ta", "te", "th", "tl", "tr", "ur",
                 "vi", "zh-CN", "zh-TW"]:
        try:
            rows = load_dataset("mteb/amazon_massive_intent", code, split="train")
        except Exception as error:
            log("  massive", code, "unavailable:", str(error)[:80])
            continue
        for row in rows.select(range(min(4000, len(rows)))):
            add(code, row["text"], "massive")
    log("wildchat (scripts other than Latin) …")
    wild = load_dataset("allenai/WildChat-1M", split="train", streaming=True).shuffle(seed=SEED, buffer_size=20000)
    taken: Counter = Counter()
    seen = 0
    for row in wild:
        seen += 1
        code = WILD_LANGS.get(row.get("language"))
        if code and taken[code] < 3000:
            for message in row["conversation"][:4]:
                if message["role"] == "user" and message.get("content"):
                    for seg in segments(judged_part(message["content"][:1500])):
                        if non_latin_share(seg) >= 0.6:
                            by_lang[code].append(" ".join(seg.split()[:40]))
                            taken[code] += 1
        if seen > 120000 or all(taken[c] >= 3000 for c in WILD_LANGS.values()):
            break
    log("English from the labelled message pool …")
    for row in read_jsonl(DATA / "pool_messages.jsonl"):
        add("en", row["text"], "pool")

    samples: list[tuple[str, str]] = []
    for code, texts in by_lang.items():
        texts = list(dict.fromkeys(texts))
        rng.shuffle(texts)
        texts = texts[:70000 if code == "en" else 7000]
        if len(texts) < 600:
            code = "xx"  # too little to learn it on its own: still not English
        for text in texts:
            samples.append((code, text))
            words = text.split()
            if code != "en" and non_latin_share(text) < 0.3 and rng.random() < 0.35:
                samples.append((code, deaccent(text)))  # typed without accents
            if code != "en" and len(words) >= 4 and rng.random() < 0.2:
                at = rng.randrange(len(words))
                samples.append((code, " ".join(words[:at] + [rng.choice(TECH)] + words[at:])))
            if len(words) >= 3 and rng.random() < 0.3:
                size = rng.randint(1, 4)
                at = rng.randrange(max(1, len(words) - size + 1))
                samples.append((code, " ".join(words[at:at + size])))
    # English around a foreign word or name stays English.
    foreign = [text for code, text in samples if code not in ("en", "xx") and len(text.split()) <= 3]
    english = [text for code, text in samples if code == "en" and len(text.split()) >= 5]
    for _ in range(12000):
        words = rng.choice(english).split()
        at = rng.randrange(len(words))
        samples.append(("en", " ".join(words[:at] + rng.choice(foreign).split()[:1] + words[at:])))
    rng.shuffle(samples)
    write_jsonl(DATA / "lang.jsonl", [{"lang": code, "text": text} for code, text in samples])
    log("language samples:", len(samples), Counter(code for code, _ in samples).most_common())


# ---------------------------------------------------------------- features

_engine: PyEngine | None = None


def engine() -> PyEngine:
    global _engine
    if _engine is None:
        _engine = PyEngine(BLOB.read_bytes())
    return _engine


def feats_text(args):
    text, flags, buckets = args
    raw = text.encode("utf-8")
    eng = engine()
    dense = eng.embed(raw)[0] if flags & F_EMB else []
    words = [w for w in raw.split(b" ") if w][:MAX_WORDS]
    hashes = [fnv(w) for w in words]
    groups = []
    if flags & F_WORDS:
        groups.append([h % buckets for h in word_features(hashes)])
    if flags & F_CHARS:
        groups.append([h % buckets for h in char_features(words, hashes)])
    return dense, groups


def feats_pair(args):
    a, b, flags, buckets = args
    ra, rb = a.encode("utf-8"), b.encode("utf-8")
    eng = engine()
    dense = eng.pair_inputs(ra, rb, bool(flags & F_EMB))
    groups = []
    if flags & F_WORDS:
        groups.append([h % buckets for h in word_features([fnv(w) for w in rb.split(b" ") if w][:MAX_WORDS])])
    if flags & F_WORDS_A:
        groups.append([h % buckets for h in word_features([fnv(w) for w in ra.split(b" ") if w][:MAX_WORDS])])
    return dense, groups


def feats_tokens(args):
    text, flags, buckets = args
    raw = text.encode("utf-8")
    eng = engine()
    words = [w for w in raw.split(b" ") if w][:MAX_WORDS][:MAX_TOKENS]
    out = []
    for word, row in zip(words, token_features(words)):
        dense = eng.embed(word)[0] if flags & F_EMB else []
        out.append((dense, [[h % buckets for h in row]] if flags & F_WORDS else []))
    return out


def featurize(fn, jobs: list) -> list:
    engine()  # loaded once, shared by the forked workers
    with mp.get_context("fork").Pool(max(1, os.cpu_count() - 2)) as pool:
        return pool.map(fn, jobs, chunksize=256)


def pack(rows: list, cap: int, buckets: int):
    """dense float32 [N, D]; per group an index matrix [N, cap] padded with `buckets`."""
    dense = np.asarray([r[0] for r in rows], dtype=np.float32) if rows and rows[0][0] else np.zeros((len(rows), 0), np.float32)
    groups = []
    for g in range(len(rows[0][1]) if rows else 0):
        mat = np.full((len(rows), cap), buckets, dtype=np.int32)
        for i, r in enumerate(rows):
            ids = r[1][g][:cap]
            mat[i, :len(ids)] = ids
        groups.append(mat)
    return dense, groups


# ---------------------------------------------------------------- the network

def train_head(tag: str, rows: list, targets: np.ndarray, sizes: list[int], dev_share: float = 0.08,
               epochs: int = 8, weights: list | None = None) -> dict:
    """targets [N, len(sizes)]: one class index per output group (-1 = no label)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    spec = SPECS[tag]
    torch.manual_seed(SEED)
    buckets, dim, hidden = spec["buckets"], spec["dim"], spec["hidden"]
    dense, groups = pack(rows, spec["cap"], buckets)
    n = len(rows)
    order = np.random.RandomState(SEED).permutation(n)
    cut = int(n * dev_share)
    dev, tr = order[:cut], order[cut:]
    classes = sum(sizes)
    inputs = dense.shape[1] + dim * len(groups)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Parameter(torch.randn(buckets, dim) * 0.25)
            self.l1 = nn.Linear(inputs, hidden)
            self.l2 = nn.Linear(hidden, classes)
            self.drop = nn.Dropout(0.1)

        def quantized(self):
            t = self.table.clamp(-1, 1)
            q = t + (torch.round(t * 127) / 127 - t).detach()
            return torch.cat([q, torch.zeros(1, dim, device=q.device)])

        def forward(self, x, gs):
            q = self.quantized()
            parts = [x]
            for g in gs:
                count = (g < buckets).sum(1, keepdim=True).clamp(min=1).float()
                parts.append(q[g].sum(1) / count)
            return self.l2(self.drop(F.relu(self.l1(torch.cat(parts, 1)))))

    net = Net().to(device)
    opt = torch.optim.Adam([
        {"params": [net.table], "lr": 1e-2},
        {"params": list(net.l1.parameters()) + list(net.l2.parameters()), "lr": 2e-3, "weight_decay": 1e-5},
    ])
    X = torch.from_numpy(dense)
    G = [torch.from_numpy(g) for g in groups]
    Y = torch.from_numpy(np.asarray(targets, dtype=np.int64))

    def batch_of(idx):
        b = torch.from_numpy(np.asarray(idx, dtype=np.int64))
        return X[b].to(device), [g[b].to(device).long() for g in G], Y[b].to(device)
    cw = [torch.tensor(w, device=device, dtype=torch.float32) if w is not None else None
          for w in (weights or [None] * len(sizes))]

    def loss_of(logits, y):
        total, at = 0.0, 0
        for k, size in enumerate(sizes):
            part = logits[:, at:at + size]
            at += size
            mask = y[:, k] >= 0
            if mask.any():
                total = total + F.cross_entropy(part[mask], y[mask, k], weight=cw[k], label_smoothing=0.05)
        return total

    def evaluate(idx):
        net.eval()
        correct = np.zeros(len(sizes))
        seen = np.zeros(len(sizes))
        with torch.no_grad():
            for s in range(0, len(idx), 4096):
                x, gs, y = batch_of(idx[s:s + 4096])
                logits = net(x, gs)
                at = 0
                for k, size in enumerate(sizes):
                    pred = logits[:, at:at + size].argmax(1)
                    at += size
                    mask = y[:, k] >= 0
                    correct[k] += (pred[mask] == y[:, k][mask]).sum().item()
                    seen[k] += mask.sum().item()
        net.train()
        return correct / np.maximum(seen, 1)

    def save(best: float) -> None:
        """Written at every better epoch, so a stopped run still leaves its best head."""
        table = torch.round(net.table.detach().clamp(-1, 1) * 127).to(torch.int8).cpu().numpy()
        scalars = PAIR_SCALARS if spec["kind"] == KIND_PAIR else 0
        raw = struct.pack("<7If", spec["kind"], spec["flags"], buckets, dim, hidden, classes, scalars, 1 / 127)
        body = table.tobytes()
        body += b"\0" * (-len(body) % 4)
        w = [net.l1.weight, net.l1.bias, net.l2.weight, net.l2.bias]
        section = raw + body + b"".join(p.detach().cpu().numpy().astype("<f4").tobytes() for p in w)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{tag}.bin").write_bytes(section)
        log(f"  {tag}: {len(section)} bytes, best dev {round(best, 4)}")

    best = -1.0
    batch = 512
    for epoch in range(epochs):
        started = time.time()
        perm = np.random.RandomState(SEED + epoch).permutation(tr)
        for s in range(0, len(perm), batch):
            x, gs, y = batch_of(perm[s:s + batch])
            loss = loss_of(net(x, gs), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                net.table.clamp_(-1, 1)
        acc = evaluate(dev)
        log(f"  {tag} epoch {epoch + 1}: dev accuracy {np.round(acc, 3).tolist()} ({time.time() - started:.0f} s)")
        if acc.mean() > best:
            best = acc.mean()
            save(best)
    return {"dev": [int(i) for i in dev]}


# ---------------------------------------------------------------- heads

def lang() -> None:
    rows = read_jsonl(DATA / "lang.jsonl")
    per = int(os.environ.get("KS_LANG_PER", 0))  # at most this many samples per language (0: all)
    if per:
        rng = random.Random(SEED)
        rng.shuffle(rows)
        seen: Counter = Counter()
        rows = [r for r in rows if (seen.update([r["lang"]]) or seen[r["lang"]]) <= (per * 8 if r["lang"] == "en" else per)]
    codes = sorted({r["lang"] for r in rows}, key=lambda c: (c != "en", c))
    index = {c: i for i, c in enumerate(codes)}
    spec = SPECS["Hlng"]
    texts = [lang_text(r["text"]) for r in rows]
    feats = featurize(feats_text, [(t, spec["flags"], spec["buckets"]) for t in texts])
    keep = [i for i, t in enumerate(texts) if t]
    y = np.asarray([[index[rows[i]["lang"]]] for i in keep])
    # English that is turned back is the costly mistake: English counts double.
    weights = [[2.0 if c == "en" else 1.0 for c in codes]]
    split = train_head("Hlng", [feats[i] for i in keep], y, [len(codes)], dev_share=0.05,
                       epochs=int(os.environ.get("KS_LANG_EPOCHS", 6)), weights=weights)
    (OUT / "Hlng.json").write_text(json.dumps({"languages": codes}))
    write_jsonl(OUT / "Hlng.dev.jsonl", [rows[keep[i]] for i in split["dev"][:4000]])


def labelled_messages() -> list[dict]:
    rows = read_jsonl(DATA / "labeled_messages.jsonl")
    return [r for r in rows if normalize(r["text"])]


def msg() -> None:
    rows = labelled_messages()
    spec = SPECS["Hmsg"]
    feats = featurize(feats_text, [(normalize(r["text"]), spec["flags"], spec["buckets"]) for r in rows])
    y = np.asarray([[labels.index(r[name]) for name, labels in GROUPS] for r in rows])
    counts = [Counter(y[:, k]) for k in range(len(GROUPS))]
    weights = [[min(4.0, (len(y) / (len(labels) * max(1, counts[k][c]))) ** 0.5) for c in range(len(labels))]
               for k, (_, labels) in enumerate(GROUPS)]
    split = train_head("Hmsg", feats, y, [len(labels) for _, labels in GROUPS], epochs=10, weights=weights)
    write_jsonl(OUT / "Hmsg.dev.jsonl", [rows[i] for i in split["dev"]])


def bio(words: list[str], phrases: dict[str, list[str]]) -> list[int]:
    tags = [0] * len(words)
    for kind in ("KEY", "ASK", "TIME"):  # later kinds win where they overlap
        for phrase in phrases.get(kind, []):
            target = normalize(phrase).split()
            if not target:
                continue
            for start in range(len(words) - len(target) + 1):
                if words[start:start + len(target)] == target:
                    tags[start] = TAGS.index(f"B-{kind}")
                    for k in range(start + 1, start + len(target)):
                        tags[k] = TAGS.index(f"I-{kind}")
                    break
    return tags


def tag() -> None:
    rows = labelled_messages()
    spec = SPECS["Htag"]
    texts = [normalize(r["text"]) for r in rows]
    feats = featurize(feats_tokens, [(t, spec["flags"], spec["buckets"]) for t in texts])
    flat, y = [], []
    for r, t, tokens in zip(rows, texts, feats):
        words = t.split()[:MAX_TOKENS]
        tags = bio(words, {"TIME": r.get("time_phrases", []), "ASK": r.get("asks", []),
                           "KEY": r.get("keyphrases", [])})
        for tok, label in zip(tokens, tags):
            flat.append(tok)
            y.append([label])
    y = np.asarray(y)
    log("  token tags:", Counter(TAGS[v] for v in y[:, 0]).most_common())
    train_head("Htag", flat, y, [len(TAGS)], epochs=6, weights=[[0.6] + [1.5] * (len(TAGS) - 1)])


def shift_before(row: dict) -> str:
    return (row["prev"][:600] + " " + row.get("reply", "")[:300]).strip()


def shift() -> None:
    rows = read_jsonl(DATA / "labeled_shift.jsonl")
    spec = SPECS["Hshf"]
    feats = featurize(feats_pair, [(normalize(shift_before(r)), normalize(r["next"]), spec["flags"], spec["buckets"])
                                   for r in rows])
    y = np.asarray([[SHIFT.index(r["label"])] for r in rows])
    log("  shift labels:", Counter(r["label"] for r in rows).most_common())
    split = train_head("Hshf", feats, y, [len(SHIFT)], epochs=10)
    write_jsonl(OUT / "Hshf.dev.jsonl", [rows[i] for i in split["dev"]])


def rel() -> None:
    rows = read_jsonl(DATA / "labeled_relate.jsonl")
    spec = SPECS["Hrel"]
    feats = featurize(feats_pair, [(normalize(r["item"]), normalize(r["message"]), spec["flags"], spec["buckets"])
                                   for r in rows])
    y = np.asarray([[1 if r["label"] == "yes" else 0] for r in rows])
    log("  relate labels:", Counter(r["label"] for r in rows).most_common())
    split = train_head("Hrel", feats, y, [2], epochs=10)
    write_jsonl(OUT / "Hrel.dev.jsonl", [rows[i] for i in split["dev"]])


# ---------------------------------------------------------------- export and check

def export() -> None:
    old = blobfile.unpack(BLOB.read_bytes())
    sections = {tag: bytes(data) for tag, data in old.items() if not tag.startswith("H")}
    meta = json.loads(bytes(old["META"])) if "META" in old else {}
    languages = json.loads((OUT / "Hlng.json").read_text())["languages"]
    meta["helpers"] = {
        "languages": languages,
        "language_names": {c: LANG_NAMES.get(c, "") for c in languages},
        "thresholds": {"english_share": 0.3, "english_segment": 0.8, "english_mixed": 0.15, "language_named": 0.6},
        "trained_on": "teacher labels (Qwen3.8-27B) of open chat datasets; see scripts/helpers",
    }
    for tag in ("Hlng", "Hmsg", "Htag", "Hshf", "Hrel"):
        sections[tag] = (OUT / f"{tag}.bin").read_bytes()
    sections["META"] = json.dumps(meta, sort_keys=True).encode("utf-8")
    data = blobfile.pack(sections)
    BLOB.write_bytes(data)
    log("wrote", BLOB, len(data), "bytes")


def check() -> None:
    """Accuracy of the written file, run by the Python twin exactly as Kiwibro runs it."""
    import importlib

    import kiwibro.sense as sense_pkg

    importlib.reload(sense_pkg)
    sense = sense_pkg.Sense(BLOB.parent, engine="python")
    helpers = sense.helpers
    log("heads:", helpers.tags)
    dev = read_jsonl(OUT / "Hmsg.dev.jsonl")[:1500]
    hits = Counter()
    for row in dev:
        reading = helpers.read(row["text"])
        for name, _ in GROUPS:
            hits[name] += getattr(reading, name) == row[name]
    log("message (teacher agreement):", {k: round(v / len(dev), 3) for k, v in hits.items()})
    web_yes = lambda w: w in ("asked", "live", "lookup")  # noqa: E731
    tp = sum(1 for r in dev if web_yes(r["web"]) and web_yes(helpers.read(r["text"]).web))
    pos = sum(1 for r in dev if web_yes(r["web"]))
    pred = sum(1 for r in dev if web_yes(helpers.read(r["text"]).web))
    log(f"web search needed: recall {tp / max(1, pos):.3f} precision {tp / max(1, pred):.3f}")
    keep_yes = lambda k: k in ("fact", "preference")  # noqa: E731
    tp = sum(1 for r in dev if keep_yes(r["keep"]) and keep_yes(helpers.read(r["text"]).keep))
    pos = sum(1 for r in dev if keep_yes(r["keep"]))
    pred = sum(1 for r in dev if keep_yes(helpers.read(r["text"]).keep))
    log(f"worth keeping: recall {tp / max(1, pos):.3f} precision {tp / max(1, pred):.3f}")
    lang_dev = read_jsonl(OUT / "Hlng.dev.jsonl")
    wrong_en = sum(1 for r in lang_dev if r["lang"] == "en" and not helpers.language(r["text"]).english)
    caught = sum(1 for r in lang_dev if r["lang"] != "en" and len(r["text"].split()) >= 3
                 and not helpers.language(r["text"]).english)
    foreign = sum(1 for r in lang_dev if r["lang"] != "en" and len(r["text"].split()) >= 3)
    english = sum(1 for r in lang_dev if r["lang"] == "en")
    log(f"language: English turned back {wrong_en}/{english}; foreign (3+ words) caught {caught}/{foreign}")
    shift_dev = read_jsonl(OUT / "Hshf.dev.jsonl")[:1500]
    log("shift agreement:", round(sum(helpers.shift(shift_before(r), r["next"])[0] == r["label"]
                                      for r in shift_dev) / len(shift_dev), 3))
    rel_dev = read_jsonl(OUT / "Hrel.dev.jsonl")[:1500]
    log("relate agreement:", round(sum((helpers.relates(r["message"], r["item"]) >= 0.5) == (r["label"] == "yes")
                                       for r in rel_dev) / len(rel_dev), 3))


STAGES = {"langdata": langdata, "lang": lang, "msg": msg, "tag": tag, "shift": shift, "rel": rel,
          "export": export, "check": check}


def main() -> None:
    for name in sys.argv[1:] or list(STAGES):
        log(f"[{name}] …")
        STAGES[name]()


if __name__ == "__main__":
    main()
