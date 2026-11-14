#!/usr/bin/env python3
"""Check a Kiwi Mind pack's engine against the model it was exported from.

`train.py parity` writes parity.json: held-back prompts and the token the trained model picks next on each.
This runs the pack's WebAssembly engine on the same prompts the way Kiwibro does (a session turn by turn, each
turn under its own skill) and compares the picks.

    KIWIBRO_DATA_DIR=<scratch> python3 scripts/mind/parity.py --data-dir <folder that holds mind/> --cases parity.json

A different pick where the model's two best tokens were almost level (a margin under 1) is rounding. Anything
else means the engine does not compute what was trained.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kiwibro.mind.client import MindClient  # noqa: E402
from kiwibro.mind.host import MindHost  # noqa: E402
from kiwibro.mind.pack import Pack  # noqa: E402

LEVEL = 1.0  # below this margin between the model's two best tokens a different pick is rounding
END = "<|im_end|>"


def pick(client: MindClient, case: dict) -> tuple[str, dict]:
    """The engine's next piece of text for a case, and what it reports about the request."""
    *earlier, last = case["steps"]
    for step in earlier:
        client.warm(step["text"], "parity", step["skill"])
    text, done = "", {}
    for event in client.generate({"prompt": last["text"], "slot": "parity", "skill": last["skill"],
                                  "max_tokens": 1, "temperature": 0.0}):
        if event.get("done"):
            done = event
            break
        text += str(event.get("text") or "")
    return text, done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="a folder with the pack in mind/ (not the one Kiwibro runs on)")
    parser.add_argument("--cases", required=True)
    args = parser.parse_args()
    cases = json.loads(Path(args.cases).read_text())["cases"]
    host = MindHost(Pack(args.data_dir))
    client = MindClient(host)
    same: dict[str, int] = {}
    total: dict[str, int] = {}
    level, wrong = [], []
    try:
        for number, case in enumerate(cases):
            text, done = pick(client, case)
            read = int((done.get("usage") or {}).get("prompt_tokens") or 0)
            if read != case["tokens"]:
                wrong.append(f"case {number}: the engine read {read} tokens, the model {case['tokens']}")
                continue
            total[case["kind"]] = total.get(case["kind"], 0) + 1
            wanted = "" if case["next_text"] == END else case["next_text"]
            if text == wanted:
                same[case["kind"]] = same.get(case["kind"], 0) + 1
            elif case["margin"] < LEVEL:
                level.append(round(case["margin"], 2))
            else:
                wrong.append(f"case {number} ({case['kind']}): engine {text!r}, model {case['next_text']!r}, "
                             f"margin {case['margin']:.2f}")
    finally:
        host.stop()
    print("same pick: " + ", ".join(f"{kind} {same.get(kind, 0)} of {count}" for kind, count in sorted(total.items())))
    if level:
        print(f"different where the two best were level (margins {level})")
    for line in wrong:
        print("WRONG " + line)
    return 1 if wrong or not total else 0


if __name__ == "__main__":
    raise SystemExit(main())
