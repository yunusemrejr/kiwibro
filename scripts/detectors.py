#!/usr/bin/env python3
"""How well Kiwibro reads a complaint about its last answer and a request for news.

Runs both readers over sets of made-up chat messages (tests/data/messages/*.json) and counts
what was caught, what was missed and what was read wrongly. A wrong reading matters most when
Kiwibro would act on it by itself, so those are counted apart: a normal message that would
start a new search of the request before.

    python3 scripts/detectors.py                 # every set
    python3 scripts/detectors.py set3.json -v    # one set, with the messages read wrongly
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kiwibro import plan, reactions, readings, research  # noqa: E402

SETS = ROOT / "tests" / "data" / "messages"


def read_reactions(cases: list[dict], helpers: bool) -> dict:
    out = {"complaints": 0, "caught": 0, "normal": 0, "false_alarms": 0, "would_search_again": 0, "wrong": []}
    for case in cases:
        earlier = [{"role": "user", "content": case["asked"]}, {"role": "assistant", "content": case["answered"]}]
        reading = readings.read(case["message"], f"{case['asked']} {case['answered']}") if helpers else None
        found = reactions.read(case["message"], reading, earlier)
        complaint = case["label"] == "reaction"
        out["complaints" if complaint else "normal"] += 1
        if complaint:
            out["caught"] += found is not None
            if found is None:
                out["wrong"].append(("missed", case["message"]))
            continue
        if found is None:
            continue
        out["false_alarms"] += 1
        # agent._search_first: a reaction with a question of its own is answered as asked
        acts = not found.asks and (found.wants_web or found.fresher or bool(case.get("searched"))
                                   or bool(plan.search_request(found.request)[0]) or research.asks_for_news(found.request))
        out["would_search_again"] += acts
        out["wrong"].append(("searches again" if acts else "note only", case["message"]))
    return out


def read_news(cases: list[dict], helpers: bool) -> dict:
    out = {"news": 0, "caught": 0, "searched_anyway": 0, "other": 0, "false_alarms": 0, "wrong": []}
    for case in cases:
        said = research.asks_for_news(case["message"])
        if case["news"]:
            out["news"] += 1
            out["caught"] += said
            if not said:  # no dated bulletin, but is the web asked at all?
                reading = readings.read(case["message"]) if helpers else readings.TurnReading()
                searched = bool(plan.search_request(case["message"])[0]) or (helpers and readings.wants_search(reading))
                out["searched_anyway"] += searched
                out["wrong"].append(("plain search" if searched else "missed", case["message"]))
        else:
            out["other"] += 1
            out["false_alarms"] += said
            if said:
                out["wrong"].append(("false alarm", case["message"]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sets", nargs="*", help="set files (default: every set in tests/data/messages)")
    parser.add_argument("-v", "--verbose", action="store_true", help="list the messages read wrongly")
    args = parser.parse_args()
    paths = [Path(name) if Path(name).exists() else SETS / name for name in args.sets] or sorted(SETS.glob("*.json"))
    have_helpers = readings.helpers() is not None
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"{path.name}")
        for helpers in ([True, False] if have_helpers else [False]):
            r = read_reactions(data.get("reactions", []), helpers)
            n = read_news(data.get("news", []), helpers)
            print(f"  helpers {'on ' if helpers else 'off'}  complaints caught {r['caught']}/{r['complaints']}   "
                  f"false alarms {r['false_alarms']}/{r['normal']} (would search again: {r['would_search_again']})   "
                  f"news caught {n['caught']}/{n['news']} (+{n['searched_anyway']} plain search)   "
                  f"news false alarms {n['false_alarms']}/{n['other']}")
            if args.verbose and helpers == have_helpers:
                for kind, message in [*r["wrong"], *n["wrong"]]:
                    print(f"      {kind:15} {message[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
