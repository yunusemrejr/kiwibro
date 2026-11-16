#!/usr/bin/env python3
"""Report asterisk-emphasis compatibility alongside a frozen action replay's strict score."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.mind.agent_corpus import score


LEGACY_PREVIEW_CHARS = 1200
# Code spans are left as written. Only paired asterisk delimiters around prose are removed.
CODE_OR_EMPHASIS = re.compile(
    r"(?P<code>(?<!\x60)(?P<ticks>\x60+)(?!\x60).*?(?<!\x60)(?P=ticks)(?!\x60))"
    r"|(?<![\\*])(?P<marks>\*{1,2})(?P<text>[^\s*](?:[^*\n]*?[^\s*])?)(?P=marks)(?!\*)",
    re.DOTALL,
)


def without_emphasis(answer: str) -> str:
    return CODE_OR_EMPHASIS.sub(lambda match: match["text"] or match["code"], answer)


def context_rejection(error: str | None) -> dict[str, int] | None:
    if not error or not error.startswith("HTTP 400: "):
        return None
    try:
        problem = json.loads(error.removeprefix("HTTP 400: "))["error"]
        if problem["type"] == "exceed_context_size_error":
            return {"prompt_tokens": int(problem["n_prompt_tokens"]), "context_tokens": int(problem["n_ctx"])}
    except (KeyError, TypeError, ValueError):
        pass
    return None


def audit(path: Path) -> tuple[dict, tuple]:
    raw = path.read_bytes()
    metadata, *rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    ids = {row["case"] for row in rows}
    if not rows or len(ids) != len(rows) or len(rows) != metadata["count"]:
        raise ValueError(f"Replay has missing or duplicate cases: {path}")
    report = {
        "model": metadata["model"], "probes_sha256": metadata["probes_sha256"],
        "replay_sha256": hashlib.sha256(raw).hexdigest(),
        "count": len(rows), "strict_passed": 0, "asterisk_emphasis_compatible": 0,
        "emphasis_gains": [], "emphasis_losses": [], "context_rejections": {},
        "possibly_truncated_answers": [],
    }
    for row in rows:
        expected = row["expected"]
        answer = row["answer"]
        actual = {"content": answer, "tool_calls": [
            {"function": {"name": name, "arguments": arguments}} for name, arguments in row["calls"]
        ]}
        complete = row["error"] is None and row["finish_reason"] != "length"
        strict = complete and score(expected, actual)
        if strict != row["passed"]:
            raise ValueError(f"Stored strict outcome disagrees with the oracle for {row['case']}")
        compatible = complete and score(expected, {**actual, "content": without_emphasis(answer)})
        report["strict_passed"] += strict
        report["asterisk_emphasis_compatible"] += compatible
        if strict != compatible:
            report["emphasis_gains" if compatible else "emphasis_losses"].append(row["case"])
        if context := context_rejection(row["error"]):
            report["context_rejections"][row["case"]] = context
        if metadata.get("answer_capture") != "full" and len(answer) == LEGACY_PREVIEW_CHARS:
            report["possibly_truncated_answers"].append(row["case"])
    provenance = (metadata["probes_sha256"], metadata["seed"], metadata["thinking"],
                  {row["case"]: row["expected"] for row in rows})
    return report, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path, nargs="+")
    args = parser.parse_args()
    baseline, provenance = audit(args.replay[0])
    reports = [baseline]
    for path in args.replay[1:]:
        report, paired = audit(path)
        if paired != provenance:
            parser.error("Reports must share the frozen probes, sample, thinking setting and exact oracles.")
        reports.append(report)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
