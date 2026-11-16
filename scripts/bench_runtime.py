"""Benchmark workspace search, recall and prompt tracking on synthetic local data.

From the checkout root, run `python3 -m scripts.bench_runtime --baseline REVISION`.
Omit the baseline to time only the current checkout.
"""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType

from kiwibro import context, inference, sandbox, sense, terms


def baseline_module(revision, module):
    path = Path(module.__file__).relative_to(Path.cwd())
    source = subprocess.check_output(["git", "show", f"{revision}:{path}"])
    baseline = ModuleType(f"{module.__name__}_baseline")
    baseline.__file__ = module.__file__
    baseline.__package__ = module.__package__
    sys.modules[baseline.__name__] = baseline
    exec(compile(source, str(path), "exec"), baseline.__dict__)
    return baseline


def measure(label, trials, current, baseline):
    methods = [("current", current)]
    result = current()
    if baseline:
        assert baseline() == result, f"{label} produced different results"
        methods.insert(0, ("baseline", baseline))
    samples = {name: [] for name, _ in methods}
    for trial in range(trials):
        for name, call in methods if trial % 2 else reversed(methods):
            started = time.perf_counter()
            call()
            samples[name].append(1000 * (time.perf_counter() - started))
    for name, runs in samples.items():
        p95 = sorted(runs)[math.ceil(0.95 * len(runs)) - 1]
        print(f"{label} ({name}): median {statistics.median(runs):.2f} ms, p95 {p95:.2f} ms (n={trials})")


def prompt_rounds():
    source = "\n".join(f"{index:03d}  assert checkout.status({index}) == 'ready'" for index in range(80))
    schema = [{
        "type": "function",
        "function": {"name": "read_file", "description": "Read source text.", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
        }},
    }]

    def read(index):
        call_id = f"read_{index}"
        return [
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": "read_file", "arguments": {"path": f"tests/test_stage_{index}.py"}},
            }]},
            {"role": "tool", "tool_call_id": call_id, "content": source},
        ]

    history = []
    for index in range(20):
        history += [
            {"role": "user", "content": f"What does checkout stage {index} verify?"},
            *read(index),
            {"role": "assistant", "content": "The tests check that the checkout reports readiness."},
        ]
    history.append({"role": "user", "content": "Review and compare the next eight checkout stages."})
    messages = context.build_llm_messages("You are reviewing an internal test repository.", history)
    rounds = [messages]
    for index in range(20, 28):
        messages = [*messages, *context.build_llm_messages("", read(index))]
        rounds.append(messages)

    def run(module, memo=None):
        tracker = module.PrefixTracker()
        options = {"chat_template_kwargs": {"enable_thinking": False}}
        extra = {"memo": memo} if memo is not None else {}
        return [tracker.observe("native", request, schema, options, **extra) for request in rounds]

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", metavar="REVISION")
    parser.add_argument("--trials", type=int, default=25)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")

    with tempfile.TemporaryDirectory(prefix="kiwibro-bench-") as folder:
        root = Path(folder)
        for directory in range(50):
            package = root / f"package_{directory:02d}"
            package.mkdir()
            for filename in range(70):
                marker = "service_identity_override" if (directory, filename) == (1, 5) else "enabled"
                (package / f"provider_{filename:02d}.py").write_text(
                    f"def decorated_handler(event):\n    {marker} = True\n    return event\n", encoding="utf-8"
                )
        workspace = sandbox.Workspace(root)
        old_workspace = baseline_module(args.baseline, sandbox).Workspace(root) if args.baseline else None
        measure(
            "Workspace search, 3,500 files", args.trials,
            lambda: workspace.grep_files("service_identity_override", glob="*.py"),
            (lambda: old_workspace.grep_files("service_identity_override", glob="*.py")) if old_workspace else None,
        )

    passages = (
        "The job runner writes each event to a ledger and rotates the log on Friday.",
        "We bought more apples and cinnamon for a pie to bring to the garden party.",
        "The city changed the last bus time following a review of weekend schedules.",
        "Use a separate branch to review the error handling patch before merging changes.",
        "Rosa has a small herb garden with basil and rosemary growing near the window.",
        "The workshop has a generator and a stack of old paperback books near the entrance.",
    )
    messages = [f"Message {index}: {passages[index % len(passages)]}" for index in range(240)]
    messages[178] = "Somebody left an old paperback and Maria's car keys by the front door."
    reader = sense.Sense()
    old_reader = baseline_module(args.baseline, sense).Sense() if args.baseline else None

    def nearest(engine):
        return engine.nearest(terms.gist("where are the keys to the vehicle"), [terms.gist(text) for text in messages])

    assert nearest(reader)[0][0] == 178
    measure(
        f"Warm semantic recall, 240 messages ({reader.engine})", args.trials,
        lambda: nearest(reader), (lambda: nearest(old_reader)) if old_reader else None,
    )
    run_rounds = prompt_rounds()
    old_inference = baseline_module(args.baseline, inference) if args.baseline else None
    measure(
        "Native prompt tracking, eight tool rounds over 20 prior turns", args.trials,
        lambda: run_rounds(inference, inference.PrefixMemo()),
        (lambda: run_rounds(old_inference)) if old_inference else None,
    )


if __name__ == "__main__":
    main()
