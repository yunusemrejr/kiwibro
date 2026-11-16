#!/usr/bin/env python3
"""Build training-only traces for grounded repository discovery and scoped checks.

    python -m scripts.mind.repo_discovery_shard OUT

Git, file, patch, and command observations come from scratch repositories.
The denied command is a simulated approval result for a command never run.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Iterator

from kiwibro.sandbox import Workspace
from kiwibro.tools import ToolContext, ToolResult, execute_tool, tool_defs, validate_args
from scripts.mind import agent_corpus as corpus
from scripts.mind.agent_supplement import check, teach

TOOLS = ("list_files", "git", "grep_files", "read_files", "read_file", "apply_patch", "run_command")
NAMES = ("Sluicegate", "Kilnworks", "Cloudpier", "Flintmill")
SEED = 78041


@contextmanager
def scratch(episode: corpus.Episode, files: dict[str, str]) -> Iterator[ToolContext]:
    with tempfile.TemporaryDirectory(prefix="kiwi-repo-discovery-") as directory:
        workspace = Workspace(Path(directory))
        for name, text in files.items():
            workspace.write_file(name, text)
        yield ToolContext(workspace, None, None, episode.id)


def observe(episode: corpus.Episode, ctx: ToolContext, name: str, *, ok: bool = True, **args) -> ToolResult:
    result = execute_tool(name, validate_args(name, args), ctx)
    if result.ok != ok:
        raise RuntimeError(f"{episode.id}: unexpected {name} outcome: {result.content}")
    content = result.content.replace(str(ctx.workspace.root) + "/", "")
    if name == "list_files":
        content = re.sub(r" \(\d{4}-\d{2}-\d{2} \d{2}:\d{2}\)", "", content)
    episode.observe(name, content, **args)
    return result


def diagnose(s: corpus.Sample, _out: Path) -> corpus.Episode:
    root = f"signalworks/{s.slug}"
    left, right = f"{root}/dispatch/tape.py", f"{root}/imports/courier.py"
    files = {
        left: '# Dispatch interface.\n\ndef normalize_probe_label(value):\n    return value.strip()\n',
        right: '# Courier import boundary.\n\ndef normalize_probe_label(value):\n    return value.strip().upper()\n',
    }
    e = corpus.Episode(s.id, "repo_grounded_diagnosis", f"{s.label}'s probe labels disagree between the dispatch "
                       "and courier tools. The function is normalize_probe_label. Find both definitions, explain "
                       "the discrepancy with file references, and leave the repository as it is.", TOOLS)
    with scratch(e, files) as ctx:
        listing = observe(e, ctx, "list_files", path="signalworks")
        assert root in listing.content
        hits = observe(e, ctx, "grep_files", pattern="def normalize_probe_label", path=root, glob="*.py", regex=False)
        assert left in hits.content and right in hits.content
        reads = observe(e, ctx, "read_files", paths=[left, right], start=3, end=4)
        assert f"=== {right} (lines 3-4 of 4) ===" in reads.content
        assert "return value.strip().upper()" in reads.content
    e.answer(f"[{left}:4]({left}) strips spaces. [{right}:4]({right}) also uppercases the value. "
             "That accounts for the different labels.", (left, right, "uppercases"))
    return teach(e, "List the known project area to get its actual checkout path.",
                 "Search for the named definition inside that checkout.",
                 "Both paths are in the search results. Read the definition and return line in one batch.",
                 "The return expressions differ in case normalization. Cite the observed lines.")


def git_changed(s: corpus.Sample, _out: Path) -> corpus.Episode:
    root = f"foundry/{s.slug}"
    rel = "production/press_limits.toml"
    path = f"{root}/{rel}"
    before = f'[press]\nfixture = "{s.marker}"\nlimit_psi = {s.value}\n'
    after = before.replace(f"limit_psi = {s.value}", f"limit_psi = {s.value + 9}")
    e = corpus.Episode(s.id, "repo_grounded_diagnosis", f"Find the {s.label} checkout under foundry and examine its "
                       "uncommitted press configuration. Give the working-copy pressure limit and file reference. "
                       "This is a review request.", TOOLS)
    with scratch(e, {path: before}) as ctx:
        repo = ctx.workspace.resolve(root)
        git_env = {key: val for key, val in os.environ.items() if not key.startswith("GIT_")}
        git_env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
        for command in (["init", "-q", "--initial-branch=fixture"], ["add", "--", rel],
                        ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                         "commit", "-qm", "seed press settings"]):
            subprocess.run(["git", "-C", str(repo), *command], env=git_env, check=True,
                           capture_output=True, timeout=10)
        ctx.workspace.write_file(path, after)
        listing = observe(e, ctx, "list_files", path="foundry")
        assert root in listing.content
        status = subprocess.run(["git", "-C", str(repo), "status", "--short", "--branch"],
                                env=git_env, check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        assert f" M {rel}" in status
        e.observe("git", f"git status in {root}:\n{status}", action="status", path=root)
        result = observe(e, ctx, "read_file", path=path, start=3, end=3)
        assert f"limit_psi = {s.value + 9}" in result.content
    e.answer(f"The working copy sets the press limit to {s.value + 9} psi at [{path}:3]({path}). "
             "Git status shows that file as modified.", (path, f"{s.value + 9} psi", "modified"))
    return teach(e, "Find the actual checkout inside the named project directory.",
                 "Read Git status in the discovered repository to learn which file changed.",
                 "Join the observed Git-relative path to its checkout root and read the current line.",
                 "Use the working-copy value from the file and report the modification status.")


def recover_path(s: corpus.Sample, _out: Path) -> corpus.Episode:
    root = f"survey/{s.slug}"
    stale = f"{root}/overview/alarms.toml"
    path = f"{root}/control/horizon/current.toml"
    content = f'[horizon]\nalarm_zone = "{s.marker}"\n'
    e = corpus.Episode(s.id, "repo_path_recovery", f"My {s.label} handoff says alarm_zone lives in {stale}. "
                       "Find the current value anywhere in that survey checkout if the old location moved.", TOOLS)
    with scratch(e, {path: content}) as ctx:
        observe(e, ctx, "read_file", path=stale, ok=False)
        hits = observe(e, ctx, "grep_files", pattern="alarm_zone", path=root, glob="*.toml", regex=False)
        assert path in hits.content
        read = observe(e, ctx, "read_file", path=path, start=2, end=2)
        assert s.marker in read.content
    e.answer(f"The handoff path is missing. The current alarm zone is {s.marker} at [{path}:2]({path}).",
             ("missing", path, s.marker))
    return teach(e, "Try the exact path supplied by the user.",
                 "The supplied file is missing. Search the known checkout for the key instead of guessing another path.",
                 "The search returned a concrete file and line. Read that location for the source value.",
                 "Report the confirmed value and cite its current path.")


def scoped_check(s: corpus.Sample, outcome: str) -> corpus.Episode:
    root = f"observatory/{s.slug}"
    filenames = {"pass": ("logic/window", "verify_window"),
                 "denied": ("rules/envelope", "check_envelope"),
                 "error": ("sampler/band", "replay_band")}
    implementation, check_name = filenames[outcome]
    verifier_name = f"{check_name}_{s.label.lower()}.py"
    module, verifier = f"{root}/{implementation}.py", f"{root}/{verifier_name}"
    command = f"python3 -B {verifier_name}"
    low = s.value
    high = low + 8
    script = (
        "# Run from this package root.\n"
        f"from {implementation.replace('/', '.')} import in_window\n"
        f"assert in_window({low}, {low}, {high})\n"
        f"assert in_window({high}, {low}, {high})\n"
        f"assert not in_window({high + 1}, {low}, {high})\n"
        + (f'print("WINDOW {s.marker} OK")\n' if outcome != "error"
           else 'print("Acquisition fixture unavailable")\nraise SystemExit(23)\n')
    )
    files = {module: '# Closed interval for the sweep.\n\ndef in_window(value, low, high):\n    return low <= value < high\n',
             verifier: script}
    requests = {
        "pass": f"{s.label}'s sweep linter should make in_window inclusive at both ends. Find the implementation "
                f"under observatory and patch it. Run exactly `{command}` from that checkout.",
        "denied": f"The {s.label} envelope drops samples on its upper bound. Fix in_window in the checkout under "
                  f"observatory. Request permission for `{command}` in that checkout and report whether it ran.",
        "error": f"The {s.label} sampler discards readings equal to the maximum. Find and repair in_window under "
                 f"observatory, then run `{command}` from the checkout and report its exit status.",
    }
    e = corpus.Episode(s.id, "repo_scoped_check", requests[outcome] + " That is the only command I want run.", TOOLS)
    with scratch(e, files) as ctx:
        listing = observe(e, ctx, "list_files", path="observatory")
        assert root in listing.content
        hits = observe(e, ctx, "grep_files", pattern="in_window", path=root, glob="*.py", regex=False)
        assert module in hits.content and verifier in hits.content
        observe(e, ctx, "read_files", paths=[module, verifier], start=2, end=5)
        observe(e, ctx, "apply_patch", path=module,
                edits=[{"old": "return low <= value < high", "new": "return low <= value <= high"}])
        if outcome == "denied":
            e.observe("run_command", "Approval declined. The command was not executed.", command=command, cwd=root)
        else:
            result = observe(e, ctx, "run_command", command=command, cwd=root, ok=(outcome == "pass"))
            exit_code, evidence = (0, s.marker) if outcome == "pass" else (23, "Acquisition fixture unavailable")
            assert result.meta["exit_code"] == exit_code and evidence in result.content
    edit = f"Updated [{module}]({module}) to include the upper endpoint."
    if outcome == "denied":
        e.answer(f"{edit} Approval for `{command}` in `{root}` was declined. Verification did not run.",
                 (module, root, "declined", "did not run"), ("passed", "I bypassed"))
        last = "The user declined the check. Report the completed edit and the pending verification."
    elif outcome == "error":
        e.answer(f"{edit} `{command}` in `{root}` exited 23 because the acquisition fixture is unavailable.",
                 (module, root, "exited 23", "unavailable"), ("passed", "I installed"))
        last = "The exact command exited nonzero. Report the output and leave the user's command scope intact."
    else:
        e.answer(f"{edit} `{command}` in `{root}` passed with marker {s.marker}.",
                 (module, root, "passed", s.marker))
        last = "The requested check exited zero and printed the marker. Report the verified edit."
    return teach(e, "List the project area to find the checkout before reading files.",
                 "Search the observed checkout for the exact symbol.",
                 "The hits identify both the implementation and its check. Read their relevant lines with an integer range.",
                 "The upper comparison excludes the endpoint. Change only that expression.",
                 "The patch succeeded. Run the user's exact command in the discovered checkout.",
                 last)


RECIPES = {
    "route-diagnose": diagnose, "route-git": git_changed, "route-missing": recover_path,
    "route-check-pass": lambda sample, _out: scoped_check(sample, "pass"),
    "route-check-denied": lambda sample, _out: scoped_check(sample, "denied"),
    "route-check-error": lambda sample, _out: scoped_check(sample, "error"),
}


def build(out: Path) -> dict:
    if any((out / name).exists() for name in ("heldout.jsonl", "probes.jsonl")):
        raise ValueError("Use a separate training output folder")
    out.mkdir(parents=True, exist_ok=True)
    catalog = {tool["function"]["name"]: tool for tool in tool_defs()}
    rows = []
    for index, episode in enumerate(corpus.samples(out, RECIPES, NAMES, len(NAMES), SEED)):
        row = episode.record(catalog)
        if index % 2 == 0:
            row["thinking"] = True
        else:
            for message in row["messages"]:
                message.pop("reasoning", None)
        rows.append(row)
    training = out / "train.jsonl"
    corpus.jsonl(training, rows)
    summary = check(out)
    summary["seed"] = SEED
    summary["sha256"] = hashlib.sha256(training.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    print(json.dumps(build(parser.parse_args().out), indent=2))


if __name__ == "__main__":
    main()
