#!/usr/bin/env python3
"""Run isolated KiwiBro finalist acceptance and write evidence without chat text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import site
import subprocess
import sys
import tempfile
import time

import pytest

from scripts.mind.bench_successor import MemorySamples, host_info, proc_values
from scripts.mind.tune_local import sha256

JOURNAL = "KIWIBRO_FINALIST_JOURNAL"
PLACEMENT = "KIWIBRO_ACCEPTANCE_PLACEMENT"
DIRECT_GPU = {"auto": "auto", "cpu": "off"}
EXTRACTIVE_HISTORY = "KIWIBRO_NATIVE_EXTRACTIVE_HISTORY"
SUPPLEMENT = "tests/test_finalist_acceptance.py"
ACCEPTANCE = "tests/test_successor_acceptance.py"
CASES = {
    "repo": (SUPPLEMENT, "test_finalist_reviews_then_repairs_in_same_chat", ()),
    "history": (SUPPLEMENT, "test_finalist_recovers_earlier_exact_constraint_after_long_history", ()),
    "recovery": (ACCEPTANCE, "test_stale_data_path_and_typed_argument_recovery", ()),
    "artifact": (ACCEPTANCE, "test_attachment_to_interactive_rendered_artifact",
                 ("handoff.html", "screenshots/look-handoff.png")),
    "hatch": (ACCEPTANCE, "test_builds_and_checks_an_interactive_3d_hatch",
              ("inspection-hatch.html", "screenshots/look-inspection-hatch.png")),
}
ROWS = {name: (label, assets) for label, (_, name, assets) in CASES.items()}


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if not (journal := os.environ.get(JOURNAL)) or item.name not in ROWS:
        return report
    label, assets = ROWS[item.name]
    result = {"scenario": label, "phase": report.when, "outcome": report.outcome,
              "duration_ms": round(report.duration * 1000)}
    if report.when == "call":
        values = [(key, value) for key, value in report.user_properties if key in ("turn", "evidence")]
        result["turns"] = [value for key, value in values if key == "turn"]
        result["evidence"] = [value for key, value in values if key == "evidence"]
        workspace = item.funcargs["acceptance"].workspace
        result["artifacts"] = {}
        for name in assets:
            path = workspace / name
            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(workspace.resolve()):
                result["artifacts"][name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    if report.failed:
        info = call.excinfo
        error = str(info.value) if info else ""
        result["failure"] = "turn_timeout" if error.startswith("Acceptance timeout") else report.when
        result["failure_kind"] = {
            "FixtureLookupError": "fixture_lookup",
            "FileNotFoundError": "missing_file",
            "PermissionError": "permission",
            "AssertionError": "assertion",
        }.get(info.typename if info else "", "test_failure")
    with Path(journal).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")
    return report


def children_of(pid: int) -> set[int]:
    children = set()
    for listing in (Path(f"/proc/{pid}") / "task").glob("*/children"):
        try:
            children.update(map(int, listing.read_text().split()))
        except (OSError, ValueError):
            continue
    return children


def descendants(pid: int) -> set[int]:
    pending = [pid]
    found = {pid}
    while pending:
        ids = children_of(pending.pop()) - found
        found.update(ids)
        pending.extend(ids)
    return found - {pid}


class FinalistSamples(MemorySamples):
    def __init__(self, pid: int, runtime: Path | None):
        super().__init__(pid)
        self.runtime = runtime
        self.model_samples: list[dict[str, int]] = []

    def sample(self) -> None:
        super().sample()
        if self.runtime is None:
            return
        for child in children_of(self.pid):
            try:
                executable = Path(os.readlink(f"/proc/{child}/exe"))
                if executable != self.runtime and executable.name != "llama-server":
                    continue
            except OSError:
                continue
            self.model_samples.append(proc_values(Path(f"/proc/{child}/status"), ("VmRSS", "VmSwap")))

    def report(self) -> dict:
        values = super().report()
        values["pytest_rss_mib_peak"] = values.pop("server_rss_mib_peak")
        values["pytest_swap_mib_peak"] = values.pop("server_swap_mib_peak")
        values.pop("server_rss_mib_before")
        for key, label in (("VmRSS", "native_rss_mib_peak"), ("VmSwap", "native_swap_mib_peak")):
            samples = [row[key] / 1024 for row in self.model_samples if key in row]
            values[label] = round(max(samples), 2) if samples else None
        return values


def abort_suite(process) -> None:
    # Browser and command fixtures may have their own process groups.
    handles = []
    for pid in descendants(process.pid):
        try:
            handles.append(os.pidfd_open(pid))
        except OSError:
            continue
    try:
        process.terminate()
        for fd in handles:
            try:
                signal.pidfd_send_signal(fd, signal.SIGTERM)
            except ProcessLookupError:
                continue
        try:
            process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    finally:
        for fd in handles:
            try:
                signal.pidfd_send_signal(fd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                os.close(fd)


def staged_inputs(data: Path, server: Path) -> dict:
    raw = (data / "mind" / "candidate" / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest["format"] != "native-gguf" or manifest["engine"] != "llama":
        raise ValueError("Expected a staged native GGUF pack")
    files = manifest["files"]
    return {"pack_manifest_sha256": hashlib.sha256(raw).hexdigest(), "context": int(manifest["ctx"]),
            "model_sha256": files[manifest["model"]]["sha256"], "staged_model_bytes": int(files[manifest["model"]]["bytes"]),
            "projector_sha256": files[manifest["vision"]["file"]]["sha256"], "server_sha256": sha256(server)}


def run(args, scratch: Path, report: dict) -> None:
    for folder in ("home", "tmp"):
        (scratch / folder).mkdir()
    journal = scratch / "events.jsonl"
    environment = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ}
    environment.update({
        "HOME": str(scratch / "home"), "TMPDIR": str(scratch / "tmp"),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "KIWIBRO_OFFLINE": "1", "KIWIBRO_OWN_MODELS_ONLY": "1", "KIWIBRO_OWN_RUNTIME_ONLY": "1",
        "KIWIBRO_ACCEPTANCE_TURN_TIMEOUT": str(args.turn_timeout),
        "KIWIBRO_ACCEPTANCE_HISTORY_TOKENS": str(args.history_tokens),
        "KIWIBRO_ACCEPTANCE_REASONING": args.reasoning, JOURNAL: str(journal),
    })
    if site.ENABLE_USER_SITE:
        environment["PYTHONPATH"] = site.getusersitepackages()
    if args.native_data:
        environment.update({
            "KIWIBRO_ACCEPTANCE_NATIVE_DATA": str(args.native_data),
            "KIWIBRO_LLAMA_SERVER": str(args.server),
            PLACEMENT: args.placement,
        })
        if args.extractive_history:
            environment[EXTRACTIVE_HISTORY] = "1"
    nodes = [f"{CASES[label][0]}::{CASES[label][1]}" for label in args.case]
    command = [sys.executable, "-m", "pytest", "-q", "-p", "scripts.mind.accept_finalist", *nodes]
    root = Path(__file__).resolve().parents[2]
    started = time.monotonic()
    with subprocess.Popen(command, cwd=root, env=environment, start_new_session=True,
                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) as process:
        with FinalistSamples(process.pid, args.server) as samples:
            try:
                process.wait(timeout=args.suite_timeout)
            except subprocess.TimeoutExpired:
                report["suite_timed_out"] = True
                abort_suite(process)
            except BaseException:
                abort_suite(process)
                raise
    report.update(elapsed_ms=round((time.monotonic() - started) * 1000), returncode=process.returncode,
                  resources=samples.report())
    rows = journal.read_text().splitlines() if journal.exists() else []
    if not rows and not report["suite_timed_out"]:
        report["driver_failure"] = {"phase": "pytest_initialization", "kind": "no_scenarios_reported"}
    for line in rows:
        try:
            report["phases"].append(json.loads(line))
        except ValueError:
            report["unfinished_evidence_write"] = True
    report["passed"] = process.returncode == 0 and {row["scenario"] for row in report["phases"]
                        if row["phase"] == "call" and row["outcome"] == "passed"} == set(args.case)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="New JSON file for the result.")
    parser.add_argument("--native-data", type=Path, help="Data root containing mind/candidate/manifest.json.")
    parser.add_argument("--server", type=Path, help="Verified native llama-server executable.")
    parser.add_argument("--scripted", action="store_true", help="Check the driver without a live model.")
    parser.add_argument("--placement", choices=DIRECT_GPU, default="auto",
                        help="Use the app's automatic backend selection or force CPU in its isolated settings.")
    parser.add_argument("--extractive-history", action="store_true",
                        help="Enable experimental native extractive history compaction.")
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--history-tokens", type=int, default=18000)
    parser.add_argument("--reasoning", choices=("off", "on"), default="off")
    parser.add_argument("--turn-timeout", type=int, default=300)
    parser.add_argument("--suite-timeout", type=int, default=2400)
    args = parser.parse_args()
    if args.scripted == bool(args.native_data or args.server) or (not args.scripted and not (args.native_data and args.server)):
        parser.error("Use --scripted, or provide both --native-data and --server.")
    if args.scripted and (args.placement != "auto" or args.extractive_history):
        parser.error("Backend placement and extractive history only apply to a native run.")
    if not 1000 <= args.history_tokens <= 65536 or min(args.turn_timeout, args.suite_timeout) < 1:
        parser.error("Expected a history budget of 1000–65536 and positive timeouts.")
    args.case = list(dict.fromkeys(args.case or CASES))
    if args.native_data:
        args.native_data, args.server = args.native_data.expanduser().resolve(), args.server.expanduser().resolve()
    result = {"mode": "scripted" if args.scripted else "native", "passed": False,
              "cases": args.case, "reasoning": args.reasoning, "history_tokens_target": args.history_tokens,
              "placement": None if args.scripted else args.placement,
              "extractive_history": args.extractive_history,
              "turn_timeout_s": args.turn_timeout, "suite_timeout_s": args.suite_timeout,
              "suite_timed_out": False, "phases": [], "host": host_info()}
    with args.report.open("x", encoding="utf-8") as output:
        try:
            if args.native_data:
                if not os.access(args.server, os.X_OK):
                    raise ValueError("The supplied server is not executable")
                result["inputs"] = staged_inputs(args.native_data, args.server)
            with tempfile.TemporaryDirectory(prefix="kiwibro-finalist-") as folder:
                run(args, Path(folder), result)
        except (OSError, ValueError, KeyError) as error:
            result["driver_failure"] = {"phase": "setup", "kind": type(error).__name__}
        finally:
            json.dump(result, output, indent=2)
            output.write("\n")
    print(f"{'PASS' if result['passed'] else 'FAIL'} ({result['mode']})")
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
