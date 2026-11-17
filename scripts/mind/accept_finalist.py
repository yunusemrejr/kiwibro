#!/usr/bin/env python3
"""Run isolated KiwiBro finalist acceptance and write evidence without chat text.

Use --case interaction to warm one app instance and measure short chat, a file
read and a same-chat render follow-up. Adjust --interaction-repeats (default 5).
First text is the arrival of displayable assistant text in the app's SSE stream.
Failed rounds are recorded and later rounds proceed once the application is idle.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import site
import statistics
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
DIRECT_IMAGE = "KIWIBRO_NATIVE_DIRECT_IMAGE"
MANAGED_RUNTIME = "KIWIBRO_ACCEPTANCE_MANAGED_RUNTIME"
INTERACTION_REPEATS = "KIWIBRO_ACCEPTANCE_INTERACTION_REPEATS"
REPOSITORY_PROFILE = "KIWIBRO_ACCEPTANCE_REPOSITORY_PROFILE"
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
DEFAULT_CASES = tuple(CASES)
CASES["interaction"] = ("tests/test_interaction_acceptance.py", "test_warm_app_interaction", ())
ROWS = {name: (label, assets) for label, (_, name, assets) in CASES.items()}


def interaction_summary(turns: list[dict], repeats: int) -> dict:
    steps = {}
    for step in ("text", "file", "render"):
        observed = [row for row in turns if row["step"] == step]
        skipped = [row for row in observed if row.get("skip_reason")]
        attempted = [row for row in observed if not row.get("skip_reason")]
        verified = [row for row in attempted if row["verified"]]
        failures = Counter(row.get("failure_kind", "timeout" if row.get("timed_out") else "unverified")
                           for row in attempted if not row["verified"])
        warm = [row for row in verified if not (native := row.get("native_execution")) or (
            native["backend_before"] is not None and native["backend_before"] == native["backend_after"]
            and native.get("reused", True) and not native["launches"] and not native["gpu_fallbacks"]
        )]
        metrics = {}
        for key in ("first_text_ms", "first_verified_tool_ms", "first_render_ms", "done_ms"):
            samples = sorted(row[key] for row in warm if row.get(key) is not None)
            metrics[key] = {"n": len(samples), "p50": statistics.median(samples) if samples else None,
                            "p90": samples[math.ceil(len(samples) * 0.9) - 1] if samples else None}
        steps[step] = {"observed": len(observed), "attempted": len(attempted), "verified": len(verified),
                       "warm_verified": len(warm), "failed": len(attempted) - len(verified), "skipped": len(skipped),
                       "unreported": max(0, repeats - len(observed)), "failure_counts": dict(failures),
                       "skip_counts": dict(Counter(row["skip_reason"] for row in skipped)),
                       "timeouts": sum(bool(row.get("timed_out")) for row in attempted), "metrics": metrics}
    return {"planned_per_step": repeats, "p90_method": "nearest-rank", "steps": steps}


def append_journal(result: dict) -> None:
    if journal := os.environ.get(JOURNAL):
        with Path(journal).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")


def collect_journal(journal: Path, report: dict, interaction_repeats: int | None) -> None:
    partial_turns = []
    for line in journal.read_text(encoding="utf-8").splitlines() if journal.exists() else ():
        try:
            row = json.loads(line)
        except ValueError:
            report["unfinished_evidence_write"] = True
            continue
        if "interaction_turn" in row:
            partial_turns.append(row["interaction_turn"])
        else:
            report["phases"].append(row)
    if not report["phases"] and not report["suite_timed_out"]:
        report["driver_failure"] = {"phase": "pytest_initialization", "kind": "no_scenarios_reported"}
    if interaction_repeats is not None:
        turns = next((row["turns"] for row in report["phases"]
                      if row["scenario"] == "interaction" and row["phase"] == "call"), None)
        report["interaction"] = interaction_summary(turns if turns is not None else partial_turns, interaction_repeats)
        if turns is None and partial_turns:
            report["interaction"]["partial_turns"] = partial_turns


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if not os.environ.get(JOURNAL) or item.name not in ROWS:
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
            "InteractionBusy": "turn_still_active",
        }.get(info.typename if info else "", "test_failure")
    append_journal(result)
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
    checksums = {"pack_manifest_sha256": hashlib.sha256(raw).hexdigest(), "context": int(manifest["ctx"]),
                 "model_sha256": files[manifest["model"]]["sha256"],
                 "staged_model_bytes": int(files[manifest["model"]]["bytes"]),
                 "projector_sha256": files[manifest["vision"]["file"]]["sha256"], "server_sha256": sha256(server)}
    if draft := manifest.get("mtp"):
        checksums["draft_sha256"] = files[draft["file"]]["sha256"]
    return checksums


def cpu_pressure_avg10() -> float | None:
    try:
        for line in Path("/proc/pressure/cpu").read_text(encoding="utf-8").splitlines():
            if line.startswith("some "):
                for field in line.split()[1:]:
                    if field.startswith("avg10="):
                        return float(field.removeprefix("avg10="))
    except (OSError, ValueError):
        pass
    return None


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
        if args.direct_image:
            environment[DIRECT_IMAGE] = "1"
        if args.managed_runtime:
            environment[MANAGED_RUNTIME] = args.managed_runtime
    if "interaction" in args.case:
        environment[INTERACTION_REPEATS] = str(args.interaction_repeats)
    if "repo" in args.case:
        environment[REPOSITORY_PROFILE] = args.repository_profile
    nodes = [f"{CASES[label][0]}::{CASES[label][1]}" for label in args.case]
    command = [sys.executable, "-m", "pytest", "-q", "-p", "scripts.mind.accept_finalist", *nodes]
    root = Path(__file__).resolve().parents[2]
    cpu_pressure_start = cpu_pressure_avg10()
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
    resources = samples.report()
    resources["cpu_pressure_some_avg10"] = {"start": cpu_pressure_start, "finish": cpu_pressure_avg10()}
    report.update(elapsed_ms=round((time.monotonic() - started) * 1000), returncode=process.returncode, resources=resources)
    collect_journal(journal, report, args.interaction_repeats if "interaction" in args.case else None)
    passed = {row["scenario"] for row in report["phases"] if row["phase"] == "call" and row["outcome"] == "passed"}
    report["passed"] = not report["suite_timed_out"] and process.returncode == 0 and passed == set(args.case)


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
    parser.add_argument("--direct-image", action="store_true",
                        help="Test the native direct attachment prompt.")
    parser.add_argument("--managed-runtime", choices=("b12443-cpu", "b12443-vulkan"),
                        help="Install the supplied runtime directory into the isolated app's owned runtime tree.")
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--interaction-repeats", type=int, default=5, help="Warm interaction rounds (1–20).")
    parser.add_argument("--repository-profile", choices=("full", "auto"), default="full",
                        help="Tool profile for the repository diagnosis and repair.")
    parser.add_argument("--history-tokens", type=int, default=18000)
    parser.add_argument("--reasoning", choices=("off", "on"), default="off")
    parser.add_argument("--turn-timeout", type=int, default=300)
    parser.add_argument("--suite-timeout", type=int, default=2400)
    args = parser.parse_args()
    if args.scripted == bool(args.native_data or args.server) or (not args.scripted and not (args.native_data and args.server)):
        parser.error("Use --scripted, or provide both --native-data and --server.")
    if args.scripted and (args.placement != "auto" or args.extractive_history or args.direct_image or args.managed_runtime):
        parser.error("Backend placement, native prompt and managed runtime options require native data.")
    if not 1000 <= args.history_tokens <= 65536 or min(args.turn_timeout, args.suite_timeout) < 1:
        parser.error("Expected a history budget of 1000–65536 and positive timeouts.")
    if not 1 <= args.interaction_repeats <= 20:
        parser.error("Expected 1–20 interaction rounds.")
    args.case = list(dict.fromkeys(args.case or DEFAULT_CASES))
    if args.native_data:
        args.native_data, args.server = args.native_data.expanduser().resolve(), args.server.expanduser().resolve()
    result = {"mode": "scripted" if args.scripted else "native", "passed": False,
              "cases": args.case, "reasoning": args.reasoning, "history_tokens_target": args.history_tokens,
              "placement": None if args.scripted else args.placement,
              "extractive_history": args.extractive_history,
              "direct_image": args.direct_image, "managed_runtime": args.managed_runtime,
              "repository_profile": args.repository_profile if "repo" in args.case else None,
              "turn_timeout_s": args.turn_timeout, "suite_timeout_s": args.suite_timeout,
              "suite_timed_out": False, "phases": [], "host": host_info()}
    with args.report.open("x", encoding="utf-8") as output:
        try:
            if args.native_data:
                if not os.access(args.server, os.X_OK):
                    raise ValueError("The supplied server is not executable")
                if args.managed_runtime and args.server.name != "llama-server":
                    raise ValueError("The managed runtime must contain the executable llama-server")
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
