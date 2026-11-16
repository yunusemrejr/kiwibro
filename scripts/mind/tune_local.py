"""Measure native llama-server profiles on a local model before choosing flags.

Example:
    python3 -m scripts.mind.tune_local --server /path/llama-server \
      --model /path/gemma-4-26B-A4B-it.gguf --mmproj /path/mmproj.gguf \
      --output /tmp/gemma-ryzen-tuning.json

The sweep uses one model, build and slot at fixed context length. It compares
physical-core CPU decode with fewer decode threads, symmetric Q8 KV, and
bounded Vulkan offload with flash attention enabled and disabled. Q4 KV and
full Vulkan are explicit opt-ins. Each server is isolated and stopped before
the next one starts. The report chooses the lowest sum of per-probe median
wall times among profiles that pass every probe, and retains the raw rows.
Linux process CPU time and page faults help spot scheduling and paging costs.
Pass --trace-vulkan /tmp/vulkan-kernels.log to profile one separate retrieval
pair with the backend's per-operation timing logger. Tracing changes runtime
cost and is excluded from profile selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import socket
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import requests

from kiwibro import gguf

from .bench_successor import MEMINFO, cases, complete, host_info, proc_values

MIB = 1024 * 1024
GPU_MARGIN_MIB = 1536
GPU_RETRY_MIB = 1024
TRACE_VARIABLES = {"GGML_VK_PERF_LOGGER", "GGML_VK_PERF_LOGGER_CONCURRENT",
                   "GGML_VK_PERF_LOGGER_FREQUENCY", "GGML_VK_SYNC_LOGGER", "GGML_VK_MEMORY_LOGGER"}


@dataclass(frozen=True)
class Profile:
    name: str
    threads: int
    batch_threads: int
    gpu_layers: int
    kv: str
    flash: str

    def options(self, device: str | None, flags: set[str]) -> list[str]:
        result = [
            "--threads", str(self.threads), "--threads-batch", str(self.batch_threads),
            "--gpu-layers", str(self.gpu_layers), "--parallel", "1",
        ]
        if "--device" in flags:
            result += ["--device", device if self.gpu_layers else "none"]
        if "--flash-attn" in flags:
            result += ["--flash-attn", self.flash]
        if {"--cache-type-k", "--cache-type-v"} <= flags:
            result += ["--cache-type-k", self.kv, "--cache-type-v", self.kv]
        return result


def cpu_cores() -> int:
    """Count physical cores in the caller's affinity, accounting for SMT."""
    affinity = os.sched_getaffinity(0)
    packages = set()
    for cpu in affinity:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            packages.add(((topology / "physical_package_id").read_text(), (topology / "core_id").read_text()))
        except OSError:
            return len(affinity)
    return len(packages)


def available_mib() -> int:
    return proc_values(MEMINFO, ("MemAvailable",)).get("MemAvailable", 0) // 1024


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as data:
        for chunk in iter(lambda: data.read(MIB), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def server_environment(server: Path) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("LLAMA_ARG_") and key != "LLAMA_API_KEY" and key not in TRACE_VARIABLES}
    directory = str(server.parent)
    if environment.get("LD_LIBRARY_PATH"):
        directory += os.pathsep + environment["LD_LIBRARY_PATH"]
    return {**environment, "LD_LIBRARY_PATH": directory}


def inspect(server: Path, environment: dict[str, str], flag: str) -> str:
    result = subprocess.run(
        [str(server), flag], cwd=server.parent, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=25, check=False,
    )
    return result.stdout


def vulkan_device(text: str) -> tuple[str | None, int]:
    for match in re.finditer(r"^\s*(Vulkan\d+): (.+) \(\d+ MiB, (\d+) MiB free\)\s*$", text, flags=re.M):
        name, description, available = match.groups()
        if "llvmpipe" not in description.lower() and "lavapipe" not in description.lower():
            return name, int(available)
    return None, 0


def partial_layers(weight_mib: int, projector_mib: int, context: int, layers: int, free_mib: int) -> int:
    """Pick a conservative explicit split for this measurement, before the backend measures allocations."""
    # Leave 128 KiB per context token as an initial KV and scratch estimate.
    budget = max(0, free_mib - projector_mib - GPU_MARGIN_MIB - context // 8)
    # GEMM buffers and layer weights need room together. The cap retains a
    # genuine partial-offload variant even on a desktop where full fits.
    return min(layers // 2, int(layers * budget / max(1, weight_mib)))


def profiles(args, flags: set[str], device: str | None, free_mib: int) -> list[Profile]:
    decode = args.threads
    fewer = max(1, (decode * 2 + 2) // 3)
    result = [Profile("cpu-f16", decode, decode, 0, "f16", "auto")]
    if decode > fewer:
        result.append(Profile("cpu-fewer-threads", fewer, decode, 0, "f16", "auto"))
    symmetric_kv = {"--cache-type-k", "--cache-type-v", "--flash-attn"} <= flags
    if symmetric_kv:
        result.append(Profile("cpu-q8", decode, decode, 0, "q8_0", "on"))
        if args.q4_kv:
            result.append(Profile("cpu-q4", decode, decode, 0, "q4_0", "on"))
    if device is None or not {"--device", "--flash-attn"} <= flags:
        return result
    weight = args.model.stat().st_size // MIB
    projector = args.mmproj.stat().st_size // MIB if args.mmproj else 0
    split = partial_layers(weight, projector, args.context, args.layers, free_mib)
    if split:
        result += [
            Profile("vulkan-partial", decode, decode, split, "f16", "on"),
            Profile("vulkan-partial-no-fa", decode, decode, split, "f16", "off"),
        ]
        if symmetric_kv:
            result.append(Profile("vulkan-partial-q8", decode, decode, split, "q8_0", "on"))
    if args.full_vulkan and free_mib >= weight + projector + GPU_MARGIN_MIB + args.context // 8:
        result.append(Profile("vulkan-full", decode, decode, args.layers + 1, "f16", "on"))
    return result


def open_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_ready(session, process, url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < deadline:
        try:
            if session.get(f"{url}/health", timeout=1).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.15)
    raise RuntimeError(f"llama-server never reported ready (exit={process.poll()})")


def stop(process) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def process_usage(pid: int) -> tuple[int, int, int] | None:
    """Process CPU ticks, minor and major page faults from Linux /proc."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[11]) + int(fields[12]), int(fields[7]), int(fields[9])
    except (OSError, IndexError, ValueError):
        return None


def usage_delta(before: tuple[int, int, int], after: tuple[int, int, int]) -> dict[str, float | int]:
    ticks, minor, major = (end - start for start, end in zip(before, after))
    return {
        "process_cpu_ms": round(ticks * 1000 / os.sysconf("SC_CLK_TCK"), 2),
        "minor_faults": minor, "major_faults": major,
    }


def summarize(rows: list[dict]) -> float | None:
    if not rows or any(not row["passed"] for row in rows):
        return None
    times = {}
    for row in rows:
        key = row["case"], row["cache_requested"]
        times.setdefault(key, []).append(row["total_ms"])
    return round(sum(statistics.median(values) for values in times.values()), 2)


def pressure() -> dict[str, str]:
    result = {}
    for resource in ("cpu", "memory", "io"):
        try:
            result[resource] = Path(f"/proc/pressure/{resource}").read_text().strip()
        except OSError:
            continue
    return result


def write_report(path: Path, report: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            json.dump(report, output, indent=2, ensure_ascii=False)
            output.write("\n")
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def measure(args, options, environment, profile, device, folder, workload, trace: Path | None = None) -> dict:
    report = {"profile": asdict(profile), "flags": options, "available_mib_before": available_mib(),
              "system_pressure_before": pressure(), "score_ms": None, "rows": []}
    if profile.gpu_layers and device:
        _, free = vulkan_device(inspect(args.server, environment, "--list-devices"))
        planned_weight = args.model.stat().st_size / MIB * min(1, profile.gpu_layers / args.layers)
        if free < planned_weight + GPU_RETRY_MIB:
            report["error"] = f"live Vulkan budget dropped to {free} MiB before this profile"
            return report

    port = open_port()
    cmd = [str(args.server), "--host", "127.0.0.1", "--port", str(port), *options]
    url = f"http://127.0.0.1:{port}"
    log = folder / f"{profile.name}.log"
    harness = SimpleNamespace(max_tokens=args.max_tokens, timeout=args.timeout, thinking="off")
    with log.open("wb") as output, requests.Session() as session:
        session.trust_env = False
        started = time.perf_counter()
        process = subprocess.Popen(cmd, cwd=args.server.parent, env=environment, stdout=output, stderr=subprocess.STDOUT)
        try:
            wait_for_ready(session, process, url, args.startup_timeout)
            report["startup_ms"] = round((time.perf_counter() - started) * 1000, 2)
            warm = session.post(f"{url}/v1/chat/completions", json={
                "model": "kiwi-tune", "messages": [{"role": "user", "content": "Say Ready."}],
                "max_tokens": 4, "temperature": 0, "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }, timeout=(5, args.timeout))
            warm.raise_for_status()
            for repetition in range(args.repeats):
                for case in workload:
                    for cache in ((False, True) if case.category == "retrieval" else (False,)):
                        before = process_usage(process.pid)
                        row = complete(session, url, "kiwi-tune", case, cache, process.pid, harness)
                        after = process_usage(process.pid)
                        if before and after:
                            row.update(usage_delta(before, after))
                        report["rows"].append({**row, "repetition": repetition + 1})
                        if row["error"]:
                            raise RuntimeError(row["error"])
            report["score_ms"] = summarize(report["rows"])
        except (RuntimeError, requests.RequestException) as error:
            report["error"] = str(error)
        finally:
            stop(process)
    report["system_pressure_after"] = pressure()
    if trace is not None:
        shutil.copyfile(log, trace)
    if report.get("error"):
        with log.open("rb") as server_log:
            server_log.seek(max(0, log.stat().st_size - 3000))
            report["log_tail"] = server_log.read().decode(errors="replace")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, type=Path, help="Existing native llama-server executable.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--layers", type=int, help="Override the GGUF transformer block count for GPU splits.")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--filler-lines", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=cpu_cores())
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--profile", action="append", help="Only benchmark these named profiles.")
    parser.add_argument("--q4-kv", action="store_true")
    parser.add_argument("--full-vulkan", action="store_true")
    parser.add_argument("--trace-vulkan", type=Path, metavar="LOG")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.trace_vulkan and args.trace_vulkan.resolve() == args.output.resolve():
        parser.error("trace log and JSON output need different paths")
    files = [args.server, args.model] + ([args.mmproj] if args.mmproj else [])
    if any(not path.expanduser().is_file() for path in files):
        parser.error("provide existing server, model, and projector files")
    if min(args.context, args.threads, args.repeats, args.max_tokens,
           args.startup_timeout, args.timeout) < 1 or args.filler_lines < 0:
        parser.error("settings need positive values, and filler lines must be nonnegative")
    args.server, args.model, *projector = [path.expanduser().resolve() for path in files]
    args.mmproj = projector[0] if projector else None
    if not os.access(args.server, os.X_OK):
        parser.error("llama-server is not executable")
    try:
        metadata = gguf.read_metadata(args.model)
    except (OSError, gguf.GGUFError) as error:
        parser.error(str(error))
    model_info = gguf.summarize(metadata)
    if args.layers is None:
        args.layers = metadata.get(f"{model_info['arch']}.block_count")
    if not isinstance(args.layers, int) or args.layers < 1:
        parser.error("GGUF transformer block count is unavailable; give --layers")
    environment = server_environment(args.server)
    flags = set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]+)", inspect(args.server, environment, "--help")))
    if not {"--model", "--parallel", "--threads-batch", "--gpu-layers"} <= flags:
        parser.error("llama-server help does not list the required native arguments")

    device, free = vulkan_device(inspect(args.server, environment, "--list-devices"))
    candidates = profiles(args, flags, device, min(free, available_mib()))
    if args.profile:
        missing = set(args.profile) - {profile.name for profile in candidates}
        if missing:
            parser.error(f"profiles unavailable in this setup: {', '.join(sorted(missing))}")
        candidates = [profile for profile in candidates if profile.name in args.profile]
    if args.trace_vulkan and not any(profile.gpu_layers for profile in candidates):
        parser.error("a Vulkan profile must be available for --trace-vulkan")
    random.Random(7530).shuffle(candidates)
    selected_cases = {"tool-symbol", "long-retrieval"}
    if args.mmproj:
        selected_cases.add("screenshot-release")
    workload = [case for case in cases(args.filler_lines) if case.name in selected_cases]

    base = ["--model", str(args.model), "--ctx-size", str(args.context), "--alias", "kiwi-tune"]
    if args.mmproj:
        base += ["--mmproj", str(args.mmproj)]
    for flag, value in (("--cache-ram", "0"), ("--fit", "off"), ("--reasoning-format", "deepseek")):
        if flag in flags:
            base += [flag, value]
    base.extend(flag for flag in ("--jinja", "--no-webui") if flag in flags)
    report = {
        "host": host_info(), "server": inspect(args.server, environment, "--version").strip(),
        "server_sha256": sha256(args.server),
        "model": {"name": args.model.name, "bytes": args.model.stat().st_size, "layers": args.layers,
                  "architecture": model_info["arch"], "quantization": model_info["quantization"]},
        "projector": {"name": args.mmproj.name, "bytes": args.mmproj.stat().st_size} if args.mmproj else None,
        "vulkan": {"device": device, "free_mib": free},
        "context": args.context, "filler_lines": args.filler_lines, "repeats": args.repeats,
        "max_tokens": args.max_tokens, "runs": [], "selected": None,
    }
    with tempfile.TemporaryDirectory(prefix="kiwi-native-tune-") as folder:
        for profile in candidates:
            print(f"Measuring {profile.name}", flush=True)
            options = base + profile.options(device, flags)
            result = measure(args, options, environment, profile, device, Path(folder), workload)
            report["runs"].append(result)
            valid = [run for run in report["runs"] if run["score_ms"] is not None]
            if valid:
                fastest = min(valid, key=lambda run: run["score_ms"])
                report["selected"] = {"profile": fastest["profile"]["name"], "score_ms": fastest["score_ms"],
                                      "flags": fastest["flags"]}
            print(f"{profile.name}: {result.get('error') or result['score_ms'] or 'probe quality failure'}", flush=True)
            write_report(args.output, report)
        if args.trace_vulkan:
            usable = [run for run in report["runs"] if run["profile"]["gpu_layers"] and not run.get("error")]
            diagnostics = {"log": str(args.trace_vulkan)}
            if usable:
                chosen = min(usable, key=lambda run: run["score_ms"] if run["score_ms"] is not None else float("inf"))
                tracing = argparse.Namespace(**{**vars(args), "repeats": 1})
                trace_env = {**environment, "GGML_VK_PERF_LOGGER": "1", "GGML_VK_PERF_LOGGER_FREQUENCY": "2"}
                diagnostics.update(measure(
                    tracing, chosen["flags"], trace_env, Profile(**chosen["profile"]), device, Path(folder),
                    [case for case in workload if case.category == "retrieval"], trace=args.trace_vulkan,
                ))
            else:
                diagnostics["error"] = "no completed Vulkan profile was available for tracing"
            report["vulkan_diagnostics"] = diagnostics
            write_report(args.output, report)


if __name__ == "__main__":
    main()
