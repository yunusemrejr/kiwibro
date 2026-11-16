"""Run a repeatable Kiwi Mind successor benchmark against local llama-server.

Example from the checkout root:
    python3 -m scripts.mind.bench_successor --endpoint spark http://127.0.0.1:8081 spark-model --pid spark 12345 --output /tmp/spark-bench.json

Pass --endpoint LABEL URL MODEL again for another running server. Give MODEL as
"auto" to use the first ID from /v1/models. If authentication is required, set
KIWIBRO_BENCH_API_KEY or select another environment variable with --key-env.
Tools, screenshots and conversations are synthetic fixtures. The tool tests
score the model's proposed calls, and scripted history supplies tool results.

Follow-on product evaluation
----------------------------
Put the held-out workflow corpus through KiwiBro's real conversation API in a
fresh disposable workspace and data directory. Use the oracle-prefix probes to
identify individual decision errors, then rerun whole episodes with the
candidate's own tool results. Give models identical fixtures, tool schemas,
permissions, context limits and retry budgets. Record actual tool calls and
filesystem or database state along with each final answer.

* Build an interactive HTML 3D scene from a screenshot or description. Exercise
  Chrome at desktop and mobile sizes, capture console errors and pixels, verify
  a camera/control interaction, and compare a requested visual property against
  a reference. Check a repair round with a seeded blank canvas or broken asset.
* Search a generated multi-package repo for a planted regression, edit only
  the intended files, build and test the result, and inspect the exact Git diff.
  Use a local Git remote and local web fixtures so tool effects are replayable.
* Save and recall notebook entries and durable facts across chats. Assert exact
  SQLite state, duplicate handling, private chat exclusion, and recall of a
  correction. For optional collaboration, exercise an available helper, a
  timeout and a skipped handoff with identical permissions and task goals.
* Test math and physics against numerical or dimensional oracles. Compile C++
  fixtures and run behavioral tests. Assess software architecture by the
  delivered code's stated interface, dependency direction and failure behavior.
  Include a missing executable or offline resource and check reported evidence.

Use tool-effect assertions as the primary score. Compare successful episodes'
wall time, total prompt and generated tokens, cache reuse, retries and peak RAM
under the same backend. Report partial completion and failed-tool recovery
separately. Keep all fixtures, workspace edits, server state and test artifacts
isolated from a regular local installation.
"""

from __future__ import annotations

import argparse
import base64
import io
import ipaddress
import json
import os
import platform
import re
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
from PIL import Image, ImageDraw, ImageFont

SAMPLING_SEED = 94037

CONNECT_TIMEOUT = 5
MEMORY_SAMPLE_SECONDS = 0.2
MEMINFO = Path("/proc/meminfo")
DRM = Path("/sys/class/drm")


@dataclass
class Case:
    name: str
    category: str
    messages: list[dict[str, Any]]
    expected: str | list[tuple[str, dict[str, Any]]]
    tools: list[dict[str, Any]] | None = None


def user(text: Any) -> dict[str, Any]:
    return {"role": "user", "content": text}


def tool(name: str, description: str, properties: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name, "description": description,
            "parameters": {
                "type": "object",
                "properties": {key: {"type": kind} for key, kind in properties.items()},
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def image_part(image: Image.Image) -> dict[str, Any]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def visual_cases() -> list[Case]:
    panel = Image.new("RGB", (900, 460), "#F2F4F8")
    draw = ImageDraw.Draw(panel)
    heading = ImageFont.load_default(size=32)
    body = ImageFont.load_default(size=24)
    draw.text((58, 35), "Release dashboard", fill="#0E2335", font=heading)
    for left, component, status, code in (
        (50, "Docs", "Passed", "J4LN7"),
        (320, "Desktop", "Failed", "K9R2Q"),
        (590, "Mobile", "Passed", "M5XT8"),
    ):
        draw.rounded_rectangle((left, 118, left + 250, 352), radius=12, fill="white")
        draw.text((left + 25, 148), component, fill="#142130", font=heading)
        draw.text((left + 25, 220), status, fill="#9B2130" if status == "Failed" else "#136451", font=body)
        draw.text((left + 25, 288), code, fill="#142130", font=body)

    chart = Image.new("RGB", (740, 420), "white")
    draw = ImageDraw.Draw(chart)
    draw.text((42, 30), "Average build time", fill="#142130", font=heading)
    draw.text((42, 85), "Minutes", fill="#142130", font=body)
    for index, (label, minutes) in enumerate((("Osprey", 23), ("Finch", 46), ("Raven", 31))):
        top = 149 + 80 * index
        draw.text((42, top), label, fill="#142130", font=body)
        draw.rectangle((180, top - 2, 180 + 8 * minutes, top + 30), fill="#2760A2")
        draw.text((190 + 8 * minutes, top), str(minutes), fill="#142130", font=body)

    return [
        Case("screenshot-release", "vision", [user([
            image_part(panel),
            {"type": "text", "text": "Read the screenshot. Give only the build code of the failed component."},
        ])], r"K9R2Q"),
        Case("chart-order", "vision", [user([
            image_part(chart),
            {"type": "text", "text": "From the chart, name the builds from fastest to slowest. Give only the names."},
        ])], r"Osprey[\s,;>→-]+Raven[\s,;>→-]+Finch[.!]?"),
    ]


def long_case(filler_lines: int) -> Case:
    lines = [
        f"Service lane {index:05d}: a staging cache entry was recorded for batch {index + 417}. "
        "The support engineer verified the metrics scraper and archived its console log."
        for index in range(filler_lines)
    ]
    lines.insert(filler_lines // 5, "Reference row: the calibration token for neutral channel H7 is violet-orbit-392.")
    lines.insert(filler_lines * 3 // 4 + 1,
                 "Reference row: the calibration token for neutral channel J4 is silver-window-518.")
    catalog = "\n".join(lines)
    question = "Give the distinct calibration tokens for channels H7 and J4 in that order. Separate with one comma."
    return Case("long-retrieval", "retrieval", [user(f"{catalog}\n\n{question}")],
                r"violet-orbit-392\s*,\s*silver-window-518[.!]?")


def copy_case() -> Case:
    source = "\n".join(f"static int control_{value:02d}(void) {{ return {value * 7}; }}"
                       for value in range(1, 15))
    return Case("preserve-code-block", "decode", [user(
        "Return the C declarations verbatim. Include all 14 lines. Do not add a heading or fence.\n\n" + source
    )], source)


def cases(filler_lines: int) -> list[Case]:
    files = [
        tool("grep_files", "Find text occurrences in code files under a folder.", {"pattern": "string", "path": "string"}),
        tool("read_file", "Read the contents of a known workspace file.", {"path": "string"}),
        tool("run_command", "Run a shell command in the workspace.", {"command": "string"}),
    ]
    return [
        Case("chat-order", "chat", [user(
            "Three stages named Birch, Maple and Hazel must run exactly once each. Maple runs after Hazel, "
            "and Birch runs before Hazel. Return their names in execution order joined by ' > '."
        )], r"Birch\s*>\s*Hazel\s*>\s*Maple[.!]?"),
        Case("chat-calculation", "chat", [user(
            "A 144 KB export shrinks by 25% and then grows by 20%. What is the final size in KB? "
            "Use only the number, with a decimal point if needed."
        )], r"129\.60*"),
        Case("tool-symbol", "tools", [user(
            "Find all mentions of the exact symbol validate_invoice within src. Use the dedicated code search tool."
        )], [("grep_files", {"pattern": "validate_invoice", "path": "src"})], files),
        Case("tool-inspect-hit", "tools", [
            user("Find validate_invoice within src and read the matching file before explaining it."),
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_file1", "type": "function", "function": {
                    "name": "grep_files", "arguments": '{"pattern":"validate_invoice","path":"src"}',
                },
            }]},
            {"role": "tool", "tool_call_id": "call_file1", "name": "grep_files",
             "content": "Only match: src/billing/checks.py:48: def validate_invoice(invoice):"},
        ], [("read_file", {"path": "src/billing/checks.py"})], files),
        *visual_cases(),
        long_case(filler_lines),
    ]


def proc_values(path: Path, keys: tuple[str, ...]) -> dict[str, int]:
    try:
        rows = path.read_text().splitlines()
    except OSError:
        return {}
    result = {}
    for row in rows:
        key, _, values = row.partition(":")
        if key in keys:
            parts = values.strip().split()
            if parts and parts[0].isdigit():
                result[key] = int(parts[0])
    return result


def gpu_counters() -> list[Path]:
    """AMD's shared-memory and dedicated-memory counters, when exposed."""
    return sorted(DRM.glob("card[0-9]*/device/mem_info_gtt_used")) + sorted(
        DRM.glob("card[0-9]*/device/mem_info_vram_used")
    )


class MemorySamples:
    """Poll Linux resident memory during a request. Server RSS covers the supplied PID."""

    def __init__(self, pid: int | None):
        self.pid = pid
        self.stop = threading.Event()
        self.host: list[int] = []
        self.rss: list[int] = []
        self.swap: list[int] = []
        self.gpu = {path: [] for path in gpu_counters()}
        self.thread = threading.Thread(target=self.sample_until_stopped)

    def sample(self) -> None:
        mem = proc_values(MEMINFO, ("MemAvailable",))
        if "MemAvailable" in mem:
            self.host.append(mem["MemAvailable"])
        if self.pid is not None:
            process = proc_values(Path(f"/proc/{self.pid}/status"), ("VmRSS", "VmSwap"))
            if "VmRSS" in process:
                self.rss.append(process["VmRSS"])
            if "VmSwap" in process:
                self.swap.append(process["VmSwap"])
        for path, history in self.gpu.items():
            try:
                history.append(int(path.read_text()) // 1024)
            except (OSError, ValueError):
                continue

    def sample_until_stopped(self) -> None:
        while not self.stop.wait(MEMORY_SAMPLE_SECONDS):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *_error):
        self.stop.set()
        self.thread.join()
        self.sample()

    def report(self) -> dict[str, Any]:
        def mib(number: int) -> float:
            return round(number / 1024, 2)

        return {
            "host_available_mib_before": mib(self.host[0]) if self.host else None,
            "host_available_mib_min": mib(min(self.host)) if self.host else None,
            "server_rss_mib_before": mib(self.rss[0]) if self.rss else None,
            "server_rss_mib_peak": mib(max(self.rss)) if self.rss else None,
            "server_swap_mib_peak": mib(max(self.swap)) if self.swap else None,
            "system_gpu_memory_mib": {
                f"{path.parent.parent.name}/{path.name}": {"before": mib(values[0]), "peak": mib(max(values))}
                for path, values in self.gpu.items() if values
            },
        }


def piece_messages(response):
    for raw in response.iter_lines(chunk_size=1):
        if not raw.startswith(b"data:"):
            continue
        value = raw[5:].strip()
        if value == b"[DONE]":
            return
        yield json.loads(value)


def append_call(calls, delta):
    for piece in delta.get("tool_calls") or []:
        index = piece.get("index", 0)
        function = piece.get("function") or {}
        current = calls.setdefault(index, ["", ""])
        current[0] += function.get("name") or ""
        current[1] += function.get("arguments") or ""


def decode_calls(calls):
    decoded = []
    for _, (name, payload) in sorted(calls.items()):
        try:
            arguments = json.loads(payload)
        except ValueError:
            arguments = payload
        decoded.append((name, arguments))
    return decoded


def score(case: Case, answer: str, calls) -> bool:
    if isinstance(case.expected, list):
        return calls == case.expected and not answer.strip()
    if case.category == "decode":
        return not calls and answer.strip() == case.expected
    clean = re.sub(r"[\x60*]", "", answer).strip()
    return not calls and bool(re.fullmatch(case.expected, clean, flags=re.I))


def complete(session, url, model, case, cache, pid, args, *, answer_limit=1200):
    request: dict[str, Any] = {
        "model": model, "messages": case.messages, "temperature": 0, "seed": SAMPLING_SEED,
        "max_tokens": args.max_tokens, "stream": True, "cache_prompt": cache,
        "stream_options": {"include_usage": True},
    }
    if args.thinking != "default":
        request["chat_template_kwargs"] = {"enable_thinking": args.thinking == "on"}
    if case.tools:
        request["tools"] = case.tools

    first = visible = None
    answer: list[str] = []
    calls: dict[int, list[str]] = {}
    timings: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    finish = None
    sampler = MemorySamples(pid)

    with sampler:
        started = time.perf_counter()
        try:
            with session.post(f"{url}/v1/chat/completions", json=request, stream=True,
                              timeout=(CONNECT_TIMEOUT, args.timeout)) as response:
                if response.status_code != 200:
                    raise ValueError(f"HTTP {response.status_code}: {response.text[:300]}")
                for event in piece_messages(response):
                    arrival = time.perf_counter()
                    if "error" in event:
                        raise ValueError(str(event["error"])[:300])
                    if isinstance(event.get("timings"), dict):
                        timings.update(event["timings"])
                    if isinstance(event.get("usage"), dict):
                        usage.update(event["usage"])
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        text = delta.get("content") or ""
                        if text or delta.get("tool_calls"):
                            visible = arrival if visible is None else visible
                            answer.append(text)
                            append_call(calls, delta)
                        if first is None and (visible is not None or delta.get("reasoning_content") or delta.get("reasoning")):
                            first = arrival
                        finish = choice.get("finish_reason") or finish
            if finish is None:
                raise ValueError("stream ended without a finish reason")
        except (requests.RequestException, ValueError) as error:
            failure = str(error)
        else:
            failure = None
        finished = time.perf_counter()

    text = "".join(answer)
    decoded = decode_calls(calls)
    token_details = usage.get("prompt_tokens_details") or {}
    prompt_tokens = usage.get("prompt_tokens")
    cached_tokens = timings.get("cache_n", token_details.get("cached_tokens"))
    fresh_tokens = timings.get("prompt_n")
    if fresh_tokens is None and cached_tokens is not None and prompt_tokens is not None:
        fresh_tokens = prompt_tokens - cached_tokens

    def elapsed(moment):
        return round((moment - started) * 1000, 2) if moment is not None else None

    return {
        "case": case.name, "category": case.category,
        "cache_requested": cache, "matched_tokens": cached_tokens, "fresh_prompt_tokens": fresh_tokens,
        "finish_reason": finish, "error": failure,
        "passed": failure is None and finish != "length" and score(case, text, decoded),
        "ttft_ms": elapsed(first), "first_content_or_tool_ms": elapsed(visible), "total_ms": elapsed(finished),
        "prompt_ms": timings.get("prompt_ms"), "tokens_per_second": timings.get("predicted_per_second"),
        "output_tokens": timings.get("predicted_n", usage.get("completion_tokens")), "memory": sampler.report(),
        "answer": text[:answer_limit], "calls": decoded,
    }


def host_info() -> dict[str, Any]:
    cpus = re.findall(r"^model name\s+:\s+(.+)$", Path("/proc/cpuinfo").read_text(), re.M)
    memory = proc_values(MEMINFO, ("MemTotal", "SwapTotal", "SwapFree"))
    return {
        "cpu": cpus[0] if cpus else platform.machine(),
        "logical_cpus": os.cpu_count(),
        "memtotal_mib": memory.get("MemTotal", 0) // 1024,
        "swaptotal_mib": memory.get("SwapTotal", 0) // 1024,
        "swapfree_mib": memory.get("SwapFree", 0) // 1024,
        "kernel": platform.release(),
    }


def run_endpoint(session, endpoint, pid, workload, args):
    label, url, model = endpoint
    url = url.rstrip("/").removesuffix("/v1")
    if model == "auto":
        listing = session.get(f"{url}/v1/models", timeout=(CONNECT_TIMEOUT, args.timeout))
        listing.raise_for_status()
        model = listing.json()["data"][0]["id"]
    rows = []
    for repetition in range(1, args.repeats + 1):
        for case in workload:
            prompt_cache_requests = (False, True) if case.category == "retrieval" else (False,)
            for cache in prompt_cache_requests:
                row = complete(session, url, model, case, cache, pid, args)
                row["repetition"] = repetition
                rows.append(row)
                status = "pass" if row["passed"] else "fail"
                print(f"{label} {case.name} cache={int(cache)} {status} "
                      f"TTFT {row['ttft_ms']}ms, {row['tokens_per_second']} tok/s", flush=True)
    summary = {}
    for case in workload:
        modes = summary[case.name] = {}
        for mode, cache in (("uncached_requested", False), ("cache_requested", True)):
            selected = [row for row in rows if row["case"] == case.name and row["cache_requested"] == cache]
            if not selected:
                continue
            modes[mode] = {
                "passed": sum(row["passed"] for row in selected), "requests": len(selected),
                "errors": sum(row["error"] is not None for row in selected),
            }
            for field in ("ttft_ms", "first_content_or_tool_ms", "tokens_per_second", "matched_tokens"):
                observed = [row[field] for row in selected if row[field] is not None]
                modes[mode]["median_" + field] = statistics.median(observed) if observed else None
    return {"model": model, "pid": pid, "summary": summary, "rows": rows}


def is_loopback(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.username is not None or parsed.query or parsed.fragment:
            return False
        return parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", nargs=3, metavar=("LABEL", "URL", "MODEL"), action="append", required=True)
    parser.add_argument("--pid", nargs=2, metavar=("LABEL", "PID"), action="append", default=[])
    parser.add_argument("--case", action="append", help="Run only these named cases. Repeat for multiple cases.")
    parser.add_argument("--filler-lines", type=int, default=140, help="Number of distracting records in the long prompt.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180, help="HTTP idle-read timeout in seconds.")
    parser.add_argument("--thinking", choices=("off", "on", "default"), default="off")
    parser.add_argument("--key-env", default="KIWIBRO_BENCH_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.repeats, args.max_tokens, args.timeout) < 1 or args.filler_lines < 0:
        parser.error("repeats, max tokens and timeout must be positive, and filler lines nonnegative")
    endpoints = {label for label, _, _ in args.endpoint}
    if len(endpoints) != len(args.endpoint):
        parser.error("endpoint labels must be unique")
    if any(not is_loopback(url) for _, url, _ in args.endpoint):
        parser.error("use a loopback HTTP address for each endpoint")
    pids = {}
    for label, pid in args.pid:
        if label not in endpoints or not pid.isdecimal() or int(pid) < 1 or not Path(f"/proc/{pid}").is_dir():
            parser.error(f"invalid server PID for {label}")
        pids[label] = int(pid)
    workload = cases(args.filler_lines)
    if args.case:
        workload.append(copy_case())
        available = {case.name for case in workload}
        if set(args.case) - available:
            parser.error(f"unknown case; choose from {', '.join(sorted(available))}")
        workload = [case for case in workload if case.name in args.case]
    report = {
        "host": host_info(), "thinking": args.thinking, "repeats": args.repeats,
        "filler_lines": args.filler_lines, "max_tokens": args.max_tokens, "endpoints": {},
    }
    with requests.Session() as session:
        session.trust_env = False  # loopback requests stay local on machines with a proxy
        key = os.environ.get(args.key_env)
        if key:
            session.headers["Authorization"] = f"Bearer {key}"
        for endpoint in args.endpoint:
            label = endpoint[0]
            report["endpoints"][label] = run_endpoint(session, endpoint, pids.get(label), workload, args)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
