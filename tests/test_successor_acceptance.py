r"""Held-out workspace outcomes through the real KiwiBro HTTP API and agent loop.

The default uses the suite's scripted LLM boundary. An optional loopback model
uses the same task prompts and state assertions:

    KIWIBRO_ACCEPTANCE_ENDPOINT=http://127.0.0.1:8080 \
      KIWIBRO_ACCEPTANCE_MODEL=your-model python3 -m pytest -q tests/test_successor_acceptance.py

The endpoint must already be running. The runner creates its own temporary app
state and workspace. Model transport credentials can be passed through
KIWIBRO_SERVER_KEY, as with a normal attached KiwiBro installation.

To test the installed Kiwi Mind path, use an existing verified native pack at
<staged-data>/mind/candidate/ (see docs/MIND-CANDIDATE.md) and a supported runtime:

    KIWIBRO_ACCEPTANCE_NATIVE_DATA=/absolute/path/to/staged-data \
      KIWIBRO_LLAMA_SERVER=/absolute/path/to/llama-server \
      python3 -m pytest -q tests/test_successor_acceptance.py

This mode starts the native engine on demand. It checks the pack once in scratch
space, links the weights into a separate app state per test, and stops that app's
engine at teardown. Use one live mode per run.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

import pytest
from PIL import Image, ImageChops, ImageDraw, ImageFont

from kiwibro import create_app, pagecheck, tools
from kiwibro.browser_render import _page_socket
from kiwibro.llm import ChatResult, Completion
from kiwibro.mind import KIWIMIND
from kiwibro.mind.native_generic import NATIVE_FORMAT, describe_pack
from kiwibro.mind.pack import CHECK_CACHE, MANIFEST, Pack, PackError
from test_backend import Env, SSEStream, tool_call_result

PKG = "stack/packages/queue_policy/policy.py"
NOTE = "stack/notes/staging.txt"
CHECK = "python3 -B -m unittest discover -s tests"
PANEL = "handoff.html"
STAMP = "LX-731-Q"
ORIGINAL_LIMIT = "return requested or DEFAULT"
FIXED_LIMIT = "return DEFAULT if requested is None else requested"
NATIVE_DATA = "KIWIBRO_ACCEPTANCE_NATIVE_DATA"


def link_native_pack(source: Pack, data_dir: Path) -> Pack:
    manifest = source.manifest()
    if manifest is None or manifest.get("format") != NATIVE_FORMAT:
        pytest.fail(f"Missing native pack manifest in {source.dir}")
    pack = Pack(data_dir, subdir="candidate")
    pack.dir.mkdir(parents=True)
    for name in manifest["files"]:
        destination = pack.path(name)
        target = source.path(name).resolve()
        if destination.suffix != ".gguf" or not target.is_file():
            pytest.fail(f"Expected a staged GGUF file at {source.path(name)}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(target)
    pack.path(MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return pack


@pytest.fixture(scope="session")
def verified_native_pack(tmp_path_factory: pytest.TempPathFactory) -> Pack:
    source = Pack(Path(os.environ[NATIVE_DATA]).expanduser(), subdir="candidate")
    # Verify in scratch space so Pack.check never writes a cache to the source.
    linked = link_native_pack(source, tmp_path_factory.mktemp("acceptance-verified-pack"))
    try:
        describe_pack(linked)
    except PackError as error:
        pytest.fail(f"Cannot use native pack in {source.dir}: {error}")
    return linked


@pytest.fixture
def acceptance(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Env]:
    endpoint = os.environ.get("KIWIBRO_ACCEPTANCE_ENDPOINT")
    native = bool(os.environ.get(NATIVE_DATA))
    if endpoint and native:
        pytest.fail(f"Set either {NATIVE_DATA} or KIWIBRO_ACCEPTANCE_ENDPOINT for a live run.")
    if not (endpoint or native):
        yield request.getfixturevalue("env")
        return

    data_dir = tmp_path / "data"
    if native:
        verified = request.getfixturevalue("verified_native_pack")
        pack = link_native_pack(verified, data_dir)
        shutil.copyfile(verified.path(CHECK_CACHE), pack.path(CHECK_CACHE))
        monkeypatch.setenv("KIWIMIND_CANDIDATE", "1")
        monkeypatch.setenv("KIWIBRO_OWN_MODELS_ONLY", "1")
        monkeypatch.delenv("KIWIBRO_MODELS_DIR", raising=False)
        monkeypatch.setenv("KIWIBRO_OFFLINE", "1")
        server_url = ""
    else:
        address = urlsplit(endpoint)
        if address.scheme != "http" or address.hostname not in {"127.0.0.1", "localhost", "::1"} or address.username:
            pytest.fail("Acceptance requires an already-running HTTP model on loopback.")
        model = os.environ.get("KIWIBRO_ACCEPTANCE_MODEL")
        if not model:
            pytest.fail("Set KIWIBRO_ACCEPTANCE_MODEL to the model ID under test.")
        server_url = endpoint.rstrip("/").removesuffix("/v1")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = create_app({
        "TESTING": True, "KIWIBRO_DATA_DIR": str(data_dir), "KIWIBRO_WARMUP": False,
        "KIWIBRO_WORKSPACE": str(workspace), "KIWIBRO_SERVER_URL": server_url,
    })
    scenario = Env(app, app.extensions["kiwibro"], None, workspace)
    if not native:
        listing = scenario.client.get("/api/models?refresh=1")
        assert listing.status_code == 200
        assert model in {row["id"] for row in listing.get_json()["models"]}, f"Model {model!r} is unavailable"
        yield scenario
        return

    llm = scenario.svc.llm
    mind = llm.mind()
    try:
        response = scenario.client.get("/api/mind")
        assert response.status_code == 200
        status = response.get_json()
        assert status["usable"] and status["engine"] == NATIVE_FORMAT and status["active"] == "candidate", status
        yield scenario
    finally:
        try:
            mind.stop()
        finally:
            llm.stop_direct()


def turn(
    env: Env, prompt: str, allowed: set[str], *, allowed_paths: set[str] | None = None,
    command: tuple[str, str] | None = None, attachments: list[dict] | None = None,
) -> tuple[list[tuple[str, dict[str, Any]]], list[dict[str, Any]]]:
    # Keep generated code in scratch space. The test driver may approve only
    # the predeclared test invocation and edits to the requested fixture file.
    writes = {"apply_patch", "replace_in_file", "write_file"}
    permissions = {name: ("ask" if name in writes | {"run_command"} else "auto") if name in allowed else "deny"
                   for name in tools.tool_names()}
    env.settings_patch({"autonomy": "supervised", "memory_enabled": False, "temperature": 0,
                        "tool_profile": "full", "max_tool_rounds": 16, "reasoning": "off",
                        "permissions": permissions})
    payload: dict[str, Any] = {"message": prompt}
    if attachments:
        payload["attachments"] = attachments
    native = env.fake is None and bool(os.environ.get(NATIVE_DATA))
    if env.fake is None:
        payload["model"] = KIWIMIND if native else os.environ["KIWIBRO_ACCEPTANCE_MODEL"]
    response = env.client.post("/api/chat", json=payload, buffered=False)
    assert response.status_code == 200, response.get_data(as_text=True)
    stream = SSEStream(response).start()
    timeout = 45 if env.fake else 300
    deadline = time.monotonic() + timeout
    handled = set()
    while not stream.done.wait(0.04):
        if time.monotonic() > deadline:
            meta = next((data for event, data in stream.events if event == "meta"), {})
            env.new_client().post("/api/chat/stop", json={"conversation_id": meta.get("conversation_id")})
            pytest.fail(f"Acceptance timeout; events={stream.names()}")
        with stream._lock:
            approvals = [data for event, data in stream.events if event == "approval" and data["approval_id"] not in handled]
        for approval in approvals:
            identity, name, args = approval["approval_id"], approval["tool"], approval["args"]
            permitted = name in writes and args.get("path") in (allowed_paths or set())
            if name == "run_command" and command:
                permitted = (args.get("command"), args.get("cwd")) == command and simple_policy(env.workspace)
            result = env.new_client().post(f"/api/approvals/{identity}", json={"decision": "allow" if permitted else "deny"})
            assert result.status_code == 200
            handled.add(identity)
    events = stream.finish(timeout=timeout)
    assert events[0][0] == "meta"
    assert events[-1][0] == "done", [(name, data) for name, data in events if name == "error"]
    assert not events[-1][1]["aborted"]
    if native:
        assert events[0][1]["model"] == events[-1][1]["model"] == KIWIMIND
        assert any(event == "brain" and data["brain"] == KIWIMIND for event, data in events)
    conversation = events[0][1]["conversation_id"]
    detail = env.client.get(f"/api/conversations/{conversation}")
    assert detail.status_code == 200
    return events, detail.get_json()["messages"]


def write(root: Path, path: str, content: str) -> None:
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content, encoding="utf-8")


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True, timeout=10).stdout


def call(index: int, name: str, **args: Any) -> ChatResult:
    return tool_call_result(name, args, f"accept_{index}")


def repo_fixture(root: Path) -> str:
    write(root, PKG, f"DEFAULT = 16\n\ndef queue_limit(requested):\n    {ORIGINAL_LIMIT}\n")
    write(root, "stack/services/dispatch/service.py", "from packages.queue_policy.policy import queue_limit\n"
          "\ndef queue_depth(config):\n    return queue_limit(config.get('queue_limit'))\n")
    write(root, "stack/services/audit/ledger.py", "from packages.queue_policy.policy import queue_limit\n"
          "\ndef planned_quota(record):\n    return queue_limit(record.get('quota'))\n")
    write(root, "stack/tests/test_dispatch.py", "import unittest\nfrom services.dispatch.service import queue_depth\n"
          "\nclass Dispatch(unittest.TestCase):\n    def test_zero(self):\n        self.assertEqual(queue_depth({'queue_limit': 0}), 0)\n"
          "    def test_default(self):\n        self.assertEqual(queue_depth({}), 16)\n")
    write(root, "stack/tests/test_audit.py", "import unittest\nfrom services.audit.ledger import planned_quota\n"
          "\nclass Audit(unittest.TestCase):\n    def test_zero(self):\n        self.assertEqual(planned_quota({'quota': 0}), 0)\n"
          "    def test_positive(self):\n        self.assertEqual(planned_quota({'quota': 6}), 6)\n")
    write(root, "stack/examples/queue_policy.py", "def example(requested):\n    return requested or 16\n")
    write(root, NOTE, "Stable release notes.\n")
    nested = root / "stack"
    git(nested, "init", "-q")
    git(nested, "add", ".")
    git(nested, "-c", "user.name=KiwiBro QA", "-c", "user.email=qa@example.test",
        "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Initial fixture")
    write(root, NOTE, "Hand edited before the task. Keep this.\n")
    return git(nested, "rev-parse", "HEAD")


def simple_policy(root: Path) -> bool:
    """Keep executable fixture edits to the shared policy's small pure expression."""
    try:
        tree = ast.parse((root / PKG).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    simple = (ast.Module, ast.Assign, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
              ast.Name, ast.Load, ast.Store, ast.BoolOp, ast.Or, ast.IfExp, ast.If, ast.Compare,
              ast.Is, ast.IsNot, ast.Constant)
    return all(isinstance(node, simple) and (not isinstance(node, ast.Name) or node.id in {"requested", "DEFAULT"})
               for node in ast.walk(tree))


def run_fixture_tests(root: Path) -> subprocess.CompletedProcess[str]:
    assert simple_policy(root)
    return subprocess.run(CHECK.split(), cwd=root / "stack", capture_output=True, text=True, timeout=15)


def test_repairs_two_consumers_in_nested_repository(acceptance: Env) -> None:
    env = acceptance
    head = repo_fixture(env.workspace)
    initial = run_fixture_tests(env.workspace)
    assert initial.returncode != 0 and "failures=2" in initial.stderr
    if env.fake:
        env.fake.stream_script = [
            call(1, "git", action="status", path="stack"),
            call(2, "grep_files", pattern="def queue_limit", path="stack/packages"),
            call(3, "read_files", paths=[PKG, "stack/services/dispatch/service.py", "stack/services/audit/ledger.py"]),
            call(4, "apply_patch", path=PKG, edits=[{"old": ORIGINAL_LIMIT, "new": FIXED_LIMIT}]),
            call(5, "run_command", command=CHECK, cwd="stack"),
            ChatResult(content="The shared queue policy preserves zero now. The dispatch and audit tests both passed."),
        ]
    events, _ = turn(
        env,
        "The stack/ checkout has two consumers that silently ignore an explicit zero queue limit. "
        "Trace the shared policy and make a minimal fix that preserves the omitted-value fallback. "
        "Keep staging.txt and the example untouched, and don't commit. "
        f"Run `{CHECK}` with stack as the working directory after the change.",
        {"git", "grep_files", "read_file", "read_files", "list_files", "apply_patch", "replace_in_file", "write_file", "run_command"},
        allowed_paths={PKG}, command=(CHECK, "stack"),
    )
    assert run_fixture_tests(env.workspace).returncode == 0
    assert git(env.workspace / "stack", "rev-parse", "HEAD") == head
    assert set(git(env.workspace / "stack", "diff", "--name-only").splitlines()) == {"packages/queue_policy/policy.py", "notes/staging.txt"}
    assert git(env.workspace / "stack", "diff", "--cached", "--name-only") == ""
    assert set(git(env.workspace / "stack", "status", "--short").splitlines()) == {
        " M packages/queue_policy/policy.py", " M notes/staging.txt",
    }
    assert (env.workspace / NOTE).read_text() == "Hand edited before the task. Keep this.\n"
    assert any(name == "tool_result" and item["name"] == "run_command" and item["ok"] for name, item in events)


def snapshot(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_stale_data_path_and_typed_argument_recovery(acceptance: Env) -> None:
    env = acceptance
    old = "shipments/round-2/old.tsv"
    path = "shipments/round-2/priority-status.tsv"
    write(env.workspace, path, "job\tstatus\tretry\nTX-10\tqueued\t0\nTX-11\tfailed\t1\n"
          "TX-12\tready\t0\nTX-13\tfailed\t2\nTX-14\tready\t0\nTX-15\tqueued\t0\nTX-16\tready\t0\n")
    write(env.workspace, "shipments/archive/priority-status.tsv", "job\tstatus\nOLD\tfailed\n")
    write(env.workspace, "shipments/round-2/by-month.csv", "month,total\nJuly,200\n")
    before = snapshot(env.workspace)
    if env.fake:
        env.fake.stream_script = [
            call(1, "analyze_table", path=old, limit=0),
            call(2, "analyze_table", path=old),
            call(3, "list_files", path="shipments/round-2"),
            call(4, "analyze_table", path=path),
            call(5, "analyze_table", path=path, where='status == "failed"', aggregate="count"),
            ChatResult(content=f"Failed: 2 of 7 rows in {path}."),
        ]
    _events, messages = turn(
        env,
        f"The handoff names {old}. Try that path first. If it fails, find the current TSV "
        "inside shipments/round-2 and report the number of failed status rows out of the total. "
        "Give both counts and the actual source file. This is a read-only check.",
        {"analyze_table", "list_files", "glob_files", "read_files", "read_file", "calculate"},
    )
    assert snapshot(env.workspace) == before
    assistant = [row["content"] for row in messages if row["role"] == "assistant"][-1]
    assert re.search(r"\b2\b.*\b7\b|\b7\b.*\b2\b", assistant, re.S), assistant
    assert path in assistant
    observed = [row for row in messages if row["role"] == "tool"]
    assert any(row["tool_name"] == "analyze_table" and "2 rows matched" in row["content"] for row in observed)
    if env.fake:
        errors = [row["content"] for row in observed if row["tool_name"] == "analyze_table"][:2]
        assert len(errors) == 2 and "limit" in errors[0] and ("does not exist" in errors[1] or "not found" in errors[1])


def upload_panel(env: Env) -> tuple[dict[str, Any], bytes]:
    image = Image.new("RGB", (780, 320), "#e7eef6")
    pen = ImageDraw.Draw(image)
    big = ImageFont.load_default(size=38)
    small = ImageFont.load_default(size=28)
    pen.text((38, 25), "Handoff dashboard", font=big, fill="#122b42")
    pen.rounded_rectangle((32, 100, 740, 270), radius=14, fill="white")
    pen.text((62, 130), "Dispatch queue", font=small, fill="#122b42")
    pen.text((390, 130), STAMP, font=big, fill="#2c518b")
    pen.text((62, 200), "Needs acknowledgment", font=small, fill="#8d3030")
    packet = io.BytesIO()
    image.save(packet, format="PNG")
    raw = packet.getvalue()
    response = env.client.post("/api/attachments", data={"file": (io.BytesIO(raw), "handoff.png")})
    assert response.status_code == 200
    return response.get_json()["attachment"], raw


def render_controls(root: Path) -> None:
    original = (root / PANEL).read_text(encoding="utf-8")
    probe = """
<script>
document.addEventListener("DOMContentLoaded", () => {
  const button = document.getElementById("acknowledge");
  const state = document.getElementById("state");
  const before = state.textContent;
  button.focus();
  button.click();
  if (before.includes("Needs acknowledgment") && state.textContent.includes("Acknowledged")
    && button.getAttribute("aria-pressed") === "true" && document.activeElement === button) {
    document.body.append("WORKLOAD_CONTROL_OK");
  }
});
</script>
"""
    page = root / "_acceptance_controls.html"
    page.write_text(original + probe, encoding="utf-8")
    result = pagecheck.chrome_render(page, root / "_acceptance_controls.png", *pagecheck.PHONE, pagecheck.BUDGET_MS)
    assert result.report, result.error
    assert "WORKLOAD_CONTROL_OK" in result.report["text"], result.report
    assert result.report["w"] == 360 and result.report["sw"] <= 360


def test_attachment_to_interactive_rendered_artifact(acceptance: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    env = acceptance
    if not pagecheck.find_chrome():
        pytest.skip("Chrome is required to measure the artifact and the control state.")
    attachment, raw = upload_panel(env)
    seen_dimensions = []
    if env.fake:
        # Only the vision inference is scripted. Upload, per-turn attachment
        # routing, tool execution, browser rendering, and storage are real.
        env.fake.models[0].update(vision=True, capabilities=["tool_use", "vision"])
        original_complete = env.fake.complete

        def complete(**kwargs):
            for message in kwargs["messages"]:
                if not isinstance(message.get("content"), list):
                    continue
                image_url = next((item["image_url"]["url"] for item in message["content"] if item["type"] == "image_url"), "")
                if image_url.startswith("data:image/"):
                    with Image.open(io.BytesIO(base64.b64decode(image_url.partition(",")[2]))) as frame:
                        seen_dimensions.append(frame.size)
                    return Completion(f"Dispatch queue · {STAMP} · Needs acknowledgment")
            return original_complete(**kwargs)

        monkeypatch.setattr(env.svc.llm, "complete", complete)
        html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Dispatch handoff</title>
<style>body{{margin:1rem;font:1rem system-ui}} main{{max-width:40em;margin:auto}} button{{font:inherit}}</style>
<main><h1>Dispatch queue</h1><p>{STAMP}</p>
<p id="state" aria-live="polite">Needs acknowledgment</p>
<button id="acknowledge" type="button" aria-pressed="false">Acknowledge</button></main>
<script>
const button=document.getElementById("acknowledge"), state=document.getElementById("state");
button.addEventListener("click", () => {{
  const enabled=button.getAttribute("aria-pressed")!=="true";
  button.setAttribute("aria-pressed", String(enabled));
  state.textContent=enabled?"Acknowledged":"Needs acknowledgment";
}});
</script></html>"""
        bad_width = "body{min-width:720px;margin:1rem"
        good_width = "body{margin:1rem"
        env.fake.stream_script = [
            call(1, "write_file", path=PANEL, content=html.replace(good_width, bad_width)),
            ChatResult(content=f"Handoff saved as {PANEL}."),
            call(2, "apply_patch", path=PANEL, edits=[{"old": bad_width, "new": good_width}]),
            ChatResult(content=f"Handoff saved as {PANEL} with build code {STAMP}. It fits the phone viewport."),
        ]
    before = env.workspace / attachment["path"]
    assert before.read_bytes() == raw
    events, messages = turn(
        env,
        "Read the attached dashboard and save a self-contained handoff.html with its queue name, build code "
        "and exact initial status. Provide a keyboard-focusable button id acknowledge, initially aria-pressed=false. "
        "When activated, set aria-pressed=true and change the visible status element id state to Acknowledged. "
        "Keep the page readable on a phone at 360 px and check the saved render.",
        {"write_file", "read_file", "apply_patch", "replace_in_file", "describe_image", "look_at_page"},
        allowed_paths={PANEL}, attachments=[{"path": attachment["path"]}],
    )
    assert before.read_bytes() == raw
    page = (env.workspace / PANEL).read_text()
    assert STAMP in page and "Dispatch queue" in page
    assert any(event == "tool_result" and data["name"] == "look_at_page" and data["ok"] for event, data in events)
    assert (env.workspace / "screenshots" / "look-handoff.png").is_file()
    rendered = [row["content"] for row in messages if row.get("tool_name") == "look_at_page"]
    assert "fits a phone screen" in rendered[-1], rendered
    if env.fake:
        assert (780, 320) in seen_dimensions
        assert len(rendered) == 2 and "wider than the screen" in rendered[0]
    render_controls(env.workspace)


@pytest.fixture
def offline_chrome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    chrome = pagecheck.find_chrome()
    if not chrome:
        pytest.skip("Chrome is required for the 3D artifact check.")
    # The app tool and the independent oracle both use Chrome with the same
    # refused proxy, including for loopback, and no direct DNS or WebRTC.
    with socket.socket() as blocked:
        blocked.bind(("127.0.0.1", 0))
        flags = [
            "--disable-background-networking", "--disable-extensions", "--disable-quic",
            "--disable-sync", "--no-pings", "--host-resolver-rules=MAP * ~NOTFOUND",
            "--force-webrtc-ip-handling-policy=disable_non_proxied_udp", "--proxy-bypass-list=<-loopback>",
            f"--proxy-server=http://127.0.0.1:{blocked.getsockname()[1]}",
        ]
        wrapper = tmp_path / "offline-chrome"
        wrapper.write_text(f'#!/bin/sh\nexec {shlex.join([chrome, *flags])} "$@"\n', encoding="utf-8")
        wrapper.chmod(0o700)
        monkeypatch.setattr(pagecheck, "find_chrome", lambda: str(wrapper))
        monkeypatch.setenv("KIWIBRO_OFFLINE", "1")
        yield wrapper


HATCH_STATE = """(() => {
    const byId = id => document.getElementById(id);
    const rect = node => node.getBoundingClientRect().toJSON();
    const lid = byId("lid"), button = byId("hatch");
    const style = getComputedStyle(lid);
    const matrix = style.transform === "none" ? new DOMMatrix() : new DOMMatrix(style.transform);
    const originY = parseFloat(style.transformOrigin.split(" ")[1]);
    let perspective = false;
    for (let element = lid.parentElement; element; element = element.parentElement) {
        if (getComputedStyle(element).perspective !== "none") perspective = true;
    }
    return {
        width: innerWidth, contentWidth: document.documentElement.scrollWidth,
        active: document.activeElement.id, expanded: button.getAttribute("aria-expanded"),
        namedButton: button.tagName === "BUTTON" && !!(button.getAttribute("aria-label") || button.textContent).trim(),
        control: rect(button), scene: rect(byId("scene")), base: rect(byId("base")), lid: rect(lid),
        depth: matrix.m23, hingeY: originY, lidHeight: lid.offsetHeight, perspective
    };
})()"""


def check_hatch(chrome: Path, page: Path) -> None:
    """Observe three steady poses and trusted keyboard activation at phone size."""
    import websocket

    width, height = pagecheck.PHONE
    deadline = time.monotonic() + 18
    with tempfile.TemporaryDirectory(prefix="kiwi-hatch-chrome-") as profile, subprocess.Popen([
        str(chrome), "--headless=new", "--no-first-run", "--no-default-browser-check",
        "--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--hide-scrollbars",
        f"--user-data-dir={profile}", "--remote-debugging-port=0", "about:blank",
    ], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as process:
        ws = None
        try:
            ws = _page_socket(Path(profile), process, deadline)
            assert ws is not None, "Chrome's local control endpoint did not start"
            sequence = 0
            errors = []

            def command(method: str, **params) -> dict:
                nonlocal sequence
                sequence += 1
                ws.send(json.dumps({"id": sequence, "method": method, "params": params}))
                while time.monotonic() < deadline:
                    try:
                        packet = json.loads(ws.recv())
                    except websocket.WebSocketTimeoutException:
                        continue
                    if packet.get("id") == sequence:
                        assert "error" not in packet, packet
                        return packet["result"]
                    event = packet.get("method")
                    detail = packet.get("params", {})
                    if event == "Runtime.exceptionThrown" or (
                        event == "Runtime.consoleAPICalled" and detail.get("type") == "error"
                    ):
                        errors.append(detail)
                pytest.fail(f"Timed out waiting for Chrome command {method}")

            def evaluate(expression: str, *, await_promise: bool = False):
                result = command("Runtime.evaluate", expression=expression,
                                 returnByValue=True, awaitPromise=await_promise)
                assert "exceptionDetails" not in result, result
                return result["result"].get("value")

            def key(name: str, code: int) -> None:
                args = {"key": name, "code": name, "windowsVirtualKeyCode": code}
                if name == "Enter":
                    command("Input.dispatchKeyEvent", type="keyDown", text="\r", unmodifiedText="\r", **args)
                else:
                    command("Input.dispatchKeyEvent", type="rawKeyDown", **args)
                command("Input.dispatchKeyEvent", type="keyUp", **args)

            def screenshot(bounds: dict) -> Image.Image:
                png = command("Page.captureScreenshot", format="png")["data"]
                with Image.open(io.BytesIO(base64.b64decode(png))) as image:
                    return image.convert("RGB").crop(tuple(
                        round(bounds[edge]) for edge in ("left", "top", "right", "bottom")
                    ))

            command("Runtime.enable")
            command("Network.enable")
            command("Network.setBlockedURLs", urls=["http://*", "https://*", "ws://*", "wss://*"])
            command("Emulation.setDeviceMetricsOverride", width=width, height=height,
                    deviceScaleFactor=1, mobile=True)
            command("Emulation.setEmulatedMedia", features=[{"name": "prefers-reduced-motion", "value": "reduce"}])
            assert not command("Page.navigate", url=page.resolve().as_uri()).get("errorText")
            while evaluate("document.readyState") != "complete" or evaluate("location.href") != page.resolve().as_uri():
                assert time.monotonic() < deadline
                time.sleep(0.03)
            closed = evaluate(HATCH_STATE)
            assert closed["width"] == width and closed["contentWidth"] <= width, closed
            control = closed["control"]
            assert closed["namedButton"] and control["width"] >= 40 and control["height"] >= 40, closed
            assert 0 <= control["left"] < control["right"] <= width, control
            assert 0 <= control["top"] < control["bottom"] <= height, control
            assert closed["expanded"] == "false", closed
            bounds = closed["scene"]
            assert 0 <= bounds["left"] < bounds["right"] <= width, bounds
            assert 0 <= bounds["top"] < bounds["bottom"] <= height, bounds
            assert closed["base"]["width"] >= 50 and closed["lid"]["width"] >= 50 and closed["lidHeight"] >= 50, closed
            assert closed["perspective"] and abs(closed["hingeY"]) <= 5, closed
            key("Tab", 9)
            focused = evaluate(HATCH_STATE)
            assert focused["active"] == "hatch" and focused["expanded"] == "false", focused
            assert focused["depth"] == closed["depth"], focused
            initial_pixels = screenshot(bounds)
            assert initial_pixels.getcolors(maxcolors=1) is None, "The initial scene is blank"

            # Reduced motion should reach the next pose on the next paint.
            painted = "new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
            key("Enter", 13)
            evaluate(painted, await_promise=True)
            opened = evaluate(HATCH_STATE)
            assert opened["active"] == "hatch" and opened["expanded"] == "true", opened
            assert abs(opened["depth"] - closed["depth"]) >= 0.35, (closed, opened)
            assert any(abs(opened["lid"][edge] - closed["lid"][edge]) >= 8
                       for edge in ("left", "top", "right", "bottom")), (closed, opened)
            assert all(abs(opened["base"][edge] - closed["base"][edge]) <= 1
                       for edge in ("left", "top", "right", "bottom")), (closed, opened)
            opened_pixels = screenshot(bounds)

            key("Enter", 13)
            evaluate(painted, await_promise=True)
            restored = evaluate(HATCH_STATE)
            assert restored["active"] == "hatch" and restored["expanded"] == "false", restored
            assert abs(restored["depth"] - closed["depth"]) <= 0.02, (closed, restored)
            restored_pixels = screenshot(bounds)
            assert not errors, errors
        finally:
            if ws is not None:
                ws.close(timeout=0)
            pagecheck._quit(process)
            process.wait(timeout=3)

    summed = initial_pixels.width * initial_pixels.height
    changed = ImageChops.difference(initial_pixels, opened_pixels).convert("L").histogram()
    returned = ImageChops.difference(initial_pixels, restored_pixels).convert("L").histogram()
    assert sum(changed[20:]) > 0.03 * summed, "The hinge moved in CSS without visibly changing the scene."
    assert sum(returned[20:]) < 0.005 * summed, "Closing did not restore the initial rendered pose."


def test_builds_and_checks_an_interactive_3d_hatch(acceptance: Env, offline_chrome: Path) -> None:
    env = acceptance
    filename = "inspection-hatch.html"
    if env.fake:
        reference = Path(__file__).with_name("fixtures") / "successor_inspection_hatch.html"
        env.fake.stream_script = [
            call(1, "write_file", path=filename, content=reference.read_text(encoding="utf-8")),
            ChatResult(content=f"Saved and checked {filename}."),
        ]
    events, _ = turn(
        env,
        "Save inspection-hatch.html, a single self-contained HTML inspection housing. Show an inspection bay "
        "under a service lid. Use CSS perspective and real 3D transforms so the lid opens toward the rear "
        "on its top-edge hinge while the base stays fixed. Give the scene, base and lid the IDs scene, base and lid. "
        "Start closed. Use a single labeled native button with ID hatch that opens and closes the lid, keeps "
        "keyboard focus, and sets aria-expanded for the current state. Make the button at least 40 × 40 px and "
        "the scene and controls fit a 360 px phone without horizontal scrolling. Hold each pose until activated. "
        "Transitions may take up to 250 ms and should be immediate if reduced motion is requested. Use inline CSS "
        "and JS only. Check the saved page using look_at_page.",
        {"write_file", "read_file", "apply_patch", "replace_in_file", "look_at_page"},
        allowed_paths={filename},
    )
    assert any(event == "tool_result" and data["name"] == "look_at_page" and data["ok"] for event, data in events)
    assert (env.workspace / "screenshots" / "look-inspection-hatch.png").is_file()
    check_hatch(offline_chrome, env.workspace / filename)
