"""Reusable end-to-end harness for Kiwibro.

Contains:
- ``free_port`` / ``wait_for_health`` helpers
- ``AppServer``: owns a real ``python3 app.py`` child process on a scratch port
  with an isolated data dir + workspace (never touches the user's real data)
- ``ApiClient`` / ``SSEChat``: thin requests-based client with a strict SSE parser
- tiny evidence/report helpers

The harness is intentionally stdlib+requests only: tests/e2e must be runnable
standalone (``python3 tests/e2e/run_e2e.py``) as well as from pytest.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPO_ROOT / ".dev" / "qa" / "runtime"
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def free_port() -> int:
    """Reserve an ephemeral loopback port (best-effort: released immediately)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def port_is_closed(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def wait_until(predicate: Callable[[], bool], timeout: float, interval: float = 0.05) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError("condition was not met in time")


def http_json(url: str, timeout: float = 2.0) -> dict[str, Any] | None:
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return None
        data = response.json()
        return data if isinstance(data, dict) else None
    except (requests.RequestException, ValueError):
        return None


# ---------------------------------------------------------------------------
# app process lifecycle
# ---------------------------------------------------------------------------


class AppServerError(RuntimeError):
    pass


class AppServer:
    """Start/stop the real Flask app as a child process with scratch state."""

    def __init__(
        self,
        label: str,
        *,
        lmstudio_url: str | None = None,
        port: int | None = None,
        data_dir: Path | None = None,
        workspace: Path | None = None,
        extra_env: dict[str, str] | None = None,
        fresh: bool = True,
        boot_timeout: float = 25.0,
    ) -> None:
        self.label = label
        self.port = port or free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        run_dir = RUNTIME_ROOT / label
        if fresh and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.data_dir = Path(data_dir) if data_dir else run_dir / "data"
        self.workspace = Path(workspace) if workspace else run_dir / "workspace"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.log_path = run_dir / "app.log"
        self.lmstudio_url = lmstudio_url
        self.extra_env = dict(extra_env or {})
        self.boot_timeout = boot_timeout
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_handle = None

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "AppServer":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "KIWIBRO_PORT": str(self.port),
                "KIWIBRO_HOST": "127.0.0.1",
                "KIWIBRO_DATA_DIR": str(self.data_dir),
                "KIWIBRO_WORKSPACE": str(self.workspace),
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        if self.lmstudio_url:
            env["KIWIBRO_LMSTUDIO_URL"] = self.lmstudio_url
        env.update(self.extra_env)
        self._log_handle = open(self.log_path, "wb")
        self._proc = subprocess.Popen(
            [PYTHON, "app.py"],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + self.boot_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise AppServerError(
                    f"app for {self.label!r} exited during boot (code {self._proc.returncode})\n"
                    + self.log_tail()
                )
            payload = http_json(f"{self.base_url}/health", timeout=1.0)
            if payload and payload.get("ok") is True:
                return
            time.sleep(0.05)
        raise AppServerError(f"app for {self.label!r} did not become healthy\n" + self.log_tail())

    def stop(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self._proc = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    # -- introspection -----------------------------------------------------

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    def client(self) -> "ApiClient":
        return ApiClient(self.base_url)

    def log_text(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def log_tail(self, lines: int = 40) -> str:
        text = self.log_text().splitlines()
        return "\n".join(text[-lines:])

    def app_log_text(self) -> str:
        path = self.data_dir / "kiwibro.log"
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""


# ---------------------------------------------------------------------------
# HTTP API client + SSE parser
# ---------------------------------------------------------------------------


@dataclass
class Response:
    status: int
    data: Any
    headers: dict[str, str]
    text: str
    elapsed_ms: float = 0.0

    def json_field(self, name: str, default: Any = None) -> Any:
        return (self.data or {}).get(name, default) if isinstance(self.data, dict) else default


class ApiClient:
    """HTTP client for the app.

    Deliberately does not pool connections: Werkzeug's development server always
    sends ``Connection: close`` on responses (and our SSE contract asks for
    ``keep-alive``, so the header appears twice). A pooled requests connection
    after an SSE stream can be stale, so each call opens a fresh one.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def close(self) -> None:
        return None

    # -- raw requests ------------------------------------------------------

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        data: bytes | str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        allow_redirects: bool = False,
    ) -> Response:
        started = time.monotonic()
        response = requests.request(
            method.upper(),
            f"{self.base_url}{path}",
            json=json_body,
            data=data,
            headers=headers,
            timeout=timeout,
            allow_redirects=allow_redirects,
        )
        elapsed_ms = (time.monotonic() - started) * 1000
        try:
            parsed = response.json()
        except ValueError:
            parsed = None
        return Response(
            status=response.status_code,
            data=parsed,
            headers={key.lower(): value for key, value in response.headers.items()},
            text=response.text,
            elapsed_ms=elapsed_ms,
        )

    def get(self, path: str, **kwargs: Any) -> Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, json_body: Any = None, **kwargs: Any) -> Response:
        return self.request("POST", path, json_body=json_body, **kwargs)

    def patch(self, path: str, json_body: Any = None, **kwargs: Any) -> Response:
        return self.request("PATCH", path, json_body=json_body, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> Response:
        return self.request("DELETE", path, **kwargs)

    # -- chat / SSE --------------------------------------------------------

    def chat(self, payload: dict[str, Any], *, timeout: float = 30.0) -> "SSEChat":
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=(5.0, timeout),
            headers={"Accept": "text/event-stream"},
        )
        return SSEChat(response)

    def wait_health(self, timeout: float = 10.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            payload = http_json(f"{self.base_url}/health")
            if payload:
                return payload
            time.sleep(0.05)
        raise TimeoutError("health endpoint did not answer")


# -- SSE parsing -------------------------------------------------------------


def parse_sse_block(block: str) -> tuple[str | None, dict[str, Any] | None]:
    """Parse one SSE block into (event, data). Returns (None, None) for comments."""
    event: str | None = None
    data_lines: list[str] = []
    saw_content = False
    for line in block.split("\n"):
        line = line.rstrip("\r")
        if not line:
            continue
        if line.startswith(":"):
            continue
        saw_content = True
        if ":" in line:
            field, value = line.split(":", 1)
            if value.startswith(" "):
                value = value[1:]
        else:
            field, value = line, ""
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)
    if not saw_content or not data_lines:
        return None, None
    raw = "\n".join(data_lines)
    try:
        data = json.loads(raw)
    except ValueError:
        return event, {"_raw": raw}
    return event or "message", data


def parse_sse_text(text: str) -> tuple[list[tuple[str, dict[str, Any]]], list[str]]:
    """Parse a complete SSE body into events + comment lines (for raw API tests)."""
    events: list[tuple[str, dict[str, Any]]] = []
    comments: list[str] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        if not block.strip():
            continue
        had_data = any(line.startswith("data:") for line in block.splitlines())
        if not had_data:
            for line in block.splitlines():
                if line.startswith(":"):
                    comments.append(line)
            continue
        event, data = parse_sse_block(block)
        if event and data is not None:
            events.append((event, data))
    return events, comments


class SSEChat:
    """Reads an SSE chat response on a background thread.

    Allows deterministic tests to wait for specific events (e.g. an approval)
    while another HTTP request resolves it.
    """

    def __init__(self, response: requests.Response) -> None:
        self.response = response
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.comments: list[str] = []
        self.raw_frames: list[str] = []
        self.error: BaseException | None = None
        self.closed = False
        self._scanned = 0
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._read, name="sse-reader", daemon=True)
        self._thread.start()

    # -- reader ------------------------------------------------------------

    def _read(self) -> None:
        buffer = ""
        try:
            for line in self.response.iter_lines(decode_unicode=True):
                if line is None:
                    continue
                line = line if isinstance(line, str) else line.decode("utf-8", "replace")
                if line == "":
                    if buffer:
                        self._dispatch(buffer)
                        buffer = ""
                    continue
                if line.startswith(":"):
                    with self._cond:
                        self.comments.append(line)
                        self._cond.notify_all()
                    continue
                buffer += line + "\n"
            if buffer.strip():
                self._dispatch(buffer)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the test
            self.error = exc
        finally:
            with self._cond:
                self.closed = True
                self._cond.notify_all()
            try:
                self.response.close()
            except Exception:
                pass

    def _dispatch(self, block: str) -> None:
        event, data = parse_sse_block(block)
        if event is None or data is None:
            return
        with self._cond:
            self.events.append((event, data))
            self.raw_frames.append(block)
            self._cond.notify_all()

    # -- waiting -----------------------------------------------------------

    def wait_for(self, event_name: str, timeout: float = 20.0) -> dict[str, Any]:
        """Wait until an event with this name arrives (scanning unread events)."""
        name, data = self.wait_any((event_name,), timeout=timeout)
        return data

    def wait_any(self, event_names: tuple[str, ...], timeout: float = 20.0) -> tuple[str, dict[str, Any]]:
        """Wait for the first unread event whose name is in ``event_names``."""
        wanted = set(event_names)
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                while self._scanned < len(self.events):
                    name, data = self.events[self._scanned]
                    self._scanned += 1
                    if name in wanted:
                        return name, data
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    seen = [name for name, _ in self.events]
                    raise TimeoutError(
                        f"timed out waiting for SSE event {sorted(wanted)!r}; saw {seen}"
                        + (f"; reader error: {self.error!r}" if self.error else "")
                    )
                if self.closed:
                    raise TimeoutError(
                        f"SSE stream closed before event {sorted(wanted)!r}; saw "
                        f"{[name for name, _ in self.events]}"
                        + (f"; reader error: {self.error!r}" if self.error else "")
                    )
                self._cond.wait(min(remaining, 0.5))

    def wait_done(self, timeout: float = 25.0) -> dict[str, Any]:
        """Wait for a terminal ``done`` or ``error`` event."""
        _name, data = self.wait_any(("done", "error"), timeout=timeout)
        return data

    def wait_closed(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("SSE stream did not close")
                self._cond.wait(min(remaining, 0.5))
        self._thread.join(timeout=2)

    def wait_comment(self, timeout: float = 20.0) -> str:
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self.comments:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("no SSE heartbeat comment seen")
                self._cond.wait(min(remaining, 0.5))
            return self.comments[0]

    # -- inspection --------------------------------------------------------

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def of(self, event_name: str) -> list[dict[str, Any]]:
        return [data for name, data in self.events if name == event_name]

    def text(self) -> str:
        return "".join(str(data.get("text", "")) for data in self.of("delta"))

    def close(self) -> None:
        try:
            self.response.close()
        except Exception:
            pass
        with self._cond:
            self.closed = True
            self._cond.notify_all()

    def wait_until_user_visible(self) -> None:
        """Best-effort: wait for meta; used by helpers."""
        try:
            self.wait_for("meta", timeout=5)
        except TimeoutError:
            pass


# ---------------------------------------------------------------------------
# check-result reporting
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    group: str
    ok: bool
    duration_ms: int
    detail: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "ok": self.ok,
            "duration_ms": self.duration_ms,
            "detail": self.detail,
            "error": self.error,
        }


@dataclass
class Runner:
    verbose: bool = True
    results: list[CheckResult] = field(default_factory=list)

    def check(self, group: str, name: str, func: Callable[..., str | None], *args: Any, **kwargs: Any) -> CheckResult:
        started = time.monotonic()
        try:
            detail = func(*args, **kwargs)
            result = CheckResult(
                name=name,
                group=group,
                ok=True,
                duration_ms=int((time.monotonic() - started) * 1000),
                detail=detail or "",
            )
        except Exception as exc:  # noqa: BLE001 - every failure becomes evidence
            result = CheckResult(
                name=name,
                group=group,
                ok=False,
                duration_ms=int((time.monotonic() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )
            if self.verbose:
                print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
                if os.environ.get("KIWIBRO_E2E_TRACEBACKS") == "1":
                    traceback.print_exc()
        self.results.append(result)
        if self.verbose and result.ok:
            print(f"  ok   {name} ({result.duration_ms} ms)")
        return result

    @property
    def failures(self) -> list[CheckResult]:
        return [result for result in self.results if not result.ok]

    def write_evidence(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "total": len(self.results),
            "passed": len(self.results) - len(self.failures),
            "failed": len(self.failures),
            "checks": [result.to_dict() for result in self.results],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        lines = [
            "| # | check | group | result | ms | note |",
            "|---|-------|-------|--------|----|------|",
        ]
        for index, result in enumerate(self.results, start=1):
            status = "PASS" if result.ok else "FAIL"
            note = (result.error if not result.ok else result.detail).replace("|", "\\|")
            if len(note) > 160:
                note = note[:157] + "..."
            lines.append(
                f"| {index} | {result.name} | {result.group} | {status} | {result.duration_ms} | {note} |"
            )
        path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# assertion helpers used by check modules
# ---------------------------------------------------------------------------


def assert_eq(actual: Any, expected: Any, what: str = "value") -> None:
    if actual != expected:
        raise AssertionError(f"{what}: expected {expected!r}, got {actual!r}")


def assert_true(value: Any, what: str = "condition") -> None:
    if not value:
        raise AssertionError(f"{what}: expected truthy, got {value!r}")


def assert_in(needle: Any, haystack: Iterable[Any], what: str = "value") -> None:
    if needle not in haystack:
        raise AssertionError(f"{what}: expected {needle!r} in {list(haystack)!r}")


def error_code(response: Response) -> str | None:
    data = response.data
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        return data["error"].get("code")
    return None


def expect_status(response: Response, status: int, what: str = "request") -> None:
    if response.status != status:
        raise AssertionError(
            f"{what}: expected HTTP {status}, got {response.status} ({response.text[:200]!r})"
        )


def expect_error(response: Response, status: int, code: str, what: str = "request") -> None:
    expect_status(response, status, what)
    actual = error_code(response)
    if actual != code:
        raise AssertionError(f"{what}: expected error code {code!r}, got {actual!r}")


def collect_stream(chat: SSEChat, timeout: float = 30.0) -> list[tuple[str, dict[str, Any]]]:
    """Wait for the terminal event, then return all events."""
    chat.wait_done(timeout=timeout)
    chat.wait_closed(timeout=timeout)
    return chat.events


def wait_for_busy(client: ApiClient, conversation_id: str, message: str, *, timeout: float = 15.0) -> SSEChat:
    """Start a chat and wait until the server accepted it (meta event)."""
    chat = client.chat({"conversation_id": conversation_id, "message": message}, timeout=timeout + 10)
    try:
        chat.wait_for("meta", timeout=timeout)
    except TimeoutError:
        chat.close()
        raise
    return chat
