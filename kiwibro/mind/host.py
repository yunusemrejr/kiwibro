"""The Kiwi Mind host process: one Node.js program that holds the model once and serves every conversation.

Kiwibro starts it when it is needed, gives it a random key, and stops it with the app. It dies with its parent
(``PR_SET_PDEATHSIG`` and a stdin pipe that closes when Kiwibro goes), so a killed app leaves nothing behind."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import secrets
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pack import HEAD, TOKENIZER, TRUNK, Pack

log = logging.getLogger(__name__)

HOST_SCRIPT = Path(__file__).resolve().parent / "host" / "host.mjs"
NODE_MIN = (18, 14)  # os.availableParallelism, worker threads and the shared-memory WebAssembly the host uses
SLOTS = 4
READY_TIMEOUT_S = 90.0
LOG_LIMIT = 256 * 1024
STOP_WAIT_S = 5.0
_VERSION_RE = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")


class HostError(Exception):
    """Kiwi Mind's host could not be started or died; the message is fit for the user."""


def node_candidates() -> list[str]:
    found: list[str] = []
    on_path = shutil.which("node")
    if on_path:
        found.append(on_path)
    nvm = sorted(Path.home().glob(".nvm/versions/node/*/bin/node"), key=_nvm_version, reverse=True)
    found += [str(path) for path in nvm[:1]]
    found += ["/usr/bin/node", "/usr/local/bin/node", str(Path.home() / ".local" / "bin" / "node")]
    seen: set[str] = set()
    return [path for path in found if not (path in seen or seen.add(path)) and os.access(path, os.X_OK)]


def _nvm_version(path: Path) -> tuple[int, ...]:
    found = _VERSION_RE.search(path.parent.parent.name)
    return tuple(int(part) for part in found.groups()) if found else (0, 0, 0)


def node_version(path: str) -> tuple[int, int, int] | None:
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    found = _VERSION_RE.search(out)
    return (int(found.group(1)), int(found.group(2)), int(found.group(3))) if found else None


def find_node() -> tuple[str | None, str]:
    """(path, "") for the first Node.js that is new enough, else (None, why not)."""
    old = ""
    for path in node_candidates():
        version = node_version(path)
        if version is None:
            continue
        if version[:2] >= NODE_MIN:
            return path, ""
        old = (f"Node.js {version[0]}.{version[1]} is too old for Kiwi Mind. "
               f"Version {NODE_MIN[0]}.{NODE_MIN[1]} or newer is needed.")
    return None, old or "Kiwi Mind needs Node.js, which is not installed."


def default_threads() -> int:
    return max(2, min(8, (os.cpu_count() or 4) // 2))


@dataclass
class _Running:
    process: subprocess.Popen[bytes]
    port: int
    key: str
    pid: int
    started_at: float
    ready_ms: int


class MindHost:
    """Starts, watches and stops the host process. One per app."""

    def __init__(self, pack: Pack, *, node: str | None = None, threads: int | None = None) -> None:
        self.pack = pack
        self._node = node
        self.threads = threads or default_threads()
        self._lock = threading.RLock()
        self._running: _Running | None = None
        self._last_used = time.monotonic()
        self._in_flight = 0
        self._restarts: dict[str, int] = {}

    # --- state ---------------------------------------------------------------

    @property
    def log_path(self) -> Path:
        return self.pack.dir / "host.log"

    def node(self) -> tuple[str | None, str]:
        if self._node:
            return self._node, ""
        return find_node()

    def alive(self) -> bool:
        running = self._running
        return running is not None and running.process.poll() is None

    def describe(self) -> dict[str, Any]:
        running = self._running if self.alive() else None
        return {
            "running": running is not None,
            "pid": running.pid if running else None,
            "rss_bytes": _rss(running.pid) if running else None,
            "ready_ms": running.ready_ms if running else None,
        }

    # --- start and stop -------------------------------------------------------

    def ensure(self) -> _Running:
        """The running host, starting it if it is not. Raises HostError with a clear message."""
        with self._lock:
            if self.alive():
                assert self._running is not None
                return self._running
            if self._running is not None:
                log.warning("Kiwi Mind host (pid %s) stopped; starting it again", self._running.pid)
                self._running = None
            self._running = self._start()
            return self._running

    def restart_after_failure(self, failed: _Running | None) -> _Running:
        """The host died under a request: start a new one once for that request, else fail."""
        with self._lock:
            if self._running is not None and self._running is not failed and self.alive():
                return self._running  # somebody else already restarted it
            self.stop()
            return self.ensure()

    def _start(self) -> _Running:
        from ..engine import _SPAWNER, _die_with_parent

        state = self.pack.check()
        if not state.ok:
            raise HostError(state.reason)
        manifest = self.pack.manifest() or {}
        node, why = self.node()
        if node is None:
            raise HostError(why)
        args = [
            node, str(HOST_SCRIPT),
            "--model", str(self.pack.path(TRUNK)),
            "--tokenizer", str(self.pack.path(TOKENIZER)),
            "--threads", str(self.threads),
            "--ctx", str(int(manifest.get("ctx") or 4096)),
            "--slots", str(SLOTS),
        ]
        if (self.pack.dir / HEAD).exists():
            args += ["--head", str(self.pack.path(HEAD))]
        for name in manifest.get("skills") or []:
            args += ["--skill", f"{name}={self.pack.path(f'skills/{name}.kms')}"]
        key = secrets.token_urlsafe(24)
        env = {**os.environ, "KIWIMIND_KEY": key}
        log_file = self._open_log()

        def spawn() -> subprocess.Popen[bytes]:
            return subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log_file, env=env,
                preexec_fn=_die_with_parent, close_fds=True,
            )

        began = time.monotonic()
        try:
            process = _SPAWNER.run(spawn)
        except OSError as exc:
            raise HostError(f"Kiwi Mind could not start: {exc}") from exc
        finally:
            log_file.close()
        try:
            ready = self._read_ready(process)
        except HostError:
            self._kill(process)
            raise
        threading.Thread(target=_drain, args=(process,), name="kiwibro-mind-stdout", daemon=True).start()
        elapsed = int((time.monotonic() - began) * 1000)
        log.info("Kiwi Mind host started (pid %d, port %d, %d threads) in %d ms",
                 process.pid, ready["port"], self.threads, elapsed)
        return _Running(process, int(ready["port"]), key, process.pid, time.time(), elapsed)

    def _open_log(self) -> Any:
        self.pack.dir.mkdir(parents=True, exist_ok=True)
        path = self.log_path
        try:
            if path.exists() and path.stat().st_size > LOG_LIMIT:
                path.write_bytes(path.read_bytes()[-LOG_LIMIT // 4:])
        except OSError:
            pass
        return path.open("ab")

    def _read_ready(self, process: subprocess.Popen[bytes]) -> dict[str, Any]:
        lines: queue.Queue[bytes] = queue.Queue()

        def read() -> None:
            assert process.stdout is not None
            lines.put(process.stdout.readline())

        threading.Thread(target=read, name="kiwibro-mind-ready", daemon=True).start()
        deadline = time.monotonic() + READY_TIMEOUT_S
        while True:
            try:
                line = lines.get(timeout=0.2)
                break
            except queue.Empty:
                if process.poll() is not None or time.monotonic() > deadline:
                    line = b""
                    break
        try:
            ready = json.loads(line.decode("utf-8"))
        except ValueError:
            ready = None
        if not isinstance(ready, dict) or not ready.get("ready"):
            raise HostError(f"Kiwi Mind did not start. Its log is {self.log_path}.")
        return ready

    def stop(self) -> None:
        with self._lock:
            running, self._running = self._running, None
        if running is not None:
            self._kill(running.process)
            log.info("Kiwi Mind host stopped (pid %d)", running.pid)

    @staticmethod
    def _kill(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.close()  # the host leaves when its stdin ends
            except OSError:
                pass
            try:
                process.wait(STOP_WAIT_S)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(STOP_WAIT_S)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(STOP_WAIT_S)
        for stream in (process.stdout, process.stdin):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    # --- use accounting (for the idle rule) -------------------------------------

    def busy(self) -> "_Use":
        return _Use(self)

    def stop_idle(self, min_idle_s: float) -> int:
        with self._lock:
            if not self.alive() or self._in_flight or time.monotonic() - self._last_used < min_idle_s:
                return 0
        self.stop()
        return 1


class _Use:
    """Marks the host as in use for the length of a request."""

    def __init__(self, host: MindHost) -> None:
        self.host = host

    def __enter__(self) -> "_Use":
        with self.host._lock:
            self.host._in_flight += 1
        return self

    def __exit__(self, *_exc: object) -> None:
        with self.host._lock:
            self.host._in_flight -= 1
            self.host._last_used = time.monotonic()


def _drain(process: subprocess.Popen[bytes]) -> None:
    """The host prints nothing after its ready line; if it ever does, read it so the pipe never fills."""
    try:
        assert process.stdout is not None
        while process.stdout.read(4096):
            pass
    except (OSError, ValueError):
        pass


def _rss(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None
