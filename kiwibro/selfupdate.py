"""Keep a running Kiwibro on the newest code.

The server can outlive its window (closing the app window leaves it running), so an
update to the source tree would otherwise keep serving old code until someone kills it.
Two pieces close that gap:

* ``fingerprint(root)`` hashes the served source (Python, templates, static assets). The
  launcher compares it with the running server's ``code_version`` and restarts a stale
  idle server before opening the window.
* ``Updater`` watches the tree from inside the server. Once the code has changed, stayed
  quiet for ``settle_s`` and the new code imports cleanly, it re-executes the server in
  place the moment no reply is running. Open windows see the new ``code_version`` and
  reload themselves.

Stdlib only: the launcher runs this file as a script (``python3 kiwibro/selfupdate.py
fingerprint <root>``) without importing the package or its dependencies.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

# What the server serves or runs. Everything else in the repo (tests, docs, data) can
# change without making a running server stale.
_ROOT_FILES = ("app.py", "requirements.txt")
_TREES = ("kiwibro", "templates", "static")
_SUFFIXES = {
    ".py", ".html", ".js", ".mjs", ".css", ".json", ".svg", ".png", ".ico", ".woff2", ".txt",
}


def source_files(root: Path) -> list[Path]:
    root = Path(root)
    files = [root / name for name in _ROOT_FILES if (root / name).is_file()]
    for tree in _TREES:
        base = root / tree
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__" and not d.startswith("."))
            for name in filenames:
                if Path(name).suffix in _SUFFIXES and not name.startswith("."):
                    files.append(Path(dirpath) / name)
    return sorted(files)


def stat_signature(root: Path) -> tuple:
    """Cheap change detector: (path, size, mtime) of every source file."""
    sig = []
    for path in source_files(root):
        try:
            st = path.stat()
        except OSError:
            continue
        sig.append((str(path.relative_to(root)), st.st_size, st.st_mtime_ns))
    return tuple(sig)


def fingerprint(root: Path) -> str:
    """Content hash of the served source; equal trees give equal fingerprints."""
    root = Path(root)
    digest = hashlib.sha256()
    for path in source_files(root):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()[:16]


def preflight(root: Path, python: str | None = None, timeout: float = 90.0) -> tuple[bool, str]:
    """Does the code on disk import? A half-saved edit must not take the server down."""
    code = (
        "import pathlib, sys\n"
        "root = pathlib.Path(sys.argv[1]); sys.path.insert(0, str(root))\n"
        "for path in [root / 'app.py', *sorted((root / 'kiwibro').rglob('*.py'))]:\n"
        "    if path.is_file() and '__pycache__' not in path.parts:\n"
        "        compile(path.read_bytes(), str(path), 'exec')\n"
        "import kiwibro, kiwibro.api, kiwibro.agent, kiwibro.selfupdate\n"
    )
    try:
        proc = subprocess.run(
            [python or sys.executable, "-c", code, str(root)],
            cwd=str(root), capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "KIWIBRO_WARMUP": "0"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "").strip()[-2000:]
    return True, ""


def _restart_argv() -> list[str]:
    argv = list(getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv])
    # Keep the interpreter path we were started with (the launcher's ``kiwibro`` link
    # names the process); orig_argv[0] may be a bare name.
    argv[0] = sys.executable or argv[0]
    return argv


class Updater:
    """Re-executes the server on new code when it is idle (see module docstring)."""

    def __init__(
        self,
        root: Path,
        busy_fn: Callable[[], bool],
        *,
        poll_s: float = 3.0,
        settle_s: float = 15.0,
        exec_fn: Callable[[list[str]], None] | None = None,
        preflight_fn: Callable[[Path], tuple[bool, str]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.root = Path(root)
        self.busy_fn = busy_fn
        self.poll_s = poll_s
        self.settle_s = settle_s
        self._exec = exec_fn or self._execv
        self._preflight = preflight_fn or preflight
        self._clock = clock
        self.code_version = fingerprint(self.root)
        self._signature = stat_signature(self.root)
        self._changed_at: float | None = None
        self._disk_version = self.code_version
        self._failed_version: str | None = None  # new code that did not import; wait for the next edit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- state -------------------------------------------------------------

    @property
    def update_pending(self) -> bool:
        return self._disk_version != self.code_version

    def status(self) -> dict:
        return {
            "code_version": self.code_version,
            "update_pending": self.update_pending,
            "update_blocked": self._failed_version is not None
            and self._failed_version == self._disk_version,
        }

    # --- loop --------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="kiwibro-selfupdate", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                self.check()
            except Exception:  # never let the watcher die
                log.exception("self-update check failed")

    def check(self) -> bool:
        """One poll. Returns True when it re-executed (only observable with a fake exec)."""
        signature = stat_signature(self.root)
        now = self._clock()
        if signature != self._signature:
            self._signature = signature
            self._changed_at = now
            self._disk_version = fingerprint(self.root)
            return False
        if not self.update_pending or self._changed_at is None:
            return False
        if now - self._changed_at < self.settle_s:
            return False  # still being edited / checked out
        if self._disk_version == self._failed_version:
            return False
        if self.busy_fn():
            return False  # never cut a running reply; retry next poll
        ok, error = self._preflight(self.root)
        if not ok:
            self._failed_version = self._disk_version
            log.warning("new code does not import; staying on %s: %s", self.code_version, error)
            return False
        log.info("code changed (%s -> %s); restarting on the new version",
                 self.code_version, self._disk_version)
        self._exec(_restart_argv())
        return True

    @staticmethod
    def _execv(argv: list[str]) -> None:
        for handler in logging.getLogger().handlers:
            try:
                handler.flush()
            except Exception:
                pass
        sys.stdout.flush()
        sys.stderr.flush()
        # Werkzeug hands its listening socket over as an inheritable fd (for its reloader);
        # left open, the new image could not bind the port again. Keep only stdio.
        os.environ.pop("WERKZEUG_SERVER_FD", None)
        os.environ.pop("WERKZEUG_RUN_MAIN", None)
        os.closerange(3, 65536)
        os.execv(argv[0], argv)


def set_process_name(name: str = "kiwibro") -> None:
    """Name the process for system monitors (``comm``) when not started via the launcher link."""
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(15, name.encode()[:15], 0, 0, 0)  # PR_SET_NAME
    except Exception:
        pass


def _main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "fingerprint":
        root = Path(argv[2]) if len(argv) > 2 else Path(__file__).resolve().parent.parent
        print(fingerprint(root))
        return 0
    print("usage: selfupdate.py fingerprint [root]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
