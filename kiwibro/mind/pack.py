"""The Kiwi Mind pack on disk: what is installed, whether it is intact, and the download.

The pack lives in ``<data dir>/mind/``: ``manifest.json`` (written by the training script's export step) lists every
file with its size and sha256, and the files themselves (trunk, skills, head, tokenizer) sit next to it. A file is
only ever under its final name when it is complete and checked; a download works in ``<name>.part`` and resumes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

log = logging.getLogger(__name__)

REPO = "yunusemrejr/kiwimind"
HF_URL = "https://huggingface.co"
MANIFEST = "manifest.json"
TRUNK = "kiwimind-trunk.fkm"
HEAD = "head.kmh"
TOKENIZER = "tokenizer.json"
CHECK_CACHE = ".verified.json"
CHUNK = 1 << 20
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 60
USER_AGENT = "Kiwibro/1.0 (local)"


class PackError(Exception):
    """A pack problem with a calm message for the user."""


@dataclass(frozen=True)
class PackState:
    installed: bool  # a manifest is there
    ok: bool  # every file is there, whole and matching its checksum
    reason: str  # why it is not ok ("" when ok)


def _sha256(path: Path, stop: threading.Event | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            if stop is not None and stop.is_set():
                raise PackError("cancelled")
            digest.update(block)
    return digest.hexdigest()


def _safe_name(name: str) -> str:
    """A manifest path that stays inside the pack folder."""
    parts = Path(name).parts
    if not parts or Path(name).is_absolute() or ".." in parts:
        raise PackError(f"The pack lists an unsafe file name: {name}")
    return name


class Pack:
    def __init__(self, data_dir: str | Path, *, base_url: str | None = None, repo: str = REPO,
                 session: requests.Session | None = None) -> None:
        self.dir = Path(data_dir).expanduser() / "mind"
        self.base_url = (base_url or os.environ.get("KIWIMIND_HF_URL") or HF_URL).rstrip("/")
        self.repo = repo
        self._session = session if session is not None else requests.Session()
        self._lock = threading.Lock()
        self._memo: tuple[tuple[Any, ...], PackState] | None = None
        self._job: _Download | None = None
        self.on_installed: Callable[[], None] | None = None  # called when a download has put a pack in place

    # --- what is installed ---------------------------------------------------

    def manifest(self) -> dict[str, Any] | None:
        try:
            data = json.loads((self.dir / MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        files = data.get("files") if isinstance(data, dict) else None
        if not isinstance(files, dict) or TRUNK not in files:
            return None
        return data

    def path(self, name: str) -> Path:
        return self.dir / _safe_name(name)

    def _signature(self, manifest: dict[str, Any]) -> tuple[Any, ...]:
        parts: list[Any] = [(self.dir / MANIFEST).stat().st_mtime_ns]
        for name in sorted(manifest["files"]):
            try:
                stat = self.path(name).stat()
                parts.append((name, stat.st_size, stat.st_mtime_ns))
            except (OSError, PackError):
                parts.append((name, None, None))
        return tuple(parts)

    def check(self) -> PackState:
        """Whether the installed pack is complete and intact. Cheap after the first time: a file is only hashed
        again when its size or modification time changed."""
        manifest = self.manifest()
        if manifest is None:
            return PackState(False, False, "Kiwi Mind is not installed.")
        with self._lock:
            try:
                signature = self._signature(manifest)
            except OSError:
                return PackState(False, False, "Kiwi Mind is not installed.")
            if self._memo is not None and self._memo[0] == signature:
                return self._memo[1]
            state = self._verify(manifest)
            self._memo = (self._signature(manifest), state)
            return state

    def _verify(self, manifest: dict[str, Any]) -> PackState:
        try:
            cache = json.loads((self.dir / CHECK_CACHE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
        fresh: dict[str, Any] = {}
        for name, info in manifest["files"].items():
            try:
                path = self.path(name)
                expected, digest = int(info["bytes"]), str(info["sha256"])
                stat = path.stat()
            except (PackError, KeyError, TypeError, ValueError):
                return PackState(True, False, "The Kiwi Mind files are damaged. Download them again.")
            except OSError:
                return PackState(True, False, f"Kiwi Mind is incomplete: {name} is missing.")
            if stat.st_size != expected:
                return PackState(True, False, f"Kiwi Mind is incomplete: {name} has the wrong size.")
            seen = cache.get(name) or {}
            if not (seen.get("size") == stat.st_size and seen.get("mtime_ns") == stat.st_mtime_ns
                    and seen.get("sha256") == digest):
                if _sha256(path) != digest:
                    return PackState(True, False, f"The Kiwi Mind files are damaged ({name}). Download them again.")
            fresh[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}
        try:
            (self.dir / CHECK_CACHE).write_text(json.dumps(fresh), encoding="utf-8")
        except OSError:
            log.warning("could not save the Kiwi Mind check result", exc_info=True)
        return PackState(True, True, "")

    def describe(self) -> dict[str, Any]:
        manifest = self.manifest()
        state = self.check()
        files = (manifest or {}).get("files") or {}
        return {
            "installed": state.installed,
            "ok": state.ok,
            "reason": state.reason,
            "version": (manifest or {}).get("version"),
            "bytes": sum(int(info.get("bytes") or 0) for info in files.values()),
            "skills": list((manifest or {}).get("skills") or []),
            "ctx": int((manifest or {}).get("ctx") or 0),
        }

    # --- download -------------------------------------------------------------

    def download_state(self) -> dict[str, Any] | None:
        with self._lock:
            job = self._job
        return job.view() if job is not None else None

    def start_download(self) -> dict[str, Any]:
        """Start (or resume) the download; returns its state. A download already running is left alone."""
        with self._lock:
            if self._job is not None and self._job.running():
                return self._job.view()
            job = self._job = _Download(self)
        job.start()
        return job.view()

    def cancel_download(self) -> bool:
        with self._lock:
            job = self._job
        if job is None or not job.running():
            return False
        job.stop.set()
        return True

    def shutdown(self) -> None:
        self.cancel_download()
        with self._lock:
            job = self._job
        if job is not None and job.thread is not None:
            job.thread.join(5)

    def _url(self, name: str) -> str:
        return f"{self.base_url}/{self.repo}/resolve/main/{_safe_name(name)}"

    def _get(self, name: str, offset: int) -> requests.Response:
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        return self._session.get(self._url(name), headers=headers, stream=True,
                                 timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))


class _Download:
    def __init__(self, pack: Pack) -> None:
        self.pack = pack
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.state = "running"
        self.done = 0
        self.total = 0
        self.error = ""

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="kiwibro-mind-download", daemon=True)
        self.thread.start()

    def running(self) -> bool:
        return self.state == "running"

    def view(self) -> dict[str, Any]:
        with self.lock:
            out: dict[str, Any] = {"state": self.state, "done_bytes": self.done, "total_bytes": self.total}
            if self.error:
                out["error"] = self.error
            return out

    def _set(self, **fields: Any) -> None:
        with self.lock:
            for key, value in fields.items():
                setattr(self, key, value)

    def _run(self) -> None:
        try:
            self._download_all()
            self._set(state="done")
            if self.pack.on_installed is not None:
                self.pack.on_installed()
        except _Cancelled:
            self._set(state="cancelled")
        except (PackError, OSError, requests.RequestException) as exc:
            log.warning("Kiwi Mind download failed: %s", exc)
            message = str(exc) if isinstance(exc, PackError) else (
                "The download stopped. Check the connection and start it again to resume.")
            self._set(state="error", error=message)

    def _download_all(self) -> None:
        pack = self.pack
        pack.dir.mkdir(parents=True, exist_ok=True)
        # The manifest first: it says what else to fetch. A new one replaces an old one only once complete.
        self._fetch(MANIFEST, None, None, count=False)
        manifest = pack.manifest()
        if manifest is None:
            raise PackError("The download is not a Kiwi Mind pack.")
        files = {_safe_name(name): info for name, info in manifest["files"].items()}
        self._set(total=sum(int(info["bytes"]) for info in files.values()))
        for name, info in sorted(files.items(), key=lambda item: item[0] != TOKENIZER):
            target = pack.path(name)
            size, digest = int(info["bytes"]), str(info["sha256"])
            if target.exists() and target.stat().st_size == size:
                with self.lock:
                    self.done += size
                continue
            self._fetch(name, size, digest)

    def _fetch(self, name: str, size: int | None, digest: str | None, *, count: bool = True) -> None:
        pack = self.pack
        target = pack.path(name)
        part = target.with_name(target.name + ".part")
        target.parent.mkdir(parents=True, exist_ok=True)
        offset = part.stat().st_size if part.exists() else 0
        if size is not None and offset > size:
            part.unlink()
            offset = 0
        if size is None or offset < size:
            self._transfer(name, part, offset, size, count)
        if size is not None and part.stat().st_size != size:
            raise PackError("The download stopped early. Start it again to resume.")
        if digest is not None:
            try:
                ok = _sha256(part, self.stop) == digest
            except PackError as exc:
                raise _Cancelled from exc
            if not ok:
                part.unlink(missing_ok=True)
                raise PackError(f"The downloaded file {name} did not match its checksum. Start it again.")
        os.replace(part, target)

    def _transfer(self, name: str, part: Path, offset: int, size: int | None, count: bool) -> None:
        response = self.pack._get(name, offset)
        try:
            if response.status_code == 416:  # the part is already past what the server has: start over
                response.close()
                part.unlink(missing_ok=True)
                offset = 0
                response = self.pack._get(name, 0)
            if response.status_code == 404:
                raise PackError("Kiwi Mind is not available to download yet.")
            if response.status_code not in (200, 206):
                raise PackError(f"The download server answered {response.status_code}.")
            if response.status_code == 200:
                offset = 0  # the server ignored Range and sent everything
            if count:
                with self.lock:
                    self.done += offset
            with part.open("ab" if offset else "wb") as out:
                for block in response.iter_content(CHUNK):
                    if self.stop.is_set():
                        raise _Cancelled
                    if block:
                        out.write(block)
                        if count:
                            with self.lock:
                                self.done += len(block)
        finally:
            response.close()


class _Cancelled(Exception):
    pass
