"""Model hub: find GGUF models on Hugging Face, download them resumably, and free the disk again.

Every model lives at ``models/author/name/file.gguf`` so the engine finds it and deleting
a model can remove its folder again. Downloads are written to a ``.part`` file first, so a
dropped connection or a cancel never leaves a half-written model where the engine could
load it, and the partial bytes are kept so the next attempt continues where it stopped.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import logging
import os
import re
import secrets
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

log = logging.getLogger(__name__)

HF_URL = "https://huggingface.co"
USER_AGENT = "kiwibro"
REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
GIB = 1024**3
DISK_RESERVE = 512 * 1024 * 1024  # left free after a download, for the OS and the engine's own files
CHUNK = 256 * 1024
HASH_BLOCK = 1024 * 1024
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 60
PROGRESS_EVERY = 0.25  # seconds; the UI polls at about this rate, so faster updates are invisible
SHUTDOWN_WAIT = 10.0

_QUANT_RE = re.compile(r"(?<![A-Z0-9])(BF16|F16|F32|IQ\d_[A-Z]{1,4}|Q\d(?:_[A-Z0-9]+)*)(?![A-Z0-9])")
_SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


class HubError(Exception):
    """A failure the UI can show. ``code`` is one of offline, not_found, bad_request, no_space, busy, failed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def quant_of(filename: str) -> str:
    """The quantization written in a GGUF name ("Q4_K_M"), or "" when the name has none."""
    match = _QUANT_RE.search(_basename(filename).upper())
    return match.group(1) if match else ""


def is_projector(filename: str) -> bool:
    """Vision projectors (mmproj) belong to a model next to them and are not models on their own."""
    return "mmproj" in _basename(filename).lower()


def split_part(filename: str) -> tuple[int, int] | None:
    """(part, total) for one piece of a split GGUF, None for a whole file."""
    match = _SPLIT_RE.search(_basename(filename))
    if match is None:
        return None
    part, total = int(match.group(1)), int(match.group(2))
    return (part, total) if 1 <= part <= total else None


def fit(size_bytes: int, ram_bytes: int) -> str:
    """Whether a model runs comfortably in RAM: the file plus overhead, leaving room for the OS."""
    need = size_bytes * 1.25 + GIB
    if need <= ram_bytes * 0.6:
        return "good"
    if need <= ram_bytes * 0.85:
        return "tight"
    return "too_big"


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _gib(num_bytes: int) -> str:
    return f"{num_bytes / GIB:.1f} GiB"


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _nearest_existing(path: Path) -> Path:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return probe


def _valid_repo(repo: str) -> bool:
    if not REPO_RE.fullmatch(repo):
        return False
    return all(part not in (".", "..") for part in repo.split("/"))


def _split_repo(repo: str) -> tuple[str, str]:
    if not isinstance(repo, str) or not _valid_repo(repo):
        raise HubError("bad_request", "A model id looks like author/model-name.")
    author, name = repo.split("/", 1)
    return author, name


def _anonymous(request: requests.PreparedRequest) -> requests.PreparedRequest:
    """Passed as ``auth`` so requests does not add credentials from ~/.netrc to a public request."""
    return request


class _PublicSession(requests.Session):
    """Redirects re-read ~/.netrc inside requests; the hub only ever talks to public repos."""

    def rebuild_auth(self, prepared_request: Any, response: Any) -> None:
        return None


@dataclass(eq=False)
class _Job:
    id: str
    repo: str
    file: str
    name: str
    path: Path
    part: Path
    url: str
    size_bytes: int = 0
    sha256: str = ""
    state: str = "queued"
    received: int = 0
    total: int = 0
    speed: float = 0.0
    eta_s: int | None = None
    error: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    mark_at: float = 0.0
    mark_bytes: int = 0


class _Cancelled(Exception):
    """Raised inside a transfer once the job was cancelled; the partial file is kept."""


class _Failed(Exception):
    def __init__(self, message: str, *, discard: bool = False) -> None:
        super().__init__(message)
        self.discard = discard  # the partial bytes are wrong, not merely unfinished


def _view(job: _Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "repo": job.repo,
        "file": job.file,
        "name": job.name,
        "state": job.state,
        "received": job.received,
        "total": job.total,
        "speed_bps": int(job.speed),
        "eta_s": job.eta_s,
        "error": job.error,
        "path": str(job.path),
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }


def _new_id(taken: dict[str, _Job]) -> str:
    while True:
        job_id = "dl_" + secrets.token_hex(4)
        if job_id not in taken:
            return job_id


def _mark_cancelled(job: _Job) -> None:
    job.cancel.set()
    job.state = "cancelled"
    job.speed = 0.0
    job.eta_s = None
    job.finished_at = time.time()


def _settle_locked(job: _Job, state: str, error: str = "") -> None:
    if job.finished_at is not None:  # a cancel that landed first wins
        return
    job.state, job.error, job.speed, job.eta_s = state, error, 0.0, None
    job.finished_at = time.time()


def _raise_for_status(resp: requests.Response) -> None:
    if resp.status_code == 404:
        raise _Failed("This file is no longer on Hugging Face.")
    if resp.status_code in (401, 403):
        raise _Failed("This model needs a Hugging Face login, which Kiwibro does not use.")
    if resp.status_code >= 400:
        raise _Failed(f"Hugging Face answered with HTTP {resp.status_code}. Start the download again later.")


def _content_total(resp: requests.Response, offset: int) -> int:
    length = resp.headers.get("Content-Length", "")
    if length.isdigit():
        return offset + int(length)
    for answer in [resp, *resp.history]:
        linked = answer.headers.get("x-linked-size", "")
        if linked.isdigit():
            return int(linked)
    return 0


def _disk_error(exc: OSError) -> str:
    if exc.errno == errno.ENOSPC:
        return "The disk is full. Free some space, then start the download again."
    return f"Could not write the file: {exc.strerror or exc}"


def _discard(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


class Hub:
    """Searches Hugging Face and downloads GGUF files into ``models_dir``, one at a time."""

    def __init__(self, models_dir: str | Path, *, base_url: str | None = None,
                 session: requests.Session | None = None) -> None:
        self.models_dir = Path(models_dir).expanduser()
        self.base_url = (base_url or os.environ.get("KIWIBRO_HF_URL") or HF_URL).rstrip("/")
        self._session = session if session is not None else _PublicSession()
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._jobs: dict[str, _Job] = {}
        self._queue: deque[_Job] = deque()
        self._worker: threading.Thread | None = None
        self._stop: threading.Event | None = None

    # Hugging Face API

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        # "conversational" keeps this to chat models: GGUF files also hold speech, music and
        # embedding models, which this engine cannot talk to.
        params: dict[str, Any] = {"filter": ["gguf", "conversational"], "sort": "downloads", "direction": -1,
                                  "limit": max(1, min(int(limit), 100))}
        text = (query or "").strip()
        if text:
            params["search"] = text
        out: list[dict[str, Any]] = []
        for item in self._get_json("/api/models", params):
            if not isinstance(item, dict):
                continue
            repo = str(item.get("id") or item.get("modelId") or "")
            if not _valid_repo(repo) or item.get("private") or item.get("gated"):
                continue
            author, name = repo.split("/", 1)
            raw_tags = item.get("tags")
            tags = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else []
            license_tag = next((tag for tag in tags if tag.startswith("license:")), "")
            out.append({
                "repo": repo,
                "author": author,
                "name": name,
                "downloads": _int(item.get("downloads")),
                "likes": _int(item.get("likes")),
                "updated": str(item.get("createdAt") or item.get("lastModified") or ""),
                "vision": "vision" in tags or "image" in str(item.get("pipeline_tag") or ""),
                "license": license_tag[len("license:"):],
            })
        return out

    def files(self, repo: str) -> list[dict[str, Any]]:
        _split_repo(repo)
        entries = self._get_json(f"/api/models/{quote(repo, safe='/')}/tree/main", {"recursive": "true"},
                                 missing="That repository was not found on Hugging Face, or it is private.")
        out: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("type") != "file":
                continue
            path = str(entry.get("path") or "")
            if not path.lower().endswith(".gguf"):
                continue
            lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else {}
            size = _int(lfs["size"]) if lfs.get("size") is not None else _int(entry.get("size"))
            try:
                target = self.target_path(repo, path)
            except HubError:
                continue
            name = _basename(path)
            out.append({
                "file": path,
                "name": name,
                "size_bytes": size,
                "sha256": str(lfs.get("oid") or "").lower(),
                "quant": quant_of(name),
                "projector": is_projector(name),
                "split": split_part(name) is not None,
                "installed": target.is_file() and target.stat().st_size == size,
            })
        out.sort(key=lambda item: (item["projector"], item["size_bytes"]))
        return out

    def _get_json(self, path: str, params: dict[str, Any] | None = None, missing: str | None = None) -> list[Any]:
        try:
            resp = self._session.get(self.base_url + path, params=params, headers={"User-Agent": USER_AGENT},
                                     timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), auth=_anonymous)
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise HubError("offline", "Could not reach Hugging Face. Check the internet connection.") from exc
        except requests.RequestException as exc:
            raise HubError("failed", f"Hugging Face request failed: {exc}") from exc
        if resp.status_code in (401, 404) and missing:  # 401 is what a visitor without an account gets
            raise HubError("not_found", missing)   # for a name that does not exist
        if resp.status_code >= 400:
            raise HubError("failed", f"Hugging Face answered with HTTP {resp.status_code}.")
        try:
            data = resp.json()
        except ValueError as exc:
            raise HubError("failed", "Hugging Face sent an answer Kiwibro could not read.") from exc
        if not isinstance(data, list):
            raise HubError("failed", "Hugging Face sent an answer Kiwibro could not read.")
        return data

    # Paths

    def target_path(self, repo: str, file: str) -> Path:
        author, name = _split_repo(repo)
        if (not file.lower().endswith(".gguf") or "\\" in file or file.startswith("/")
                or ".." in file.split("/") or not _basename(file)):
            raise HubError("bad_request", "That is not a GGUF file inside the repository.")
        target = self.models_dir / author / name / _basename(file)
        if not target.resolve().is_relative_to(self.models_dir.resolve()):
            raise HubError("bad_request", "That file would be saved outside the models folder.")
        return target

    def _url(self, repo: str, file: str) -> str:
        return f"{self.base_url}/{quote(repo, safe='/')}/resolve/main/{quote(file, safe='/')}"

    # Jobs

    def start(self, repo: str, file: str, *, sha256: str = "", size_bytes: int = 0) -> dict[str, Any]:
        """Queue a download, or return the job already running for it or the model already on disk."""
        target = self.target_path(repo, file)
        size = max(0, _int(size_bytes))
        digest = (sha256 or "").strip().lower()
        with self._lock:
            for job in self._jobs.values():
                if job.finished_at is None and job.repo == repo and job.file == file:
                    return _view(job)
            if target.is_file() and (not size or target.stat().st_size == size):
                done = self._new_job(_new_id(self._jobs), repo, file, target, size, digest)
                done.state = "done"
                done.received = done.total = _file_size(target)
                done.finished_at = time.time()
                return _view(done)
            free = shutil.disk_usage(_nearest_existing(self.models_dir)).free
            needed = size + DISK_RESERVE
            if free < needed:
                raise HubError("no_space", f"This model needs {_gib(needed)} free on the disk, "
                                           f"but only {_gib(free)} is free.")
            job = self._new_job(_new_id(self._jobs), repo, file, target, size, digest)
            self._jobs[job.id] = job
            self._queue.append(job)
            self._ensure_worker()
            self._wake.notify_all()
            return _view(job)

    def _new_job(self, job_id: str, repo: str, file: str, target: Path, size: int, digest: str) -> _Job:
        return _Job(id=job_id, repo=repo, file=file, name=_basename(file), path=target,
                    part=target.with_name(target.name + ".part"), url=self._url(repo, file),
                    size_bytes=size, sha256=digest, total=size)

    def jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [_view(job) for job in reversed(self._jobs.values())]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.finished_at is not None:
                return False
            _mark_cancelled(job)
            return True

    def clear_finished(self) -> int:
        with self._lock:
            finished = [job_id for job_id, job in self._jobs.items() if job.finished_at is not None]
            for job_id in finished:
                del self._jobs[job_id]
            return len(finished)

    def active(self) -> bool:
        with self._lock:
            return any(job.finished_at is None for job in self._jobs.values())

    def shutdown(self) -> None:
        with self._wake:
            for job in self._jobs.values():
                if job.finished_at is None:
                    _mark_cancelled(job)
            self._queue.clear()
            if self._stop is not None:
                self._stop.set()
            worker = self._worker
            self._wake.notify_all()
        if worker is not None and worker is not threading.current_thread():
            worker.join(SHUTDOWN_WAIT)

    # Worker

    def _ensure_worker(self) -> None:
        if (self._stop is not None and not self._stop.is_set()
                and self._worker is not None and self._worker.is_alive()):
            return
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._drain, args=(self._stop,), name="kiwibro-hub", daemon=True)
        self._worker.start()

    def _drain(self, stop: threading.Event) -> None:
        while True:
            with self._wake:
                while not self._queue and not stop.is_set():
                    self._wake.wait()
                if stop.is_set():
                    return
                job = self._queue.popleft()
                if job.finished_at is not None:
                    continue
                job.state = "downloading"
                job.started_at = time.time()
            self._run(job)

    def _run(self, job: _Job) -> None:
        log.info("Downloading %s from %s", job.file, job.repo)
        try:
            self._transfer(job)
            self._check_size(job)
            self._verify(job)
            self._install(job)
        except _Cancelled:
            log.info("Download of %s cancelled; the partial file is kept", job.file)
        except _Failed as exc:
            if exc.discard:
                _discard(job.part)
            self._settle(job, "error", str(exc))
        except requests.RequestException:
            log.warning("Connection lost while downloading %s", job.file)
            self._settle(job, "error", "The connection to Hugging Face was lost. Start it again to resume.")
        except OSError as exc:
            self._settle(job, "error", _disk_error(exc))
        except Exception:
            log.exception("Download of %s failed", job.file)
            self._settle(job, "error", "Unexpected error. The details are in the Kiwibro log.")

    def _settle(self, job: _Job, state: str, error: str = "") -> None:
        with self._lock:
            _settle_locked(job, state, error)

    def _range_get(self, url: str, offset: int) -> requests.Response:
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        return self._session.get(url, headers=headers, stream=True,
                                 timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), auth=_anonymous)

    def _transfer(self, job: _Job) -> None:
        offset = _file_size(job.part)
        resp = self._range_get(job.url, offset)
        try:
            if resp.status_code == 416:
                resp.close()
                if job.size_bytes and offset == job.size_bytes:
                    return
                _discard(job.part)
                offset = 0
                resp = self._range_get(job.url, 0)
            _raise_for_status(resp)
            if resp.status_code != 206:
                offset = 0  # the server ignored Range and sent the whole file
            total = job.size_bytes or _content_total(resp, offset)
            with self._lock:
                job.total, job.received = total, offset
            job.part.parent.mkdir(parents=True, exist_ok=True)
            got = offset
            job.mark_at, job.mark_bytes = time.monotonic(), offset
            with job.part.open("ab" if offset else "wb") as out:
                for block in resp.iter_content(CHUNK):
                    if job.cancel.is_set():
                        raise _Cancelled()
                    if not block:
                        continue
                    out.write(block)
                    got += len(block)
                    if time.monotonic() - job.mark_at >= PROGRESS_EVERY:
                        self._progress(job, got)
            self._progress(job, got)
        finally:
            resp.close()

    def _progress(self, job: _Job, got: int) -> None:
        now = time.monotonic()
        with self._lock:
            if job.finished_at is not None:
                return
            elapsed = now - job.mark_at
            if elapsed > 0:
                instant = (got - job.mark_bytes) / elapsed
                job.speed = instant if job.speed == 0 else 0.3 * instant + 0.7 * job.speed
            job.received = got
            job.eta_s = int((job.total - got) / job.speed) if job.total > got and job.speed >= 1 else None
        job.mark_at, job.mark_bytes = now, got

    def _check_size(self, job: _Job) -> None:
        actual = _file_size(job.part)
        if job.total and actual < job.total:
            raise _Failed("The download stopped early. Start it again to resume.")
        if job.size_bytes and actual != job.size_bytes:
            raise _Failed("size mismatch", discard=True)

    def _verify(self, job: _Job) -> None:
        if not job.sha256:
            return
        with self._lock:
            if job.finished_at is None:
                job.state, job.speed, job.eta_s = "verifying", 0.0, None
        digest = hashlib.sha256()
        with job.part.open("rb") as fh:
            for block in iter(lambda: fh.read(HASH_BLOCK), b""):
                if job.cancel.is_set():
                    raise _Cancelled()
                digest.update(block)
        if digest.hexdigest() != job.sha256:
            raise _Failed("checksum mismatch", discard=True)

    def _install(self, job: _Job) -> None:
        with self._lock:
            if job.cancel.is_set():
                raise _Cancelled()
            size = _file_size(job.part)
            os.replace(job.part, job.path)
            job.total = job.received = size
            _settle_locked(job, "done")
        log.info("Downloaded %s", job.path)


def delete_model(path: str | Path, models_dir: str | Path) -> int:
    """Delete one downloaded model and what only it needed; returns the bytes freed."""
    root = Path(models_dir).expanduser().resolve()
    given = Path(path).expanduser()
    target = given.resolve()
    if target == root or not target.is_relative_to(root) or target.suffix.lower() != ".gguf":
        raise HubError("bad_request", "Only downloaded models inside the models folder can be deleted.")
    if not target.is_file():
        raise HubError("not_found", "That model is not installed.")
    freed = _remove_file(target)
    freed += _remove_file(target.with_name(target.name + ".part"))
    if given.is_symlink():
        given.unlink(missing_ok=True)
    folder = target.parent
    if root in folder.parents:
        leftovers = list(folder.iterdir())
        if leftovers and all(_is_projector_file(item) for item in leftovers):
            freed += sum(_remove_file(item) for item in leftovers)
        _prune_empty(folder, root)
    return freed


def _is_projector_file(path: Path) -> bool:
    name = path.name.lower()
    return name.startswith("mmproj") and name.endswith(".gguf") and path.is_file()


def _remove_file(path: Path) -> int:
    if not path.is_symlink() and not path.is_file():
        return 0
    size = 0 if path.is_symlink() else path.stat().st_size
    path.unlink(missing_ok=True)
    return size


def _prune_empty(folder: Path, root: Path) -> None:
    while root in folder.parents:
        try:
            folder.rmdir()
        except OSError:
            return
        folder = folder.parent


def disk_usage(models_dir: str | Path) -> dict[str, int]:
    """Bytes used by models and by unfinished downloads under ``models_dir``, and what is free."""
    root = Path(models_dir).expanduser()
    models = partial = 0
    if root.is_dir():
        for dirpath, _dirs, names in os.walk(root):
            for name in names:
                full = Path(dirpath, name)
                if full.is_symlink():
                    continue
                lower = name.lower()
                if lower.endswith(".gguf"):
                    models += _file_size(full)
                elif lower.endswith(".part"):
                    partial += _file_size(full)
    return {
        "models_bytes": models,
        "partial_bytes": partial,
        "free_bytes": shutil.disk_usage(_nearest_existing(root)).free,
    }
