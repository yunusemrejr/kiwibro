"""Kiwibro's own copy of the llama.cpp engine, from the official GitHub releases.

A build is downloaded for this machine, checked against the sha256 GitHub publishes for it,
and unpacked only after every archive member has been vetted. The program inside is never
run here; the inference layer starts it later.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import sys
import tarfile
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

log = logging.getLogger(__name__)

RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
KINDS = ("vulkan", "cpu")

NIGHTLY_TAG = "nightly-tag.txt"
CURRENT = "current.json"
MANIFEST = "kiwibro-runtime.json"
CHUNK = 256 * 1024
SPACE_FACTOR = 4  # the archive, its unpacked copy and some headroom
MAX_UNPACKED = 4 * 1024**3  # an archive that expands without limit could fill the disk
_CHECKSUM = "checksum mismatch"
_OFFLINE = "Could not reach GitHub. Check the internet connection."
_UNSUPPORTED = "There is no llama.cpp build for this machine (Linux on x86-64 or arm64 only)."
_LOCAL_HOSTS = ("127.0.0.1", "localhost")
_VULKAN_LIBS = (
    "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
    "/usr/lib/aarch64-linux-gnu/libvulkan.so.1",
    "/usr/lib64/libvulkan.so.1",
    "/usr/lib/libvulkan.so.1",
)
_DRI_DIR = "/dev/dri"
_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_HELPER_LIB = re.compile(r"^libllama-(?!server-)[a-z0-9-]+-impl\.so")


class EngineInstallError(Exception):
    """A runtime step that failed; ``code`` says what kind of failure it was."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _Cancelled(Exception):
    """Unwinds the install thread once cancel() was called."""


def machine() -> str | None:
    """The arch name Kiwibro uses for this machine, or None when no build exists for it."""
    if sys.platform != "linux":
        return None
    raw = platform.machine().lower()
    if raw in ("x86_64", "amd64"):
        return "x64"
    if raw in ("aarch64", "arm64"):
        return "arm64"
    return None


def asset_name(tag: str, kind: str, arch: str) -> str:
    """The release asset that holds ``kind`` for ``arch``, named the way llama.cpp names it."""
    if kind == "vulkan":
        return f"llama-{tag}-bin-ubuntu-vulkan-{arch}.tar.gz"
    if kind == "cpu":
        return f"llama-{tag}-bin-ubuntu-{arch}.tar.gz"
    raise ValueError(f"unknown runtime kind: {kind}")


def wants_vulkan() -> bool:
    """Whether the Vulkan build can run here: a Vulkan loader and a GPU render node. Filesystem only."""
    if not any(os.path.exists(lib) for lib in _VULKAN_LIBS):
        return False
    try:
        return any(name.startswith("renderD") for name in os.listdir(_DRI_DIR))
    except OSError:
        return False


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int
    sha256: str
    kind: str


@dataclass(frozen=True)
class Release:
    tag: str
    assets: dict[str, Asset]


class Installer:
    """Installs, lists and removes llama.cpp builds under ``data_dir/runtime``.

    One install runs on a background thread at a time; status() says where it is.
    """

    def __init__(self, data_dir: str | Path, *, api_url: str | None = None,
                 session: requests.Session | None = None, arch: str | None = None) -> None:
        self.root = Path(data_dir) / "runtime"
        self._api = (api_url or os.environ.get("KIWIBRO_RUNTIME_API") or RELEASES_API).rstrip("/")
        self._local = _is_local(self._api)
        self._session = session or requests.Session()
        self._arch = arch if arch is not None else machine()
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = _idle()

    def latest(self) -> Release:
        """The newest build for this machine; falls back to the newest release that has one."""
        if self._arch is None:
            raise EngineInstallError("unsupported", _UNSUPPORTED)
        try:
            tag = self._nightly_tag()
            if tag is not None:
                release = self._release(tag)
                if release.assets:
                    return release
        except EngineInstallError as exc:
            if exc.code not in ("failed", "not_found"):
                raise
            log.info("Latest runtime tag unusable (%s); checking recent releases", exc.message)
        return self._newest_release()

    def install(self, kind: str = "auto", tag: str | None = None) -> dict[str, Any]:
        """Start installing a build in the background and return the first status.

        ``kind`` is "vulkan", "cpu" or "auto"; ``tag`` picks a release instead of the newest one.
        """
        if kind not in ("auto", *KINDS):
            raise EngineInstallError("failed", f"Unknown runtime kind: {kind}.")
        if self._arch is None:
            raise EngineInstallError("unsupported", _UNSUPPORTED)
        if tag is not None and not _TAG.fullmatch(tag):
            raise EngineInstallError("not_found", f"{tag!r} is not a llama.cpp release tag.")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise EngineInstallError("busy", "A runtime install is already running.")
            self._cancel.clear()
            self._state = _idle()
            self._state.update(state="checking", tag=tag or "")
            self._thread = threading.Thread(target=self._run, args=(kind, tag),
                                            name="kiwibro-runtime", daemon=True)
            self._thread.start()
        return self.status()

    def cancel(self) -> bool:
        """Stop the running install at its next chunk. False when nothing is running."""
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            if running:
                self._cancel.set()
        return running

    def wait(self, timeout: float | None = None) -> None:
        """Block until the running install ends (tests and scripts)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def status(self) -> dict[str, Any]:
        with self._lock:
            out = dict(self._state)
        rows = self.installed()
        out.update(supported=self._arch is not None, arch=self._arch or "", installed=rows,
                   current=next((row for row in rows if row["current"]), None))
        return out

    def installed(self) -> list[dict[str, Any]]:
        """Every installed build, newest first. Hidden folders are never builds."""
        if not self.root.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in self.root.iterdir():
            if path.name.startswith(".") or path.is_symlink() or not path.is_dir() or not _built(path):
                continue
            tag, sep, kind = path.name.rpartition("-")
            if not sep:
                tag, kind = path.name, ""
            rows.append({"name": path.name, "tag": tag, "kind": kind,
                         "path": str(path / "llama-server"), "size_bytes": _tree_size(path),
                         "installed_at": path.stat().st_mtime, "current": False})
        rows.sort(key=lambda row: row["installed_at"], reverse=True)
        chosen = self._current_name(rows)
        for row in rows:
            row["current"] = row["name"] == chosen
        return rows

    def current(self) -> dict[str, Any] | None:
        return next((row for row in self.installed() if row["current"]), None)

    def use(self, name: str) -> dict[str, Any]:
        """Make an installed build the current one."""
        row = next((r for r in self.installed() if r["name"] == name), None)
        if row is None:
            raise EngineInstallError("not_found", f"No installed runtime named {name!r}.")
        self._write_current(name)
        row["current"] = True
        return row

    def remove(self, name: str) -> int:
        """Delete an installed build and return the bytes freed. The newest one left becomes current."""
        rows = self.installed()
        row = next((r for r in rows if r["name"] == name), None)
        if row is None:
            raise EngineInstallError("not_found", f"No installed runtime named {name!r}.")
        shutil.rmtree(self.root / name)
        if row["current"]:
            rest = [r for r in rows if r["name"] != name]
            if rest:
                self._write_current(rest[0]["name"])
            else:
                (self.root / CURRENT).unlink(missing_ok=True)
        return int(row["size_bytes"])

    def _run(self, kind: str, tag: str | None) -> None:
        part: Path | None = None
        unpack: Path | None = None
        try:
            release = self.latest() if tag is None else self._release(tag)
            chosen = self._choose(release, kind)
            asset = release.assets[chosen]
            self._update(tag=release.tag, kind=chosen)
            final = self.root / f"{release.tag}-{chosen}"
            if _built(final):
                self._write_current(final.name)
                self._update(state="done")
                return
            self.root.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(self.root).free
            if free < SPACE_FACTOR * asset.size:
                raise EngineInstallError(
                    "no_space",
                    f"Not enough free disk space: about {_mb(SPACE_FACTOR * asset.size)} MB is needed.")
            part = self.root / f".download-{release.tag}-{chosen}.part"
            unpack = self.root / f".unpack-{release.tag}-{chosen}"
            log.info("Installing llama.cpp %s (%s)", release.tag, chosen)
            self._download(asset, part)
            self._update(state="unpacking")
            build = _unpack(part, unpack)
            self._raise_if_cancelled()
            _prune(build)
            os.chmod(build / "llama-server", 0o755)
            manifest = {"tag": release.tag, "kind": chosen, "asset": asset.name,
                        "sha256": asset.sha256, "installed_at": time.time()}
            (build / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            _remove(final)
            os.replace(build, final)
            self._write_current(final.name)
            self._update(state="done")
            log.info("llama.cpp %s (%s) installed", release.tag, chosen)
        except _Cancelled:
            self._update(state="cancelled", error="")
            log.info("llama.cpp install cancelled")
        except EngineInstallError as exc:
            self._fail(exc.message)
        except OSError as exc:
            log.exception("llama.cpp install hit a file error")
            self._fail(f"Could not write the runtime files: {exc}")
        except Exception:
            log.exception("llama.cpp install failed")
            self._fail("The install failed unexpectedly. See the log for details.")
        finally:
            _remove(part)
            _remove(unpack)

    def _choose(self, release: Release, kind: str) -> str:
        if kind == "auto":
            if "vulkan" in release.assets and wants_vulkan():
                return "vulkan"
            if "cpu" in release.assets:
                return "cpu"
        elif kind in release.assets:
            return kind
        raise EngineInstallError("unsupported", f"Release {release.tag} has no build for this machine ({kind}).")

    def _download(self, asset: Asset, part: Path) -> None:
        self._update(state="downloading", received=0, total=asset.size)
        digest = hashlib.sha256()
        received = 0
        try:
            with self._get(asset.url, stream=True, timeout=(10, 60)) as resp, part.open("wb") as out:
                for chunk in resp.iter_content(CHUNK):
                    self._raise_if_cancelled()
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > asset.size:
                        raise EngineInstallError("checksum", _CHECKSUM)
                    out.write(chunk)
                    digest.update(chunk)
                    self._update(received=received)
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise EngineInstallError("offline", _OFFLINE) from exc
        except requests.RequestException as exc:
            raise EngineInstallError("failed", f"Download failed: {exc}") from exc
        self._update(state="verifying")
        if received != asset.size or digest.hexdigest() != asset.sha256:
            raise EngineInstallError("checksum", _CHECKSUM)

    def _nightly_tag(self) -> str | None:
        """The build tag the stable release points at, read from its nightly-tag.txt asset."""
        for asset in _assets_of(self._json(f"{self._api}/latest")):
            url = asset.get("browser_download_url")
            if asset.get("name") == NIGHTLY_TAG and isinstance(url, str):
                with self._get(url) as resp:
                    body = resp.content
                text = body.decode("utf-8", "replace").strip() if len(body) <= 256 else ""
                return text if _TAG.fullmatch(text) else None
        return None

    def _release(self, tag: str) -> Release:
        if not _TAG.fullmatch(tag):
            raise EngineInstallError("not_found", f"{tag!r} is not a llama.cpp release tag.")
        return self._release_of(tag, _assets_of(self._json(f"{self._api}/tags/{tag}")))

    def _newest_release(self) -> Release:
        listing = self._json(f"{self._api}?per_page=30")
        if not isinstance(listing, list):
            raise EngineInstallError("failed", "GitHub sent an unexpected release list.")
        for release in listing:
            tag = release.get("tag_name") if isinstance(release, dict) else None
            if isinstance(tag, str) and _TAG.fullmatch(tag):
                found = self._release_of(tag, _assets_of(release))
                if found.assets:
                    return found
        raise EngineInstallError("failed", "No llama.cpp build for this machine was found on GitHub.")

    def _release_of(self, tag: str, raw_assets: list[dict[str, Any]]) -> Release:
        """This machine's assets of a release. Anything without a sha256 digest is left out."""
        assets: dict[str, Asset] = {}
        for raw in raw_assets:
            for kind in KINDS:
                if raw.get("name") == asset_name(tag, kind, str(self._arch)):
                    asset = _asset(raw, kind)
                    if asset is not None:
                        assets[kind] = asset
        return Release(tag, assets)

    def _json(self, url: str) -> Any:
        with self._get(url) as resp:
            try:
                return resp.json()
            except ValueError as exc:
                raise EngineInstallError("failed", "GitHub sent a reply that is not JSON.") from exc

    def _get(self, url: str, *, stream: bool = False, timeout: tuple[float, float] = (10, 30)) -> requests.Response:
        if not self._allowed(url):
            raise EngineInstallError("failed", "Refusing a download that is not https.")
        try:
            resp = self._session.get(url, headers={"User-Agent": "kiwibro"}, stream=stream, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise EngineInstallError("offline", _OFFLINE) from exc
        except requests.RequestException as exc:
            raise EngineInstallError("failed", f"Request failed: {exc}") from exc
        if resp.status_code == 404:
            resp.close()
            raise EngineInstallError("not_found", "GitHub has nothing at that address.")
        if resp.status_code >= 400:
            resp.close()
            raise EngineInstallError("failed", f"GitHub answered HTTP {resp.status_code}.")
        if not self._allowed(resp.url):
            resp.close()
            raise EngineInstallError("failed", "Refusing a redirect that is not https.")
        return resp

    def _allowed(self, url: str) -> bool:
        """https only; plain http is allowed just for a local test API, and only to a local host."""
        parts = urlsplit(url)
        if parts.scheme == "https":
            return bool(parts.hostname)
        return self._local and parts.scheme == "http" and parts.hostname in _LOCAL_HOSTS

    def _current_name(self, rows: list[dict[str, Any]]) -> str | None:
        try:
            saved = json.loads((self.root / CURRENT).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = None
        name = saved.get("name") if isinstance(saved, dict) else None
        if any(row["name"] == name for row in rows):
            return name
        return rows[0]["name"] if rows else None

    def _write_current(self, name: str) -> None:
        tmp = self.root / f".{CURRENT}.tmp"
        tmp.write_text(json.dumps({"name": name}), encoding="utf-8")
        os.replace(tmp, self.root / CURRENT)

    def _update(self, **fields: Any) -> None:
        with self._lock:
            self._state.update(fields)

    def _fail(self, message: str) -> None:
        self._update(state="error", error=message)
        log.warning("llama.cpp install failed: %s", message)

    def _raise_if_cancelled(self) -> None:
        if self._cancel.is_set():
            raise _Cancelled()


def _idle() -> dict[str, Any]:
    return {"state": "idle", "tag": "", "kind": "", "received": 0, "total": 0, "error": ""}


def _is_local(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme == "http" and parts.hostname in _LOCAL_HOSTS


def _assets_of(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict) or not isinstance(data.get("assets"), list):
        return []
    return [asset for asset in data["assets"] if isinstance(asset, dict)]


def _asset(raw: dict[str, Any], kind: str) -> Asset | None:
    """The asset as GitHub describes it, or None when it carries no usable sha256 digest."""
    digest, url, size = raw.get("digest"), raw.get("browser_download_url"), raw.get("size")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        return None
    sha = digest[len("sha256:"):].lower()
    if not _SHA256.fullmatch(sha) or not isinstance(url, str) or not isinstance(size, int) or size <= 0:
        return None
    return Asset(str(raw["name"]), url, size, sha, kind)


def _built(folder: Path) -> bool:
    server = folder / "llama-server"
    return server.is_file() and os.access(server, os.X_OK)


def _regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _tree_size(folder: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(dirpath, name)
            if not os.path.islink(path):
                try:
                    total += os.path.getsize(path)
                except OSError:
                    continue
    return total


def _mb(n: int) -> str:
    return f"{n / 2**20:.0f}"


def _remove(path: Path | None) -> None:
    if path is None:
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _invalid(why: str) -> EngineInstallError:
    return EngineInstallError("bad_archive", f"The runtime archive is invalid: {why}.")


def _unsafe(name: str) -> bool:
    """Absolute paths and any ".." segment could land outside the unpack folder."""
    if not name or name.startswith(("/", "\\")) or re.match(r"[A-Za-z]:", name):
        return True
    return ".." in re.split(r"[\\/]", name)


def _vet(members: list[tarfile.TarInfo]) -> None:
    """Refuse the whole archive before anything is written when a member is not safe to unpack."""
    total = 0
    for member in members:
        if _unsafe(member.name):
            raise _invalid(f"unsafe name {member.name!r}")
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
            raise _invalid(f"unsupported entry {member.name!r}")
        if (member.issym() or member.islnk()) and _unsafe(member.linkname):
            raise _invalid(f"unsafe link {member.name!r}")
        if member.isfile():
            total += member.size
    if total > MAX_UNPACKED:
        raise _invalid("it unpacks to more than 4 GiB")


def _build_folder(dest: Path) -> Path:
    """The folder that holds llama-server: the unpack root itself, or the one folder inside it."""
    if _regular(dest / "llama-server"):
        return dest
    homes = [p for p in dest.iterdir() if p.is_dir() and not p.is_symlink() and _regular(p / "llama-server")]
    if len(homes) != 1:
        raise _invalid("it does not contain exactly one llama-server")
    return homes[0]


def _unpack(archive: Path, dest: Path) -> Path:
    """Extract ``archive`` into a fresh ``dest`` folder and return the build folder inside it."""
    _remove(dest)
    dest.mkdir(parents=True)
    try:
        with tarfile.open(archive, "r:gz") as tar:
            _vet(tar.getmembers())
            tar.extractall(dest, filter="data")
    except (tarfile.TarError, EOFError, zlib.error) as exc:
        raise _invalid(str(exc)) from exc
    return _build_folder(dest)


def _keep(entry: Path) -> bool:
    if entry.is_dir() and not entry.is_symlink():
        return False
    name = entry.name
    if name == "llama-server":
        return _regular(entry)
    if name.startswith("LICENSE"):
        return True
    return fnmatch.fnmatchcase(name, "lib*.so*") and _HELPER_LIB.match(name) is None


def _prune(build: Path) -> None:
    """Keep the engine, its shared libraries and the license; drop the other tools and helper libraries."""
    for entry in list(build.iterdir()):
        if not _keep(entry):
            _remove(entry)
    for entry in list(build.iterdir()):
        if entry.is_symlink() and not entry.exists():
            entry.unlink()
