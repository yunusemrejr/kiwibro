"""kiwibro.runtime against a fake GitHub on 127.0.0.1: lookup, download checks, unpacking, registry.

No request leaves the machine and no archive member is ever executed.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import socket
import stat
import tarfile
import threading
import time
from collections import Counter, namedtuple
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
import requests

from kiwibro import runtime
from kiwibro.runtime import EngineInstallError, Installer, asset_name

TAG = "b12110"
NEWER = "b12111"
PREFIX = "/repos/ggml-org/llama.cpp/releases"
Member = tuple[tarfile.TarInfo, bytes | None]
Usage = namedtuple("Usage", "total used free")


class FakeGitHub:
    """The release endpoints the installer reads, served from memory on a free local port."""

    def __init__(self) -> None:
        self.stable: dict[str, Any] = {"tag_name": "v0.9.0", "assets": []}
        self.releases: list[dict[str, Any]] = []  # newest first, after the stable release
        self.tags: dict[str, dict[str, Any]] = {}
        self.files: dict[str, bytes] = {}
        self.hits: Counter[str] = Counter()
        self.agents: list[str] = []
        self.delay = 0.0  # seconds to pause after each slice of an archive
        self.slice = 32 * 1024
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                fake._serve(self)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.base = f"http://127.0.0.1:{self._server.server_address[1]}"
        self.api = self.base + PREFIX
        threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def publish(self, tag: str, archives: dict[str, bytes], *,
                digests: dict[str, str | None] | None = None) -> None:
        """Add a build as a release; the stable release's nightly-tag.txt points at it."""
        digests = digests or {}
        assets = []
        for name, data in archives.items():
            self.files[name] = data
            raw: dict[str, Any] = {"name": name, "size": len(data),
                                   "browser_download_url": f"{self.base}/dl/{name}"}
            value = digests[name] if name in digests else "sha256:" + hashlib.sha256(data).hexdigest()
            if value is not None:
                raw["digest"] = value
            assets.append(raw)
        release = {"tag_name": tag, "assets": assets}
        self.tags[tag] = release
        self.releases.insert(0, release)
        body = f"{tag}\n".encode()
        self.files[runtime.NIGHTLY_TAG] = body
        self.stable = {"tag_name": "v0.9.0", "assets": [
            {"name": runtime.NIGHTLY_TAG, "size": len(body),
             "browser_download_url": f"{self.base}/dl/{runtime.NIGHTLY_TAG}"}]}

    def _serve(self, handler: BaseHTTPRequestHandler) -> None:
        path = urlsplit(handler.path).path
        self.hits[path] += 1
        self.agents.append(handler.headers.get("User-Agent", ""))
        if path == PREFIX + "/latest":
            self._json(handler, self.stable)
        elif path == PREFIX:
            self._json(handler, [self.stable, *self.releases])
        elif path.startswith(PREFIX + "/tags/") and path.rsplit("/", 1)[1] in self.tags:
            self._json(handler, self.tags[path.rsplit("/", 1)[1]])
        elif path.startswith("/dl/") and path[4:] in self.files:
            self._send(handler, self.files[path[4:]], slow=self.delay > 0)
        else:
            self._send(handler, b'{"message": "Not Found"}', status=404, ctype="application/json")

    def _json(self, handler: BaseHTTPRequestHandler, body: Any) -> None:
        self._send(handler, json.dumps(body).encode(), ctype="application/json")

    def _send(self, handler: BaseHTTPRequestHandler, data: bytes, *, status: int = 200,
              ctype: str = "application/octet-stream", slow: bool = False) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", ctype)
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        step = self.slice if slow else max(len(data), 1)
        try:
            for start in range(0, len(data), step):
                handler.wfile.write(data[start:start + step])
                handler.wfile.flush()
                if slow:
                    time.sleep(self.delay)
        except OSError:
            pass  # the client hung up, as a cancelled download does


def reg(name: str, data: bytes = b"stub\n") -> Member:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    return info, data


def link(name: str, target: str) -> Member:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    info.mode = 0o777
    return info, None


def special(name: str, kind: bytes) -> Member:
    info = tarfile.TarInfo(name)
    info.type = kind
    return info, None


def archive(path: Path, members: list[Member]) -> bytes:
    """Write a .tar.gz with exactly these members into the test's temp folder and return its bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    return path.read_bytes()


def build_members(tag: str, *, big: bool = False) -> list[Member]:
    folder = f"llama-{tag}"
    members = [
        reg(f"{folder}/llama-server", b"engine stub\n"),
        reg(f"{folder}/llama-cli", b"tool stub\n"),
        reg(f"{folder}/libllama-cli-impl.so", b"helper\n"),
        reg(f"{folder}/libllama-server-impl.so", b"helper\n"),
        reg(f"{folder}/libggml-base.so.0", b"lib\n"),
        link(f"{folder}/libggml-base.so", "libggml-base.so.0"),
        reg(f"{folder}/LICENSE", b"MIT\n"),
    ]
    if big:
        members.append(reg(f"{folder}/libggml-big.so", os.urandom(1 << 20)))
    return members


def release_archives(folder: Path, tag: str = TAG, *, big: bool = False) -> dict[str, bytes]:
    """The four archives a release carries: vulkan and cpu, for x64 and arm64."""
    out = {}
    for arch in ("x64", "arm64"):
        for kind in runtime.KINDS:
            name = asset_name(tag, kind, arch)
            out[name] = archive(folder / name, build_members(tag, big=big))
    return out


def publish_release(github: FakeGitHub, fixtures: Path, tag: str = TAG, *, big: bool = False,
                    digests: dict[str, str | None] | None = None) -> None:
    github.publish(tag, release_archives(fixtures, tag, big=big), digests=digests)


def plain_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False  # never route the loopback fake through a proxy from the environment
    return session


def make_installer(github: FakeGitHub, tmp_path: Path, *, arch: str = "x64") -> Installer:
    return Installer(tmp_path / "data", api_url=github.api, session=plain_session(), arch=arch)


def root_of(tmp_path: Path) -> Path:
    return tmp_path / "data" / "runtime"


def leftovers(tmp_path: Path) -> list[str]:
    root = root_of(tmp_path)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.name.startswith((".download", ".unpack")))


def finish(inst: Installer) -> dict[str, Any]:
    inst.wait(10)
    return inst.status()


def wait_until(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def github() -> Iterator[FakeGitHub]:
    fake = FakeGitHub()
    yield fake
    fake.close()


@pytest.fixture
def fixtures(tmp_path: Path) -> Path:
    folder = tmp_path / "fixtures"
    folder.mkdir()
    return folder


@pytest.mark.parametrize(("system", "raw", "expected"), [
    ("linux", "x86_64", "x64"),
    ("linux", "AMD64", "x64"),
    ("linux", "aarch64", "arm64"),
    ("linux", "arm64", "arm64"),
    ("linux", "riscv64", None),
    ("darwin", "arm64", None),
    ("win32", "AMD64", None),
])
def test_machine(monkeypatch: pytest.MonkeyPatch, system: str, raw: str, expected: str | None) -> None:
    monkeypatch.setattr(runtime.sys, "platform", system)
    monkeypatch.setattr(runtime.platform, "machine", lambda: raw)
    assert runtime.machine() == expected


def test_asset_name_matches_llama_cpp_layout() -> None:
    assert asset_name("b12110", "vulkan", "x64") == "llama-b12110-bin-ubuntu-vulkan-x64.tar.gz"
    assert asset_name("b12110", "cpu", "arm64") == "llama-b12110-bin-ubuntu-arm64.tar.gz"
    with pytest.raises(ValueError):
        asset_name("b12110", "gpu", "x64")


def test_wants_vulkan_needs_a_loader_and_a_render_node(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loader = tmp_path / "libvulkan.so.1"
    dri = tmp_path / "dri"
    dri.mkdir()
    monkeypatch.setattr(runtime, "_VULKAN_LIBS", (str(loader),))
    monkeypatch.setattr(runtime, "_DRI_DIR", str(dri))
    assert runtime.wants_vulkan() is False
    loader.touch()
    assert runtime.wants_vulkan() is False
    (dri / "card0").touch()
    assert runtime.wants_vulkan() is False
    (dri / "renderD128").touch()
    assert runtime.wants_vulkan() is True


def test_wants_vulkan_returns_a_bool_on_this_machine() -> None:
    assert isinstance(runtime.wants_vulkan(), bool)


def test_latest_follows_nightly_tag_and_keeps_only_this_arch(github: FakeGitHub, fixtures: Path,
                                                            tmp_path: Path) -> None:
    publish_release(github, fixtures)
    release = make_installer(github, tmp_path).latest()
    assert release.tag == TAG
    assert {kind: asset.name for kind, asset in release.assets.items()} == {
        "vulkan": asset_name(TAG, "vulkan", "x64"),
        "cpu": asset_name(TAG, "cpu", "x64"),
    }
    cpu = release.assets["cpu"]
    assert cpu.url == f"{github.base}/dl/{cpu.name}"
    assert cpu.size == len(github.files[cpu.name])
    assert cpu.sha256 == hashlib.sha256(github.files[cpu.name]).hexdigest()
    arm = make_installer(github, tmp_path, arch="arm64").latest()
    assert set(arm.assets) == {"vulkan", "cpu"}
    assert all("arm64" in asset.name for asset in arm.assets.values())
    assert github.agents and set(github.agents) == {"kiwibro"}


def test_latest_ignores_assets_without_a_sha256_digest(github: FakeGitHub, fixtures: Path,
                                                      tmp_path: Path) -> None:
    publish_release(github, fixtures, digests={asset_name(TAG, "cpu", "x64"): None})
    assert set(make_installer(github, tmp_path).latest().assets) == {"vulkan"}


def test_latest_falls_back_to_the_newest_release_with_this_arch(github: FakeGitHub, fixtures: Path,
                                                               tmp_path: Path) -> None:
    publish_release(github, fixtures)
    github.stable = {"tag_name": "v0.9.0", "assets": []}
    github.releases.insert(0, {"tag_name": NEWER, "assets": [{
        "name": asset_name(NEWER, "cpu", "arm64"), "size": 3,
        "browser_download_url": f"{github.base}/dl/none", "digest": "sha256:" + "0" * 64}]})
    release = make_installer(github, tmp_path).latest()
    assert release.tag == TAG
    assert github.hits[PREFIX] >= 1


def test_latest_on_an_unsupported_machine(github: FakeGitHub, tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "machine", lambda: None)
    inst = Installer(tmp_path / "data", api_url=github.api, session=plain_session())
    with pytest.raises(EngineInstallError) as exc:
        inst.latest()
    assert exc.value.code == "unsupported"
    with pytest.raises(EngineInstallError) as exc:
        inst.install("cpu")
    assert exc.value.code == "unsupported"
    assert inst.status()["supported"] is False


def test_latest_reports_offline_when_nothing_listens(tmp_path: Path) -> None:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    inst = Installer(tmp_path / "data", api_url=f"http://127.0.0.1:{port}{PREFIX}",
                     session=plain_session(), arch="x64")
    with pytest.raises(EngineInstallError) as exc:
        inst.latest()
    assert exc.value.code == "offline"
    assert exc.value.message == "Could not reach GitHub. Check the internet connection."


def test_api_url_can_come_from_the_environment(github: FakeGitHub, fixtures: Path, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    publish_release(github, fixtures)
    monkeypatch.setenv("KIWIBRO_RUNTIME_API", github.api)
    inst = Installer(tmp_path / "data", session=plain_session(), arch="x64")
    assert inst.latest().tag == TAG


def test_install_cpu_build_is_unpacked_pruned_and_made_current(github: FakeGitHub, fixtures: Path,
                                                              tmp_path: Path) -> None:
    publish_release(github, fixtures)
    inst = make_installer(github, tmp_path)
    started = inst.install("cpu")
    assert started["supported"] is True and started["arch"] == "x64"
    status = finish(inst)
    assert status["state"] == "done", status["error"]
    assert set(status) == {"state", "tag", "kind", "received", "total", "error",
                           "supported", "arch", "installed", "current"}

    build = root_of(tmp_path) / f"{TAG}-cpu"
    assert {p.name for p in build.iterdir()} == {
        "llama-server", "libllama-server-impl.so", "libggml-base.so.0", "libggml-base.so", "LICENSE",
        "kiwibro-runtime.json"}
    server = build / "llama-server"
    assert server.read_bytes() == b"engine stub\n"
    assert stat.S_IMODE(server.stat().st_mode) == 0o755

    manifest = json.loads((build / "kiwibro-runtime.json").read_text(encoding="utf-8"))
    cpu_name = asset_name(TAG, "cpu", "x64")
    assert manifest["tag"] == TAG and manifest["kind"] == "cpu"
    assert manifest["asset"] == cpu_name
    assert manifest["sha256"] == hashlib.sha256(github.files[cpu_name]).hexdigest()

    current = inst.current()
    assert current is not None
    assert (current["name"], current["tag"], current["kind"], current["current"]) == (f"{TAG}-cpu", TAG, "cpu", True)
    assert Path(current["path"]) == server
    assert current["size_bytes"] > 0 and isinstance(current["installed_at"], float)
    assert status["received"] == status["total"] == len(github.files[cpu_name])
    assert leftovers(tmp_path) == []


def test_auto_picks_vulkan_only_when_the_machine_can_run_it(github: FakeGitHub, fixtures: Path,
                                                           tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    publish_release(github, fixtures)
    inst = make_installer(github, tmp_path)
    monkeypatch.setattr(runtime, "wants_vulkan", lambda: True)
    inst.install("auto")
    assert finish(inst)["kind"] == "vulkan"
    assert (root_of(tmp_path) / f"{TAG}-vulkan").is_dir()

    monkeypatch.setattr(runtime, "wants_vulkan", lambda: False)
    inst.install("auto")
    assert finish(inst)["kind"] == "cpu"
    assert (root_of(tmp_path) / f"{TAG}-cpu").is_dir()


def test_explicit_kind_missing_from_the_release_is_an_error(github: FakeGitHub, fixtures: Path,
                                                           tmp_path: Path) -> None:
    publish_release(github, fixtures)
    github.tags[TAG]["assets"] = [a for a in github.tags[TAG]["assets"] if "vulkan" not in a["name"]]
    inst = make_installer(github, tmp_path)
    inst.install("vulkan")
    status = finish(inst)
    assert status["state"] == "error"
    assert "vulkan" in status["error"]
    assert inst.installed() == []


def test_unknown_kind_is_refused_before_anything_starts(github: FakeGitHub, tmp_path: Path) -> None:
    with pytest.raises(EngineInstallError) as exc:
        make_installer(github, tmp_path).install("gpu")
    assert exc.value.code == "failed"


def test_checksum_mismatch_installs_nothing(github: FakeGitHub, fixtures: Path, tmp_path: Path) -> None:
    publish_release(github, fixtures, digests={asset_name(TAG, "cpu", "x64"): "sha256:" + "0" * 64})
    inst = make_installer(github, tmp_path)
    inst.install("cpu")
    status = finish(inst)
    assert status["state"] == "error"
    assert status["error"] == "checksum mismatch"
    assert inst.installed() == []
    assert not (root_of(tmp_path) / f"{TAG}-cpu").exists()
    assert leftovers(tmp_path) == []


def test_download_over_plain_http_is_refused_unless_the_api_is_local(github: FakeGitHub, fixtures: Path,
                                                                    tmp_path: Path) -> None:
    publish_release(github, fixtures)
    cpu = next(a for a in github.tags[TAG]["assets"] if a["name"] == asset_name(TAG, "cpu", "x64"))
    cpu["browser_download_url"] = "http://203.0.113.9/llama.tar.gz"
    inst = make_installer(github, tmp_path)
    inst.install("cpu")
    status = finish(inst)
    assert status["state"] == "error"
    assert "https" in status["error"]
    assert inst.installed() == []


@pytest.mark.parametrize("members", [
    pytest.param(lambda base: [reg("../evil.txt"), reg(f"llama-{TAG}/llama-server")], id="dotdot-member"),
    pytest.param(lambda base: [reg(str(base / "evil.txt")), reg(f"llama-{TAG}/llama-server")],
                 id="absolute-member"),
    pytest.param(lambda base: [link(f"llama-{TAG}/libpasswd.so", "/etc/passwd"),
                               reg(f"llama-{TAG}/llama-server")], id="symlink-to-etc-passwd"),
    pytest.param(lambda base: [link(f"llama-{TAG}/libx.so", "../../x"),
                               reg(f"llama-{TAG}/llama-server")], id="symlink-up-and-out"),
    pytest.param(lambda base: [special(f"llama-{TAG}/dev-null", tarfile.CHRTYPE),
                               reg(f"llama-{TAG}/llama-server")], id="device-node"),
    pytest.param(lambda base: [special(f"llama-{TAG}/pipe", tarfile.FIFOTYPE),
                               reg(f"llama-{TAG}/llama-server")], id="fifo"),
])
def test_malicious_archive_is_rejected_and_writes_nothing_outside_root(
        github: FakeGitHub, fixtures: Path, tmp_path: Path, members: Callable[[Path], list[Member]]) -> None:
    base = tmp_path / "outside"
    base.mkdir()
    bad = archive(fixtures / "bad.tar.gz", members(base))
    github.publish(TAG, {asset_name(TAG, "cpu", "x64"): bad})
    before = set(tmp_path.rglob("*"))

    inst = make_installer(github, tmp_path)
    inst.install("cpu")
    status = finish(inst)

    assert status["state"] == "error"
    assert "invalid" in status["error"]
    assert not (base / "evil.txt").exists()
    created = set(tmp_path.rglob("*")) - before
    assert all(path.is_relative_to(tmp_path / "data") for path in created)
    root = root_of(tmp_path)
    assert not root.exists() or list(root.iterdir()) == []
    assert inst.installed() == []


def test_archive_without_llama_server_is_a_bad_archive(github: FakeGitHub, fixtures: Path,
                                                      tmp_path: Path) -> None:
    without = [reg(f"llama-{TAG}/libggml-base.so.0"), reg(f"llama-{TAG}/LICENSE")]
    github.publish(TAG, {asset_name(TAG, "cpu", "x64"): archive(fixtures / "no-server.tar.gz", without)})
    inst = make_installer(github, tmp_path)
    inst.install("cpu")
    status = finish(inst)
    assert status["state"] == "error"
    assert "llama-server" in status["error"]
    assert inst.installed() == []
    assert leftovers(tmp_path) == []


def test_installed_build_is_not_downloaded_again(github: FakeGitHub, fixtures: Path, tmp_path: Path) -> None:
    publish_release(github, fixtures)
    inst = make_installer(github, tmp_path)
    inst.install("cpu")
    assert finish(inst)["state"] == "done"
    archive_path = f"/dl/{asset_name(TAG, 'cpu', 'x64')}"
    assert github.hits[archive_path] == 1

    inst.install("cpu")
    status = finish(inst)
    assert status["state"] == "done"
    assert github.hits[archive_path] == 1


def test_cancel_during_a_slow_download_leaves_nothing_behind(github: FakeGitHub, fixtures: Path,
                                                            tmp_path: Path) -> None:
    publish_release(github, fixtures, big=True)
    github.delay = 0.05
    inst = make_installer(github, tmp_path)
    inst.install("cpu")
    assert wait_until(lambda: inst.status()["received"] > 0)
    assert inst.cancel() is True
    status = finish(inst)
    assert status["state"] == "cancelled"
    assert inst.installed() == []
    assert leftovers(tmp_path) == []
    assert not (root_of(tmp_path) / f"{TAG}-cpu").exists()
    assert inst.cancel() is False

    github.delay = 0.0
    inst.install("cpu")
    assert finish(inst)["state"] == "done"
    assert inst.current()["name"] == f"{TAG}-cpu"


def test_second_install_while_one_runs_is_busy(github: FakeGitHub, fixtures: Path, tmp_path: Path) -> None:
    publish_release(github, fixtures, big=True)
    github.delay = 0.05
    inst = make_installer(github, tmp_path)
    inst.install("cpu")
    with pytest.raises(EngineInstallError) as exc:
        inst.install("cpu")
    assert exc.value.code == "busy"
    inst.cancel()
    assert finish(inst)["state"] == "cancelled"


def test_no_space_stops_before_download(github: FakeGitHub, fixtures: Path, tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    publish_release(github, fixtures)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(10**9, 10**9, 0))
    inst = make_installer(github, tmp_path)
    inst.install("cpu")
    status = finish(inst)
    assert status["state"] == "error"
    assert "disk space" in status["error"].lower()
    assert github.hits[f"/dl/{asset_name(TAG, 'cpu', 'x64')}"] == 0
    assert inst.installed() == []
    assert leftovers(tmp_path) == []


def test_installed_use_and_remove(github: FakeGitHub, fixtures: Path, tmp_path: Path) -> None:
    publish_release(github, fixtures)
    inst = make_installer(github, tmp_path)
    inst.install("cpu")
    assert finish(inst)["state"] == "done"
    publish_release(github, fixtures, NEWER)
    inst.install("cpu", tag=NEWER)
    assert finish(inst)["state"] == "done"

    root = root_of(tmp_path)
    os.utime(root / f"{TAG}-cpu", (1_000, 1_000))
    os.utime(root / f"{NEWER}-cpu", (2_000, 2_000))
    hidden = root / ".scratch"
    hidden.mkdir()
    (hidden / "llama-server").write_bytes(b"x")
    (hidden / "llama-server").chmod(0o755)
    sentinel = tmp_path / "data" / "x"
    sentinel.mkdir()

    rows = inst.installed()
    assert [row["name"] for row in rows] == [f"{NEWER}-cpu", f"{TAG}-cpu"]
    assert rows[0]["current"] is True and rows[1]["current"] is False
    assert rows[0]["tag"] == NEWER and rows[0]["kind"] == "cpu"

    used = inst.use(f"{TAG}-cpu")
    assert used["current"] is True and inst.current()["name"] == f"{TAG}-cpu"
    assert json.loads((root / "current.json").read_text(encoding="utf-8")) == {"name": f"{TAG}-cpu"}
    with pytest.raises(EngineInstallError) as exc:
        inst.use("missing")
    assert exc.value.code == "not_found"

    for bad in ("../x", ".scratch", "missing"):
        with pytest.raises(EngineInstallError) as exc:
            inst.remove(bad)
        assert exc.value.code == "not_found"
    assert sentinel.is_dir() and hidden.is_dir()

    freed = inst.remove(f"{TAG}-cpu")
    assert freed > 0
    assert inst.current()["name"] == f"{NEWER}-cpu"
    assert json.loads((root / "current.json").read_text(encoding="utf-8")) == {"name": f"{NEWER}-cpu"}

    inst.remove(f"{NEWER}-cpu")
    assert inst.installed() == []
    assert inst.current() is None
    assert not (root / "current.json").exists()
