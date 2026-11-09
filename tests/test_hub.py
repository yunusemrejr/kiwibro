"""Model hub: search, file listing, resumable downloads and disk cleanup, against a local fake Hugging Face."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import threading
import time
from collections import namedtuple
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from kiwibro import hub as hubmod
from kiwibro.hub import Hub, HubError, delete_model, disk_usage, fit, is_projector, quant_of, split_part

MB = 1024 * 1024
GIB = 1024**3


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        fake: FakeHub = self.server.fake
        parts = urlsplit(self.path)
        path = unquote(parts.path)
        query = {key: values[0] for key, values in parse_qs(parts.query).items()}
        headers = {key.lower(): value for key, value in self.headers.items()}
        fake.calls.append({"path": path, "query": query, "headers": headers})
        if path == "/api/models":
            return self._send_json(fake.search_items)
        if path.startswith("/api/models/") and path.endswith("/tree/main"):
            repo = path[len("/api/models/"):-len("/tree/main")]
            if repo in fake.trees:
                return self._send_json(fake.trees[repo])
            return self._send(404, b'{"error":"Repository Not Found"}', "application/json")
        repo, marker, file = path.lstrip("/").partition("/resolve/main/")
        data = fake.blobs.get(f"{repo}/{file}") if marker else None
        if data is None:
            return self._send(404, b"Entry not found", "text/plain")
        return self._send_file(fake, data)

    def _send_file(self, fake: FakeHub, data: bytes) -> None:
        start, status = 0, 200
        match = re.fullmatch(r"bytes=(\d+)-", self.headers.get("Range") or "") if fake.honor_range else None
        if match:
            start, status = int(match.group(1)), 206
            if start >= len(data):
                return self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{len(data)}"})
        body = data[start:]
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        self.end_headers()
        try:
            if fake.throttle:
                for at in range(0, len(body), 8192):
                    self.wfile.write(body[at:at + 8192])
                    self.wfile.flush()
                    time.sleep(fake.throttle)
            else:
                self.wfile.write(body)
        except OSError:
            pass  # the client cancelled and hung up

    def _send(self, status: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any) -> None:
        self._send(200, json.dumps(payload).encode(), "application/json")


class FakeHub:
    """A Hugging Face stand-in on 127.0.0.1 that records every request it receives."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.trees: dict[str, list[dict[str, Any]]] = {}
        self.search_items: list[dict[str, Any]] = []
        self.honor_range = True
        self.throttle = 0.0
        self.calls: list[dict[str, Any]] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.fake = self
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()

    def add(self, repo: str, file: str, data: bytes) -> None:
        self.blobs[f"{repo}/{file}"] = data

    def gets(self, repo: str, file: str) -> list[dict[str, Any]]:
        return [call for call in self.calls if call["path"] == f"/{repo}/resolve/main/{file}"]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(5)


@pytest.fixture
def hf():
    fake = FakeHub()
    yield fake
    fake.close()


@pytest.fixture
def hub(hf, tmp_path):
    instance = Hub(tmp_path / "models", base_url=hf.url)
    yield instance
    instance.shutdown()


def wait_until(check: Callable[[], Any], what: str, timeout: float = 10.0) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        value = check()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.02)


def job_now(hub: Hub, job_id: str) -> dict[str, Any]:
    return next(job for job in hub.jobs() if job["id"] == job_id)


def wait_state(hub: Hub, job_id: str, *states: str) -> dict[str, Any]:
    return wait_until(lambda: job_now(hub, job_id) if job_now(hub, job_id)["state"] in states else None,
                      f"job {job_id} to reach {states}")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def test_quant_of_reads_the_label_from_the_name():
    cases = {
        "Model.Q4_K_M.gguf": "Q4_K_M",
        "model-IQ2_M.gguf": "IQ2_M",
        "model-Q8_0.gguf": "Q8_0",
        "model-f16.gguf": "F16",
        "model-bf16.gguf": "BF16",
        "mmproj-model-f16.gguf": "F16",
        "sub/Qwen2.5-Coder-7B-Q5_K_S.gguf": "Q5_K_S",
        "big-00001-of-00003.gguf": "",
        "plain.gguf": "",
    }
    for name, expected in cases.items():
        assert quant_of(name) == expected, name


def test_projector_and_split_detection():
    assert is_projector("mmproj-model-f16.gguf")
    assert is_projector("MMPROJ-Model.gguf")
    assert not is_projector("model-Q4_K_M.gguf")
    assert split_part("big-00001-of-00003.gguf") == (1, 3)
    assert split_part("big-00002-of-00003.gguf") == (2, 3)
    assert split_part("plain.gguf") is None
    assert split_part("bad-00004-of-00003.gguf") is None


def test_fit_thresholds():
    assert fit(4 * GIB, 32 * GIB) == "good"       # 4 GiB * 1.25 + 1 GiB = 6 GiB <= 19.2 GiB
    assert fit(20 * GIB, 32 * GIB) == "tight"     # 26 GiB: above 60% of RAM, below 85%
    assert fit(30 * GIB, 32 * GIB) == "too_big"
    assert fit(1 * GIB, 0) == "too_big"


def test_search_maps_fields_skips_private_and_gated(hf, hub):
    hf.search_items = [
        {"id": "Owner/Good-GGUF", "downloads": 1200, "likes": 5, "createdAt": "2025-01-02T00:00:00.000Z",
         "tags": ["gguf", "vision", "license:apache-2.0"], "pipeline_tag": "image-text-to-text"},
        {"id": "owner/plain", "downloads": 3, "likes": 0, "lastModified": "2025-02-01",
         "tags": ["gguf"], "pipeline_tag": "text-generation"},
        {"id": "owner/secret", "private": True, "downloads": 9, "tags": []},
        {"id": "owner/gated", "gated": "auto", "downloads": 9, "tags": []},
    ]
    found = hub.search("qwen", limit=5)
    assert [item["repo"] for item in found] == ["Owner/Good-GGUF", "owner/plain"]
    assert found[0] == {
        "repo": "Owner/Good-GGUF", "author": "Owner", "name": "Good-GGUF", "downloads": 1200, "likes": 5,
        "updated": "2025-01-02T00:00:00.000Z", "vision": True, "license": "apache-2.0",
    }
    assert found[1]["updated"] == "2025-02-01"
    assert found[1]["vision"] is False
    assert found[1]["license"] == ""
    call = hf.calls[-1]
    assert call["path"] == "/api/models"
    assert call["headers"]["user-agent"] == "kiwibro"
    assert "authorization" not in call["headers"]
    assert call["query"] == {"search": "qwen", "filter": "gguf", "sort": "downloads", "direction": "-1", "limit": "5"}


def test_empty_search_omits_the_query(hf, hub):
    hf.search_items = [{"id": "a/b", "tags": ["gguf"]}]
    assert [item["repo"] for item in hub.search("")] == ["a/b"]
    assert "search" not in hf.calls[-1]["query"]


def test_files_keeps_gguf_uses_lfs_and_flags_entries(hf, hub):
    repo = "acme/tiny-7b-GGUF"
    hf.trees[repo] = [
        {"type": "directory", "path": "IQ2_XS"},
        {"type": "file", "path": "README.md", "size": 12},
        {"type": "file", "path": "tiny-7b-Q8_0.gguf", "size": 130,
         "lfs": {"oid": "ab" * 32, "size": 9000}},
        {"type": "file", "path": "IQ2_XS/tiny-7b-IQ2_XS.gguf", "size": 80},
        {"type": "file", "path": "mmproj-tiny-7b-f16.gguf", "size": 1500,
         "lfs": {"oid": "cd" * 32, "size": 1500}},
        {"type": "file", "path": "big-00001-of-00002.gguf", "size": 4000,
         "lfs": {"oid": "ef" * 32, "size": 4000}},
    ]
    target_dir = hub.models_dir / "acme" / "tiny-7b-GGUF"
    write(target_dir / "tiny-7b-Q8_0.gguf", 9000)
    write(target_dir / "tiny-7b-IQ2_XS.gguf", 5)

    files = hub.files(repo)

    assert [item["name"] for item in files] == [
        "tiny-7b-IQ2_XS.gguf", "big-00001-of-00002.gguf", "tiny-7b-Q8_0.gguf", "mmproj-tiny-7b-f16.gguf",
    ]
    assert files[0]["installed"] is False
    assert files[2] == {
        "file": "tiny-7b-Q8_0.gguf", "name": "tiny-7b-Q8_0.gguf", "size_bytes": 9000, "sha256": "ab" * 32,
        "quant": "Q8_0", "projector": False, "split": False, "installed": True,
    }
    assert files[1]["split"] is True and files[1]["quant"] == ""
    assert files[3]["projector"] is True and files[3]["quant"] == "F16" and files[3]["installed"] is False


def test_files_errors(hub):
    with pytest.raises(HubError) as bad:
        hub.files("not-a-repo")
    assert bad.value.code == "bad_request"
    with pytest.raises(HubError) as missing:
        hub.files("acme/missing")
    assert missing.value.code == "not_found"


@pytest.mark.parametrize("repo", ["nosep", "a/b/c", "../x", "a/..", "a/.", "-bad/x", "a b/c", "a/x\n"])
def test_target_path_refuses_bad_repos(hub, repo):
    with pytest.raises(HubError) as info:
        hub.target_path(repo, "x.gguf")
    assert info.value.code == "bad_request"


@pytest.mark.parametrize("file", ["../x.gguf", "a/../../x.gguf", "/abs/x.gguf", "x.bin", "a\\..\\x.gguf",
                                  "sub/../x.txt", ""])
def test_target_path_refuses_traversal_and_non_gguf(hub, file):
    with pytest.raises(HubError) as info:
        hub.target_path("owner/name", file)
    assert info.value.code == "bad_request"


def test_target_path_refuses_symlink_escape_and_keeps_good_paths(hub, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    hub.models_dir.mkdir(parents=True)
    (hub.models_dir / "evil").symlink_to(outside, target_is_directory=True)
    with pytest.raises(HubError) as info:
        hub.target_path("evil/name", "x.gguf")
    assert info.value.code == "bad_request"
    assert hub.target_path("owner/name", "sub/Model-Q4.gguf") == hub.models_dir / "owner" / "name" / "Model-Q4.gguf"


def test_download_writes_verified_file(hf, hub):
    repo, file = "owner/tiny", "Tiny-Q4_K_M.gguf"
    data = os.urandom(700_000)
    hf.add(repo, file, data)
    job = hub.start(repo, file, sha256=sha(data), size_bytes=len(data))
    assert job["state"] in ("queued", "downloading")
    done = wait_state(hub, job["id"], "done")
    target = hub.models_dir / "owner" / "tiny" / file
    assert target.read_bytes() == data
    assert not target.with_name(file + ".part").exists()
    assert done["received"] == done["total"] == len(data)
    assert done["path"] == str(target)
    assert hf.gets(repo, file)[0]["headers"]["user-agent"] == "kiwibro"
    assert "authorization" not in hf.gets(repo, file)[0]["headers"]


def test_checksum_mismatch_leaves_nothing(hf, hub):
    repo, file = "owner/tiny", "bad.gguf"
    data = os.urandom(50_000)
    hf.add(repo, file, data)
    job = hub.start(repo, file, sha256="0" * 64)
    failed = wait_state(hub, job["id"], "error", "done")
    assert failed["state"] == "error"
    assert "checksum mismatch" in failed["error"]
    target = hub.models_dir / "owner" / "tiny" / file
    assert not target.exists()
    assert not target.with_name(file + ".part").exists()


def test_resume_continues_from_partial_file(hf, hub):
    repo, file = "owner/tiny", "resume.gguf"
    data = os.urandom(300_000)
    hf.add(repo, file, data)
    target = hub.models_dir / "owner" / "tiny" / file
    target.parent.mkdir(parents=True)
    target.with_name(file + ".part").write_bytes(data[:100_000])
    job = hub.start(repo, file, sha256=sha(data))
    done = wait_state(hub, job["id"], "done")
    assert [call["headers"].get("range") for call in hf.gets(repo, file)] == ["bytes=100000-"]
    assert target.read_bytes() == data
    assert done["total"] == done["received"] == len(data)


def test_server_ignoring_range_still_ends_identical(hf, hub):
    repo, file = "owner/tiny", "norange.gguf"
    data = os.urandom(120_000)
    hf.add(repo, file, data)
    hf.honor_range = False
    target = hub.models_dir / "owner" / "tiny" / file
    target.parent.mkdir(parents=True)
    target.with_name(file + ".part").write_bytes(b"z" * 50_000)
    job = hub.start(repo, file, sha256=sha(data))
    wait_state(hub, job["id"], "done")
    assert hf.gets(repo, file)[0]["headers"]["range"] == "bytes=50000-"
    assert target.read_bytes() == data


def test_complete_partial_file_is_accepted_without_a_body(hf, hub):
    repo, file = "owner/tiny", "whole.gguf"
    data = os.urandom(40_000)
    hf.add(repo, file, data)
    target = hub.models_dir / "owner" / "tiny" / file
    target.parent.mkdir(parents=True)
    target.with_name(file + ".part").write_bytes(data)
    job = hub.start(repo, file, sha256=sha(data), size_bytes=len(data))
    wait_state(hub, job["id"], "done")
    assert target.read_bytes() == data


def test_cancel_keeps_partial_and_start_resumes(hf, hub):
    repo, file = "owner/big", "big.gguf"
    data = os.urandom(2 * MB)
    hf.add(repo, file, data)
    hf.throttle = 0.005
    job = hub.start(repo, file, sha256=sha(data), size_bytes=len(data))
    wait_until(lambda: job_now(hub, job["id"])["received"] > 0, "first bytes")
    assert hub.cancel(job["id"]) is True
    wait_state(hub, job["id"], "cancelled")
    part = hub.models_dir / "owner" / "big" / (file + ".part")
    assert 0 < part.stat().st_size < len(data)

    hf.throttle = 0.0
    again = hub.start(repo, file, sha256=sha(data), size_bytes=len(data))
    assert again["id"] != job["id"]
    wait_state(hub, again["id"], "done")
    assert (hub.models_dir / "owner" / "big" / file).read_bytes() == data
    assert int(hf.gets(repo, file)[-1]["headers"]["range"].split("=")[1].split("-")[0]) > 0


def test_second_download_waits_in_queue(hf, hub):
    repo = "owner/queue"
    first_data, second_data = os.urandom(2 * MB), os.urandom(10_000)
    hf.add(repo, "a.gguf", first_data)
    hf.add(repo, "b.gguf", second_data)
    hf.throttle = 0.005
    first = hub.start(repo, "a.gguf", size_bytes=len(first_data))
    second = hub.start(repo, "b.gguf", size_bytes=len(second_data))
    assert second["state"] == "queued"
    assert hub.start(repo, "b.gguf")["id"] == second["id"]
    assert hub.active()
    wait_state(hub, first["id"], "done")
    wait_state(hub, second["id"], "done")
    assert (hub.models_dir / "owner" / "queue" / "b.gguf").read_bytes() == second_data
    assert hub.clear_finished() == 2
    assert hub.jobs() == []


def test_start_refuses_when_disk_is_too_small(hub, monkeypatch):
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(10 * GIB, 10 * GIB - 1000, 1000))
    with pytest.raises(HubError) as info:
        hub.start("owner/tiny", "huge.gguf", size_bytes=5 * GIB)
    assert info.value.code == "no_space"
    assert hub.jobs() == []


def test_missing_file_reports_it_is_gone(hf, hub):
    hf.add("owner/tiny", "other.gguf", b"data")
    job = hub.start("owner/tiny", "gone.gguf")
    failed = wait_state(hub, job["id"], "error")
    assert "no longer on Hugging Face" in failed["error"]


def test_delete_model_frees_bytes_prunes_folders_and_orphan_projector(tmp_path):
    models = tmp_path / "models"
    write(models / "acme" / "tiny" / "Tiny-Q4.gguf", 100)
    write(models / "acme" / "tiny" / "Tiny-Q4.gguf.part", 30)
    write(models / "acme" / "tiny" / "mmproj-tiny.gguf", 50)

    freed = delete_model(models / "acme" / "tiny" / "Tiny-Q4.gguf", models)

    assert freed == 180
    assert not (models / "acme").exists()
    assert models.is_dir()


def test_delete_model_keeps_folder_that_still_holds_models(tmp_path):
    models = tmp_path / "models"
    write(models / "two" / "one" / "A-Q4.gguf", 10)
    write(models / "two" / "one" / "B-Q8.gguf", 20)
    write(models / "two" / "one" / "mmproj-x.gguf", 5)

    assert delete_model(models / "two" / "one" / "A-Q4.gguf", models) == 10
    assert (models / "two" / "one" / "B-Q8.gguf").exists()
    assert (models / "two" / "one" / "mmproj-x.gguf").exists()


def test_delete_model_refuses_outside_symlink_and_missing(tmp_path):
    models = tmp_path / "models"
    write(models / "two" / "one" / "A-Q4.gguf", 10)
    outside = write(tmp_path / "outside.gguf", 7)
    with pytest.raises(HubError) as escaped:
        delete_model(outside, models)
    assert escaped.value.code == "bad_request"
    assert outside.exists()

    link = models / "two" / "one" / "link.gguf"
    link.symlink_to(outside)
    with pytest.raises(HubError) as linked:
        delete_model(link, models)
    assert linked.value.code == "bad_request"
    assert outside.exists()

    with pytest.raises(HubError) as missing:
        delete_model(models / "two" / "one" / "nope.gguf", models)
    assert missing.value.code == "not_found"


def test_disk_usage_totals(tmp_path):
    models = tmp_path / "models"
    write(models / "a" / "b" / "x.gguf", 100)
    write(models / "a" / "y.gguf", 50)
    write(models / "a" / "z.gguf.part", 7)
    write(models / "notes.txt", 1000)

    usage = disk_usage(models)

    assert usage["models_bytes"] == 150
    assert usage["partial_bytes"] == 7
    assert usage["free_bytes"] > 0
    assert disk_usage(tmp_path / "none")["models_bytes"] == 0


def test_unreachable_hub_is_offline(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    offline = Hub(tmp_path / "models", base_url=f"http://127.0.0.1:{port}")
    with pytest.raises(HubError) as info:
        offline.search("x")
    assert info.value.code == "offline"


def test_base_url_comes_from_environment(hf, tmp_path, monkeypatch):
    monkeypatch.setenv("KIWIBRO_HF_URL", hf.url + "/")
    env_hub = Hub(tmp_path / "models")
    try:
        hf.search_items = [{"id": "a/b", "tags": []}]
        assert [item["repo"] for item in env_hub.search("")] == ["a/b"]
    finally:
        env_hub.shutdown()


def test_constructing_a_hub_starts_no_thread(tmp_path):
    idle = Hub(tmp_path / "models")
    assert not any(thread.name == "kiwibro-hub" for thread in threading.enumerate())
    idle.shutdown()
    assert hubmod.HF_URL == "https://huggingface.co"
