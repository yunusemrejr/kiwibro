"""Model manager routes: library, Hugging Face downloads and the llama.cpp builds.

The hub and installer are fakes, and the engine is a fake too, so nothing touches the
network or runs a downloaded program. Delete tests use real files under the temp data dir.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from kiwibro import api_models
from kiwibro.engine import SOURCE_FOLDER, SOURCE_LMSTUDIO, SOURCE_OWN, SOURCE_SHARED, LocalModel
from kiwibro.hub import Hub, HubError
from kiwibro.runtime import EngineInstallError

GIB = 1024**3
SHA = "a" * 64
FILE = {
    "file": "tiny-Q4_K_M.gguf",
    "name": "tiny-Q4_K_M.gguf",
    "size_bytes": 1000,
    "sha256": SHA,
    "quant": "Q4_K_M",
    "projector": False,
    "split": False,
    "installed": False,
}


class FakeHub:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int]] = []
        self.started: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        self.cleared = 0
        self.jobs_list: list[dict[str, Any]] = []
        self.file_list: dict[str, list[dict[str, Any]]] = {"pub/tiny": [dict(FILE)]}
        self.error: HubError | None = None

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        self.search_calls.append((query, limit))
        if self.error:
            raise self.error
        return [{"repo": "pub/tiny", "author": "pub", "name": "tiny", "downloads": 3}]

    def files(self, repo: str) -> list[dict[str, Any]]:
        if repo == "pub/gone":
            raise HubError("not_found", "That repository was not found on Hugging Face.")
        if self.error:
            raise self.error
        return [dict(item) for item in self.file_list.get(repo, [])]

    def start(self, repo: str, file: str, *, sha256: str = "", size_bytes: int = 0) -> dict[str, Any]:
        job = {"id": "dl_1", "repo": repo, "file": file, "state": "downloading",
               "sha256": sha256, "size_bytes": size_bytes}
        self.started.append(job)
        return job

    def jobs(self) -> list[dict[str, Any]]:
        return [dict(job) for job in self.jobs_list]

    def cancel(self, job_id: str) -> bool:
        self.cancelled.append(job_id)
        return job_id == "dl_1"

    def clear_finished(self) -> int:
        self.cleared += 1
        return 0

    def shutdown(self) -> None:
        pass


@dataclass
class FakeInstaller:
    error: EngineInstallError | None = None
    calls: list[tuple[Any, ...]] = field(default_factory=list)
    freed: int = 1234

    def status(self) -> dict[str, Any]:
        return {"state": "idle", "tag": "", "kind": "", "received": 0, "total": 0, "error": "",
                "supported": True, "arch": "x64", "installed": [], "current": None}

    def install(self, kind: str = "auto", tag: str | None = None) -> dict[str, Any]:
        self.calls.append(("install", kind))
        if self.error:
            raise self.error
        return self.status()

    def cancel(self) -> bool:
        self.calls.append(("cancel",))
        return True

    def use(self, name: str) -> dict[str, Any]:
        self.calls.append(("use", name))
        if self.error:
            raise self.error
        return {"name": name, "current": True}

    def remove(self, name: str) -> int:
        self.calls.append(("remove", name))
        if self.error:
            raise self.error
        return self.freed


class FakeEngine:
    def __init__(self, models: list[LocalModel], servers: list[dict[str, Any]], runtime: Any = None) -> None:
        self.models_list = models
        self.servers = servers
        self.runtime = runtime
        self.sources = [{"path": "/data/models", "source": SOURCE_OWN, "label": "Kiwibro", "models": 1, "bytes": 0}]
        self.force_calls = 0
        self.unload_stops = True
        self.unloaded: list[str] = []

    def models(self, force: bool = False) -> list[LocalModel]:
        if force:
            self.force_calls += 1
        return list(self.models_list)

    def find(self, model_id: str | None) -> LocalModel | None:
        return next((m for m in self.models_list if m.id == (model_id or "").lower()), None)

    def loaded(self, model_id: str) -> dict[str, Any] | None:
        return next((s for s in self.servers if s["model"] == model_id), None)

    def unload(self, model_id: str) -> bool:
        self.unloaded.append(model_id)
        if not self.unload_stops:
            return False
        self.servers = [s for s in self.servers if s["model"] != model_id]
        return True

    def status(self) -> dict[str, Any]:
        return {"runtime": self.runtime, "sources": list(self.sources), "servers": list(self.servers)}


@dataclass
class Manager:
    client: Any
    engine: FakeEngine
    hub: FakeHub
    installer: FakeInstaller
    cleared: list[int]


def _model(path: str, *, source: str = SOURCE_OWN, size: int = 2048, embeddings: bool = False,
           model_id: str = "pub/tiny") -> LocalModel:
    return LocalModel(id=model_id, path=path, size_bytes=size, publisher="pub", repo="pub/tiny",
                      name="tiny", params="1B", quantization="Q4_K_M", context_length=4096,
                      source=source, embeddings=embeddings)


def _wire(env: Any, monkeypatch: pytest.MonkeyPatch, *, local: bool = True, models: list[LocalModel] | None = None,
          servers: list[dict[str, Any]] | None = None, runtime: Any = None) -> Manager:
    svc = env.svc
    engine = FakeEngine(models or [], servers or [], runtime)
    manager = Manager(env.client, engine, FakeHub(), FakeInstaller(), [])
    monkeypatch.setattr(svc.llm, "direct_engine", lambda: engine)
    monkeypatch.setattr(svc.llm, "direct_on", lambda: local)
    monkeypatch.setattr(svc.llm, "clear_cache", lambda: manager.cleared.append(1))
    monkeypatch.setattr(svc, "hub", manager.hub)
    monkeypatch.setattr(svc, "installer", manager.installer)
    monkeypatch.setattr(api_models, "_ram_bytes", lambda: 16 * GIB)
    return manager


def _own_file(env: Any, size: int = 2048) -> Path:
    path = env.svc.data_dir / "models" / "pub" / "tiny" / "tiny-Q4_K_M.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def _error(response: Any) -> tuple[str, str]:
    body = response.get_json()["error"]
    return body["code"], body["message"]


# --- wiring -----------------------------------------------------------------


def test_services_carry_hub_and_installer(env: Any) -> None:
    assert isinstance(env.svc.hub, Hub)
    assert env.svc.hub.models_dir == env.svc.data_dir / "models"
    assert env.svc.installer.root == env.svc.data_dir / "runtime"


def test_hub_is_not_shut_down_at_exit_under_testing(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from kiwibro import create_app

    registered: list[Any] = []
    monkeypatch.setattr(atexit, "register", lambda fn, *args: registered.append(fn))
    create_app({"TESTING": True, "KIWIBRO_DATA_DIR": str(env.svc.data_dir.parent / "other"),
                "KIWIBRO_WORKSPACE": str(env.workspace)})
    assert not any(getattr(fn, "__name__", "") == "shutdown" and isinstance(getattr(fn, "__self__", None), Hub)
                   for fn in registered)


# --- library ----------------------------------------------------------------


def test_ram_bytes_reads_mem_total(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemFree:  100 kB\nMemTotal:       16384000 kB\n", encoding="utf-8")
    assert api_models._ram_bytes(str(meminfo)) == 16384000 * 1024
    assert api_models._ram_bytes(str(tmp_path / "missing")) == 0


def test_library_lists_local_models_with_fit_and_state(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    own = _model("/data/models/pub/tiny/a.gguf", size=5 * GIB, model_id="pub/own")
    tight = _model("/x/b.gguf", size=7 * GIB, source=SOURCE_LMSTUDIO, model_id="pub/tight")
    huge = _model("/x/c.gguf", size=40 * GIB, source=SOURCE_SHARED, model_id="pub/huge")
    embed = _model("/x/d.gguf", embeddings=True, model_id="pub/embed")
    manager = _wire(env, monkeypatch, models=[own, tight, huge, embed],
                    servers=[{"model": "pub/own", "busy": False}],
                    runtime={"path": "/rt/llama-server", "label": "Vulkan", "kind": "vulkan"})
    manager.hub.jobs_list = [{"id": "dl_9", "state": "done"}]

    body = manager.client.get("/api/library").get_json()

    assert body["mode"] == "local"
    assert [m["id"] for m in body["models"]] == ["pub/own", "pub/tight", "pub/huge"]
    first = body["models"][0]
    assert set(first) == {"id", "name", "publisher", "params", "quantization", "size_bytes", "context_length",
                          "kind", "source", "source_label", "deletable", "loaded", "fit"}
    assert first["deletable"] is True and first["loaded"] is True and first["fit"] == "good"
    assert first["source_label"] == "Kiwibro"
    assert body["models"][1]["deletable"] is False and body["models"][1]["loaded"] is False
    assert body["models"][1]["fit"] == "tight"
    assert body["models"][2]["fit"] == "too_big"
    assert body["sources"] == manager.engine.sources
    assert body["runtime"] == manager.engine.runtime
    assert body["ram_bytes"] == 16 * GIB
    assert set(body["disk"]) == {"models_bytes", "partial_bytes", "free_bytes"}
    assert body["downloads"] == [{"id": "dl_9", "state": "done"}]
    assert body["installer"]["supported"] is True


def test_library_is_empty_of_models_when_attached(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _wire(env, monkeypatch, local=False, models=[_model("/x/a.gguf")],
                    runtime={"path": "/rt", "label": "CPU", "kind": "cpu"})

    body = manager.client.get("/api/library").get_json()

    assert body["mode"] == "attached"
    assert body["models"] == []
    assert body["sources"] == []
    assert body["runtime"]["label"] == "CPU"
    assert manager.engine.force_calls == 0


# --- delete -----------------------------------------------------------------


def test_delete_removes_an_own_model_and_refreshes(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _own_file(env)
    manager = _wire(env, monkeypatch, models=[_model(str(path))])

    response = manager.client.delete("/api/library/models", json={"id": "pub/tiny"})

    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "freed_bytes": 2048}
    assert not path.exists()
    assert manager.engine.force_calls == 1
    assert manager.cleared == [1]


def test_delete_stops_a_loaded_model_first(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _own_file(env)
    manager = _wire(env, monkeypatch, models=[_model(str(path))], servers=[{"model": "pub/tiny"}])

    response = manager.client.delete("/api/library/models", json={"id": "pub/tiny"})

    assert response.status_code == 200
    assert manager.engine.unloaded == ["pub/tiny"]
    assert manager.engine.servers == []


def test_delete_is_busy_while_a_model_answers(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _own_file(env)
    manager = _wire(env, monkeypatch, models=[_model(str(path))], servers=[{"model": "pub/tiny"}])
    manager.engine.unload_stops = False

    response = manager.client.delete("/api/library/models", json={"id": "pub/tiny"})

    assert response.status_code == 409
    assert _error(response) == ("busy", "tiny is answering right now. Try again when it is done.")
    assert path.exists()


def test_delete_refuses_lmstudio_and_other_folders(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    lm = _wire(env, monkeypatch, models=[_model("/home/u/.lmstudio/models/a.gguf", source=SOURCE_LMSTUDIO)])
    response = lm.client.delete("/api/library/models", json={"id": "pub/tiny"})
    assert response.status_code == 409
    assert _error(response) == ("not_deletable", "tiny is in LM Studio's folder. Remove it in LM Studio.")

    other = _wire(env, monkeypatch, models=[_model("/srv/models/a.gguf", source=SOURCE_FOLDER)])
    response = other.client.delete("/api/library/models", json={"id": "pub/tiny"})
    assert response.status_code == 409
    assert _error(response) == (
        "not_deletable",
        "tiny is in a folder Kiwibro does not manage. Delete the file there: /srv/models/a.gguf",
    )


def test_delete_never_removes_a_file_outside_kiwibro_folder(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    outside = env.workspace / "stray.gguf"
    outside.write_bytes(b"x" * 10)
    manager = _wire(env, monkeypatch, models=[_model(str(outside))])

    response = manager.client.delete("/api/library/models", json={"id": "pub/tiny"})

    assert response.status_code == 400
    assert _error(response)[0] == "bad_request"
    assert outside.exists()


def test_delete_validates_id_and_knowledge(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _wire(env, monkeypatch, models=[_model("/x/a.gguf")])

    missing = manager.client.delete("/api/library/models", json={})
    assert missing.status_code == 400
    assert _error(missing)[0] == "bad_request"

    unknown = manager.client.delete("/api/library/models", json={"id": "pub/nope"})
    assert unknown.status_code == 404
    assert _error(unknown)[0] == "not_found"


def test_delete_in_attached_mode_is_refused(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _wire(env, monkeypatch, local=False, models=[_model("/x/a.gguf")])

    response = manager.client.delete("/api/library/models", json={"id": "pub/tiny"})

    assert response.status_code == 409
    assert _error(response)[0] == "attached"
    assert manager.engine.force_calls == 0


# --- hugging face ------------------------------------------------------------


def test_search_passes_query_and_limit(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _wire(env, monkeypatch)

    default = manager.client.get("/api/hub/search")
    assert default.status_code == 200
    assert default.get_json()["results"][0]["repo"] == "pub/tiny"
    assert manager.hub.search_calls[-1] == ("", 20)

    manager.client.get("/api/hub/search?q=%20qwen%20&limit=5")
    assert manager.hub.search_calls[-1] == ("qwen", 5)


@pytest.mark.parametrize("raw", ["0", "51", "abc"])
def test_search_rejects_bad_limits(env: Any, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    manager = _wire(env, monkeypatch)

    response = manager.client.get(f"/api/hub/search?limit={raw}")

    assert response.status_code == 400
    assert _error(response)[0] == "bad_request"
    assert manager.hub.search_calls == []


def test_files_add_fit_and_require_repo(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _wire(env, monkeypatch)

    response = manager.client.get("/api/hub/files?repo=pub/tiny")
    assert response.status_code == 200
    body = response.get_json()
    assert body["repo"] == "pub/tiny"
    assert body["files"][0]["file"] == "tiny-Q4_K_M.gguf"
    assert body["files"][0]["fit"] == "good"

    assert manager.client.get("/api/hub/files").status_code == 400
    gone = manager.client.get("/api/hub/files?repo=pub/gone")
    assert gone.status_code == 404
    assert _error(gone)[0] == "not_found"


def test_download_takes_size_and_checksum_from_hugging_face(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _wire(env, monkeypatch)

    response = manager.client.post("/api/hub/downloads", json={
        "repo": "pub/tiny", "file": "tiny-Q4_K_M.gguf", "size_bytes": 1, "sha256": "b" * 64,
    })

    assert response.status_code == 201
    assert response.get_json()["job"]["id"] == "dl_1"
    assert manager.hub.started == [{"id": "dl_1", "repo": "pub/tiny", "file": "tiny-Q4_K_M.gguf",
                                    "state": "downloading", "sha256": SHA, "size_bytes": 1000}]


def test_download_of_unlisted_file_is_not_found(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _wire(env, monkeypatch)

    response = manager.client.post("/api/hub/downloads", json={"repo": "pub/tiny", "file": "other.gguf"})

    assert response.status_code == 404
    assert manager.hub.started == []


def test_download_needs_repo_and_file(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _wire(env, monkeypatch)

    response = env.client.post("/api/hub/downloads", json={"repo": "pub/tiny"})

    assert response.status_code == 400
    assert _error(response)[0] == "bad_request"


def test_finished_download_refreshes_the_library_once(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _wire(env, monkeypatch)
    manager.hub.jobs_list = [{"id": "dl_1", "state": "done"}, {"id": "dl_2", "state": "downloading"}]

    first = manager.client.get("/api/hub/downloads").get_json()
    assert [job["id"] for job in first["jobs"]] == ["dl_1", "dl_2"]
    assert manager.engine.force_calls == 1
    assert manager.cleared == [1]

    manager.client.get("/api/hub/downloads")
    assert manager.engine.force_calls == 1
    assert manager.cleared == [1]


def test_finished_download_in_attached_mode_only_clears_the_cache(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _wire(env, monkeypatch, local=False)
    manager.hub.jobs_list = [{"id": "dl_1", "state": "done"}]

    response = manager.client.get("/api/hub/downloads")

    assert response.status_code == 200
    assert manager.engine.force_calls == 0
    assert manager.cleared == [1]


def test_cancel_and_clear_downloads(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _wire(env, monkeypatch)

    assert manager.client.post("/api/hub/downloads/dl_1/cancel").get_json() == {"ok": True}
    assert manager.client.post("/api/hub/downloads/dl_9/cancel").get_json() == {"ok": False}
    assert manager.client.post("/api/hub/downloads/clear").get_json() == {"ok": True}
    assert manager.hub.cleared == 1


# --- runtime ----------------------------------------------------------------


def test_runtime_status_reports_active_build(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = {"path": "/rt/llama-server", "label": "Vulkan", "kind": "vulkan"}
    manager = _wire(env, monkeypatch, runtime=runtime)

    body = manager.client.get("/api/runtime").get_json()

    assert body["mode"] == "local"
    assert body["active"] == runtime
    assert body["installer"]["supported"] is True


def test_install_defaults_to_auto_and_answers_accepted(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _wire(env, monkeypatch)

    response = manager.client.post("/api/runtime/install")
    assert response.status_code == 202
    assert "installer" in response.get_json()
    assert manager.installer.calls == [("install", "auto")]

    manager.client.post("/api/runtime/install", json={"kind": "cpu"})
    assert manager.installer.calls[-1] == ("install", "cpu")


def test_install_rejects_unknown_kind(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _wire(env, monkeypatch)

    response = manager.client.post("/api/runtime/install", json={"kind": "gpu"})

    assert response.status_code == 400
    assert manager.installer.calls == []


def test_cancel_runtime_install(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _wire(env, monkeypatch)

    body = manager.client.post("/api/runtime/cancel").get_json()

    assert body["ok"] is True and "installer" in body
    assert ("cancel",) in manager.installer.calls


def test_use_switches_the_build(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _wire(env, monkeypatch)

    assert manager.client.post("/api/runtime/use", json={}).status_code == 400

    response = manager.client.post("/api/runtime/use", json={"name": "b1-vulkan"})
    assert response.status_code == 200
    assert response.get_json()["ok"] is True
    assert manager.installer.calls == [("use", "b1-vulkan")]


def test_remove_refuses_while_a_model_runs(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _wire(env, monkeypatch, servers=[{"model": "pub/tiny"}])

    response = manager.client.delete("/api/runtime/b1-vulkan")

    assert response.status_code == 409
    assert _error(response) == ("busy", "Unload the running model first.")
    assert manager.installer.calls == []


def test_remove_deletes_an_idle_build(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _wire(env, monkeypatch)

    response = manager.client.delete("/api/runtime/b1-vulkan")

    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True and body["freed_bytes"] == 1234 and "installer" in body
    assert manager.installer.calls == [("remove", "b1-vulkan")]


# --- error mapping ------------------------------------------------------------


@pytest.mark.parametrize(("code", "status"), [
    ("bad_request", 400), ("not_found", 404), ("offline", 502), ("failed", 502),
    ("no_space", 409), ("busy", 409),
])
def test_hub_errors_map_to_statuses(env: Any, monkeypatch: pytest.MonkeyPatch, code: str, status: int) -> None:
    manager = _wire(env, monkeypatch)
    manager.hub.error = HubError(code, "Something about the hub.")

    response = manager.client.get("/api/hub/search?q=x")

    assert response.status_code == status
    assert _error(response) == (code, "Something about the hub.")


@pytest.mark.parametrize(("code", "status"), [
    ("not_found", 404), ("offline", 502), ("failed", 502),
    ("busy", 409), ("checksum", 409), ("bad_archive", 409), ("unsupported", 409),
])
def test_install_errors_map_to_statuses(env: Any, monkeypatch: pytest.MonkeyPatch, code: str, status: int) -> None:
    manager = _wire(env, monkeypatch)
    manager.installer.error = EngineInstallError(code, "Something about the runtime.")

    response = manager.client.post("/api/runtime/install", json={"kind": "auto"})

    assert response.status_code == status
    assert _error(response) == (code, "Something about the runtime.")
