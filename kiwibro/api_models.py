"""HTTP routes for the model manager: installed models, Hugging Face downloads and llama.cpp builds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Response, current_app, jsonify, request

from . import hub as hubmod
from .api import ApiError, _json_body, bp, error_body, services
from .engine import SOURCE_LABELS, SOURCE_LMSTUDIO
from .hub import HubError
from .runtime import EngineInstallError

INSTALL_KINDS = ("auto", "vulkan", "cpu")
# Bad input or an unknown thing gets its own status; network trouble is a bad gateway; the rest conflict.
FAILURE_STATUS = {"bad_request": 400, "not_found": 404, "offline": 502, "failed": 502}
SEEN_DONE_KEY = "kiwibro_seen_downloads"


def _failure(code: str, message: str) -> tuple[Response, int]:
    return jsonify(error_body(code, message)), FAILURE_STATUS.get(code, 409)


@bp.errorhandler(HubError)
def handle_hub_error(exc: HubError) -> tuple[Response, int]:
    return _failure(exc.code, exc.message)


@bp.errorhandler(EngineInstallError)
def handle_install_error(exc: EngineInstallError) -> tuple[Response, int]:
    return _failure(exc.code, exc.message)


def _ram_bytes(source: str = "/proc/meminfo") -> int:
    """Total physical RAM in bytes, 0 when the system does not say."""
    try:
        with open(source, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _own_dir(svc: Any) -> Path:
    return svc.data_dir / "models"


def _mode(svc: Any) -> str:
    return "local" if svc.llm.direct_on() else "attached"


def _optional_body() -> dict[str, Any]:
    data = request.get_json(silent=True)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ApiError("bad_request", "A JSON object body is required.", 400)
    return data


def _text(data: dict[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ApiError("bad_request", f"{label} is required", 400)
    return value.strip()


def _model_view(model: Any, running: set[str], ram: int) -> dict[str, Any]:
    return {
        "id": model.id,
        "name": model.name,
        "publisher": model.publisher,
        "params": model.params,
        "quantization": model.quantization,
        "size_bytes": model.size_bytes,
        "context_length": model.context_length,
        "kind": model.kind,
        "source": model.source,
        "source_label": SOURCE_LABELS.get(model.source, model.source),
        "deletable": model.deletable,
        "loaded": model.id in running,
        "fit": hubmod.fit(model.size_bytes, ram),
    }


def _not_deletable(model: Any) -> str:
    name = model.name or model.id
    if model.source == SOURCE_LMSTUDIO:
        return f"{name} is in LM Studio's folder. Remove it in LM Studio."
    return f"{name} is in a folder Kiwibro does not manage. Delete the file there: {model.path}"


def _limit(raw: str | None, default: int = 20) -> int:
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ApiError("bad_request", "limit must be an integer", 400) from None
    if not 1 <= value <= 50:
        raise ApiError("bad_request", "limit must be between 1 and 50", 400)
    return value


# --- library ----------------------------------------------------------------


@bp.get("/api/library")
def library() -> Response:
    """Installed models with where they live, plus disk, RAM, runtime and downloads."""
    svc = services()
    engine = svc.llm.direct_engine()
    status = engine.status()
    ram = _ram_bytes()
    local = svc.llm.direct_on()
    models: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    if local:
        running = {server["model"] for server in status["servers"]}
        models = [_model_view(m, running, ram) for m in engine.models() if not m.embeddings]
        sources = status["sources"]
    return jsonify({
        "mode": "local" if local else "attached",
        "models": models,
        "sources": sources,
        "disk": hubmod.disk_usage(_own_dir(svc)),
        "ram_bytes": ram,
        "runtime": status["runtime"],
        "installer": svc.installer.status(),
        "downloads": svc.hub.jobs(),
    })


@bp.delete("/api/library/models")
def delete_library_model() -> Response:
    """Delete a model Kiwibro downloaded itself; stops its server first when it is loaded."""
    svc = services()
    model_id = _text(_json_body(), "id", "id")
    if not svc.llm.direct_on():
        raise ApiError("attached", "Kiwibro is attached to a model server it does not run, "
                                   "so it cannot delete models here.", 409)
    engine = svc.llm.direct_engine()
    model = engine.find(model_id)
    if model is None:
        raise ApiError("not_found", "Kiwibro does not know that model.", 404)
    if not model.deletable:
        raise ApiError("not_deletable", _not_deletable(model), 409)
    name = model.name or model.id
    if engine.loaded(model.id) is not None:
        stopped = engine.unload(model.id)
        if not stopped and engine.loaded(model.id) is not None:
            raise ApiError("busy", f"{name} is answering right now. Try again when it is done.", 409)
    freed = hubmod.delete_model(model.path, _own_dir(svc))
    engine.models(force=True)
    svc.llm.clear_cache()
    return jsonify({"ok": True, "freed_bytes": freed})


# --- hugging face -----------------------------------------------------------


@bp.get("/api/hub/search")
def hub_search() -> Response:
    """Models on Hugging Face matching the text; an empty text lists the popular ones."""
    svc = services()
    query = (request.args.get("q") or "").strip()[:200]
    limit = _limit(request.args.get("limit"))
    return jsonify({"results": svc.hub.search(query, limit)})


@bp.get("/api/hub/files")
def hub_files() -> Response:
    """The GGUF files of one repository, each with how well it fits this computer's RAM."""
    svc = services()
    repo = (request.args.get("repo") or "").strip()
    if not repo:
        raise ApiError("bad_request", "repo is required", 400)
    ram = _ram_bytes()
    files = [{**item, "fit": hubmod.fit(item["size_bytes"], ram)} for item in svc.hub.files(repo)]
    return jsonify({"repo": repo, "files": files})


@bp.post("/api/hub/downloads")
def start_download() -> Response:
    """Queue a file from a repository. Size and checksum come from Hugging Face, never from the client."""
    svc = services()
    data = _json_body()
    repo = _text(data, "repo", "repo")
    file = _text(data, "file", "file")
    item = next((entry for entry in svc.hub.files(repo) if entry["file"] == file), None)
    if item is None:
        raise ApiError("not_found", "That file is not in this repository.", 404)
    job = svc.hub.start(repo, file, sha256=item["sha256"], size_bytes=item["size_bytes"])
    return jsonify({"job": job}), 201


@bp.get("/api/hub/downloads")
def list_downloads() -> Response:
    """Download jobs and disk use. A download that finished shows up in the library right away."""
    svc = services()
    jobs = svc.hub.jobs()
    seen: set[str] = current_app.extensions.setdefault(SEEN_DONE_KEY, set())
    fresh = {job["id"] for job in jobs if job["state"] == "done"} - seen
    if fresh:
        seen.update(fresh)
        if svc.llm.direct_on():
            svc.llm.direct_engine().models(force=True)
        svc.llm.clear_cache()
    return jsonify({"jobs": jobs, "disk": hubmod.disk_usage(_own_dir(svc))})


@bp.post("/api/hub/downloads/<job_id>/cancel")
def cancel_download(job_id: str) -> Response:
    return jsonify({"ok": services().hub.cancel(job_id)})


@bp.post("/api/hub/downloads/clear")
def clear_downloads() -> Response:
    services().hub.clear_finished()
    return jsonify({"ok": True})


# --- llama.cpp runtime ------------------------------------------------------


@bp.get("/api/runtime")
def runtime_status() -> Response:
    """The llama.cpp build in use and the installer's state."""
    svc = services()
    return jsonify({
        "mode": _mode(svc),
        "active": svc.llm.direct_engine().status()["runtime"],
        "installer": svc.installer.status(),
    })


@bp.post("/api/runtime/install")
def install_runtime() -> Response:
    """Start installing a llama.cpp build in the background."""
    svc = services()
    kind = _optional_body().get("kind", "auto")
    if kind not in INSTALL_KINDS:
        raise ApiError("bad_request", "kind must be auto, vulkan or cpu", 400)
    svc.installer.install(kind)
    return jsonify({"installer": svc.installer.status()}), 202


@bp.post("/api/runtime/cancel")
def cancel_runtime_install() -> Response:
    svc = services()
    svc.installer.cancel()
    return jsonify({"ok": True, "installer": svc.installer.status()})


@bp.post("/api/runtime/use")
def use_runtime() -> Response:
    """Make an installed llama.cpp build the one Kiwibro starts models with."""
    svc = services()
    name = _text(_json_body(), "name", "name")
    svc.installer.use(name)
    return jsonify({"ok": True, "installer": svc.installer.status()})


@bp.delete("/api/runtime/<name>")
def remove_runtime(name: str) -> Response:
    """Delete an installed llama.cpp build, unless a model is running on it."""
    svc = services()
    if svc.llm.direct_engine().status()["servers"]:
        raise ApiError("busy", "Unload the running model first.", 409)
    freed = svc.installer.remove(name)
    return jsonify({"ok": True, "freed_bytes": freed, "installer": svc.installer.status()})
