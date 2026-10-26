"""Kiwibro application factory and service wiring."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable

from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import HTTPException

from . import config
from .agent import AgentManager, retitle_small_talk_chats
from .api import bp, error_body
from .context import ContextEngine
from .db import Database
from .llm import LMStudioClient
from .memory import MemoryEngine
from .readiness import Readiness
from .search import SearchEngine

log = logging.getLogger(__name__)

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass
class Services:
    db: Database
    llm: LMStudioClient
    context: ContextEngine
    memory: MemoryEngine
    search: SearchEngine
    agent: AgentManager
    settings_provider: Callable[[], dict[str, Any]]
    data_dir: Path


def _configure_logging(data_dir: Path) -> None:
    root = logging.getLogger()
    if getattr(root, "_kiwibro_configured", False):
        return
    root.setLevel(logging.INFO)
    file_handler = RotatingFileHandler(
        data_dir / "kiwibro.log", maxBytes=2 * 1024 * 1024, backupCount=2, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    root._kiwibro_configured = True  # type: ignore[attr-defined]


def _host_header() -> str:
    host = request.host or ""
    if host.startswith("["):
        end = host.find("]")
        return host[1:end].lower() if end != -1 else ""
    if ":" in host:
        return host.split(":", 1)[0].lower()
    return host.lower()


def create_app(config_overrides: dict[str, Any] | None = None) -> Flask:
    repo_root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        static_folder=str(repo_root / "static"),
        template_folder=str(repo_root / "templates"),
    )
    app.config.update(config_overrides or {})
    app.json.compact = True
    app.config.setdefault("TEMPLATES_AUTO_RELOAD", True)
    if not app.config.get("MAX_CONTENT_LENGTH"):
        app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

    data_dir = Path(app.config.get("KIWIBRO_DATA_DIR") or config.data_dir()).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    app.config["KIWIBRO_DATA_DIR"] = str(data_dir)
    if not app.config.get("TESTING"):
        _configure_logging(data_dir)

    db = Database(config.database_path(data_dir))
    db.initialize()
    try:
        retitle_small_talk_chats(db)
    except Exception:  # titles are cosmetic; never block startup
        logging.getLogger(__name__).warning("could not re-title chats", exc_info=True)

    defaults = config.default_settings(
        lmstudio_url=app.config.get("KIWIBRO_LMSTUDIO_URL"),
        workspace=app.config.get("KIWIBRO_WORKSPACE"),
    )
    workspace_override = app.config.get("KIWIBRO_WORKSPACE")
    if workspace_override:
        defaults["workspace"] = str(Path(workspace_override).expanduser().resolve())
    try:
        Path(defaults["workspace"]).expanduser().mkdir(parents=True, exist_ok=True)
    except OSError:
        log.warning("could not create default workspace %s", defaults["workspace"])

    _settings_cache: dict[str, Any] = {"version": -1, "value": None}

    def settings_provider() -> dict[str, Any]:
        # Settings are read many times per turn (agent, context, memory, llm,
        # tools) but written rarely; serve from cache keyed by the DB write
        # version so reads never go stale.
        version = db.settings_version
        cached = _settings_cache["value"]
        if cached is None or _settings_cache["version"] != version:
            cached = db.get_settings(defaults)
            _settings_cache["value"] = cached
            _settings_cache["version"] = db.settings_version
        return {**cached, "permissions": dict(cached.get("permissions") or {})}

    llm = LMStudioClient(settings_provider, data_dir=data_dir)
    context = ContextEngine(db, llm, settings_provider)
    memory = MemoryEngine(db, llm, settings_provider)
    search = SearchEngine()
    agent = AgentManager(db, llm, context, memory, search, settings_provider)
    # Load + KV warm-up of the selected model ahead of the first message. Off in tests:
    # scripted fakes would answer the warm-up request instead of the turn.
    warmup = app.config.get("KIWIBRO_WARMUP")
    if warmup is None:
        warmup = not app.config.get("TESTING") and os.environ.get("KIWIBRO_WARMUP", "1") != "0"
    agent.readiness = Readiness(
        llm, agent.warm_prefix, busy_fn=agent.active_turns, on_job=agent._cancel_memory_extraction,
        enabled=bool(warmup),
    )
    services = Services(
        db=db,
        llm=llm,
        context=context,
        memory=memory,
        search=search,
        agent=agent,
        settings_provider=settings_provider,
        data_dir=data_dir,
    )
    app.extensions["kiwibro"] = services

    app.register_blueprint(bp)

    @app.before_request
    def enforce_loopback() -> Response | None:
        if _host_header() not in _LOCAL_HOSTS:
            return jsonify(error_body("forbidden_host", "Only loopback requests are allowed.")), 403
        return None

    @app.after_request
    def api_headers(response: Response) -> Response:
        if request.path.startswith("/api/") or request.path == "/health":
            if "Cache-Control" not in response.headers:
                response.headers["Cache-Control"] = "no-store"
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers.setdefault("Referrer-Policy", "no-referrer")
        elif request.path == "/" or request.path.startswith("/static/"):
            # Revalidate every load (cheap 304s on loopback): without this, the browser's
            # heuristic cache can keep serving old JS/CSS after an update.
            response.headers["Cache-Control"] = "no-cache"
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        return response

    @app.errorhandler(HTTPException)
    def handle_http_exception(exc: HTTPException) -> Response:
        if request.path.startswith("/api/") or request.path == "/health":
            code = {
                400: "bad_request",
                404: "not_found",
                405: "method_not_allowed",
                413: "too_large",
            }.get(exc.code or 500, "http_error")
            return jsonify(error_body(code, exc.description or "Request failed.")), exc.code or 500
        return exc.get_response()

    @app.errorhandler(Exception)
    def handle_unexpected(exc: Exception) -> Response:
        log.exception("unhandled error on %s", request.path)
        if request.path.startswith("/api/") or request.path == "/health":
            return jsonify(error_body("internal", "Something went wrong.")), 500
        return Response("Internal Server Error", status=500, mimetype="text/plain")

    return app


__all__ = ["create_app", "Services", "config"]
