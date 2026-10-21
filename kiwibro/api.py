"""HTTP API and SSE endpoints, implementing docs/API.md."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, current_app, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from . import config
from .agent import BusyError, ConversationNotFound
from .sandbox import SandboxError, Workspace

log = logging.getLogger(__name__)

bp = Blueprint("api", __name__)


class ApiError(Exception):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def services():
    return current_app.extensions["kiwibro"]


def error_body(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def _json_body() -> dict[str, Any]:
    max_length = current_app.config.get("MAX_CONTENT_LENGTH")
    length = request.content_length
    if max_length and length and length > max_length:
        raise ApiError("too_large", "Request body is too large.", 413)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError("bad_request", "A JSON object body is required.", 400)
    return data


def _conversation_full(conversation: dict[str, Any], *, with_messages: bool = False) -> dict[str, Any]:
    summary = conversation.get("summary") or ""
    view: dict[str, Any] = {
        "id": conversation["id"],
        "title": conversation["title"],
        "model": conversation["model"],
        "pinned": conversation["pinned"],
        "title_auto": conversation["title_auto"],
        "created_at": conversation["created_at"],
        "updated_at": conversation["updated_at"],
        "summary": summary or None,
        "summary_through_seq": conversation["summary_through_seq"],
        "compacted": bool(summary),
    }
    if with_messages:
        view["messages"] = []
    return view


def _conversation_list_item(conversation: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": conversation["id"],
        "title": conversation["title"],
        "model": conversation["model"],
        "pinned": conversation["pinned"],
        "message_count": conversation.get("message_count", 0),
        "created_at": conversation["created_at"],
        "updated_at": conversation["updated_at"],
    }


def _memory_view(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": memory["id"],
        "kind": memory["kind"],
        "content": memory["content"],
        "pinned": memory["pinned"],
        "confidence": memory["confidence"],
        "created_at": memory["created_at"],
        "updated_at": memory["updated_at"],
        "last_used_at": memory["last_used_at"],
    }


def _current_model(models: list[dict[str, Any]] | None) -> str | None:
    svc = services()
    if models:
        return svc.llm.pick_model(models)
    return svc.settings_provider().get("model")


# --- health & bootstrap -----------------------------------------------------


@bp.get("/health")
def health() -> Response:
    svc = services()
    settings = svc.settings_provider()
    ok, models, error = svc.llm.models_status()
    payload: dict[str, Any] = {
        "ok": True,
        "version": config.VERSION,
        "started_at": config.STARTED_AT,
        "lmstudio": ok,
        "lmstudio_url": settings.get("lmstudio_url"),
        "model": _current_model(models) if ok else settings.get("model"),
    }
    if error:
        payload["lmstudio_error"] = error
    return jsonify(payload)


@bp.get("/api/bootstrap")
def bootstrap() -> Response:
    svc = services()
    settings = svc.settings_provider()
    ok, models, error = svc.llm.models_status()
    payload: dict[str, Any] = {
        "health": {
            "ok": True,
            "version": config.VERSION,
            "lmstudio": ok,
            "lmstudio_url": settings.get("lmstudio_url"),
        },
        "settings": settings,
        "models": models if ok else [],
        "current_model": _current_model(models) if ok else settings.get("model"),
        "conversations": [_conversation_list_item(c) for c in svc.db.list_conversations()],
        "memories_count": svc.db.count_memories(),
        "workspace": settings.get("workspace"),
    }
    if error:
        payload["health"]["lmstudio_error"] = error
    return jsonify(payload)


# --- models -----------------------------------------------------------------


@bp.get("/api/models")
def list_models() -> Response:
    svc = services()
    ok, models, error = svc.llm.models_status(force=request.args.get("refresh") == "1")
    payload: dict[str, Any] = {
        "ok": True,
        "lmstudio": ok,
        "current": _current_model(models) if ok else svc.settings_provider().get("model"),
        "models": models if ok else [],
    }
    if error:
        payload["lmstudio_error"] = error
    return jsonify(payload)


@bp.post("/api/models/select")
def select_model() -> Response:
    svc = services()
    data = _json_body()
    model_id = data.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ApiError("bad_request", "id is required", 400)
    model_id = model_id.strip()
    ok, models, _error = svc.llm.models_status(force=True)
    if ok and model_id not in {model["id"] for model in models}:
        raise ApiError("unknown_model", f"Unknown model: {model_id}", 400)
    svc.db.set_setting("model", model_id)
    return jsonify({"ok": True, "current": model_id})


# --- conversations ----------------------------------------------------------


@bp.get("/api/conversations")
def list_conversations() -> Response:
    svc = services()
    return jsonify(
        {"ok": True, "conversations": [_conversation_list_item(c) for c in svc.db.list_conversations()]}
    )


@bp.post("/api/conversations")
def create_conversation() -> Response:
    svc = services()
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        raise ApiError("bad_request", "A JSON object body is required.", 400)
    title = data.get("title")
    if title is not None:
        if not isinstance(title, str) or not title.strip():
            raise ApiError("bad_request", "title must be a non-empty string", 400)
        if len(title) > 200:
            raise ApiError("bad_request", "title is too long", 400)
        title = title.strip()
    else:
        title = "New chat"
    model = data.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ApiError("bad_request", "model must be a non-empty string", 400)
    conversation = svc.db.create_conversation(title=title, model=model or svc.settings_provider().get("model"))
    view = _conversation_full(conversation)
    view["summary"] = None
    view["messages"] = []
    return jsonify(view), 201


@bp.get("/api/conversations/<conversation_id>")
def get_conversation(conversation_id: str) -> Response:
    svc = services()
    conversation = svc.db.get_conversation(conversation_id)
    if conversation is None:
        raise ApiError("not_found", "Conversation not found.", 404)
    messages = svc.db.list_messages(conversation_id)
    model = conversation.get("model") or svc.settings_provider().get("model")
    context = (
        svc.context.estimate_conversation(
            conversation_id, model or "", conversation=conversation, db_messages=messages
        )
        if model
        else {"used": 0, "budget": 30720, "percent": 0}
    )
    return jsonify(
        {
            "ok": True,
            "conversation": _conversation_full(conversation),
            "messages": [
                {
                    "id": message["id"],
                    "seq": message["seq"],
                    "role": message["role"],
                    "content": message["content"],
                    "reasoning": message["reasoning"],
                    "tool_calls": message["tool_calls"],
                    "tool_call_id": message["tool_call_id"],
                    "tool_name": message["tool_name"],
                    "model": message["model"],
                    "created_at": message["created_at"],
                    "meta": message["meta"],
                }
                for message in messages
            ],
            "context": context,
        }
    )


@bp.patch("/api/conversations/<conversation_id>")
def patch_conversation(conversation_id: str) -> Response:
    svc = services()
    conversation = svc.db.get_conversation(conversation_id)
    if conversation is None:
        raise ApiError("not_found", "Conversation not found.", 404)
    data = _json_body()
    fields: dict[str, Any] = {}
    if "title" in data:
        title = data["title"]
        if not isinstance(title, str) or not title.strip():
            raise ApiError("bad_request", "title must be a non-empty string", 400)
        if len(title) > 200:
            raise ApiError("bad_request", "title is too long", 400)
        fields["title"] = title.strip()
        fields["title_auto"] = False
    if "pinned" in data:
        if not isinstance(data["pinned"], bool):
            raise ApiError("bad_request", "pinned must be a boolean", 400)
        fields["pinned"] = data["pinned"]
    if not fields:
        raise ApiError("bad_request", "nothing to update", 400)
    updated = svc.db.update_conversation(conversation_id, **fields)
    return jsonify({"ok": True, "conversation": _conversation_full(updated or conversation)})


@bp.delete("/api/conversations/<conversation_id>")
def delete_conversation(conversation_id: str) -> Response:
    svc = services()
    if svc.db.get_conversation(conversation_id) is None:
        raise ApiError("not_found", "Conversation not found.", 404)
    svc.agent.stop(conversation_id)
    if not svc.db.delete_conversation(conversation_id):
        raise ApiError("not_found", "Conversation not found.", 404)
    return jsonify({"ok": True})


@bp.post("/api/conversations/<conversation_id>/clear")
def clear_conversation(conversation_id: str) -> Response:
    svc = services()
    conversation = svc.db.clear_conversation(conversation_id)
    if conversation is None:
        raise ApiError("not_found", "Conversation not found.", 404)
    return jsonify({"ok": True, "conversation": _conversation_full(conversation)})


# --- chat -------------------------------------------------------------------


@bp.post("/api/chat")
def chat() -> Response:
    svc = services()
    data = _json_body()
    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ApiError("bad_request", "message is required", 400)
    if len(message) > config.MAX_MESSAGE_CHARS:
        raise ApiError("message_too_long", "message is too long", 400)
    conversation_id = data.get("conversation_id")
    if conversation_id is not None and not isinstance(conversation_id, str):
        raise ApiError("bad_request", "conversation_id must be a string", 400)
    model = data.get("model")
    if model is not None:
        if not isinstance(model, str) or not model.strip():
            raise ApiError("bad_request", "model must be a non-empty string", 400)
        model = model.strip()
    try:
        turn = svc.agent.start_chat(conversation_id or None, message, model)
    except ConversationNotFound:
        raise ApiError("not_found", "Conversation not found.", 404) from None
    except BusyError:
        raise ApiError("busy", "A reply is already being generated for this chat.", 409) from None
    response = Response(svc.agent.event_stream(turn), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Connection"] = "keep-alive"
    return response


@bp.post("/api/chat/stop")
def stop_chat() -> Response:
    svc = services()
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        raise ApiError("bad_request", "A JSON object body is required.", 400)
    conversation_id = data.get("conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise ApiError("bad_request", "conversation_id is required", 400)
    svc.agent.stop(conversation_id)
    return jsonify({"ok": True})


@bp.post("/api/approvals/<approval_id>")
def resolve_approval(approval_id: str) -> Response:
    svc = services()
    data = _json_body()
    decision = data.get("decision")
    if decision not in ("allow", "always", "deny"):
        raise ApiError("bad_request", "decision must be allow, always or deny", 400)
    resolved = svc.agent.resolve_approval(approval_id, decision)
    if resolved is None:
        raise ApiError("not_found", "Approval not found or already expired.", 404)
    return jsonify({"ok": True, "decision": resolved})


# --- memories ---------------------------------------------------------------


@bp.get("/api/memories")
def list_memories() -> Response:
    svc = services()
    query = request.args.get("q") or None
    kind = request.args.get("kind") or None
    if kind is not None and kind not in config.MEMORY_KINDS:
        raise ApiError("bad_request", f"unknown memory kind: {kind}", 400)
    memories = svc.db.list_memories(query=query, kind=kind)
    total = svc.db.count_memories(kind=kind, query=query)
    return jsonify({"ok": True, "memories": [_memory_view(m) for m in memories], "total": total})


@bp.post("/api/memories")
def create_memory() -> Response:
    svc = services()
    data = _json_body()
    content = data.get("content")
    kind = data.get("kind") or "fact"
    try:
        memory = svc.memory.remember(content, kind)
    except ValueError as exc:
        raise ApiError("bad_request", str(exc), 400) from None
    return jsonify({"ok": True, "memory": _memory_view(memory)})


@bp.patch("/api/memories/<memory_id>")
def patch_memory(memory_id: str) -> Response:
    svc = services()
    if svc.db.get_memory(memory_id) is None:
        raise ApiError("not_found", "Memory not found.", 404)
    data = _json_body()
    try:
        memory = svc.memory.update(
            memory_id,
            content=data.get("content"),
            kind=data.get("kind"),
            pinned=data.get("pinned"),
        )
    except ValueError as exc:
        raise ApiError("bad_request", str(exc), 400) from None
    if memory is None:
        raise ApiError("not_found", "Memory not found.", 404)
    return jsonify({"ok": True, "memory": _memory_view(memory)})


@bp.delete("/api/memories/<memory_id>")
def delete_memory(memory_id: str) -> Response:
    svc = services()
    if not svc.db.delete_memory(memory_id):
        raise ApiError("not_found", "Memory not found.", 404)
    return jsonify({"ok": True})


@bp.delete("/api/memories")
def delete_all_memories() -> Response:
    svc = services()
    deleted = svc.db.delete_all_memories()
    return jsonify({"ok": True, "deleted": deleted})


# --- settings ---------------------------------------------------------------


@bp.get("/api/settings")
def get_settings() -> Response:
    svc = services()
    return jsonify(
        {
            "ok": True,
            "settings": svc.settings_provider(),
            "data_dir": str(svc.data_dir),
        }
    )


@bp.patch("/api/settings")
def patch_settings() -> Response:
    svc = services()
    data = _json_body()
    current = svc.settings_provider()
    try:
        cleaned = config.validate_settings_patch(data, current)
    except ValueError as exc:
        raise ApiError("bad_request", str(exc), 400) from None
    if not cleaned:
        raise ApiError("bad_request", "nothing to update", 400)
    svc.db.patch_settings(cleaned)
    if "workspace" in cleaned:
        try:
            Path(cleaned["workspace"]).mkdir(parents=True, exist_ok=True)
        except OSError:  # validation already created it
            pass
    return jsonify({"ok": True, "settings": svc.settings_provider()})


# --- workspace --------------------------------------------------------------


@bp.get("/api/workspace")
def workspace_files() -> Response:
    svc = services()
    root = svc.settings_provider().get("workspace") or str(config.default_workspace())
    try:
        files = Workspace(root).list_files(".")
    except SandboxError as exc:
        raise ApiError("workspace_error", str(exc), 400) from None
    return jsonify({"ok": True, "root": str(Path(root).expanduser().resolve()), "files": files})


# --- static shell -----------------------------------------------------------


@bp.get("/")
def index() -> Response:
    try:
        return render_template("index.html")
    except Exception:
        return Response(
            "<!doctype html><title>Kiwibro</title><h1>Kiwibro</h1>"
            "<p>The UI has not been installed yet.</p>",
            mimetype="text/html",
        )


@bp.get("/favicon.ico")
def favicon() -> Response:
    static_folder = Path(current_app.static_folder or "")
    for name in ("favicon.ico", "favicon.svg"):
        candidate = static_folder / "img" / name
        if candidate.is_file():
            return current_app.send_static_file(f"img/{name}")
    return Response(status=404)


# --- errors -----------------------------------------------------------------


@bp.errorhandler(ApiError)
def handle_api_error(exc: ApiError) -> Response:
    return jsonify(error_body(exc.code, exc.message)), exc.status


@bp.errorhandler(HTTPException)
def handle_http_error(exc: HTTPException) -> Response:
    if request.path.startswith("/api/") or request.path == "/health":
        code = {400: "bad_request", 404: "not_found", 405: "method_not_allowed", 413: "too_large"}.get(
            exc.code or 500, "http_error"
        )
        return jsonify(error_body(code, exc.description or "Request failed.")), exc.code or 500
    return exc.get_response()


@bp.errorhandler(Exception)
def handle_unexpected(exc: Exception) -> Response:
    log.exception("unhandled API error on %s", request.path)
    if request.path.startswith("/api/") or request.path == "/health":
        return jsonify(error_body("internal", "Something went wrong.")), 500
    return Response("Internal Server Error", status=500, mimetype="text/plain")


__all__ = ["bp", "services", "error_body", "ApiError"]
