"""HTTP API and SSE endpoints, implementing docs/API.md."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, current_app, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from . import config, media, routing
from .agent import BusyError, ConversationNotFound
from .llm import LMStudioError
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
        "models": [
            {**m, "context_choices": routing.context_choices(m)} if m.get("state") == "loaded" else m
            for m in models
        ] if ok else [],
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


@bp.post("/api/models/context")
def resize_model_context() -> Response:
    """Reload the selected loaded model with a larger context window (long chats)."""
    svc = services()
    data = _json_body()
    model_id, size = data.get("id"), data.get("context_length")
    if not isinstance(model_id, str) or isinstance(size, bool) or not isinstance(size, int):
        raise ApiError("bad_request", "id and context_length are required", 400)
    ok, models, error = svc.llm.models_status(force=True)
    model = next((m for m in models if m["id"] == model_id), None) if ok else None
    if model is None or model.get("state") != "loaded":
        raise ApiError("unknown_model", error or f"{model_id} is not loaded in LM Studio.", 400)
    if size not in routing.context_choices(model):
        raise ApiError("bad_request", "That context length is not available for this model.", 400)
    if svc.agent.active_turns():
        raise ApiError("busy", "Wait for the current reply to finish.", 409)
    try:
        current = routing.reload_with_context(svc.llm, model, size)
    except LMStudioError as exc:
        raise ApiError(exc.kind, exc.message, 409 if exc.kind == "no_memory" else 502) from exc
    svc.db.set_setting("model", current)
    return jsonify({"ok": True, "current": current, "context_length": size})


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


def _page_params() -> tuple[int | None, int | None]:
    """Opt-in keyset pagination for long conversations (?limit=&before_seq=)."""
    raw_limit = request.args.get("limit")
    raw_before = request.args.get("before_seq")
    limit: int | None = None
    before: int | None = None
    if raw_limit is not None:
        try:
            limit = int(raw_limit)
        except ValueError:
            raise ApiError("bad_request", "limit must be an integer", 400) from None
        if not 1 <= limit <= 500:
            raise ApiError("bad_request", "limit must be between 1 and 500", 400)
    if raw_before is not None:
        try:
            before = int(raw_before)
        except ValueError:
            raise ApiError("bad_request", "before_seq must be an integer", 400) from None
        if before < 1:
            raise ApiError("bad_request", "before_seq must be positive", 400)
    return limit, before


@bp.get("/api/conversations/<conversation_id>")
def get_conversation(conversation_id: str) -> Response:
    svc = services()
    conversation = svc.db.get_conversation(conversation_id)
    if conversation is None:
        raise ApiError("not_found", "Conversation not found.", 404)
    limit, before_seq = _page_params()
    page: dict[str, Any] = {}
    if limit is None:
        messages = svc.db.list_messages(conversation_id)
        estimate_messages = messages
    else:
        messages, has_more = svc.db.list_messages_page(
            conversation_id, before_seq=before_seq, limit=limit
        )
        page = {
            "has_more": has_more,
            "oldest_seq": messages[0]["seq"] if messages else None,
        }
        estimate_messages = None  # estimate from the summary window, not the page
    model = conversation.get("model") or svc.settings_provider().get("model")
    context = (
        svc.context.estimate_conversation(
            conversation_id, model or "", conversation=conversation, db_messages=estimate_messages
        )
        if model
        else {"used": 0, "budget": 30720, "percent": 0}
    )
    return jsonify(
        {
            "ok": True,
            **({"page": page} if page else {}),
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


@bp.post("/api/conversations/<conversation_id>/fork")
def fork_conversation(conversation_id: str) -> Response:
    svc = services()
    conv = svc.db.get_conversation(conversation_id)
    if conv is None:
        raise ApiError("not_found", "Conversation not found.", 404)
    data = request.get_json(silent=True) or {}
    max_seq = data.get("seq")
    new_title = f"{conv['title']} (fork)"
    new_conv = svc.db.create_conversation(
        title=new_title,
        model=conv.get("model") or svc.settings_provider().get("model"),
    )
    messages = svc.db.list_messages(conversation_id)
    for m in messages:
        if max_seq is not None:
            try:
                if int(m.get("seq", 0)) > int(max_seq):
                    continue
            except (ValueError, TypeError):
                pass
        svc.db.add_message(
            conversation_id=new_conv["id"],
            role=m["role"],
            content=m.get("content", ""),
            reasoning=m.get("reasoning", ""),
            tool_calls=m.get("tool_calls"),
            tool_call_id=m.get("tool_call_id"),
            tool_name=m.get("tool_name"),
            model=m.get("model"),
            tokens=m.get("tokens"),
            meta=m.get("meta"),
        )
    return jsonify({"ok": True, "conversation": _conversation_full(new_conv)}), 201


@bp.post("/api/conversations/<conversation_id>/compact")
def compact_conversation(conversation_id: str) -> Response:
    svc = services()
    conv = svc.db.get_conversation(conversation_id)
    if conv is None:
        raise ApiError("not_found", "Conversation not found.", 404)
    model = conv.get("model") or svc.settings_provider().get("model") or ""
    ok = False
    if hasattr(svc.context, "recover_aggressive"):
        try:
            ok = svc.context.recover_aggressive(conversation_id, model)
        except Exception:
            ok = False
    if not ok and hasattr(svc.context, "hard_recover"):
        try:
            ok = svc.context.hard_recover(conversation_id)
        except Exception:
            ok = False
    updated = svc.db.get_conversation(conversation_id) or conv
    return jsonify({"ok": True, "conversation": _conversation_full(updated), "compacted": bool(ok)})



# --- chat -------------------------------------------------------------------


@bp.post("/api/chat")
def chat() -> Response:
    svc = services()
    data = _json_body()
    message = data.get("message")
    attachments = _chat_attachments(svc, data.get("attachments"))
    if not isinstance(message, str) or not (message.strip() or attachments):
        raise ApiError("bad_request", "message is required", 400)
    if not message.strip():  # attachments alone: name them, like a caption
        message = ", ".join(item["name"] for item in attachments)
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
        turn = svc.agent.start_chat(conversation_id or None, message, model, attachments=attachments)
    except ConversationNotFound:
        raise ApiError("not_found", "Conversation not found.", 404) from None
    except BusyError:
        raise ApiError("busy", "A reply is already being generated for this chat.", 409) from None
    response = Response(svc.agent.event_stream(turn), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Connection"] = "keep-alive"
    return response


def _workspace(svc: Any) -> Workspace:
    return Workspace(svc.settings_provider().get("workspace") or str(config.default_workspace()))


def _chat_attachments(svc: Any, raw: Any) -> list[dict[str, Any]]:
    """Attachments named in a chat request, re-derived from the files (never trusted as sent)."""
    if raw in (None, []):
        return []
    if not isinstance(raw, list) or len(raw) > media.MAX_ATTACHMENTS:
        raise ApiError("bad_request", f"attachments must be a list of at most {media.MAX_ATTACHMENTS}", 400)
    workspace = _workspace(svc)
    result = []
    for item in raw:
        path = item.get("path") if isinstance(item, dict) else None
        if not isinstance(path, str) or not path.startswith(f"{media.ATTACHMENTS_DIR}/"):
            raise ApiError("bad_request", "each attachment needs the path returned by /api/attachments", 400)
        try:
            resolved = workspace.resolve(path, must_exist=True)
        except SandboxError as exc:
            raise ApiError("bad_attachment", str(exc), 400) from None
        name = item.get("name") if isinstance(item.get("name"), str) else resolved.name
        result.append({
            "path": workspace.public(resolved),
            "name": name.strip()[:120] or resolved.name,
            "kind": media.media_kind(resolved),
            "size": resolved.stat().st_size,
        })
    return result


@bp.post("/api/attachments")
def upload_attachment() -> Response:
    """Store one file the user attached, inside the workspace so every tool can reach it."""
    svc = services()
    request.max_content_length = media.MAX_ATTACHMENT_BYTES + 64 * 1024  # this route only
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        raise ApiError("bad_request", "Send the file as the multipart field 'file'.", 400)
    try:
        attachment = media.save_attachment(_workspace(svc).root, upload.filename, upload.read())
    except ValueError as exc:
        raise ApiError("bad_attachment", str(exc), 400) from None
    except OSError as exc:
        raise ApiError("storage", f"Could not store the file: {exc.strerror or exc}", 500) from None
    return jsonify({"ok": True, "attachment": attachment})


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


@bp.post("/api/chat/steer")
def steer_chat() -> Response:
    """Queue a steering note into a running turn (applied at the next round)."""
    svc = services()
    data = _json_body()
    conversation_id = data.get("conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise ApiError("bad_request", "conversation_id is required", 400)
    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ApiError("bad_request", "message is required", 400)
    if len(message) > config.MAX_MESSAGE_CHARS:
        raise ApiError("message_too_long", "message is too long", 400)
    if svc.db.get_conversation(conversation_id) is None:
        raise ApiError("not_found", "Conversation not found.", 404)
    queued = svc.agent.steer(conversation_id, message)
    if not queued:
        raise ApiError("not_busy", "No reply is being generated for this chat.", 409)
    return jsonify({"ok": True, "queued": True})


@bp.get("/api/turns")
def get_turns() -> Response:
    svc = services()
    return jsonify({
        "ok": True,
        "active_count": svc.agent.active_turns(),
        "conversations": list(svc.agent._turns.keys()),
    })


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


# --- terminals --------------------------------------------------------------


@bp.get("/api/terminals")
def list_terminals() -> Response:
    from .sandbox import TERMINALS

    conv_id = request.args.get("conversation_id")
    terms = TERMINALS.list_sessions(conversation_id=conv_id)
    return jsonify({"ok": True, "terminals": terms})


@bp.post("/api/terminals/<session_id>/kill")
def kill_terminal(session_id: str) -> Response:
    from .sandbox import SandboxError, TERMINALS

    try:
        info = TERMINALS.kill(session_id)
    except SandboxError as exc:
        raise ApiError("not_found", str(exc), 404) from exc
    return jsonify({"ok": True, "terminal": info})


# --- tasks ------------------------------------------------------------------


@bp.get("/api/tasks")
def list_tasks_route() -> Response:
    from .tasks import TASK_BOARD

    conv_id = request.args.get("conversation_id")
    status = request.args.get("status")
    tasks = TASK_BOARD.list_tasks(conversation_id=conv_id, status=status)
    return jsonify({"ok": True, "tasks": tasks})


@bp.post("/api/tasks")
def modify_tasks_route() -> Response:
    from .tasks import TASK_BOARD

    data = request.get_json(silent=True) or {}
    action = data.get("action", "add")
    conv_id = data.get("conversation_id") or request.args.get("conversation_id")
    res = TASK_BOARD.execute(action, data, conversation_id=conv_id)
    code = 200 if res.get("ok", True) else 400
    return jsonify(res), code



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
