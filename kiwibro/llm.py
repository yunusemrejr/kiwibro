"""LM Studio client: model discovery, streaming chat, non-streaming JSON helpers."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import requests

from . import capabilities as caps
from . import config
from .inference import model_profile, reasoning_payload

log = logging.getLogger(__name__)

CONNECT_TIMEOUT = 5
STREAM_READ_TIMEOUT = 600
COMPLETE_READ_TIMEOUT = 300
MODEL_CACHE_TTL = 5.0
USER_AGENT = "Kiwibro/1.0 (local)"

_CORE_PAYLOAD_KEYS = frozenset({"model", "messages", "stream", "tools"})
_CONTEXT_MARKERS = ("context", "n_ctx", "token", "too long", "too many", "exceeds")
_TOOL_MARKERS = ("tool",)


class LMStudioError(Exception):
    """Recoverable/expected LM Studio failure with a calm, user-facing message."""

    def __init__(self, kind: str, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status = status


class ContextLimitError(LMStudioError):
    def __init__(self, message: str = "The conversation is too long for the model's context window.") -> None:
        super().__init__("context", message)


class ToolsUnsupportedError(LMStudioError):
    def __init__(self, message: str = "This model does not support tools.") -> None:
        super().__init__("tools", message)


class TurnAborted(Exception):
    """Raised inside the streaming callback when the user stops the turn."""


@dataclass
class ChatResult:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    tools_disabled: bool = False


@dataclass
class Completion:
    text: str = ""
    usage: dict[str, int] | None = None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage(raw: Any) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    prompt = _int(raw.get("prompt_tokens"))
    completion = _int(raw.get("completion_tokens"))
    total = _int(raw.get("total_tokens"))
    if prompt is None and completion is None and total is None:
        return None
    return {
        "prompt_tokens": prompt or 0,
        "completion_tokens": completion or 0,
        "total_tokens": total if total is not None else (prompt or 0) + (completion or 0),
    }


def normalize_model(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise one /api/v0/models entry for the UI. Returns None for embeddings."""
    model_id = raw.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    model_type = raw.get("type") or "llm"
    if model_type not in ("llm", "vlm"):
        return None
    state = raw.get("state") or "not-loaded"
    max_ctx = _int(raw.get("max_context_length"))
    loaded_ctx = _int(raw.get("loaded_context_length"))
    if state == "loaded" and loaded_ctx:
        context = loaded_ctx
    else:
        context = min(max_ctx or 32768, 32768)
    capabilities = raw.get("capabilities") or []
    if not isinstance(capabilities, list):
        capabilities = []
    lowered_caps = {str(cap).lower() for cap in capabilities}
    vision = (
        model_type == "vlm"
        or "vision" in lowered_caps
        or bool(raw.get("vision"))
    )
    return {
        "id": model_id,
        "name": raw.get("name") or model_id,
        "type": model_type,
        "state": "loaded" if state == "loaded" else "not-loaded",
        "context": context,
        "max_context": max_ctx or context,
        "tools": "tool_use" in capabilities or bool(raw.get("tool_use")),
        "vision": vision,
        "capabilities": [str(cap) for cap in capabilities],
        "publisher": raw.get("publisher") or "",
        "quantization": raw.get("quantization") or "",
    }


def merge_native_metadata(models: list[dict[str, Any]], native: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add LM Studio ``/api/v1/models`` facts to model records (additive; returns copies).

    v1 adds what v0 lacks: reasoning toggle options, parameter count, file size,
    architecture and the loaded instance's config. A v0 id is either a v1
    ``key`` or one of its ``loaded_instances[].id``.
    """
    by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for entry in native:
        if not isinstance(entry, dict) or not entry.get("key"):
            continue
        instances = [i for i in entry.get("loaded_instances") or [] if isinstance(i, dict)]
        by_id.setdefault(str(entry["key"]), (entry, (instances[0].get("config") or {}) if instances else {}))
        for instance in instances:
            if instance.get("id"):
                by_id[str(instance["id"])] = (entry, instance.get("config") or {})
    out = []
    for model in models:
        match = by_id.get(str(model.get("id")))
        if match is None:
            out.append(model)
            continue
        entry, loaded_config = match
        caps_v1 = entry.get("capabilities") if isinstance(entry.get("capabilities"), dict) else {}
        reasoning = caps_v1.get("reasoning") if isinstance(caps_v1.get("reasoning"), dict) else {}
        quant = entry.get("quantization") if isinstance(entry.get("quantization"), dict) else {}
        merged = dict(model)
        merged.update({
            "key": entry["key"],
            "display_name": entry.get("display_name") or model.get("name"),
            "arch": entry.get("architecture") or "",
            "params": entry.get("params_string") or "",
            "size_bytes": _int(entry.get("size_bytes")),
            "quant_bits": quant.get("bits_per_weight"),
            "reasoning": [str(o) for o in reasoning.get("allowed_options") or []],
            "reasoning_default": reasoning.get("default"),
            "vision": bool(model.get("vision") or caps_v1.get("vision")),
            "tools": bool(model.get("tools") or caps_v1.get("trained_for_tool_use")),
        })
        if loaded_config and model.get("state") == "loaded":
            merged["loaded_config"] = {
                k: loaded_config.get(k) for k in ("context_length", "parallel", "flash_attention")
                if k in loaded_config
            }
        out.append(merged)
    return out


def classify_http_error(status: int, body: str) -> LMStudioError:
    """Map a non-200 response from LM Studio to a typed error."""
    text = (body or "").lower()
    if status >= 500:
        return LMStudioError("server", "LM Studio had a server error. Try again.", status)
    if status in (400, 413, 422):
        if any(marker in text for marker in _TOOL_MARKERS):
            return ToolsUnsupportedError()
        if any(marker in text for marker in _CONTEXT_MARKERS):
            return ContextLimitError()
    if status in (401, 403):
        return LMStudioError("http", "LM Studio rejected the request (check its server settings).", status)
    return LMStudioError("http", "LM Studio could not handle that request.", status)


def parse_json_loose(text: str) -> Any:
    """Extract the first JSON object/array from model text, tolerating fences and prose."""
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            value, _end = decoder.raw_decode(cleaned[index:])
        except ValueError:
            continue
        if isinstance(value, (dict, list)):
            return value
    try:
        return json.loads(cleaned)
    except ValueError:
        return None


class LMStudioClient:
    """Small, synchronous LM Studio client shared by all request threads."""

    def __init__(
        self,
        settings_provider: Callable[[], dict[str, Any]],
        *,
        cache_ttl: float = MODEL_CACHE_TTL,
    ) -> None:
        self._settings = settings_provider
        self._cache: list[dict[str, Any]] | None = None
        self._cache_at = 0.0
        self._cache_ttl = cache_ttl
        self._lock = threading.Lock()
        self._fetch_lock = threading.Lock()
        self._load_api_supported: bool | None = None
        self._instances: dict[str, str] = {}  # model key -> instance id we loaded

    # --- model discovery ---------------------------------------------------

    def base_url(self) -> str:
        url = self._settings().get("lmstudio_url") or config.DEFAULT_LMSTUDIO_URL
        return str(url).rstrip("/")

    def clear_cache(self) -> None:
        with self._lock:
            self._cache = None
            self._cache_at = 0.0

    def list_models(self, force: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            if not force and self._cache is not None and time.monotonic() - self._cache_at < self._cache_ttl:
                return [dict(item) for item in self._cache]
        # Single-flight: concurrent callers share one fetch instead of stampeding.
        with self._fetch_lock:
            with self._lock:
                if not force and self._cache is not None and time.monotonic() - self._cache_at < self._cache_ttl:
                    return [dict(item) for item in self._cache]
            models = self._fetch_models()
            with self._lock:
                self._cache = models
                self._cache_at = time.monotonic()
            return [dict(item) for item in models]

    def _fetch_models(self) -> list[dict[str, Any]]:
        base = self.base_url()
        try:
            response = requests.get(
                f"{base}/api/v0/models", timeout=(CONNECT_TIMEOUT, 15), headers={"User-Agent": USER_AGENT}
            )
        except requests.ConnectionError as exc:
            raise LMStudioError("connection", self._connection_message(base)) from exc
        except requests.Timeout as exc:
            raise LMStudioError("timeout", "LM Studio did not answer in time.") from exc
        except requests.RequestException as exc:  # pragma: no cover - defensive
            raise LMStudioError("connection", self._connection_message(base)) from exc

        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, list):
                models = [m for m in (normalize_model(raw) for raw in data if isinstance(raw, dict)) if m]
                return merge_native_metadata(models, self._native_models())
        # Older LM Studio builds: fall back to the OpenAI-style list.
        try:
            fallback = requests.get(
                f"{base}/v1/models", timeout=(CONNECT_TIMEOUT, 15), headers={"User-Agent": USER_AGENT}
            )
        except requests.RequestException as exc:
            raise LMStudioError("connection", self._connection_message(base)) from exc
        if fallback.status_code != 200:
            raise LMStudioError("http", "LM Studio did not return a model list.", fallback.status_code)
        try:
            payload = fallback.json()
        except ValueError as exc:
            raise LMStudioError("bad_response", "LM Studio returned an invalid model list.") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        models = []
        for raw in data or []:
            if not isinstance(raw, dict) or not raw.get("id"):
                continue
            models.append(normalize_model(raw) or {
                "id": raw["id"],
                "name": raw["id"],
                "type": "llm",
                "state": "not-loaded",
                "context": 32768,
                "max_context": 32768,
                "tools": False,
                "vision": False,
                "capabilities": [],
                "publisher": "",
                "quantization": "",
            })
        return models

    def models_status(self, force: bool = False) -> tuple[bool, list[dict[str, Any]], str | None]:
        try:
            return True, self.list_models(force=force), None
        except LMStudioError as exc:
            return False, [], exc.message

    def pick_model(self, models: list[dict[str, Any]], requested: str | None = None) -> str | None:
        ids = [m["id"] for m in models]
        preferred = requested or self._settings().get("model")
        if preferred and preferred in ids:
            return preferred
        for model in models:
            if model["state"] == "loaded":
                return model["id"]
        return models[0]["id"] if models else None

    def resolve_model(self, requested: str | None = None) -> str:
        ok, models, error = self.models_status()
        base = self.base_url()
        if not ok:
            raise LMStudioError("connection", self._connection_message(base))
        if not models:
            raise LMStudioError("no_model", "No chat model is available in LM Studio. Load a model to start.")
        model = self.pick_model(models, requested)
        if not model:  # pragma: no cover - guarded by models check
            raise LMStudioError("no_model", error or "No chat model is available in LM Studio.")
        return model

    def model_info(self, model_id: str) -> dict[str, Any] | None:
        try:
            models = self.list_models()
        except LMStudioError:
            return None
        for model in models:
            if model["id"] == model_id:
                return model
        return None

    @staticmethod
    def _connection_message(base: str) -> str:
        return f"LM Studio is not reachable at {base}. Open LM Studio and load a model."

    # --- streaming chat ----------------------------------------------------

    def _payload(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
        stream: bool,
        response_format: dict[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format
        if extra:
            # Server knobs such as reasoning_effort, ttl or draft_model; never the core fields.
            payload.update({k: v for k, v in extra.items() if k not in _CORE_PAYLOAD_KEYS})
        return payload

    def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = config.DEFAULT_TEMPERATURE,
        max_tokens: int = config.DEFAULT_MAX_OUTPUT_TOKENS,
        on_delta: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        abort: threading.Event | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ChatResult:
        base = self.base_url()
        url = f"{base}/v1/chat/completions"
        use_tools = bool(tools)

        def attempt(with_tools: bool) -> ChatResult:
            payload = self._payload(
                model, messages, tools if with_tools else None, temperature, max_tokens, stream=True,
                extra=extra,
            )
            result = ChatResult()
            tool_acc: dict[int, dict[str, Any]] = {}
            try:
                with requests.post(
                    url,
                    json=payload,
                    stream=True,
                    timeout=(CONNECT_TIMEOUT, STREAM_READ_TIMEOUT),
                    headers={"Accept": "text/event-stream", "User-Agent": USER_AGENT},
                ) as response:
                    if response.status_code != 200:
                        body = self._read_body(response)
                        raise classify_http_error(response.status_code, body)
                    # LM Studio streams UTF-8 but requests would decode SSE text as
                    # ISO-8859-1; iterate raw bytes and decode explicitly so non-ASCII
                    # content survives (fixes mojibake such as emoji and em-dashes).
                    response.encoding = "utf-8"
                    for raw_line in response.iter_lines(decode_unicode=False):
                        if abort is not None and abort.is_set():
                            raise TurnAborted()
                        if not raw_line:
                            continue
                        if isinstance(raw_line, bytes):
                            line = raw_line.decode("utf-8", errors="replace").strip()
                        else:  # pragma: no cover - requests always yields bytes here
                            line = str(raw_line).strip()
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except ValueError:
                            log.debug("skipping malformed stream chunk: %.200s", data)
                            continue
                        self._consume_chunk(chunk, result, tool_acc, on_delta, on_reasoning)
                    if abort is not None and abort.is_set():
                        raise TurnAborted()
            except requests.ConnectionError as exc:
                raise LMStudioError("connection", self._connection_message(base)) from exc
            except requests.Timeout as exc:
                raise LMStudioError("timeout", "LM Studio stopped responding. Try again.") from exc
            except requests.RequestException as exc:
                raise LMStudioError("stream", "The connection to LM Studio was interrupted.") from exc
            result.tool_calls = self._finish_tool_calls(tool_acc)
            return result

        try:
            return attempt(use_tools)
        except ToolsUnsupportedError:
            if not use_tools:
                raise
            log.info("model %s rejected tools; retrying without tools", model)
            result = attempt(False)
            result.tools_disabled = True
            return result

    @staticmethod
    def _read_body(response: requests.Response) -> str:
        try:
            return response.text[:4000]
        except Exception:  # pragma: no cover - defensive
            return ""

    @staticmethod
    def _consume_chunk(
        chunk: dict[str, Any],
        result: ChatResult,
        tool_acc: dict[int, dict[str, Any]],
        on_delta: Callable[[str], None] | None,
        on_reasoning: Callable[[str], None] | None,
    ) -> None:
        if not isinstance(chunk, dict):
            return
        usage = _usage(chunk.get("usage"))
        if usage:
            result.usage = usage
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        choice = choices[0] if isinstance(choices[0], dict) else {}
        if choice.get("finish_reason"):
            result.finish_reason = str(choice["finish_reason"])
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            return
        content = delta.get("content")
        if isinstance(content, str) and content:
            result.content += content
            if on_delta:
                on_delta(content)
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            result.reasoning += reasoning
            if on_reasoning:
                on_reasoning(reasoning)
        fragments = delta.get("tool_calls")
        if isinstance(fragments, list):
            for fragment in fragments:
                if not isinstance(fragment, dict):
                    continue
                index = fragment.get("index")
                if not isinstance(index, int):
                    index = len(tool_acc)
                entry = tool_acc.setdefault(
                    index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                )
                if fragment.get("id"):
                    entry["id"] = str(fragment["id"])
                if fragment.get("type"):
                    entry["type"] = str(fragment["type"])
                function = fragment.get("function") or {}
                if not isinstance(function, dict):
                    continue
                if function.get("name"):
                    entry["function"]["name"] += str(function["name"])
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    entry["function"]["arguments"] += arguments
                elif isinstance(arguments, dict):
                    entry["function"]["arguments"] += json.dumps(
                        arguments, ensure_ascii=False, separators=(",", ":")
                    )

    @staticmethod
    def _finish_tool_calls(tool_acc: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for index in sorted(tool_acc):
            entry = tool_acc[index]
            if not entry["function"]["name"]:
                continue
            if not entry["id"]:
                entry["id"] = f"call_{index}"
            calls.append(entry)
        return calls

    # --- non-streaming -----------------------------------------------------

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 900,
        temperature: float = 0.0,
        read_timeout: int = COMPLETE_READ_TIMEOUT,
        retries: int = 1,
        response_format: dict[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Completion:
        base = self.base_url()
        url = f"{base}/v1/chat/completions"
        if extra is None:
            extra = self._utility_extra(model)
        payload = self._payload(
            model, messages, None, temperature, max_tokens, stream=False,
            response_format=response_format, extra=extra,
        )
        last_error: LMStudioError | None = None
        for attempt in range(retries + 1):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=(CONNECT_TIMEOUT, read_timeout),
                    headers={"User-Agent": USER_AGENT},
                )
            except requests.ConnectionError as exc:
                last_error = LMStudioError("connection", self._connection_message(base))
                if attempt < retries:
                    time.sleep(0.4)
                    continue
                raise last_error from exc
            except requests.Timeout as exc:
                last_error = LMStudioError("timeout", "LM Studio did not answer in time.")
                if attempt < retries:
                    continue
                raise last_error from exc
            except requests.RequestException as exc:  # pragma: no cover - defensive
                last_error = LMStudioError("connection", self._connection_message(base))
                if attempt < retries:
                    continue
                raise last_error from exc
            if response.status_code != 200:
                error = classify_http_error(response.status_code, self._read_body(response))
                if isinstance(error, (ContextLimitError, ToolsUnsupportedError)):
                    raise error
                last_error = error
                if attempt < retries:
                    time.sleep(0.4)
                    continue
                raise error
            try:
                data = response.json()
            except ValueError as exc:
                raise LMStudioError("bad_response", "LM Studio returned an invalid response.") from exc
            choices = data.get("choices") if isinstance(data, dict) else None
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise LMStudioError("bad_response", "LM Studio returned an empty response.")
            message = choices[0].get("message") or {}
            content = message.get("content") or message.get("reasoning_content") or ""
            return Completion(text=str(content), usage=_usage(data.get("usage")))
        raise last_error or LMStudioError("http", "LM Studio could not handle that request.")

    def _utility_extra(self, model: str) -> dict[str, Any] | None:
        """Non-streaming calls are utility work (summaries, memory, titles, helpers), so they
        never think: on a laptop, reasoning tokens cost as much as answer tokens, and an
        answer that is all reasoning would otherwise be returned as the content."""
        info = self.model_info(model)
        if info is None:
            return None
        return reasoning_payload(model_profile(model, info), think=False) or None

    def complete_json(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 900,
        temperature: float = 0.0,
        read_timeout: int = COMPLETE_READ_TIMEOUT,
        retries: int = 1,
        response_format: dict[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Any:
        try:
            completion = self.complete(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                read_timeout=read_timeout,
                retries=retries,
                response_format=response_format,
                extra=extra,
            )
        except LMStudioError as exc:
            log.debug("complete_json failed: %s", exc.message)
            return None
        return parse_json_loose(completion.text)

    # --- capability catalog, vision, embeddings, load control -------------
    # Serving primitives consumed by the routing layers. Selection policy
    # (which helper model to run) is NOT here; see routing.py.

    def _native_models(self) -> list[dict[str, Any]]:
        """``/api/v1/models`` entries, or [] on builds/servers without it (optional metadata)."""
        try:
            response = requests.get(
                f"{self.base_url()}/api/v1/models", timeout=(CONNECT_TIMEOUT, 5),
                headers={"User-Agent": USER_AGENT},
            )
            payload = response.json() if response.status_code == 200 else None
        except Exception:  # metadata is optional: never fail the model list over it
            return []
        native = payload.get("models") if isinstance(payload, dict) else None
        return [entry for entry in native if isinstance(entry, dict)] if isinstance(native, list) else []

    def _fetch_raw_models(self) -> list[dict[str, Any]]:
        base = self.base_url()
        for path in ("/api/v0/models", "/v1/models"):
            try:
                response = requests.get(
                    f"{base}{path}", timeout=(CONNECT_TIMEOUT, 15),
                    headers={"User-Agent": USER_AGENT},
                )
            except requests.RequestException as exc:
                raise LMStudioError("connection", self._connection_message(base)) from exc
            if response.status_code != 200:
                continue
            try:
                payload = response.json()
            except ValueError:
                continue
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, list):
                return [raw for raw in data if isinstance(raw, dict)]
        raise LMStudioError("http", "LM Studio did not return a model list.")

    def catalog(self, force: bool = False) -> list[dict[str, Any]]:
        """ALL installed models (chat + vision + embeddings), capability-annotated.

        Unlike :meth:`list_models` (the UI's chat picker), nothing is filtered
        out, and each entry carries ``vision``/``audio``/``embeddings``/
        ``params_b``/``est_memory_gb`` from :mod:`kiwibro.capabilities`.
        """
        entries: list[dict[str, Any]] = []
        for raw in self._fetch_raw_models():
            entry = normalize_model(raw) or {
                "id": raw.get("id"), "name": raw.get("id"),
                "type": raw.get("type") or "embeddings",
                "state": "loaded" if raw.get("state") == "loaded" else "not-loaded",
                "context": _int(raw.get("loaded_context_length"))
                or _int(raw.get("max_context_length")) or 2048,
                "tools": False, "vision": False,
                "capabilities": raw.get("capabilities") or [],
                "publisher": raw.get("publisher") or "",
                "quantization": raw.get("quantization") or "",
            }
            entries.append(entry)
        return [caps.infer_capabilities(e) for e in merge_native_metadata(entries, self._native_models())]

    def complete_vision(
        self,
        *,
        model: str,
        prompt: str,
        image_data_uris: list[str],
        system: str | None = None,
        max_tokens: int = 900,
        temperature: float = 0.0,
        read_timeout: int = COMPLETE_READ_TIMEOUT,
        extra: Mapping[str, Any] | None = None,
    ) -> Completion:
        """One non-streaming vision call. ``image_data_uris``: ``data:image/...;base64,...``."""
        uris = [u for u in image_data_uris if isinstance(u, str) and u.startswith("data:image/")]
        if not uris:
            raise LMStudioError("bad_request", "No valid image data was provided.")
        if len(uris) > 4 or sum(len(u) for u in uris) > 12_000_000:
            raise LMStudioError("bad_request", "Too much image data for one request.")
        parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        parts += [{"type": "image_url", "image_url": {"url": uri}} for uri in uris]
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": parts})
        return self.complete(
            model=model, messages=messages, max_tokens=max_tokens,
            temperature=temperature, read_timeout=read_timeout, retries=0, extra=extra,
        )

    def embed(
        self,
        *,
        model: str,
        texts: list[str],
        read_timeout: int = 120,
    ) -> list[list[float]]:
        """Embed texts via ``/v1/embeddings``; returns vectors in input order."""
        if not texts:
            return []
        base = self.base_url()
        try:
            response = requests.post(
                f"{base}/v1/embeddings",
                json={"model": model, "input": texts},
                timeout=(CONNECT_TIMEOUT, read_timeout),
                headers={"User-Agent": USER_AGENT},
            )
        except requests.ConnectionError as exc:
            raise LMStudioError("connection", self._connection_message(base)) from exc
        except requests.RequestException as exc:
            raise LMStudioError("timeout", "LM Studio did not answer in time.") from exc
        if response.status_code != 200:
            raise classify_http_error(response.status_code, self._read_body(response))
        try:
            payload = response.json()
        except ValueError as exc:
            raise LMStudioError("bad_response", "LM Studio returned an invalid embedding response.") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise LMStudioError("bad_response", "LM Studio returned no embeddings.")
        ordered = sorted(
            (d for d in data if isinstance(d, dict)),
            key=lambda d: d.get("index") if isinstance(d.get("index"), int) else 0,
        )
        return [[float(v) for v in d.get("embedding") or []] for d in ordered]

    def _load_api_call(self, action: str, model: str, context_length: int | None = None) -> bool:
        """LM Studio's v1 load control: ``POST /api/v1/models/load|unload``.

        ``load`` takes ``{"model", "context_length"?}`` and answers
        ``{"instance_id", "status": "loaded", ...}``; ``unload`` takes that
        ``instance_id`` (remembered here; it can differ from the model key).
        Builds without the API answer ``Unexpected endpoint``; that is cached.
        """
        if self._load_api_supported is False:
            return False
        if action == "load":
            body: dict[str, Any] = {"model": model}
            if context_length:
                body["context_length"] = int(context_length)
        else:
            with self._lock:
                body = {"instance_id": self._instances.get(model, model)}
        try:
            response = requests.post(
                f"{self.base_url()}/api/v1/models/{action}",
                json=body,
                timeout=(CONNECT_TIMEOUT, 180),
                headers={"User-Agent": USER_AGENT},
            )
        except requests.RequestException:
            return False
        text = self._read_body(response)
        if response.status_code == 404 or "Unexpected endpoint" in text:
            self._load_api_supported = False
            return False
        if response.status_code != 200:
            return False
        self._load_api_supported = True
        instance = None
        if action == "load":
            try:
                instance = (response.json() or {}).get("instance_id")
            except (ValueError, AttributeError):
                instance = None
        with self._lock:
            if action == "load":
                self._instances[model] = str(instance or model)
            else:
                self._instances.pop(model, None)
        self.clear_cache()
        return True

    def load_model(self, model: str, context_length: int | None = None) -> bool:
        """Best-effort explicit load; False means rely on LM Studio JIT loading."""
        return self._load_api_call("load", model, context_length)

    def unload_model(self, model: str) -> bool:
        """Best-effort explicit unload of a model this client loaded (or a loaded instance id)."""
        return self._load_api_call("unload", model)
