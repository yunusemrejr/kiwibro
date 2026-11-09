"""Model client: discovery, streaming chat, non-streaming JSON helpers.

Kiwibro runs llama.cpp itself (:mod:`kiwibro.engine`) and this client talks to the
servers it starts. With ``KIWIBRO_SERVER_URL`` set it attaches to an llama.cpp-compatible
server that is already running instead; everything above this module is the same."""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import requests

from . import capabilities as caps
from . import config
from .inference import NO_THINKING

log = logging.getLogger(__name__)

CONNECT_TIMEOUT = 5
STREAM_READ_TIMEOUT = 600
COMPLETE_READ_TIMEOUT = 300
MODEL_CACHE_TTL = 5.0
USER_AGENT = "Kiwibro/1.0 (local)"

_CORE_PAYLOAD_KEYS = frozenset({"model", "messages", "stream", "tools"})
_CONTEXT_MARKERS = ("context", "n_ctx", "token", "too long", "too many", "exceeds")
_TOOL_MARKERS = ("tool",)
NO_MODELS = "No models yet. Open Models to download one, or drop a GGUF file into Kiwibro's models folder."
NO_MODELS_ATTACHED = "The attached model server lists no chat models."


class LLMError(Exception):
    """Recoverable/expected model-server failure with a calm, user-facing message."""

    def __init__(self, kind: str, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status = status


class ContextLimitError(LLMError):
    def __init__(self, message: str = "The conversation is too long for the model's context window.") -> None:
        super().__init__("context", message)


class ToolsUnsupportedError(LLMError):
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
    leaked_reasoning: bool = False  # the model "thought" in plain content (moved to reasoning)
    timings: dict[str, float] | None = None  # llama.cpp's own counts (direct engine): see _timings


@dataclass
class Completion:
    text: str = ""
    usage: dict[str, int] | None = None
    finish_reason: str | None = None


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


_TIMING_KEYS = ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "predicted_per_second")


def _timings(raw: Any) -> dict[str, float] | None:
    """llama-server's per-request ``timings``: tokens served from the KV cache (``cache_n``),
    tokens prefilled (``prompt_n``) and their times. Exact where Kiwibro otherwise estimates."""
    if not isinstance(raw, dict):
        return None
    out = {key: float(raw[key]) for key in _TIMING_KEYS
           if isinstance(raw.get(key), (int, float)) and not isinstance(raw.get(key), bool) and raw[key] >= 0}
    return out if "prompt_n" in out else None


def normalize_model(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise one model entry for the UI (the engine's, or an attached server's
    ``/v1/models`` item). Returns None for embeddings."""
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
    """Add the engine's file facts to model records (additive; returns copies): reasoning
    toggle options, parameter count, file size, architecture and the running server's
    config. A record id is either an entry ``key`` or one of its ``loaded_instances[].id``.
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
            # Where the file lives; only Kiwibro's own downloads can be deleted from the app.
            "source": str(entry.get("source") or ""),
            "deletable": bool(entry.get("deletable")),
        })
        if loaded_config and model.get("state") == "loaded":
            merged["loaded_config"] = {
                k: loaded_config.get(k) for k in ("context_length", "parallel", "flash_attention")
                if k in loaded_config
            }
        out.append(merged)
    return out


def classify_http_error(status: int, body: str) -> LLMError:
    """Map a non-200 response from the model server to a typed error."""
    text = (body or "").lower()
    if status >= 500:
        return LLMError("server", "The model server had an error. Try again.", status)
    if status in (400, 413, 422):
        if any(marker in text for marker in _TOOL_MARKERS):
            return ToolsUnsupportedError()
        if any(marker in text for marker in _CONTEXT_MARKERS):
            return ContextLimitError()
    if status in (401, 403):
        return LLMError("http", "The model server rejected the request (check its API key).", status)
    return LLMError("http", "The model could not handle that request.", status)


def parse_json_loose(text: str) -> Any:
    """The first JSON object/array in model text, repaired the way small models need
    (fences, prose, trailing commas, single quotes, Python literals). One repair
    implementation for the whole app: toolcall.repair_json."""
    from .toolcall import repair_json  # stdlib-only module; imported lazily to keep llm.py a leaf

    return repair_json(text) if text else None


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Drop <think> blocks (and an unclosed trailing one) so a JSON example the model
    considered while thinking is never parsed as its answer."""
    text = _THINK_RE.sub("", text or "")
    head, sep, tail = text.partition("</think>")
    if sep:  # template opened the block for us; only the closer is in the text
        text = tail
    opened = text.lower().find("<think>")
    return text[:opened] if opened >= 0 else text


def _socket_of(response: requests.Response) -> socket.socket | None:
    """The socket under a streaming response. urllib3 2.x detaches it from the connection
    while the body streams; it sits under the http.client response's file object then."""
    raw = response.raw
    sock = getattr(getattr(raw, "_connection", None), "sock", None)
    if sock is None:
        reader = getattr(getattr(raw, "_fp", None), "fp", None)  # BufferedReader over SocketIO
        sock = getattr(getattr(reader, "raw", None), "_sock", None)
    return sock if isinstance(sock, socket.socket) else None


class LLMClient:
    """Small, synchronous client for the model servers, shared by all request threads."""

    def __init__(
        self,
        settings_provider: Callable[[], dict[str, Any]],
        *,
        cache_ttl: float = MODEL_CACHE_TTL,
        data_dir: str | Path | None = None,
    ) -> None:
        self._settings = settings_provider
        self._data_dir = data_dir
        self._direct: Any = None  # engine.DirectEngine, created on first use
        self._direct_lock = threading.Lock()
        self._cache: list[dict[str, Any]] | None = None
        self._cache_at = 0.0
        self._cache_ttl = cache_ttl
        self._lock = threading.Lock()
        self._fetch_lock = threading.Lock()
        self._props: dict[str, dict[str, Any]] = {}  # attached server: /props per model id

    # --- engine switch -----------------------------------------------------

    def server_url(self) -> str:
        """The already-running server Kiwibro is attached to ('' when it runs its own)."""
        return str(self._settings().get("server_url") or "").strip().rstrip("/")

    def engine_mode(self) -> str:
        return "attached" if self.server_url() else "local"

    def direct_on(self) -> bool:
        """Kiwibro starts and stops the llama.cpp servers itself (the normal case)."""
        return not self.server_url()

    def direct_engine(self) -> Any:
        """The engine that runs llama.cpp (created lazily: an attached client never needs it)."""
        with self._direct_lock:
            if self._direct is None:
                from .engine import DirectEngine

                self._direct = DirectEngine(self._settings, self._data_dir)
            return self._direct

    def stop_direct(self) -> None:
        """Stop every llama.cpp server the engine runs (frees their memory at once)."""
        with self._direct_lock:
            engine = self._direct
        if engine is not None:
            engine.shutdown()
        self.clear_cache()

    @contextmanager
    def _serving(self, model: str) -> Iterator[str]:
        """The base URL that serves ``model``: a server of our own held busy for the
        request (started just in time), or the attached one."""
        if not self.direct_on():
            yield self.server_url()
            return
        from .engine import EngineError

        engine = self.direct_engine()
        try:
            server = engine.acquire(model)
        except EngineError as exc:
            raise LLMError("engine", str(exc)) from exc
        try:
            yield server.url
        finally:
            engine.release(server)

    # --- model discovery ---------------------------------------------------

    def base_url(self) -> str:
        return self.server_url()

    def _headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        """Request headers; our own servers take the engine's key, an attached one the
        key in ``KIWIBRO_SERVER_KEY`` (when it has authentication on)."""
        headers = {"User-Agent": USER_AGENT, **(extra or {})}
        if self.direct_on():
            headers["Authorization"] = f"Bearer {self.direct_engine().api_key}"
            return headers
        token = str(os.environ.get("KIWIBRO_SERVER_KEY") or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def clear_cache(self) -> None:
        with self._lock:
            self._cache = None
            self._cache_at = 0.0
            self._props.clear()

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
        models = [m for m in (normalize_model(raw) for raw in self._fetch_raw_models()) if m]
        return merge_native_metadata(models, self._native_models())

    def models_status(self, force: bool = False) -> tuple[bool, list[dict[str, Any]], str | None]:
        try:
            return True, self.list_models(force=force), None
        except LLMError as exc:
            return False, [], exc.message

    def pick_model(self, models: list[dict[str, Any]], requested: str | None = None) -> str | None:
        ids = [m["id"] for m in models]
        preferred = requested or self._settings().get("model")
        if preferred and preferred in ids:
            return preferred
        if preferred and self.direct_on():
            # An id saved by an older version (e.g. liquid/lfm2.5-1.2b) names the same file here.
            canonical = self.direct_engine().canonical(preferred)
            if canonical in ids:
                return canonical
        for model in models:
            if model["state"] == "loaded":
                return model["id"]
        return models[0]["id"] if models else None

    def resolve_model(self, requested: str | None = None) -> str:
        ok, models, error = self.models_status()
        if not ok:
            if self.direct_on():
                raise LLMError("no_engine", error or NO_MODELS)
            raise LLMError("connection", self._connection_message(self.server_url()))
        if not models:
            raise LLMError("no_model", NO_MODELS if self.direct_on() else NO_MODELS_ATTACHED)
        model = self.pick_model(models, requested)
        if not model:  # pragma: no cover - guarded by models check
            raise LLMError("no_model", error or NO_MODELS)
        return model

    def model_info(self, model_id: str) -> dict[str, Any] | None:
        try:
            models = self.list_models()
        except LLMError:
            return None
        for model in models:
            if model["id"] == model_id:
                return model
        return None

    def _connection_message(self, base: str) -> str:
        if self.direct_on():
            return "The model engine stopped responding. Try again."
        return f"The model server at {base} is not reachable."

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
        if self.direct_on():
            from .engine import adapt_payload

            payload = adapt_payload(payload)
            try:  # task adapters: which LoRA is on for this request (adapters.py)
                lora = self.direct_engine().lora_for(model, messages)
            except Exception:  # noqa: BLE001 - an adapter problem must never fail a request
                lora = None
            if lora is not None:
                payload["lora"] = lora
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
        on_rewrite: Callable[[str, str], None] | None = None,
    ) -> ChatResult:
        use_tools = bool(tools)
        base = url = ""  # set by _serving below, before attempt() runs

        def attempt(with_tools: bool) -> ChatResult:
            payload = self._payload(
                model, messages, tools if with_tools else None, temperature, max_tokens, stream=True,
                extra=extra,
            )
            result = ChatResult()
            tool_acc: dict[int, dict[str, Any]] = {}
            finished = threading.Event()
            try:
                with requests.post(
                    url,
                    json=payload,
                    stream=True,
                    timeout=(CONNECT_TIMEOUT, STREAM_READ_TIMEOUT),
                    headers=self._headers({"Accept": "text/event-stream"}),
                ) as response:
                    if abort is not None:
                        # Stop must work during prefill too, when the server sends nothing for
                        # minutes: closing the socket makes it drop the request at its next
                        # batch (measured: a 75 s prefill stopped ~20 s after the abort).
                        threading.Thread(
                            target=self._close_on_abort, args=(response, abort, finished),
                            name="kiwibro-stream-watch", daemon=True,
                        ).start()
                    if response.status_code != 200:
                        body = self._read_body(response)
                        raise classify_http_error(response.status_code, body)
                    # The server streams UTF-8 but requests would decode SSE text as
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
                        self._consume_chunk(chunk, result, tool_acc, on_delta, on_reasoning, on_rewrite)
                    if abort is not None and abort.is_set():
                        raise TurnAborted()
            except requests.RequestException as exc:
                if abort is not None and abort.is_set():
                    raise TurnAborted() from exc  # our own watcher closed the connection
                if isinstance(exc, requests.ConnectionError):
                    raise LLMError("connection", self._connection_message(base)) from exc
                if isinstance(exc, requests.Timeout):
                    raise LLMError("timeout", "The model stopped responding. Try again.") from exc
                raise LLMError("stream", "The connection to the model was interrupted.") from exc
            finally:
                finished.set()
            result.tool_calls = self._finish_tool_calls(tool_acc)
            return result

        with self._serving(model) as base:
            url = f"{base}/v1/chat/completions"
            try:
                return attempt(use_tools)
            except ToolsUnsupportedError:
                if not use_tools:
                    raise
                log.info("model %s rejected tools; retrying without tools", model)
                result = attempt(False)
                result.tools_disabled = True
                return result

    def prefill(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        extra: Mapping[str, Any] | None = None,
        abort: threading.Event | None = None,
    ) -> None:
        """Process a prompt and stop at its first token (KV-cache warm-up).

        The server sends nothing while it prefills, so an abort flag checked per stream
        line would wait minutes; a watcher shuts the socket down instead, which ends
        the request mid-prefill."""
        with self._serving(model) as base:
            payload = self._payload(model, messages, tools, 0.0, 1, stream=True, extra=extra)
            finished = threading.Event()
            try:
                with requests.post(
                    f"{base}/v1/chat/completions", json=payload, stream=True,
                    timeout=(CONNECT_TIMEOUT, STREAM_READ_TIMEOUT),
                    headers=self._headers({"Accept": "text/event-stream"}),
                ) as response:
                    if abort is not None:
                        threading.Thread(
                            target=self._close_on_abort, args=(response, abort, finished),
                            name="kiwibro-prefill-watch", daemon=True,
                        ).start()
                    if response.status_code != 200:
                        raise classify_http_error(response.status_code, self._read_body(response))
                    for raw_line in response.iter_lines(decode_unicode=False):
                        if raw_line and raw_line.startswith(b"data:"):
                            break  # the first token: the whole prompt is in the cache now
            except requests.RequestException as exc:
                if abort is not None and abort.is_set():
                    raise TurnAborted() from exc
                raise LLMError("connection", self._connection_message(base)) from exc
            finally:
                finished.set()
            if abort is not None and abort.is_set():
                raise TurnAborted()

    @staticmethod
    def _close_on_abort(response: requests.Response, abort: threading.Event, finished: threading.Event) -> None:
        """Shut the socket down once ``abort`` is set: the reading thread wakes with EOF or an
        error and closes the response itself (closing it from here races its read)."""
        while not finished.wait(0.2):
            if abort.is_set():
                sock = _socket_of(response)
                if sock is not None:  # unknown layout: the abort is seen at the next line instead
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                return

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
        on_rewrite: Callable[[str, str], None] | None = None,
    ) -> None:
        if not isinstance(chunk, dict):
            return
        usage = _usage(chunk.get("usage"))
        if usage:
            result.usage = usage
        timings = _timings(chunk.get("timings"))
        if timings:
            result.timings = timings
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
            if not result.leaked_reasoning and "</think>" in result.content:
                LLMClient._split_leaked_reasoning(result, on_rewrite)
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
    def _split_leaked_reasoning(result: ChatResult, on_rewrite: Callable[[str, str], None] | None) -> None:
        """Move reasoning a model wrote into plain content (ending in a bare ``</think>``).

        With thinking switched off, the template pre-fills an empty think block and
        some small models think anyway, closing a tag the server never saw opened.
        """
        before, _sep, after = result.content.partition("</think>")
        opened = before.lstrip().startswith("<think>")
        if "<think>" in before and not opened:
            return  # a literal tag pair inside an answer, not leaked reasoning
        reasoning = before.lstrip()[len("<think>"):] if opened else before
        result.leaked_reasoning = True
        result.reasoning += reasoning.strip()
        result.content = after.lstrip()
        if on_rewrite:
            on_rewrite(result.content, reasoning.strip())

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
        abort: threading.Event | None = None,
    ) -> Completion:
        if extra is None:
            # Non-streaming calls are utility work (summaries, memory, titles, helpers):
            # they never think. Reasoning tokens cost as much as answer tokens on a
            # laptop, and an all-reasoning answer would come back as the content.
            extra = NO_THINKING
        if abort is not None and response_format is None:
            # Cancellable background work (memory extraction) streams, so an abort can
            # close the connection mid-prefill; raises TurnAborted when it does.
            streamed = self.stream_chat(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                abort=abort, extra=extra,
            )
            return Completion(
                text=streamed.content or streamed.reasoning or "", usage=streamed.usage,
                finish_reason=streamed.finish_reason,
            )
        with self._serving(model) as base:
            url = f"{base}/v1/chat/completions"
            payload = self._payload(
                model, messages, None, temperature, max_tokens, stream=False,
                response_format=response_format, extra=extra,
            )
            last_error: LLMError | None = None
            for attempt in range(retries + 1):
                try:
                    response = requests.post(
                        url,
                        json=payload,
                        timeout=(CONNECT_TIMEOUT, read_timeout),
                        headers=self._headers(),
                    )
                except requests.ConnectionError as exc:
                    last_error = LLMError("connection", self._connection_message(base))
                    if attempt < retries:
                        time.sleep(0.4)
                        continue
                    raise last_error from exc
                except requests.Timeout as exc:
                    last_error = LLMError("timeout", "The model did not answer in time.")
                    if attempt < retries:
                        continue
                    raise last_error from exc
                except requests.RequestException as exc:  # pragma: no cover - defensive
                    last_error = LLMError("connection", self._connection_message(base))
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
                    raise LLMError("bad_response", "The model server returned an invalid response.") from exc
                choices = data.get("choices") if isinstance(data, dict) else None
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    raise LLMError("bad_response", "The model server returned an empty response.")
                message = choices[0].get("message") or {}
                content = message.get("content") or message.get("reasoning_content") or ""
                return Completion(
                    text=str(content),
                    usage=_usage(data.get("usage")),
                    finish_reason=str(choices[0].get("finish_reason") or "") or None,
                )
            raise last_error or LLMError("http", "The model could not handle that request.")

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
        abort: threading.Event | None = None,
    ) -> Any:
        """JSON from a utility call, or None. With ``abort``, TurnAborted propagates."""
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
                **({"abort": abort} if abort is not None else {}),
            )
        except LLMError as exc:
            log.debug("complete_json failed: %s", exc.message)
            return None
        text = strip_reasoning(completion.text)
        if completion.finish_reason == "length":
            # Cut off by max_tokens: a leniently repaired tail ("User lives in Berl")
            # would be stored as fact, so only fully closed JSON is trusted.
            try:
                return json.loads(text.strip().removeprefix("```json").strip("`").strip())
            except ValueError:
                log.debug("complete_json: truncated JSON reply ignored")
                return None
        return parse_json_loose(text)

    # --- capability catalog, vision, embeddings, load control -------------
    # Serving primitives consumed by the routing layers. Selection policy
    # (which helper model to run) is NOT here; see routing.py.

    def _native_models(self) -> list[dict[str, Any]]:
        """File facts for the engine's models; [] for an attached server (optional metadata)."""
        return self.direct_engine().native_entries() if self.direct_on() else []

    def _fetch_raw_models(self) -> list[dict[str, Any]]:
        if self.direct_on():
            engine = self.direct_engine()
            if not engine.has_runtime():
                from .engine import NO_RUNTIME

                raise LLMError("no_engine", NO_RUNTIME)
            return engine.raw_entries()
        base = self.server_url()
        try:
            response = requests.get(f"{base}/v1/models", timeout=(CONNECT_TIMEOUT, 15), headers=self._headers())
        except requests.Timeout as exc:
            raise LLMError("timeout", "The model server did not answer in time.") from exc
        except requests.RequestException as exc:
            raise LLMError("connection", self._connection_message(base)) from exc
        if response.status_code != 200:
            raise LLMError("http", "The model server did not return a model list.", response.status_code)
        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMError("bad_response", "The model server returned an invalid model list.") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        return [self._described(base, raw) for raw in data or [] if isinstance(raw, dict) and raw.get("id")]

    def _described(self, base: str, raw: dict[str, Any]) -> dict[str, Any]:
        """What a plain llama.cpp ``/v1/models`` item leaves out (window, tools, vision),
        read from ``/props``. An item that already describes itself is used as it is."""
        if "max_context_length" in raw or "capabilities" in raw:
            return raw
        entry = dict(raw)
        status = raw.get("status") if isinstance(raw.get("status"), dict) else {}
        entry.setdefault("state", "not-loaded" if status.get("value") == "unloaded" else "loaded")
        props = self._server_props(base, str(raw["id"]))
        generation = props.get("default_generation_settings")
        meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
        window = _int((generation or {}).get("n_ctx") if isinstance(generation, dict) else None) or _int(meta.get("n_ctx_train"))
        if window:
            entry["max_context_length"] = entry["loaded_context_length"] = window
        template = str(props.get("chat_template") or "").lower()
        capabilities = ["tool_use"] if "tools" in template or "tool_call" in template else []
        modalities = props.get("modalities") if isinstance(props.get("modalities"), dict) else {}
        if modalities.get("vision"):
            entry["type"] = "vlm"
            capabilities.append("vision")
        entry["capabilities"] = capabilities
        return entry

    def _server_props(self, base: str, model_id: str) -> dict[str, Any]:
        with self._lock:
            cached = self._props.get(model_id)
        if cached is not None:
            return cached
        try:
            response = requests.get(
                f"{base}/props", params={"model": model_id}, timeout=(CONNECT_TIMEOUT, 5), headers=self._headers()
            )
            props = response.json() if response.status_code == 200 else {}
        except (requests.RequestException, ValueError):  # optional facts: never fail the list over them
            props = {}
        props = props if isinstance(props, dict) else {}
        with self._lock:
            self._props[model_id] = props
        return props

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
            raise LLMError("bad_request", "No valid image data was provided.")
        if len(uris) > 4 or sum(len(u) for u in uris) > 12_000_000:
            raise LLMError("bad_request", "Too much image data for one request.")
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
        with self._serving(model) as base:
            try:
                response = requests.post(
                    f"{base}/v1/embeddings",
                    json={"model": model, "input": texts},
                    timeout=(CONNECT_TIMEOUT, read_timeout),
                    headers=self._headers(),
                )
            except requests.ConnectionError as exc:
                raise LLMError("connection", self._connection_message(base)) from exc
            except requests.RequestException as exc:
                raise LLMError("timeout", "The model did not answer in time.") from exc
            if response.status_code != 200:
                raise classify_http_error(response.status_code, self._read_body(response))
            try:
                payload = response.json()
            except ValueError as exc:
                raise LLMError("bad_response", "The model server returned an invalid embedding response.") from exc
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, list):
                raise LLMError("bad_response", "The model server returned no embeddings.")
            ordered = sorted(
                (d for d in data if isinstance(d, dict)),
                key=lambda d: d.get("index") if isinstance(d.get("index"), int) else 0,
            )
            return [[float(v) for v in d.get("embedding") or []] for d in ordered]

    def load_model(self, model: str, context_length: int | None = None, parallel: int | None = None) -> bool:
        """Start ``model`` now (optionally with a window size). False for an attached
        server, which loads what it was started with."""
        if not self.direct_on():
            return False
        loaded = self.direct_engine().load(model, context_length, parallel)
        self.clear_cache()
        return loaded

    def loaded_instance(self, model: str) -> str | None:
        """The id requests should use for ``model`` once it is loaded."""
        return self.direct_engine().canonical(model) if self.direct_on() else None

    def unload_model(self, model: str) -> bool:
        """Stop ``model``'s server and free its memory (False for an attached server)."""
        if not self.direct_on():
            return False
        unloaded = self.direct_engine().unload(model)
        self.clear_cache()
        return unloaded
