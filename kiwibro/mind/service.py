"""Kiwi Mind as the app sees it: is it usable, what is its state, and the one place that owns its parts."""

from __future__ import annotations

import atexit
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..engine import DirectEngine, LocalModel
from ..llm import ChatResult, Completion, LLMError, TurnAborted
from . import KIWIMIND
from .client import MindClient
from .conductor import Conductor, Settings, has_images
from .conductor3 import Conductor3
from .host import MindHost, _rss, find_node
from .little import Little
from .llama import NO_RUNTIME, LlamaClient, LlamaHost, LlamaRuntime
from .native_generic import NATIVE_FORMAT, NATIVE_MODEL, NativeChatClient, NativeConductor, describe_pack
from .pack import LEGACY_REVISION, MEMORY_FILE, MEMORY_REVISION, TRUNK, TRUNK3, Pack, PackError

log = logging.getLogger(__name__)

NODE_RECHECK_S = 30.0
DEFAULT_CTX = 8192
PRIMARY_RETRY_S = 60.0


class Mind:
    """Pack + host process + conductor. One per app; nothing starts until it is used.

    A 3.x pack runs on llama.cpp and a 2.x pack on Kiwibro's own engine under Node.js; the parts are built
    for the pack that is installed and built again when a pack of the other kind arrives."""

    def __init__(
        self, data_dir: str | Path | None, *, helper: Callable[[], dict[str, Any] | None],
        helper_stream: Callable[..., ChatResult], base_url: str | None = None, node: str | None = None,
        runtime: Callable[[], LlamaRuntime | None] | None = None,
        native_engine: Callable[[], DirectEngine] | None = None,
    ) -> None:
        self.enabled = data_dir is not None
        folder = data_dir if data_dir is not None else Path("/nonexistent-kiwibro")
        self.pack = Pack(folder, base_url=base_url)
        self.candidate_pack = Pack(folder, base_url=base_url, subdir="candidate", prefix="candidate", entry=TRUNK3,
                                   expected={"engine": "llama", "format": NATIVE_FORMAT})
        self._native = NativeChatClient(native_engine) if native_engine is not None else None
        self.legacy_pack = Pack(folder, base_url=base_url, subdir="legacy", revision=LEGACY_REVISION, entry=TRUNK,
                                expected={"base": "Qwen/Qwen3-0.6B"})
        self.memory_pack = Pack(folder, base_url=base_url, subdir="experts", prefix="experts",
                                revision=MEMORY_REVISION, entry=MEMORY_FILE,
                                expected={"name": "kiwimind-memory-expert", "engine": "llama",
                                          "base": "Qwen/Qwen3-0.6B", "job": "memory"})
        self.helper = helper
        self._helper_stream = helper_stream
        self._node = node
        self._find_runtime = runtime or (lambda: None)
        self._node_memo: tuple[float, str | None, str] | None = None
        self._runtime_memo: tuple[float, LlamaRuntime | None] | None = None
        self._notified = False
        self._lock = threading.Lock()
        self._primary_retry_at = 0.0
        self._build()
        self.legacy_host = MindHost(self.legacy_pack, node=self._node)
        self.legacy_conductor = Conductor(
            MindClient(self.legacy_host),
            Settings(lambda: self._ctx(self.legacy_pack), self.helper, self._helper_stream,
                     lambda: self._open_at(self.legacy_pack)),
        )
        self.little = Little(data_dir, self.runtime, self._chat_busy)
        self.memory_expert = Little(data_dir, self.runtime, self._chat_busy, model_file=self._memory_file,
                                    model_name="Kiwi Mind Memory Expert", log_name="memory-expert.log")
        self.pack.on_installed = self._installed
        self.candidate_pack.on_installed = self._candidate_installed
        self.legacy_pack.on_installed = self.legacy_host.stop
        self.memory_pack.on_installed = self.memory_expert.stop
        atexit.register(self.stop)

    def _build(self) -> None:
        self.engine = self.pack.engine()
        settings = Settings(self.ctx, self.helper, self._helper_stream, self.open_at, self.heavy_at, self._primary_sees)
        if self.engine == "llama":
            self.host: MindHost = LlamaHost(self.pack, self.runtime)
            self.client: MindClient = LlamaClient(self.host)
            self.conductor: Conductor = Conductor3(self.client, settings)
        else:
            self.host = MindHost(self.pack, node=self._node)
            self.client = MindClient(self.host)
            self.conductor = Conductor(self.client, settings)

    def _installed(self) -> None:
        """A host that is running keeps the files it loaded: after a new download it starts again on the new
        ones, with other parts when the new pack runs on the other engine."""
        self.host.stop()
        self._primary_retry_at = 0.0
        if self.pack.engine() != self.engine:
            self._build()

    @property
    def native_mode(self) -> bool:
        return self._native is not None

    def download_pack(self, target: str) -> Pack | None:
        if target == "candidate" or (self.native_mode and target == "primary"):
            return self.candidate_pack
        if self.native_mode:
            return None
        return {"primary": self.pack, "legacy": self.legacy_pack, "memory": self.memory_pack}.get(target)

    def _candidate_installed(self) -> None:
        if self._native is not None:
            self._native.unload()

    def native_schema_tokens(self, tools: list[dict[str, Any]]) -> int | None:
        return self._native.schema_tokens(tools) if self._native is not None else None

    def managed_native_model(self) -> LocalModel | None:
        if self._native is None:
            return None
        try:
            return describe_pack(self.candidate_pack).model
        except PackError:
            return None

    def _candidate_conductor(self) -> NativeConductor:
        assert self._native is not None
        try:
            candidate = describe_pack(self.candidate_pack)
        except PackError as exc:
            raise LLMError("engine", str(exc)) from exc
        return NativeConductor(self._native, candidate)

    def _candidate_reason(self) -> str:
        if not self.enabled:
            return "Kiwi Mind is not available here."
        state = self.candidate_pack.check()
        if not state.ok:
            return state.reason
        try:
            describe_pack(self.candidate_pack)
        except PackError as exc:
            return str(exc)
        return "" if self.runtime() is not None else NO_RUNTIME

    def _memory_file(self) -> Path | None:
        return self.memory_pack.path(MEMORY_FILE) if self.enabled and self.memory_pack.check().ok else None

    def _chat_busy(self) -> bool:
        return self.host.in_flight() > 0 or self.legacy_host.in_flight() > 0

    # --- is it usable --------------------------------------------------------------

    def ctx(self) -> int:
        return self._ctx(self.candidate_pack if self.native_mode else self.pack)

    @staticmethod
    def _ctx(pack: Pack) -> int:
        return int((pack.manifest() or {}).get("ctx") or DEFAULT_CTX)

    def open_at(self) -> float | None:
        """From which reading of the pack's action unit on a reply is opened as a tool call. A pack names the
        mark only when its own held-back cases came out better that way; packs before 2.0 have no unit."""
        return self._open_at(self.pack)

    @staticmethod
    def _open_at(pack: Pack) -> float | None:
        mark = (pack.manifest() or {}).get("open_at")
        return float(mark) if isinstance(mark, (int, float)) and not isinstance(mark, bool) and 0.5 <= mark <= 1 else None

    def heavy_at(self) -> float | None:
        """A 3.x pack: the share of "heavy" in its route from which a request is heavy, as its held-back
        conversations set it."""
        route = (self.pack.manifest() or {}).get("route")
        mark = route.get("heavy_at") if isinstance(route, dict) else None
        return float(mark) if isinstance(mark, (int, float)) and not isinstance(mark, bool) and 0 < mark < 1 else None

    def sees(self) -> bool:
        """Whether the installed pack has vision layers, and the skill that reads what they pass on."""
        if self.native_mode:
            model = self.managed_native_model()
            return bool(model and model.projector)
        return not self._legacy_active() and self._primary_sees()

    def _primary_sees(self) -> bool:
        manifest = self.pack.manifest() or {}
        vision = manifest.get("vision")
        return self.engine == "llama" and isinstance(vision, dict) and bool(vision.get("file")) \
            and "see" in (manifest.get("skills") or [])

    def node(self) -> tuple[str | None, str]:
        if self._node:
            return self._node, ""
        with self._lock:
            memo = self._node_memo
            if memo is not None and time.monotonic() - memo[0] < NODE_RECHECK_S:
                return memo[1], memo[2]
        path, why = find_node()
        with self._lock:
            self._node_memo = (time.monotonic(), path, why)
        return path, why

    def runtime(self) -> LlamaRuntime | None:
        """The llama.cpp build a 3.x pack runs on, looked for at most every half minute."""
        with self._lock:
            memo = self._runtime_memo
            if memo is not None and time.monotonic() - memo[0] < NODE_RECHECK_S:
                return memo[1]
        try:
            found = self._find_runtime()
        except Exception:  # noqa: BLE001 - a broken install of llama.cpp reads as none
            log.warning("could not look for llama.cpp", exc_info=True)
            found = None
        with self._lock:
            self._runtime_memo = (time.monotonic(), found)
        return found

    def primary_reason(self, *, honor_backoff: bool = True) -> str:
        """Why the primary pack cannot be used right now."""
        if not self.enabled:
            return "Kiwi Mind is not available here."
        state = self.pack.check()
        if not state.ok:
            return state.reason
        if self.pack.engine() != self.engine:  # a pack put in place by hand
            with self._lock:
                self.host.stop()
                self._build()
        if honor_backoff and time.monotonic() < self._primary_retry_at:
            return "The primary Kiwi Mind runtime stopped answering."
        if self.engine == "llama":
            return "" if self.runtime() is not None else NO_RUNTIME
        path, why = self.node()
        return "" if path else why

    def _legacy_ready(self) -> bool:
        return self.enabled and self.legacy_pack.check().ok and self.node()[0] is not None

    def _legacy_active(self) -> bool:
        return bool(self.primary_reason()) and self._legacy_ready()

    def reason(self) -> str:
        """Why the selected Kiwi Mind runtime cannot be used right now."""
        if self.native_mode:
            return self._candidate_reason()
        why = self.primary_reason()
        return "" if not why or self._legacy_ready() else why

    def usable(self) -> bool:
        return not self.reason()

    def model_info(self) -> dict[str, Any]:
        """Kiwi Mind as a model record: the agent talks to it with native tool lists and plain answers."""
        if self.native_mode:
            native = self._candidate_conductor().candidate.model
            manifest = self.candidate_pack.manifest() or {}
            ctx = native.context_length or DEFAULT_CTX
            vision = bool(native.projector)
            return {
                "id": KIWIMIND, "name": native.name, "type": native.kind, "state": "loaded",
                "context": ctx, "max_context": ctx, "tools": True, "vision": vision,
                "capabilities": ["tool_use", "vision"] if vision else ["tool_use"],
                "reasoning": {"allowed_options": ["off", "on"], "default": "off"} if native.thinking else False,
                "publisher": native.publisher, "quantization": native.quantization, "params": native.params,
                "arch": native.arch, "size_bytes": native.size_bytes, "version": manifest.get("version"),
                "engine": NATIVE_FORMAT,
            }
        legacy = self._legacy_active()
        pack = self.legacy_pack if legacy else self.pack
        manifest = pack.manifest() or {}
        ctx = int(manifest.get("ctx") or DEFAULT_CTX)
        sees = not legacy and self._primary_sees()
        return {
            "id": KIWIMIND, "name": "Kiwi Mind", "type": "llm", "state": "loaded", "context": ctx,
            "max_context": ctx, "tools": True, "vision": sees,
            "capabilities": ["tool_use", "vision"] if sees else ["tool_use"],
            "publisher": "Kiwibro", "quantization": str(manifest.get("quant") or "Q4"), "reasoning": False,
            "version": manifest.get("version"), "engine": "wasm" if legacy else self.engine,
        }

    def notice(self, answering: str) -> str:
        """One line to show, once per run, when Kiwi Mind is not used and another model answers instead."""
        if self.usable():
            return ""
        with self._lock:
            if self._notified:
                return ""
            self._notified = True
        reason = self.reason()
        if "not installed" in reason:
            return f"Kiwi Mind is not installed: {answering} is answering."
        return f"{reason} {answering} is answering."

    # --- generation -----------------------------------------------------------------

    def _execute(
        self, call: Callable[[Conductor], Any], *, abort: threading.Event | None = None,
        allow_legacy: bool = True, retry: Callable[[], bool] = lambda: True,
        on_legacy: Callable[[], None] = lambda: None,
    ) -> Any:
        if abort is not None and abort.is_set():
            raise TurnAborted()
        if self._legacy_active():
            if not allow_legacy:
                # A text retry may have cooled the primary down. Give a request
                # that needs its vision features a chance to probe it directly.
                if self.primary_reason(honor_backoff=False):
                    raise LLMError("vision", "The legacy Qwen expert needs a vision helper for images.")
                return call(self.conductor)
            on_legacy()
            return call(self.legacy_conductor)
        try:
            return call(self.conductor)
        except LLMError as exc:
            if abort is not None and abort.is_set():
                raise TurnAborted() from exc
            if exc.kind not in {"connection", "stream"} or not allow_legacy or not retry() or not self._legacy_ready():
                raise
            self._primary_retry_at = time.monotonic() + PRIMARY_RETRY_S
            log.info("The primary Kiwi Mind runtime failed; using the installed legacy pack: %s", exc.message)
            on_legacy()
            return call(self.legacy_conductor)

    def stream(self, **kwargs: Any) -> ChatResult:
        if self.native_mode:
            kwargs.pop("conversation", None)
            return self._candidate_conductor().stream(**kwargs)
        callback, events = kwargs.get("on_delta"), kwargs.get("events")
        has_picture = has_images(kwargs["messages"])
        showed_text = False

        def delta(text: str) -> None:
            nonlocal showed_text
            showed_text = showed_text or bool(text)
            if callback is not None:
                callback(text)

        def on_legacy() -> None:
            if events is not None:
                events("status", {"phase": "thinking", "label": "Kiwi Mind is using the legacy Qwen expert."})

        # Once the user has seen text, the caller owns recovery for the partial stream.
        return self._execute(lambda conductor: conductor.stream(**{**kwargs, "on_delta": delta}),
                             abort=kwargs.get("abort"), on_legacy=on_legacy,
                             allow_legacy=not has_picture or bool((self.helper() or {}).get("vision")),
                             retry=lambda: not showed_text)

    def complete(
        self, messages: list[dict[str, Any]], *, max_tokens: int, temperature: float,
        abort: threading.Event | None = None, response_format: dict[str, Any] | None = None,
    ) -> Completion:
        if self.native_mode:
            return self._candidate_conductor().complete(
                messages, max_tokens=max_tokens, temperature=temperature, abort=abort, response_format=response_format,
            )
        return self._execute(lambda conductor: conductor.complete(
            messages, max_tokens=max_tokens, temperature=temperature, abort=abort,
        ), abort=abort, allow_legacy=not has_images(messages))

    def warm(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, slot: str | None, **options: Any,
    ) -> None:
        if self.native_mode:
            self._candidate_conductor().warm(messages, tools, **options)
            return
        self._execute(lambda conductor: conductor.warm(messages, tools, slot), retry=lambda: not has_images(messages))

    # --- state for the API -------------------------------------------------------------

    def _native_health(self) -> dict[str, Any]:
        assert self._native is not None
        why = self._candidate_reason()
        server = None if why else self._native.direct_engine().loaded(NATIVE_MODEL)
        return {"usable": not why, "running": server is not None, "pid": server.process.pid if server else None,
                "reason": why, "engine": NATIVE_FORMAT}

    def describe(self) -> dict[str, Any]:
        if self.native_mode:
            status = self._native_health()
            pack = self.candidate_pack.describe()
            details: dict[str, Any] = {"vision": False}
            try:
                candidate = describe_pack(self.candidate_pack)
            except PackError:
                pass
            else:
                model = candidate.model
                details = {
                    "base": model.repo, "architecture": model.arch, "quant": model.quantization,
                    "vision": bool(model.projector), "expert_bank": candidate.expert_bank,
                    "mtp": bool(model.pack_mtp),
                    "training_steps": candidate.training_steps, "feed_forward": candidate.feed_forward,
                    "capabilities": {"tools": model.tools, "thinking": model.thinking,
                                     "reasoning_effort": candidate.reasoning_effort},
                }
                if model.pack_adapters:
                    adapter = model.pack_adapters[0]
                    details["task_adapter"] = {"tasks": list(adapter.tasks), "scale": adapter.scale}
            return {**pack, **status, **details, "active": "candidate" if status["usable"] else None,
                    "node": None, "rss_bytes": _rss(status["pid"]) if status["pid"] else None,
                    "helper": None, "download": self.candidate_pack.download_state()}
        why = self.primary_reason()  # builds the parts for a pack that was put in place by hand
        legacy = self.legacy_pack.describe()
        using_legacy = bool(why) and legacy["ok"] and self.node()[0] is not None
        reason = "" if using_legacy else why
        pack = legacy if using_legacy else self.pack.describe()
        host = (self.legacy_host if using_legacy else self.host).describe()
        engine = "wasm" if using_legacy else self.engine
        path = self.node()[0] if self.enabled and engine == "wasm" else None
        helper = self.helper()
        return {
            "usable": not reason,
            "installed": pack["installed"],
            "version": pack["version"],
            "bytes": pack["bytes"],
            "engine": engine,
            "active": "legacy" if using_legacy else (None if reason else "primary"),
            "vision": not using_legacy and self._primary_sees(),
            "node": path,
            "running": host["running"],
            "pid": host["pid"],
            "rss_bytes": host["rss_bytes"],
            "skills": pack["skills"],
            "helper": {"id": helper["id"], "name": helper.get("name") or helper["id"]} if helper else None,
            "little": self.little.describe(),
            "memory_expert": {**self.memory_pack.describe(), **self.memory_expert.describe(),
                              "download": self.memory_pack.download_state()},
            "legacy": {**legacy, **self.legacy_host.describe(), "download": self.legacy_pack.download_state()},
            "primary": {"installed": self.pack.manifest() is not None, "reason": why},
            "reason": reason,
            "download": self.pack.download_state(),
        }

    def health(self) -> dict[str, Any]:
        if self.native_mode:
            return self._native_health()
        why = self.primary_reason()
        using_legacy = bool(why) and self._legacy_ready()
        host = (self.legacy_host if using_legacy else self.host).describe()
        result = {"usable": using_legacy or not why, "running": host["running"],
                  "pid": host["pid"], "little": self.little.health()}
        if self.legacy_pack.manifest() is not None:
            result["legacy"] = {"available": self._legacy_ready(), "active": using_legacy,
                                "running": self.legacy_host.alive()}
        if self.memory_pack.manifest() is not None:
            result["memory_expert"] = self.memory_expert.health()
        return result

    # --- lifecycle -------------------------------------------------------------------

    def start_background(self) -> None:
        """Start the host without making anyone wait, then read the shared start of a conversation into it."""
        why = self._candidate_reason() if self.native_mode else self.primary_reason()
        if why:
            return
        threading.Thread(target=self._start, name="kiwibro-mind-start", daemon=True).start()

    def _start(self) -> None:
        try:
            if self._native is not None:
                self._native.direct_engine().ensure(NATIVE_MODEL)
            else:
                self.host.ensure()
        except Exception as exc:  # noqa: BLE001 - starting early is only an optimisation
            log.info("Kiwi Mind did not start in the background: %s", exc)

    def stop(self) -> None:
        self.pack.shutdown()
        self.candidate_pack.shutdown()
        self._candidate_installed()
        self.legacy_pack.shutdown()
        self.memory_pack.shutdown()
        self.host.stop()
        self.legacy_host.stop()
        self.little.stop()
        self.memory_expert.stop()

    def stop_idle(self, min_idle_s: float) -> int:
        return sum(host.stop_idle(min_idle_s)
                   for host in (self.host, self.legacy_host, self.little, self.memory_expert))
