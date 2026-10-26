"""Model readiness: the selected model is loaded with a working window and the
prompt prefix of the next request is prefilled before the user sends anything.

Measured on this laptop (Ryzen 5 7530U, CPU only, LM Studio 0.4.26): Kiwibro's
stable prefix (identity + the auto tool set) is ~4.7k tokens. With
``liquid/lfm2.5-1.2b`` the first request took 47 s cold and 6 s once that prefix
was in LM Studio's KV cache. So opening Kiwibro, picking a model or opening a
chat starts one background job:

1. **Load** the model explicitly when it is not loaded (or is loaded with a
   window too small to work in). LM Studio's just-in-time load uses the model's
   own defaults — sometimes a single parallel slot, where any background call
   (memory, titles) evicts the chat's cache.
2. **Warm**: send the exact leading messages of the next request (system
   prompt + tools, plus the chat's history when a chat is open) with
   ``max_tokens=1``, so the real request only prefills the new message.

A turn that starts while a job prefills the same prefix waits for it instead
of prefilling the same tokens again in another slot; a job for some other chat
is cancelled so it never competes with a live reply.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from . import inference, routing
from .llm import LMStudioError, TurnAborted

log = logging.getLogger(__name__)

TARGET_CONTEXT = 32768  # window Kiwibro loads models with (capped by the model's maximum)
MIN_CONTEXT = 16384  # a loaded model below this is reloaded larger while nothing is running
LOAD_LADDER = (32768, 16384, 8192)  # tried in order when LM Studio's memory guardrails refuse
PARALLEL_SLOTS = 4  # one slot for the chat, others for background calls (shared KV pool)
WARM_HISTORY_TOKENS = 24000  # longer chats are not prefilled ahead of time
_WARM_MEMO = 32

PHASES = ("idle", "loading", "warming", "ready", "error")

PrefixFn = Callable[[str, "str | None"], "tuple[list[dict[str, Any]], list[dict[str, Any]] | None, dict[str, Any]] | None"]


def signature(model: str, messages: list[dict[str, Any]], tools: Any, extra: Any, context: Any = None) -> str:
    """Identity of one prefix on one loaded instance (a reload empties the cache)."""
    encoded = json.dumps([model, context, messages, tools, extra], sort_keys=True, ensure_ascii=False)
    return hashlib.blake2b(encoded.encode("utf-8"), digest_size=16).hexdigest()


def load_sizes(max_context: int) -> list[int]:
    """Windows to try for an explicit load, largest first, never above the model's maximum."""
    if max_context <= 0:
        return list(LOAD_LADDER)
    sizes = [min(size, max_context) for size in LOAD_LADDER]
    return list(dict.fromkeys(sizes))


@dataclass
class _Job:
    model: str | None
    conversation_id: str | None
    abort: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)


class Readiness:
    def __init__(
        self,
        llm: Any,
        prefix_fn: PrefixFn,
        *,
        busy_fn: Callable[[], int],
        on_job: Callable[[], None] | None = None,
        enabled: bool = True,
    ) -> None:
        self.llm = llm
        self._prefix = prefix_fn
        self._busy = busy_fn
        self._on_job = on_job  # e.g. pause background memory extraction: the user is waiting on this
        self.enabled = enabled
        self._lock = threading.Lock()
        self._pending: list[_Job] = []
        self._current: _Job | None = None
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._warm: OrderedDict[str, float] = OrderedDict()
        self._loaded: set[str] = set()  # models this worker loaded (and so may unload again)
        self._status: dict[str, Any] = {"model": None, "phase": "idle", "detail": "", "context_length": None}

    # --- public -------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def request(self, model: str | None = None, conversation_id: str | None = None) -> dict[str, Any]:
        """Queue a load + warm job (deduplicated). ``model=None`` means the selected model."""
        if not self.enabled:
            return self.status()
        with self._lock:
            wanted = (model, conversation_id)
            if self._current and (self._current.model, self._current.conversation_id) == wanted:
                return dict(self._status)
            if not any((job.model, job.conversation_id) == wanted for job in self._pending):
                # Newest request first; the base prefix (no chat) is shared by every chat.
                self._pending = [job for job in self._pending if job.conversation_id is None][-1:]
                self._pending.append(_Job(model, conversation_id))
            self._start_locked()
        self._wake.set()
        return self.status()

    def release_all(self) -> None:
        """Unload every model this worker loaded (the engine is switching: free their RAM)."""
        self._release_others("")
        with self._lock:
            self._warm.clear()

    def invalidate(self) -> None:
        """Settings that shape the prefix changed: forget what was warmed."""
        with self._lock:
            self._warm.clear()

    def before_turn(
        self,
        model: str,
        conversation_id: str,
        *,
        abort: threading.Event | None = None,
        on_wait: Callable[[], None] | None = None,
    ) -> None:
        """Called by a turn before its first request: share the job's work or stop it."""
        with self._lock:
            self._pending.clear()
            job = self._current
        if job is None:
            return
        same_model = job.model in (None, model)
        if not same_model or job.conversation_id not in (None, conversation_id):
            job.abort.set()  # another chat or model: never compete with a live reply
            return
        if not job.done.is_set() and on_wait is not None:
            on_wait()
        while not job.done.wait(0.25):
            if abort is not None and abort.is_set():
                return

    # --- worker -------------------------------------------------------------

    def _start_locked(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name="kiwibro-readiness", daemon=True)
            self._thread.start()

    def _set(self, **fields: Any) -> None:
        with self._lock:
            self._status.update(fields)

    def _loop(self) -> None:
        while True:
            self._wake.wait(30.0)
            self._wake.clear()
            while True:
                with self._lock:
                    if not self._pending:
                        self._current = None
                        break
                    job = self._pending.pop(0)
                    self._current = job
                try:
                    if not self._busy():  # a live reply prefills for itself
                        if self._on_job is not None:
                            self._on_job()
                        self._run(job)
                except Exception:  # readiness is an optimisation: never crash the worker
                    log.exception("model readiness job failed")
                    self._set(phase="error", detail="Could not prepare the model.")
                finally:
                    job.done.set()

    def _run(self, job: _Job) -> None:
        try:
            model = job.model or self.llm.resolve_model()
        except LMStudioError as exc:
            self._set(model=job.model, phase="error", detail=exc.message, context_length=None)
            return
        job.model = model
        info = self._info(model, force=True)
        if info is None:
            self._set(model=model, phase="error", detail=f"{model} is not available in LM Studio.")
            return
        info = self._ensure_loaded(model, info, job)
        if job.abort.is_set():
            return
        context = (info.get("loaded_config") or {}).get("context_length") or info.get("context")
        prefix = self._prefix(model, job.conversation_id)
        if prefix is None:
            self._set(model=model, phase="ready", detail="", context_length=context)
            return
        messages, tools, extra = prefix
        key = signature(model, messages, tools, extra, context if info.get("state") == "loaded" else None)
        with self._lock:
            warmed = key in self._warm
        if warmed and info.get("state") == "loaded":
            self._set(model=model, phase="ready", detail="", context_length=context)
            return
        self._set(model=model, phase="warming", detail="Reading the conversation" if job.conversation_id
                  else "Getting ready", context_length=context)
        reuse = inference.PREFIX_TRACKER.observe(model, messages, tools, extra)
        started = time.monotonic()
        try:
            self.llm.prefill(model=model, messages=messages, tools=tools, extra=extra or None, abort=job.abort)
        except TurnAborted:
            self._set(phase="ready", detail="")
            return
        except LMStudioError as exc:
            log.info("warm-up for %s failed: %s", model, exc.message)
            self._set(phase="error", detail=exc.message)
            return
        elapsed_ms = int((time.monotonic() - started) * 1000)
        inference.SPEEDS.note(
            model,
            new_tokens=int(reuse.get("prompt_tokens_est", 0)) - int(reuse.get("reused_tokens_est", 0)),
            ttft_ms=elapsed_ms, tokens_per_s=None,
        )
        info = self._info(model, force=True) or info
        context = (info.get("loaded_config") or {}).get("context_length") or info.get("context")
        with self._lock:
            self._warm[signature(model, messages, tools, extra, context)] = time.time()
            while len(self._warm) > _WARM_MEMO:
                self._warm.popitem(last=False)
        log.info("warmed %s (%s, ~%s new tokens) in %.1fs", model,
                 "chat " + job.conversation_id if job.conversation_id else "base prefix",
                 reuse.get("prompt_tokens_est"), elapsed_ms / 1000)
        self._set(phase="ready", detail="", context_length=context)

    def _release_others(self, key: str) -> None:
        """Unload the models this worker loaded for an earlier selection: each holds
        gigabytes of RAM, and a CPU laptop with two 4B models resident swaps. Models the
        user loaded in LM Studio, and helpers leased by routing, are never touched; jobs
        only run while no reply streams, so none of these is in use."""
        with self._lock:
            stale, self._loaded = self._loaded - {key}, self._loaded & {key}
        for model in stale:
            if self.llm.unload_model(model):
                log.info("unloaded %s: %s is the selected model now", model, key)

    def _info(self, model: str, *, force: bool = False) -> dict[str, Any] | None:
        try:
            models = self.llm.list_models(force=force)
        except LMStudioError:
            return None
        return next((m for m in models if m.get("id") == model or m.get("key") == model), None)

    def _ensure_loaded(self, model: str, info: dict[str, Any], job: _Job) -> dict[str, Any]:
        """Load (or enlarge) the model's window; on any refusal LM Studio's JIT load remains."""
        max_context = int(info.get("max_context") or 0)
        if info.get("state") == "loaded":
            current = int((info.get("loaded_config") or {}).get("context_length") or info.get("context") or 0)
            target = min(TARGET_CONTEXT, max_context)
            if current >= MIN_CONTEXT or target <= current or self._busy():
                return info
            self._set(model=model, phase="loading", detail=f"Enlarging the window to {target // 1024}K")
            try:
                routing.reload_with_context(self.llm, info, target)
            except LMStudioError as exc:
                log.info("could not enlarge %s: %s", model, exc.message)
            return self._info(model, force=True) or info
        self._set(model=model, phase="loading", detail="Loading the model", context_length=None)
        key = str(info.get("key") or model)
        self._release_others(key)  # first: both resident at once can make LM Studio refuse the load
        for size in load_sizes(max_context):
            if job.abort.is_set():
                break
            if self.llm.load_model(key, size, parallel=PARALLEL_SLOTS):
                with self._lock:
                    self._loaded.add(key)
                break
            direct = callable(getattr(self.llm, "direct_on", None)) and self.llm.direct_on()
            if not direct and getattr(self.llm, "_load_api_supported", None) is False:
                break  # this LM Studio has no load API: the warm-up request JIT-loads it
        return self._info(model, force=True) or info
