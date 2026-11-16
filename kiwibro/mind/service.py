"""Kiwi Mind as the app sees it: is it usable, what is its state, and the one place that owns its parts."""

from __future__ import annotations

import atexit
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..llm import ChatResult
from . import KIWIMIND
from .client import MindClient
from .conductor import Conductor, Settings
from .conductor3 import Conductor3
from .host import MindHost, find_node
from .llama import NO_RUNTIME, LlamaClient, LlamaHost, LlamaRuntime
from .pack import Pack

log = logging.getLogger(__name__)

NODE_RECHECK_S = 30.0
DEFAULT_CTX = 8192


class Mind:
    """Pack + host process + conductor. One per app; nothing starts until it is used.

    A 3.x pack runs on llama.cpp and a 2.x pack on Kiwibro's own engine under Node.js; the parts are built
    for the pack that is installed and built again when a pack of the other kind arrives."""

    def __init__(
        self, data_dir: str | Path | None, *, helper: Callable[[], dict[str, Any] | None],
        helper_stream: Callable[..., ChatResult], base_url: str | None = None, node: str | None = None,
        runtime: Callable[[], LlamaRuntime | None] | None = None,
    ) -> None:
        self.enabled = data_dir is not None
        self.pack = Pack(data_dir if data_dir is not None else Path("/nonexistent-kiwibro"), base_url=base_url)
        self.helper = helper
        self._helper_stream = helper_stream
        self._node = node
        self._find_runtime = runtime or (lambda: None)
        self._node_memo: tuple[float, str | None, str] | None = None
        self._runtime_memo: tuple[float, LlamaRuntime | None] | None = None
        self._notified = False
        self._lock = threading.Lock()
        self._build()
        self.pack.on_installed = self._installed
        atexit.register(self.stop)

    def _build(self) -> None:
        self.engine = self.pack.engine()
        settings = Settings(self.ctx, self.helper, self._helper_stream, self.open_at, self.heavy_at, self.sees)
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
        if self.pack.engine() != self.engine:
            self._build()

    # --- is it usable --------------------------------------------------------------

    def ctx(self) -> int:
        return int((self.pack.manifest() or {}).get("ctx") or DEFAULT_CTX)

    def open_at(self) -> float | None:
        """From which reading of the pack's action unit on a reply is opened as a tool call. A pack names the
        mark only when its own held-back cases came out better that way; packs before 2.0 have no unit."""
        mark = (self.pack.manifest() or {}).get("open_at")
        return float(mark) if isinstance(mark, (int, float)) and not isinstance(mark, bool) and 0.5 <= mark <= 1 else None

    def heavy_at(self) -> float | None:
        """A 3.x pack: the share of "heavy" in its route from which a request is heavy, as its held-back
        conversations set it."""
        route = (self.pack.manifest() or {}).get("route")
        mark = route.get("heavy_at") if isinstance(route, dict) else None
        return float(mark) if isinstance(mark, (int, float)) and not isinstance(mark, bool) and 0 < mark < 1 else None

    def sees(self) -> bool:
        """Whether the installed pack has vision layers, and the skill that reads what they pass on."""
        manifest = self.pack.manifest() or {}
        vision = manifest.get("vision")
        return self.engine == "llama" and isinstance(vision, dict) and bool(vision.get("file")) \
            and "see" in (manifest.get("skills") or [])

    def node(self) -> tuple[str | None, str]:
        if self.host._node:
            return self.host._node, ""
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

    def reason(self) -> str:
        """Why Kiwi Mind cannot be used right now ("" when it can)."""
        if not self.enabled:
            return "Kiwi Mind is not available here."
        state = self.pack.check()
        if not state.ok:
            return state.reason
        if self.pack.engine() != self.engine:  # a pack put in place by hand
            with self._lock:
                self.host.stop()
                self._build()
        if self.engine == "llama":
            return "" if self.runtime() is not None else NO_RUNTIME
        path, why = self.node()
        return "" if path else why

    def usable(self) -> bool:
        return not self.reason()

    def model_info(self) -> dict[str, Any]:
        """Kiwi Mind as a model record: the agent talks to it with native tool lists and plain answers."""
        manifest = self.pack.manifest() or {}
        ctx = self.ctx()
        sees = self.sees()
        return {
            "id": KIWIMIND, "name": "Kiwi Mind", "type": "llm", "state": "loaded", "context": ctx,
            "max_context": ctx, "tools": True, "vision": sees,
            "capabilities": ["tool_use", "vision"] if sees else ["tool_use"],
            "publisher": "Kiwibro", "quantization": str(manifest.get("quant") or "Q4"), "reasoning": False,
            "version": manifest.get("version"),
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

    # --- state for the API -------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        pack = self.pack.describe()
        reason = self.reason()  # first: it builds the parts for a pack that was put in place by hand
        host = self.host.describe()
        path = self.node()[0] if self.enabled and self.engine != "llama" else None
        helper = self.helper()
        return {
            "usable": not reason,
            "installed": pack["installed"],
            "version": pack["version"],
            "bytes": pack["bytes"],
            "engine": self.engine,
            "vision": self.sees(),
            "node": path,
            "running": host["running"],
            "pid": host["pid"],
            "rss_bytes": host["rss_bytes"],
            "skills": pack["skills"],
            "helper": {"id": helper["id"], "name": helper.get("name") or helper["id"]} if helper else None,
            "reason": reason,
            "download": self.pack.download_state(),
        }

    def health(self) -> dict[str, Any]:
        usable = self.usable()
        host = self.host.describe()
        return {"usable": usable, "running": host["running"], "pid": host["pid"]}

    # --- lifecycle -------------------------------------------------------------------

    def start_background(self) -> None:
        """Start the host without making anyone wait, then read the shared start of a conversation into it."""
        if not self.usable():
            return
        threading.Thread(target=self._start, name="kiwibro-mind-start", daemon=True).start()

    def _start(self) -> None:
        try:
            self.host.ensure()
        except Exception as exc:  # noqa: BLE001 - starting early is only an optimisation
            log.info("Kiwi Mind did not start in the background: %s", exc)

    def stop(self) -> None:
        self.pack.shutdown()
        self.host.stop()

    def stop_idle(self, min_idle_s: float) -> int:
        return self.host.stop_idle(min_idle_s)
