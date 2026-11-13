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
from .host import MindHost, find_node
from .pack import Pack

log = logging.getLogger(__name__)

NODE_RECHECK_S = 30.0
DEFAULT_CTX = 8192


class Mind:
    """Pack + host process + conductor. One per app; nothing starts until it is used."""

    def __init__(
        self, data_dir: str | Path | None, *, helper: Callable[[], dict[str, Any] | None],
        helper_stream: Callable[..., ChatResult], base_url: str | None = None, node: str | None = None,
    ) -> None:
        self.enabled = data_dir is not None
        self.pack = Pack(data_dir if data_dir is not None else Path("/nonexistent-kiwibro"), base_url=base_url)
        self.host = MindHost(self.pack, node=node)
        # A host that is running keeps the files it loaded: after a new download it starts again on the new ones.
        self.pack.on_installed = self.host.stop
        self.client = MindClient(self.host)
        self.helper = helper
        self.conductor = Conductor(self.client, Settings(self.ctx, helper, helper_stream))
        self._node_memo: tuple[float, str | None, str] | None = None
        self._notified = False
        self._lock = threading.Lock()
        atexit.register(self.stop)

    # --- is it usable --------------------------------------------------------------

    def ctx(self) -> int:
        return int((self.pack.manifest() or {}).get("ctx") or DEFAULT_CTX)

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

    def reason(self) -> str:
        """Why Kiwi Mind cannot be used right now ("" when it can)."""
        if not self.enabled:
            return "Kiwi Mind is not available here."
        state = self.pack.check()
        if not state.ok:
            return state.reason
        path, why = self.node()
        return "" if path else why

    def usable(self) -> bool:
        return not self.reason()

    def model_info(self) -> dict[str, Any]:
        """Kiwi Mind as a model record: the agent talks to it with native tool lists and plain answers."""
        manifest = self.pack.manifest() or {}
        ctx = self.ctx()
        return {
            "id": KIWIMIND, "name": "Kiwi Mind", "type": "llm", "state": "loaded", "context": ctx,
            "max_context": ctx, "tools": True, "vision": False, "capabilities": ["tool_use"],
            "publisher": "Kiwibro", "quantization": "Q4", "reasoning": False,
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
        host = self.host.describe()
        path, _why = self.node() if self.enabled else (None, "")
        helper = self.helper()
        reason = self.reason()
        return {
            "usable": not reason,
            "installed": pack["installed"],
            "version": pack["version"],
            "bytes": pack["bytes"],
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
        host = self.host.describe()
        return {"usable": self.usable(), "running": host["running"], "pid": host["pid"]}

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
