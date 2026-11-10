"""Stops the local model once no Kiwibro window is open.

An open window polls /api/version every 10 s and says it is closing on pagehide. A window
counts as gone when it said so and did not poll again within CLOSE_GRACE_S (a reload polls
at once), or when it has been silent for STALE_S (a crash). Idle model servers are then
stopped; the model starts again on the next request.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)

CLOSE_GRACE_S = 15.0  # longer than the 10 s poll, so a second open window always speaks up first
STALE_S = 180.0
IDLE_S = 20.0
CHECK_EVERY_S = 3.0


class Presence:
    def __init__(
        self,
        *,
        busy: Callable[[], bool],
        stop_idle: Callable[[float], int],
        clock: Callable[[], float] = time.monotonic,
        close_grace_s: float = CLOSE_GRACE_S,
        stale_s: float = STALE_S,
        idle_s: float = IDLE_S,
    ) -> None:
        self._busy = busy
        self._stop_idle = stop_idle
        self._clock = clock
        self._close_grace_s = close_grace_s
        self._stale_s = stale_s
        self._idle_s = idle_s
        self._lock = threading.Lock()
        self._seen_at: float | None = None
        self._closing_at: float | None = None

    def seen(self) -> None:
        """A window polled: it is open, and any earlier close was a reload."""
        with self._lock:
            self._seen_at = self._clock()
            self._closing_at = None

    def closing(self) -> None:
        """A window is closing (or reloading): it was open, even if it never polled."""
        with self._lock:
            self._closing_at = self._clock()
            if self._seen_at is None:
                self._seen_at = self._closing_at

    def gone(self) -> bool:
        """A window was seen since start and none is open now."""
        with self._lock:
            if self._seen_at is None:
                return False
            now = self._clock()
            closed = self._closing_at is not None and now - self._closing_at >= self._close_grace_s
            return closed or now - self._seen_at >= self._stale_s

    def check(self) -> int:
        """Stop idle model servers when no window is open and nothing is running. Servers stopped."""
        if not self.gone() or self._busy():
            return 0
        stopped = self._stop_idle(self._idle_s)
        if stopped:
            log.info("no Kiwibro window open: stopped the model (frees its memory)")
        return stopped

    def start(self) -> None:
        threading.Thread(target=self._run, name="kiwibro-presence", daemon=True).start()

    def _run(self) -> None:
        while True:
            time.sleep(CHECK_EVERY_S)
            try:
                self.check()
            except Exception:  # a failed check must not end the loop
                log.warning("presence check failed", exc_info=True)
