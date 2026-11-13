"""Whether this machine can reach the internet right now.

A plain TCP connection to a few public addresses answers the question without DNS or HTTP,
so it still works when name lookup itself is what broke, and it costs nothing when it
succeeds. The answer is remembered for a short while: a request that fails at the network
level says so (``suspect``) and the next look happens at once. Standard library only, so
every web tool can ask before it starts a request that would otherwise wait out its timeout.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import Any

OFFLINE_ERROR = (
    "no internet connection right now. Web tools cannot reach anything until it is back; "
    "local files, notes, saved memory and indexed documents still work."
)
OFFLINE_LINE = (
    "Internet: not reachable right now. Web tools will fail until it is back: answer from what you know "
    "and from local files, notes, memory and documents, and say what may be outdated."
)

PROBE_ADDRESSES = (("1.1.1.1", 443), ("8.8.8.8", 53), ("9.9.9.9", 443))
PROBE_TIMEOUT_S = 1.2
ONLINE_TTL_S = 45.0  # a working connection is not asked again for this long
OFFLINE_TTL_S = 8.0  # a missing one is checked again sooner: it may come back any moment
ENV_SWITCH = "KIWIBRO_OFFLINE"  # "1" says offline, "0" says online, neither probes

_lock = threading.Lock()
_probe_lock = threading.Lock()  # one probe at a time: callers arriving together share its answer
_checked = 0.0  # time.monotonic() of the last probe; 0 before the first one
_checked_unix = 0.0  # the same moment as a unix time, for the API
_online = False  # what that probe found


def _forced() -> bool | None:
    """The environment's answer, when it gives one, instead of probing."""
    value = os.environ.get(ENV_SWITCH, "").strip()
    if value == "1":
        return False
    if value == "0":
        return True
    return None


def _probe() -> bool:
    """Try every address at once; online as soon as one accepts the connection."""
    stop = threading.Event()
    left = [len(PROBE_ADDRESSES)]
    lock = threading.Lock()
    reached: list[bool] = []

    def attempt(address: tuple[str, int]) -> None:
        try:
            connection = socket.create_connection(address, timeout=PROBE_TIMEOUT_S)
        except OSError:
            pass
        else:
            connection.close()
            reached.append(True)
            stop.set()
        finally:
            with lock:
                left[0] -= 1
                if not left[0]:
                    stop.set()

    for address in PROBE_ADDRESSES:
        name = f"kiwibro-probe-{address[0]}"
        threading.Thread(target=attempt, args=(address,), name=name, daemon=True).start()
    stop.wait(PROBE_TIMEOUT_S + 0.1)
    return bool(reached)


def _remember(result: bool) -> None:
    global _checked, _checked_unix, _online
    with _lock:
        _checked = time.monotonic()
        _checked_unix = time.time()
        _online = result


def _fresh() -> bool:
    with _lock:
        if not _checked:
            return False
        ttl = ONLINE_TTL_S if _online else OFFLINE_TTL_S
        return time.monotonic() - _checked < ttl


def online() -> bool:
    """Whether the internet is reachable; a remembered answer is reused until it ages out."""
    forced = _forced()
    if forced is not None:
        return forced
    if _fresh():
        with _lock:
            return _online
    with _probe_lock:
        if _fresh():  # another caller probed while this one waited
            with _lock:
                return _online
        result = _probe()
        _remember(result)
        return result


def suspect() -> None:
    """A real request failed at the network level: the next look happens at once."""
    global _checked
    with _lock:
        _checked = 0.0


def state() -> dict[str, Any]:
    """The answer and when it was taken (unix time), for the API.

    A fresh answer is returned without a probe, so polling costs nothing. When the
    environment gives the answer, it counts as taken now.
    """
    forced = _forced()
    if forced is not None:
        return {"online": forced, "checked": int(time.time())}
    result = online()
    with _lock:
        return {"online": result, "checked": int(_checked_unix)}


def _set_state(online_now: bool, age: float = 0.0) -> None:
    """For tests: as if a probe ran ``age`` seconds ago and found ``online_now``."""
    global _checked, _checked_unix, _online
    with _lock:
        _checked = time.monotonic() - age
        _checked_unix = time.time() - age
        _online = online_now


def _reset() -> None:
    """For tests: forget the remembered answer, so the next look probes."""
    global _checked, _checked_unix, _online
    with _lock:
        _checked = 0.0
        _checked_unix = 0.0
        _online = False
