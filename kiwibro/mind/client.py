"""HTTP client for the Kiwi Mind host (see kiwibro/mind/host/host.mjs for the endpoints)."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Iterator

import requests

from .host import HostError, MindHost, _Running

log = logging.getLogger(__name__)

CONNECT_TIMEOUT = 5
READ_TIMEOUT = 600


class MindRequestError(Exception):
    """The host refused a request. ``context`` is True when the prompt does not fit the window."""

    def __init__(self, message: str, status: int = 0, *, context: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.context = context


class HostGone(Exception):
    """The host could not be reached (it died or stopped answering)."""


class MindClient:
    def __init__(self, host: MindHost) -> None:
        self.host = host

    # --- one request, with one restart if the host died -------------------------

    def _call(self, path: str, body: dict[str, Any] | None, *, stream: bool = False,
              abort: threading.Event | None = None, method: str = "POST") -> requests.Response:
        try:
            running = self.host.ensure()
        except HostError as err:
            raise HostGone(str(err)) from err
        for attempt in (0, 1):
            try:
                return self._send(running, method, path, body, stream, abort)
            except requests.ConnectionError as exc:
                if abort is not None and abort.is_set():
                    raise
                if attempt:
                    raise HostGone("Kiwi Mind stopped answering. Try again.") from exc
                log.warning("Kiwi Mind host did not answer; restarting it once")
                try:
                    running = self.host.restart_after_failure(running)
                except HostError as err:
                    raise HostGone(str(err)) from err
        raise HostGone("Kiwi Mind stopped answering. Try again.")  # pragma: no cover

    def _send(self, running: _Running, method: str, path: str, body: dict[str, Any] | None, stream: bool,
              abort: threading.Event | None) -> requests.Response:
        response = requests.request(
            method, f"http://127.0.0.1:{running.port}{path}", json=body, stream=stream,
            headers={"Authorization": f"Bearer {running.key}"}, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        if abort is not None and stream:
            finished = threading.Event()
            response.kiwibro_finished = finished  # type: ignore[attr-defined]
            threading.Thread(target=_close_on_abort, args=(response, abort, finished),
                             name="kiwibro-mind-watch", daemon=True).start()
        if response.status_code != 200:
            try:
                message = str(response.json().get("error") or response.text[:300])
            except ValueError:
                message = response.text[:300]
            response.close()
            raise MindRequestError(message, response.status_code, context="context holds" in message)
        return response

    def _json(self, path: str, body: dict[str, Any] | None = None, *, method: str = "POST") -> dict[str, Any]:
        with self.host.busy():
            response = self._call(path, body, method=method)
            try:
                return response.json()
            finally:
                response.close()

    # --- endpoints ---------------------------------------------------------------

    def tokenize(self, text: str) -> int:
        return int(self._json("/tokenize", {"text": text})["count"])

    def control(self, prompt: str, slot: str) -> dict[str, Any]:
        return self._json("/control", {"prompt": prompt, "slot": slot})

    def warm(self, prompt: str, slot: str, skill: str = "none") -> dict[str, Any]:
        return self._json("/warm", {"prompt": prompt, "slot": slot, "skill": skill})

    def health(self) -> dict[str, Any]:
        return self._json("/health", None, method="GET")

    def generate(self, body: dict[str, Any], abort: threading.Event | None = None) -> Iterator[dict[str, Any]]:
        """The events of a streamed /generate: {"text": ...} pieces, then one {"done": true, ...}."""
        with self.host.busy():
            response = self._call("/generate", {**body, "stream": True}, stream=True, abort=abort)
            finished = getattr(response, "kiwibro_finished", None)
            try:
                for raw in response.iter_lines(decode_unicode=False):
                    if not raw.startswith(b"data:"):
                        continue
                    data = raw[5:].strip()
                    if data == b"[DONE]":
                        return
                    try:
                        event = json.loads(data.decode("utf-8"))
                    except ValueError:
                        continue
                    yield event
            except requests.RequestException as exc:
                if abort is not None and abort.is_set():
                    return
                raise HostGone("The connection to Kiwi Mind was interrupted.") from exc
            finally:
                if finished is not None:
                    finished.set()
                response.close()


def _close_on_abort(response: requests.Response, abort: threading.Event, finished: threading.Event) -> None:
    """Drop the connection when the caller aborts: the host stops generating when its client disconnects."""
    from ..llm import _socket_of

    while not finished.wait(0.1):
        if abort.is_set():
            sock = _socket_of(response)
            if sock is not None:
                try:
                    sock.shutdown(2)  # SHUT_RDWR
                except OSError:
                    pass
            return
