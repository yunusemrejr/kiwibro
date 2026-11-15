"""Tests for {{name}}.client; a local HTTP server plays the part of the API."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from {{name}}.client import ApiClient, ApiError


class Handler(BaseHTTPRequestHandler):
    """Answers with the next scripted reply, so a test decides status, type and body."""

    script: list[tuple[int, str, bytes]] = []
    calls: list[str] = []
    received: bytes = b""

    def do_GET(self) -> None:
        self._answer()

    def do_POST(self) -> None:
        Handler.received = self.rfile.read(int(self.headers.get("content-length", "0")))
        self._answer()

    def _answer(self) -> None:
        Handler.calls.append(self.path)
        status, kind, body = (
            Handler.script.pop(0) if Handler.script else (404, "application/json", b'{"error": "no route"}')
        )
        self.send_response(status)
        self.send_header("content-type", kind)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, pattern: str, *args: object) -> None:
        pass


@pytest.fixture()
def api():
    Handler.script = []
    Handler.calls = []
    Handler.received = b""
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def client_of(server, **options) -> ApiClient:
    return ApiClient(f"http://127.0.0.1:{server.server_address[1]}", **options)


def reply(body: bytes, status: int = 200, kind: str = "application/json") -> tuple[int, str, bytes]:
    return status, kind, body


def test_get_returns_the_decoded_json(api) -> None:
    Handler.script = [reply(b'{"id": 3, "title": "Task"}')]

    assert client_of(api).get("/tasks/3") == {"id": 3, "title": "Task"}


def test_post_sends_json_and_returns_the_answer(api) -> None:
    Handler.script = [reply(b'{"id": 4}', status=201)]

    assert client_of(api).post("/tasks", {"title": "Task"}) == {"id": 4}
    assert json.loads(Handler.received) == {"title": "Task"}


def test_a_server_error_is_retried_and_then_succeeds(api) -> None:
    Handler.script = [reply(b"{}", status=503), reply(b'{"ok": true}')]

    assert client_of(api, backoff=0).get("/health") == {"ok": True}
    assert Handler.calls == ["/health", "/health"]


def test_retries_are_bounded_and_the_error_says_what_arrived(api) -> None:
    Handler.script = [reply(b"{}", status=503)] * 3

    with pytest.raises(ApiError) as caught:
        client_of(api, retries=2, backoff=0).get("/health")

    assert "expected status 200, got 503" in str(caught.value)
    assert "gave up after 3 attempts" in str(caught.value)
    assert len(Handler.calls) == 3


def test_a_client_error_is_not_retried(api) -> None:
    Handler.script = [reply(b'{"error": "missing"}', status=404)]

    with pytest.raises(ApiError) as caught:
        client_of(api, backoff=0).get("/tasks/9")

    assert "got 404" in str(caught.value)
    assert len(Handler.calls) == 1


def test_html_instead_of_json_names_both_sides(api) -> None:
    Handler.script = [reply(b"<html></html>", kind="text/html")]

    with pytest.raises(ApiError) as caught:
        client_of(api).get("/tasks")

    assert "expected JSON, got text/html" in str(caught.value)


def test_broken_json_is_reported_with_its_first_bytes(api) -> None:
    Handler.script = [reply(b'{"id": ')]

    with pytest.raises(ApiError) as caught:
        client_of(api).get("/tasks")

    assert "expected valid JSON" in str(caught.value)


def test_a_closed_port_fails_saying_what_was_expected() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    with pytest.raises(ApiError) as caught:
        ApiClient(f"http://127.0.0.1:{port}", timeout=0.5, retries=0).get("/health")

    assert "expected an answer within 0.5s" in str(caught.value)
    assert "gave up after 1 attempt)" in str(caught.value)
