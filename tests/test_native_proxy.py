"""Private model HTTP uses the managed loopback lease even with proxy env set."""

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace

import pytest
import requests

from kiwibro.engine import DirectEngine, loopback_request
from kiwibro.llm import LLMClient, TurnAborted
from kiwibro.mind.native_generic import NATIVE_MODEL, NativeChatClient
from tests.test_mind_native_generic import MESSAGES, TOOLS, VISION


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch):
    records = []
    blocked, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            records.append((self.path, self.headers.get("Authorization"), None))
            self.respond(b"{}")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            records.append((self.path, self.headers.get("Authorization"), body))
            if self.path == "/apply-template":
                self.respond(json.dumps({"prompt": "tools" if body["tools"] else "plain"}).encode())
            elif self.path == "/tokenize":
                self.respond(json.dumps({"tokens": [0, 1, 2] if body["content"] == "tools" else [0]}).encode())
            elif self.path == "/redirect":
                self.send_response(307)
                self.send_header("Location", "/unexpected-redirect")
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif body.get("messages") == [{"role": "user", "content": "stall"}]:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", "8192")
                self.end_headers()
                self.wfile.flush()
                blocked.set()
                release.wait(timeout=4)
                self.close_connection = True
            else:
                self.respond(b'data: {"choices":[{"delta":{"content":"ready"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
                             "text/event-stream")

        def respond(self, body: bytes, mime: str = "application/json"):
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        origin = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("HTTP_PROXY", origin)
        for key in ("http_proxy", "no_proxy", "NO_PROXY", "REQUEST_METHOD"):
            monkeypatch.delenv(key, raising=False)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        try:
            yield origin, records, blocked, release
        finally:
            release.set()
            server.shutdown()
            thread.join(timeout=2)


def private_client(origin):
    lease = SimpleNamespace(url=origin, alive=lambda: True)
    acquired, released = [], []

    def acquire(model):
        acquired.append(model)
        return lease

    engine = SimpleNamespace(api_key="managed-fixture-key", acquire=acquire, release=released.append,
                             loaded=lambda _model: lease, lora_for=lambda *_args, **_kwargs: None)
    return NativeChatClient(lambda: engine), engine, lease, acquired, released


def test_managed_chat_schema_and_health_bypass_proxy_but_attached_requests_use_it(wire, monkeypatch):
    origin, records, _blocked, _release = wire
    monkeypatch.setenv("KIWIBRO_SERVER_KEY", "attached-fixture-key")
    native, engine, lease, acquired, released = private_client(origin)
    DirectEngine._wait_ready(engine, lease)
    assert native.stream_chat(model=NATIVE_MODEL, messages=VISION, tools=TOOLS).content == "ready"
    native.prefill(model=NATIVE_MODEL, messages=MESSAGES, tools=TOOLS)
    assert native.schema_tokens(TOOLS) == 2
    with loopback_request("POST", f"{origin}/redirect", json={}) as response:
        assert response.status_code == 307
    assert acquired == [NATIVE_MODEL, NATIVE_MODEL] and released == [lease, lease]
    assert [row[0] for row in records] == [
        "/health", "/v1/chat/completions", "/v1/chat/completions", "/apply-template", "/tokenize",
        "/apply-template", "/tokenize", "/redirect",
    ]
    assert all(row[1] == "Bearer managed-fixture-key" for row in records[:-1])
    assert records[1][2]["messages"] == VISION and records[1][2]["tools"] == TOOLS

    attached = LLMClient(lambda: {"server_url": "http://attached.invalid"})
    assert attached.stream_chat(model="attached", messages=MESSAGES).content == "ready"
    assert records[-1][0] == "http://attached.invalid/v1/chat/completions"
    assert records[-1][1] == "Bearer attached-fixture-key"


def test_managed_stream_cancels_during_prefill_and_releases_the_lease(wire):
    origin, records, blocked, release = wire
    native, _engine, lease, acquired, released = private_client(origin)
    abort = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        stream = pool.submit(native.stream_chat, model=NATIVE_MODEL, messages=[{"role": "user", "content": "stall"}],
                             abort=abort)
        try:
            assert blocked.wait(timeout=2)
            abort.set()
            with pytest.raises(TurnAborted):
                stream.result(timeout=2)
        finally:
            release.set()
    assert acquired == [NATIVE_MODEL] and released == [lease]
    assert [row[0] for row in records] == ["/v1/chat/completions"]


@pytest.mark.parametrize("url", ["http://attached.invalid", "https://127.0.0.1", "http://name@127.0.0.1"])
def test_managed_transport_rejects_destinations_outside_its_expected_url_format(url):
    with pytest.raises(requests.exceptions.InvalidURL, match="literal loopback"):
        loopback_request("POST", url)
