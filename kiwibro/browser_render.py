"""Render a public page in disposable Chrome with all network access brokered by Kiwibro."""

from __future__ import annotations

import base64
from concurrent.futures import Future, ThreadPoolExecutor
import http.client
import ipaddress
import json
import logging
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlsplit

from .pagecheck import _quit, find_chrome
from .search import FETCH_MAX_BYTES, USER_AGENT, _public_ip

log = logging.getLogger(__name__)
MAX_REQUESTS = 64
MAX_TOTAL_BYTES = 12_000_000
REQUEST_TIMEOUT = 3
_LOCK = threading.Lock()
_POOL = ThreadPoolExecutor(max_workers=6, thread_name_prefix="browser-resource")
_SKIP = frozenset({"Image", "Media", "Font", "Ping", "Prefetch"})
_PASSTHROUGH = frozenset({"accept", "accept-language", "origin", "referer", "user-agent"})
_HOP_HEADERS = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                          "te", "trailer", "transfer-encoding", "upgrade"})


def _resource(url: str, method: str, headers: dict[str, str]) -> dict | None:
    """Perform one bounded GET/HEAD, pinning the connection to the validated public IP."""
    if method not in ("GET", "HEAD"):
        return None
    try:
        parts = urlsplit(url)
        host = parts.hostname
        if parts.scheme not in ("http", "https") or not host or parts.username is not None:
            return None
        tls = parts.scheme == "https"
        port = parts.port or (443 if tls else 80)
        addresses = list(dict.fromkeys(info[4][0] for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)))
        if not addresses or any(not (ip.is_global and _public_ip(ip)) for ip in map(ipaddress.ip_address, addresses)):
            return None

        def connect(_address, timeout=REQUEST_TIMEOUT, source_address=None):
            # Most desktop connections have IPv4 even when DNS returns AAAA first.
            target = min(addresses, key=lambda address: ":" in address)
            return socket.create_connection((target, port), timeout, source_address)

        cls = http.client.HTTPSConnection if tls else http.client.HTTPConnection
        connection = cls(host, port, timeout=REQUEST_TIMEOUT)
        connection._create_connection = connect
        try:
            path = parts.path or "/"
            if parts.query:
                path += "?" + parts.query
            allowed = {key.lower(): value for key, value in headers.items() if key.lower() in _PASSTHROUGH}
            allowed.setdefault("user-agent", USER_AGENT)
            connection.request(method, path, headers=allowed)
            with connection.getresponse() as response:
                body = response.read(FETCH_MAX_BYTES + 1)
                if len(body) > FETCH_MAX_BYTES:
                    return None
                return {
                    "responseCode": response.status,
                    "responseHeaders": [{"name": key, "value": value} for key, value in response.getheaders()
                                        if key.lower() not in _HOP_HEADERS],
                    "body": base64.b64encode(body).decode("ascii"),
                }
        finally:
            connection.close()
    except (OSError, ValueError, http.client.HTTPException):
        return None


def _page_socket(profile: Path, process: subprocess.Popen, deadline: float):
    import websocket
    import requests

    # Chrome chooses an exclusive debug port, recorded in the temporary private profile.
    # Using the system's proxy settings for this local control connection would leak the socket.
    with requests.Session() as session:
        session.trust_env = False
        while time.monotonic() < deadline and process.poll() is None:
            try:
                port = int((profile / "DevToolsActivePort").read_text().splitlines()[0])
                targets = session.get(f"http://127.0.0.1:{port}/json/list", timeout=0.5).json()
                page = next(target for target in targets if target.get("type") == "page")
                return websocket.create_connection(page["webSocketDebuggerUrl"], timeout=0.5,
                                                   suppress_origin=True, http_no_proxy=["127.0.0.1"])
            except (OSError, ValueError, StopIteration, requests.RequestException, websocket.WebSocketException):
                time.sleep(0.05)
    return None


def _observe(ws, url: str, deadline: float) -> tuple[str, str] | None:
    import websocket

    last_id = 0

    def send(method: str, **params) -> int:
        nonlocal last_id
        last_id += 1
        ws.send(json.dumps({"id": last_id, "method": method, "params": params}))
        return last_id

    def fail(request_id: str) -> None:
        send("Fetch.failRequest", requestId=request_id, errorReason="BlockedByClient")

    setup = set()
    for method, params in (("Page.enable", {}), ("Network.enable", {}), ("Fetch.enable", {}),
                           ("Network.setBypassServiceWorker", {"bypass": True})):
        setup.add(send(method, **params))
    navigate_id = send("Page.navigate", url=url)
    in_flight: dict[str, Future] = {}
    requests_started = 0
    byte_count = 0
    last_activity = time.monotonic()
    dom_ready = None
    evaluate_id = None
    main_frame = None
    document_status = None
    ws.settimeout(0.08)
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            for request_id, future in list(in_flight.items()):
                if not future.done():
                    continue
                del in_flight[request_id]
                last_activity = now
                response = future.result()
                if response:
                    byte_count += len(response["body"]) * 3 // 4
                if response and byte_count <= MAX_TOTAL_BYTES:
                    send("Fetch.fulfillRequest", requestId=request_id, **response)
                else:
                    fail(request_id)
            settled = not in_flight and now - last_activity >= 0.45
            if evaluate_id is None and dom_ready and document_status is not None and (settled or now - dom_ready > 3):
                if document_status != 200:
                    return None
                evaluate_id = send("Runtime.evaluate", returnByValue=True, expression=(
                    "(() => [location.href, document.documentElement?.outerHTML.slice(0, "
                    f"{FETCH_MAX_BYTES}) || ''])()"
                ))
            try:
                message = json.loads(ws.recv())
            except websocket.WebSocketTimeoutException:
                continue

            message_id = message.get("id")
            result = message.get("result", {})
            if message_id in setup and "error" in message:
                return None
            if message_id == navigate_id:
                if result.get("errorText") or "error" in message:
                    return None
                main_frame = result.get("frameId") or main_frame
            if evaluate_id and message_id == evaluate_id:
                value = result.get("result", {}).get("value")
                if isinstance(value, list) and len(value) == 2:
                    final_url, html = value
                    if urlsplit(final_url).scheme in ("http", "https") and html:
                        return final_url, html
                return None
            method = message.get("method")
            params = message.get("params", {})
            if method == "Fetch.requestPaused":
                request_id = params["requestId"]
                request = params["request"]
                if (params.get("resourceType") in _SKIP or requests_started >= MAX_REQUESTS or
                        byte_count >= MAX_TOTAL_BYTES or len(in_flight) >= 18):
                    fail(request_id)
                else:
                    requests_started += 1
                    last_activity = now
                    in_flight[request_id] = _POOL.submit(_resource, request["url"], request["method"], request["headers"])
            elif method == "Page.frameNavigated" and not params["frame"].get("parentId"):
                main_frame = params["frame"]["id"]
                last_activity = now
                dom_ready = None
            elif method == "Page.domContentEventFired":
                dom_ready = now
            elif method == "Network.responseReceived" and params.get("type") == "Document":
                if main_frame is None or params.get("frameId") == main_frame:
                    document_status = params["response"]["status"]
            elif method == "Page.javascriptDialogOpening":
                send("Page.handleJavaScriptDialog", accept=False)
    finally:
        for future in in_flight.values():
            future.cancel()
    return None


def render(url: str, timeout: float = 12) -> tuple[str, str] | None:
    """Return (landed URL, live HTML), or None so the caller can report a static fallback."""
    try:
        import websocket
    except ImportError:
        return None
    chrome = find_chrome()
    if not chrome:
        return None
    deadline = time.monotonic() + timeout
    if not _LOCK.acquire(timeout=timeout):
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="kiwibro-browser-") as dirname, socket.socket() as blocked_proxy:
            profile = Path(dirname)
            # Reserve a port which refuses connections for this Chrome process's lifetime.
            blocked_proxy.bind(("127.0.0.1", 0))
            proxy_port = blocked_proxy.getsockname()[1]
            process = subprocess.Popen([
                chrome, "--headless=new", "--no-first-run", "--no-default-browser-check",
                "--disable-background-networking", "--disable-dev-shm-usage", "--disable-extensions",
                "--disable-quic", "--mute-audio", "--remote-debugging-port=0",
                "--force-webrtc-ip-handling-policy=disable_non_proxied_udp", "--proxy-bypass-list=<-loopback>",
                f"--proxy-server=http://127.0.0.1:{proxy_port}", "--host-resolver-rules=MAP * ~NOTFOUND",
                f"--user-data-dir={profile}", "about:blank",
            ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            try:
                ws = _page_socket(profile, process, deadline)
                if ws is None:
                    return None
                try:
                    return _observe(ws, url, deadline)
                finally:
                    ws.close(timeout=0.1)
            finally:
                if process.poll() is None:
                    _quit(process)
                process.wait(timeout=1)
    except (OSError, ValueError, subprocess.TimeoutExpired, websocket.WebSocketException):
        log.debug("Browser render unavailable", exc_info=True)
        return None
    finally:
        _LOCK.release()
