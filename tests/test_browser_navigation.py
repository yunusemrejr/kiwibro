from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from kiwibro import browser, browser_render, connectivity, tools
from kiwibro.search import SearchError


def test_browser_tool_follows_observed_redirected_links_and_goes_back(monkeypatch):
    monkeypatch.setenv(connectivity.ENV_SWITCH, "0")
    monkeypatch.setattr(browser, "assert_public_host", lambda host: None)
    requested = []

    class Pages:
        def _guarded_get(self, url, **bounds):
            requested.append(url)
            if url == "https://example.test/start":
                return (
                    "https://example.test/help/index.html", "text/html", "latin-1",
                    '''<main><h1>Café</h1><a href="page.html">Guide</a></main>
                    <nav><a href="/faq">FAQ</a><a href="javascript:alert(0)">Ignore</a></nav>'''.encode(),
                )
            if url == "https://example.test/help/page.html":
                return url, "text/html", "utf-8", b"<h1>Guide</h1><p>Source passage</p>"
            raise SearchError("Page was unavailable.")

    monkeypatch.setattr(browser, "BROWSER", browser.BrowserEngine(Pages()))

    def act(action, chat="first", **args):
        validated = tools.validate_args("browser", {"action": action, **args})
        return tools.execute_tool("browser", validated, SimpleNamespace(conversation_id=chat))

    first = act("open", url="https://example.test/start")
    assert first.ok and "Café" in first.content
    assert "[1] Guide" in first.content and "[2] FAQ" in first.content
    assert "Ignore" not in first.content
    tab_id = first.meta["tab_id"]
    repeated = tools.validate_args("browser", {"action": "follow", "link_number": 1})
    before = tools.tool_call_signature("browser", repeated, conversation_id="first")
    assert not act("follow", chat="different", tab_id=tab_id, link_number=1).ok
    assert not act("follow", link_number=3).ok and len(requested) == 1
    linked = act("follow", tab_id=tab_id, link_number=1)
    assert linked.ok and linked.meta["tab_id"] == tab_id
    assert linked.meta["url"] == "https://example.test/help/page.html"
    assert tools.tool_call_signature("browser", repeated, conversation_id="first") != before
    assert "Source passage" in linked.content
    assert not act("navigate", url="https://example.test/broken").ok
    assert act("inspect").meta["url"] == linked.meta["url"]
    back = act("back")
    assert back.ok and back.meta["url"].endswith("/help/index.html")
    assert back.meta["tab_id"] == tab_id
    assert tools.tool_call_signature("browser", repeated, conversation_id="first") == before
    assert not act("back").ok and len(requested) == 3  # history restore and inspect each perform no fetch


def test_headless_resource_rejects_private_hosts_and_post_before_connect(monkeypatch):
    def must_not_connect(*args, **kwargs):
        pytest.fail("a rejected browser request tried to connect")

    monkeypatch.setattr(browser_render.socket, "create_connection", must_not_connect)
    assert browser_render._resource("http://127.0.0.1:8777/api/settings", "GET", {}) is None
    assert browser_render._resource("http://[::ffff:127.0.0.1]/", "GET", {}) is None
    assert browser_render._resource("http://100.64.0.9/", "GET", {}) is None
    assert browser_render._resource("https://8.8.8.8/", "POST", {}) is None


def test_auto_mode_uses_headless_only_for_a_script_shell(monkeypatch):
    monkeypatch.setenv(connectivity.ENV_SWITCH, "0")
    monkeypatch.setattr(browser, "assert_public_host", lambda host: None)
    urls_rendered = []

    class Pages:
        def _guarded_get(self, url, **bounds):
            html = b"<main><h1>Static</h1></main>" if url.endswith("/static") else b"<script src='app.js'></script>"
            return url, "text/html", "utf-8", html

    def render(url):
        urls_rendered.append(url)
        return url, "<main><h1>Rendered</h1></main>"

    monkeypatch.setattr(browser_render, "render", render)
    engine = browser.BrowserEngine(Pages())
    assert engine.inspect_page("https://fixtures.example/static", mode="auto")["mode_used"] == "http"
    assert engine.inspect_page("https://fixtures.example/app", mode="auto")["text"] == "Rendered"
    assert urls_rendered == ["https://fixtures.example/app"]


@pytest.mark.skipif(browser_render.find_chrome() is None, reason="Chrome is not installed")
def test_browser_tool_renders_javascript_and_uses_the_landed_url(monkeypatch):
    host = "fixtures.example"
    monkeypatch.setenv(connectivity.ENV_SWITCH, "0")
    monkeypatch.setattr(browser, "assert_public_host", lambda name: None)
    real_resource = browser_render._resource
    requested = []

    def response(code, source="", **headers):
        return {
            "responseCode": code,
            "responseHeaders": [{"name": key, "value": val} for key, val in headers.items()],
            "body": base64.b64encode(source.encode()).decode(),
        }

    def resource(url, method, headers):
        requested.append((url, method))
        if url == f"https://{host}/start":
            return response(302, Location="/docs/page.html")
        if url == f"https://{host}/docs/page.html":
            return response(200, '<!doctype html><title>Browser test</title><main></main><script src="app.js"></script>',
                            **{"Content-Type": "text/html; charset=utf-8"})
        if url == f"https://{host}/docs/app.js":
            js = '''document.querySelector('main').innerHTML = '<h1>Rendered by JavaScript</h1><a href="guide.html">Guide</a>';
                fetch('http://127.0.0.1/private').catch(() => {});
                fetch('/mutation', {method: 'POST',body: 'forbidden'}).catch(() => {});'''
            return response(200, js, **{"Content-Type": "text/javascript"})
        return real_resource(url, method, headers) if url.startswith("http://127.0.0.1/") else None

    monkeypatch.setattr(browser_render, "_resource", resource)
    result = tools.execute_tool("browser", tools.validate_args("browser", {
        "action": "inspect", "url": f"https://{host}/start", "mode": "headless",
    }), SimpleNamespace(conversation_id="headless"))
    assert result.ok, result.content
    assert result.meta["mode"] == "headless"
    assert result.meta["url"] == f"https://{host}/docs/page.html"
    assert "Rendered by JavaScript" in result.content and f"https://{host}/docs/guide.html" in result.content
    assert (f"https://{host}/docs/app.js", "GET") in requested
