"""Tests for {{name}}; a local HTTP server serves the sample page."""

from __future__ import annotations

import csv
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from {{name}} import FetchError, fetch, link_rows, main, table_rows

PAGE = b"""<!doctype html>
<html><body>
  <a href="/about">About us</a>
  <a href="https://example.com/help">Help</a>
  <table>
    <tr><th>Name</th><th>Size</th></tr>
    <tr><td>alpha</td><td>10</td></tr>
    <tr><td>beta</td></tr>
  </table>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    status = 200
    kind = "text/html"
    body = b""

    def do_GET(self) -> None:
        self.send_response(Handler.status)
        self.send_header("content-type", Handler.kind)
        self.send_header("content-length", str(len(Handler.body)))
        self.end_headers()
        self.wfile.write(Handler.body)

    def log_message(self, pattern: str, *args: object) -> None:
        pass


@pytest.fixture()
def page_server():
    Handler.status, Handler.kind, Handler.body = 200, "text/html", PAGE
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture()
def closed_url() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}/"


def test_links_keep_their_text_and_resolve_relative_urls() -> None:
    rows = link_rows(PAGE.decode("utf-8"), "https://example.com/docs/")

    assert rows[0] == ["text", "url"]
    assert rows[1] == ["About us", "https://example.com/about"]
    assert rows[2] == ["Help", "https://example.com/help"]


def test_a_table_comes_back_padded_to_one_width() -> None:
    assert table_rows(PAGE.decode("utf-8")) == [["Name", "Size"], ["alpha", "10"], ["beta", ""]]


def test_a_page_without_a_table_gives_no_rows() -> None:
    assert table_rows("<p>No table here</p>") == []


def test_fetch_reads_the_page_over_http(page_server) -> None:
    assert "About us" in fetch(page_server)


def test_fetch_says_what_arrived_when_it_is_not_html(page_server) -> None:
    Handler.kind, Handler.body = "application/json", b"{}"

    with pytest.raises(FetchError) as caught:
        fetch(page_server)

    assert "expected HTML, got application/json" in str(caught.value)


def test_fetch_reports_the_status_it_got(page_server) -> None:
    Handler.status, Handler.body = 404, b"missing"

    with pytest.raises(FetchError) as caught:
        fetch(page_server)

    assert "expected status 200, got 404" in str(caught.value)


def test_fetch_gives_up_when_nothing_answers(closed_url) -> None:
    with pytest.raises(FetchError) as caught:
        fetch(closed_url, timeout=0.5)

    assert "expected an answer within 0.5s" in str(caught.value)


def test_main_saves_the_links_as_csv(page_server, tmp_path) -> None:
    out = tmp_path / "links.csv"

    assert main([page_server, "--out", str(out)]) == 0

    with out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[1] == ["About us", f"{page_server}/about"]


def test_main_reports_a_page_it_cannot_read(closed_url) -> None:
    assert main([closed_url, "--timeout", "0.5"]) == 1
