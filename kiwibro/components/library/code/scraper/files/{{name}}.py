"""Fetch one page and save its links or its first table as CSV."""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.error
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urljoin

USER_AGENT = "{{name}}/0.1"
HTML_TYPES = ("text/html", "application/xhtml+xml")


class FetchError(Exception):
    """The page could not be read, saying what was expected and what arrived."""


def fetch(url: str, *, timeout: float = 10.0) -> str:
    """Return the HTML at url, or raise FetchError with both sides of the mismatch."""
    request = urllib.request.Request(url, headers={"user-agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            kind = response.headers.get_content_type()
            if kind not in HTML_TYPES:
                raise FetchError(f"{url}: expected HTML, got {kind} with {len(response.read())} bytes")
            charset = response.headers.get_content_charset("utf-8")
            return response.read().decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        raise FetchError(f"{url}: expected status 200, got {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise FetchError(f"{url}: expected an answer within {timeout}s, got {exc.reason}") from exc
    except TimeoutError as exc:
        raise FetchError(f"{url}: expected an answer within {timeout}s, got a timeout") from exc


class _Collector(HTMLParser):
    """One pass over the page, keeping anchor text with its href and table cells in rows."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.tables: list[list[list[str]]] = []
        self._href: str | None = None
        self._link_text: list[str] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "a" and values.get("href") is not None:
            self._href = values["href"]
            self._link_text = []
        elif tag == "table":
            self.tables.append([])
        elif tag == "tr" and self.tables:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            self.links.append((_spaced(self._link_text), self._href))
            self._href = None
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(_spaced(self._cell))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.tables[-1].append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._link_text.append(data)
        if self._cell is not None:
            self._cell.append(data)


def _spaced(parts: list[str]) -> str:
    return " ".join("".join(parts).split())


def _collect(html: str) -> _Collector:
    collector = _Collector()
    collector.feed(html)
    collector.close()
    return collector


def link_rows(html: str, base_url: str) -> list[list[str]]:
    """A header row, then one row per anchor with its text and absolute URL."""
    return [["text", "url"], *([text, urljoin(base_url, href)] for text, href in _collect(html).links)]


def table_rows(html: str) -> list[list[str]]:
    """Every row of the first table, padded so each row has the same number of cells."""
    tables = _collect(html).tables
    if not tables:
        return []
    width = max((len(row) for row in tables[0]), default=0)
    return [row + [""] * (width - len(row)) for row in tables[0]]


def write_csv(rows: list[list[str]], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="{{name}}", description="Save a page's links or first table as CSV.")
    parser.add_argument("url", help="the page to read, for example https://example.com")
    parser.add_argument("--what", choices=("links", "tables"), default="links")
    parser.add_argument("--out", default="-", help="CSV file to write; - prints to standard output")
    parser.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for the page")
    args = parser.parse_args(argv)

    try:
        html = fetch(args.url, timeout=args.timeout)
    except FetchError as problem:
        print(f"{{name}}: {problem}", file=sys.stderr)
        return 1

    rows = link_rows(html, args.url) if args.what == "links" else table_rows(html)
    if not rows or (args.what == "links" and len(rows) == 1):
        print(f"{{name}}: {args.url} has no {args.what}", file=sys.stderr)
        return 1
    if args.out == "-":
        csv.writer(sys.stdout).writerows(rows)
    else:
        write_csv(rows, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
