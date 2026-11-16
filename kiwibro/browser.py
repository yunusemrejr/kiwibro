"""Read and navigate public pages in isolated, per-chat browser tabs."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import re
import threading
from typing import Any
from urllib.parse import urljoin, urlsplit
import uuid

from bs4 import BeautifulSoup, Comment, Tag

from . import browser_render
from .search import (
    FETCH_MAX_BYTES,
    BROWSE_MAX_LINKS,
    SearchEngine,
    SearchError,
    assert_public_host,
    require_online,
)

MAX_PAGE_CHARS = 8000
MAX_HISTORY = 16
MAX_TABS = 12
MAX_SESSIONS = 32


def _link_url(base_url: str, href: str) -> str:
    """Hand back only links the browser can actually follow."""
    if not href or href.startswith("#") or len(href) > 2048 or any(ord(char) < 32 for char in href):
        return ""
    try:
        absolute = urljoin(base_url, href.replace(" ", "%20"))
        parts = urlsplit(absolute)
        if parts.scheme in ("http", "https") and parts.hostname and parts.username is None:
            return absolute
        if not base_url and not parts.scheme and not parts.netloc:
            return absolute
    except ValueError:
        pass
    return ""


def _render_table(table_tag: Tag) -> str:
    """Format an HTML table into a standard Markdown pipe table."""
    rows: list[list[str]] = []
    for tr in table_tag.find_all("tr"):
        cells = [
            " ".join(c.get_text(" ", strip=True).replace("|", "\\|").split())
            for c in tr.find_all(["th", "td"])
        ]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    max_cols = max(len(r) for r in rows)
    if max_cols == 0:
        return ""
    padded = [r + [""] * (max_cols - len(r)) for r in rows]
    header = padded[0]
    sep = ["---"] * max_cols
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in padded[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n\n" + "\n".join(lines) + "\n\n"


def _render_list(
    list_tag: Tag,
    is_ol: bool = False,
    depth: int = 0,
    include_links: bool = True,
    base_url: str = "",
) -> str:
    """Recursively format unordered and ordered lists into Markdown."""
    indent = "  " * depth
    lines: list[str] = []
    idx = 1
    for child in list_tag.find_all("li", recursive=False):
        prefix = f"{indent}{idx}. " if is_ol else f"{indent}* "
        idx += 1
        parts: list[str] = []
        nested: list[Tag] = []
        for c in child.children:
            if isinstance(c, Tag) and c.name in ("ul", "ol"):
                nested.append(c)
            else:
                parts.append(_render_node(c, include_links, base_url))
        item_text = " ".join("".join(parts).split())
        lines.append(f"{prefix}{item_text}")
        for n in nested:
            lines.append(
                _render_list(
                    n,
                    is_ol=(n.name == "ol"),
                    depth=depth + 1,
                    include_links=include_links,
                    base_url=base_url,
                )
            )
    return "\n".join(lines)


def _render_node(node: Any, include_links: bool, base_url: str) -> str:
    """Render a DOM node into Markdown recursively."""
    if isinstance(node, Comment):
        return ""
    if not isinstance(node, Tag):
        return str(node)

    tag = node
    name = (tag.name or "").lower()

    if name in ("script", "style", "noscript", "svg", "iframe", "head"):
        return ""

    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        lvl = min(int(name[1]), 4)
        txt = " ".join(tag.get_text(" ", strip=True).split())
        return f"\n\n{'#' * lvl} {txt}\n\n"

    if name == "pre":
        code_tag = tag.find("code")
        lang = ""
        if code_tag and code_tag.get("class"):
            for cls in code_tag["class"]:
                if cls.startswith("language-") or cls.startswith("lang-"):
                    lang = cls.split("-", 1)[1]
                    break
        raw_code = tag.get_text()
        return f"\n\n```{lang}\n{raw_code}\n```\n\n"

    if name == "code":
        return f" `{tag.get_text(strip=True)}` "

    if name == "blockquote":
        inner = "".join(_render_node(c, include_links, base_url) for c in tag.children).strip()
        quoted = "\n".join(f"> {line}" for line in inner.splitlines() if line.strip())
        return f"\n\n{quoted}\n\n"

    if name == "table":
        return _render_table(tag)

    if name in ("ul", "ol"):
        return f"\n\n{_render_list(tag, is_ol=(name == 'ol'), include_links=include_links, base_url=base_url)}\n\n"

    if name == "p":
        inner = "".join(_render_node(c, include_links, base_url) for c in tag.children).strip()
        return f"\n\n{inner}\n\n" if inner else ""

    if name == "a":
        txt = " ".join(tag.get_text(" ", strip=True).split())
        href = _link_url(base_url, tag.get("href", "").strip())
        if not href:
            return f" {txt} "
        if include_links:
            return f" [{txt or href}]({href}) "
        return f" {txt or href} "

    if name in ("strong", "b"):
        inner = "".join(_render_node(c, include_links, base_url) for c in tag.children).strip()
        return f" **{inner}** " if inner else ""

    if name in ("em", "i"):
        inner = "".join(_render_node(c, include_links, base_url) for c in tag.children).strip()
        return f" *{inner}* " if inner else ""

    if name == "br":
        return "\n"

    if name == "hr":
        return "\n\n---\n\n"

    return "".join(_render_node(c, include_links, base_url) for c in tag.children)


def html_to_markdown(
    html: str,
    base_url: str = "",
    selector: str | None = None,
    include_links: bool = True,
) -> dict[str, Any]:
    """Parse HTML with BeautifulSoup, extract links/forms, and convert into structured Markdown."""
    soup = BeautifulSoup(html or "", "html.parser")

    title_tag = soup.find("title")
    title = " ".join(title_tag.get_text(" ", strip=True).split())[:300] if title_tag else ""
    base = soup.find("base", href=True)
    link_base = (_link_url(base_url, base["href"].strip()) if base else "") or base_url
    for tag in soup.select("script, style, noscript, template, svg, iframe, [hidden]"):
        tag.decompose()

    if selector:
        try:
            scope = soup.select_one(selector)
        except Exception as exc:
            raise SearchError(f"Invalid CSS selector '{selector}': {exc}") from exc
        if scope is None:
            raise SearchError(f"CSS selector '{selector}' matched no elements.")
    else:
        scope = soup.find("article") or soup.find("main") or soup.body or soup

    # Extract deduplicated links
    links: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    body = soup.body or soup
    for root in (scope, body) if not selector and scope is not body else (scope,):
        for a in root.find_all("a", href=True):
            abs_url = _link_url(link_base, a["href"].strip())
            if not abs_url or abs_url == base_url or abs_url in seen_urls:
                continue
            seen_urls.add(abs_url)
            txt = " ".join((a.get_text(" ", strip=True) or a.get("aria-label", "")).split())
            links.append({"n": len(links) + 1, "text": (txt or abs_url)[:120], "url": abs_url})
            if len(links) >= BROWSE_MAX_LINKS:
                break
        if len(links) >= BROWSE_MAX_LINKS:
            break

    # Extract forms summary
    forms: list[dict[str, Any]] = []
    for f in scope.find_all("form", limit=12):
        action = f.get("action", "")
        if base_url and action:
            try:
                action = urljoin(base_url, action)
            except Exception:
                pass
        method = f.get("method", "GET").upper()
        fields: list[dict[str, Any]] = []
        for el in f.find_all(["input", "textarea", "select", "button"], limit=24):
            tag_name = el.name
            name = el.get("name") or el.get("id") or ""
            field_type = el.get("type", "text" if tag_name == "input" else tag_name)
            fields.append({
                "name": name,
                "type": field_type,
                "tag": tag_name,
            })
        forms.append({
            "action": action,
            "method": method,
            "fields": fields,
        })

    # Structured Markdown rendering
    raw_md = _render_node(scope, include_links, link_base)
    markdown = re.sub(r"\n{3,}", "\n\n", raw_md).strip()

    # Plain text extraction
    plain_text = re.sub(r"\n{2,}", "\n\n", scope.get_text("\n", strip=True)).strip()

    return {
        "title": title,
        "markdown": markdown,
        "text": plain_text,
        "links": links,
        "forms": forms,
    }


def render_headless(url: str, timeout: int = 12) -> str | None:
    """Return rendered HTML for a public URL without opening a chat tab."""
    current = _link_url("", (url or "").strip())
    if current:
        try:
            assert_public_host(urlsplit(current).hostname or "")
        except SearchError:
            return None
        rendered = browser_render.render(current, timeout)
        return rendered[1] if rendered else None
    return None


def inspect_page(
    url: str,
    *,
    mode: str = "markdown",
    selector: str | None = None,
    max_chars: int = 6000,
    include_links: bool = True,
    search_engine: SearchEngine | None = None,
) -> dict[str, Any]:
    """Fetch URL safely via SSRF guard or headless mode, and parse into structured Markdown."""
    engine = search_engine or SearchEngine()
    current = (url or "").strip()
    if not current:
        raise SearchError("URL is empty.")
    parts = urlsplit(current)
    if parts.scheme not in ("http", "https"):
        raise SearchError("Only http(s) URLs can be fetched.")
    if not parts.hostname:
        raise SearchError("URL has no host.")
    if parts.username is not None:
        raise SearchError("Browser URLs cannot contain credentials.")
    require_online()
    assert_public_host(parts.hostname)

    rendered = browser_render.render(current) if mode == "headless" else None
    if rendered:
        final_url, html = rendered
        mode_used = "headless"
    else:
        final_url, _ct, encoding, raw = engine._guarded_get(
            current, read_timeout=10, max_bytes=FETCH_MAX_BYTES
        )
        html = SearchEngine._decode(raw, encoding)
        mode_used = "http_fallback" if mode == "headless" else "http"

    parsed = html_to_markdown(html, base_url=final_url, selector=selector, include_links=include_links)
    empty = not (parsed["text"] or parsed["links"] or parsed["forms"])
    if mode == "auto" and empty and re.search(r"<script(?:\s|>)", html, re.IGNORECASE):
        rendered = browser_render.render(final_url)
        if rendered:
            final_url, html = rendered
            parsed = html_to_markdown(html, base_url=final_url, selector=selector, include_links=include_links)
        mode_used = "headless" if rendered else "http_fallback"
    md = parsed["markdown"]
    txt = parsed["text"]
    if not txt and not parsed["links"] and not parsed["forms"]:
        retry = {
            "http": " Try headless mode if the page needs JavaScript.",
            "http_fallback": " JavaScript rendering was unavailable.",
        }.get(mode_used, "")
        raise SearchError("The page has no readable text or links." + retry)
    truncated = len(md) > max_chars or len(txt) > max_chars

    return {
        "title": parsed["title"] or parts.hostname or "",
        "url": final_url,
        "markdown": md[:max_chars],
        "text": txt[:max_chars],
        "links": parsed["links"],
        "forms": parsed["forms"],
        "truncated": truncated,
        "mode_used": mode_used,
        "mode_requested": mode,
    }


@dataclass
class BrowserTab:
    tab_id: str
    last_page: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_summary(self, is_active: bool = False) -> dict[str, Any]:
        return {
            "tab_id": self.tab_id,
            "url": self.last_page["url"],
            "title": self.last_page["title"],
            "active": is_active,
            "history_len": len(self.history) + 1,
        }


class BrowserEngine:
    """Multi-tab browser session manager per conversation."""

    def __init__(self, search_engine: SearchEngine | None = None) -> None:
        self._lock = threading.RLock()
        self._search_engine = search_engine or SearchEngine()
        self._sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def _get_session(self, conv_id: str) -> dict[str, Any]:
        if conv_id not in self._sessions:
            if len(self._sessions) >= MAX_SESSIONS:
                self._sessions.popitem(last=False)
            self._sessions[conv_id] = {
                "tabs": {},
                "active_tab_id": None,
            }
        self._sessions.move_to_end(conv_id)
        return self._sessions[conv_id]

    def _list_tabs_unlocked(self, conv_id: str) -> list[dict[str, Any]]:
        sess = self._get_session(conv_id)
        active_id = sess["active_tab_id"]
        return [tab.to_summary(is_active=(tid == active_id)) for tid, tab in sess["tabs"].items()]

    def _tab_unlocked(self, conv: str, tab_id: str | None) -> BrowserTab:
        sess = self._get_session(conv)
        tab = sess["tabs"].get(tab_id or sess["active_tab_id"])
        if tab is None:
            raise SearchError(f"Tab '{tab_id}' is not open. List tabs to see which are available."
                              if tab_id else "No active tab. Open a page first.")
        return tab

    def _outcome(self, conv: str, tab: BrowserTab, max_chars: int) -> dict[str, Any]:
        page = tab.last_page
        return {
            "ok": True,
            "tab_id": tab.tab_id,
            "page": {
                **page,
                "markdown": page["markdown"][:max_chars],
                "text": page["text"][:max_chars],
                "truncated": page["truncated"] or max(len(page["markdown"]), len(page["text"])) > max_chars,
            },
            "tabs": self._list_tabs_unlocked(conv),
        }

    def open_tab(
        self,
        url: str,
        conversation_id: str | None = None,
        mode: str = "markdown",
        selector: str | None = None,
        max_chars: int = 6000,
        include_links: bool = True,
    ) -> dict[str, Any]:
        conv = conversation_id or "default"
        page = self.inspect_page(
            url,
            mode=mode,
            selector=selector,
            max_chars=MAX_PAGE_CHARS,
            include_links=include_links,
        )
        with self._lock:
            sess = self._get_session(conv)
            if len(sess["tabs"]) >= MAX_TABS:
                raise SearchError(f"This chat already has {MAX_TABS} tabs. Close one before opening another.")
            tab_id = f"tab_{uuid.uuid4().hex[:6]}"
            tab = BrowserTab(tab_id, page)
            sess["tabs"][tab_id] = tab
            sess["active_tab_id"] = tab_id
            return self._outcome(conv, tab, max_chars)

    def navigate(
        self,
        conversation_id: str | None = None,
        *,
        url: str | None = None,
        link_number: int | None = None,
        tab_id: str | None = None,
        mode: str | None = None,
        selector: str | None = None,
        max_chars: int = 6000,
        include_links: bool = True,
    ) -> dict[str, Any]:
        conv = conversation_id or "default"
        with self._lock:
            tab = self._tab_unlocked(conv, tab_id)
            previous = tab.last_page
            if link_number is not None:
                links = previous["links"]
                if not 1 <= link_number <= len(links):
                    raise SearchError("Link number is unavailable on the current page. Inspect the tab for its links.")
                url = links[link_number - 1]["url"]

        page = self.inspect_page(url or "", mode=mode or previous.get("mode_requested", "markdown"),
                                 selector=selector, max_chars=MAX_PAGE_CHARS, include_links=include_links)
        with self._lock:
            sess = self._sessions.get(conv)
            if not sess or sess["tabs"].get(tab.tab_id) is not tab or tab.last_page is not previous:
                raise SearchError("The tab changed while the page loaded. Inspect its current page before navigating again.")
            if previous["url"] != page["url"]:
                tab.history.append(previous)
                del tab.history[:-MAX_HISTORY]
            tab.last_page = page
            sess["active_tab_id"] = tab.tab_id
            return self._outcome(conv, tab, max_chars)

    def current_page(self, conversation_id: str | None = None, *, tab_id: str | None = None,
                     max_chars: int = 6000) -> dict[str, Any]:
        conv = conversation_id or "default"
        with self._lock:
            return self._outcome(conv, self._tab_unlocked(conv, tab_id), max_chars)

    def position(self, conversation_id: str, tab_id: str | None = None) -> tuple[str, str, int] | None:
        """Identify the logical starting page for a stateful tool call's repeat guard."""
        with self._lock:
            session = self._sessions.get(conversation_id)
            if session is None:
                return None
            tab = session["tabs"].get(tab_id or session["active_tab_id"])
            return (tab.tab_id, tab.last_page["url"], len(tab.history)) if tab else None

    def list_tabs(self, conversation_id: str | None = None) -> list[dict[str, Any]]:
        conv = conversation_id or "default"
        with self._lock:
            return self._list_tabs_unlocked(conv)

    def switch_tab(self, tab_id: str, conversation_id: str | None = None, *, max_chars: int = 6000) -> dict[str, Any]:
        conv = conversation_id or "default"
        with self._lock:
            tab = self._tab_unlocked(conv, tab_id)
            self._sessions[conv]["active_tab_id"] = tab_id
            return self._outcome(conv, tab, max_chars)

    def close_tab(self, tab_id: str, conversation_id: str | None = None) -> bool:
        conv = conversation_id or "default"
        with self._lock:
            sess = self._get_session(conv)
            if tab_id not in sess["tabs"]:
                return False
            del sess["tabs"][tab_id]
            if sess["active_tab_id"] == tab_id:
                sess["active_tab_id"] = next(reversed(sess["tabs"]), None)
            if not sess["tabs"]:
                del self._sessions[conv]
            return True

    def back(
        self,
        conversation_id: str | None = None,
        *,
        tab_id: str | None = None,
        max_chars: int = 6000,
    ) -> dict[str, Any]:
        conv = conversation_id or "default"
        with self._lock:
            tab = self._tab_unlocked(conv, tab_id)
            if not tab.history:
                return {
                    "ok": False,
                    "reason": "Tab has no previous history.",
                    "tab_id": tab.tab_id,
                }
            tab.last_page = tab.history.pop()
            self._sessions[conv]["active_tab_id"] = tab.tab_id
            return self._outcome(conv, tab, max_chars)

    def inspect_page(
        self,
        url: str,
        *,
        mode: str = "markdown",
        selector: str | None = None,
        max_chars: int = 6000,
        include_links: bool = True,
    ) -> dict[str, Any]:
        return inspect_page(
            url,
            mode=mode,
            selector=selector,
            max_chars=max_chars,
            include_links=include_links,
            search_engine=self._search_engine,
        )


BROWSER = BrowserEngine()

