"""Advanced web browser and page inspection engine for Kiwibro agents."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
import shutil
import subprocess
import threading
import time
from typing import Any
from urllib.parse import urljoin, urlsplit
import uuid

from bs4 import BeautifulSoup, Tag

from .search import (
    FETCH_MAX_BYTES,
    HEADERS,
    SearchEngine,
    SearchError,
    assert_public_host,
    normalize_url,
)

log = logging.getLogger(__name__)


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
        href = tag.get("href", "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            return f" {txt} "
        if base_url:
            try:
                href = urljoin(base_url, href)
            except Exception:
                pass
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
    title = " ".join(title_tag.get_text(" ", strip=True).split()) if title_tag else ""

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
    for a in scope.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        abs_url = urljoin(base_url, href) if base_url else href
        norm = normalize_url(abs_url) if abs_url.startswith(("http://", "https://")) else abs_url
        if norm in seen_urls:
            continue
        seen_urls.add(norm)
        txt = " ".join(a.get_text(" ", strip=True).split())
        links.append({
            "n": len(links) + 1,
            "text": txt[:120] or abs_url,
            "url": abs_url,
        })

    # Extract forms summary
    forms: list[dict[str, Any]] = []
    for f in scope.find_all("form"):
        action = f.get("action", "")
        if base_url and action:
            try:
                action = urljoin(base_url, action)
            except Exception:
                pass
        method = f.get("method", "GET").upper()
        fields: list[dict[str, Any]] = []
        for el in f.find_all(["input", "textarea", "select", "button"]):
            tag_name = el.name
            name = el.get("name") or el.get("id") or ""
            field_type = el.get("type", "text" if tag_name == "input" else tag_name)
            value = el.get("value", "")
            fields.append({
                "name": name,
                "type": field_type,
                "value": value,
                "tag": tag_name,
            })
        forms.append({
            "action": action,
            "method": method,
            "fields": fields,
        })

    # Structured Markdown rendering
    raw_md = _render_node(scope, include_links, base_url)
    markdown = re.sub(r"\n{3,}", "\n\n", raw_md).strip()

    # Plain text extraction
    scope_copy = BeautifulSoup(str(scope), "html.parser")
    for t in scope_copy(["script", "style", "noscript", "svg", "iframe"]):
        t.decompose()
    plain_text = re.sub(r"\n{2,}", "\n\n", scope_copy.get_text("\n", strip=True)).strip()

    return {
        "title": title,
        "markdown": markdown,
        "text": plain_text,
        "links": links,
        "forms": forms,
    }


def render_headless(url: str, timeout: int = 12) -> str | None:
    """Render page using local headless Chromium/Chrome with SSRF validation."""
    current = (url or "").strip()
    if not current:
        return None
    parts = urlsplit(current)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    try:
        assert_public_host(parts.hostname)
    except Exception as exc:
        log.warning("Headless request blocked by SSRF check for %s: %s", current, exc)
        return None

    candidates = ["chromium", "chromium-browser", "google-chrome", "chrome"]
    chrome_bin: str | None = None
    for name in candidates:
        found = shutil.which(name)
        if found:
            chrome_bin = found
            break
    if not chrome_bin:
        return None

    cmd = [
        chrome_bin,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--dump-dom",
        current,
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout
        return None
    except Exception as exc:
        log.warning("render_headless execution failed for %s: %s", current, exc)
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
    assert_public_host(parts.hostname)

    mode_used = mode
    final_url = current
    html = ""

    if mode == "headless":
        rendered = render_headless(current)
        if rendered:
            html = rendered
            mode_used = "headless"
        else:
            final_url, _ct, encoding, raw = engine._guarded_get(
                current, read_timeout=10, max_bytes=FETCH_MAX_BYTES
            )
            html = raw.decode(encoding or "utf-8", errors="replace")
            mode_used = "http_fallback"
    else:
        final_url, _ct, encoding, raw = engine._guarded_get(
            current, read_timeout=10, max_bytes=FETCH_MAX_BYTES
        )
        html = raw.decode(encoding or "utf-8", errors="replace")
        mode_used = "http"

    parsed = html_to_markdown(
        html,
        base_url=final_url,
        selector=selector,
        include_links=include_links,
    )
    md = parsed["markdown"]
    txt = parsed["text"]
    truncated = len(md) > max_chars

    return {
        "title": parsed["title"] or parts.hostname or "",
        "url": final_url,
        "markdown": md[:max_chars],
        "text": txt[:max_chars],
        "links": parsed["links"],
        "forms": parsed["forms"],
        "truncated": truncated,
        "mode_used": mode_used,
    }


@dataclass
class BrowserTab:
    tab_id: str
    url: str
    title: str
    history: list[str] = field(default_factory=list)
    last_page: dict[str, Any] = field(default_factory=dict)
    opened_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_summary(self, is_active: bool = False) -> dict[str, Any]:
        return {
            "tab_id": self.tab_id,
            "url": self.url,
            "title": self.title,
            "active": is_active,
            "history_len": len(self.history),
        }


class BrowserEngine:
    """Multi-tab browser session manager per conversation."""

    def __init__(self, search_engine: SearchEngine | None = None) -> None:
        self._lock = threading.RLock()
        self._search_engine = search_engine or SearchEngine()
        self._sessions: dict[str, dict[str, Any]] = {}

    def _get_session(self, conv_id: str) -> dict[str, Any]:
        if conv_id not in self._sessions:
            self._sessions[conv_id] = {
                "tabs": {},
                "tab_order": [],
                "active_tab_id": None,
            }
        return self._sessions[conv_id]

    def _list_tabs_unlocked(self, conv_id: str) -> list[dict[str, Any]]:
        sess = self._get_session(conv_id)
        active_id = sess["active_tab_id"]
        result: list[dict[str, Any]] = []
        for tid in sess["tab_order"]:
            if tid in sess["tabs"]:
                result.append(sess["tabs"][tid].to_summary(is_active=(tid == active_id)))
        return result

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
        page = inspect_page(
            url,
            mode=mode,
            selector=selector,
            max_chars=max_chars,
            include_links=include_links,
            search_engine=self._search_engine,
        )
        with self._lock:
            sess = self._get_session(conv)
            tab_id = f"tab_{uuid.uuid4().hex[:6]}"
            tab = BrowserTab(
                tab_id=tab_id,
                url=page["url"],
                title=page["title"],
                history=[page["url"]],
                last_page=page,
                opened_at=time.time(),
                updated_at=time.time(),
            )
            sess["tabs"][tab_id] = tab
            sess["tab_order"].append(tab_id)
            sess["active_tab_id"] = tab_id
            return {
                "ok": True,
                "tab_id": tab_id,
                "page": page,
                "tabs": self._list_tabs_unlocked(conv),
            }

    def list_tabs(self, conversation_id: str | None = None) -> list[dict[str, Any]]:
        conv = conversation_id or "default"
        with self._lock:
            return self._list_tabs_unlocked(conv)

    def switch_tab(self, tab_id: str, conversation_id: str | None = None) -> dict[str, Any]:
        conv = conversation_id or "default"
        with self._lock:
            sess = self._get_session(conv)
            if tab_id not in sess["tabs"]:
                raise ValueError(f"Tab '{tab_id}' does not exist in conversation '{conv}'.")
            sess["active_tab_id"] = tab_id
            tab = sess["tabs"][tab_id]
            return {
                "ok": True,
                "tab_id": tab_id,
                "page": tab.last_page,
                "tabs": self._list_tabs_unlocked(conv),
            }

    def close_tab(self, tab_id: str, conversation_id: str | None = None) -> bool:
        conv = conversation_id or "default"
        with self._lock:
            sess = self._get_session(conv)
            if tab_id not in sess["tabs"]:
                return False
            del sess["tabs"][tab_id]
            if tab_id in sess["tab_order"]:
                sess["tab_order"].remove(tab_id)
            if sess["active_tab_id"] == tab_id:
                sess["active_tab_id"] = sess["tab_order"][-1] if sess["tab_order"] else None
            return True

    def back(
        self,
        conversation_id: str | None = None,
        mode: str = "markdown",
        max_chars: int = 6000,
        include_links: bool = True,
    ) -> dict[str, Any]:
        conv = conversation_id or "default"
        with self._lock:
            sess = self._get_session(conv)
            active_id = sess["active_tab_id"]
            if not active_id or active_id not in sess["tabs"]:
                raise ValueError(f"No active tab open in conversation '{conv}'.")
            tab = sess["tabs"][active_id]
            if len(tab.history) <= 1:
                return {
                    "ok": False,
                    "reason": "Tab has no previous history.",
                    "tab_id": tab.tab_id,
                    "page": tab.last_page,
                }
            tab.history.pop()
            prev_url = tab.history[-1]

        page = inspect_page(
            prev_url,
            mode=mode,
            max_chars=max_chars,
            include_links=include_links,
            search_engine=self._search_engine,
        )

        with self._lock:
            sess = self._get_session(conv)
            if active_id in sess["tabs"]:
                tab = sess["tabs"][active_id]
                tab.url = page["url"]
                tab.title = page["title"]
                tab.last_page = page
                tab.updated_at = time.time()
                return {
                    "ok": True,
                    "tab_id": active_id,
                    "page": page,
                    "tabs": self._list_tabs_unlocked(conv),
                }
            return {"ok": False, "reason": "Active tab was closed during navigation."}

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


def _exec_browser_inspect(args: dict[str, Any], ctx: Any) -> Any:
    from .extensions import tool_result

    action = args.get("action", "inspect")
    conv_id = getattr(ctx, "conversation_id", None)
    url = args.get("url", "")
    mode = args.get("mode", "markdown")
    selector = args.get("selector")
    max_chars = int(args.get("max_chars", 6000))
    include_links = bool(args.get("include_links", True))
    tab_id = args.get("tab_id")

    try:
        if action == "inspect":
            data = inspect_page(
                url,
                mode=mode,
                selector=selector,
                max_chars=max_chars,
                include_links=include_links,
            )
            content = f"Page: {data['title']} ({data['url']})\n\n{data['markdown']}"
            return tool_result(ok=True, content=content, meta=data)
        elif action == "open_tab":
            res = BROWSER.open_tab(
                url,
                conversation_id=conv_id,
                mode=mode,
                selector=selector,
                max_chars=max_chars,
                include_links=include_links,
            )
            page = res["page"]
            content = f"Opened tab {res['tab_id']}: {page['title']} ({page['url']})\n\n{page['markdown']}"
            return tool_result(ok=True, content=content, meta=res)
        elif action == "list_tabs":
            tabs = BROWSER.list_tabs(conversation_id=conv_id)
            lines = [
                f"- Tab {t['tab_id']} ({'active' if t['active'] else 'background'}): {t['title']} ({t['url']})"
                for t in tabs
            ]
            return tool_result(ok=True, content="\n".join(lines) or "No open tabs.", meta={"tabs": tabs})
        elif action == "switch_tab":
            if not tab_id:
                return tool_result(ok=False, content="tab_id is required for switch_tab")
            res = BROWSER.switch_tab(tab_id, conversation_id=conv_id)
            page = res["page"]
            return tool_result(
                ok=True,
                content=f"Switched to tab {tab_id}: {page.get('title', '')}\n\n{page.get('markdown', '')}",
                meta=res,
            )
        elif action == "close_tab":
            if not tab_id:
                return tool_result(ok=False, content="tab_id is required for close_tab")
            closed = BROWSER.close_tab(tab_id, conversation_id=conv_id)
            return tool_result(
                ok=closed,
                content=f"Closed tab {tab_id}." if closed else f"Tab {tab_id} not found.",
                meta={"tab_id": tab_id, "closed": closed},
            )
        elif action == "back":
            res = BROWSER.back(
                conversation_id=conv_id,
                mode=mode,
                max_chars=max_chars,
                include_links=include_links,
            )
            if not res.get("ok"):
                return tool_result(ok=False, content=res.get("reason", "Failed to navigate back"), meta=res)
            page = res["page"]
            return tool_result(
                ok=True,
                content=f"Navigated back to: {page['title']} ({page['url']})\n\n{page['markdown']}",
                meta=res,
            )
        else:
            return tool_result(ok=False, content=f"Unknown browser action: '{action}'")
    except Exception as exc:
        return tool_result(ok=False, content=f"Browser action failed: {exc}")


def _browser_summary(args: dict[str, Any]) -> str:
    act = args.get("action", "inspect")
    target = args.get("url") or args.get("tab_id") or ""
    return f"browser: {act} {target}".strip()


from .extensions import ExtensionTool, schema

TOOLS = (
    ExtensionTool(
        name="browser_inspect",
        description="Inspect and navigate web pages, render headless dynamic DOM, extract structured Markdown, tables, links, and forms.",
        parameters=schema(
            properties={
                "url": {
                    "type": "string",
                    "description": "Web URL to inspect or navigate.",
                },
                "action": {
                    "type": "string",
                    "enum": ["inspect", "open_tab", "list_tabs", "switch_tab", "close_tab", "back"],
                    "description": "Browser action to execute (default: inspect).",
                },
                "mode": {
                    "type": "string",
                    "enum": ["markdown", "headless"],
                    "description": "Fetch mode: 'markdown' for fast SSRF-guarded fetch, 'headless' for dynamic JS rendering.",
                },
                "selector": {
                    "type": "string",
                    "description": "Optional CSS selector to scope content extraction.",
                },
                "tab_id": {
                    "type": "string",
                    "description": "Tab ID for switch_tab or close_tab.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max characters of Markdown content to return.",
                },
                "include_links": {
                    "type": "boolean",
                    "description": "Whether to include markdown links in the output.",
                },
            },
            required=["url"],
        ),
        risk="web",
        phase="web",
        label="Browser Inspect",
        summary=_browser_summary,
        validate=lambda args: args,
        execute=_exec_browser_inspect,
    ),
)

