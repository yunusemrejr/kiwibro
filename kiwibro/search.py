"""Web search providers and SSRF-guarded page fetching (no third-party APIs)."""

from __future__ import annotations

import base64
import ipaddress
import logging
import re
import socket
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any, Callable
from urllib.parse import parse_qs, quote, urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}
PROVIDER_TIMEOUT = 8
SEARCH_CACHE_TTL = 600.0
SEARCH_CACHE_MAX = 128
FETCH_MAX_BYTES = 2 * 1024 * 1024
META_MAX_BYTES = 512 * 1024
MAX_REDIRECTS = 5
TEXT_TYPES = ("text/html", "application/xhtml", "text/plain", "text/markdown", "application/json")


class SearchError(Exception):
    """Raised when a page cannot be fetched safely; message is user-facing."""


# --- SSRF guard -------------------------------------------------------------


def _public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def assert_public_host(hostname: str) -> None:
    """Reject loopback, private, link-local and reserved hosts (DNS re-resolved per hop)."""
    host = (hostname or "").strip().strip("[]").lower()
    if not host:
        raise SearchError("URL has no host")
    if host in ("localhost", "localhost.localdomain") or host.endswith(".localhost"):
        raise SearchError("Refusing to fetch a local address.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror as exc:
            raise SearchError(f"Could not resolve host: {host}") from exc
        addresses = []
        for info in infos:
            try:
                addresses.append(ipaddress.ip_address(info[4][0]))
            except ValueError:
                continue
        if not addresses:
            raise SearchError(f"Could not resolve host: {host}")
        if not all(_public_ip(address) for address in addresses):
            raise SearchError("Refusing to fetch a local or private address.")
        return
    if not _public_ip(address):
        raise SearchError("Refusing to fetch a local or private address.")


# --- small helpers ----------------------------------------------------------


def _strip_html(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", BeautifulSoup(text, "html.parser").get_text(" ", strip=True)).strip()


def _clean(text: str, limit: int = 500) -> str:
    collapsed = re.sub(r"\s+", " ", text or "").strip()
    return collapsed[:limit]


def _decode_bing_url(href: str) -> str:
    """Bing wraps result links in /ck/a?...&u=a1<base64url>; decode when possible."""
    try:
        parsed = urlsplit(href)
    except ValueError:
        return href
    if "bing.com" not in parsed.netloc or "u=" not in parsed.query:
        return href
    values = parse_qs(parsed.query).get("u", [])
    if not values:
        return href
    raw = values[0]
    if raw.startswith("a1"):
        raw = raw[2:]
    try:
        padded = raw + "=" * (-len(raw) % 4)
        decoded = base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
    except (ValueError, TypeError):
        return href
    return decoded if decoded.startswith(("http://", "https://")) else href


def _get(url: str, *, timeout: int = PROVIDER_TIMEOUT, params: dict[str, Any] | None = None) -> requests.Response:
    return requests.get(url, params=params, headers=HEADERS, timeout=(5, timeout))


def _post(url: str, *, data: dict[str, Any], timeout: int = PROVIDER_TIMEOUT) -> requests.Response:
    return requests.post(url, data=data, headers=HEADERS, timeout=(5, timeout), allow_redirects=True)


# --- providers --------------------------------------------------------------


def brave_search(query: str, limit: int) -> list[dict[str, str]]:
    """Brave HTML: div.snippet[data-type=web] with .title / .generic-snippet .content."""
    response = _get("https://search.brave.com/search", params={"q": query, "source": "web"})
    if response.status_code != 200:
        raise SearchError(f"Brave returned HTTP {response.status_code}")
    soup = BeautifulSoup(response.text, "html.parser")
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for snippet in soup.select('div.snippet[data-type="web"]'):
        link = snippet.select_one("a[href^='http']")
        if link is None:
            continue
        url = link.get("href", "")
        if not url.startswith("http") or url in seen:
            continue
        title_el = snippet.select_one(".title")
        title = _clean(title_el.get_text(" ", strip=True) if title_el else link.get_text(" ", strip=True), 200)
        content_el = snippet.select_one(".generic-snippet .content") or snippet.select_one(
            ".snippet-description"
        )
        snippet_text = _clean(content_el.get_text(" ", strip=True) if content_el else "", 500)
        if not title:
            continue
        seen.add(url)
        results.append({"title": title, "url": url, "snippet": snippet_text})
        if len(results) >= limit:
            break
    return results


def _bing_rss(query: str, limit: int) -> list[dict[str, str]]:
    response = _get("https://www.bing.com/search", params={"format": "rss", "q": query, "setlang": "en"})
    if response.status_code != 200:
        return []
    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        return []
    results: list[dict[str, str]] = []
    for item in root.iter("item"):
        title = _clean(item.findtext("title") or "", 200)
        url = _clean(item.findtext("link") or "", 1000)
        description = _clean(item.findtext("description") or "", 500)
        if not title or not url.startswith(("http://", "https://")):
            continue
        results.append({"title": title, "url": url, "snippet": description})
        if len(results) >= limit:
            break
    return results


def bing_search(query: str, limit: int) -> list[dict[str, str]]:
    """Bing HTML (li.b_algo, redirect links via u= base64). Empty when bot-blocked."""
    try:
        response = _get("https://www.bing.com/search", params={"q": query, "setlang": "en", "cc": "us"})
    except requests.RequestException as exc:
        raise SearchError("Bing is not reachable") from exc
    if response.status_code != 200:
        raise SearchError(f"Bing returned HTTP {response.status_code}")
    soup = BeautifulSoup(response.text, "html.parser")
    results: list[dict[str, str]] = []
    for item in soup.select("li.b_algo"):
        link = item.select_one("h2 a[href]")
        if link is None:
            continue
        url = _decode_bing_url(link.get("href", ""))
        if not url.startswith("http"):
            continue
        title = _clean(link.get_text(" ", strip=True), 200)
        description = item.select_one(".b_caption p") or item.select_one("p")
        snippet = _clean(description.get_text(" ", strip=True) if description else "", 500)
        if title:
            results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


def bing_rss_search(query: str, limit: int) -> list[dict[str, str]]:
    """Bing's RSS output: last-resort provider (HTML is frequently bot-blocked).

    Its relevance is noticeably lower than Brave/Wikipedia, so it is tried after the
    documented chain rather than in Bing's place.
    """
    return _bing_rss(query, limit)


def ddg_search(query: str, limit: int) -> list[dict[str, str]]:
    response = _post("https://lite.duckduckgo.com/lite/", data={"q": query, "kl": "wt-wt"})
    if response.status_code != 200:
        raise SearchError(f"DuckDuckGo returned HTTP {response.status_code}")
    page = response.text.lower()
    if "captcha" in page or "anomaly" in page:
        raise SearchError("DuckDuckGo returned a captcha page")
    soup = BeautifulSoup(response.text, "html.parser")
    results: list[dict[str, str]] = []
    for link in soup.select("a.result-link[href]"):
        url = link.get("href", "")
        if not url.startswith(("http://", "https://")):
            continue
        title = _clean(link.get_text(" ", strip=True), 200)
        snippet = ""
        row = link.find_parent("tr")
        if row is not None:
            next_row = row.find_next_sibling("tr")
            if next_row is not None:
                snippet_cell = next_row.select_one(".result-snippet")
                if snippet_cell is not None:
                    snippet = _clean(snippet_cell.get_text(" ", strip=True), 500)
        if title:
            results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


def wikipedia_search(query: str, limit: int) -> list[dict[str, str]]:
    """Wikipedia action API with generator=search + intro extracts (real snippets)."""
    response = _get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": max(1, limit),
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "exchars": 600,
            "format": "json",
            "utf8": 1,
        },
    )
    if response.status_code != 200:
        raise SearchError(f"Wikipedia returned HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise SearchError("Wikipedia returned invalid JSON") from exc
    pages = ((payload.get("query") or {}).get("pages") or {}) if isinstance(payload, dict) else {}
    ordered = sorted(
        (page for page in pages.values() if isinstance(page, dict)),
        key=lambda page: int(page.get("index") or 0),
    )
    results: list[dict[str, str]] = []
    for page in ordered:
        title = str(page.get("title") or "").strip()
        if not title:
            continue
        snippet = _clean(str(page.get("extract") or ""), 500)
        results.append(
            {
                "title": title,
                "url": "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")),
                "snippet": snippet,
            }
        )
        if len(results) >= limit:
            break
    return results


PROVIDERS: tuple[tuple[str, Callable[[str, int], list[dict[str, str]]]], ...] = (
    ("brave", brave_search),
    ("bing", bing_search),
    ("ddg", ddg_search),
    ("wikipedia", wikipedia_search),
    ("bing_rss", bing_rss_search),
)
PROVIDER_COOLDOWN_S = 30.0
FETCH_CACHE_TTL = 300.0
FETCH_CACHE_MAX = 64
BROWSE_MAX_LINKS = 40

_WORDS = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_url(url: str) -> str:
    """Canonical key for deduplicating search results (www/trailing slash/fragment-insensitive)."""
    try:
        parts = urlsplit((url or "").strip())
    except ValueError:
        return (url or "").strip().lower()
    host = (parts.hostname or "").lower().removeprefix("www.")
    path = parts.path.rstrip("/") or "/"
    query = f"?{parts.query}" if parts.query else ""
    return f"{parts.scheme.lower()}://{host}{path}{query}"


def _query_terms(query: str) -> set[str]:
    return {term.lower() for term in _WORDS.findall(query or "") if len(term) > 2}


def rerank_results(query: str, results: list[dict[str, str]]) -> list[dict[str, str]]:
    """Stable re-ranking by query-term overlap (title weighted 3x, snippet 1x)."""
    terms = _query_terms(query)
    if not terms or len(results) < 2:
        return list(results)

    def score(result: dict[str, str]) -> int:
        title_terms = {term.lower() for term in _WORDS.findall(result.get("title") or "")}
        snippet_terms = {term.lower() for term in _WORDS.findall(result.get("snippet") or "")}
        return 3 * len(terms & title_terms) + len(terms & snippet_terms)

    return sorted(results, key=score, reverse=True)


def dedupe_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop duplicate URLs (normalised), keeping the first occurrence."""
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for result in results:
        key = normalize_url(result.get("url") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


class SearchEngine:
    """Provider fallback chain, snippet enrichment and a small in-memory cache."""

    def __init__(self, cache_ttl: float = SEARCH_CACHE_TTL) -> None:
        self._cache: dict[str, tuple[float, list[dict[str, str]]]] = {}
        self._fetch_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._cache_ttl = cache_ttl
        self._cooldowns: dict[str, float] = {}

    def web_search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        query = (query or "").strip()
        if not query:
            raise SearchError("search query is empty")
        max_results = max(1, min(int(max_results or 5), 8))
        key = f"{query.lower()}|{max_results}"
        with self._lock:
            cached = self._cache.get(key)
            if cached and time.monotonic() - cached[0] < self._cache_ttl:
                return [dict(item) for item in cached[1]]
        results: list[dict[str, str]] = []
        now = time.monotonic()
        for name, provider in PROVIDERS:
            if self._cooldowns.get(name, 0.0) > now:
                log.info("search provider %s cooling down after a rate limit", name)
                continue
            try:
                results = provider(query, max_results)
            except (SearchError, requests.RequestException) as exc:
                message = str(exc).lower()
                if "429" in message or "captcha" in message:
                    self._cooldowns[name] = now + PROVIDER_COOLDOWN_S
                log.info("search provider %s unavailable: %s", name, exc)
                continue
            except Exception:  # pragma: no cover - parser drift is non-fatal
                log.warning("search provider %s failed", name, exc_info=True)
                continue
            if results:
                log.info("search provider %s returned %d results", name, len(results))
                results = results[:max_results]
                break
        if results:
            results = self._enrich_snippets(results)
        with self._lock:
            if len(self._cache) >= SEARCH_CACHE_MAX:
                oldest = min(self._cache, key=lambda item: self._cache[item][0])
                self._cache.pop(oldest, None)
            self._cache[key] = (time.monotonic(), [dict(item) for item in results])
        return [dict(item) for item in results]

    def _enrich_snippets(self, results: list[dict[str, str]], limit: int = 3) -> list[dict[str, str]]:
        """Fill empty snippets from page metadata (bounded, SSRF-guarded, 4s each)."""
        filled = 0
        for result in results:
            if filled >= limit:
                break
            if result.get("snippet"):
                continue
            try:
                snippet = self.fetch_snippet(result["url"])
            except (SearchError, requests.RequestException, ValueError):
                continue
            except Exception:  # pragma: no cover - enrichment is best effort
                continue
            if snippet:
                result["snippet"] = snippet
                filled += 1
        return results

    # --- guarded fetching --------------------------------------------------

    def _guarded_get(
        self,
        url: str,
        *,
        read_timeout: int,
        max_bytes: int,
        max_redirects: int = MAX_REDIRECTS,
    ) -> tuple[str, str, str, bytes]:
        """GET with per-hop SSRF checks. Returns (final_url, content_type, encoding, body)."""
        current = (url or "").strip()
        if not current:
            raise SearchError("URL is empty")
        response: requests.Response | None = None
        try:
            for _hop in range(max_redirects + 1):
                parts = urlsplit(current)
                if parts.scheme not in ("http", "https"):
                    raise SearchError("Only http(s) URLs can be fetched.")
                if not parts.hostname:
                    raise SearchError("URL has no host.")
                assert_public_host(parts.hostname)
                try:
                    response = requests.get(
                        current,
                        headers=HEADERS,
                        timeout=(5, read_timeout),
                        allow_redirects=False,
                        stream=True,
                    )
                except requests.RequestException as exc:
                    raise SearchError("The page could not be fetched.") from exc
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    response.close()
                    response = None
                    if not location:
                        raise SearchError("Redirect without a target.")
                    current = urljoin(current, location)
                    continue
                break
            else:
                raise SearchError("Too many redirects.")
            if response is None:  # pragma: no cover - defensive
                raise SearchError("The page could not be fetched.")
            if response.status_code != 200:
                raise SearchError(f"The page returned HTTP {response.status_code}.")
            content_type = (response.headers.get("Content-Type") or "").lower()
            if content_type and not any(kind in content_type for kind in TEXT_TYPES):
                raise SearchError("Only text and HTML pages can be fetched.")
            raw = bytearray()
            for chunk in response.iter_content(chunk_size=65536):
                raw.extend(chunk)
                if len(raw) > max_bytes:
                    break
            encoding = response.encoding or "utf-8"
            return current, content_type, encoding, bytes(raw)
        finally:
            if response is not None:
                response.close()

    @staticmethod
    def _decode(raw: bytes, encoding: str) -> str:
        try:
            return raw.decode(encoding, errors="replace")
        except LookupError:
            return raw.decode("utf-8", errors="replace")

    def fetch_snippet(self, url: str) -> str:
        """Best-effort short description for a search result (meta description / first paragraph)."""
        try:
            _final_url, content_type, encoding, raw = self._guarded_get(
                url, read_timeout=4, max_bytes=META_MAX_BYTES, max_redirects=3
            )
        except SearchError:
            return ""
        if content_type and "html" not in content_type:
            return ""
        soup = BeautifulSoup(self._decode(raw, encoding), "html.parser")
        for meta in soup.find_all("meta"):
            name = (meta.get("name") or meta.get("property") or "").lower()
            if name in ("description", "og:description", "twitter:description"):
                content = _clean(meta.get("content") or "", 400)
                if content:
                    return content
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        first_paragraph = soup.find("p")
        if first_paragraph is not None:
            text = _clean(first_paragraph.get_text(" ", strip=True), 400)
            if len(text) > 40:
                return text
        body = soup.body or soup
        return _clean(body.get_text(" ", strip=True), 400)

    def fetch_page(self, url: str, max_chars: int = 4000) -> dict[str, Any]:
        max_chars = max(200, min(int(max_chars or 4000), 8000))
        cache_key = f"{normalize_url(url)}|{max_chars}"
        with self._lock:
            cached = self._fetch_cache.get(cache_key)
            if cached and time.monotonic() - cached[0] < FETCH_CACHE_TTL:
                return dict(cached[1])
        page = self._fetch_page_uncached(url, max_chars)
        with self._lock:
            if len(self._fetch_cache) >= FETCH_CACHE_MAX:
                oldest = min(self._fetch_cache, key=lambda item: self._fetch_cache[item][0])
                self._fetch_cache.pop(oldest, None)
            self._fetch_cache[cache_key] = (time.monotonic(), dict(page))
        return page

    def _fetch_page_uncached(self, url: str, max_chars: int) -> dict[str, Any]:
        final_url, _content_type, encoding, raw = self._guarded_get(
            url, read_timeout=15, max_bytes=FETCH_MAX_BYTES
        )
        text = self._decode(raw, encoding)
        soup = BeautifulSoup(text, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        for tag in soup(
            ["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "iframe"]
        ):
            tag.decompose()
        root = soup.find("article") or soup.find("main") or soup.body or soup
        extracted = root.get_text("\n", strip=True)
        extracted = re.sub(r"\n{2,}", "\n\n", extracted).strip()
        if not extracted:
            raise SearchError("The page contained no readable text.")
        return {
            "title": _clean(title, 300) or urlsplit(final_url).hostname or "",
            "url": final_url,
            "text": extracted[:max_chars],
            "truncated": len(extracted) > max_chars,
        }

    def browse_page(
        self,
        url: str,
        max_chars: int = 4000,
        max_links: int = BROWSE_MAX_LINKS,
    ) -> dict[str, Any]:
        """Readable text plus a numbered link list, for step-by-step browsing.

        The agent picks a link by number and calls ``fetch_page``/``browse_page``
        on it next. Links are absolute, deduplicated, and same-page anchors are
        dropped. SSRF guarding happens again when a link is actually fetched.
        """
        max_chars = max(200, min(int(max_chars or 4000), 8000))
        max_links = max(1, min(int(max_links or BROWSE_MAX_LINKS), BROWSE_MAX_LINKS))
        final_url, _content_type, encoding, raw = self._guarded_get(
            url, read_timeout=15, max_bytes=FETCH_MAX_BYTES
        )
        soup = BeautifulSoup(self._decode(raw, encoding), "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        links: list[dict[str, str]] = []
        seen: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            if len(links) >= max_links:
                break
            href = str(anchor.get("href") or "").strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            absolute = urljoin(final_url, href)
            if urlsplit(absolute).scheme not in ("http", "https"):
                continue
            key = normalize_url(absolute)
            if key in seen or key == normalize_url(final_url):
                continue
            seen.add(key)
            label = _clean(anchor.get_text(" ", strip=True), 120) or absolute
            links.append({"n": str(len(links) + 1), "text": label, "url": absolute})
        for tag in soup(
            ["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "iframe"]
        ):
            tag.decompose()
        root = soup.find("article") or soup.find("main") or soup.body or soup
        extracted = re.sub(r"\n{2,}", "\n\n", root.get_text("\n", strip=True)).strip()
        if not extracted and not links:
            raise SearchError("The page contained no readable text.")
        return {
            "title": _clean(title, 300) or urlsplit(final_url).hostname or "",
            "url": final_url,
            "text": extracted[:max_chars],
            "truncated": len(extracted) > max_chars,
            "links": links,
        }
