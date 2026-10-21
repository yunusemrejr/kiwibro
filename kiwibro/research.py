"""Parallel web research: multi-query search, batch fetching, evidence packs.

Deterministic and LLM-free: these helpers fan out over
:class:`~kiwibro.search.SearchEngine` with a small thread pool, dedupe and
rerank results, and format compact evidence text with numbered citations the
model can reference. The point is fewer, denser tool rounds for research
tasks instead of one search/fetch per round.
"""

from __future__ import annotations

import logging
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .search import SearchEngine, SearchError, dedupe_results, rerank_results

log = logging.getLogger(__name__)

MAX_QUERIES = 4
MAX_URLS = 5
MAX_WORKERS = 4
EVIDENCE_PAGE_CAP = 2500


def multi_search(
    engine: SearchEngine,
    queries: list[str],
    *,
    max_results: int = 5,
) -> dict[str, Any]:
    """Run several searches in parallel; merge, dedupe and rerank per query."""
    cleaned = [str(query).strip() for query in queries if str(query or "").strip()]
    if not cleaned:
        raise SearchError("no search queries were given")
    cleaned = cleaned[:MAX_QUERIES]
    max_results = max(1, min(int(max_results or 5), 8))

    def run(query: str) -> tuple[str, list[dict[str, str]]]:
        try:
            return query, engine.web_search(query, max_results)
        except (SearchError, requests.RequestException) as exc:
            log.info("multi_search query failed (%s): %s", query, exc)
            return query, []

    merged: list[dict[str, str]] = []
    per_query: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(cleaned))) as pool:
        for query, results in pool.map(run, cleaned):
            ranked = rerank_results(query, results)
            per_query[query] = len(ranked)
            for result in ranked:
                item = dict(result)
                item["query"] = query
                merged.append(item)
    unique = dedupe_results(merged)
    return {"queries": cleaned, "results": unique, "count": len(unique), "per_query": per_query}


def fetch_pages(
    engine: SearchEngine,
    urls: list[str],
    *,
    max_chars: int = EVIDENCE_PAGE_CAP,
) -> dict[str, Any]:
    """Fetch several pages in parallel; failures become error entries, not drops."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for url in urls:
        value = str(url or "").strip()
        if value and value not in seen:
            seen.add(value)
            cleaned.append(value)
    if not cleaned:
        raise SearchError("no URLs were given")
    cleaned = cleaned[:MAX_URLS]
    max_chars = max(200, min(int(max_chars or EVIDENCE_PAGE_CAP), 8000))

    def run(url: str) -> dict[str, Any]:
        try:
            page = engine.fetch_page(url, max_chars)
            return {"ok": True, **page}
        except (SearchError, requests.RequestException, ValueError) as exc:
            return {"ok": False, "url": url, "title": "", "text": "", "error": str(exc)}

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(cleaned))) as pool:
        pages = list(pool.map(run, cleaned))
    return {
        "pages": pages,
        "count": len(pages),
        "ok_count": sum(1 for page in pages if page.get("ok")),
    }


def format_search_results(results: list[dict[str, str]]) -> str:
    """Numbered result list (title, URL, snippet) for tool output."""
    if not results:
        return "No web results were found."
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        lines.append(f"[{index}] {result.get('title', '')}\n{result.get('url', '')}")
        if result.get("snippet"):
            lines.append(result["snippet"])
        lines.append("")
    return "\n".join(lines).strip()


def format_evidence(pages: list[dict[str, Any]]) -> str:
    """Numbered evidence pack: one section per fetched source."""
    if not pages:
        return "No pages could be fetched."
    lines: list[str] = []
    for index, page in enumerate(pages, start=1):
        if not page.get("ok"):
            lines.append(f"[{index}] (unavailable) {page.get('url', '')} — {page.get('error', 'failed')}")
            lines.append("")
            continue
        lines.append(f"[{index}] {page.get('title') or page.get('url', '')}")
        lines.append(str(page.get("url", "")))
        lines.append(str(page.get("text") or "").strip())
        lines.append("")
    return "\n".join(lines).strip()


def web_research(
    engine: SearchEngine,
    query: str,
    *,
    depth: int = 3,
    max_results: int = 6,
) -> dict[str, Any]:
    """One-call research: search, rerank, fetch the top sources, pack evidence."""
    query = (query or "").strip()
    if not query:
        raise SearchError("research query is empty")
    depth = max(1, min(int(depth or 3), MAX_URLS))
    max_results = max(1, min(int(max_results or 6), 8))
    results = dedupe_results(rerank_results(query, engine.web_search(query, max_results)))
    if not results:
        return {"query": query, "results": [], "pages": [], "evidence": "No web results were found."}
    top_urls = [result["url"] for result in results[:depth] if result.get("url")]
    fetched = fetch_pages(engine, top_urls, max_chars=EVIDENCE_PAGE_CAP)
    evidence = (
        f"Search results for “{query}”:\n{format_search_results(results)}\n\n"
        f"Fetched sources:\n{format_evidence(fetched['pages'])}"
    )
    return {
        "query": query,
        "results": results,
        "pages": fetched["pages"],
        "evidence": evidence,
    }

