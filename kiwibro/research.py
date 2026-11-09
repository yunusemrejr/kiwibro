"""Parallel web research: multi-query search, batch fetching, evidence packs.

Deterministic and LLM-free: these helpers fan out over
:class:`~kiwibro.search.SearchEngine` with a small thread pool, dedupe and
rerank results, and format compact evidence text with numbered citations the
model can reference. The point is fewer, denser tool rounds for research
tasks instead of one search/fetch per round.
"""

from __future__ import annotations

import logging
import re
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import terms
from .search import SearchEngine, SearchError, dedupe_results, rerank_results

log = logging.getLogger(__name__)

MAX_QUERIES = 4
MAX_URLS = 5
MAX_WORKERS = 4
EVIDENCE_PAGE_CAP = 2500
NO_RESULTS = (
    "No relevant web results were found. Search again with fewer, more specific keywords "
    "(names, versions, error text), or answer from what you know and say it is unverified."
)


# Words that point at something named earlier instead of naming it.
_POINTERS = frozenset("he she him his her hers it its they them their theirs this that these those there".split())
_CAPITAL_WORD = re.compile(r"[^\W\d_][\w'’.-]*|\d[\w.-]*")
_JOINERS = frozenset({"of", "the", "de", "van", "von", "da", "del", "la", "le", "bin", "and"})
_SHORT_REPLY = 300  # a reply this short is an answer whose names the next question can point at


def names_in(text: str) -> list[str]:
    """Proper names in a text, in order: runs of capitalised words ("Barack Obama", "Eiffel
    Tower"). A word capitalised only because it opens a sentence is not a name."""
    found: list[str] = []
    for sentence in re.split(r"(?<=[.!?:;])\s+|\n+", text or ""):
        tokens = _CAPITAL_WORD.findall(sentence)
        run: list[str] = []
        runs: list[list[str]] = []
        for index, token in enumerate(tokens):
            capital = token[0].isupper() and token.lower() not in terms.STOPWORDS
            if capital and index == 0 and not (len(tokens) > 1 and tokens[1][0].isupper()):
                capital = False  # sentence opener
            if capital or (run and token.lower() in _JOINERS):
                run.append(token)
            else:
                runs.append(run)
                run = []
        runs.append(run)
        for run in runs:
            while run and run[-1].lower() in _JOINERS:
                run.pop()
            name = " ".join(run).strip(".'’-")
            if len(name) > 1 and name not in found:
                found.append(name)
    return found


def _subject(message: str, earlier: list[dict[str, Any]]) -> list[str]:
    """What a pointing question is about: the names in the user's own message, else those of
    the question before it, plus the person or place a short answer to it introduced."""
    own = names_in(message)
    if own:
        return own[:2]
    answered = ""
    for item in reversed(earlier[-6:]):
        content = str(item.get("content") or "")
        if item.get("role") == "assistant":
            if not answered and content.strip() and len(content) <= _SHORT_REPLY:
                # One-word capitals in a reply are mostly adjectives ("English"); a full name is two words or more.
                answered = next((name for name in names_in(content) if " " in name), "")
        elif item.get("role") == "user":
            named = names_in(content)[:2]
            if not named and not set(terms.words(content)) & _POINTERS:
                named = [" ".join(terms.content_words(content)[:4])]  # typed without capitals
            named = [name for name in named if name]
            if named:
                return named + ([answered] if answered and answered not in named else [])
    return [answered] if answered else []


def anchor_query(query: str, message: str, earlier: list[dict[str, Any]]) -> str:
    """A follow-up search made to stand alone. Small models search for "how old is he" as
    typed; when the query points at something without naming it, the name the conversation
    is about is added. ``earlier`` holds the messages before ``message``, oldest first."""
    if not set(terms.words(query)) & _POINTERS or len(terms.content_words(query)) > 3:
        return query
    lowered = query.lower()
    missing = [name for name in _subject(message, earlier) if name.lower() not in lowered]
    return f"{query} {' '.join(missing)}" if missing else query


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
        return NO_RESULTS
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        lines.append(f"[{index}] {result.get('title', '')}\n{result.get('url', '')}")
        if result.get("snippet"):
            lines.append(result["snippet"])
        lines.append("")
    return "\n".join(lines).strip()


_RESULT_HEAD = re.compile(r"^(?:\d+\.|\[\d+\]) ")


def drop_seen_snippets(content: str, seen: set[str]) -> str:
    """Search output with the snippets of pages already shown this turn left out.

    Different queries keep returning the same pages; each repeat re-sends its snippet,
    and on a CPU every 500 prompt tokens cost ~15 s of prefill. The title and URL stay
    (they are what gets cited); ``seen`` collects the URLs shown so far."""
    blocks = content.split("\n\n")
    for index, block in enumerate(blocks):
        lines = block.split("\n")
        if len(lines) < 2 or not _RESULT_HEAD.match(lines[0]) or not lines[1].startswith(("http://", "https://")):
            continue
        if lines[1] in seen:
            if len(lines) > 2:
                blocks[index] = f"{lines[0]}\n{lines[1]}\n(already shown above)"
        else:
            seen.add(lines[1])
    return "\n\n".join(blocks)


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
        return {"query": query, "results": [], "pages": [], "evidence": NO_RESULTS}
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

