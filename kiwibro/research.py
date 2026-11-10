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
from dataclasses import dataclass
from datetime import datetime, timedelta
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


# --- news -------------------------------------------------------------------
# A news question goes to dated news feeds instead of a web index: a web index answers
# "today's AI news" with evergreen pages, and a small model then fills the gap from memory.
_NEWS_ASK = re.compile(r"\b(?:news|headlines?|bulletin|breaking|what(?:'s|\s+is)\s+new|announcements?|announced)\b", re.I)
_TODAY = re.compile(r"\b(?:today'?s?|tonight'?s?|this\s+(?:morning|evening|afternoon)|so\s+far\s+today)\b", re.I)
_YESTERDAY = re.compile(r"\b(?:yesterday'?s?|last\s+night|past\s+(?:two|2)\s+days|last\s+(?:two|2)\s+days|48\s+hours)\b", re.I)
_HAPPENED = re.compile(r"\bhappen(?:ed|ing|s)?\b", re.I)
_DAY = re.compile(r"\b(?:(?:past|last)\s+(?:day|24\s+hours))\b", re.I)
_WEEK = re.compile(r"\b(?:this|past|last)\s+(?:week|7\s+days|few\s+days)\b", re.I)
_NEWS_FILLER = re.compile(
    r"\b(?:news|headlines?|bulletin|breaking|announcements?|latest|newest|recent|top|current|any|the|some|"
    r"about|on|of|for|in|from|only|and|or|me|please|stories|updates?|developments?|what'?s|what\s+is|what|new|happened|"
    r"explain|why|how|does|do|did|is|are|was|were|it|that|this|matters?|summari[sz]e|tell|give|show|list|get|"
    r"brief|catch|up|bring|i|my|you|can|could|would|need|want|to|a|an|search|look|find|google)\b",
    re.I,
)


@dataclass(frozen=True)
class NewsAsk:
    topic: str  # what to search the feeds for ("" = top stories)
    days: int  # how far back the feeds look
    since: datetime | None  # nothing older is listed (local time)
    window: str  # in words, for the model ("today", "today and yesterday", "the past week", "")


def news_request(query: str, now: datetime | None = None) -> NewsAsk | None:
    """The news a search asks for, with its time window; None for a search that is not about news."""
    text = (query or "").replace("\u2019", "'")
    if not _NEWS_ASK.search(text) and not (_HAPPENED.search(text) and (_TODAY.search(text) or _YESTERDAY.search(text))):
        return None
    now = (now or datetime.now()).astimezone()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if _YESTERDAY.search(text):
        days, since, window = 2, midnight - timedelta(days=1), "today and yesterday" if _TODAY.search(text) else "since yesterday"
    elif _TODAY.search(text):
        days, since, window = 1, midnight, "today"
    elif _DAY.search(text):
        days, since, window = 1, now - timedelta(days=1), "the past 24 hours"
    elif _WEEK.search(text):
        days, since, window = 7, now - timedelta(days=7), "the past week"
    else:
        days, since, window = 7, None, ""
    topic = _NEWS_FILLER.sub(" ", _WEEK.sub(" ", _DAY.sub(" ", _YESTERDAY.sub(" ", _TODAY.sub(" ", text)))))
    topic = " ".join(re.sub(r"[^\w.+#&'-]+", " ", topic).split()).strip(" '-")
    return NewsAsk(topic, days, since, window)


_NEWS_REQUEST_START = re.compile(
    r"^\s*(?:(?:please|hey|ok|okay|so)\W+)*(?:summari[sz]e|explain|tell|give|show|list|what|whats|what's|any|get|"
    r"brief|catch|bring|read|find|search|look|check|fetch|can|could|would|will|i\s+(?:want|need)|let'?s)\b",
    re.IGNORECASE,
)


def asks_for_news(message: str) -> bool:
    """The message is a request for news (not a remark that mentions it: "I read the news today")."""
    text = message.strip()
    return news_request(text) is not None and (text.endswith("?") or bool(_NEWS_REQUEST_START.match(text)))


def with_time_words(query: str, message: str) -> str:
    """A news query with the time window the rest of the message gives ("… today and yesterday
    only"), so the feeds list only those days."""
    if news_request(query) is None:
        return query
    extra = []
    for pattern in (_TODAY, _YESTERDAY, _DAY, _WEEK):
        found = pattern.search(message or "")
        if found and not pattern.search(query):
            extra.append(found.group(0))
    return " ".join([query, *extra])


NEWS_ITEMS = 8  # a bulletin needs more than the five a web search returns
NEWS_MIN_ITEMS = 3  # fewer than this inside "today": show the past 24 hours, and say so


def news_search(engine: SearchEngine, ask: NewsAsk, limit: int = 8) -> tuple[list[dict[str, str]], str]:
    """(items newest first, a line saying which window they cover)."""
    items = engine.news_search(ask.topic, limit, days=ask.days, since=ask.since)
    note = f"Only items from {ask.window} are listed." if ask.window else ""
    if ask.since is not None and len(items) < NEWS_MIN_ITEMS and ask.days <= 2:
        wider_since = min(ask.since, datetime.now().astimezone() - timedelta(days=1))
        wider = engine.news_search(ask.topic, limit, days=max(ask.days, 1) + 1, since=wider_since)
        if len(wider) > len(items):
            note = (f"Only {len(items)} {'story is' if len(items) == 1 else 'stories are'} dated {ask.window}, "
                    "so the past 24 hours are listed." if items else
                    f"Nothing is dated {ask.window} yet, so the past 24 hours are listed.")
            items = wider
    return items, note


def _moment(published: str, now: datetime) -> datetime | None:
    try:
        return datetime.fromisoformat(published).astimezone(now.tzinfo)
    except (TypeError, ValueError):
        return None


def _day_heading(day: Any, now: datetime) -> str:
    named = f"{day:%A} {day.day} {day:%B}"
    if day == now.date():
        return f"Today, {named}"
    if day == (now - timedelta(days=1)).date():
        return f"Yesterday, {named}"
    return named


SUMMARY_CHARS = 220


def _first_sentence(text: str) -> str:
    text = " ".join(text.split()).rstrip(". ")
    cut = re.search(r"(?<=[.!?])\s", text)
    text = (text[: cut.start()] if cut and cut.start() >= 40 else text).rstrip(". ")
    return text if len(text) <= SUMMARY_CHARS else text[: SUMMARY_CHARS - 1].rsplit(" ", 1)[0] + "…"


# The model only adds something to a bulletin when the message asks for more than the news
# itself: an explanation, a comparison, a shorter or translated version.
_NEWS_MORE = re.compile(
    r"\b(why|how|explain\w*|compar\w*|analy[sz]\w*|opinion|think|mean\w*|impact|affect\w*|summar\w*|"
    r"short\w*|brief\w*|top \d+|most important|biggest|translat\w*|in (?!the\b)[a-z]+ish|in [a-z]+an)\b",
    re.IGNORECASE,
)


def bulletin_is_the_answer(message: str) -> bool:
    """The message asks for the news and nothing more, in English: the finished bulletin is the
    reply, word for word, so no model can pad it with stories or details it made up."""
    letters = [ch for ch in message if ch.isalpha()]
    foreign = sum(1 for ch in letters if not ch.isascii())
    return not _NEWS_MORE.search(message) and foreign * 20 <= len(letters)


NEWS_HEAD = "News bulletin, newest first"


def format_news(items: list[dict[str, str]], note: str = "", now: datetime | None = None) -> str:
    """News as a finished bulletin: grouped by day, each story with its outlet, time and (when the
    feed gave one) its first sentence and link. Small models copy a bulletin faithfully; asked to
    compose one from raw results they pad headlines with made-up detail and repeat stories."""
    if not items:
        return NO_NEWS
    now = (now or datetime.now()).astimezone()
    head = f"{NEWS_HEAD} (it is now {now:%A} {now.day} {now:%B %Y}, {now:%H:%M} local time)."
    lines = [" ".join(filter(None, (head, note)))]
    day = None
    for item in items:
        moment = _moment(item.get("published", ""), now)
        if (moment.date() if moment else None) != day or not lines[1:]:
            day = moment.date() if moment else None
            lines.extend(("", f"**{_day_heading(day, now) if day else 'Date unknown'}**"))
        title = item.get("title", "").replace("[", "(").replace("]", ")")
        url = item.get("url", "")
        headline = f"[{title}]({url})" if url and "news.google.com/" not in url else title
        where = ", ".join(filter(None, (item.get("source", ""), f"{moment:%H:%M}" if moment else "")))
        summary = f" {_first_sentence(item['snippet'])}." if item.get("snippet") else ""
        lines.append(f"- {headline} ({where}).{summary}")
    return "\n".join(lines).strip()


NO_NEWS = (
    "No news items were found for this. Say so plainly; do not list stories from memory. Searching again "
    "with a broader topic (fewer words) may help."
)


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

