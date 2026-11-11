"""Parallel web research: multi-query search, batch fetching, evidence packs.

Deterministic and LLM-free: these helpers fan out over
:class:`~kiwibro.search.SearchEngine` with a small thread pool, dedupe and
rerank results, and format compact evidence text with numbered citations the
model can reference. The point is fewer, denser tool rounds for research
tasks instead of one search/fetch per round.
"""

from __future__ import annotations

import logging
import math
import re
import requests
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from . import terms
from .search import SearchEngine, SearchError, dedupe_results, pick_news, rerank_results

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
_NEWS_ASK = re.compile(
    r"\b(?:news\b(?=\W*$|\s*[?.!,;:]|\s+(?:about|on|in|from|for|of|at|around|regarding|today|tonight|this|now|lately|"
    r"recently|yesterday|and|or|pls|plz|please|stories|headlines|updates?|bulletin|summary|roundup|digest|only|so|"
    r"i|we|you|to|is|are|was|has|have|there|here|out|with|that)\b)|headlines|"
    r"(?<!\ba\s)headline|bulletin|breaking|top\s+stories|announc(?:e|es|ed|ing|ements?)|"
    r"what(?:'?s|\s+is)\s+new\b(?=\s*(?:$|[?.!,]|(?:in|with|on|at|about|from|for|today|this|lately|recently)\b)))",
    re.I)
# Asked about the word or the trade, not for the news: "what does the word news mean".
_ABOUT_NEWS = re.compile(r"\b(?:the\s+word|word|term|meaning\s+of|definition\s+of|write|writing|wrote|draft)\b", re.I)
_MINE = re.compile(r"\b(?:my|our|mine|i'?m)\b", re.I)
_LATELY = re.compile(r"\b(?:lately|recently|these\s+days|this\s+(?:month|weekend|year|season)|so\s+far|right\s+now|"
                     r"this\s+(?:mon|tues|wednes|thurs|fri|satur|sun)day)\b", re.I)
_TODAY = re.compile(r"\b(?:today'?s?|tonight'?s?|this\s+(?:morning|evening|afternoon)|so\s+far\s+today)\b", re.I)
_YESTERDAY = re.compile(r"\b(?:yesterday'?s?|last\s+night|past\s+(?:two|2)\s+days|last\s+(?:two|2)\s+days|48\s+hours)\b", re.I)
_HAPPENED = re.compile(r"\bhappen(?:ed|ing|s)?\b", re.I)
# A question about how things stand now, asked without the word "news" ("what is going on
# in the AI world", "any updates on the strike", "what's the latest on Mars"): only dated
# reports answer it.
_GOING_ON = re.compile(
    r"\b(?:what(?:'?s|\s+is|\s+has\s+been)?\s+(?:going\s+on|happening)|what\s+went\s+on|"
    r"what(?:'?s|\s+is)\s+the\s+latest(?=\s+(?:on|in|about|from|with)\b|\s*[?.!]*$)|"
    r"(?:going\s+on|happening)\s+(?:in|with|at|around)|(?:the\s+)?latest\s+(?:on|in|about|from|with)|"
    r"any\s+(?:updates?|developments?)|(?:updates?|developments?)\s+(?:on|in|about|with)|"
    r"catch\s+me\s+up|bring\s+me\s+up\s+to\s+(?:date|speed))\b",
    re.I,
)
# "in the AI world", "the world of chess", "in the crypto space": the field named is the
# topic, the frame around it is not.
_WORLD_OF = re.compile(r"\b(?:(?:in|on|around|across|from|about)\s+)?the\s+world\s+of\b", re.I)
_FIELD_FRAME = re.compile(
    r"\b(?:in|on|around|across|from)\s+the\s+(?P<field>[\w.+#&'-]{2,30}(?:\s+[\w.+#&'-]{2,30})?)\s+"
    r"(?:world|scene|space|industry|sector|field|community)\b",
    re.I,
)
# What the user has in front of them is not news: "what is going on in my code", "… in this function".
_OWN_THING = re.compile(r"\b(?:my|our|this|that|these|those|here|i|i'?m|i'?ve|we|me|us|it|its|your|you)\b", re.I)
_DAY = re.compile(r"\b(?:(?:past|last)\s+(?:day|24\s+hours))\b", re.I)
_WEEK = re.compile(r"\b(?:this|past|last)\s+(?:week|7\s+days|few\s+days)\b", re.I)
_NEWS_FILLER = re.compile(
    r"\b(?:news|headlines?|bulletin|breaking|announc(?:e|es|ed|ing|ements?)|latest|newest|recent|top|current|any|the|some|"
    r"about|on|of|for|in|from|only|and|or|me|please|stories|updates?|developments?|what'?s|what\s+is|what|new|happened|"
    r"explain|why|how|does|do|did|is|are|was|were|it|that|this|matters?|summari[sz]e|tell|give|show|list|get|"
    r"brief|catch|up|bring|i|my|you|can|could|would|need|want|to|a|an|search|look|find|google|"
    r"going|happening|has|been|with|around|at|lately|currently|right\s+now|these\s+days|date|speed|"
    r"went|there|pls|plz|whats|top\s+stories|world)\b",
    re.I,
)


@dataclass(frozen=True)
class NewsAsk:
    topic: str  # what to search the feeds for ("" = top stories)
    days: int  # how far back the feeds look
    since: datetime | None  # nothing older is listed (local time)
    window: str  # in words, for the model ("today", "today and yesterday", "the past week", "")


def news_request(query: str, now: datetime | None = None, *, message: bool = False) -> NewsAsk | None:
    """The news a search asks for, with its time window; None for a search that is not about news.
    A search query that says "news" is after news; in a whole ``message`` the word has to be what
    is asked for ("news team" and "news programmes" name other things)."""
    text = (query or "").replace("\u2019", "'")
    dated = bool(_TODAY.search(text) or _YESTERDAY.search(text) or _DAY.search(text) or _WEEK.search(text))
    # "this week" and "tell me" are not the user's own thing; "my code" and "this function" are.
    timeless = text
    for pattern in (_TODAY, _YESTERDAY, _DAY, _WEEK, _LATELY, _NEWS_REQUEST_START):
        timeless = pattern.sub(" ", timeless)
    timeless = re.sub(r"\b(?:tell|give|show)\s+me\b|\bfor\s+me\b|\bcan\s+you\b|\bcould\s+you\b", " ", timeless, flags=re.I)
    own = bool(_OWN_THING.search(_GOING_ON.sub(" ", timeless)))
    going_on = bool(_GOING_ON.search(text)) and not own
    happened = bool(_HAPPENED.search(text)) and (dated or bool(_LATELY.search(text))) and not own
    # "news" before another word of its own ("news team", "news programmes") names a thing, unless a
    # day is asked for ("why today's ai news matters").
    named = bool(_NEWS_ASK.search(text) or ((dated or not message) and re.search(r"\bnews\b", text, re.I))) \
        and not _ABOUT_NEWS.search(text) and not (_MINE.search(text) and not dated)
    if not named and not going_on and not happened:
        return None
    text = _FIELD_FRAME.sub(lambda found: f" {found.group('field')} ", _WORLD_OF.sub(" ", text))
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
    topic = _NEWS_FILLER.sub(" ", _LATELY.sub(" ", _WEEK.sub(" ", _DAY.sub(" ", _YESTERDAY.sub(" ", _TODAY.sub(" ", text))))))
    topic = " ".join(word for word in re.sub(r"[^\w.+#&'-]+", " ", topic).split()
                     if any(ch.isalnum() for ch in word)).strip(" '-.")
    if not topic and not dated and not named:
        return None  # "what is going on?" alone asks about the conversation, not the world
    return NewsAsk(topic, days, since, window)


_NEWS_REQUEST_START = re.compile(
    r"^\s*(?:(?:please|hey|ok|okay|so)\W+)*(?:summari[sz]e|explain|tell|give|show|list|what|whats|what's|any|get|"
    r"brief|catch|bring|read|find|search|look|check|fetch|can|could|would|will|i\s+(?:want|need)|let'?s|"
    r"are\s+there|is\s+there|has\s+there|have\s+there)\b",
    re.IGNORECASE,
)


def asks_for_news(message: str) -> bool:
    """The message is a request for news (not a remark that mentions it: "I read the news today")."""
    text = message.strip()
    if news_request(text, message=True) is None:
        return False
    # "global ai news today. find news": the request can be any sentence of the message.
    if text.endswith("?") or any(_NEWS_REQUEST_START.match(part) for part in re.split(r"(?<=[.!?;])\s+", text)):
        return True
    # "tech headlines for today pls", "latest news in the ai world": a bare phrase that names the news
    # asks for it. "good news, I got the job" tells some.
    return bool(_NEWS_PHRASE.match(text)) and not _TELLS.search(text)


_NEWS_PHRASE = re.compile(
    r"^\W*(?!(?:good|great|bad|sad|big|huge|exciting|amazing|terrible|awful|fake|old|no|some|sorry|happy|wonderful)\b)"
    r"(?:[\w'.+#&-]+\s+){0,4}(?:news|headlines|stories)\b", re.I)
_TELLS = re.compile(r"\b(?:i|we|he|she|they)\s+(?:got|have|had|am|was|were|just|heard|read|saw|watched|is|are)\b|\b(?:i'?m|we'?re)\b",
                    re.I)


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


NOT_SHOWN_BEFORE = "These were not in the answer before."
NOTHING_NEWER = "A new search just now found no other stories: these are the newest dated reports the news feeds have."
# A request done again is also searched in other words: the feeds answer each wording with other stories.
REWORDINGS = ("{topic} news", "latest {topic} developments")


def again(engine: SearchEngine, ask: NewsAsk, items: list[dict[str, str]], note: str, limit: int,
          shown_before: str = "", must_mention: tuple[str, ...] = ()) -> tuple[list[dict[str, str]], str]:
    """News for a request that is being done again because the user turned the answer down:
    stories that stay on the request's subject, and ones not yet shown instead of the same list.
    ``items`` and ``note`` are what the plain search gave."""
    def on_subject(found: list[dict[str, str]]) -> list[dict[str, str]]:
        return [item for item in found
                if any(word in f"{item.get('title', '')} {item.get('snippet', '')}".lower() for word in must_mention)]

    def unseen(found: list[dict[str, str]]) -> list[dict[str, str]]:
        return [item for item in found if item.get("title", "").replace("[", "(").replace("]", ")") not in shown_before]

    pool_of = getattr(engine, "news_pool", None)
    since = ask.since
    if since is not None and "past 24 hours" in note:  # the plain search already had to look further back
        since = min(since, datetime.now().astimezone() - timedelta(days=1))

    def pool(topic: str) -> list[dict[str, str]]:
        if pool_of is None:
            return engine.news_search(topic, limit * 3, days=ask.days, since=since)
        return pool_of(topic, days=ask.days, since=since)

    if must_mention:
        kept = on_subject(pool(ask.topic)) or on_subject(items)
        return (pick_news(kept, limit), note) if kept else (items[:limit], note)
    if not shown_before:
        return items[:limit], note
    found: dict[str, dict[str, str]] = {}
    for wording in ("{topic}", *(REWORDINGS if ask.topic else ())):
        try:
            more = pool(wording.format(topic=ask.topic).strip())
        except Exception:  # one wording failing must not lose the others
            log.info("news search in other words failed", exc_info=True)
            continue
        for item in unseen(more):
            found.setdefault(item.get("title", ""), item)
        if len(found) >= limit:
            break
    fresh = sorted(found.values(), key=lambda item: item.get("published") or "", reverse=True)
    if len(fresh) >= NEWS_MIN_ITEMS:
        return pick_news(fresh, limit), " ".join(filter(None, (note, NOT_SHOWN_BEFORE)))
    return items[:limit], " ".join(filter(None, (note, NOTHING_NEWER)))


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
WEATHER_HEAD = "Weather, from forecast data"


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


# --- reading the pages a search found ------------------------------------------
# A result's snippet is a line or two chosen by the search engine; the answer is usually
# further down the page. Kiwibro opens the best pages itself and hands the model the
# passages that bear on the question, so a small model has the facts in front of it
# instead of a headline to guess from.
DEEP_PAGES = 3  # pages opened for one search
DEEP_DEADLINE_S = 9.0  # all of them together; a page that is slower is left out
DEEP_PAGE_CHARS = 8000
PASSAGE_CHARS = 360
PASSAGES_PER_PAGE = 2
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"“(\[]?[A-Z0-9])|\n+")


def passages(query: str, text: str, limit: int = PASSAGES_PER_PAGE, chars: int = PASSAGE_CHARS) -> list[str]:
    """The parts of a page that bear on a query, best first.

    The page is cut into sentences; each window of neighbouring sentences is scored by the
    query's words it holds, a rare word counting for more than one found everywhere on the
    page, and a window with a figure in it slightly ahead of one without. Windows that hold
    none of the query's words are never returned."""
    wanted = {terms.stem(word) for word in terms.content_words(query)}
    sentences = [" ".join(part.split()) for part in _SENTENCE_END.split(text or "")]
    sentences = [part for part in sentences if len(part) >= 25]
    if not wanted or not sentences:
        return []
    held = [{terms.stem(word) for word in terms.words(part)} & wanted for part in sentences]
    weight = {word: math.log(1.0 + len(sentences) / (1.0 + sum(word in found for found in held))) for word in wanted}
    windows: list[tuple[float, int, int]] = []
    for start in range(len(sentences)):
        end, size, found = start, 0, set()
        while end < len(sentences) and (end == start or size + len(sentences[end]) + 1 <= chars):
            size += len(sentences[end]) + 1
            found |= held[end]
            end += 1
        if not found:
            continue
        score = sum(weight[word] for word in found) * (1.1 if any(ch.isdigit() for ch in " ".join(sentences[start:end])) else 1.0)
        windows.append((score, start, end))
    picked: list[tuple[int, int]] = []
    for _score, start, end in sorted(windows, key=lambda item: (-item[0], item[1])):
        if len(picked) >= limit:
            break
        if all(end <= other_start or start >= other_end for other_start, other_end in picked):
            picked.append((start, end))
    out = []
    for start, end in picked:
        passage = " ".join(sentences[start:end])
        out.append(passage if len(passage) <= chars else passage[: chars - 1].rsplit(" ", 1)[0] + "…")
    return out


def broader(query: str) -> str:
    """A query cut down to the words that carry it, for a search that found too little."""
    kept = [word for word in terms.content_words(query)]
    return " ".join(dict.fromkeys(kept))[:160]


def read_results(
    engine: SearchEngine, query: str, results: list[dict[str, str]], *,
    depth: int = DEEP_PAGES, deadline: float = DEEP_DEADLINE_S,
) -> list[dict[str, str]]:
    """Open the best results and attach to each the passages that bear on the query
    (``passage``: one string, passages joined by " … "). Results come back in their order; a
    page that cannot be read in time keeps its snippet and nothing else."""
    targets = [result for result in results if str(result.get("url") or "").startswith(("http://", "https://"))][:depth]
    if not targets:
        return results
    pool = ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(targets)), thread_name_prefix="kiwibro-read")
    try:
        futures = {pool.submit(engine.fetch_page, result["url"], DEEP_PAGE_CHARS): result for result in targets}
        done, _late = wait(futures, timeout=deadline)
        for future in done:
            try:
                page = future.result()
            except Exception:  # an unreadable page: its snippet stays
                continue
            found = passages(query, str(page.get("text") or ""))
            if found:
                futures[future]["passage"] = " … ".join(found)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return results


THIN_RESULTS = 3  # fewer than this: the question's own wording was too narrow for the index


def search(engine: SearchEngine, query: str, max_results: int = 5, *, deep: bool = False) -> list[dict[str, str]]:
    """A web search that does not give up on a thin answer: when the query as asked finds
    too little, its key words are searched as well and the two lists are merged. With
    ``deep`` the best pages are opened and their passages attached (see read_results)."""
    results = engine.web_search(query, max_results)
    if deep and len(results) < THIN_RESULTS:
        wider = broader(query)
        if wider and wider.lower() != query.lower().strip():
            try:
                results = dedupe_results([*results, *engine.web_search(wider, max_results)])[:max_results]
            except SearchError:
                pass
    return read_results(engine, query, results) if deep and results else results


def format_read_results(results: list[dict[str, str]]) -> str:
    """Numbered results in the web_search layout, with what was read on each page under it."""
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.get('title', '')}\n{result.get('url', '')}")
        if result.get("snippet"):
            lines.append(result["snippet"])
        if result.get("passage"):
            lines.append(f"On the page: {result['passage']}")
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



def weather_report(engine: Any, query: str) -> str:
    """The weather a search asks about, read from forecast data instead of web pages, which hold
    their figures in scripts. "" when the search is about no place's weather, or the data is not there."""
    look = getattr(engine, "weather", None)
    try:
        report = look(query) if callable(look) else ""
    except Exception:  # a convenience; the web search runs instead
        log.warning("the weather could not be looked up", exc_info=True)
        return ""
    return f"{WEATHER_HEAD}.\n\n{report}" if isinstance(report, str) and report else ""
