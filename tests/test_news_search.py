"""News questions go to dated news feeds, with the time window the user gave."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from kiwibro import research, search, tools
from kiwibro.sandbox import Workspace

NOW = datetime(2026, 11, 11, 0, 8, tzinfo=timezone(timedelta(hours=3)))  # just after midnight in Istanbul


@pytest.mark.parametrize(
    ("query", "topic", "window", "since_day"),
    [
        ("todays ai news yesterday", "ai", "today and yesterday", 10),
        ("todays ai news", "ai", "today", 11),
        ("yesterday's headlines about the election", "election", "since yesterday", 10),
        ("AI news this week", "AI", "the past week", 4),
        ("latest news about nvidia", "nvidia", "", None),
        ("today's news", "", "today", 11),
        ("what happened in turkey today", "turkey", "today", 11),
    ],
)
def test_a_news_query_reads_its_topic_and_time_window(query: str, topic: str, window: str, since_day: int | None):
    ask = research.news_request(query, NOW)
    assert ask is not None and ask.topic == topic and ask.window == window
    assert (ask.since.day if ask.since else None) == since_day


@pytest.mark.parametrize("query", ["weather istanbul", "tallest building in Turkey", "python dataclass default list"])
def test_other_searches_are_not_news(query: str):
    assert research.news_request(query, NOW) is None


def test_the_time_window_elsewhere_in_the_message_joins_a_news_query():
    message = "search for todays ai news and tell be as bulletin. today and yesterday only"
    assert research.with_time_words("todays ai news", message) == "todays ai news yesterday"
    assert research.with_time_words("weather istanbul", "weather istanbul today") == "weather istanbul"


def _item(title: str, hours_ago: float, snippet: str = "", source: str = "Reuters") -> dict[str, str]:
    published = (NOW - timedelta(hours=hours_ago)).astimezone(timezone.utc).isoformat()
    return {"title": title, "url": "https://example.org/" + title[:5], "snippet": snippet, "source": source,
            "published": published}


def test_merging_feeds_keeps_one_copy_per_story_newest_first_and_inside_the_window():
    google = [_item("Nvidia selloff deepens, adding to AI jitters", 1), _item("Old story about chips", 60)]
    bing = [_item("Nvidia selloff deepens, adding to AI jitters", 1, snippet="Shares fell 4%."),
            _item("Mercor valued at $20 billion", 5)]
    merged = search.merge_news([google, bing], since=NOW - timedelta(days=1))
    assert [item["title"] for item in merged] == ["Nvidia selloff deepens, adding to AI jitters",
                                                  "Mercor valued at $20 billion"]
    assert merged[0]["snippet"] == "Shares fell 4%."  # the copy with a summary wins


def test_news_is_a_finished_bulletin_grouped_by_day_and_feed_redirects_are_left_out():
    items = [{**_item("Nvidia selloff deepens", 0.05), "url": "https://news.google.com/rss/articles/CBMi"},
             _item("Mercor valued at $20 billion", 5, snippet="A funding round. It was led by two firms."),
             _item("Old chip story", 50)]
    text = research.format_news(items, "Only items from today and yesterday are listed.", NOW)
    assert text.startswith(f"{research.NEWS_HEAD} (it is now Wednesday 11 November 2026, 00:08 local time).")
    assert "**Today, Wednesday 11 November**\n- Nvidia selloff deepens (Reuters, 00:05)." in text
    assert ("**Yesterday, Tuesday 10 November**\n- [Mercor valued at $20 billion](https://example.org/Merco) "
            "(Reuters, 19:08). A funding round. It was led by two firms.") in text
    assert "**Sunday 8 November**" in text and "news.google.com" not in text
    assert text.count("Mercor") == 1


def test_a_quiet_today_widens_to_the_past_day_and_says_so():
    class Feeds:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def news_search(self, topic, limit, *, days=None, since=None):
            self.calls.append((topic, days))
            return [] if len(self.calls) == 1 else [_item("Late story", 2), _item("Later story", 1)]

    feeds = Feeds()
    ask = research.news_request("todays ai news", datetime.now().astimezone())
    items, note = research.news_search(feeds, ask)
    assert len(items) == 2 and note.startswith("Nothing is dated today yet, so the past 24 hours are listed")


RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>AI startup raises funds - Reuters</title><link>https://news.google.com/rss/articles/x</link>
<pubDate>Tue, 10 Nov 2026 17:29:28 GMT</pubDate><source url="https://reuters.com">Reuters</source></item>
</channel></rss>"""


def test_google_news_items_are_parsed_without_the_outlet_suffix(monkeypatch: pytest.MonkeyPatch):
    class Response:
        status_code = 200
        content = RSS

    seen: dict = {}

    def get(url, *, timeout=search.PROVIDER_TIMEOUT, params=None):
        seen.update(url=url, params=params)
        return Response()

    monkeypatch.setattr(search, "_get", get)
    [item] = search.google_news("ai", 10, days=2)
    assert item["title"] == "AI startup raises funds" and item["source"] == "Reuters"
    assert item["published"] == "2026-11-10T17:29:28+00:00" and seen["params"]["q"] == "ai when:2d"


def test_the_search_tool_answers_news_from_the_feeds(monkeypatch: pytest.MonkeyPatch, tmp_path):
    engine = search.SearchEngine()
    monkeypatch.setattr(engine, "news_search", lambda topic, limit, days=None, since=None: [
        _item("Mercor valued at $20 billion", 5), _item("Nvidia selloff deepens", 2), _item("Third", 1)])
    monkeypatch.setattr(engine, "web_search", lambda *a, **k: pytest.fail("a news question went to the web index"))
    context = tools.ToolContext(workspace=Workspace(tmp_path), memory=None, search=engine, conversation_id="c_test")
    result = tools.execute_tool("web_search", tools.validate_args("web_search", {"query": "latest ai news"}), context)
    assert result.ok and result.meta["news"] is True and result.meta["count"] == 3
    assert "Mercor valued at $20 billion" in result.content and result.content.startswith(research.NEWS_HEAD)


def test_the_search_tool_falls_back_to_the_web_when_no_feed_answers(monkeypatch: pytest.MonkeyPatch, tmp_path):
    engine = search.SearchEngine()
    monkeypatch.setattr(engine, "news_search", lambda *a, **k: [])
    monkeypatch.setattr(engine, "web_search", lambda query, limit: [
        {"title": "AI news hub", "url": "https://example.org/ai", "snippet": "All the AI news."}])
    context = tools.ToolContext(workspace=Workspace(tmp_path), memory=None, search=engine, conversation_id="c_test")
    result = tools.execute_tool("web_search", tools.validate_args("web_search", {"query": "latest ai news"}), context)
    assert result.ok and "AI news hub" in result.content and not result.meta.get("news")


def _news_turn(env, monkeypatch: pytest.MonkeyPatch, message: str):
    from test_backend import chat_events

    from kiwibro.llm import ChatResult

    asked: list[tuple] = []

    def news_search(topic, limit, *, days=None, since=None):
        asked.append((topic, days, since))
        return [_item("Nvidia selloff deepens, adding to AI jitters", 1, snippet="Shares fell 4%."),
                _item("Mercor valued at $20 billion", 5), _item("Berkeley weighs blocking Google search", 3)]

    monkeypatch.setattr(env.svc.search, "news_search", news_search)
    env.fake.stream_script = [ChatResult(content="The selloff matters because chips lead the market.")]
    events, _raw = chat_events(env, message)
    return asked, events


def _reply(events) -> str:
    text = ""
    for name, data in events:
        if name == "delta":
            text += data.get("text", "")
        elif name == "rewrite":
            text = data.get("text", "")
    return text


def test_the_users_bulletin_request_is_answered_from_dated_news_and_nothing_else(env, monkeypatch: pytest.MonkeyPatch):
    """Seen live: qwen3.8-2b wrote a made-up "October 2026" bulletin because it answered before any
    search, and later padded a real bulletin with paragraphs it invented. The bulletin is the reply."""
    asked, events = _news_turn(env, monkeypatch,
                               "search for todays ai news and tell be as bulletin. today and yesterday only")
    [(topic, days, since)] = asked
    assert topic == "ai" and days == 2 and since is not None  # today and yesterday, from the end of the message
    assert env.fake.stream_calls == []  # no model wrote a word of it
    reply = _reply(events)
    assert reply.startswith(research.NEWS_HEAD) and "Nvidia selloff deepens, adding to AI jitters" in reply
    assert "Shares fell 4%" in reply and "matters because" not in reply and "[Kiwibro" not in reply
    names = [event for event, _ in events]
    assert names.index("tool_result") < names.index("delta") and names[-1] == "done"


def test_a_news_question_that_asks_for_more_goes_to_the_model_with_the_bulletin(env, monkeypatch: pytest.MonkeyPatch):
    from kiwibro import prompts

    _asked, events = _news_turn(env, monkeypatch, "explain why today's ai news matters")
    first = env.fake.stream_calls[0]["messages"]
    assert first[-2]["role"] == "tool" and first[-2]["content"].startswith(research.NEWS_HEAD)
    assert first[-1]["content"] == prompts.NEWS_FIRST.format(then="")
    reply = _reply(events)
    assert reply.index("matters because") < reply.index(prompts.NEWS_SOURCES) < reply.index("Mercor valued")
    assert "[Kiwibro" not in reply  # the real stories sit under the model's words, without notes for it


@pytest.mark.parametrize(("message", "plain"), [
    ("search for todays ai news and tell be as bulletin. today and yesterday only", True),
    ("latest news about nvidia", True),
    ("summarize today's ai news in two sentences", False),
    ("what are the top 3 ai stories today and why", False),
    ("bugünün yapay zekâ haberlerini ara, türkçe yaz", False),
    ("give me today's tech news in german", False),
])
def test_only_a_plain_news_request_is_answered_with_the_bulletin_as_is(message: str, plain: bool):
    assert research.bulletin_is_the_answer(message) is plain


def test_summarised_articles_come_first_and_bare_headlines_are_few():
    bare = [_item(f"Headline {n}", n) for n in range(1, 9)]
    summarised = [_item(f"Story {n}", 10 + n, snippet="What happened.") for n in range(5)]
    picked = search.pick_news(bare + summarised, 8)
    assert sum(1 for item in picked if item["snippet"]) == 5
    assert sum(1 for item in picked if not item["snippet"]) == search.NEWS_EXTRA_HEADLINES
    assert [item["published"] for item in picked] == sorted((item["published"] for item in picked), reverse=True)
    assert len(search.pick_news(bare, 8)) == 8  # no summaries anywhere: headlines are all there is


@pytest.mark.parametrize(("message", "asks"), [
    ("summarize today's ai news", True),
    ("what is the latest ai news and why does it matter", True),
    ("any news on the election?", True),
    ("I read the news today and felt sad", False),
])
def test_a_request_for_news_is_told_from_a_remark_about_it(message: str, asks: bool):
    assert research.asks_for_news(message) is asks


class _Feeds:
    """Stands in for the search engine: a pool of dated items per wording."""

    def __init__(self, pools: dict[str, list[dict[str, str]]]) -> None:
        self.pools, self.asked = pools, []

    def news_pool(self, query: str, *, days=None, since=None) -> list[dict[str, str]]:
        self.asked.append(query)
        return [dict(item) for item in self.pools.get(query, [])]

    def news_search(self, query: str, max_results: int = 8, *, days=None, since=None) -> list[dict[str, str]]:
        return search.pick_news(self.news_pool(query, days=days, since=since), max_results)


def _stories(names: str, hours_ago: float = 1) -> list[dict[str, str]]:
    return [_item(f"{name} harbour story", hours_ago + index) for index, name in enumerate(names.split())]


def test_a_news_search_done_again_lists_what_was_not_shown():
    feeds = _Feeds({"chess": _stories("Alta Bodo Molde Voss Floro Rana Forde Sogndal Odda Stord")})
    ask = research.news_request("chess news", NOW)
    first, note = research.news_search(feeds, ask, 4)
    shown = research.format_news(first, note)
    again, note = research.again(feeds, ask, first, note, 4, shown_before=shown)
    assert research.NOT_SHOWN_BEFORE in note and again
    assert not {item["title"] for item in again} & {item["title"] for item in first}


def test_a_news_search_done_again_tries_other_words_before_giving_the_same_list():
    feeds = _Feeds({"chess": _stories("Alta Bodo Molde"), "chess news": _stories("Alta Voss Floro Rana")})
    ask = research.news_request("chess news", NOW)
    first, note = research.news_search(feeds, ask, 4)
    again, note = research.again(feeds, ask, first, note, 4, shown_before=research.format_news(first, note))
    assert "chess news" in feeds.asked and research.NOT_SHOWN_BEFORE in note
    assert [item["title"].split()[0] for item in again] == ["Voss", "Floro", "Rana"]


def test_a_news_search_done_again_says_so_when_the_feeds_have_nothing_else():
    feeds = _Feeds({"chess": _stories("Alta Bodo Molde")})
    ask = research.news_request("chess news", NOW)
    first, note = research.news_search(feeds, ask, 4)
    again, note = research.again(feeds, ask, first, note, 4, shown_before=research.format_news(first, note))
    assert research.NOTHING_NEWER in note and again == first


def test_a_refined_news_search_stays_on_the_subject_of_the_request():
    pool = [_item("Voss bakery closes after software switch", 2), _item("Alta harbour story", 1),
            _item("Bakery owner in Voss speaks out", 3)]
    feeds = _Feeds({"voss bakery owner": pool})
    ask = research.news_request("voss news bakery owner", NOW)
    again, _note = research.again(feeds, ask, pool[:2], "", 4, must_mention=("voss",))
    assert [item["title"] for item in again] == ["Voss bakery closes after software switch", "Bakery owner in Voss speaks out"]
