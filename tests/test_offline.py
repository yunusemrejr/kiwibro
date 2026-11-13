"""The app notices when the internet is gone: one cheap probe, web tools that fail at once,
the model told in its turn note, and /health reporting the state.

No test here opens a real connection: the probe's ``socket.create_connection`` or the whole
probe is replaced, and the rest of the suite runs with the environment switch saying online.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kiwibro import connectivity, search, tools  # noqa: E402
from kiwibro.agent import Turn  # noqa: E402
from kiwibro.sandbox import Workspace  # noqa: E402
from kiwibro.search import SearchEngine, SearchError  # noqa: E402
from kiwibro.tools import ToolContext  # noqa: E402

OFFLINE_RESULT = f"Error: {connectivity.OFFLINE_ERROR}"


@pytest.fixture()
def probe(monkeypatch: pytest.MonkeyPatch):
    """The probe itself: no answer from the environment and nothing remembered."""
    monkeypatch.delenv(connectivity.ENV_SWITCH, raising=False)
    connectivity._reset()
    yield
    connectivity._reset()


class Reached:
    """What an accepted connection looks like to the probe."""

    def close(self) -> None:
        pass


def refuse(address: tuple[str, int], timeout: float | None = None):
    raise OSError("unreachable")


# --- the probe -----------------------------------------------------------------


def test_online_when_any_one_address_accepts(probe, monkeypatch: pytest.MonkeyPatch):
    tried: list[tuple[str, int]] = []

    def connect(address, timeout=None):
        tried.append(address)
        if address == ("8.8.8.8", 53):  # every other address refuses
            return Reached()
        raise OSError("unreachable")

    monkeypatch.setattr(connectivity.socket, "create_connection", connect)
    assert connectivity.online() is True
    assert ("8.8.8.8", 53) in tried


def test_offline_when_no_address_accepts(probe, monkeypatch: pytest.MonkeyPatch):
    tried: list[tuple[tuple[str, int], float | None]] = []

    def connect(address, timeout=None):
        tried.append((address, timeout))
        raise OSError("unreachable")

    monkeypatch.setattr(connectivity.socket, "create_connection", connect)
    assert connectivity.online() is False
    assert [address for address, _timeout in tried] == list(connectivity.PROBE_ADDRESSES)
    assert all(timeout == connectivity.PROBE_TIMEOUT_S for _address, timeout in tried)


def test_the_probe_gives_up_within_about_a_second(probe, monkeypatch: pytest.MonkeyPatch):
    def slow(address, timeout=None):
        time.sleep(2.0)
        raise OSError("unreachable")

    monkeypatch.setattr(connectivity.socket, "create_connection", slow)
    started = time.monotonic()
    assert connectivity.online() is False
    assert 1.0 <= time.monotonic() - started < 1.6


def test_a_remembered_online_answer_is_reused_until_it_ages(probe, monkeypatch: pytest.MonkeyPatch):
    tried: list[tuple[str, int]] = []

    def connect(address, timeout=None):
        tried.append(address)
        return Reached()

    monkeypatch.setattr(connectivity.socket, "create_connection", connect)
    connectivity._set_state(True, age=44.0)
    assert connectivity.online() is True and tried == []
    connectivity._set_state(True, age=46.0)
    assert connectivity.online() is True and len(tried) == 3


def test_a_remembered_offline_answer_is_asked_again_sooner(probe, monkeypatch: pytest.MonkeyPatch):
    tried: list[tuple[str, int]] = []

    def connect(address, timeout=None):
        tried.append(address)
        raise OSError("unreachable")

    monkeypatch.setattr(connectivity.socket, "create_connection", connect)
    connectivity._set_state(False, age=7.0)
    assert connectivity.online() is False and tried == []
    connectivity._set_state(False, age=9.0)
    assert connectivity.online() is False and len(tried) == 3


def test_a_failed_request_makes_the_next_look_probe(probe, monkeypatch: pytest.MonkeyPatch):
    tried: list[tuple[str, int]] = []

    def connect(address, timeout=None):
        tried.append(address)
        raise OSError("unreachable")

    monkeypatch.setattr(connectivity.socket, "create_connection", connect)
    connectivity._set_state(True)
    assert connectivity.online() is True and tried == []
    connectivity.suspect()
    assert connectivity.online() is False
    assert len(tried) == 3


def test_callers_arriving_together_share_one_probe(probe, monkeypatch: pytest.MonkeyPatch):
    tried: list[tuple[str, int]] = []

    def connect(address, timeout=None):
        tried.append(address)
        time.sleep(0.2)
        raise OSError("unreachable")

    monkeypatch.setattr(connectivity.socket, "create_connection", connect)
    answers: list[bool] = []
    threads = [threading.Thread(target=lambda: answers.append(connectivity.online())) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert answers == [False] * 6
    assert len(tried) == 3  # one probe of three addresses, not six


def test_the_environment_switch_answers_without_a_probe(probe, monkeypatch: pytest.MonkeyPatch):
    tried: list[tuple[str, int]] = []

    def connect(address, timeout=None):
        tried.append(address)
        return Reached()  # would say online

    monkeypatch.setattr(connectivity.socket, "create_connection", connect)
    monkeypatch.setenv(connectivity.ENV_SWITCH, "1")
    assert connectivity.online() is False
    assert connectivity.state()["online"] is False
    monkeypatch.setenv(connectivity.ENV_SWITCH, "0")
    assert connectivity.online() is True
    assert tried == []


def test_the_state_says_when_the_answer_was_taken(probe, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(connectivity.socket, "create_connection", refuse)
    before = int(time.time())
    assert connectivity.online() is False
    reported = connectivity.state()
    assert reported["online"] is False
    assert before <= reported["checked"] <= int(time.time())


# --- web tools fail at once and say why --------------------------------------------


@pytest.fixture()
def context(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=Workspace(tmp_path), memory=None, search=SearchEngine(), conversation_id="c_offline")


TOOL_CALLS = [
    ("web_search", {"query": "what colour is a kiwi fruit"}),
    ("multi_search", {"queries": ["what colour is a kiwi fruit", "kiwi fruit skin"]}),
    ("fetch_page", {"url": "https://example.org/kiwi"}),
    ("fetch_pages", {"urls": ["https://example.org/kiwi", "https://example.org/orchard"]}),
    ("web_research", {"query": "what colour is a kiwi fruit"}),
    ("browse_page", {"url": "https://example.org/kiwi"}),
    ("browser", {"action": "open", "url": "https://example.org/kiwi"}),
]


@pytest.mark.parametrize("name,args", TOOL_CALLS)
def test_every_web_tool_fails_at_once_with_the_offline_sentence(offline, context, name, args):
    started = time.monotonic()
    result = tools.execute_tool(name, tools.validate_args(name, args), context)
    elapsed = time.monotonic() - started
    assert result.ok is False
    assert result.content == OFFLINE_RESULT
    assert elapsed < 0.5


def test_a_search_answered_from_the_cache_is_still_served_offline(probe, monkeypatch: pytest.MonkeyPatch):
    engine = SearchEngine()
    monkeypatch.setattr(engine, "_search_providers", lambda query, limit: [
        {"title": "Kiwi harvest", "url": "https://example.org/kiwi", "snippet": "Picked in autumn."},
    ])
    first = engine.web_search("kiwi harvest dates", 3)
    monkeypatch.setenv(connectivity.ENV_SWITCH, "1")
    assert engine.web_search("kiwi harvest dates", 3) == first
    with pytest.raises(SearchError) as raised:
        engine.web_search("kiwi storage", 3)
    assert str(raised.value) == connectivity.OFFLINE_ERROR


def test_the_browser_extension_does_not_hide_the_offline_sentence(offline, context):
    from kiwibro import browser

    with pytest.raises(SearchError) as raised:
        browser._exec_browser_inspect({"action": "inspect", "url": "https://example.org/kiwi"}, context)
    assert str(raised.value) == connectivity.OFFLINE_ERROR


def test_a_search_whose_providers_cannot_connect_reports_offline(probe, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def refuse_provider(query: str, limit: int) -> list[dict[str, str]]:
        raise requests.ConnectionError("the network is down")

    monkeypatch.setattr(search, "PROVIDERS", (("example_index", refuse_provider),))
    monkeypatch.setattr(search, "PROVIDER_TIERS", {"example_index": 0})
    monkeypatch.setattr(connectivity.socket, "create_connection", refuse)
    connectivity._set_state(True)  # online when the search starts; the probe is what learns otherwise
    context = ToolContext(workspace=Workspace(tmp_path), memory=None, search=SearchEngine(), conversation_id="c_passive")
    result = tools.execute_tool(
        "web_search", tools.validate_args("web_search", {"query": "what colour is a kiwi fruit"}), context
    )
    assert result.ok is False
    assert result.content == OFFLINE_RESULT


def test_news_feeds_that_cannot_connect_report_offline(probe, monkeypatch: pytest.MonkeyPatch):
    def refuse_feed(query: str, limit: int, days: int | None = None) -> list[dict[str, str]]:
        raise requests.ConnectionError("the network is down")

    monkeypatch.setattr(search, "NEWS_PROVIDERS", (("example_feed", refuse_feed),))
    monkeypatch.setattr(connectivity.socket, "create_connection", refuse)
    connectivity._set_state(True)
    with pytest.raises(SearchError) as raised:
        SearchEngine().news_search("kiwi fruit news", 5)
    assert str(raised.value) == connectivity.OFFLINE_ERROR


# --- the model is told ------------------------------------------------------------


def _turn_note(env: Any) -> str:
    conversation = env.svc.db.create_conversation()
    turn = Turn(conversation_id=conversation["id"], message="hello there", requested_model=None)
    return env.svc.agent._turn_context(turn, {})


def test_the_turn_note_puts_the_offline_line_under_the_clock(env, offline):
    lines = _turn_note(env).splitlines()
    at = next(index for index, line in enumerate(lines) if line.startswith("Local time: "))
    assert lines[at + 1] == connectivity.OFFLINE_LINE


def test_the_turn_note_says_nothing_about_the_internet_when_online(env):
    assert connectivity.OFFLINE_LINE not in _turn_note(env)


def _search_turn() -> Turn:
    turn = Turn(conversation_id="c_offline", message="what are the latest kiwi harvest dates", requested_model=None)
    turn.facts = SimpleNamespace(small_talk=False, tools=("web_search",))
    turn.reaction = SimpleNamespace(asks=True)  # a question of its own, not a complaint to redo
    return turn


def test_no_search_runs_before_the_answer_when_offline(env, offline, monkeypatch: pytest.MonkeyPatch):
    agent = env.svc.agent
    ran: list[Any] = []
    monkeypatch.setattr(agent, "_own_search", lambda turn, reading: "kiwi harvest dates")
    monkeypatch.setattr(agent, "_run_search_first", lambda *args: ran.append(args) or True)
    assert agent._search_first(_search_turn(), {}) is False
    assert ran == []


def test_the_same_message_is_searched_when_online(env, monkeypatch: pytest.MonkeyPatch):
    agent = env.svc.agent
    ran: list[Any] = []
    monkeypatch.setattr(agent, "_own_search", lambda turn, reading: "kiwi harvest dates")
    monkeypatch.setattr(agent, "_run_search_first", lambda *args: ran.append(args) or True)
    assert agent._search_first(_search_turn(), {}) is True
    assert len(ran) == 1


# --- /health ----------------------------------------------------------------------


def test_health_reports_the_answer_and_a_fresh_one_costs_no_probe(env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(connectivity.ENV_SWITCH, raising=False)
    probes: list[int] = []
    monkeypatch.setattr(connectivity, "_probe", lambda: probes.append(1) or True)
    connectivity._set_state(False)
    assert env.client.get("/health").get_json()["online"] is False
    assert env.client.get("/health").get_json()["online"] is False
    assert probes == []
    connectivity._set_state(True, age=60.0)
    assert env.client.get("/health").get_json()["online"] is True
    assert len(probes) == 1
