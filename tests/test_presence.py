"""Window presence: the model stops once no Kiwibro window is open, and not before.

The decision logic runs on a fake clock and fake busy/stop callables. The wiring tests go
through the Flask app (the poll and closing routes), and the engine tests use the same fake
llama-server as test_engine.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from kiwibro.engine import DirectEngine
from kiwibro.llm import LLMClient
from kiwibro.presence import CLOSE_GRACE_S, IDLE_S, STALE_S, Presence
from test_backend import Env, env  # noqa: F401  (re-exported fixture)
from test_engine import (  # noqa: F401  (re-exported fixtures and helpers)
    FAKE_LLAMA_BODY,
    add_tiny_model,
    direct_client,
    fake_calls,
    models_root,
    write_gguf,
    write_script,
)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Model:
    """Stands in for the agent (busy) and the engine (stop_idle)."""

    def __init__(self, stopped: int = 1) -> None:
        self.busy = False
        self.stopped = stopped
        self.stop_calls: list[float] = []

    def is_busy(self) -> bool:
        return self.busy

    def stop_idle(self, min_idle_s: float) -> int:
        self.stop_calls.append(min_idle_s)
        return self.stopped


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def model() -> Model:
    return Model()


@pytest.fixture
def presence(clock: Clock, model: Model) -> Presence:
    return Presence(busy=model.is_busy, stop_idle=model.stop_idle, clock=clock)


def test_closed_window_that_does_not_poll_again_stops_the_model(presence: Presence, clock: Clock, model: Model):
    presence.seen()
    clock.advance(1)
    presence.closing()
    clock.advance(CLOSE_GRACE_S - 0.5)
    assert presence.check() == 0  # still inside the grace: a reload may be on its way
    assert model.stop_calls == []
    clock.advance(1)
    assert presence.check() == 1
    assert model.stop_calls == [IDLE_S]


def test_reload_that_polls_within_the_grace_cancels_the_close(presence: Presence, clock: Clock, model: Model):
    presence.seen()
    presence.closing()
    clock.advance(2)
    presence.seen()  # the reloaded page is back
    clock.advance(CLOSE_GRACE_S * 3)
    assert presence.check() == 0
    assert model.stop_calls == []


def test_no_window_ever_seen_never_stops(presence: Presence, clock: Clock, model: Model):
    clock.advance(STALE_S * 10)
    assert presence.check() == 0  # used without a window (scripts, the API): the model stays
    assert model.stop_calls == []


def test_a_window_open_across_a_restart_that_closes_before_polling_stops_the_model(
        presence: Presence, clock: Clock, model: Model):
    presence.closing()  # only a window says it is closing, so one was open
    clock.advance(CLOSE_GRACE_S + 1)
    assert presence.check() == 1


def test_busy_reply_or_extraction_holds_the_model_until_idle(presence: Presence, clock: Clock, model: Model):
    presence.seen()
    presence.closing()
    clock.advance(CLOSE_GRACE_S + 1)
    model.busy = True  # a reply, or memory extraction, is running
    assert presence.check() == 0
    assert model.stop_calls == []
    model.busy = False
    assert presence.check() == 1
    assert model.stop_calls == [IDLE_S]


def test_silent_window_counts_as_gone_after_the_stale_timeout(presence: Presence, clock: Clock, model: Model):
    presence.seen()  # no close signal: a crash or a killed browser
    clock.advance(STALE_S - 1)
    assert presence.check() == 0
    clock.advance(2)
    assert presence.check() == 1
    assert model.stop_calls == [IDLE_S]


def test_model_started_with_no_window_is_stopped_again_later(presence: Presence, clock: Clock, model: Model):
    presence.seen()
    presence.closing()
    clock.advance(CLOSE_GRACE_S + 1)
    model.stopped = 0  # nothing running yet
    assert presence.check() == 0
    model.stopped = 1  # a scheduled task starts the model; it goes idle later and is stopped
    assert presence.check() == 1
    assert len(model.stop_calls) == 2


def test_closing_route_records_the_close(env: Env, clock: Clock, model: Model):  # noqa: F811
    presence = Presence(busy=model.is_busy, stop_idle=model.stop_idle, clock=clock)
    env.app.extensions["kiwibro_presence"] = presence
    presence.seen()
    response = env.client.post("/api/window/closing")  # no body, no JSON: what a beacon sends
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}
    clock.advance(CLOSE_GRACE_S + 1)
    assert presence.check() == 1


@pytest.mark.parametrize("path", ["/api/version", "/api/bootstrap"])
def test_window_polls_mark_the_window_as_open(env: Env, clock: Clock, model: Model, path: str):  # noqa: F811
    presence = Presence(busy=model.is_busy, stop_idle=model.stop_idle, clock=clock)
    env.app.extensions["kiwibro_presence"] = presence
    assert env.client.get(path).status_code == 200
    env.client.post("/api/window/closing")
    clock.advance(CLOSE_GRACE_S + 1)
    assert presence.check() == 1  # the poll made the window open; the close then made it gone


def test_llm_stop_idle_does_nothing_before_an_engine_exists(tmp_path: Path):
    client = LLMClient(lambda: {}, data_dir=tmp_path / "data")
    assert client.stop_idle(0) == 0


def test_llm_stop_idle_stops_the_running_model(direct_client: LLMClient, models_root: Path):  # noqa: F811
    add_tiny_model(models_root)
    assert direct_client.load_model("pub/tiny") is True
    assert direct_client.stop_idle(0) == 1
    assert direct_client.direct_engine().loaded("pub/tiny") is None


def test_stop_idle_stops_only_servers_unused_for_the_window(tmp_path: Path, models_root: Path):  # noqa: F811
    for name in ("a", "b"):
        write_gguf(models_root / "Pub" / f"{name}-GGUF" / f"{name}-Q4_K_M.gguf")
    script = write_script(tmp_path / "bin" / "llama-server", FAKE_LLAMA_BODY)
    eng = DirectEngine(lambda: {"direct_runtime": str(script)}, tmp_path / "data")
    try:
        idle = eng.ensure("pub/a")
        busy = eng.acquire("pub/b")  # a request is streaming from it
        idle.last_used = busy.last_used = time.monotonic() - 100
        assert eng.stop_idle(20) == 1
        assert not idle.alive()
        assert busy.alive()
        eng.release(busy)  # just used: not idle for 20 s yet
        assert eng.stop_idle(20) == 0
        assert busy.alive()
        busy.last_used -= 100
        assert eng.stop_idle(20) == 1
        assert not busy.alive()
    finally:
        eng.shutdown()
