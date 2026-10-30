"""Model readiness: explicit load with a working window, KV warm-up, and how a turn
shares (or stops) a warm-up that is running when it starts."""

from __future__ import annotations

import threading
import time
from typing import Any

from kiwibro import readiness
from kiwibro.llm import LMStudioError, TurnAborted


class FakeLLM:
    def __init__(self, state: str = "not-loaded", context: int = 32768, max_context: int = 131072) -> None:
        self.model = {
            "id": "m", "key": "m", "state": state, "context": context, "max_context": max_context,
            "loaded_config": {"context_length": context} if state == "loaded" else {},
        }
        self.loads: list[tuple[str, int | None, int | None]] = []
        self.unloads: list[str] = []
        self.prefills: list[dict[str, Any]] = []
        self.refuse_above: int | None = None
        self._load_api_supported: bool | None = None
        self.gate: threading.Event | None = None
        self.started = threading.Event()

    def resolve_model(self, requested: str | None = None) -> str:
        return "m"

    def list_models(self, force: bool = False) -> list[dict[str, Any]]:
        return [dict(self.model)]

    def load_model(self, model: str, context_length: int | None = None, parallel: int | None = None) -> bool:
        self.loads.append((model, context_length, parallel))
        if self.refuse_above and context_length and context_length > self.refuse_above:
            return False
        self.model.update(state="loaded", context=context_length, loaded_config={"context_length": context_length})
        return True

    def unload_model(self, model: str) -> bool:
        self.unloads.append(model)
        self.model.update(state="not-loaded", loaded_config={})
        return True

    def loaded_instance(self, model: str) -> str | None:
        return model

    def prefill(self, *, model, messages, tools=None, extra=None, abort=None) -> None:
        self.started.set()
        self.prefills.append({"model": model, "messages": messages, "tools": tools, "extra": extra})
        if self.gate is not None:
            while not self.gate.wait(0.02):
                if abort is not None and abort.is_set():
                    raise TurnAborted()


def prefix(model: str, conversation_id: str | None) -> tuple[list[dict[str, Any]], None, dict[str, Any]]:
    history = [{"role": "user", "content": f"history of {conversation_id}"}] if conversation_id else []
    return [{"role": "system", "content": "stable"}, *history, {"role": "user", "content": "…"}], None, {}


def make(llm: FakeLLM, busy: int = 0, prefix_fn=prefix) -> readiness.Readiness:
    return readiness.Readiness(llm, prefix_fn, busy_fn=lambda: busy)


def wait_phase(ready: readiness.Readiness, phase: str, timeout: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = ready.status()
        if status["phase"] == phase:
            return status
        time.sleep(0.01)
    raise AssertionError(f"phase {phase!r} not reached: {ready.status()}")


def test_unloaded_model_is_loaded_with_a_working_window_then_warmed():
    llm = FakeLLM()
    ready = make(llm)
    ready.request()
    status = wait_phase(ready, "ready")
    assert llm.loads == [("m", 32768, readiness.PARALLEL_SLOTS)]
    assert status["context_length"] == 32768 and status["model"] == "m"
    assert llm.prefills[0]["messages"][0] == {"role": "system", "content": "stable"}


def test_load_falls_back_to_smaller_windows_when_refused():
    llm = FakeLLM()
    llm.refuse_above = 8192
    ready = make(llm)
    ready.request()
    wait_phase(ready, "ready")
    assert [size for _model, size, _parallel in llm.loads] == [32768, 16384, 8192]


def test_load_ladder_never_exceeds_the_model_maximum():
    assert readiness.load_sizes(12000) == [12000, 8192]
    assert readiness.load_sizes(0) == list(readiness.LOAD_LADDER)


def test_small_loaded_window_is_enlarged_when_idle():
    llm = FakeLLM(state="loaded", context=4096)
    ready = make(llm)
    ready.request()
    wait_phase(ready, "ready")
    assert llm.unloads == ["m"] and llm.loads[-1][1] == readiness.TARGET_CONTEXT


def test_working_window_is_kept_and_a_warm_prefix_is_not_sent_twice():
    llm = FakeLLM(state="loaded", context=32768)
    ready = make(llm)
    ready.request()
    wait_phase(ready, "ready")
    ready.request()
    time.sleep(0.2)
    wait_phase(ready, "ready")
    assert llm.loads == [] and len(llm.prefills) == 1
    ready.invalidate()  # e.g. the user renamed themselves: message 0 changed
    ready.request()
    time.sleep(0.2)
    wait_phase(ready, "ready")
    assert len(llm.prefills) == 2


def test_nothing_runs_while_a_reply_is_streaming():
    llm = FakeLLM(state="loaded")
    ready = make(llm, busy=1)
    ready.request()
    time.sleep(0.3)
    assert llm.prefills == [] and llm.loads == []


def test_disabled_readiness_is_a_no_op():
    llm = FakeLLM()
    ready = readiness.Readiness(llm, prefix, busy_fn=lambda: 0, enabled=False)
    assert ready.request()["phase"] == "idle"
    time.sleep(0.1)
    assert llm.loads == [] and llm.prefills == []


def test_turn_waits_for_a_warm_up_of_its_own_chat():
    llm = FakeLLM(state="loaded")
    llm.gate = threading.Event()
    ready = make(llm)
    ready.request(conversation_id="c1")
    assert llm.started.wait(5)
    waited = threading.Event()
    finished = threading.Event()

    def turn() -> None:
        ready.before_turn("m", "c1", on_wait=waited.set)
        finished.set()

    threading.Thread(target=turn, daemon=True).start()
    assert waited.wait(5)
    assert not finished.wait(0.2)  # still prefilling: the turn shares that work
    llm.gate.set()
    assert finished.wait(5)


def test_turn_in_another_chat_cancels_the_warm_up():
    llm = FakeLLM(state="loaded")
    llm.gate = threading.Event()
    ready = make(llm)
    ready.request(conversation_id="c1")
    assert llm.started.wait(5)
    started = time.monotonic()
    ready.before_turn("m", "c2")
    assert time.monotonic() - started < 1.0
    status = wait_phase(ready, "ready")  # the aborted job ends quietly
    assert status["model"] == "m"


def test_no_prefix_means_ready_without_prefilling():
    llm = FakeLLM(state="loaded")
    ready = make(llm, prefix_fn=lambda model, conversation_id: None)
    ready.request(conversation_id="long-chat")
    wait_phase(ready, "ready")
    assert llm.prefills == []


def test_lm_studio_errors_surface_as_status():
    llm = FakeLLM(state="loaded")

    def broken(**_kwargs: Any) -> None:
        raise LMStudioError("connection", "LM Studio is not reachable.")

    llm.prefill = broken  # type: ignore[method-assign]
    ready = make(llm)
    ready.request()
    status = wait_phase(ready, "error")
    assert "not reachable" in status["detail"]


def test_signature_changes_with_the_loaded_window():
    messages = [{"role": "system", "content": "x"}]
    assert readiness.signature("m", messages, None, {}, 8192) != readiness.signature("m", messages, None, {}, 32768)


def test_switching_models_unloads_the_one_kiwibro_loaded_before():
    llm = FakeLLM()
    ready = make(llm)
    ready.request("m")
    wait_phase(ready, "ready")
    assert llm.unloads == []
    llm.model.update(id="n", key="n", state="not-loaded", loaded_config={})  # the user picks another model
    llm.resolve_model = lambda requested=None: "n"
    ready.request("n")
    deadline = time.monotonic() + 5
    while not llm.unloads and time.monotonic() < deadline:
        time.sleep(0.01)
    assert [load[0] for load in llm.loads] == ["m", "n"]
    assert llm.unloads == ["m"]  # freed: only "n" stays resident


def test_models_kiwibro_did_not_load_are_never_unloaded():
    llm = FakeLLM(state="loaded")  # loaded by the user in LM Studio
    ready = make(llm)
    ready.request("m")
    wait_phase(ready, "ready")
    llm.model.update(id="n", key="n", state="not-loaded", loaded_config={})
    ready.request("n")
    deadline = time.monotonic() + 1
    while len(llm.loads) < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)
    assert llm.unloads == []


def test_warm_limit_follows_the_measured_prefill_speed(monkeypatch) -> None:
    """Warming an opened chat is speculative: it may take minutes, not a quarter of an hour."""
    from kiwibro import inference

    book = inference.SpeedBook()
    monkeypatch.setattr(inference, "SPEEDS", book)
    assert readiness.warm_limit("m") == readiness.WARM_MIN_TOKENS  # unmeasured: pessimistic CPU default
    book.note("m", new_tokens=3500, ttft_ms=36_000, tokens_per_s=None)  # ~97 tok/s (an iGPU)
    assert readiness.warm_limit("m") == int(book.prefill_tps("m") * readiness.WARM_MAX_S)
    book.note("fast", new_tokens=3000, ttft_ms=3_000, tokens_per_s=None)  # a discrete GPU
    assert readiness.warm_limit("fast") == readiness.WARM_HISTORY_TOKENS
