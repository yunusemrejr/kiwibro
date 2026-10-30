"""Tests for kiwibro.inference: model profiles, reasoning policy, prefix reuse, timing.

Offline and deterministic: no LM Studio, no sockets, no sleeps. An injected
clock replaces time.monotonic in RoundTimer.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from kiwibro import inference as inf

# --- fixtures: real LM Studio /api/v1/models shapes and a Kiwibro-normalised record -------

SPARK_RAW: dict[str, Any] = {
    "type": "llm",
    "key": "bartowski/spark-x2.6-4b",
    "architecture": "spark2_5",
    "quantization": {"name": "IQ2_M", "bits_per_weight": 2},
    "size_bytes": 1565852288,
    "params_string": "4B",
    "loaded_instances": [{"id": "spark-x2.6-4b", "config": {"context_length": 32768}}],
    "max_context_length": 1048576,
    "capabilities": {
        "vision": False,
        "trained_for_tool_use": True,
        "reasoning": {"allowed_options": ["off", "on"], "default": "on"},
    },
}

QWEN_27B_RAW: dict[str, Any] = {
    "type": "llm",
    "key": "qwen3.8-27b",
    "params_string": "27B",
    "capabilities": {"vision": True, "trained_for_tool_use": True},
}

KIWI_RECORD: dict[str, Any] = {
    "id": "qwen3.8-2b",
    "type": "llm",
    "state": "loaded",
    "context": 32768,
    "max_context": 262144,
    "tools": True,
    "vision": False,
    "quantization": "Q4_K_M",
}


class FakeClock:
    """Injected monotonic clock: tests set ``now`` explicitly."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# --- 1. model_profile on raw LM Studio entries and Kiwibro records -----------------------


def test_model_profile_raw_quantized_reasoning_model():
    p = inf.model_profile(SPARK_RAW["key"], SPARK_RAW)
    assert p.family == "spark"
    assert p.params_b == 4.0
    assert p.active_params_b == 4.0
    assert p.tier == "small"
    assert p.tools is True
    assert p.vision is False
    assert p.reasoning == ("off", "on")
    assert p.reasoning_default == "on"
    assert p.reasoning_toggle is True
    assert p.loaded is True
    assert p.quant_bits == 2.0
    assert p.size_bytes == 1565852288
    assert p.max_context == 1048576
    assert p.is_small is True


def test_raw_entry_reports_loaded_context_not_max():
    assert inf.model_profile(SPARK_RAW["key"], SPARK_RAW).context == 32768


def test_model_profile_raw_large_vision_model_without_toggle():
    p = inf.model_profile(QWEN_27B_RAW["key"], QWEN_27B_RAW)
    assert p.params_b == 27.0
    assert p.tier == "large"
    assert p.vision is True
    assert p.tools is True
    assert p.reasoning_toggle is False
    assert p.is_small is False


def test_model_profile_string_params_in_millions_is_tiny():
    p = inf.model_profile("some-model", {"params_string": "773M"})
    assert p.params_b == pytest.approx(0.773)
    assert p.tier == "tiny"


def test_model_profile_kiwibro_normalised_record():
    p = inf.model_profile(KIWI_RECORD["id"], KIWI_RECORD)
    assert p.context == 32768
    assert p.max_context == 262144
    assert p.tools is True
    assert p.loaded is True
    assert p.vision is False
    assert p.params_b == 2.0


def test_model_profile_from_id_only_moe():
    p = inf.model_profile("qwen3-30b-a3b")
    assert p.params_b == 30.0
    assert p.active_params_b == 3.0
    assert p.tier == "small"


def test_model_profile_unknown_id_ranks_last():
    p = inf.model_profile("mystery-model")
    assert p.tier == "unknown"
    assert p.cost == float("inf")


def test_model_profile_vlm_type_means_vision():
    assert inf.model_profile("", {"type": "vlm"}).vision is True


# --- 2. parse_params / params_from_id ----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("4.4B", 4.4), ("773M", 0.773), ("27B", 27.0), (None, None), ("", None), ("abc", None)],
)
def test_parse_params(text, expected):
    result = inf.parse_params(text)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("qwen3.8_4b_distilled_gguf", (4.0, 4.0)),
        ("qwen3.5-0.8b-distilled", (0.8, 0.8)),
        ("v1.5", (None, None)),
        ("qwen3-30b-a3b", (30.0, 3.0)),
    ],
)
def test_params_from_id(model_id, expected):
    assert inf.params_from_id(model_id) == expected


# --- 3. cost ordering --------------------------------------------------------------------


def test_cost_uses_size_bytes_when_known():
    sized = inf.model_profile("tiny-1b", {"size_bytes": 3_000_000_000})
    assert sized.cost == 3_000_000_000.0


def test_cost_falls_back_to_params_times_bits_per_byte():
    unsized = inf.model_profile("big-4b", {"quantization": "Q4_K_M"})
    assert unsized.size_bytes is None
    assert unsized.cost == pytest.approx(4 * 1e9 * 4.5 / 8)


def test_cost_orders_mixed_sized_and_unsized_by_bytes():
    sized = inf.model_profile("tiny-1b", {"size_bytes": 3_000_000_000})  # 3.0e9 bytes
    unsized = inf.model_profile("big-4b", {"quantization": "Q4_K_M"})  # 2.25e9 bytes
    assert sorted([sized, unsized], key=lambda p: p.cost)[0] is unsized


# --- 4. wants_reasoning -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("hi", False),
        ("why does my build fail?", True),
        ("what is 17*23", True),
    ],
)
def test_wants_reasoning_auto_on_trivial_and_hint_messages(message, expected):
    assert inf.wants_reasoning("auto", message) is expected


def test_wants_reasoning_auto_long_plain_message_thinks():
    assert inf.wants_reasoning("auto", "hello " * 50) is True  # 300 chars, no hint words


def test_turn_reasoning_auto_switches_only_while_prompt_is_cheap():
    hard = "why does my build fail?"
    assert inf.turn_reasoning("auto", hard, previous=None, prompt_tokens=50_000) is True
    # a flip re-prefills the whole prompt: allowed while short, refused when long
    assert inf.turn_reasoning("auto", hard, previous=False, prompt_tokens=800) is True
    assert inf.turn_reasoning("auto", hard, previous=False, prompt_tokens=5_000) is False
    assert inf.turn_reasoning("auto", "hi", previous=True, prompt_tokens=5_000) is True
    assert inf.turn_reasoning("auto", "hi", previous=True, prompt_tokens=100) is False


def test_turn_reasoning_explicit_modes_ignore_history():
    assert inf.turn_reasoning("on", "hi", previous=False, prompt_tokens=10**6) is True
    assert inf.turn_reasoning("off", "why?", previous=True, prompt_tokens=10**6) is False


def test_prefix_tracker_options_are_folded_into_message_zero():
    tracker = inf.PrefixTracker(token_fn=lambda m: len(m.get("content") or ""))
    msgs = [{"role": "system", "content": "s" * 10}, {"role": "user", "content": "u" * 5}]
    tracker.observe("m", msgs, None, {"reasoning_effort": "none"})
    assert tracker.observe("m", msgs, None, {"reasoning_effort": "none"})["reused_tokens_est"] == 15
    assert tracker.observe("m", msgs, None, None)["reused_tokens_est"] == 0

@pytest.mark.parametrize("message", ["hi", "why does my build fail?", "x" * 400])
def test_wants_reasoning_explicit_modes_ignore_message(message):
    assert inf.wants_reasoning("on", message) is True
    assert inf.wants_reasoning("on", message) is True
    assert inf.wants_reasoning("off", message) is False
    assert inf.wants_reasoning("off", message) is False


# --- 5. reasoning_payload -----------------------------------------------------------------


def test_reasoning_payload_toggle_off_when_not_thinking():
    profile = inf.model_profile(SPARK_RAW["key"], SPARK_RAW)
    assert inf.reasoning_payload(profile, think=False) == {"reasoning_effort": "none"}


def test_reasoning_payload_toggle_default_on_when_thinking_is_empty():
    profile = inf.model_profile(SPARK_RAW["key"], SPARK_RAW)
    assert inf.reasoning_payload(profile, think=True) == {}


def test_reasoning_payload_default_off_needs_explicit_medium_to_think():
    raw = {"key": "x-4b", "capabilities": {"reasoning": {"allowed_options": ["off", "on"], "default": "off"}}}
    profile = inf.model_profile("x-4b", raw)
    assert inf.reasoning_payload(profile, think=True) == {"reasoning_effort": "medium"}


def test_reasoning_payload_empty_for_model_without_toggle():
    profile = inf.model_profile(QWEN_27B_RAW["key"], QWEN_27B_RAW)
    assert inf.reasoning_payload(profile, think=True) == {}
    assert inf.reasoning_payload(profile, think=False) == {}


# --- 5b. thinking budgets (direct engine) ---------------------------------------------------


def test_budgets_apply_only_to_thinking_models_on_the_direct_engine():
    spark = inf.model_profile(SPARK_RAW["key"], SPARK_RAW)
    assert inf.uses_budgets(spark, direct=True)
    assert not inf.uses_budgets(spark, direct=False)
    assert not inf.uses_budgets(inf.model_profile(QWEN_27B_RAW["key"], QWEN_27B_RAW), direct=True)


def test_budget_payload_keeps_the_template_in_thinking_mode_whatever_the_decision():
    """The prompt never changes with the decision: 'no thinking' is a budget of 0."""
    spark = inf.model_profile(SPARK_RAW["key"], SPARK_RAW)
    off = inf.budget_payload(spark, think=False, must_act=False)
    acting = inf.budget_payload(spark, think=True, must_act=True)
    opening = inf.budget_payload(spark, think=True, must_act=False)
    assert off == {"chat_template_kwargs": {"enable_thinking": True}, "thinking_budget_tokens": 0}
    assert acting["thinking_budget_tokens"] == inf.THINK_BUDGET_TOKENS
    assert opening["thinking_budget_tokens"] == inf.OPEN_THINK_BUDGET_TOKENS > inf.THINK_BUDGET_TOKENS
    assert {k: v for k, v in off.items() if k not in inf.BUDGET_KEYS} == \
        {k: v for k, v in acting.items() if k not in inf.BUDGET_KEYS}
    assert "reasoning_effort" not in acting
    # a cut-off thought is closed in the model's own voice; never with a budget of 0 (it leaked)
    assert "act on it" in acting["reasoning_budget_message"] and "answer" in opening["reasoning_budget_message"]
    assert "reasoning_budget_message" not in off


def test_budget_payload_for_a_model_that_always_thinks_sets_only_the_budget():
    always = inf.model_profile("x-1b", {"key": "x-1b", "capabilities": {"reasoning": {"allowed_options": ["on"]}}})
    assert inf.budget_payload(always, think=False, must_act=False) == {"thinking_budget_tokens": 0}


def test_server_reuse_counts_exactly_from_llama_cpp_timings():
    assert inf.server_reuse({"cache_n": 3416.0, "prompt_n": 104.0}) == {
        "prompt_tokens_est": 3520, "reused_tokens_est": 3416, "reuse_percent": 97,
    }
    assert inf.server_reuse({"cache_n": 0.0, "prompt_n": 0.0})["reuse_percent"] == 0
    assert inf.server_reuse(None) is None
    assert inf.server_reuse({"prompt_n": 5.0}) is None  # LM Studio-style partial data: keep the estimate


# --- 10. PrefixTracker -------------------------------------------------------------------


def _tokens(message: dict[str, Any]) -> int:
    return len(str(message.get("content", "")))


def _chat(system: str = "SYS", *rest: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for index, text in enumerate(rest):
        out.append({"role": "user" if index % 2 == 0 else "assistant", "content": text})
    return out


def test_prefix_tracker_first_observe_reuses_nothing():
    tracker = inf.PrefixTracker(token_fn=_tokens)
    result = tracker.observe("m", _chat("SYS", "hello"))
    assert result["reused_tokens_est"] == 0
    assert result["prompt_tokens_est"] == 3 + 5


def test_prefix_tracker_growing_conversation_reuses_shared_prefix():
    tracker = inf.PrefixTracker(token_fn=_tokens)
    base = _chat("SYS", "hello")
    tracker.observe("m", base)
    grown = base + [{"role": "assistant", "content": "hi"}]
    result = tracker.observe("m", grown)
    shared_tokens = sum(_tokens(m) for m in base)
    total = sum(_tokens(m) for m in grown)
    assert result["reused_tokens_est"] == shared_tokens == 8
    assert result["prompt_tokens_est"] == total == 10
    assert result["reuse_percent"] == round(100 * shared_tokens / total) == 80


def test_prefix_tracker_changed_message_zero_reuses_nothing():
    tracker = inf.PrefixTracker(token_fn=_tokens)
    tracker.observe("m", _chat("SYS", "hello"))
    result = tracker.observe("m", _chat("SYS-CHANGED", "hello", "reply"))
    assert result["reused_tokens_est"] == 0


def test_prefix_tracker_tools_are_folded_into_message_zero():
    tools_a = [{"type": "function", "function": {"name": "read_file"}}]
    tools_b = [{"type": "function", "function": {"name": "write_file"}}]
    messages = _chat("SYS", "hello")

    tracker = inf.PrefixTracker(token_fn=_tokens)
    tracker.observe("m", messages, tools_a)
    assert tracker.observe("m", messages, tools_b)["reused_tokens_est"] == 0

    tracker = inf.PrefixTracker(token_fn=_tokens)
    tracker.observe("m", messages, tools_a)
    same_tools = tracker.observe("m", messages, tools_a)
    assert same_tools["reused_tokens_est"] == same_tools["prompt_tokens_est"]


def test_prefix_tracker_keeps_models_separate():
    tracker = inf.PrefixTracker(token_fn=_tokens)
    tracker.observe("model-a", _chat("SYS", "hello"))
    other = tracker.observe("model-b", _chat("SYS", "hello", "more"))
    assert other["reused_tokens_est"] == 0
    back = tracker.observe("model-a", _chat("SYS", "hello", "more"))
    assert back["reused_tokens_est"] > 0


def test_prefix_tracker_slots_evict_oldest_conversation():
    tracker = inf.PrefixTracker(slots=2, token_fn=_tokens)
    first, second, third = _chat("AAA", "x"), _chat("BBB", "x"), _chat("CCC", "x")
    for conversation in (first, second, third):
        tracker.observe("m", conversation)
    assert tracker.observe("m", first)["reused_tokens_est"] == 0


def test_prefix_tracker_forget_clears_history():
    tracker = inf.PrefixTracker(token_fn=_tokens)
    tracker.observe("m", _chat("SYS", "hello"))
    tracker.forget("m")
    assert tracker.observe("m", _chat("SYS", "hello", "more"))["reused_tokens_est"] == 0

    tracker.observe("m", _chat("SYS", "hello"))
    tracker.observe("other", _chat("SYS", "hello"))
    tracker.forget()
    assert tracker.observe("m", _chat("SYS", "hello", "more"))["reused_tokens_est"] == 0
    assert tracker.observe("other", _chat("SYS", "hello", "more"))["reused_tokens_est"] == 0


# --- 11. RoundTimer ----------------------------------------------------------------------


def test_round_timer_reports_ttft_and_decode_speed():
    clock = FakeClock(0.0)
    timer = inf.RoundTimer(clock)
    clock.now = 0.5
    timer.mark_first()
    clock.now = 2.5
    timer.finish()
    assert timer.stats(50) == {"total_ms": 2500, "ttft_ms": 500, "tokens_per_s": 25.0}


def test_round_timer_before_first_token_has_only_total():
    clock = FakeClock(0.0)
    timer = inf.RoundTimer(clock)
    clock.now = 1.25
    assert timer.stats(10) == {"total_ms": 1250}


def test_round_timer_only_first_mark_counts():
    clock = FakeClock(0.0)
    timer = inf.RoundTimer(clock)
    clock.now = 0.5
    timer.mark_first()
    clock.now = 1.0
    timer.mark_first()
    clock.now = 2.5
    timer.finish()
    assert timer.stats(50)["ttft_ms"] == 500


@pytest.mark.parametrize("completion_tokens", [None, 0])
def test_round_timer_omits_tokens_per_s_without_tokens(completion_tokens):
    clock = FakeClock(0.0)
    timer = inf.RoundTimer(clock)
    clock.now = 0.5
    timer.mark_first()
    clock.now = 2.5
    timer.finish()
    assert "tokens_per_s" not in timer.stats(completion_tokens)


def test_round_timer_omits_tokens_per_s_for_negligible_decode_time():
    clock = FakeClock(0.0)
    timer = inf.RoundTimer(clock)
    clock.now = 1.0
    timer.mark_first()
    clock.now = 1.02
    timer.finish()
    assert "tokens_per_s" not in timer.stats(50)


# --- leaked reasoning (bare </think> in content) ----------------------------------------


class _SSE:
    """Minimal stand-in for a streamed requests.Response."""

    status_code = 200
    encoding = "utf-8"

    def __init__(self, deltas: list[str]) -> None:
        chunks = [{"choices": [{"delta": {"content": text}}]} for text in deltas]
        self._lines = [f"data: {json.dumps(chunk)}".encode() for chunk in chunks] + [b"data: [DONE]"]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)


def _stream(monkeypatch, deltas: list[str]):
    from kiwibro.llm import LMStudioClient

    monkeypatch.setattr("kiwibro.llm.requests.post", lambda *a, **k: _SSE(deltas))
    shown: list[str] = []
    rewrites: list[tuple[str, str]] = []
    client = LMStudioClient(lambda: {"lmstudio_url": "http://lm.test"})
    result = client.stream_chat(
        model="m", messages=[{"role": "user", "content": "hi"}],
        on_delta=shown.append, on_rewrite=lambda text, why: rewrites.append((text, why)),
    )
    return result, shown, rewrites


def test_bare_close_tag_moves_leaked_reasoning_out_of_the_answer(monkeypatch):
    result, shown, rewrites = _stream(monkeypatch, ["The user wants", " a greeting.</th", "ink>\n\nHello", " there!"])
    assert result.content == "Hello there!" and result.reasoning == "The user wants a greeting."
    assert result.leaked_reasoning and rewrites == [("Hello", "The user wants a greeting.")]
    assert "".join(shown).endswith("Hello there!")  # deltas still streamed; the UI rewrites once


def test_full_think_block_in_content_is_split_too(monkeypatch):
    result, _shown, rewrites = _stream(monkeypatch, ["<think>plan it</think>Answer."])
    assert (result.content, result.reasoning, len(rewrites)) == ("Answer.", "plan it", 1)


def test_literal_tags_inside_an_answer_are_left_alone(monkeypatch):
    text = "Use `<think>` and `</think>` to mark reasoning."
    result, _shown, rewrites = _stream(monkeypatch, [text])
    assert result.content == text and not result.leaked_reasoning and rewrites == []


def test_agent_turn_emits_rewrite_and_stores_the_clean_answer(env, monkeypatch):
    from kiwibro.llm import ChatResult
    from tests.test_backend import chat_events

    def leaky_stream(*, on_delta=None, on_rewrite=None, **_kw):
        on_delta("The user wants a greeting.</think>Hello!")
        on_rewrite("Hello!", "The user wants a greeting.")
        return ChatResult(content="Hello!", reasoning="The user wants a greeting.", leaked_reasoning=True)

    monkeypatch.setattr(env.svc.llm, "stream_chat", leaky_stream)
    events, _raw = chat_events(env, "hi")
    names = [name for name, _data in events]
    assert names[0] == "meta" and names[-1] == "done"
    assert ("rewrite", {"text": "Hello!", "reasoning": "The user wants a greeting."}) in events
    conversation_id = events[0][1]["conversation_id"]
    reply = [m for m in env.svc.db.list_messages(conversation_id) if m["role"] == "assistant"][-1]
    assert reply["content"] == "Hello!" and reply["reasoning"] == "The user wants a greeting."


# --- measured speeds size background-call timeouts -------------------------------


def test_speed_book_budgets_fit_the_machine() -> None:
    book = inf.SpeedBook()
    slow = book.budget_s("m", prompt_tokens=2500, max_tokens=400)
    assert slow == 180.0  # unmeasured: pessimistic CPU defaults, capped
    book.note("m", new_tokens=2000, ttft_ms=1000, tokens_per_s=50.0)  # a fast GPU round
    fast = book.budget_s("m", prompt_tokens=2500, max_tokens=400)
    assert 8.0 < fast < 20.0
    assert book.budget_s("m", prompt_tokens=10, max_tokens=10) == 8.0  # floor
    book.note("m", new_tokens=10, ttft_ms=5000, tokens_per_s=None)  # too few tokens to measure prefill
    assert book.budget_s("m", prompt_tokens=2500, max_tokens=400) == fast


def test_memory_extraction_gets_a_speed_sized_budget(env, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def spy(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return {"ops": []}

    monkeypatch.setattr(env.svc.llm, "complete_json", spy)
    monkeypatch.setattr(inf, "SPEEDS", inf.SpeedBook())  # unmeasured model: CPU-sized budget
    conversation = env.svc.db.create_conversation()
    env.svc.db.add_message(conversation["id"], "user", "I live in Ankara and I prefer short answers.")
    env.svc.db.add_message(conversation["id"], "assistant", "Noted!")
    env.svc.memory.extract_and_apply(conversation["id"], "fake-1")
    assert seen["read_timeout"] > 8  # the old fixed 8 s never finished on a laptop CPU
    assert env.svc.db.get_conversation(conversation["id"])["extracted_through_seq"] == 2


def _drain(agent, rounds: int = 250) -> None:
    import threading

    for _ in range(rounds):
        with agent._extracting_guard:
            if not agent._extracting and not agent._memory_queue:
                return
        threading.Event().wait(0.02)


def test_memory_extractions_queue_and_run_one_at_a_time(env, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    release, calls, running = threading.Event(), [], []

    def slow_extract(conversation_id: str, model: str, *, abort=None) -> dict[str, int]:
        running.append(len(env.svc.agent._extracting))
        calls.append(conversation_id)
        release.wait(5)
        return {"added": 0, "updated": 0, "removed": 0}

    monkeypatch.setattr(env.svc.memory, "extract_and_apply", slow_extract)
    agent = env.svc.agent
    agent._start_memory_extraction("c1", "fake-1")
    agent._start_memory_extraction("c1", "fake-1")  # already queued: one run covers both turns
    agent._start_memory_extraction("c2", "fake-1")
    release.set()
    _drain(agent)
    assert calls == ["c1", "c2"] and set(running) == {1} and not agent._extracting


def test_a_reply_starting_cancels_extraction_and_it_runs_again_later(env, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    started, calls = threading.Event(), []

    def cancellable(conversation_id: str, model: str, *, abort=None) -> dict[str, int]:
        calls.append(conversation_id)
        if len(calls) == 1:
            started.set()
            abort.wait(5)
            return {"added": 0, "updated": 0, "removed": 0, "cancelled": 1}
        return {"added": 1, "updated": 0, "removed": 0}

    monkeypatch.setattr(env.svc.memory, "extract_and_apply", cancellable)
    agent = env.svc.agent
    agent._start_memory_extraction("c1", "fake-1")
    assert started.wait(5)
    chat_events_like_a_new_turn = agent._cancel_memory_extraction  # what start_chat does first
    chat_events_like_a_new_turn()
    _drain(agent)
    assert calls == ["c1", "c1"]
