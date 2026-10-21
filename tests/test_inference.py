"""Tests for kiwibro.inference: model profiles, reasoning policy, prefix reuse, timing.

Offline and deterministic: no LM Studio, no sockets, no sleeps. An injected
clock replaces time.monotonic in RoundTimer.
"""

from __future__ import annotations

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


def test_wants_reasoning_auto_follow_up_round_does_not_think():
    assert inf.wants_reasoning("auto", "why does my build fail?", round_index=1) is False


@pytest.mark.parametrize("message", ["hi", "why does my build fail?", "x" * 400])
def test_wants_reasoning_explicit_modes_ignore_message(message):
    assert inf.wants_reasoning("on", message) is True
    assert inf.wants_reasoning("on", message, round_index=3) is True
    assert inf.wants_reasoning("off", message) is False
    assert inf.wants_reasoning("off", message, round_index=0) is False


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
