"""When Kiwi Mind is stuck: the chosen helper model writes the rest of the turn; with none, Kiwibro never
shows the same reply twice and says so itself; a steering note that came too late is still answered."""

from __future__ import annotations

from typing import Any

import pytest
from test_backend import Env, chat_events

from kiwibro import context, prompts
from kiwibro.agent import HELPER_AFTER_FAILED_REPAIRS, Turn
from kiwibro.llm import ChatResult, LLMError
from kiwibro.mind import KIWIMIND

HELPER = "fake-1"
VENUS = "Venus is the closest planet to the sun, about 108 million km away."
MERCURY = "Mercury is the closest planet to the sun, about 58 million km away."
PLANET = "which planet is closest to the sun"
TURNED_DOWN = "that is wrong, try again"
CAPITAL = "Canberra is the capital of Australia, not Sydney."
CAPITAL_AGAIN = "The capital is Canberra, which lies between Sydney and Melbourne."
FIRST = "what is the capital of australia"
SECOND = "which city is the capital of australia"


@pytest.fixture
def mind(env: Env, monkeypatch: pytest.MonkeyPatch) -> Env:
    """Kiwi Mind leads every turn; the selected model (the fake) is its helper when one is selected."""
    monkeypatch.setattr(type(env.svc.llm.mind()), "usable", lambda self: True)
    return env


def choose_helper(env: Env, model: str | None = HELPER) -> None:
    env.settings_patch({"model": model})


def asked(env: Env, message: str, conversation_id: str | None = None) -> tuple[list[tuple[str, dict]], str]:
    events, _raw = chat_events(env, message, conversation_id)
    return events, next(payload["conversation_id"] for name, payload in events if name == "meta")


def models_called(env: Env) -> list[str]:
    return [call["model"] for call in env.fake.stream_calls]


def prompt_text(call: dict[str, Any]) -> str:
    return "\n".join(str(m.get("content") or "") for m in call["messages"])


def stored(env: Env, conversation_id: str) -> list[dict[str, Any]]:
    return [m for m in env.svc.db.list_messages(conversation_id) if m["role"] == "assistant"]


def labels(events: list[tuple[str, dict]]) -> list[str]:
    return [payload.get("label", "") for name, payload in events if name == "status"]


def brains(events: list[tuple[str, dict]]) -> list[dict]:
    return [payload for name, payload in events if name == "brain"]


# --- B: the helper leads when Kiwi Mind is stuck --------------------------------------------------

def test_a_turned_down_answer_is_redone_by_the_helper_and_the_next_turn_goes_back_to_kiwi_mind(mind: Env) -> None:
    choose_helper(mind)
    mind.fake.stream_script = [ChatResult(content=VENUS), ChatResult(content=MERCURY), ChatResult(content="Welcome.")]
    _events, cid = asked(mind, PLANET)
    events, _ = asked(mind, TURNED_DOWN, cid)
    assert models_called(mind) == [KIWIMIND, HELPER]
    assert "fake-1 is taking over." in labels(events)
    assert brains(events) == [{"brain": "helper", "name": "fake-1", "lead": True}]
    assert [m["model"] for m in stored(mind, cid)] == [KIWIMIND, HELPER]
    assert stored(mind, cid)[-1]["content"] == MERCURY
    again, _ = asked(mind, "thanks, and what is the next planet out", cid)
    assert models_called(mind)[-1] == KIWIMIND and brains(again) == []
    # Kiwi Mind's message 0 is the same text before and after the helper's turn.
    first, last = mind.fake.stream_calls[0], mind.fake.stream_calls[-1]
    assert first["messages"][0] == last["messages"][0]


def test_the_helper_never_sees_the_answer_it_replaces(mind: Env) -> None:
    choose_helper(mind)
    mind.fake.stream_script = [ChatResult(content=VENUS), ChatResult(content=MERCURY)]
    _events, cid = asked(mind, PLANET)
    asked(mind, TURNED_DOWN, cid)
    redo = prompt_text(mind.fake.stream_calls[1])
    assert VENUS not in redo and context.REJECTED_STUB in redo


def test_a_repeated_reply_is_written_again_by_the_helper(mind: Env) -> None:
    choose_helper(mind)
    mind.fake.stream_script = [ChatResult(content=CAPITAL), ChatResult(content=CAPITAL), ChatResult(content=CAPITAL_AGAIN)]
    _events, cid = asked(mind, FIRST)
    events, _ = asked(mind, SECOND, cid)
    assert models_called(mind) == [KIWIMIND, KIWIMIND, HELPER]
    assert brains(events) == [{"brain": "helper", "name": "fake-1", "lead": True}]
    assert stored(mind, cid)[-1]["content"] == CAPITAL_AGAIN and stored(mind, cid)[-1]["model"] == HELPER
    assert CAPITAL not in prompt_text(mind.fake.stream_calls[2])


def test_the_helper_is_told_once_however_many_generations_it_writes(mind: Env) -> None:
    choose_helper(mind)
    mind.fake.stream_script = [ChatResult(content=VENUS), ChatResult(content=MERCURY)]
    _events, cid = asked(mind, PLANET)
    events, _ = asked(mind, TURNED_DOWN, cid)
    assert labels(events).count("fake-1 is taking over.") == 1


def test_a_helper_that_cannot_answer_is_given_up_and_the_status_says_so(mind: Env) -> None:
    choose_helper(mind)
    mind.fake.stream_script = [
        ChatResult(content=VENUS), LLMError("connection", "not reachable"), ChatResult(content=MERCURY)]
    _events, cid = asked(mind, PLANET)
    events, _ = asked(mind, TURNED_DOWN, cid)
    assert models_called(mind) == [KIWIMIND, HELPER, KIWIMIND]
    assert "fake-1 could not answer. Kiwi Mind is answering." in labels(events)
    assert stored(mind, cid)[-1]["content"] == MERCURY and stored(mind, cid)[-1]["model"] == KIWIMIND
    assert events[-1][0] == "done"


def test_without_a_helper_kiwi_mind_answers_the_turned_down_question_itself(mind: Env) -> None:
    choose_helper(mind, None)
    mind.fake.stream_script = [ChatResult(content=VENUS), ChatResult(content=MERCURY)]
    _events, cid = asked(mind, PLANET)
    events, _ = asked(mind, TURNED_DOWN, cid)
    assert models_called(mind) == [KIWIMIND, KIWIMIND] and brains(events) == []
    assert not any("taking over" in label for label in labels(events))


def turn_on_kiwi_mind(env: Env, repairs: int) -> Turn:
    turn = Turn(conversation_id="c", message="hi", requested_model=None, model=KIWIMIND)
    for _ in range(repairs):
        env.svc.agent._repair_failed(turn)
    return turn


def test_the_second_failed_repair_in_a_turn_hands_it_to_the_helper_the_first_does_not(mind: Env) -> None:
    choose_helper(mind)
    assert HELPER_AFTER_FAILED_REPAIRS == 2
    one = turn_on_kiwi_mind(mind, 1)
    assert one.model == KIWIMIND and one.lead is None
    two = turn_on_kiwi_mind(mind, 2)
    assert (two.model, two.lead, two.lead_name) == (HELPER, HELPER, "fake-1")
    assert "failed their repair" in two.inference["helper_led"]


def test_failed_repairs_hand_nothing_over_when_no_helper_is_chosen(mind: Env) -> None:
    choose_helper(mind, None)
    turn = turn_on_kiwi_mind(mind, 3)
    assert turn.model == KIWIMIND and turn.lead is None


# --- C: Kiwi Mind alone never shows the same reply twice ----------------------------------------------

def test_the_redo_prompt_leaves_out_the_answer_that_was_turned_down(mind: Env) -> None:
    choose_helper(mind, None)
    mind.fake.stream_script = [ChatResult(content=VENUS), ChatResult(content=MERCURY)]
    _events, cid = asked(mind, PLANET)
    asked(mind, TURNED_DOWN, cid)
    redo = prompt_text(mind.fake.stream_calls[1])
    assert VENUS not in redo
    assert context.REJECTED_STUB == "[An answer the user turned down as wrong or out of date stood here. It is not repeated.]"
    assert context.REJECTED_STUB in redo


def test_the_redo_prompt_after_a_repeat_leaves_out_the_repeated_text(mind: Env) -> None:
    choose_helper(mind, None)
    mind.fake.stream_script = [ChatResult(content=CAPITAL), ChatResult(content=CAPITAL), ChatResult(content=CAPITAL_AGAIN)]
    _events, cid = asked(mind, FIRST)
    asked(mind, SECOND, cid)
    redo = prompt_text(mind.fake.stream_calls[2])
    assert CAPITAL not in redo and context.REPEATED_STUB in redo
    assert stored(mind, cid)[-1]["content"] == CAPITAL_AGAIN


def test_the_answer_left_out_of_the_new_attempt_is_shown_to_the_model_again_next_turn(mind: Env) -> None:
    choose_helper(mind, None)
    mind.fake.stream_script = [ChatResult(content=CAPITAL), ChatResult(content=CAPITAL),
                               ChatResult(content=CAPITAL_AGAIN), ChatResult(content="About 450,000 people live there.")]
    _events, cid = asked(mind, FIRST)
    asked(mind, SECOND, cid)
    asked(mind, "how many people live there", cid)
    last = prompt_text(mind.fake.stream_calls[-1])
    assert CAPITAL in last and context.REPEATED_STUB not in last


@pytest.mark.parametrize("second", [ChatResult(content=CAPITAL), ChatResult(content="")], ids=["again", "nothing"])
def test_a_redo_that_repeats_again_gets_kiwibros_own_line_never_an_empty_reply(mind: Env, second: ChatResult) -> None:
    choose_helper(mind, None)
    mind.fake.stream_script = [ChatResult(content=CAPITAL), ChatResult(content=CAPITAL), second]
    _events, cid = asked(mind, FIRST)
    events, _ = asked(mind, SECOND, cid)
    assert models_called(mind) == [KIWIMIND] * 3
    assert stored(mind, cid)[-1]["content"] == prompts.REPEATED_ANSWER
    assert prompts.REPEATED_ANSWER.startswith("I only came up with the same answer as before")
    assert next(p for n, p in events if n == "done")["inference"]["said_by_kiwibro"] == "repeat"
    assert any(p["text"] == prompts.REPEATED_ANSWER for n, p in events if n == "rewrite")


def test_a_turned_down_answer_given_again_is_not_shown_again(mind: Env) -> None:
    choose_helper(mind, None)
    mind.fake.stream_script = [ChatResult(content=VENUS), ChatResult(content=VENUS), ChatResult(content=VENUS)]
    _events, cid = asked(mind, PLANET)
    asked(mind, TURNED_DOWN, cid)
    assert stored(mind, cid)[-1]["content"] == prompts.REPEATED_ANSWER
    assert [m["content"] for m in stored(mind, cid)].count(VENUS) == 1  # the first one, which was turned down


# --- D: a steering note that arrives too late is answered ----------------------------------------------

def test_a_note_that_arrives_after_the_last_reply_was_written_is_answered(env: Env) -> None:
    _events, cid = asked(env, "name a tall mountain")
    note = "also give its height in feet"

    def reply_while_the_user_adds_a_note() -> ChatResult:
        assert env.svc.agent.steer(cid, note)  # the model has read the chat; the note comes after
        return ChatResult(content="Mount Everest is the tallest mountain on Earth.")

    env.fake.stream_script = [reply_while_the_user_adds_a_note, ChatResult(content="It is about 29,032 feet tall.")]
    events, _ = asked(env, "and the tallest one in europe too", cid)
    assert len(env.fake.stream_calls) == 3  # the first turn, this turn, the note
    answering = env.fake.stream_calls[-1]["messages"]
    assert answering[-1]["role"] == "user" and note in answering[-1]["content"]
    assert "Answering your note" in labels(events)
    texts = [m["content"] for m in stored(env, cid)]
    assert texts[-2:] == ["Mount Everest is the tallest mountain on Earth.", "It is about 29,032 feet tall."]
    assert events[-1][0] == "done"


def test_a_note_that_a_generation_read_is_not_answered_a_second_time(env: Env) -> None:
    _events, cid = asked(env, "name a tall mountain")
    env.fake.stream_script = [ChatResult(content="Mount Everest.")]
    events, _ = asked(env, "and a deep ocean trench", cid)
    assert len(env.fake.stream_calls) == 2 and "Answering your note" not in labels(events)
