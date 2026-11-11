"""A message read as a complaint by mistake must cost nothing: no search of the request before,
no answer set aside. And greetings never end in a template."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from test_backend import Env, chat_events

from kiwibro import plan, prompts, reactions
from kiwibro.llm import ChatResult

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import detectors  # noqa: E402

EARLIER = [{"role": "user", "content": "how do i get a red wine stain out of a shirt"},
           {"role": "assistant", "content": "Blot it, then cold water works on a fresh stain."}]


@pytest.mark.parametrize(("message", "asks"), [
    ("nope, mine is old and already washed, what then", "what then"),
    ("nah not today, i ran out, will it be bad without them", "will it be bad without them"),
    ("no, i meant the mat for the bathroom, and whats the weather like in izmir this weekend",
     "whats the weather like in izmir this weekend"),
])
def test_a_reaction_with_a_question_of_its_own_carries_that_question(message: str, asks: str):
    found = reactions.read(message, None, EARLIER)
    assert found is not None and found.asks == asks
    assert "Answer the new message as it is written" in found.note()


@pytest.mark.parametrize("message", ["wrong, tell me the real one", "thats wrong, can you search again", "no thats wrong",
                                     "these are from last year"])
def test_a_plain_complaint_carries_no_question_of_its_own(message: str):
    found = reactions.read(message, None, EARLIER)
    assert found is not None and found.asks == ""


@pytest.mark.parametrize("message", ["no its fine, i already fixed the time", "nah its fine, honestly im broke this month anyway",
                                     "no, not needed, thanks", "no thanks", "nope thats all"])
def test_turning_down_an_offer_is_not_a_complaint(message: str):
    assert reactions.read(message, None, EARLIER) is None


def test_age_words_about_the_users_own_things_do_not_ask_for_newer_sources():
    assert reactions.read("these are from last year", None, EARLIER).fresher
    assert reactions.read("thats old news give me todays", None, EARLIER).fresher
    mine = reactions.read("nope, mine is old and already washed, what then", None, EARLIER)
    assert mine is not None and not mine.fresher


@pytest.mark.parametrize(("message", "said"), [("hi", prompts.HELLO), ("hello kiwi", prompts.HELLO), ("thanks a lot", prompts.WELCOME),
                                               ("good night", prompts.GOODBYE)])
def test_small_talk_has_a_plain_reply(message: str, said: str):
    assert plan.small_talk_reply(message) == said


def test_a_greeting_answered_with_a_template_gets_a_plain_greeting(env: Env):
    env.fake.stream_script = [ChatResult(content="[Your greeting here]")]
    events, _raw = chat_events(env, "hi")
    conversation_id = events[0][1]["conversation_id"]
    shown = [m["content"] for m in env.svc.db.list_messages(conversation_id) if m["role"] == "assistant"]
    assert shown == [prompts.HELLO]
    assert len(env.fake.stream_calls) == 1  # there is no value to ask the model for


STAND_IN = "Here is the summary as one paragraph. [One coherent paragraph summarizing the main points of the topic above.]"


def test_a_bracketed_line_that_stands_in_for_the_answer_never_reaches_the_user(env: Env):
    env.fake.stream_script = [ChatResult(content=STAND_IN), ChatResult(content=STAND_IN)]
    events, _raw = chat_events(env, "tell me about kiwi birds in one paragraph")
    conversation_id = events[0][1]["conversation_id"]
    shown = [m["content"] for m in env.svc.db.list_messages(conversation_id) if m["role"] == "assistant"]
    assert shown == [prompts.NO_ANSWER]
    assert len(env.fake.stream_calls) == 2  # asked once more, then said plainly


@pytest.mark.parametrize(("reply", "blank"), [
    (STAND_IN, True),
    ("Dear [Name], thank you for your order.", True),
    ("Here it is:\n\n[A short paragraph that covers the main stories found above]", True),
    ("Read [the guide](https://example.org/guide) first.", False),
    ("In Python: `rows = [row for row in table if row is not None and row.ok]`.", False),
    ("The winning numbers were [3, 14, 15, 22, 31, 40] this week.", False),
    ("- [ ] buy milk\n- [x] call the plumber about the kitchen sink today", False),
])
def test_what_counts_as_a_placeholder(env: Env, reply: str, blank: bool):
    assert bool(env.svc.agent._blanks(reply)) is blank


# The standing check (scripts/detectors.py) on made-up messages the rules were not written from.
# Floors, not targets: they hold what was measured so a later change cannot quietly lose it.
@pytest.mark.parametrize(("name", "caught", "false_alarms", "news_caught", "news_false"), [
    ("set1.json", 54, 2, 49, 1),
    ("set2.json", 44, 18, 36, 5),
    ("set3.json", 27, 1, 26, 2),
])
def test_the_readers_hold_on_unseen_messages(name: str, caught: int, false_alarms: int, news_caught: int, news_false: int):
    import json

    data = json.loads((detectors.SETS / name).read_text(encoding="utf-8"))
    read = detectors.read_reactions(data["reactions"], helpers=False)
    news = detectors.read_news(data["news"], helpers=False)
    assert read["caught"] >= caught and read["false_alarms"] <= false_alarms
    assert read["would_search_again"] <= 1  # a wrong reading must not start a search
    assert news["caught"] >= news_caught and news["false_alarms"] <= news_false
