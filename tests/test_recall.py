"""A question about a saved fact is answered with that fact."""

from __future__ import annotations

import pytest
from test_backend import Env, chat_events

from kiwibro import recall
from kiwibro.llm import ChatResult

FACTS = ["User lives in Ankara", "User has a cat named Pamuk", "User's name is Ana", "User works as a nurse",
         "User likes to cook Italian food", "User's next exam is on 12 May", "User studies physics at METU"]


@pytest.mark.parametrize(("question", "fact", "telling"), [
    ("where do i live?", "User lives in Ankara", ("Ankara",)),
    ("what's my cat's name?", "User has a cat named Pamuk", ("Pamuk",)),
    ("do you remember my cat's name", "User has a cat named Pamuk", ("Pamuk",)),
    ("whats my name", "User's name is Ana", ("Ana",)),
    ("what do i do for work", "User works as a nurse", ("nurse",)),
    ("where do i study", "User studies physics at METU", ("METU",)),
    ("when is my next exam", "User's next exam is on 12 May", ("12", "May")),
])
def test_the_fact_a_question_asks_for_is_found_with_its_answer_words(question: str, fact: str, telling: tuple[str, ...]):
    assert recall.asks_for_a_fact(question)
    assert recall.answers(question, FACTS) == [(fact, telling)]


@pytest.mark.parametrize("question", [
    "what should i cook tonight?",  # the model's to answer; a saved taste is only a hint
    "what did i ask you to buy",
    "how do i move a file",
    "can you write my cover letter",
])
def test_other_questions_about_the_user_are_not_checked(question: str):
    assert not recall.asks_for_a_fact(question)


def test_a_question_no_fact_covers_finds_nothing():
    assert recall.answers("what's my next step?", FACTS) == []  # "next" alone is not the exam
    assert recall.answers("what's my cpu", FACTS) == []
    assert recall.answers("how old am i", FACTS) == []


def test_the_reply_has_to_give_the_answer_word():
    found = recall.answers("what's my cat's name?", FACTS)
    assert recall.said("Your cat's name is **Pamuk**.", found)
    assert not recall.said("You can tell me your cat's name!", found)
    assert not recall.said("Your cat's name is Whiskers.", found)
    assert recall.said("you are a nurse", recall.answers("what do i do for work", FACTS))


@pytest.mark.parametrize(("fact", "said"), [
    ("User lives in Ankara", "you live in Ankara"),
    ("User has a cat named Pamuk", "you have a cat named Pamuk"),
    ("User's name is Ana", "your name is Ana"),
    ("User studies physics at METU", "you study physics at METU"),
    ("User is a nurse.", "you are a nurse"),
    ("User teaches maths", "you teach maths"),
])
def test_a_saved_fact_is_said_to_the_user(fact: str, said: str):
    assert recall.to_the_user(fact) == said


def test_a_reply_without_the_fact_is_asked_again_with_the_fact_shown(env: Env):
    env.svc.memory.remember("User lives in Ankara")
    env.fake.stream_script = [ChatResult(content="I don't know where you live. You can tell me!"),
                              ChatResult(content="You live in Ankara.")]
    events, _raw = chat_events(env, "where do i live?")
    assert "User lives in Ankara" in env.fake.stream_calls[1]["messages"][-1]["content"]
    conversation_id = events[0][1]["conversation_id"]
    shown = [m["content"] for m in env.svc.db.list_messages(conversation_id) if m["role"] == "assistant"]
    assert shown == ["You live in Ankara."]


def test_a_fact_the_model_leaves_out_twice_is_said_by_kiwibro(env: Env):
    env.svc.memory.remember("User lives in Ankara")
    env.fake.stream_script = [ChatResult(content="I don't know where you live. You can tell me!"),
                              ChatResult(content="I have no information about that.")]
    events, _raw = chat_events(env, "where do i live?")
    done = [payload for event, payload in events if event == "done"][0]
    assert done["inference"]["from_memory"] == "said by Kiwibro"
    conversation_id = events[0][1]["conversation_id"]
    shown = [m["content"] for m in env.svc.db.list_messages(conversation_id) if m["role"] == "assistant"]
    assert shown == ["From what you told me: you live in Ankara."]


def test_a_reply_that_gives_the_fact_is_left_alone(env: Env):
    env.svc.memory.remember("User lives in Ankara")
    env.fake.stream_script = [ChatResult(content="You live in Ankara.")]
    chat_events(env, "where do i live?")
    assert len(env.fake.stream_calls) == 1
