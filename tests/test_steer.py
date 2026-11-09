"""Steering: Kiwi Sense reads a reply, the facts of the turn decide whether it is wrong."""

from __future__ import annotations

import pytest
from test_backend import Env, chat_events, tool_call_result

from kiwibro import prompts, steer
from kiwibro.db import Database
from kiwibro.llm import ChatResult

WEB = frozenset({"web_search", "fetch_page"})
FILES = frozenset({"write_file", "read_file"})
CHATGPT = "I am ChatGPT, a large language model trained by OpenAI."


def kind(reply: str, facts: steer.Facts, **kwargs) -> str | None:
    verdict = steer.review(reply, facts, **kwargs)
    return verdict.kind if verdict else None


# ---- who is speaking ----

def test_speaking_as_another_assistant_is_caught():
    verdict = steer.review(CHATGPT, steer.Facts(question="who are you?"))
    assert verdict.kind == steer.OTHER_IDENTITY
    assert "You are Kiwibro" in verdict.note and verdict.sentence == CHATGPT


def test_talking_about_another_assistant_is_fine():
    facts = steer.Facts(question="what is ChatGPT and who makes it?")
    assert kind("ChatGPT is a chatbot made by OpenAI. It was released in 2022.", facts) is None
    # Asked to play one: the user brought the name up, so it is their wish.
    assert kind(CHATGPT, steer.Facts(question="pretend you are ChatGPT and introduce yourself")) is None


def test_naming_the_engine_as_kiwibro_is_fine():
    facts = steer.Facts(question="which model are you running on?")
    assert kind("I'm Kiwibro, and right now I run on the qwen3 model on your computer.", facts) is None


def test_taking_the_users_identity_is_caught():
    facts = steer.Facts(question="hi", user_names=("Yunus Emre",))
    verdict = steer.review("Hi! I'm Yunus, a developer from Istanbul.", facts)
    assert verdict.kind == steer.ADOPTED_USER and "Yunus is the user" in verdict.note
    assert kind("Hi Yunus! What shall we build today?", facts) is None


# ---- what it says it can and cannot do ----

def test_denying_a_tool_it_has_is_caught_only_when_the_tool_exists():
    reply = "I'm sorry, but I can't browse the internet."
    verdict = steer.review(reply, steer.Facts(question="what's the weather in Izmir right now?", tools=WEB))
    assert verdict.kind == steer.DENIAL and "`web_search`" in verdict.note
    assert kind(reply, steer.Facts(question="what's the weather in Izmir right now?", memory_on=False)) is None
    assert kind(reply, steer.Facts(question="open example.com", tools=WEB, tools_ran=1, memory_on=False)) is None


def test_denying_its_memory_is_caught_while_memory_is_on():
    reply = "I don't have the ability to remember previous conversations."
    assert kind(reply, steer.Facts(question="do you remember my name?")) == steer.DENIAL
    assert kind(reply, steer.Facts(question="do you remember my name?", memory_on=False)) is None


def test_claiming_an_action_without_a_tool_run_is_caught():
    reply = "I've saved the file to your desktop."
    with_tools = steer.review(reply, steer.Facts(question="save my notes to a file", tools=FILES))
    assert with_tools.kind == steer.UNBACKED_CLAIM and "no tool has run" in with_tools.note
    without = steer.review(reply, steer.Facts(question="save my notes to a file"))
    assert without.kind == steer.UNBACKED_CLAIM and "you have no tools" in without.note
    assert kind(reply, steer.Facts(question="save my notes to a file", tools=FILES, tools_ran=1)) is None


def test_a_true_memory_claim_is_left_alone():
    facts = steer.Facts(question="remember that my cat is called Pamuk", memory_saved=True)
    assert kind("I've saved that to memory.", facts) is None


def test_announcing_a_step_without_taking_it_is_caught_at_the_end_only():
    facts = steer.Facts(question="what is the newest Linux kernel?", tools=WEB)
    assert kind("Let me search the web for that.", facts) == steer.UNKEPT_PROMISE
    assert kind("Let me search the web for that.", facts, final=False) is None  # it did call the tool
    assert kind("Let me search the web for that.", steer.Facts(question="what is the newest Linux kernel?")) is None


def test_describing_the_request_instead_of_answering_is_caught():
    facts = steer.Facts(question="what's the weather like?")
    assert kind("The user is asking about the weather. I should answer briefly.", facts) == steer.META_LEAK


@pytest.mark.parametrize("reply", [
    "Hello! How can I help you today? Feel free to ask me anything you like.",
    "Great question! I would be happy to help you with that today, just give me a moment.",
    "I'm ready to help! Tell me what you would like to know.",
    "Absolutely, I can help with that! What exactly do you need?",
])
def test_an_offer_of_help_instead_of_an_answer_is_caught(reply: str):
    facts = steer.Facts(question="what time zone is Tokyo in?")
    verdict = steer.review(reply, facts)
    assert verdict.kind == steer.OFF_TOPIC and "what time zone is Tokyo in?" in verdict.note


@pytest.mark.parametrize("question, reply", [
    ("what time zone is Tokyo in?", "Japan Standard Time, nine hours ahead of UTC, with no daylight saving."),
    ("tell me a joke about farmers", "Why did the scarecrow win an award? Because he was outstanding in his field."),
    ("can you write a haiku for me?", "Morning fog lifts slow, a heron folds into reeds, the pond holds its breath."),
    ("how are you doing today?", "I'm doing well, thanks for asking. What would you like to work on?"),
    ("I feel a bit down today", "I'm sorry to hear that. Do you want to talk about it, or would a distraction help more?"),
    ("can you help me with my taxes?", "I can help with that. First, tell me whether you file jointly or alone."),
    ("could you explain that more simply?", "Imagine a library where every book has an index card that names its shelf."),
])
def test_real_answers_are_not_called_off_topic(question: str, reply: str):
    assert kind(reply, steer.Facts(question=question)) is None


def test_a_greeting_may_be_answered_with_an_offer_of_help():
    facts = steer.Facts(question="hello there, good morning!", small_talk=True)
    assert kind("Hello! How can I help you today? Feel free to ask me anything you like.", facts) is None


def test_answering_the_previous_question_again_is_caught():
    old = "How do I reverse a list in Python?"
    stale = "To reverse a list in Python, call reversed(items) or slice it with items[::-1]."
    fresh = steer.Facts(question="what should I cook for dinner tonight?", asked_before=(old,))
    assert kind(stale, fresh) == steer.OFF_TOPIC
    # A follow-up points back at the earlier topic, so the same reply is on topic there.
    follow_up = steer.Facts(question="can you show that again with a tuple?", asked_before=(old,))
    assert kind(stale, follow_up) is None
    assert kind("Try a quick tomato pasta with garlic, basil and plenty of olive oil.", fresh) is None


def test_other_languages_are_not_judged_by_the_english_probe():
    facts = steer.Facts(question="Türkiye'nin başkenti neresi?", tools=WEB)
    assert kind("Türkiye'nin başkenti Ankara'dır.", facts) is None


def test_ordinary_replies_pass():
    facts = steer.Facts(question="what is the capital of France?", tools=WEB | FILES, user_names=("Sam",))
    for reply in (
        "Paris is the capital of France.",
        "I can search the web for that if you like.",
        "You can save the file with Ctrl+S.",
        "Paris. Let me know if you need anything else.",
    ):
        assert kind(reply, facts) is None, reply


# ---- while the reply is still streaming ----

def test_early_check_waits_for_the_sentence_to_end():
    facts = steer.Facts(question="who are you?")
    verdict, checked = steer.early("I am ChatGPT, a large language", facts)
    assert verdict is None and checked == 0
    verdict, checked = steer.early(CHATGPT + " How can", facts)
    assert verdict.kind == steer.OTHER_IDENTITY and checked == 1


def test_early_check_reads_each_sentence_once_and_only_the_opening():
    facts = steer.Facts(question="what is the capital of France?")
    verdict, checked = steer.early("Paris is the capital. It is on the Seine.", facts)
    assert verdict is None and checked == 2
    assert steer.early("Paris is the capital. It is on the Seine. More", facts, checked) == (None, 2)
    long_text = "Paris is lovely. " * 40 + CHATGPT
    assert steer.early(long_text, facts) == (None, 0)


# ---- last resort and standing reminders ----

def test_repair_removes_only_the_wrong_sentences():
    facts = steer.Facts(question="what is the capital of France?")
    reply = f"{CHATGPT} Paris is the capital of France."
    verdict = steer.review(reply, facts)
    assert steer.repair(reply, verdict, facts) == "Paris is the capital of France."
    assert steer.repair(CHATGPT, verdict, facts) == ""


def test_reminders_follow_what_the_model_got_wrong_before():
    facts = steer.Facts(user_names=("Sam",), tools=WEB)
    assert steer.reminders({}, facts) == []
    lines = steer.reminders({steer.ADOPTED_USER: 1, steer.DENIAL: 2, steer.UNKEPT_PROMISE: 1, steer.META_LEAK: 1}, facts)
    assert len(lines) == 4 and "Sam is the user." in lines[0] and "Your tools work" in lines[1]
    assert steer.reminders({steer.DENIAL: 3}, steer.Facts()) == []  # no tools this turn: nothing to remind


def test_every_kind_has_a_plain_label():
    kinds = (steer.OTHER_IDENTITY, steer.ADOPTED_USER, steer.DENIAL, steer.UNBACKED_CLAIM,
             steer.UNKEPT_PROMISE, steer.META_LEAK, steer.OFF_TOPIC, steer.LOOP)
    assert set(steer.LABELS) == set(kinds)


# ---- what a model tends to get wrong (stored per model) ----

def test_traits_count_per_model_and_kind(tmp_path):
    db = Database(tmp_path / "kiwibro.sqlite3")
    db.initialize()
    assert db.note_trait("tiny-1", steer.DENIAL) == 1
    assert db.note_trait("tiny-1", steer.DENIAL) == 2
    db.note_trait("tiny-1", steer.LOOP)
    db.note_trait("other", steer.LOOP)
    assert db.traits("tiny-1") == {steer.DENIAL: 2, steer.LOOP: 1}
    assert db.traits("tiny-1", since="2999-01-01T00:00:00") == {}
    assert {row["model"] for row in db.all_traits()} == {"tiny-1", "other"}
    db.clear_traits("tiny-1")
    assert db.traits("tiny-1") == {} and db.traits("other") == {steer.LOOP: 1}
    db.clear_traits()
    assert db.all_traits() == []


# ---- in the agent loop ----

def _events_of(events, name):
    return [data for event, data in events if event == name]


def _stored_reply(env: Env) -> str:
    conversation_id = env.svc.db.list_conversations()[0]["id"]
    replies = [m for m in env.svc.db.list_messages(conversation_id) if m["role"] == "assistant"]
    return replies[-1]["content"]


def test_a_wrong_opening_is_cut_off_and_the_model_tries_again(env: Env):
    env.fake.stream_script = [
        ChatResult(content=CHATGPT),
        ChatResult(content="I'm Kiwibro, an AI agent running on your computer."),
    ]
    events, _raw = chat_events(env, "who are you?")
    assert len(env.fake.stream_calls) == 2
    correction = env.fake.stream_calls[1]["messages"][-1]
    assert correction["role"] == "user" and correction["content"].startswith("[Kiwibro: You are Kiwibro")
    assert {"text": "", "reasoning": ""} in _events_of(events, "rewrite")  # the wrong text leaves the screen
    assert _stored_reply(env) == "I'm Kiwibro, an AI agent running on your computer."
    assert _events_of(events, "done")[0]["inference"]["steered"] == [steer.OTHER_IDENTITY]
    assert env.svc.db.traits("fake-1") == {steer.OTHER_IDENTITY: 1}
    # The correction was for one request only: nothing of it is stored in the chat.
    conversation_id = env.svc.db.list_conversations()[0]["id"]
    assert not any("[Kiwibro:" in (m["content"] or "") for m in env.svc.db.list_messages(conversation_id))


def test_a_second_failure_keeps_only_what_is_true(env: Env):
    env.fake.stream_script = [
        ChatResult(content=f"{CHATGPT} Paris is the capital of France."),
        ChatResult(content=f"{CHATGPT} Paris is the capital of France."),
    ]
    chat_events(env, "what is the capital of France?")
    assert len(env.fake.stream_calls) == 2  # one corrected attempt, never a loop
    assert _stored_reply(env) == "Paris is the capital of France."


def test_nothing_true_left_falls_back_to_the_identity_line(env: Env):
    env.fake.stream_script = [ChatResult(content=CHATGPT), ChatResult(content=CHATGPT)]
    chat_events(env, "who are you?")
    assert _stored_reply(env) == prompts.IDENTITY_LINE


def test_a_claimed_action_gets_one_chance_to_become_real(env: Env):
    env.fake.stream_script = [
        ChatResult(content="I've saved the file to your workspace."),
        tool_call_result("write_file", {"path": "notes.txt", "content": "milk, eggs"}),
        ChatResult(content="Saved your list to notes.txt."),
    ]
    events, _raw = chat_events(env, "save my shopping list (milk, eggs) to notes.txt")
    assert (env.workspace / "notes.txt").read_text() == "milk, eggs"
    assert _stored_reply(env) == "Saved your list to notes.txt."
    assert steer.UNBACKED_CLAIM in _events_of(events, "done")[0]["inference"]["steered"]


def test_a_model_that_needed_correcting_gets_a_standing_reminder(env: Env):
    env.svc.db.note_trait("fake-1", steer.OTHER_IDENTITY)
    chat_events(env, "what is the capital of France?")
    newest = env.fake.stream_calls[-1]["messages"][-1]["content"]
    assert "Never speak as the user or as another assistant." in newest
    assert "Never speak as the user" not in env.fake.stream_calls[-1]["messages"][0]["content"]  # message 0 stays fixed


def test_sense_can_be_switched_off(env: Env):
    env.settings_patch({"sense_enabled": False})
    env.fake.stream_script = [ChatResult(content=CHATGPT)]
    events, _raw = chat_events(env, "who are you?")
    assert len(env.fake.stream_calls) == 1 and _stored_reply(env) == CHATGPT
    assert "steered" not in _events_of(events, "done")[0].get("inference", {})


@pytest.mark.parametrize("silent_again", [False, True])
def test_an_empty_turn_never_ends_empty(env: Env, silent_again: bool):
    answer = ChatResult(content="" if silent_again else "Paris.")
    env.fake.stream_script = [ChatResult(content=""), ChatResult(content=""), answer]
    events, _raw = chat_events(env, "what is the capital of France?")
    assert env.fake.stream_calls[-1]["messages"][-1]["content"] == prompts.ANSWER_NOW
    assert _stored_reply(env) == (prompts.NO_ANSWER if silent_again else "Paris.")
    assert _events_of(events, "done")[0]["message_id"] != _events_of(events, "meta")[0]["user_message_id"]


# ---- task adapters under test ----

def _with_adapter(env: Env, monkeypatch) -> list:
    """The model runs with one task adapter on; returns the list blamed adapters land in."""
    from kiwibro.adapters import Adapter

    coder = Adapter("/adapters/coder.gguf", "coder", "x", tasks=("*",))
    blamed: list = []
    monkeypatch.setattr(env.svc.llm, "has_adapters", lambda model: True)
    monkeypatch.setattr(env.svc.llm, "adapters_on", lambda model, messages, task=None: [coder])
    monkeypatch.setattr(env.svc.llm, "blame_adapters",
                        lambda found: blamed.extend(found) or {a.name: len(blamed) for a in found})
    return blamed


def test_a_reply_that_is_right_without_its_adapter_counts_against_the_adapter(env: Env, monkeypatch):
    from kiwibro.llm import ADAPTER_TASK, ADAPTERS_OFF

    blamed = _with_adapter(env, monkeypatch)
    env.fake.stream_script = [ChatResult(content=CHATGPT),
                              ChatResult(content="I'm Kiwibro, an AI agent running on your computer.")]
    events, _raw = chat_events(env, "who are you?")
    first, second = env.fake.stream_calls
    assert ADAPTER_TASK in first["extra"] and ADAPTERS_OFF not in first["extra"]
    assert second["extra"][ADAPTERS_OFF] is True
    assert second["messages"] == first["messages"]  # the same request, only the adapter is gone
    assert [a.name for a in blamed] == ["coder"]
    done = _events_of(events, "done")[0]["inference"]
    assert done["adapters_turned_down"] == [{"name": "coder", "trouble": 1, "strength": 1.0}]
    assert done["steered"] == [steer.OTHER_IDENTITY]
    assert env.svc.db.traits("fake-1") == {}  # the adapter's doing, not the model's habit


def test_wrong_with_and_without_the_adapter_is_the_models_own_doing(env: Env, monkeypatch):
    blamed = _with_adapter(env, monkeypatch)
    env.fake.stream_script = [ChatResult(content=CHATGPT), ChatResult(content=CHATGPT),
                              ChatResult(content="I'm Kiwibro, an AI agent running on your computer.")]
    events, _raw = chat_events(env, "who are you?")
    assert len(env.fake.stream_calls) == 3
    assert env.fake.stream_calls[2]["messages"][-1]["content"].startswith("[Kiwibro: You are Kiwibro")
    assert blamed == [] and "adapters_turned_down" not in _events_of(events, "done")[0]["inference"]
    assert env.svc.db.traits("fake-1") == {steer.OTHER_IDENTITY: 1}
    assert _stored_reply(env) == "I'm Kiwibro, an AI agent running on your computer."
