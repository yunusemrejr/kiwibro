"""The Kiwi Sense helpers at work: English only, search, memory, time ranges, topic shifts.

The cases below were written by hand and never seen in training."""

from __future__ import annotations

from datetime import date

import pytest
from test_backend import Env, chat_events

from kiwibro import readings, sense
from kiwibro.llm import ChatResult
from kiwibro.sense.helpers import Language, judged_part, lang_text, spans

TODAY = date(2026, 11, 11)  # a Wednesday


@pytest.mark.parametrize("phrase, meant", [
    ("yesterday", "10 November 2026"),
    ("tomorrow morning", "12 November 2026"),
    ("last week", "2 November 2026 to 8 November 2026"),
    ("next week", "16 November 2026 to 22 November 2026"),
    ("this weekend", "14 November 2026 to 15 November 2026"),
    ("3 days ago", "8 November 2026"),
    ("the past 2 weeks", "28 October 2026 to 11 November 2026"),
    ("last friday", "6 November 2026"),
    ("next monday", "16 November 2026"),
    ("since March", "1 March 2026 to 11 November 2026"),
    ("next month", "December 2026"),
    ("last year", "2025"),
])
def test_time_phrases_become_dates(phrase, meant):
    assert readings.resolve_time(phrase, TODAY) == meant


def test_a_bare_year_needs_no_working_out():
    assert readings.resolve_times(["in 2019", "last week"], TODAY) == [("last week", "2 November 2026 to 8 November 2026")]


def test_spans_read_bio_tags():
    words = "write it in 3 bullet points about the eiffel tower by tomorrow".split()
    tags = ["O", "O", "B-ASK", "I-ASK", "I-ASK", "I-ASK", "O", "O", "B-KEY", "I-KEY", "B-TIME", "I-TIME"]
    assert spans(words, tags) == {"ASK": ["in 3 bullet points"], "KEY": ["eiffel tower"], "TIME": ["by tomorrow"]}


def test_the_language_judge_leaves_out_code_links_and_quotes():
    text = 'what does "ich liebe dich" mean? see https://de.wikipedia.org/wiki/Liebe ```print("hola")```'
    assert "liebe dich" not in judged_part(text) and "wikipedia" not in judged_part(text)
    assert lang_text("Ça VA très bien!") == "ça va très bien !"


# ---- the English-only reply, written by code ----

def test_a_message_not_in_english_is_answered_by_code(env: Env, monkeypatch):
    monkeypatch.setattr(readings, "language", lambda message: Language(False, "tr", "Turkish", 0.98, True))
    events, _raw = chat_events(env, "Bugün hava nasıl olacak?")
    assert env.fake.stream_calls == []  # no model was asked
    replies = [data["text"] for event, data in events if event == "delta"]
    assert replies == [readings.english_only_reply(Language(False, "tr", "Turkish"))]
    assert "Turkish" in replies[0] and "English" in replies[0]
    done = [data for event, data in events if event == "done"][0]
    assert done["inference"]["english_only"] == "tr"
    assert [event for event, _ in events][0] == "meta"


def test_english_only_can_be_switched_off(env: Env, monkeypatch):
    monkeypatch.setattr(readings, "language", lambda message: Language(False, "tr", "Turkish", 0.98, True))
    env.settings_patch({"english_only": False})
    env.fake.stream_script = [ChatResult(content="Güneşli.")]
    chat_events(env, "Bugün hava nasıl olacak?")
    assert len(env.fake.stream_calls) == 1


def test_the_reading_rides_in_the_turn_note(env: Env, monkeypatch):
    found = readings.TurnReading(ranges=[("last week", "2 November 2026 to 8 November 2026")])
    monkeypatch.setattr(readings, "read", lambda message, before="": found)
    env.fake.stream_script = [ChatResult(content="Done.")]
    chat_events(env, "summarise what I did last week")
    newest = env.fake.stream_calls[-1]["messages"][-1]["content"]
    assert "Dates the message means: last week = 2 November 2026 to 8 November 2026." in newest


# ---- the trained helpers themselves (held-out, handwritten) ----

def _helpers():
    reader = sense.shared()
    found = reader.helpers if reader is not None else None
    if found is None or not all(index is not None for index in found.tags.values()):
        pytest.skip("this data file carries no helpers")
    return found


ENGLISH = [
    "hey, can you help me plan a trip to lisbon next month?",
    "what does 'je ne sais quoi' mean?",
    "how do I say 'good morning' in Turkish?",
    "fix this: ```def f(x): return x+1``` it throws TypeError",
    "my flat in Berlin is too cold, any tips?",
    "translate to english: merhaba, nasılsın?",
    "ok",
    "ciao",
    "thanks!!",
    "lol nice",
    "is müsli healthy for breakfast?",
    "Summarize the plot of Les Misérables in 5 bullet points",
    "whats the weather like in são paulo",
    "git push -f origin main keeps failing with 403, why",
    "turn 1: " + "kiwi " * 50,
]
FOREIGN = [
    ("Bugün hava nasıl olacak, şemsiye almalı mıyım?", "tr"),
    ("Kannst du mir bitte helfen, eine E-Mail an meinen Chef zu schreiben?", "de"),
    ("¿Cuál es la mejor manera de aprender a programar en Python?", "es"),
    ("Je voudrais réserver une table pour deux personnes ce soir.", "fr"),
    ("Как мне настроить сервер nginx для моего сайта?", "ru"),
    ("请帮我写一封邮件给我的老板", "zh"),
    ("Qual é a capital da Austrália e quantas pessoas moram lá?", "pt"),
    ("Ik wil graag weten hoe ik mijn laptop sneller kan maken.", "nl"),
    ("como puedo hacer que mi codigo sea mas rapido", "es"),
    ("bunu nasil yapabilirim bilmiyorum yardim eder misin", "tr"),
    ("Potresti spiegarmi come funziona la fotosintesi?", "it"),
    ("この文章を英語に翻訳してください", "ja"),
]


@pytest.mark.parametrize("text", ENGLISH)
def test_english_is_never_turned_back(text):
    assert _helpers().language(text).english, text


def test_other_languages_are_caught():
    helpers = _helpers()
    caught = [text for text, _code in FOREIGN if not helpers.language(text).english]
    assert len(caught) == len(FOREIGN), set(t for t, _ in FOREIGN) - set(caught)


MESSAGES = [  # (message, field, wanted)
    ("what's the weather in Paris tomorrow?", "web", {"live"}),
    ("who won the champions league final this year?", "web", {"live", "lookup"}),
    ("search the web for reviews of the framework laptop 16", "web", {"asked"}),
    ("write a haiku about autumn leaves", "web", {"none"}),
    ("explain how a binary search works", "web", {"none"}),
    ("what did I save in my notes about the garden?", "web", {"own"}),
    ("I have two cats called Miso and Tofu", "keep", {"fact"}),
    ("I prefer short answers without emojis", "keep", {"preference"}),
    ("if I were a doctor, what would I earn?", "keep", {"unsure", "none"}),
    ("I'm so tired today", "keep", {"passing"}),
    ("remember that my sister's birthday is on 4 May", "intent", {"remember"}),
    ("thanks, that worked!", "intent", {"feedback", "chat"}),
    ("go on", "intent", {"continue"}),
    ("what time is it?", "time", {"clock"}),
    ("what happened in the news last week?", "time", {"recent"}),
    ("how do I fix a segfault in my C program", "topic", {"software"}),
]


def test_message_readings_agree_with_hand_labels():
    helpers = _helpers()
    wrong = [(text, name, getattr(helpers.read(text), name)) for text, name, wanted in MESSAGES
             if getattr(helpers.read(text), name) not in wanted]
    assert len(wrong) <= 2, wrong


def test_demands_and_times_are_found():
    reading = _helpers().read("give me 3 bullet points on what happened in tech last week, under 100 words")
    assert any("bullet" in ask for ask in reading.asks), reading.asks
    assert any("last week" in phrase for phrase in reading.times), reading.times


BEFORE = "how do I read a csv file in python? Use pandas.read_csv('file.csv') to load it into a DataFrame."


def test_new_topics_are_told_apart():
    helpers = _helpers()
    assert helpers.shift(BEFORE, "what's a good recipe for banana bread?")[0] == "new"
    assert helpers.shift(BEFORE, "why does it need pandas?")[0] == "followup"
    # only a plain change of topic gets the "answer it on its own" line
    assert not readings.new_topic(readings.TurnReading(shift=helpers.shift(BEFORE, "explain the second line")))


@pytest.mark.xfail(reason="short elliptical follow-ups come out close to a tie; the shift helper needs more of them in training")
def test_elliptical_follow_ups():
    assert _helpers().shift(BEFORE, "and in rust?")[0] == "followup"


def test_relate_tells_bearing_items_from_lookalikes():
    helpers = _helpers()
    assert helpers.relates("what should I cook tonight?", "User is vegetarian.") > 0.5
    assert helpers.relates("what should I cook tonight?", "User works as a sound engineer.") < 0.5
