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


CSV = "how do I read a csv file in python? Use pandas.read_csv('file.csv') to load it into a DataFrame."
TRIP = ("I'm going to Lisbon in March, what should I see? Lisbon's highlights are the Alfama quarter, "
        "Belém Tower, the Jerónimos Monastery and the view from the São Jorge castle.")
PROTEIN = ("how much protein should I eat a day? Most adults need about 0.8 g per kg of body weight, "
           "more if you train hard.")
SHIFTS = [  # (before, message, what it is)
    (CSV, "and in rust?", "followup"),
    (CSV, "what about rust?", "followup"),
    (CSV, "can you show the same in rust", "followup"),
    (CSV, "why does it need pandas?", "followup"),
    (CSV, "explain the second line", "followup"),
    (CSV, "shorter please", "followup"),
    (CSV, "does that work for big files?", "followup"),
    (CSV, "and if the file has no header?", "followup"),
    (TRIP, "how many days do I need for that?", "followup"),
    (TRIP, "and porto?", "followup"),
    (TRIP, "which of those is free?", "followup"),
    (PROTEIN, "and if I lift weights?", "followup"),
    (PROTEIN, "what does that mean for 70 kg?", "followup"),
    (CSV, "how do I read an excel file in python with pandas?", "same"),
    (TRIP, "what's the best way to get from Lisbon airport to the city centre?", "same"),
    (PROTEIN, "which foods have the most protein per calorie?", "same"),
    (CSV, "what's a good recipe for banana bread?", "new"),
    (CSV, "who won the world cup in 2018?", "new"),
    (CSV, "my cat keeps sneezing, should I worry?", "new"),
    (TRIP, "how do I reset my router?", "new"),
    (TRIP, "explain quantum entanglement simply", "new"),
    (PROTEIN, "fix this python error: KeyError 'name'", "new"),
    (PROTEIN, "write a short poem about the sea", "new"),
]


@pytest.mark.parametrize("before, message, meant", SHIFTS, ids=[message for _, message, _ in SHIFTS])
def test_follow_ups_and_new_topics_are_told_apart(before, message, meant):
    found, sure = _helpers().shift(before, message)
    reading = readings.TurnReading(shift=(found, sure))
    if meant == "new":
        assert found == "new", (message, found, round(sure, 2))
    else:  # follow-up or same subject: looked up with the exchange before, never told to drop it
        assert readings.continues(reading) and not readings.new_topic(reading), (message, found, round(sure, 2))


def test_most_follow_ups_are_read_as_follow_ups():
    """Follow-up or same subject is the helper's least sure call (a follow-up read as "same"
    still carries the exchange before into lookups). At most one handwritten follow-up may slip."""
    helpers = _helpers()
    slipped = [message for before, message, meant in SHIFTS
               if meant == "followup" and helpers.shift(before, message)[0] != "followup"]
    assert len(slipped) <= 1, slipped


def test_relate_tells_bearing_items_from_lookalikes():
    helpers = _helpers()
    assert helpers.relates("what should I cook tonight?", "User is vegetarian.") > 0.5
    assert helpers.relates("what should I cook tonight?", "User works as a sound engineer.") < 0.5


def test_a_message_that_carries_on_is_looked_up_with_the_exchange_before(env: Env, monkeypatch):
    lookups = []
    memory = env.svc.agent.memory
    found = memory.build_block
    monkeypatch.setattr(memory, "build_block", lambda query, **kw: (lookups.append(query), found(query, **kw))[1])
    for shift in (("same", 0.6), ("followup", 0.5), ("new", 0.95)):
        monkeypatch.setattr(readings, "read", lambda message, before="": readings.TurnReading())
        env.fake.stream_script = [ChatResult(content="Use pandas.read_csv('file.csv').")]
        events, _raw = chat_events(env, "how do I read a csv file in python?")
        monkeypatch.setattr(readings, "read", lambda message, before="", s=shift: readings.TurnReading(shift=s))
        env.fake.stream_script = [ChatResult(content="Done.")]
        chat_events(env, "does it handle big files?", events[0][1]["conversation_id"])
    carried, followed, alone = lookups[1::2]
    assert "read a csv file" in carried and "read a csv file" in followed
    assert alone == "does it handle big files?"


@pytest.mark.parametrize("message, local", [
    ("write smart.txt", True),
    ("summarise ~/projects/kiwi/notes.md", True),
    ("what does https://example.com/status say?", True),
    ("open ./src/main.py", True),
    ("what's new in node.js 24?", False),
    ("bitcoin price now", False),
    ("is 1/2 cup enough?", False),
])
def test_files_paths_and_links_are_for_kiwibros_own_tools(message, local):
    assert readings.names_a_file_or_link(message) is local
