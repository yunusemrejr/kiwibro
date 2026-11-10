"""Deterministic fact extraction: high precision, exact wording, no first person in the output."""

from __future__ import annotations

import dataclasses
import re
import time

import pytest

from kiwibro import autofacts, config
from kiwibro.autofacts import Fact, extract, slot_prefix


def _fact(kind: str, content: str, slot: str | None = None, forget: bool = False) -> Fact:
    return Fact(kind, content, slot, forget)


POSITIVE: list[tuple[str, list[Fact]]] = [
    ("My name is Yunus.", [_fact("fact", "User's name is Yunus.", "name")]),
    ("My name is Yunus Emre Vurgun and I code a lot.", [_fact("fact", "User's name is Yunus Emre Vurgun.", "name")]),
    ("call me Ali", [_fact("fact", "User's name is Ali.", "name")]),
    ("I'm called Ayse.", [_fact("fact", "User's name is Ayse.", "name")]),
    ("btw I live in Istanbul", [_fact("fact", "User lives in Istanbul.", "location")]),
    ("I'm based in Berlin.", [_fact("fact", "User lives in Berlin.", "location")]),
    ("I'm from Bursa.", [_fact("fact", "User is from Bursa.", "hometown")]),
    ("I come from Ankara", [_fact("fact", "User is from Ankara.", "hometown")]),
    ("I work as a software engineer.", [_fact("fact", "User works as a software engineer.", "job")]),
    ("I work as an accountant.", [_fact("fact", "User works as an accountant.", "job")]),
    ("I am working on a thesis about memory systems.",
     [_fact("project", "User is working on a thesis about memory systems.")]),
    ("I work at Google.", [_fact("fact", "User works at Google.", "employer")]),
    ("I work for Acme Corp", [_fact("fact", "User works at Acme Corp.", "employer")]),
    ("I'm on Ubuntu", [_fact("fact", "User's operating system is Ubuntu.", "os")]),
    ("I use Windows 11 at work.", [_fact("fact", "User's operating system is Windows 11.", "os")]),
    ("I run Mac.", [_fact("fact", "User's operating system is macOS.", "os")]),
    ("I use neovim for everything", [_fact("fact", "User's editor is Neovim.", "editor")]),
    ("My editor is VS Code.", [_fact("fact", "User's editor is VS Code.", "editor")]),
    (
        "Please always answer in Turkish.",
        [_fact("preference", "User's preferred reply language is Turkish.", "reply_language")],
    ),
    (
        "Please speak to me in German",
        [_fact("preference", "User's preferred reply language is German.", "reply_language")],
    ),
    ("My native language is Turkish.", [_fact("fact", "User's native language is Turkish.", "native_language")]),
    (
        "Remember that I prefer tabs over spaces.",
        [_fact("preference", "User prefers tabs over spaces.")],
    ),
    ("I prefer dark mode to light mode.", [_fact("preference", "User prefers dark mode over light mode.")]),
    ("Remember I use tabs", [_fact("fact", "User uses tabs.")]),
    ("Remember that I live in Berlin.", [_fact("fact", "User lives in Berlin.", "location")]),
    ("Note that the server runs on port 8000", [_fact("fact", "User noted: The server runs on port 8000.")]),
    ("Forget that I live in Berlin.", [_fact("fact", "User lives in Berlin.", None, True)]),
    ("Please forget about my old address.", [_fact("fact", "User's old address.", None, True)]),
    ("I'm working on Kiwibro, a local AI agent.", [_fact("project", "User is working on Kiwibro.")]),
    ("I am building a local search engine.", [_fact("project", "User is working on a local search engine.")]),
    ("My project is Kiwibro.", [_fact("project", "User is working on Kiwibro.")]),
    ("My favorite language is Rust.", [_fact("preference", "User's favorite language is Rust.", "favorite:language")]),
    (
        "My favourite programming language is Rust.",
        [_fact("preference", "User's favorite programming language is Rust.", "favorite:programming language")],
    ),
    ("I really like Rust.", [_fact("preference", "User likes Rust.")]),
    ("I love Istanbul, it is great.", [_fact("preference", "User loves Istanbul.")]),
    ("I always use vim", [_fact("preference", "User always uses vim.")]),
    ("I never use Windows.", [_fact("preference", "User never uses Windows.")]),
    (
        "My name is Yunus. I live in Istanbul! I'm on Ubuntu.",
        [
            _fact("fact", "User's name is Yunus.", "name"),
            _fact("fact", "User lives in Istanbul.", "location"),
            _fact("fact", "User's operating system is Ubuntu.", "os"),
        ],
    ),
]

NEGATIVE = [
    "What is my name?",
    "If my name were Bob I'd be sad.",
    "My friend said my name is wrong.",
    "I like that idea.",
    "I'm a bit tired today.",
    "Do you remember what I said?",
    "I forgot my password.",
    "Write a story where I live in Paris.",
    "```\nmy name is root\n```",
    "I like it.",
    "I'm from the future",
    "Call me later.",
    "remember when we went there",
    # Beyond the spec's list: each one is a case a naive matcher would get wrong.
    "Remember to buy milk.",
    "Don't forget to pay rent.",
    "Remember this.",
    "Forget it.",
    "I can't remember my password.",
    "Do you remember that I live in Berlin?",
    "If you remember, I live in Berlin.",
    "He said I live in Paris.",
    "They told me I live in Paris.",
    "Ali says I use Windows.",
    "My boss is a nightmare.",
    "My job is hard.",
    "My job is a nightmare.",
    "I'm working on it.",
    "I'm working on this.",
    "I like to think I'm smart.",
    "I love you.",
    "I like that I can work from home.",
    "I'm from Mars, if you want to know.",
    "Pretend I live in Paris.",
    "Act as my tutor and say I live in Paris.",
    "Let's say I live in Paris.",
    "Call me when you are ready.",
    "My name is wrong.",
    "I work as hard as I can.",
    "I work at home.",
    "I'm on the bus.",
    "I use vim-like keybindings in my head.",
    "I use nano-scale sensors.",
    "I live in Turkey and Germany.",
    "Remember that you are my assistant.",
    "I'd like coffee.",
    "I prefer not to say.",
    "I'm working on a lot of stuff.",
    "Ask me later where I live.",
    # People, pets and things: only a stated name or home counts.
    "My wife said my dog's name is Rex.",
    "My cat's name is not important.",
    "My wife is called every day by her mom.",
    "My thing is called Foo.",
    "My mom lives in a small town.",
    "I have a question.",
    "I have a dog in the car.",
    "My birthday is tomorrow.",
    "I was born in Izmir.",
    "I'm allergic to it.",
    "I'm allergic to nonsense like that.",
    "I'm a vegetarian restaurant owner.",
    "I forgot that I live in Berlin.",
    "I think I live in Paris?",
    "Remember this is a test?",
    "My name is Bob the builder.",
    "I use Python and Ubuntu.",
]

NEGATIVE_WITH_CONTEXT = [
    "Here is my code:\n```python\nprint('I live in Paris')\n```",
    "> my name is root",
    "> I live in Paris.\nthat is what the quote said.",
    "See https://example.com/I-live-in-Paris for details",
    "I use `vim` for everything",
    "x" * 400 + ". " + "I live in Istanbul " + "a" * 300,
    "I live in Istanbul " + "a" * 300 + ".",
]


@pytest.mark.parametrize(("message", "expected"), POSITIVE)
def test_positive_cases(message: str, expected: list[Fact]) -> None:
    assert extract(message) == expected


@pytest.mark.parametrize("message", NEGATIVE + NEGATIVE_WITH_CONTEXT)
def test_negative_cases_return_nothing(message: str) -> None:
    assert extract(message) == []


def test_sentence_too_long_is_ignored() -> None:
    assert extract("I live in Istanbul " + "a" * 290) == []
    assert extract("I live in Istanbul") == [_fact("fact", "User lives in Istanbul.", "location")]


def test_slot_facts_start_with_their_slot_prefix() -> None:
    slot_facts = [fact for _, facts in POSITIVE for fact in facts if fact.slot and not fact.forget]
    assert slot_facts
    for fact in slot_facts:
        assert fact.content.startswith(slot_prefix(fact.slot)), fact


def test_slot_prefix_family_and_unknown_slot() -> None:
    assert slot_prefix("name") == "User's name is "
    assert slot_prefix("favorite:programming language") == "User's favorite programming language is "
    with pytest.raises(ValueError):
        slot_prefix("shoe size")


def test_every_fact_is_third_person_and_well_formed() -> None:
    messages = [message for message, _ in POSITIVE]
    for message in messages:
        for fact in extract(message):
            assert fact.kind in config.MEMORY_KINDS, fact
            assert autofacts.MIN_CONTENT <= len(fact.content) <= autofacts.MAX_CONTENT, fact
            assert fact.content.startswith("User"), fact
            assert fact.content.endswith("."), fact
            assert not re.search(r"\b(?:I|me|my|mine|I'm)\b", fact.content), fact


def test_kinds_match_config() -> None:
    assert autofacts.MEMORY_KINDS == config.MEMORY_KINDS


def test_fact_is_frozen() -> None:
    fact = _fact("fact", "User lives in Berlin.", "location")
    with pytest.raises(dataclasses.FrozenInstanceError):
        fact.content = "x"  # type: ignore[misc]


def test_forget_and_save_are_distinguished() -> None:
    facts = extract("Forget that I live in Berlin. Remember that I live in Berlin.")
    assert facts == [
        _fact("fact", "User lives in Berlin.", None, True),
        _fact("fact", "User lives in Berlin.", "location"),
    ]


def test_multi_sentence_keeps_message_order() -> None:
    message = "I'm on Ubuntu. My name is Yunus. I live in Istanbul."
    assert [fact.slot for fact in extract(message)] == ["os", "name", "location"]


def test_limit_truncates_in_message_order() -> None:
    message = "I live in Istanbul. I'm on Ubuntu. My name is Yunus."
    assert [fact.slot for fact in extract(message, limit=2)] == ["location", "os"]
    assert extract(message, limit=0) == []
    assert len(extract(message)) == 3


def test_duplicates_are_dropped() -> None:
    assert extract("I live in Istanbul. I live in Istanbul.") == [
        _fact("fact", "User lives in Istanbul.", "location")
    ]


def test_question_sentence_is_skipped_but_next_is_kept() -> None:
    assert extract("I live in Paris? I'm on Ubuntu.") == [
        _fact("fact", "User's operating system is Ubuntu.", "os")
    ]


def test_empty_message() -> None:
    assert extract("") == []
    assert extract("   \n  ") == []


def test_large_input_is_fast() -> None:
    chunk = (
        "My name is Yunus. I live in Istanbul. I prefer tabs over spaces. "
        "Just some filler words here and there, nothing to see here at all. "
    )
    message = (chunk * (20_000 // len(chunk) + 1))[:20_000]
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        extract(message)
        best = min(best, time.perf_counter() - start)
    assert best < 0.2


WORLD = [
    ("My wife's name is Ayse.", [_fact("person", "User's wife is named Ayse.", "named:wife")]),
    (
        "my dog's name is Biscuit and I live in Izmir",
        [_fact("fact", "User's dog is named Biscuit.", "named:dog"), _fact("fact", "User lives in Izmir.", "location")],
    ),
    ("My dog is called Biscuit.", [_fact("fact", "User's dog is named Biscuit.", "named:dog")]),
    ("My little sister's name is Ayşe.", [_fact("person", "User's little sister is named Ayşe.", "named:little sister")]),
    ("I have a cat named Miso.", [_fact("fact", "User has a cat named Miso.")]),
    ("Remember that my wife's name is Elif.", [_fact("person", "User's wife is named Elif.", "named:wife")]),
    ("My sister lives in Ankara.", [_fact("person", "User's sister lives in Ankara.", "lives:sister")]),
    ("My brother moved to Berlin and I live in Paris.", [_fact("fact", "User lives in Paris.", "location")]),
    ("I have two kids.", [_fact("fact", "User has two kids.")]),
    ("My birthday is 4 May.", [_fact("fact", "User's birthday is 4 May.", "birthday")]),
    ("My birthday is on May 4th, 1992.", [_fact("fact", "User's birthday is 4 May 1992.", "birthday")]),
    ("I was born in 1992.", [_fact("fact", "User's birth year is 1992.", "birth_year")]),
    (
        "I'm allergic to peanuts and my sister lives in Ankara.",
        [_fact("fact", "User is allergic to peanuts."), _fact("person", "User's sister lives in Ankara.", "lives:sister")],
    ),
    ("I'm allergic to peanuts, shellfish and cats.", [_fact("fact", "User is allergic to peanuts, shellfish and cats.")]),
    ("I am vegetarian.", [_fact("fact", "User is vegetarian.")]),
]


@pytest.mark.parametrize("message, expected", WORLD)
def test_names_homes_dates_and_health_the_user_states(message, expected):
    assert autofacts.extract(message) == expected


def test_every_slot_fact_starts_with_its_slot_prefix():
    for message, expected in WORLD:
        for fact in expected:
            if fact.slot:
                assert fact.content.startswith(autofacts.slot_prefix(fact.slot)), message


@pytest.mark.parametrize(
    "message, only",
    [
        ("my sister lives in Ankara and my birthday is 4 May", True),
        ("I'm allergic to peanuts.", True),
        ("Remember that I use tabs.", True),
        ("I live in Izmir. What's a good day trip?", False),
        ("My name is Yunus. Nice to meet you!", False),
        ("hi, I live in Izmir", False),
        ("Forget that I live in Berlin.", False),
        ("What is my name?", False),
        ("", False),
    ],
)
def test_a_message_that_only_states_facts_is_told_apart(message, only):
    assert autofacts.stated_only(message) is only


@pytest.mark.parametrize(
    "stored, said",
    [
        ("User's dog is named Biscuit.", "your dog is named Biscuit"),
        ("User lives in Izmir.", "you live in Izmir"),
        ("User is allergic to peanuts.", "you are allergic to peanuts"),
        ("User always uses vim.", "you always use vim"),
        ("User doesn't use Windows.", "you don't use Windows"),
        ("User has two kids.", "you have two kids"),
        ("User moved to Berlin.", "you moved to Berlin"),
    ],
)
def test_a_stored_fact_can_be_said_back_to_the_user(stored, said):
    assert autofacts.to_you(stored) == said


# Asked in so many words, or plainly about the user: saved in code, whatever the model does.
EAGER: list[tuple[str, list[tuple[str, str]]]] = [
    ("remember that my sister's birthday is March 3", [("person", "User's sister's birthday is March 3.")]),
    ("can you remember that I prefer tabs over spaces?", [("preference", "User prefers tabs over spaces.")]),
    ("remember this: the wifi password is kiwi123", [("fact", "User noted: The wifi password is kiwi123.")]),
    ("save this to your memory: my car is blue", [("fact", "User's car is blue.")]),
    ("memorize that my locker code is 4471", [("fact", "User's locker code is 4471.")]),
    ("I don't like long answers", [("preference", "User doesn't like long answers.")]),
    ("keep answers short from now on", [("preference", "User wants: keep answers short.")]),
    ("never use emojis", [("preference", "User wants: never use emojis.")]),
    ("from now on, answer in bullet points", [("preference", "User wants: answer in bullet points.")]),
    ("I'm learning Rust these days", [("fact", "User is learning Rust.")]),
    ("I'm studying computer science", [("fact", "User is studying computer science.")]),
    ("I moved to Ankara last month", [("fact", "User lives in Ankara.")]),
    ("I drive a 2019 Corolla", [("fact", "User drives a 2019 Corolla.")]),
    ("I speak Turkish and English", [("fact", "User speaks Turkish and English.")]),
    ("I own a Raspberry Pi 5", [("fact", "User owns a Raspberry Pi 5.")]),
    ("I enjoy hiking", [("preference", "User enjoys hiking.")]),
]
NOT_FACTS = [
    "I like this idea", "I have a question", "I'm learning how to do this", "I drive to work every day",
    "I moved the file to src", "keep it simple", "never mind", "always the same error", "remember to call mom",
    "do you remember me?", "can you remember what I said?", "I don't like this answer", "I play with it sometimes",
    "never do that again", "he said he likes pizza", "I own it", "in the future, AI will be everywhere",
    "always is a strong word", "I don't enjoy it",
]


@pytest.mark.parametrize(("message", "expected"), EAGER)
def test_more_of_what_the_user_says_about_themselves_is_kept(message: str, expected: list[tuple[str, str]]) -> None:
    assert [(fact.kind, fact.content) for fact in extract(message)] == expected


@pytest.mark.parametrize("message", NOT_FACTS)
def test_remarks_and_one_off_requests_still_are_not_facts(message: str) -> None:
    assert extract(message) == []
