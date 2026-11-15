"""Tests for the exact letter and word answers (kiwibro/textfacts.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kiwibro import compute, textfacts  # noqa: E402


@pytest.mark.parametrize("message, expected", [
    ("how many r's are in strawberry", [('the letter "r" in "strawberry"', "3")]),
    ("How many times does the letter e appear in \"Mississippi experience\"?",
     [('the letter "e" in "Mississippi experience"', "4")]),
    ("how many s in possess", [('the letter "s" in "possess"', "4")]),
    ("how many words are in \"the quick brown fox\"", [('words in "the quick brown fox"', "4")]),
    ("how many letters in the word banana", [('letters in "banana"', "6")]),
    ("spell banana backwards", [('"banana" backwards', "ananab")]),
    ("reverse the word kiwi", [('"kiwi" backwards', "iwik")]),
])
def test_letter_and_word_questions_are_answered_in_code(message, expected):
    assert textfacts.worked(message) == expected


@pytest.mark.parametrize("message", [
    "how many people are in Turkey", "spell necessary", "reverse this list for me", "hello",
    "how many days are in a week", "what is 2 + 2", "write a function that reverses a string", "",
])
def test_other_messages_give_nothing(message):
    assert textfacts.worked(message) == []


def test_a_reply_is_checked_against_the_result():
    (_asked, count), = textfacts.worked("how many r's are in strawberry")
    assert compute.gives("There are 3 r's in strawberry.", count)
    assert not compute.gives("There are 2 r's in strawberry.", count)
