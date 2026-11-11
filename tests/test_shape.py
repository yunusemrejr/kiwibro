"""The form of a finished reply (kiwibro/shape.py)."""

import pytest

from kiwibro import shape

LIST = """Here is the answer:

**The latest items:**

- A lighthouse keeper in Brevik has won the national chess title.
- The ferry company Nordvind cancelled all Sunday sailings
- A lighthouse keeper in Brevik has won the national chess title.
- Nordvind's chief has been asked to explain the cancellations.
- a lighthouse keeper in Brevik has won the national chess title"""


def test_a_line_said_again_is_kept_once():
    kept = shape.without_repeats(LIST)
    assert kept.lower().count("lighthouse keeper") == 1
    assert "Nordvind cancelled" in kept and "asked to explain" in kept


def test_code_tables_and_short_lines_may_repeat():
    reply = "Yes.\nYes.\n```\nprint('the same line of code')\nprint('the same line of code')\n```\n| a long table cell here | 1 |\n| a long table cell here | 1 |"
    assert shape.without_repeats(reply) == reply


@pytest.mark.parametrize("message", [
    "summarize them as one pararaph", "in a short paragraf please", "one paragraph", "sum it up in a single paragraph",
    "put that into a paragraph",
])
def test_one_paragraph_is_heard(message):
    assert shape.wants_one_paragraph(message)


@pytest.mark.parametrize("message", [
    "give me three paragraphs", "what is a paragraph", "write 2 paragraphs about cats", "summarize them",
    "one more thing about the parade", "how long should a paragraph be",
])
def test_other_messages_do_not_ask_for_one_paragraph(message):
    assert not shape.wants_one_paragraph(message)


def test_the_demands_a_helper_read_count_too():
    assert shape.wants_one_paragraph("shorter", ("one paragraph",))


def test_a_list_becomes_one_paragraph():
    text = shape.as_one_paragraph(shape.without_repeats(LIST))
    assert "\n" not in text and "-" not in text.split()[0]
    assert text == ("A lighthouse keeper in Brevik has won the national chess title. The ferry company Nordvind "
                    "cancelled all Sunday sailings. Nordvind's chief has been asked to explain the cancellations.")


def test_one_paragraph_and_code_are_left_alone():
    assert shape.as_one_paragraph("Just one line.") == "Just one line."
    assert shape.as_one_paragraph("```py\nx = 1\n```\nmore") == "```py\nx = 1\n```\nmore"
