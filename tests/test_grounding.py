"""Grounding: a reply written from sources is checked against them, piece by piece."""

from __future__ import annotations

import pytest

from kiwibro import grounding

BANK = ["Commonwealth Bank reported a rise in profit. The bank said trading was strong."]
QUIET = ["The market was quiet."]


def missing(reply: str, sources: list[str], known: tuple[str, ...] = ()) -> list[str]:
    return grounding.missing(grounding.check(reply, sources, known))


def pieces(reply: str, sources: list[str]) -> list[tuple[int, str, tuple[str, ...]]]:
    return [(finding.line, finding.piece, finding.missing) for finding in grounding.check(reply, sources)]


# ---- names ----

def test_a_name_in_the_sources_is_backed():
    assert missing("Commonwealth Bank reported a rise in profit.", BANK) == []


def test_a_name_the_sources_never_mention_is_flagged():
    reply = "The Australian Broadcasting Corporation said profit rose."
    assert missing(reply, BANK) == ["Australian Broadcasting Corporation"]


# ---- figures ----

def test_a_figure_the_sources_never_give_is_flagged():
    assert missing("The bank reported $32,585,678,799 in profit.", BANK) == ["$32,585,678,799"]


def test_a_figure_the_sources_give_is_backed():
    assert missing("The bank made $1.2 billion.", ["The bank made $1.2 billion last year."]) == []


@pytest.mark.parametrize("reply", ["There are three stories.", "There are 5 results.", "There are 12 items."])
def test_small_plain_counts_are_never_flagged(reply):
    assert missing(reply, QUIET) == []


def test_a_count_above_the_small_limit_is_flagged():
    assert missing(f"There are {grounding.SMALL_NUMBER + 1} items.", QUIET) == [str(grounding.SMALL_NUMBER + 1)]


# ---- scale words and rounding ----

@pytest.mark.parametrize("reply", ["The fund is worth 2.5 million dollars.", "The fund is worth 2.5M dollars."])
def test_scale_words_match_however_they_are_written(reply):
    assert missing(reply, ["The fund is worth 2.5 million dollars."]) == []


def test_a_figure_within_the_rounding_margin_is_backed():
    assert missing("It sold 1,010 units.", ["It sold 1,000 units."]) == []


def test_a_figure_further_off_than_the_margin_is_flagged():
    assert missing("It sold 1,200 units.", ["It sold 1,000 units."]) == ["1,200"]


# ---- links ----

def test_a_url_from_the_sources_is_backed():
    assert missing("See https://example.com/news/story for more.", ["Read https://example.com/news/story today."]) == []


def test_an_invented_url_is_flagged():
    reply = "See https://invented.example.org/fake for more."
    assert missing(reply, ["Read https://example.com/news/story today."]) == ["https://invented.example.org/fake"]


def test_the_front_page_of_a_host_in_the_sources_is_backed():
    assert missing("See https://example.com for more.", ["Read https://example.com/news/story today."]) == []


# ---- what the user wrote ----

def test_a_name_the_user_wrote_is_backed_without_a_source():
    reply = "The tool was written by Selin Kaya."
    sources = ["The tool was written by a developer."]
    assert missing(reply, sources, ("Selin Kaya",)) == []
    assert missing(reply, sources) == ["Selin Kaya"]


# ---- words capitalised for other reasons ----

def test_a_sentence_opener_is_not_a_name():
    assert missing("Latest reports say prices rose.", ["Prices rose sharply this year."]) == []


@pytest.mark.parametrize(
    "reply, word",
    [
        ("On Monday the market was quiet.", "Monday"),
        ("In November the market was quiet.", "November"),
        ("Dr Smith spoke at the event.", "Dr"),
        ("The AI team met.", "AI"),
        ("The CEO spoke.", "CEO"),
    ],
)
def test_calendar_words_titles_and_common_words_are_not_flagged(reply, word):
    assert word not in missing(reply, ["A meeting was held."])


# ---- initials and expansions ----

def test_an_expansion_of_an_acronym_in_the_sources_is_backed():
    assert missing("The Radiological Society of North America met.", ["RSNA held its annual meeting."]) == []


def test_initials_of_a_name_the_sources_spell_out_are_backed():
    assert missing("The NT government acted.", ["The Northern Territory government acted."]) == []


def test_words_starting_with_the_same_letters_do_not_back_an_expansion():
    reply = "The Australian Broadcasting Corporation reported it."
    assert missing(reply, ["Adobe Brings ChatGPT to its apps"]) == ["Australian Broadcasting Corporation"]


# ---- joined words ----

def test_a_joined_token_matches_the_two_words_of_the_source():
    assert missing("OpenAI released GPT-5 today.", ["OpenAI released GPT 5 today."]) == []


def test_a_common_prefix_in_a_hyphenated_word_is_not_flagged():
    assert missing("The AI-related rules are new.", ["AI rules are new."]) == []


# ---- headings ----

@pytest.mark.parametrize("reply", ["## 2. Recent Breakthroughs\n\nNothing much happened.",
                                   "**2. Recent Breakthroughs**\n\nNothing much happened."])
def test_a_heading_of_ordinary_capitalised_words_is_not_flagged(reply):
    assert missing(reply, ["Nothing much happened."]) == []


# ---- tables ----

TABLE = "| Name | Figure |\n| --- | --- |\n| Zorblax Corp | 44,321 |\n| Quentaris Ltd | 77,888 |"


def test_a_table_of_unknown_names_and_figures_is_flagged_row_by_row():
    found = pieces(TABLE, ["A short report about markets."])
    assert [line for line, _, _ in found] == [2, 3]
    assert "44,321" in found[0][2]
    assert "77,888" in found[1][2]


def test_a_reply_that_is_only_a_table_is_stripped_to_nothing():
    assert grounding.strip(TABLE, grounding.check(TABLE, ["A short report about markets."])) == ""


# ---- strip ----

def test_strip_keeps_the_clean_sentences_and_drops_the_flagged_one():
    reply = "The market was quiet. The Zorblax Corporation rose. Trading closed early."
    sources = ["The market was quiet. Trading closed early."]
    assert grounding.strip(reply, grounding.check(reply, sources)) == "The market was quiet. Trading closed early."


def test_strip_drops_a_heading_left_with_nothing_under_it():
    reply = ("The market was quiet. Trading was thin. Prices held. Volumes stayed low.\n\n"
             "## Zorblax Corporation\n\nThe Zorblax Corporation rose.")
    assert grounding.strip(reply, grounding.check(reply, QUIET)) == (
        "The market was quiet. Trading was thin. Prices held. Volumes stayed low.")


def test_strip_leaves_a_clean_reply_alone():
    reply = "The market was quiet."
    assert grounding.strip(reply, grounding.check(reply, QUIET)) == reply


# ---- code ----

def test_names_inside_a_code_fence_are_never_flagged():
    reply = "Here is code:\n\n```python\nfrom Zorblax import Quentaris\n```\n\nThe market was quiet."
    assert missing(reply, QUIET) == []


# ---- no sources ----

def test_check_without_sources_does_not_raise():
    grounding.check("The Zorblax Corporation rose 44,321 percent.", [])
    grounding.check("The Zorblax Corporation rose.", [""])


# ---- digest ----

ITEMS = [
    {"title": "Bank posts profit", "url": "https://example.com/a", "passage": "Commonwealth Bank posted a profit."},
    {"title": "Market quiet", "url": "https://example.com/b", "snippet": "The market was quiet."},
]


def test_digest_gives_one_line_per_item_with_its_link():
    text = grounding.digest(ITEMS)
    assert text.startswith(grounding.DIGEST_HEAD)
    assert "https://example.com/a" in text and "https://example.com/b" in text
    assert "Commonwealth Bank posted a profit." in text
    assert "The market was quiet." in text


def test_digest_stops_at_the_line_limit():
    many = [{"title": f"Story {n}", "url": f"https://example.com/{n}", "snippet": "text"} for n in range(12)]
    text = grounding.digest(many)
    assert text.count("\n- ") == grounding.DIGEST_ITEMS
    assert f"https://example.com/{grounding.DIGEST_ITEMS}" not in text


def test_digest_of_nothing_is_empty():
    assert grounding.digest([]) == ""


# ---- listed ----

def test_listed_quotes_what_is_missing():
    found = grounding.check("The Zorblax Corporation rose.", ["Markets rose."])
    assert grounding.listed(found) == "“Zorblax Corporation”"


def test_listed_stops_at_the_limit_and_says_there_is_more():
    names = ["Zorblax Corp", "Quentaris Ltd", "Vantablack Inc", "Nimbulus Group", "Oracular Systems",
             "Pellucid Holdings", "Xanthippe Partners", "Yggdrasil Trust", "Zephyrine Capital"]
    reply = " ".join(f"The {name} rose." for name in names)
    found = grounding.check(reply, ["Markets rose."])
    assert len(grounding.missing(found)) == len(names)
    shown = grounding.listed(found)
    assert shown.count("“") == grounding.MAX_LISTED
    assert shown.endswith("and more")


# --- one statement, one result ------------------------------------------------------------

STORIES = [
    "- Brightwell lodges its prospectus with the Harbour Exchange (Ledger Post, 10:02). The lender filed listing papers.",
    "- Tamsin Tools unveils a pocket welder (Workshop Weekly, 09:00). Tamsin Tools says it weighs 900 grams.",
    "- A 13-Year-Old Built a Weather Station at Her School (Coast Herald, 08:00). It reports to the town hall.",
    "- US-China talks resume in Geneva (Daily Wire Service, 15:54). This follows a visit by Okafor to the US.",
]
BULLETIN = ["News bulletin, newest first (it is now Wednesday 11 November 2026, 17:20 local time).\n" + "\n".join(STORIES)]


def _missing(reply, results=STORIES):
    return [finding.missing for finding in grounding.check(reply, BULLETIN, [], results)]


def test_names_of_two_stories_joined_into_one_claim_are_caught():
    assert _missing("The Harbour Exchange approves the Tamsin Tools prospectus.") != []


def test_a_statement_from_one_story_passes():
    assert _missing("Tamsin Tools unveiled a pocket welder that weighs 900 grams.") == []
    assert _missing("Brightwell lodged its prospectus with the Harbour Exchange.") == []


def test_a_sentence_that_joins_two_stories_with_and_passes():
    assert _missing("Tamsin Tools unveiled a pocket welder, and Brightwell lodged a prospectus with the Harbour Exchange.") == []


def test_a_figure_the_story_gives_differently_is_caught_even_when_the_date_has_it():
    assert _missing("An 11-year-old built a weather station at her school.") == [("11 year",)]
    assert _missing("A 13-year-old built a weather station at her school.") == []


def test_a_country_spelled_out_is_backed_by_its_initials_and_a_title_by_the_name():
    assert _missing("The United States and China resumed talks in Geneva after President Okafor's visit.") == []


def test_without_separate_results_only_the_whole_is_checked():
    assert _missing("The Harbour Exchange approves the Tamsin Tools prospectus.", results=[]) == []


def test_a_file_name_with_underscores_is_read_as_written():
    listing = ["file\tASP_ST_Report_21x28cm_v2.pdf\t14.7 MiB"]
    assert grounding.check("The largest is `ASP_ST_Report_21x28cm_v2.pdf` (14.7 MiB).", listing) == []


def test_strip_takes_the_sentence_that_points_back_at_a_removed_one():
    reply = ("The market was quiet. Trading was thin. Prices held. Volumes stayed low. Buyers waited. Sellers too. "
             "The Zorblax Corporation rose. This follows a strong week.")
    left = grounding.strip(reply, grounding.check(reply, QUIET))
    assert "Zorblax" not in left and "This follows" not in left and left.startswith("The market was quiet.")


def test_results_and_headlines_are_read_from_a_bulletin():
    assert len(grounding.results_in(BULLETIN)) == 4
    assert grounding.headlines(BULLETIN)[:2] == [
        "Brightwell lodges its prospectus with the Harbour Exchange", "Tamsin Tools unveils a pocket welder"]


def test_initials_are_backed_by_a_name_spelled_out_inside_a_headline():
    sources = ["- A Turning Point for the United Nations and the Future of Fishing (Harbour Press, 17:29).\n"
               "- U.S., Norway agree on cod quotas, minister says (Coast Wire, 00:11)."]
    assert grounding.check("The UN faces a turning point on fishing, Harbour Press reports.", sources) == []
    assert grounding.check("The US and Norway agreed on cod quotas.", sources) == []
    assert grounding.missing(grounding.check("The talks moved to NATO this week.", sources)) == ["NATO"]


# ---- citation markers that point at nothing ----

def test_a_marker_with_no_source_behind_it_is_taken_out():
    text = "Version 3.12 is the newest [1] and 3.13 is in testing [2]."
    assert grounding.without_unsourced_markers(text, 0) == "Version 3.12 is the newest and 3.13 is in testing."
    assert grounding.without_unsourced_markers(text, 1) == "Version 3.12 is the newest [1] and 3.13 is in testing."
    assert grounding.without_unsourced_markers(text, 2) == text


def test_a_run_of_markers_keeps_only_the_ones_with_a_source():
    assert grounding.without_unsourced_markers("It rains in May [1][3].", 2) == "It rains in May [1]."


def test_reference_lines_for_missing_sources_go_with_their_lead_in():
    text = "The pass opens in June [1].\n\nSources:\n[1] https://example.org/pass\n[2] https://example.org/road"
    assert grounding.without_unsourced_markers(text, 0) == "The pass opens in June."
    assert grounding.without_unsourced_markers(text, 1) == (
        "The pass opens in June [1].\n\nSources:\n[1] https://example.org/pass"
    )


def test_code_links_and_indexes_are_not_markers():
    text = "Use `items[1]` or items[2].\n\n```python\nvalues = [1, 2][1]\nprint(values[1])\n```\n\nSee [1](https://example.org)."
    assert grounding.without_unsourced_markers(text, 0) == text


def test_a_reply_is_never_blanked():
    assert grounding.without_unsourced_markers("[1]", 0) == "[1]"
    assert grounding.without_unsourced_markers("", 0) == ""
