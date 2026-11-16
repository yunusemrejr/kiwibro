"""Tests for the rules in app.py; no terminal is needed."""

from pathlib import Path

import pytest

from app import (ItemError, clamp, detail_lines, filter_items, fit, header, layout,
                 load_items, scroll_for, sort_items, width_of)

ITEMS = [{"title": t, "status": s, "tags": g, "date": d, "body": b} for t, s, g, d, b in (
    ("Renew passport", "todo", ["travel"], "2026-12-01", "Photo booth."),
    ("Fix tap", "blocked", ["home"], "2026-11-12", "Washer ordered."),
    ("Return books", "done", ["errands"], "2026-11-14", "Receipt kept."),
    ("Write report", "doing", ["writing"], "2026-11-20", "One page."))]


def saved(tmp_path, text):
    path = tmp_path / "items.json"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_sample_file_loads_as_six_records():
    found = load_items(Path(__file__).with_name("items.json"))
    assert len(found) == 6 and all(i["title"] and i["status"] for i in found)


def test_missing_fields_get_safe_defaults(tmp_path):
    assert load_items(saved(tmp_path, '[{"title": "Bare"}]')) == [
        {"title": "Bare", "status": "todo", "tags": [], "date": "", "body": ""}]


@pytest.mark.parametrize("text, error", [
    ("", "could not be read"), ('[{"status": "todo"}]', "record 1"), ('{"nope": 1}', "list of records")])
def test_a_broken_file_names_the_problem(tmp_path, text, error):
    with pytest.raises(ItemError, match=error):
        load_items(saved(tmp_path, text))


def test_a_missing_file_names_itself(tmp_path):
    with pytest.raises(ItemError, match="items.json is missing"):
        load_items(tmp_path / "items.json")


def test_wide_characters_count_twice_and_fit_is_exact():
    assert (width_of("日本"), width_of("é")) == (4, 1)
    assert fit("abc", 5) == "abc  " and fit("abcdef", 4) == "abc…" and fit("x", 0) == ""
    assert width_of(fit("日本語のテキスト", 5)) == 5


def test_filter_searches_every_field_in_any_case():
    titles = lambda query: [i["title"] for i in filter_items(ITEMS, query)]
    assert titles("TAP") == ["Fix tap"] and titles("travel") == ["Renew passport"]
    assert titles("receipt") == ["Return books"] and titles("home washer") == ["Fix tap"]
    assert titles("home receipt") == [] and filter_items(ITEMS, " ") == ITEMS


def test_the_three_sorts():
    assert [i["date"] for i in sort_items(ITEMS, "date")][0] == "2026-12-01"
    assert [i["title"] for i in sort_items(ITEMS, "title")][0] == "Fix tap"
    assert [i["status"] for i in sort_items(ITEMS, "status")] == ["doing", "todo", "blocked", "done"]
    odd = {"title": "Someday", "status": "someday", "tags": [], "date": "", "body": ""}
    assert sort_items(ITEMS + [odd], "status")[-1] is odd


def test_selection_and_the_scroll_window_stay_inside_the_list():
    assert (clamp(-2, 4), clamp(9, 4), clamp(2, 0)) == (0, 3, 0)
    assert (scroll_for(0, 0, 20, 5), scroll_for(3, 0, 20, 5), scroll_for(5, 0, 20, 5)) == (0, 0, 1)
    assert (scroll_for(2, 8, 20, 5), scroll_for(19, 0, 20, 5), scroll_for(0, 15, 20, 5)) == (2, 15, 0)


def test_a_small_window_asks_for_more_room():
    assert "40x12" in layout(30, 8)[3]


def test_wide_panes_sit_side_by_side_and_narrow_ones_stack():
    assert layout(100, 30) == ((1, 0, 28, 40), (1, 42, 28, 58), True, "")
    listing, detail, side_by_side, _ = layout(50, 20)
    assert (listing, side_by_side) == ((1, 0, 10, 50), False)
    assert detail[0] > listing[0] + listing[2] - 1


def test_the_header_counts_statuses_and_names_the_filter_and_sort():
    assert header(ITEMS, 4, "", "date") == (
        "{{title}}   4 {{items}}   1 doing · 1 todo · 1 blocked · 1 done", "sort: date")
    left, right = header(ITEMS[:1], 4, " passport ", "title")
    assert "1 {{items}} of 4" in left and right == "filter: passport   sort: title"


def test_the_detail_lists_heading_status_meta_and_the_wrapped_body():
    item = ITEMS[0]
    rows = detail_lines(item, 20)
    assert rows[:3] == [("Renew passport", "heading"), ("todo", "status"), ("2026-12-01   travel", "muted")]
    assert rows[3] == ("", "text")
    assert "".join(text for text, _ in rows[4:]).replace(" ", "") == item["body"].replace(" ", "")
