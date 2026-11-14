"""What fills the disk is measured on the computer and answered from the figures, never searched on the web."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_backend import Env, chat_events

from kiwibro import desktop, lookfirst, readings

TOOLS = frozenset({"disk_usage", "system_info", "list_files", "git"})
ROOT = Path("/nonexistent-workspace")

TODAY = """~: 3.3 GiB in 2 entries changed since 14 Nov 00:00
Disk: 17.9 GiB free of 18.6 GiB
   2.9 GiB   88%  Videos/
 400.0 MiB   12%  Downloads/

Largest files written since 14 Nov 00:00:
   2.9 GiB  09:12  Videos/trip.mp4
 400.0 MiB  10:40  Downloads/setup.iso"""
EVERYTHING = """~: 120.0 GiB in 25 entries
Disk: 17.9 GiB free of 18.6 GiB
  45.0 GiB   38%  Downloads/
  30.0 GiB   25%  Videos/
(23 more not listed; ask for a higher top)
2 folders could not be read"""


def looks(message: str) -> list[tuple[str, dict]]:
    return [(look.name, look.args) for look in lookfirst.calls(message, ROOT, TOOLS)]


@pytest.mark.parametrize(("message", "args"), [
    ("what's eating my disk", {}),
    ("why is my disk full", {}),
    ("my disk is full, can you help", {}),
    ("which folders are the biggest", {}),
    ("what are the biggest folders on my computer", {}),
    ("where did all my storage go", {}),
    ("who ate all my disk space", {}),
    ("what filled my drive", {}),
    ("what is taking the space today", {"changed": "today"}),
    ("what used up the space this week", {"changed": "week"}),
    ("what has been using my storage lately", {"changed": "week"}),
    ("what filled my disk this month", {"changed": "month"}),
])
def test_a_question_about_what_fills_the_disk_gets_a_disk_look(message: str, args: dict):
    assert looks(message) == [("disk_usage", args)]


@pytest.mark.parametrize("message", [
    "how much free disk space do i have",  # the free figure is system_info's
    "what takes up space in the universe",
    "what is taking up space in this repo",
    "what's the biggest star",
    "how much ram is my laptop using",
])
def test_other_questions_get_no_disk_look(message: str):
    assert ("disk_usage", {}) not in looks(message) and all(name != "disk_usage" for name, _ in looks(message))


def test_a_file_question_about_a_folder_stays_with_the_folder_listing():
    assert [name for name, _ in looks("which file takes the most space on my desktop")] == ["list_files"]


def test_the_answer_is_written_from_the_look_largest_first():
    said = lookfirst.answer("what's eating my disk", {"disk_usage": {"output": EVERYTHING}})
    assert said.splitlines() == [
        "Your home folder holds 120.0 GiB in 25 entries, largest first:",
        "- Downloads/: 45.0 GiB (38%)", "- Videos/: 30.0 GiB (25%)",
        "23 more entries are smaller and not listed.", "",
        "17.9 GiB is free of 18.6 GiB on this disk.", "2 folders could not be read"]


def test_a_time_window_answer_names_the_files_written_in_it():
    said = lookfirst.answer("what is taking the space today", {"disk_usage": {"output": TODAY}})
    assert said.startswith("Since 14 Nov 00:00, 3.3 GiB was written in your home folder. Largest first:")
    assert "- Videos/trip.mp4: 2.9 GiB, 09:12" in said and "Largest files written since 14 Nov 00:00:" in said


def test_nothing_written_is_said_as_that():
    empty = "~: 0 B in 0 entries changed since 14 Nov 00:00\nDisk: 1.0 GiB free of 2.0 GiB"
    assert lookfirst.answer("what used up the space today", {"disk_usage": {"output": empty}}) == \
        "Nothing in your home folder was written since 14 Nov 00:00."


@pytest.mark.parametrize("message", ["what's eating my disk and can i delete it", "why is my disk full, is it the cache"])
def test_a_question_that_wants_more_than_the_figures_goes_to_the_model(message: str):
    assert lookfirst.answer(message, {"disk_usage": {"output": EVERYTHING}}) == ""


# --- in a turn ----------------------------------------------------------------------------------

@pytest.fixture
def measured(env: Env, monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """disk_usage answers with a fixed result (the real one would walk the home folder); each call is kept."""
    calls: list[tuple] = []

    def fake(path=None, changed=None, top=10):
        calls.append((path, changed, top))
        return TODAY if changed else EVERYTHING

    monkeypatch.setattr(desktop, "disk_usage", fake)
    return calls


def tools_called(env: Env, conversation_id: str) -> list[str]:
    return [call["function"]["name"] for m in env.svc.db.list_messages(conversation_id)
            for call in m.get("tool_calls") or []]


def test_the_follow_up_about_today_measures_the_disk_instead_of_searching_the_web(env: Env, measured, monkeypatch):
    # what the helpers made of "today": a live question, so the web
    monkeypatch.setattr(readings, "wants_search", lambda _reading: True)
    events, _raw = chat_events(env, "how much free disk space do i have")
    conversation_id = events[0][1]["conversation_id"]
    events, _raw = chat_events(env, "so what is taking the space today", conversation_id)
    done = [payload for event, payload in events if event == "done"][0]
    assert measured == [(None, "today", 10)]
    assert tools_called(env, conversation_id) == ["system_info", "disk_usage"]
    assert "searched_first" not in done["inference"] and done["inference"]["answered_from_look"] is True
    assert env.fake.stream_calls == []  # the figures are the answer
    last = [m for m in env.svc.db.list_messages(conversation_id) if m["role"] == "assistant" and m["content"]][-1]
    assert last["content"].startswith("Since 14 Nov 00:00, 3.3 GiB was written in your home folder.")


def test_a_short_follow_up_to_a_look_does_not_start_a_web_search(env: Env, measured, monkeypatch):
    monkeypatch.setattr(readings, "wants_search", lambda _reading: True)
    monkeypatch.setattr(readings, "follows_up", lambda _reading: False)
    events, _raw = chat_events(env, "what's eating my disk")
    conversation_id = events[0][1]["conversation_id"]
    events, _raw = chat_events(env, "and is any of it from today", conversation_id)
    done = [payload for event, payload in events if event == "done"][0]
    assert "searched_first" not in done["inference"]
    assert "web_search" not in tools_called(env, conversation_id)


def test_after_a_chat_answer_the_same_message_still_searches(env: Env, monkeypatch):
    """The rule above is about the computer only: a short question after an ordinary answer is as before."""
    monkeypatch.setattr(readings, "wants_search", lambda _reading: True)
    monkeypatch.setattr(readings, "follows_up", lambda _reading: False)
    events, _raw = chat_events(env, "tell me a fact about kiwi birds")
    conversation_id = events[0][1]["conversation_id"]
    events, _raw = chat_events(env, "and is any of it from today", conversation_id)
    done = [payload for event, payload in events if event == "done"][0]
    assert done["inference"].get("searched_first") is True
