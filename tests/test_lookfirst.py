"""Questions the look at the computer answers outright are answered in its own figures."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_backend import Env, chat_events

from kiwibro import lookfirst
from kiwibro.llm import ChatResult

INFO = {"cpu_model": "Example Core 7", "cpu_cores": "6 cores, 12 threads", "memory_total": "15.0 GB",
        "memory_available": "6.1 GB", "gpu": "Example Graphics", "disk_free": "3 GB", "disk_total": "7 GB",
        "home_disk_free": "210 GB", "home_disk_total": "476 GB", "os": "Example Linux 26.04", "hostname": "box"}
FOLDER = {"files_here": 12, "folders_here": 5, "files": 75, "folders": 9, "largest": "a/video.mp4",
          "largest_size": "1.2 GB", "newest": "notes.txt", "newest_time": "2026-11-11 10:00", "shown": "~/Desktop", "depth": 2}


@pytest.mark.parametrize(("message", "said"), [
    ("what cpu do i have", "The CPU is Example Core 7 (6 cores, 12 threads)."),
    ("how many cores does my cpu have", "The CPU is Example Core 7 (6 cores, 12 threads)."),
    ("how much memory does this laptop have", "This computer has 15.0 GB of RAM; 6.1 GB of it is available right now."),
    ("how much free disk space do i have", "210 GB is free of 476 GB on the disk your home folder is on."),
    ("which gpu is in this pc", "The GPU is Example Graphics."),
    ("what cpu and how much ram does this computer have?",
     "- CPU: Example Core 7 (6 cores, 12 threads)\n- RAM: 15.0 GB, 6.1 GB available right now"),
])
def test_a_question_about_a_part_is_answered_with_that_part(message: str, said: str):
    assert lookfirst.answer(message, {"system_info": INFO}) == said


def test_specs_lists_the_main_parts():
    said = lookfirst.answer("tell me my pc specs please", {"system_info": INFO})
    assert said.splitlines() == ["- CPU: Example Core 7 (6 cores, 12 threads)", "- RAM: 15.0 GB, 6.1 GB available right now",
                                 "- GPU: Example Graphics", "- Disk: 210 GB free of 476 GB"]


@pytest.mark.parametrize("message", [
    "how much ram do i have and is it enough for gaming",  # an opinion is asked too
    "is my cpu good for gaming",
    "how much ram is in use",
    "what kind of battery do i have",  # the look has no battery: the model says so
])
def test_a_question_that_wants_more_than_the_figures_goes_to_the_model(message: str):
    assert lookfirst.answer(message, {"system_info": INFO}) == ""


def test_how_many_files_gives_the_folder_itself_first_then_what_is_inside():
    said = lookfirst.answer("How many files are on my Desktop?", {"list_files": FOLDER})
    assert said == ("~/Desktop has 12 files and 5 folders directly in it (hidden ones not counted). "
                    "Counting what is inside those folders too, 2 levels deep: 75 files and 9 folders.")


def test_one_file_and_one_folder_are_not_plural():
    said = lookfirst.answer("how many files are in my documents", {"list_files": {**FOLDER, "files_here": 1, "folders_here": 1,
                                                                                   "files": 1, "folders": 1}})
    assert said == "~/Desktop has 1 file and 1 folder directly in it (hidden ones not counted)."


def test_the_biggest_and_the_newest_are_named():
    assert lookfirst.answer("whats the biggest file on my desktop", {"list_files": FOLDER}) == \
        "The largest file in ~/Desktop is a/video.mp4 (1.2 GB)."
    assert lookfirst.answer("which one is the biggest", {"list_files": FOLDER}) == \
        "The largest file in ~/Desktop is a/video.mp4 (1.2 GB)."
    assert "notes.txt (2026-11-11 10:00)" in lookfirst.answer("what is the newest file in my downloads folder", {"list_files": FOLDER})


@pytest.mark.parametrize("message", [
    "list the files on my desktop",
    "what is on my desktop",
    "how many pdfs are on my desktop",  # a kind of file: the count of all files is not the answer
    "how many files on my desktop are older than a year",
    "how many files did i delete from my desktop",
])
def test_a_folder_question_with_a_word_of_its_own_goes_to_the_model(message: str):
    assert lookfirst.answer(message, {"list_files": FOLDER}) == ""


def test_a_part_question_is_answered_without_asking_the_model(env: Env):
    env.fake.stream_script = [ChatResult(content="You have 64 GB of RAM.")]  # what a model would make up
    events, _raw = chat_events(env, "how much ram does this computer have")
    done = [payload for event, payload in events if event == "done"][0]
    assert done["inference"]["answered_from_look"] is True
    assert env.fake.stream_calls == []
    conversation_id = events[0][1]["conversation_id"]
    shown = [m["content"] for m in env.svc.db.list_messages(conversation_id) if m["role"] == "assistant" and m["content"]]
    assert len(shown) == 1 and shown[0].startswith("This computer has ") and "64 GB" not in shown[0]


# --- is my local copy the same commit as the remote? ---------------------------------

def _checkout(tmp_path: Path, name: str = "") -> Path:
    folder = tmp_path / name if name else tmp_path
    (folder / ".git").mkdir(parents=True)
    return folder


@pytest.mark.parametrize(("message", "remote"), [
    ("is my local copy the same commit as github.com/acme/app?", "https://github.com/acme/app"),
    ("in my github repo at https://github.com/acme/app i updated the readme and on my local dev copy i have "
     "some changes but are they same commit?", "https://github.com/acme/app"),
    ("is this repo up to date with origin?", "origin"),
    ("did i push everything in my repo", "origin"),
    ("check if my local branch is behind github", "origin"),
])
def test_a_same_commit_question_is_put_to_git(tmp_path: Path, message: str, remote: str):
    _checkout(tmp_path)
    look = lookfirst.repo(message, tmp_path)
    assert look is not None and look.name == "git"
    assert look.args == {"action": "compare", "remote": remote, "path": "."}
    assert lookfirst.calls(message, tmp_path, frozenset({"git", "system_info", "list_files"})) == [look]


def test_the_checkout_a_message_names_is_the_one_compared(tmp_path: Path):
    _checkout(tmp_path, "site")
    look = lookfirst.repo("is my repo in ./site in sync with github.com/acme/site?", tmp_path)
    assert look is not None and look.args["path"] == "site" and look.args["remote"] == "https://github.com/acme/site"


@pytest.mark.parametrize("message", [
    "what is the difference between git merge and git rebase?",
    "in this repo, rename every function with the same prefix",
    "is my git knowledge up to date?",  # asked where there is no checkout: see below
    "write a commit message for my last change",
    "how do i push my branch to github",
])
def test_other_git_talk_starts_no_compare(tmp_path: Path, message: str):
    if "knowledge" not in message:
        _checkout(tmp_path)
    assert lookfirst.repo(message, tmp_path) is None
    assert lookfirst.calls(message, tmp_path, frozenset({"git"})) == []


def test_without_the_git_tool_nothing_is_compared(tmp_path: Path):
    _checkout(tmp_path)
    assert lookfirst.calls("is this repo up to date with origin?", tmp_path, frozenset({"system_info"})) == []


SAME = {"same": True, "output": "Same commit: yes. The local copy and origin (main) are both at 1a2b3c4d5.\n"
        "Local: 1a2b3c4 tidy the readme (2 hours ago)\nUncommitted changes in the local copy: 2 files\nRemote branches: main"}
AHEAD = {"same": False, "output": "Same commit: no. The local copy is at 1a2b3c4d5 (main), origin (main) is at 9f8e7d6c5.\n"
         "The local copy is 2 commits ahead.\nLocal: 1a2b3c4 tidy the readme (2 hours ago)\n"
         "Uncommitted changes in the local copy: none\nRemote branches: main, dev"}


def test_gits_verdict_is_the_answer():
    said = lookfirst.answer("is this repo up to date with origin?", {"git": SAME})
    assert said.startswith("Yes, the same commit. The local copy and origin (main) are both at 1a2b3c4d5.")
    assert "- Uncommitted changes in the local copy: 2 files" in said and "Remote branches" not in said
    said = lookfirst.answer("are they the same commit?", {"git": AHEAD})
    assert said.startswith("No, not the same commit. The local copy is at 1a2b3c4d5 (main), origin (main) is at 9f8e7d6c5. "
                           "The local copy is 2 commits ahead.")
    assert "- Last commit in the local copy: 1a2b3c4 tidy the readme (2 hours ago)" in said


def test_a_compare_that_failed_is_left_to_the_model():
    failed = {"same": None, "output": "Could not compare: the remote origin could not be reached, or it does not exist or is private."}
    assert lookfirst.answer("is this repo up to date with origin?", {"git": failed}) == ""
