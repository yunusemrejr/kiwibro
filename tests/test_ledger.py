"""Ledger tests: the exact record a long conversation keeps of its own steps.

The ledger is written by code from tool names and arguments, so these tests use made-up
calls only: no app, no model, and a database in a temporary folder.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kiwibro import ledger, tools
from kiwibro.db import Database
from kiwibro.memory import MemoryEngine, asked_steps, normalize


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "data" / "kiwibro.db")
    database.initialize()
    return database


@pytest.fixture
def chat(db: Database) -> str:
    return db.create_conversation()["id"]


@pytest.fixture
def memory(db: Database) -> MemoryEngine:
    return MemoryEngine(db, None, lambda: {})  # type: ignore[arg-type]


def _rows(db: Database, chat: str) -> list[tuple[int, str, str, str]]:
    return [(row["seq"], row["kind"], row["key"], row["detail"]) for row in db.list_ledger(chat)]


# --- what the user asked ----------------------------------------------------


def test_record_user_keeps_the_first_words_of_the_ask(db: Database, chat: str) -> None:
    ledger.record_user(db, chat, 1, "  Build a to-do API.\nWith due dates and tags.  ")
    ledger.record_user(db, chat, 2, "   \n  ")
    ledger.record_user(db, chat, 3, "x" * 400)
    rows = db.list_ledger(chat)
    assert [row["seq"] for row in rows] == [1, 3]
    assert rows[0]["kind"] == "ask"
    assert rows[0]["key"] == "Build a to-do API. With due dates and tags."
    assert len(rows[1]["key"]) == ledger.ASK_CHARS


# --- what the tools did -----------------------------------------------------


def test_file_tools_record_the_path_and_the_change(db: Database, chat: str) -> None:
    ledger.record_tool(db, chat, 4, "write_file", {"path": "app.py", "content": "x"}, True, "Wrote app.py")
    ledger.record_tool(db, chat, 5, "replace_in_file", {"path": "app.py", "old": "a", "new": "b"}, True, "Edited")
    ledger.record_tool(db, chat, 6, "apply_patch", {"path": "app.py", "edits": [{"old": "a", "new": "b"}]}, True, "ok")
    ledger.record_tool(db, chat, 7, "file_manage", {"action": "mkdir", "path": "src"}, True, "Created folder src.")
    ledger.record_tool(
        db, chat, 8, "file_manage", {"action": "move", "path": "a.py", "destination": "src/a.py"}, True, "Moved"
    )
    ledger.record_tool(
        db, chat, 9, "file_manage", {"action": "copy", "path": "a.py", "destination": "b.py"}, True, "Copied"
    )
    ledger.record_tool(db, chat, 10, "convert_document", {"path": "report.md", "to": "pdf"}, True, "Converted")
    ledger.record_tool(
        db, chat, 11, "convert_document", {"path": "notes.md", "to": "html", "output": "out/notes.html"}, True, "ok"
    )
    ledger.record_tool(db, chat, 12, "undo_changes", {"path": "app.py"}, True, "Restored app.py")
    ledger.record_tool(db, chat, 13, "component", {"action": "use", "path": "library/chart"}, True, "Used")
    assert _rows(db, chat) == [
        (4, "file", "app.py", "written"),
        (5, "file", "app.py", "patched"),
        (6, "file", "app.py", "patched"),
        (7, "file", "src", "folder made"),
        (8, "file", "a.py", "moved to src/a.py"),
        (9, "file", "a.py", "copied to b.py"),
        (10, "file", "report.md", "converted to pdf"),
        (11, "file", "notes.md", "converted to out/notes.html"),
        (12, "file", "app.py", "undone"),
        (13, "file", "library/chart", "used"),
    ]


def test_command_tools_record_the_command_and_its_exit_code(db: Database, chat: str) -> None:
    listed = "$ ls -la demo\nexit code: 0\ncwd: .\n\ntotal 0"
    ledger.record_tool(db, chat, 3, "run_command", {"command": "ls -la demo"}, True, listed)
    failed = "$ python3 -m pytest -q\nexit code: 1\ncwd: .\n\n1 failed"
    ledger.record_tool(db, chat, 14, "run_command", {"command": "python3 -m pytest -q"}, False, failed)
    ledger.record_tool(
        db, chat, 15, "terminal_session", {"action": "start", "command": "python3 app.py"}, True,
        "Started session s1 (running, cwd .):\n$ python3 app.py\n(no output)",
    )
    ledger.record_tool(
        db, chat, 16, "terminal_session", {"action": "run", "command": "curl localhost", "session_id": "s1"}, True,
        "[s1] $ curl localhost\nexit code: 0 · cwd: .\n\nok",
    )
    ledger.record_tool(
        db, chat, 17, "run_python", {"code": "print(1)"}, True,
        "python snippet · exit code: 0 · 12ms\nscratch: /tmp/s\n\n1",
    )
    ledger.record_tool(
        db, chat, 18, "run_javascript", {"code": "throw 1"}, False,
        "javascript snippet · exit code: 2 · 4ms\nscratch: /tmp/s\n\n",
    )
    ledger.record_tool(
        db, chat, 19, "remote", {"action": "run", "host": "box", "command": "systemctl status demo"}, True,
        "$ box: systemctl status demo\n(exit 0)\nactive",
    )
    ledger.record_tool(db, chat, 20, "remote", {"action": "hosts"}, True, "hosts: box")
    ledger.record_tool(db, chat, 21, "terminal_session", {"action": "list"}, True, "s1 · running")
    assert _rows(db, chat) == [
        (3, "command", "ls -la demo", "exit 0"),
        (14, "command", "python3 -m pytest -q", "exit 1"),
        (15, "command", "python3 app.py", ""),
        (16, "command", "curl localhost", "exit 0"),
        (17, "command", "python snippet", "exit 0"),
        (18, "command", "javascript snippet", "exit 2"),
        (19, "command", "systemctl status demo", "exit 0"),
    ]


def test_multi_command_records_one_row_per_command_with_its_own_exit(db: Database, chat: str) -> None:
    result = (
        "$ pip install flask\nexit code: 0\nInstalled\n\n"
        "$ python3 -m pytest -q\nexit code: 1\n1 failed\n\n"
        "(stopped after the failing command)"
    )
    ledger.record_tool(
        db, chat, 20, "multi_command",
        {"commands": ["pip install flask", "python3 -m pytest -q", "python3 -m pytest -q --lf"]}, False, result,
    )
    assert _rows(db, chat) == [
        (20, "command", "pip install flask", "exit 0"),
        (20, "command", "python3 -m pytest -q", "exit 1"),
        (20, "command", "python3 -m pytest -q --lf", "failed"),  # never ran: still not passed
    ]


def test_page_and_search_tools_record_what_was_read(db: Database, chat: str) -> None:
    ledger.record_tool(db, chat, 22, "fetch_page", {"url": "https://example.org/docs"}, True, "text")
    ledger.record_tool(db, chat, 23, "fetch_pages", {"urls": ["https://a.test", "https://b.test"]}, True, "text")
    ledger.record_tool(db, chat, 24, "browse_page", {"url": "https://c.test"}, True, "text")
    ledger.record_tool(db, chat, 25, "browser", {"action": "open", "url": "https://d.test"}, True, "opened")
    ledger.record_tool(db, chat, 26, "browser", {"action": "back"}, True, "https://c.test")
    ledger.record_tool(db, chat, 27, "web_search", {"query": "repair a bike tyre"}, True, "results")
    ledger.record_tool(db, chat, 28, "multi_search", {"queries": ["harbour tides", "moon phase"]}, True, "results")
    ledger.record_tool(db, chat, 29, "web_research", {"query": "tide tables"}, True, "report")
    assert _rows(db, chat) == [
        (22, "page", "https://example.org/docs", "read"),
        (23, "page", "https://a.test", "read"),
        (23, "page", "https://b.test", "read"),
        (24, "page", "https://c.test", "read"),
        (25, "page", "https://d.test", "read"),
        (27, "page", '"repair a bike tyre"', "searched"),
        (28, "page", '"harbour tides"', "searched"),
        (28, "page", '"moon phase"', "searched"),
        (29, "page", '"tide tables"', "searched"),
    ]


def test_note_and_task_tools_record_their_titles(db: Database, chat: str) -> None:
    ledger.record_tool(db, chat, 5, "note", {"action": "save", "title": "Shopping", "content": "eggs"}, True, "ok")
    ledger.record_tool(db, chat, 6, "note", {"action": "append", "title": "Shopping", "content": "tea"}, True, "ok")
    ledger.record_tool(db, chat, 7, "note", {"action": "read", "title": "Shopping"}, True, "eggs")
    ledger.record_tool(db, chat, 8, "note", {"action": "search", "content": "eggs"}, True, "Shopping")
    ledger.record_tool(db, chat, 9, "remember", {"action": "save", "content": "User keeps bees in Lisbon"}, True, "ok")
    ledger.record_tool(db, chat, 10, "task_board", {"action": "add", "title": "Ship the demo"}, True, "Added")
    ledger.record_tool(db, chat, 11, "task_board", {"action": "update", "task_id": "t_9"}, True, "Done")
    ledger.record_tool(db, chat, 12, "task_board", {"action": "list"}, True, "Tasks")
    assert _rows(db, chat) == [
        (5, "note", "Shopping", ""),
        (6, "note", "Shopping", ""),
        (9, "note", "User keeps bees in Lisbon", ""),
        (10, "task", "Ship the demo", ""),
        (11, "task", "t_9", ""),
    ]


def test_failed_calls_are_kept_except_failed_writes(db: Database, chat: str) -> None:
    ledger.record_tool(db, chat, 12, "write_file", {"path": "a.py", "content": "x"}, False, "Error: denied")
    assert db.list_ledger(chat) == []
    ledger.record_tool(db, chat, 13, "fetch_page", {"url": "https://a.test"}, False, "Error: timeout")
    ledger.record_tool(db, chat, 14, "run_command", {"command": "make"}, False, "Timed out after 60s.")
    ledger.record_tool(db, chat, 15, "note", {"action": "save", "title": "Trip"}, False, "Error: private chat")
    assert _rows(db, chat) == [
        (13, "page", "https://a.test", "failed"),
        (14, "command", "make", "failed"),
        (15, "note", "Trip", "failed"),
    ]


@pytest.mark.parametrize(("name", "arguments"), [
    ("write_file", {}),
    ("write_file", None),
    ("write_file", {"path": None, "content": None}),
    ("file_manage", {"action": "move", "path": "a.py"}),
    ("run_command", {"command": None}),
    ("run_command", {"command": ["ls", "-la"]}),
    ("multi_command", {"commands": "not-a-list"}),
    ("multi_command", {"commands": [None, 3]}),
    ("fetch_pages", {"urls": [None, 3]}),
    ("terminal_session", {"action": "start"}),
    ("note", {"action": "save"}),
    ("task_board", {"action": "add"}),
    ("browser", {"action": "open"}),
    ("component", {"action": "use"}),
    ("unknown_tool", {"anything": 1}),
])
def test_odd_arguments_never_raise(db: Database, chat: str, name: str, arguments: object) -> None:
    rows = ledger.record_tool(db, chat, 1, name, arguments, True, None)
    assert isinstance(rows, list)
    assert all(row["key"] for row in rows)  # an empty key is never stored


def test_a_result_that_is_not_text_is_tolerated(db: Database, chat: str) -> None:
    ledger.record_tool(db, chat, 1, "run_command", {"command": "ls"}, True, {"output": "a.py"})
    ledger.record_tool(db, chat, 2, "write_file", {"path": "a.py"}, True, {"written": True})
    assert [row["key"] for row in db.list_ledger(chat)] == ["ls", "a.py"]


# --- the note the model sees ------------------------------------------------


def _fill(db: Database, chat: str, rows: list[tuple[int, str, str, str]]) -> None:
    for seq, kind, key, detail in rows:
        db.add_ledger_entry(chat, seq, kind, key, detail)


def test_block_is_empty_until_a_step_has_scrolled_out(db: Database, chat: str) -> None:
    db.add_ledger_entry(chat, 3, "ask", "build a to-do API")
    assert ledger.block(db, chat, 0) == ""


def test_block_opens_with_the_two_fixed_lines(db: Database, chat: str) -> None:
    db.add_ledger_entry(chat, 1, "ask", "build a to-do API")
    text = ledger.block(db, chat, 5)
    assert text.splitlines()[0] == "## Earlier in this session"
    assert text.splitlines()[1] == (
        "Every earlier step of this conversation is stored in full. To read one again: "
        'recall("words to find"), or recall("step N") with a step number.'
    )


def test_block_renders_one_line_per_kind(db: Database, chat: str) -> None:
    _fill(db, chat, [
        (1, "ask", "build a to-do API", ""),
        (3, "command", "pip install flask", "exit 0"),
        (5, "note", "Shopping", ""),
        (9, "ask", "add due dates", ""),
        (12, "file", "app.py", "patched"),
        (14, "command", "python3 -m pytest -q", "exit 1"),
        (15, "file", "tests/test_api.py", "written"),
        (18, "file", "app.py", "patched"),
        (22, "page", "https://example.org/docs", "read"),
        (31, "ask", "fix the failing test", ""),
        (40, "file", "app.py", "patched"),
        (41, "command", "python3 -m pytest -q", "exit 0"),
    ])
    lines = ledger.block(db, chat, 41).splitlines()
    assert lines[2] == "Asked: (step 1) build a to-do API; (step 9) add due dates; (step 31) fix the failing test"
    assert lines[3] == (
        "Files changed: app.py (steps 12, 18, 40: patched), tests/test_api.py (step 15: written)"
    )
    assert lines[4] == (
        "Commands: pip install flask (step 3: exit 0); "
        "python3 -m pytest -q (step 14: exit 1; step 41: exit 0)"
    )
    assert lines[5] == "Pages: https://example.org/docs (step 22: read)"
    assert lines[6] == 'Notes and tasks: "Shopping" (step 5)'
    assert len(lines) == 7


def test_block_shows_the_last_four_steps_of_a_repeated_key(db: Database, chat: str) -> None:
    for seq in (2, 5, 8, 11, 14, 17):
        db.add_ledger_entry(chat, seq, "file", "app.py", "patched")
    db.add_ledger_entry(chat, 18, "command", "make", "exit 0")
    db.add_ledger_entry(chat, 19, "command", "make test", "exit 1")
    db.add_ledger_entry(chat, 20, "command", "make lint", "exit 0")
    db.add_ledger_entry(chat, 21, "command", "make lint", "exit 0")
    assert ledger.block(db, chat, 21).splitlines()[2:] == [
        "Files changed: app.py (steps 8, 11, 14, 17 and 2 more: patched)",
        "Commands: make (step 18: exit 0); make test (step 19: exit 1); make lint (steps 20, 21: exit 0)",
    ]


def test_block_counts_only_the_steps_that_scrolled_out(db: Database, chat: str) -> None:
    db.add_ledger_entry(chat, 4, "file", "a.py", "written")
    db.add_ledger_entry(chat, 9, "file", "b.py", "written")
    text = ledger.block(db, chat, 5)
    assert "a.py" in text and "b.py" not in text


def test_block_drops_the_oldest_entries_of_the_longest_line(db: Database, chat: str) -> None:
    for seq in range(1, 13):
        db.add_ledger_entry(chat, seq, "ask", f"question number {seq} about the demo", "")
    db.add_ledger_entry(chat, 20, "file", "app.py", "written")
    text = ledger.block(db, chat, 20, budget_chars=330)
    lines = text.splitlines()
    assert lines[0] == "## Earlier in this session"
    assert len(text) <= 330
    asked = next(line for line in lines if line.startswith("Asked:"))
    assert "(step 12)" in asked and "(step 1)" not in asked
    assert re.search(r"; and \d+ earlier$", asked)
    assert "Files changed: app.py (step 20: written)" in lines  # the short line was left alone


def test_block_skips_rows_without_a_key(db: Database, chat: str) -> None:
    db.add_ledger_entry(chat, 1, "file", "", "written")
    db.add_ledger_entry(chat, 2, "ask", "build a to-do API")
    assert ledger.block(db, chat, 5).splitlines()[2:] == ["Asked: (step 2) build a to-do API"]


# --- reading one step again -------------------------------------------------


def test_step_text_reads_a_message_back_as_one_recall_line(db: Database, chat: str) -> None:
    user = db.add_message(chat, "user", "Build a to-do API.\nWith tags.")["seq"]
    answer = db.add_message(chat, "assistant", "Here is a plan.")["seq"]
    tool = db.add_message(chat, "tool", "exit code: 0\nok", tool_name="run_command")["seq"]
    assert ledger.step_text(db, chat, user) == f"- [this chat, step {user}, user] Build a to-do API. With tags."
    assert ledger.step_text(db, chat, answer) == f"- [this chat, step {answer}, Kiwibro] Here is a plan."
    assert ledger.step_text(db, chat, tool) == f"- [this chat, step {tool}, tool result] exit code: 0 ok"
    assert ledger.step_text(db, chat, 99) is None


def test_step_text_cuts_a_long_message_and_says_how_much_is_left(db: Database, chat: str) -> None:
    seq = db.add_message(chat, "user", "a" * (ledger.STEP_READ_CHARS + 500))["seq"]
    line = ledger.step_text(db, chat, seq) or ""
    assert line.startswith(f"- [this chat, step {seq}, user] aaa")
    assert re.search(r"\[cut: \d+ more characters\]$", line)
    assert len(line) > ledger.STEP_READ_CHARS


# --- recall over steps and words --------------------------------------------


def test_only_a_bare_step_reference_means_a_step() -> None:
    assert asked_steps("step 14") == [14]
    assert asked_steps("STEP14") == [14]
    assert asked_steps("  #14 ") == [14]
    assert asked_steps("steps 14-16") == [14, 15, 16]
    assert asked_steps("steps 16 to 14") == [14, 15, 16]
    assert asked_steps("steps 1-40") == [1, 2, 3, 4, 5]
    assert asked_steps("step by step instructions") == []
    assert asked_steps("what was step 4 about?") == []


def test_recall_reads_a_step_back_in_full(db: Database, chat: str, memory: MemoryEngine) -> None:
    seq = db.add_message(chat, "user", "The launch code is kiwi-42")["seq"]
    items = memory.recall(f"step {seq}", conversation_id=chat)
    assert len(items) == 1
    assert items[0]["source"] == f"this chat, step {seq}, user"
    assert items[0]["line"] == f"- [this chat, step {seq}, user] The launch code is kiwi-42"
    assert items[0]["content"] == "The launch code is kiwi-42"


def test_recall_reads_a_range_hash_and_tight_form(db: Database, chat: str, memory: MemoryEngine) -> None:
    for n in range(1, 6):
        db.add_message(chat, "user", f"detail {n}")
    assert [item["seq"] for item in memory.recall("steps 1-2", conversation_id=chat)] == [1, 2]
    assert [item["seq"] for item in memory.recall("#3", conversation_id=chat)] == [3]
    assert [item["seq"] for item in memory.recall("step2", conversation_id=chat)] == [2]
    assert [item["seq"] for item in memory.recall("steps 1-9", conversation_id=chat)] == [1, 2, 3, 4, 5]


def test_recall_says_when_a_step_does_not_exist(db: Database, chat: str, memory: MemoryEngine) -> None:
    db.add_message(chat, "user", "hello")
    items = memory.recall("step 14", conversation_id=chat)
    assert len(items) == 1
    assert items[0]["line"] == "No step 14 in this conversation (it has 1 step)."
    db.add_message(chat, "assistant", "hi")
    db.add_message(chat, "user", "how are you")
    assert memory.recall("step 14", conversation_id=chat)[0]["content"] == (
        "No step 14 in this conversation (it has 3 steps)."
    )


def test_recall_labels_this_chats_excerpts_with_their_step(db: Database, memory: MemoryEngine) -> None:
    chat = db.create_conversation()["id"]
    early = db.add_message(chat, "user", "Remember the harbor gate code is 4815")["seq"]
    db.add_message(chat, "assistant", "Noted: the harbor gate code is 4815")
    for n in range(60):
        db.add_message(chat, "user", f"Filler sailing topic {n} " + "chat " * 30)
        db.add_message(chat, "assistant", f"Filler reply {n} " + "talk " * 30)
    db.update_conversation(chat, summary="Harbor trip planned.", summary_through_seq=2, touch=False)
    items = memory.recall("harbor gate code", conversation_id=chat, limit=4)
    excerpts = [item for item in items if item["source"] != "memory"]
    assert excerpts
    asked = next(item for item in excerpts if item["seq"] == early)
    assert asked["source"] == f"this chat, step {early}, user"
    assert asked["line"].startswith(f"- [this chat, step {early}, user] ")
    assert "4815" in asked["content"]
    assert all(item["source"].startswith("this chat, step ") for item in excerpts)


def test_recall_labels_a_tool_result_step(db: Database, memory: MemoryEngine) -> None:
    chat = db.create_conversation()["id"]
    db.add_message(chat, "user", "Check the demo logs")
    seq = db.add_message(chat, "tool", "ERROR: the disk is full", tool_name="run_command")["seq"]
    for n in range(40):
        db.add_message(chat, "user", f"Filler sailing topic {n} " + "chat " * 30)
    items = memory.recall("disk full", conversation_id=chat)
    tool_items = [item for item in items if item.get("role") == "tool"]
    assert tool_items
    assert tool_items[0]["source"] == f"this chat, step {seq}, tool result"
    assert tool_items[0]["line"].startswith(f"- [this chat, step {seq}, tool result] ")


def test_long_excerpts_are_capped_at_the_recall_limit(db: Database, memory: MemoryEngine) -> None:
    chat = db.create_conversation()["id"]
    db.add_message(chat, "user", "the harbor gate " + "detail " * 400)
    db.add_message(chat, "tool", "harbor gate ok " + "x" * 4000, tool_name="run_command")
    items = memory.recall("harbor gate", conversation_id=chat)
    excerpts = [item for item in items if item["source"] != "memory"]
    assert excerpts
    assert all(len(item["content"]) <= 900 for item in excerpts)


# --- life with the conversation ---------------------------------------------


def test_ledger_rows_go_when_the_conversation_is_deleted(db: Database, chat: str) -> None:
    ledger.record_user(db, chat, 1, "build a to-do API")
    ledger.record_tool(db, chat, 2, "run_command", {"command": "ls"}, True, "$ ls\nexit code: 0\n")
    assert len(db.list_ledger(chat)) == 2
    assert db.delete_conversation(chat) is True
    assert db.list_ledger(chat) == []


def test_clearing_a_conversation_clears_its_ledger(db: Database, chat: str) -> None:
    ledger.record_user(db, chat, 1, "build a to-do API")
    db.clear_conversation(chat)
    assert db.list_ledger(chat) == []


def test_delete_ledger_removes_only_that_conversations_rows(db: Database, chat: str) -> None:
    other = db.create_conversation()["id"]
    ledger.record_user(db, chat, 1, "build a to-do API")
    ledger.record_user(db, chat, 2, "add due dates")
    ledger.record_user(db, other, 1, "plan the trip")
    assert db.delete_ledger(chat) == 2
    assert db.list_ledger(chat) == []
    assert [row["key"] for row in db.list_ledger(other)] == ["plan the trip"]


# --- the recall tool over steps and saved memories ---------------------------


def _recall_tool(memory: MemoryEngine, chat: str, query: str, limit: int = 6) -> tools.ToolResult:
    ctx = tools.ToolContext(
        workspace=None, memory=memory, search=None, conversation_id=chat,  # type: ignore[arg-type]
    )
    return tools.execute_tool("recall", tools.validate_args("recall", {"query": query, "limit": limit}), ctx)


def test_recall_tool_returns_a_long_tool_result_in_full(
    db: Database, chat: str, memory: MemoryEngine
) -> None:
    db.add_message(chat, "user", "Check the demo logs")
    log = " ".join(f"row {n} of the demo log" for n in range(1, 90))
    seq = db.add_message(chat, "tool", log, tool_name="run_command")["seq"]
    assert len(log) > 1500
    result = _recall_tool(memory, chat, f"step {seq}")
    assert result.ok
    assert result.content == f"- [this chat, step {seq}, tool result] {log}"
    assert "[cut:" not in result.content


def test_recall_tool_returns_a_missing_step_message_unlabelled(
    db: Database, chat: str, memory: MemoryEngine
) -> None:
    db.add_message(chat, "user", "Plan the harbor trip")
    result = _recall_tool(memory, chat, "step 14")
    assert result.ok
    assert result.content == "No step 14 in this conversation (it has 1 step)."
    assert not result.content.startswith("- [")


def test_recall_tool_keeps_saved_memories_labelled_and_cut_at_600(
    db: Database, chat: str, memory: MemoryEngine
) -> None:
    short = "User keeps bees in Lisbon"
    db.create_memory("fact", short, normalize(short))
    long_fact = "the harbor gate code is 4815. " + " ".join(f"detail {n}" for n in range(1, 60))
    db.create_memory("fact", long_fact, normalize(long_fact))
    assert _recall_tool(memory, chat, "bees").content == f"- [memory] {short}"
    line = _recall_tool(memory, chat, "harbor gate code").content.splitlines()[0]
    assert line == f"- [memory] {long_fact[:600]}"
    assert len(long_fact) > 600


def test_recall_tool_returns_both_steps_of_a_range(
    db: Database, chat: str, memory: MemoryEngine
) -> None:
    db.add_message(chat, "user", "Plan the harbor trip")
    db.add_message(chat, "user", "Book the ferry for Tuesday")
    db.add_message(chat, "assistant", "Ferry booked for Tuesday morning.")
    result = _recall_tool(memory, chat, "steps 2-3")
    assert result.ok
    assert result.content.splitlines() == [
        "- [this chat, step 2, user] Book the ferry for Tuesday",
        "- [this chat, step 3, Kiwibro] Ferry booked for Tuesday morning.",
    ]
