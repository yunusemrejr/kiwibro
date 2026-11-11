"""Turn plan: multi-step requests are split, kept on track, verified and stopped."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_backend import Env, chat_events, tool_call_result

from kiwibro import plan, prompts, research
from kiwibro.context import ContextEngine
from kiwibro.llm import ChatResult

READ_RESULTS = research.read_results  # the real one: conftest switches page reading off for every other test

EXAMPLE = (
    "Hey, tell me briefly what a Raspberry Pi is, then search the web for the latest Raspberry Pi 5 news, "
    "and finally write me a markdown doc at ~/Desktop/goal/raspberry.md summarizing what you found."
)
DOC = "# Raspberry Pi 5 news\n\nThe 16GB model went on sale for $120; memory prices pushed costs up.\n"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    (home / "Desktop").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-config"))
    return home


@pytest.fixture
def web(env: Env, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    queries: list[str] = []

    def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
        queries.append(query)
        return [{"title": "16GB Raspberry Pi 5 on sale now at $120", "url": "https://www.raspberrypi.com/news/16gb",
                 "snippet": "Today we're launching the 16GB variant, priced at $120."}]

    monkeypatch.setattr(env.svc.search, "web_search", web_search)
    return queries


def _tool_messages(env: Env) -> list[dict]:
    conversation_id = env.svc.db.list_conversations()[0]["id"]
    return [m for m in env.svc.db.list_messages(conversation_id) if m["role"] == "tool"]


# --- splitting and classification ------------------------------------------------------


def test_the_example_request_becomes_three_ordered_steps(home: Path, tmp_path: Path) -> None:
    built = plan.build(EXAMPLE, tmp_path / "workspace")
    assert [s.kind for s in built.steps] == ["answer", "web", "write"]
    assert built.steps[0].text == "tell me briefly what a Raspberry Pi is"
    target = built.steps[2].target
    assert target.path == home / "Desktop" / "goal" / "raspberry.md" and target.outside and not target.is_dir
    note = built.note()
    assert note.startswith("This request has 3 steps.")
    assert 'write_file with path "~/Desktop/goal/raspberry.md"' in note
    assert "do not paste a saved file's content again" in note and "final reply" not in note.lower()


@pytest.mark.parametrize(
    ("message", "steps"),
    [
        ("search the web about cats and dogs and write a doc on my desktop",
         ["search the web about cats and dogs", "write a doc on my desktop"]),
        ("find and fix the bug in app.py", ["find and fix the bug in app.py"]),
        ("1. Search for Python web frameworks\n2. Compare Flask and Django\n3. Save the comparison to fw.md",
         ["Search for Python web frameworks", "Compare Flask and Django", "Save the comparison to fw.md"]),
        ("Look up the weather in Istanbul today, then tell me what to wear.",
         ["Look up the weather in Istanbul today", "tell me what to wear."]),
        ("What is DNS?", ["What is DNS?"]),
        ("ok so there has been a cyberattack recently, like yesterday, on PHP. php zeroshell? phpshell? "
         "something like that. search that and summarize it to be in a bulletin",
         ["search that", "summarize it to be in a bulletin"]),  # the story before it is context
        ("I moved to Ankara last week.", ["I moved to Ankara last week."]),
    ],
)
def test_split_steps(message: str, steps: list[str]) -> None:
    assert plan.split_steps(message) == steps


def test_plain_chat_and_single_questions_get_no_plan(home: Path, tmp_path: Path) -> None:
    for message in ("hi there", "What is the capital of France?", "explain how DNS works",
                    "write a poem about the sea", "find and fix the bug in app.py"):
        assert plan.build(message, tmp_path) is None, message


def test_constraints_are_not_answers_to_check(home: Path, tmp_path: Path) -> None:
    built = plan.build("Create a.txt containing kiwi. Make sure it ends with a newline.", tmp_path)
    assert [s.kind for s in built.steps] == ["write", "other"]
    assert not built.checkable


def test_places_and_file_names_combine(home: Path, tmp_path: Path) -> None:
    built = plan.build("Write me a doc about kiwis and save it on my desktop as kiwis.md", tmp_path)
    assert len(built.steps) == 1  # "…and save it…" is the same deliverable
    assert built.steps[0].target.path == home / "Desktop" / "kiwis.md"
    folder = plan.build("I want a doc on my desktop with the top 5 AI news", tmp_path).steps[0].target
    assert folder.path == home / "Desktop" and folder.is_dir


def test_forbidden_places_are_never_targets(home: Path, tmp_path: Path) -> None:
    for message in ("write the report to ~/.ssh/config", "save the notes to /etc/hosts", "save notes.md in ~/"):
        built = plan.build(message, tmp_path / "workspace")
        assert built is None or all(not s.target or not s.target.outside for s in built.steps), message


def test_xdg_folders_are_honoured(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "cfg"
    config.mkdir()
    (config / "user-dirs.dirs").write_text('XDG_DESKTOP_DIR="$HOME/Masaüstü"\n', encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    built = plan.build("save a summary file on my desktop", tmp_path / "workspace")
    assert built.steps[0].target.path == home / "Masaüstü"


def test_search_helpers() -> None:
    first = plan.search_terms("web_search", {"query": "Raspberry Pi 5 latest news release"})
    again = plan.search_terms("web_search", {"query": "latest Raspberry Pi 5 release news"})
    other = plan.search_terms("web_search", {"query": "Pi 5 power supply requirements"})
    assert plan.repeated(again, [(first, "Raspberry Pi 5 latest news release")]) == "Raspberry Pi 5 latest news release"
    assert plan.repeated(other, [(first, "q")]) is None
    assert plan.search_terms("fetch_page", {"url": "https://x.org"}) == frozenset()
    assert plan.is_small_talk("hi") and plan.is_small_talk("Thanks a lot!") and not plan.is_small_talk("continue")
    assert not plan.is_small_talk("ok") and not plan.is_small_talk("great")  # often "go ahead"


# --- the agent loop -------------------------------------------------------------------


def test_multi_step_request_runs_to_completion_without_asking(env: Env, home: Path, web: list[str]) -> None:
    target = home / "Desktop" / "goal" / "raspberry.md"
    env.fake.stream_script = [  # Kiwibro makes the search itself, before the first round
        tool_call_result("write_file", {"path": "~/Desktop/goal/raspberry.md", "content": DOC}, "call_2"),
        ChatResult(content="A Raspberry Pi is a small, cheap single-board computer. I saved the news to "
                           "~/Desktop/goal/raspberry.md."),
    ]
    events, _raw = chat_events(env, EXAMPLE)
    names = [event for event, _ in events]
    assert names[0] == "meta" and "plan" in names  # meta is always first
    assert "approval" not in names  # the user named the place: no prompt, no refusal
    assert target.read_text() == DOC
    calls = env.fake.stream_calls
    assert len(calls) == 2  # write, reply — no extra review round
    assert web == ["the latest Raspberry Pi 5 news"]
    assert any("This request has 3 steps" in str(m["content"]) for m in calls[0]["messages"] if m["role"] == "user")
    tools_out = [m["content"] for m in _tool_messages(env)]
    assert "Next: 3. write me a markdown doc" in tools_out[0]
    assert "[Location: ~/Desktop/goal/raspberry.md]" in tools_out[1]
    assert "every step that needs a tool is done" in tools_out[1] and "raspberry.md exists" in tools_out[1]
    done = events[-1][1]
    assert events[-1][0] == "done" and all(step["done"] for step in done["plan"])
    shown = [payload for event, payload in events if event == "tool_call" and payload["name"] == "write_file"]
    assert shown[0]["args"]["path"] == "~/Desktop/goal/raspberry.md" and not shown[0]["requires_approval"]
    assert shown[0]["level"] == "low"


def test_a_dropped_folder_is_routed_to_the_named_file(env: Env, home: Path, web: list[str]) -> None:
    env.fake.stream_script = [
        tool_call_result("web_search", {"query": "Raspberry Pi 5 news"}),
        tool_call_result("write_file", {"path": "raspberry.md", "content": DOC}, "call_2"),
        ChatResult(content="A Raspberry Pi is a tiny computer; the summary is on your Desktop."),
    ]
    chat_events(env, EXAMPLE)
    assert (home / "Desktop" / "goal" / "raspberry.md").read_text() == DOC
    assert not (env.workspace / "raspberry.md").exists()


def test_an_unnamed_place_outside_the_workspace_stays_refused(env: Env, home: Path, web: list[str]) -> None:
    env.fake.stream_script = [
        tool_call_result("write_file", {"path": "~/Desktop/other.md", "content": DOC}),
        tool_call_result("write_file", {"path": "~/Desktop/goal/raspberry.md", "content": DOC}, "call_2"),
        ChatResult(content="Saved."),
    ]
    chat_events(env, "Write a doc about kiwis to ~/Desktop/goal/raspberry.md")
    assert not (home / "Desktop" / "other.md").exists()
    assert (home / "Desktop" / "goal" / "raspberry.md").exists()


def test_stopping_early_gets_one_corrective_nudge(env: Env, home: Path) -> None:
    env.fake.stream_script = [
        ChatResult(content="I'll write that doc for you."),  # announces, does nothing
        tool_call_result("write_file", {"path": "~/Desktop/goal/raspberry.md", "content": DOC}),
        ChatResult(content="Saved the doc to your Desktop."),
    ]
    chat_events(env, "Write a doc about the Raspberry Pi to ~/Desktop/goal/raspberry.md")
    nudge = env.fake.stream_calls[1]["messages"][-1]["content"]
    assert nudge.startswith("The user's request is not finished.") and "1. Write a doc" in nudge
    assert (home / "Desktop" / "goal" / "raspberry.md").exists()
    stored = [m["content"] for m in env.svc.db.list_messages(env.svc.db.list_conversations()[0]["id"])]
    assert not any("is not finished" in str(content) for content in stored)  # transient


def test_the_search_a_request_asks_for_is_made_before_the_model_answers(env: Env, home: Path, web: list[str]) -> None:
    env.fake.stream_script = [
        tool_call_result("write_file", {"path": "~/Desktop/goal/raspberry.md", "content": DOC}),
        ChatResult(content="Saved the news to your Desktop."),
    ]
    events, _raw = chat_events(env, EXAMPLE)
    assert web == ["the latest Raspberry Pi 5 news"]  # Kiwibro's own call, worded from the request
    nudge = env.fake.stream_calls[0]["messages"][-1]["content"]  # the model's very first request has the results
    assert nudge == prompts.SEARCHED_FIRST.format(then=prompts.SEARCHED_FIRST_THEN)
    assert env.fake.stream_calls[0]["messages"][-2]["role"] == "tool"
    names = [event for event, _ in events]
    assert names.index("tool_result") < names.index("delta")  # nothing was written before the search
    messages = env.svc.db.list_messages(env.svc.db.list_conversations()[0]["id"])
    first = next(m for m in messages if m["role"] == "assistant")
    assert first["content"] == "" and first["tool_calls"][0]["function"]["name"] == "web_search"
    assert (home / "Desktop" / "goal" / "raspberry.md").exists()
    assert [payload for event, payload in events if event == "done"][0]["inference"]["searched_first"] is True


def test_a_follow_up_that_asks_to_look_it_up_is_searched_before_the_answer(env: Env, web: list[str]) -> None:
    env.fake.stream_script = [ChatResult(content="Inception was directed by Christopher Nolan.")]
    events, _raw = chat_events(env, "Who directed Inception?")
    conversation_id = events[0][1]["conversation_id"]
    assert web == []  # a plain question is not a search request
    env.fake.stream_script = [ChatResult(content="The results only cover the 16GB Raspberry Pi 5, not his age.")]
    chat_events(env, "how old is he? look it up", conversation_id=conversation_id)
    assert len(web) == 1 and web[0].startswith("how old is he") and "Christopher Nolan" in web[0]
    assert env.fake.stream_calls[-1]["messages"][-1]["content"] == prompts.SEARCHED_FIRST.format(then="")
    stored = [m["content"] for m in env.svc.db.list_messages(conversation_id) if m["role"] == "assistant"]
    assert not any("85 years" in content for content in stored) and stored[-1].endswith("not his age.")


def test_a_model_that_repeats_the_search_kiwibro_made_is_not_searched_twice(env: Env, web: list[str]) -> None:
    env.fake.stream_script = [tool_call_result("web_search", {"query": "tallest building in Turkey"}),
                              ChatResult(content="It is the Çamlıca Tower.")]
    events, _raw = chat_events(env, "look up the tallest building in Turkey")
    assert web == ["the tallest building in Turkey"]
    assert _tool_messages(env)[1]["content"].startswith("Not searched: this turn already searched")
    assert [payload for event, payload in events if event == "done"][0]["inference"]["searched_first"] is True


def test_an_unfinished_turn_says_so_honestly(env: Env, home: Path, web: list[str]) -> None:
    env.fake.stream_script = [ChatResult(content="Sure!") for _ in range(4)]  # never does the work
    events, _raw = chat_events(env, EXAMPLE)
    assert len(env.fake.stream_calls) == 1 + plan.MAX_FOLLOWUPS  # the search was made first; the doc never was
    text = "".join(payload.get("text", "") for event, payload in events if event == "delta")
    assert "Not finished: 3. write me a markdown doc" in text and "2. search the web" not in text


def test_a_cut_off_write_never_saves_a_truncated_file(env: Env, home: Path, web: list[str]) -> None:
    cut = tool_call_result("write_file", {"path": "~/Desktop/goal/raspberry.md", "content": DOC[:30]})
    cut.finish_reason = "length"
    env.fake.stream_script = [
        cut,
        tool_call_result("write_file", {"path": "~/Desktop/goal/raspberry.md", "content": DOC}, "call_2"),
        ChatResult(content="Saved the doc about kiwis."),
    ]
    chat_events(env, "Write a doc about kiwis to ~/Desktop/goal/raspberry.md")
    assert _tool_messages(env)[0]["content"] == prompts.CUT_OFF_CALL
    assert (home / "Desktop" / "goal" / "raspberry.md").read_text() == DOC


def test_near_repeat_searches_and_endless_research_are_cut_short(env: Env, home: Path, web: list[str]) -> None:
    def search(query: str, call_id: str) -> ChatResult:
        return tool_call_result("web_search", {"query": query}, call_id)

    env.fake.stream_script = [  # Kiwibro's own search comes first
        search("the PHP2Shell zero day advisory", "c2"),  # nearly the same
        search("PHP2Shell patched versions", "c3"),  # one more round after the plan's search
        search("PHP2Shell exploitation timeline", "c4"),  # refused: research is over
        ChatResult(content="Bulletin: PHP2Shell is a pair of critical PHP flaws; patch now."),
    ]
    chat_events(env, "search the web for the PHP2Shell zero day and summarize it as a bulletin")
    assert web == ["the PHP2Shell zero day", "PHP2Shell patched versions"]
    contents = [m["content"] for m in _tool_messages(env)]
    assert contents[1].startswith("Not searched: this turn already searched for almost the same thing")
    assert contents[3] == prompts.SEARCH_SPENT


def test_a_greeting_runs_no_tools(env: Env) -> None:
    env.fake.stream_script = [tool_call_result("study_file", {"path": "/doc", "question": "hello"}),
                              ChatResult(content="Hi! How can I help?")]
    chat_events(env, "hi")
    assert _tool_messages(env)[0]["content"] == prompts.SMALL_TALK_NO_TOOLS


def test_writing_rounds_get_room_for_the_whole_file(env: Env, home: Path, web: list[str], monkeypatch) -> None:
    caps: list[int] = []
    original = env.fake.stream_chat

    def stream_chat(**kwargs):
        caps.append(kwargs["max_tokens"])
        return original(**kwargs)

    monkeypatch.setattr(env.svc.llm, "stream_chat", stream_chat)
    env.fake.stream_script = [
        tool_call_result("write_file", {"path": "~/Desktop/goal/raspberry.md", "content": DOC}),
        ChatResult(content="Saved."),
    ]
    chat_events(env, "Write a doc about kiwis to ~/Desktop/goal/raspberry.md")
    assert caps[0] > 2048 and caps[1] == 2048  # raised only while the file is still unwritten


# --- context: the request being worked on is never compacted away ----------------------


def test_compaction_keeps_the_current_request(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    db = env.svc.db
    conversation = db.create_conversation(title="t", model="fake-1")
    cid = conversation["id"]
    db.add_message(cid, "user", "Research kiwi birds and write a report.")
    for index in range(6):
        call = {"id": f"c{index}", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}
        db.add_message(cid, "assistant", "", tool_calls=[call])
        db.add_message(cid, "tool", "x" * 6000, tool_call_id=f"c{index}", tool_name="web_search")
    engine: ContextEngine = env.svc.context
    monkeypatch.setattr(engine, "budget_for", lambda model: (8192, 6000, None))
    monkeypatch.setattr(engine, "inventory", lambda model: (8192, None))
    prepared = engine.prepare(cid, "fake-1", "system", turn_context="[note]")
    users = [m for m in prepared.messages if m["role"] == "user"]
    assert users and users[0]["content"].startswith("Research kiwi birds") and users[0]["content"].endswith("[note]")
    assert "Research kiwi birds" not in (db.get_conversation(cid)["summary"] or "")


def test_a_named_folder_allows_new_files_only(env: Env, home: Path, web: list[str]) -> None:
    project = home / "Desktop" / "project"
    project.mkdir()
    (project / "app.py").write_text("print('mine')\n")
    env.fake.stream_script = [
        tool_call_result("write_file", {"path": "~/Desktop/project/app.py", "content": "oops\n"}),
        tool_call_result("write_file", {"path": "~/Desktop/kiwis.md", "content": DOC}, "call_2"),
        ChatResult(content="Saved the doc about kiwis on your desktop."),
    ]
    chat_events(env, "write a doc about kiwis on my desktop")
    assert (project / "app.py").read_text() == "print('mine')\n"  # an existing file there is not the user's ask
    assert (home / "Desktop" / "kiwis.md").read_text() == DOC
    assert "outside the workspace" in _tool_messages(env)[0]["content"]


def test_pages_already_shown_this_turn_lose_their_snippet(env: Env, home: Path, monkeypatch) -> None:
    def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
        return [{"title": "PHP2Shell advisory", "url": "https://www.php.net/advisory", "snippet": "Patch to 8.5.12. " * 20},
                {"title": f"Coverage of {query}", "url": f"https://news.example/{len(query)}", "snippet": "Details."}]

    monkeypatch.setattr(env.svc.search, "web_search", web_search)
    env.fake.stream_script = [  # Kiwibro's own search for the PHP2Shell zero day comes first
        tool_call_result("web_search", {"query": "PHP2Shell exploitation timeline"}, "call_2"),
        ChatResult(content="Bulletin: PHP2Shell, patch now."),
    ]
    chat_events(env, "search the web for the PHP2Shell zero day and summarize it as a bulletin")
    first, second = (m["content"] for m in _tool_messages(env))
    assert "Patch to 8.5.12." in first
    assert "https://www.php.net/advisory\n(already shown above)" in second and "Patch to 8.5.12." not in second
    assert "Details." in second  # a new page keeps its snippet


def test_thinking_past_the_budget_acts_with_thinking_off(env: Env, home: Path, monkeypatch) -> None:
    """Seen live: a 4B model drafted the file in its thinking for 20 minutes and never called write_file."""
    from kiwibro import inference

    env.fake.models[0]["reasoning"] = {"allowed_options": ["off", "on"], "default": "on"}
    env.settings_patch({"reasoning": "on"})
    calls: list[dict] = []

    def stream_chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:  # drafts in its head without end (varied, so no loop detector fires)
            for i in range(200):
                kwargs["on_reasoning"](f"Section {i} could cover kiwi fact {i * 7 % 23}, citing source {i % 5}; ")
            return ChatResult(reasoning="…")
        if len(calls) == 2:
            return tool_call_result("write_file", {"path": "~/Desktop/goal/raspberry.md", "content": DOC})
        return ChatResult(content="Saved the doc about kiwis to ~/Desktop/goal/raspberry.md.")

    monkeypatch.setattr(env.svc.llm, "stream_chat", stream_chat)
    chat_events(env, "Write a doc about kiwis to ~/Desktop/goal/raspberry.md")
    assert len(calls) == 3
    retry = calls[1]
    assert retry["extra"]["reasoning_effort"] == "none"  # this one request only
    assert retry["messages"][-1]["content"].startswith("You have thought about this long enough.")
    assert 'write_file with path "~/Desktop/goal/raspberry.md"' in retry["messages"][-1]["content"]
    assert (calls[2]["extra"] or {}).get("reasoning_effort") != "none"  # later rounds think again
    assert (home / "Desktop" / "goal" / "raspberry.md").read_text() == DOC
    assert inference.THINK_BUDGET_CHARS < 200 * 50  # the fake really overran it


def test_direct_engine_steers_thinking_with_budgets_and_never_changes_the_template(
    env: Env, home: Path, monkeypatch
) -> None:
    """llama.cpp ends the thought at the budget inside the request: no cut-off, no re-prefill."""
    from kiwibro import inference

    env.fake.models[0]["reasoning"] = {"allowed_options": ["off", "on"], "default": "on"}
    env.settings_patch({"reasoning": "auto"})
    monkeypatch.setattr(env.svc.llm, "direct_on", lambda: True)
    calls: list[dict] = []

    def stream_chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            return tool_call_result("write_file", {"path": "~/Desktop/goal/raspberry.md", "content": DOC})
        return ChatResult(content="Done.")

    monkeypatch.setattr(env.svc.llm, "stream_chat", stream_chat)
    chat_events(env, "hi there")  # small talk: no thinking, but the same template
    env.settings_patch({"reasoning": "on"})
    chat_events(env, "Write a doc about kiwis to ~/Desktop/goal/raspberry.md")  # thinks, and must act
    templates = [{k: v for k, v in c["extra"].items() if k not in inference.BUDGET_KEYS} for c in calls]
    assert all(t == {"chat_template_kwargs": {"enable_thinking": True}} for t in templates), templates
    assert [c["extra"]["thinking_budget_tokens"] for c in calls] == [
        0, inference.THINK_BUDGET_TOKENS, inference.THINK_BUDGET_TOKENS,
    ]
    assert (home / "Desktop" / "goal" / "raspberry.md").read_text() == DOC


def test_direct_engine_that_ignores_the_budget_falls_back_to_thinking_off(env: Env, home: Path, monkeypatch) -> None:
    env.fake.models[0]["reasoning"] = {"allowed_options": ["off", "on"], "default": "on"}
    env.settings_patch({"reasoning": "on"})
    monkeypatch.setattr(env.svc.llm, "direct_on", lambda: True)
    calls: list[dict] = []

    def stream_chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:  # an old llama.cpp build: the budget field is ignored
            for i in range(400):
                kwargs["on_reasoning"](f"Section {i} could cover kiwi fact {i * 7 % 23}, citing source {i % 5}; ")
            return ChatResult(reasoning="…")
        if len(calls) == 2:
            return tool_call_result("write_file", {"path": "~/Desktop/goal/raspberry.md", "content": DOC})
        return ChatResult(content="Saved.")

    monkeypatch.setattr(env.svc.llm, "stream_chat", stream_chat)
    chat_events(env, "Write a doc about kiwis to ~/Desktop/goal/raspberry.md")
    retry = calls[1]["extra"]
    assert retry["thinking_budget_tokens"] == 0 and retry["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_budget_message" not in retry
    assert calls[1]["messages"][-1]["content"].startswith("You have thought about this long enough.")
    assert calls[2]["extra"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert (home / "Desktop" / "goal" / "raspberry.md").read_text() == DOC


def test_routing_never_escapes_the_named_place(home: Path, tmp_path: Path) -> None:
    from kiwibro.sandbox import Workspace

    (home / ".ssh").mkdir()
    workspace = Workspace(tmp_path / "workspace")
    built = plan.build("write a doc about kiwis to ~/Desktop/goal/raspberry.md", workspace.root)
    assert built.route("write_file", {"path": "~/Desktop/goal/raspberry.md", "content": "x"}, workspace).args["path"] == "raspberry.md"
    for escape in ("~/Desktop/goal/../../.ssh/id_rsa", "~/Desktop/goal/other.md", "../.bashrc", "/etc/passwd"):
        assert built.route("write_file", {"path": escape, "content": "x"}, workspace) is None, escape
    assert built.route("run_command", {"path": "~/Desktop/goal/raspberry.md"}, workspace) is None  # file tools only


@pytest.mark.parametrize(
    "message, query",
    [
        ("search the web for the next solar eclipse", "the next solar eclipse"),
        ("google the weather in Izmir", "the weather in Izmir"),
        ("find out who wrote Dune", "who wrote Dune"),
        ("do some research on heat pumps and write me a summary", "heat pumps"),
        ("can you tell me the latest news on the Fed?", "the latest news on the Fed"),
        ("who is the current president of France?", "who is the current president of France"),
        ("look up my flight TK1234 status", "my flight TK1234 status"),  # told to, in so many words
    ],
)
def test_a_search_kiwibro_may_run_by_itself(message, query):
    assert plan.search_request(message) == (query, False)


@pytest.mark.parametrize(
    "message",
    [
        # The user's own things never go to a search engine unasked.
        "what did we decide this week?",
        "show my recent notes",
        "search my notes for the dentist",
        "summarise this week's notes",
        "find out what my sister said",
        "what did you tell me recently?",
        "latest version of my code looks wrong, why?",
        "currently reading Dune, any tips?",
        # Not a request to look anything up.
        "Summarize the PHP2Shell news as a bulletin.",
        "research plan for my thesis: outline it",
        "explain what a web server is",
        "is online banking safe?",
        "what's the current time?",
        "what is the current directory?",
        "tell me about kiwi birds",
        "read https://example.com/latest and summarise it",
    ],
)
def test_no_search_is_made_unasked(message):
    assert plan.search_request(message) == ("", False)


def test_the_separate_questions_of_a_message_are_found():
    assert plan.questions("who is my wife and what am I allergic to?") == ["who is my wife", "what am I allergic to?"]
    assert plan.questions("what is 2+2?") == []
    assert plan.questions("search the web for kiwi birds and write a summary to kiwi.md") == []  # work, not questions


def test_a_question_the_reply_skipped_is_noticed_but_a_differently_worded_answer_is_not():
    asked = plan.questions("who is my wife and what am I allergic to?")
    assert plan.unanswered(asked, "Your wife is Elif.") == ["what am I allergic to?"]
    assert plan.unanswered(asked, "Your wife is Elif and you are allergic to peanuts.") == []
    asked = plan.questions("how tall is Everest and when was it first climbed?")
    assert plan.unanswered(asked, "It is 8,849 m and was first summited in 1953.") == []


def test_a_skipped_question_gets_one_push(env: Env):
    env.fake.stream_script = [ChatResult(content="Your wife is Elif."),
                              ChatResult(content="You are allergic to peanuts.")]
    events, _raw = chat_events(env, "who is my wife and what am I allergic to?")
    assert "The message asks 2 things." in env.fake.stream_calls[0]["messages"][-1]["content"]
    assert env.fake.stream_calls[1]["messages"][-1]["content"].startswith("[Kiwibro: your reply left part")
    assert "“what am I allergic to?”" in env.fake.stream_calls[1]["messages"][-1]["content"]
    assert len(env.fake.stream_calls) == 2


def test_a_question_about_kiwibro_itself_runs_no_tools(env: Env):
    env.fake.stream_script = [tool_call_result("recall", {"query": "who are you"}),
                              ChatResult(content="I'm Kiwibro, an AI agent on your computer. I can search, keep notes and work on files.")]
    chat_events(env, "who are you and what can you do?")
    assert _tool_messages(env)[0]["content"] == prompts.ABOUT_YOU_NO_TOOLS


@pytest.mark.parametrize(
    "message, about",
    [
        ("who are you and what can you do?", True),
        ("hi, what can you do for me?", True),
        ("are you an AI?", True),
        ("who are you and what's the weather?", False),
        ("what can you do with this file?", False),
        ("what are you working on?", False),
    ],
)
def test_questions_about_kiwibro_are_told_from_requests(message, about):
    assert plan.is_about_kiwibro(message) is about


@pytest.mark.parametrize("message", [
    "what's the weather in Istanbul?", "who won the F1 race yesterday?", "bitcoin price",
    "who is the president of Argentina?", "when is the release date of GTA 6?",
    "is the new macbook out yet?", "what are the best laptops in 2026?",
])
def test_live_world_questions_are_searched(message):
    query, borrowed = plan.search_request(message)
    assert query and not borrowed


@pytest.mark.parametrize("message", [
    "how tall is mount everest?", "what time is it?", "what's the date today?", "what should I do today?",
    "what did we talk about yesterday?", "run the tests and show me the results", "explain recursion",
    "write a file prices.csv with these prices",
])
def test_timeless_own_and_clock_questions_are_not_searched(message):
    assert plan.search_request(message) == ("", False)


FORECAST = ("Izmir, Türkiye right now: 18 °C (64 °F), clear sky, feels like 17 °C. Wind 9 km/h, humidity 60%.\n\n"
            "- Today: 12 to 19 °C, clear sky, 0% chance of rain\n\nSource: Open-Meteo, as of 19:15 local time there.")


def test_a_weather_question_is_answered_from_forecast_data(env: Env, web: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []
    monkeypatch.setattr(env.svc.search, "weather", lambda question: asked.append(question) or FORECAST)
    events, _raw = chat_events(env, "whats the weather in izmir")
    assert asked == ["whats the weather in izmir"] and web == []  # the data, not a page of results
    chat_events(env, "how hot is it in Dubai")  # worded without "weather": looked up all the same
    assert asked[-1] == "how hot is it in Dubai" and web == [] and env.fake.stream_calls == []
    assert env.fake.stream_calls == []  # no model wrote a number
    messages = env.svc.db.list_messages(env.svc.db.list_conversations()[-1]["id"])
    assert messages[-1]["content"] == FORECAST
    assert [payload for event, payload in events if event == "done"][0]["inference"]["answered_from_look"] is True


def test_without_forecast_data_a_weather_question_is_a_web_search(env: Env, web: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env.svc.search, "weather", lambda question: "")
    env.fake.stream_script = [ChatResult(content="The results do not give the temperature.")]
    chat_events(env, "whats the weather in izmir")
    assert len(web) == 1 and len(env.fake.stream_calls) == 1


INVENTED = ("Elon Musk unveiled the Zorblax 9000 at Contoso Labs in Reykjavik for $999, and Satya Nadella "
            "praised the Quantum Falcon chip from Wonka Industries at a Helsinki event with 4,200 guests.")


def test_a_second_attempt_that_says_nothing_is_not_the_answer(env: Env, web: list[str]) -> None:
    env.fake.stream_script = [ChatResult(content=INVENTED), ChatResult(content="The answer is.\n\nThe answer is.")]
    events, _raw = chat_events(env, "look up the latest Raspberry Pi 5 news")
    messages = env.svc.db.list_messages(env.svc.db.list_conversations()[0]["id"])
    shown = messages[-1]["content"]
    assert "The answer is" not in shown and "Zorblax" not in shown
    assert "16GB Raspberry Pi 5" in shown  # what the search found, in its own words
    done = [payload for event, payload in events if event == "done"][0]
    assert done["inference"]["grounding"]["shown_as_found"] is True


def test_a_short_second_attempt_that_names_the_fact_is_kept(env: Env, web: list[str]) -> None:
    env.fake.stream_script = [ChatResult(content=INVENTED), ChatResult(content="The 16GB Raspberry Pi 5 costs $120.")]
    chat_events(env, "look up the latest Raspberry Pi 5 news")
    messages = env.svc.db.list_messages(env.svc.db.list_conversations()[0]["id"])
    assert messages[-1]["content"] == "The 16GB Raspberry Pi 5 costs $120."


def test_kiwibros_own_search_reads_the_pages_and_the_models_does_not(env: Env, web: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []

    def fetch_page(url: str, max_chars: int = 4000) -> dict:
        opened.append(url)
        return {"title": "16GB Raspberry Pi 5", "url": url, "truncated": False,
                "text": "The 16GB Raspberry Pi 5 went on sale on 9 January and ships from approved resellers worldwide."}

    monkeypatch.setattr(research, "read_results", READ_RESULTS)
    monkeypatch.setattr(env.svc.search, "fetch_page", fetch_page)
    env.fake.stream_script = [ChatResult(content="The 16GB Raspberry Pi 5 went on sale on 9 January.")]
    chat_events(env, "look up the latest Raspberry Pi 5 news")
    assert opened == ["https://www.raspberrypi.com/news/16gb"]
    assert "On the page: The 16GB Raspberry Pi 5 went on sale on 9 January" in _tool_messages(env)[0]["content"]
    opened.clear()
    env.fake.stream_script = [tool_call_result("web_search", {"query": "raspberry pi 5 resellers"}),
                              ChatResult(content="Sixteen gigabytes hum / approved resellers ship it / a small board, big dreams")]
    chat_events(env, "Write a haiku about it.", conversation_id=env.svc.db.list_conversations()[0]["id"])
    assert len(web) == 2
    assert opened == []  # a search the model asks for stays a quick one
