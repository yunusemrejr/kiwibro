"""Agent loop control for small models: failing-call coaching and the tool pause."""

from __future__ import annotations

import json

import pytest

from test_backend import Env, chat_events

from kiwibro.agent import FAILURE_COACH_AT, FAILURE_PAUSE_AT
from kiwibro.llm import ChatResult


def _call(path: str, index: int) -> ChatResult:
    call = {"id": f"call_{index}", "type": "function",
            "function": {"name": "list_files", "arguments": json.dumps({"path": path})}}
    return ChatResult(tool_calls=[call], usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25})


def _tool_messages(env: Env) -> list[dict]:
    conversation_id = env.svc.db.list_conversations()[0]["id"]
    return [m for m in env.svc.db.list_messages(conversation_id) if m["role"] == "tool"]


def test_failing_calls_get_coached_then_paused(env: Env) -> None:
    guesses = ["/tmp", "/Desktop", "/", "/etc", "/usr", "/opt", "/srv", "/var"]
    env.fake.stream_script = [_call(path, i) for i, path in enumerate(guesses)] + [
        ChatResult(content="I could not find those folders; the workspace is empty."),
    ]
    events, _raw = chat_events(env, "List the files in your workspace.")
    tool_messages = _tool_messages(env)
    assert all("outside the workspace; use a path relative to it" in m["content"] for m in tool_messages[:FAILURE_PAUSE_AT])
    assert "[Kiwibro: the last 3 tool calls failed" in tool_messages[FAILURE_COACH_AT - 1]["content"]
    paused = tool_messages[FAILURE_PAUSE_AT:]
    assert paused and all(m["content"].startswith("Tool use is paused") for m in paused)
    assert events[-1][0] == "done"


def test_a_pause_stops_only_the_tools_that_kept_failing(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """16 Nov 2026: six calls for a component that does not exist paused tools, and the write_file that
    finally made the page was refused with them."""
    from kiwibro import pagecheck
    from test_pagecheck import Fake

    monkeypatch.setattr(pagecheck, "chrome_render", Fake())
    guesses = ["/tmp", "/Desktop", "/", "/etc", "/usr", "/opt", "/srv"]
    write = {"id": "call_w", "type": "function",
             "function": {"name": "write_file", "arguments": json.dumps({"path": "page.html", "content": "<p>hi</p>"})}}
    env.fake.stream_script = [_call(path, i) for i, path in enumerate(guesses)] + [
        ChatResult(tool_calls=[write], usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}),
        ChatResult(content="Wrote page.html."),
    ]
    chat_events(env, "List the files in your workspace, then write page.html.")
    contents = [m["content"] for m in _tool_messages(env)]
    assert contents[FAILURE_PAUSE_AT].startswith("Tool use is paused") and "list_files" in contents[FAILURE_PAUSE_AT]
    assert any(c.startswith("Wrote") for c in contents[FAILURE_PAUSE_AT:]) and (env.workspace / "page.html").exists()


def _kit_call(message_index: int) -> ChatResult:
    call = {"id": f"call_k{message_index}", "type": "function",
            "function": {"name": "component", "arguments": json.dumps(
                {"action": "use", "name": "web/animation-2d", "path": "anim", "values": {}})}}
    return ChatResult(tool_calls=[call], usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25})


def test_a_finished_kit_runs_only_when_the_request_calls_for_it(env: Env) -> None:
    """16 Nov 2026: asked for a kiwi walking to a tree, the 2D animation kit drew bouncing balls."""
    env.fake.stream_script = [_kit_call(0), ChatResult(content="I will draw it by hand.")]
    chat_events(env, "Create a hand-drawn kiwitree.html that animates a kiwi walking towards a tree.")
    refused = _tool_messages(env)[0]["content"]
    assert refused.startswith("This call did not run: web/animation-2d makes only") and "write_file" in refused
    assert not (env.workspace / "anim").exists()


def test_a_finished_kit_still_runs_when_asked_for(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    from kiwibro import pagecheck
    from test_pagecheck import Fake

    monkeypatch.setattr(pagecheck, "chrome_render", Fake())
    env.fake.stream_script = [_kit_call(0), ChatResult(content="Here are the balls.")]
    chat_events(env, "Make a bouncing balls animation.")
    assert (env.workspace / "anim" / "index.html").exists()


def test_a_named_supported_scene_is_built_and_checked_without_a_model(env: Env, monkeypatch: pytest.MonkeyPatch):
    from kiwibro import pagecheck
    from test_pagecheck import Fake

    monkeypatch.setattr(pagecheck, "chrome_render", Fake())
    events, _ = chat_events(env, "Create a single HTML file named dancer.html that animates a kiwi dancing.")
    assert (env.workspace / "dancer.html").is_file()
    assert not env.fake.stream_calls
    [done] = [payload for event, payload in events if event == "done"]
    assert done["inference"]["looked"] == "good"


def test_a_named_hinged_panel_is_created_and_reviewed_without_generation(env: Env,
                                                                        monkeypatch: pytest.MonkeyPatch):
    from kiwibro import pagecheck
    from test_pagecheck import Fake

    monkeypatch.setattr(pagecheck, "chrome_render", Fake())
    message = "Create `inspection.html` as a 3D hinged panel with a left-edge hinge."
    events, _ = chat_events(env, message)
    page = env.workspace / "inspection.html"
    written = page.read_bytes()
    assert not env.fake.stream_calls
    assert [(data["name"], data["ok"]) for kind, data in events if kind == "tool_result"] == [
        ("component", True), ("look_at_page", True),
    ]

    retry, _ = chat_events(env, message)
    assert page.read_bytes() == written
    assert not env.fake.stream_calls
    assert any(kind == "delta" and "already exists" in data["text"] for kind, data in retry)


def test_explicit_panel_fast_path_leaves_custom_requirements_for_model(env: Env):
    env.fake.stream_script = [ChatResult(content="This mechanism needs a custom layout.")]
    chat_events(env, "Create hatch.html as a 3D hinged panel with a left hinge, also add a geared hinge.")
    assert env.fake.stream_calls
    assert "web/hinged-panel" not in env.fake.stream_calls[0]["messages"][-1]["content"]
    assert not (env.workspace / "hatch.html").exists()


def test_long_hinge_brief_offers_the_component_to_the_model(env: Env):
    message = (
        "Save hatch.html as an inspection scene with a bottom-edge hinge, keyboard control and an offline preview. "
        + "Draw an inspection hatch in CSS. " * 17
    )
    assert len(message) > 600
    env.fake.stream_script = [ChatResult(content="I'll use the hinge component for the panel.")]
    chat_events(env, message)
    first = env.fake.stream_calls[0]
    assert 'name="web/hinged-panel"' in first["messages"][-1]["content"]


def test_an_explicit_browser_open_reports_the_landed_url_without_a_model(env: Env, monkeypatch: pytest.MonkeyPatch):
    from kiwibro import tools

    def opened(args, _ctx):
        assert args["action"] == "open" and args["url"] == "https://example.com"
        return tools.ToolResult(ok=True, content="Example page",
                                meta={"url": "https://example.org/docs", "tab_id": "example"})

    monkeypatch.setattr(tools, "_execute_browser", opened)
    events, _ = chat_events(env, "Navigate to https://example.com")
    assert not env.fake.stream_calls
    assert [payload["text"] for event, payload in events if event == "delta"] == [
        "Opened https://example.org/docs in the browser.",
    ]


def test_a_success_resets_the_failure_streak(env: Env) -> None:
    env.fake.stream_script = [_call("/tmp", 0), _call("/etc", 1), _call(".", 2), _call("/usr", 3),
                              ChatResult(content="Done.")]
    chat_events(env, "List the files in your workspace.")
    contents = [m["content"] for m in _tool_messages(env)]
    assert not any("[Kiwibro: the last" in content for content in contents)  # never 3 failures in a row


# --- degeneration guard ---------------------------------------------------------

from kiwibro import inference  # noqa: E402
from test_backend import SSEStream, tool_call_result  # noqa: E402


def test_degeneration_detector() -> None:
    assert inference.is_degenerate("I will create the files now.\n\n" + "...\n\n\n" * 60)
    assert inference.is_degenerate("Let me use the workspace path. " * 20)
    assert inference.is_degenerate("\n".join(["The workspace is given."] * 9) + "\n" + "x" * 300)
    prose = ("Kiwis are flightless birds native to New Zealand. They are nocturnal and have long beaks "
             "with nostrils at the tip, which they use to find food in the soil. ") * 2  # a passage and its copy
    code = "\n".join(f"def handler_{i}(event):\n    return process(event, retries={i})" for i in range(12))
    assert not inference.is_degenerate(prose)
    assert not inference.is_degenerate(code)
    assert inference.trim_degenerate("I will create a.txt.\n\n" + "...\n\n" * 50) == "I will create a.txt."


def test_degenerate_reply_is_cut_and_retried_once(env: Env) -> None:
    junk = "I need to create the file. Let me use the workspace path.\n\n" + "...\n\n\n" * 200
    env.fake.stream_script = [
        ChatResult(content=junk),
        tool_call_result("write_file", {"path": "a.txt", "content": "kiwi\n"}),
        ChatResult(content="Created a.txt."),
    ]
    events, _raw = chat_events(env, "Create a.txt containing kiwi.")
    calls = env.fake.stream_calls
    assert len(calls) == 3
    assert calls[1]["messages"][-1]["content"].startswith("Your last reply was going in circles")
    assert (env.workspace / "a.txt").read_text() == "kiwi\n"
    rewrites = [payload for event, payload in events if event == "rewrite"]
    assert rewrites and "..." not in rewrites[0]["text"]
    stored = [m["content"] for m in env.svc.db.list_messages(env.svc.db.list_conversations()[0]["id"])]
    assert not any("...\n\n\n..." in str(content) for content in stored)  # junk never stored


def test_degenerating_twice_ends_the_round_honestly(env: Env) -> None:
    junk = "...\n\n\n" * 200
    env.fake.stream_script = [ChatResult(content=junk), ChatResult(content=junk)]
    events, _raw = chat_events(env, "Do the thing.")
    assert len(env.fake.stream_calls) == 2
    assert events[-1][0] == "done"
    text = "".join(payload.get("text", "") for event, payload in events if event == "delta")
    assert "I got stuck repeating myself" in text


def test_deliberation_loop_detector() -> None:
    from pathlib import Path

    circling = " ".join(
        "So the content should be kiwi with a newline? No, wait. Let me re-read the request. "
        f"Maybe the file already ends with a newline? Actually that is confusing. Attempt {i}."
        for i in range(40)
    )
    assert inference.is_looping(circling)
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert not inference.is_looping(readme[:3000])
    steps = "".join(f"Step {i}: create part{i}.txt with the heading Part {i} and a summary.\n" for i in range(40))
    assert not inference.is_looping(steps)


def test_circling_reply_is_cut_and_retried(env: Env) -> None:
    circling = " ".join(
        "So the content should be kiwi with a newline? No, wait. Let me re-read the request. "
        f"Maybe the file already ends with a newline? Actually that is confusing. Attempt {i}."
        for i in range(40)
    )
    env.fake.stream_script = [
        ChatResult(content=circling),
        tool_call_result("write_file", {"path": "a.txt", "content": "kiwi\n"}),
        ChatResult(content="Created a.txt."),
    ]
    chat_events(env, "Create a.txt containing kiwi. Make sure it ends with a newline.")
    assert len(env.fake.stream_calls) == 3
    assert (env.workspace / "a.txt").read_text() == "kiwi\n"


def test_a_repeating_reply_keeps_its_start_and_is_continued(env: Env, monkeypatch) -> None:
    """The slop loop seen live ("**The fix**" + the same bullet, a dozen times): cut at the third
    copy, keep the good start and one copy, continue with anti-repetition sampling."""
    from kiwibro import prompts

    good = "**PHP2Shell bulletin**\n\n- Two critical PHP flaws: CVE-2026-109618 and CVE-2026-109619.\n\n"
    loop = "**The fix**\n\n- The fix is to update PHP; the original ones are not the same shape.\n\n" * 12
    calls: list[dict] = []

    def stream_chat(**kwargs):
        calls.append(kwargs)
        text = good + loop if len(calls) == 1 else "Patch to PHP 8.5.12 or later today."
        for start in range(0, len(text), 9):  # streamed in small pieces, as a model server does
            kwargs["on_delta"](text[start:start + 9])
        return ChatResult(content=text)

    monkeypatch.setattr(env.svc.llm, "stream_chat", stream_chat)
    monkeypatch.setattr(env.svc.search, "web_search", lambda query, max_results=5: [])  # nothing to check the reply against
    events, _raw = chat_events(env, "Summarize the PHP2Shell news as a bulletin.")
    assert len(calls) == 2
    kept = calls[1]["messages"][-2]
    assert kept["role"] == "assistant" and kept["content"].startswith("**PHP2Shell bulletin**")
    assert kept["content"].count("**The fix**") == 1
    assert calls[1]["messages"][-1]["content"] == prompts.DEGENERATE_CONTINUE
    assert calls[1]["extra"]["repeat_penalty"] > 1  # anti-repetition sampling, this request only
    stored = [m for m in env.svc.db.list_messages(env.svc.db.list_conversations()[0]["id"]) if m["role"] == "assistant"]
    assert stored[-1]["content"] == kept["content"] + "\n\nPatch to PHP 8.5.12 or later today."
    rewrites = [payload["text"] for event, payload in events if event == "rewrite"]
    assert rewrites == [kept["content"] + "\n\n"]  # the UI drops the loop, keeps the start


def test_wrapper_tags_around_an_answer_are_neither_shown_nor_stored(env: Env):
    env.fake.stream_script = [ChatResult(content="<output>\n400\n</output>")]
    events, _raw = chat_events(env, "give the code word")
    shown = "".join(payload["text"] for event, payload in events if event == "delta")
    assert shown.strip() == "400" and not [event for event, _ in events if event == "rewrite"]
    conversation_id = events[0][1]["conversation_id"]
    assert [m["content"] for m in env.svc.db.list_messages(conversation_id) if m["role"] == "assistant"] == ["400"]


def test_a_reply_that_only_starts_like_a_wrapper_keeps_its_tag(env: Env):
    text = "<output> is the HTML element for a calculation result."
    env.fake.stream_script = [ChatResult(content=text)]
    events, _raw = chat_events(env, "what is the output tag in HTML for?")
    conversation_id = events[0][1]["conversation_id"]
    assert [m["content"] for m in env.svc.db.list_messages(conversation_id) if m["role"] == "assistant"] == [text]
    assert [payload["text"] for event, payload in events if event == "rewrite"][-1] == text


def test_an_answer_given_before_one_more_tool_call_and_then_again_is_kept_once(env: Env):
    early = ("The latest stable Python version is 3.15.2, released on 25 October 2026. It is the second "
             "maintenance release of the 3.15 series.")
    final = "The latest stable Python version is **3.15.2**, released on 25 October 2026 (second maintenance release of 3.15)."
    again = tool_call_result("list_files", {"path": "."})
    again.content = early
    env.fake.stream_script = [again, ChatResult(content=final)]
    events, _raw = chat_events(env, "tell me about Python 3.15.2")
    assert [payload["text"] for event, payload in events if event == "retract"] == [early]
    conversation_id = events[0][1]["conversation_id"]
    texts = [m["content"] for m in env.svc.db.list_messages(conversation_id) if m["role"] == "assistant"]
    assert texts == ["", final]


def test_a_short_preface_before_a_tool_call_stays(env: Env):
    preface = tool_call_result("list_files", {"path": "."})
    preface.content = "Let me look at the files first."
    env.fake.stream_script = [preface, ChatResult(content="There are no files in the workspace yet.")]
    events, _raw = chat_events(env, "what files do I have?")
    assert not [event for event, _ in events if event == "retract"]


def test_a_sum_is_worked_out_before_the_model_replies_and_a_wrong_answer_is_redone(env: Env):
    env.fake.stream_script = [ChatResult(content="12% of 480 = 5760."), ChatResult(content="12% of 480 is 57.6.")]
    events, _raw = chat_events(env, "what is 12% of 480, rounded to one decimal place?")
    assert "- 12% of 480 = 57.6" in env.fake.stream_calls[0]["messages"][-1]["content"]
    assert "Kiwibro worked it out exactly: 12% of 480 = 57.6" in env.fake.stream_calls[1]["messages"][-1]["content"]
    conversation_id = events[0][1]["conversation_id"]
    shown = [m["content"] for m in env.svc.db.list_messages(conversation_id) if m["role"] == "assistant"]
    assert shown == ["12% of 480 is 57.6."]
    assert [payload for event, payload in events if event == "done"][0]["inference"]["steered"] == ["wrong_result"]


def test_a_standalone_sum_skips_generation(env: Env):
    events, _raw = chat_events(env, "what is 12% of 480?")
    assert not env.fake.stream_calls
    assert [payload["text"] for event, payload in events if event == "delta"] == ["12% of 480 = 57.6."]


def test_a_read_after_a_write_in_one_batch_sees_the_change(env: Env):
    target = env.workspace / "state.txt"
    target.write_text("previous contents")
    before = tool_call_result("read_file", {"path": "state.txt"})
    write = tool_call_result("replace_in_file", {"path": "state.txt", "old": "previous", "new": "updated"})
    after = tool_call_result("read_file", {"path": "state.txt"})
    for index, response in enumerate((before, write, after)):
        response.tool_calls[0]["id"] = f"change_{index}"
    before.tool_calls.extend(write.tool_calls + after.tool_calls)
    env.fake.stream_script = [before, ChatResult(content="The state file now says updated contents.")]
    events, _ = chat_events(env, "Inspect state.txt, change previous to updated, then read it again.")
    tool_results = [payload for event, payload in events if event == "tool_result"]
    assert [payload["id"] for payload in tool_results[:3]] == ["change_0", "change_1", "change_2"]
    assert "previous contents" in tool_results[0]["preview"]
    assert "updated contents" in tool_results[2]["preview"]


def test_stopping_at_later_approval_keeps_the_completed_tool_result(env: Env):
    env.settings_patch({"autonomy": "supervised", "permissions": {"replace_in_file": "auto", "file_manage": "ask"}})
    target = env.workspace / "state.txt"
    target.write_text("before")
    write = tool_call_result("replace_in_file", {"path": "state.txt", "old": "before", "new": "after"}, "write")
    delete = tool_call_result("file_manage", {"action": "delete", "path": "state.txt"}, "delete")
    later = tool_call_result("write_file", {"path": "later.txt", "content": "unexpected"}, "later")
    write.tool_calls.extend(delete.tool_calls + later.tool_calls)
    env.fake.stream_script = [write]

    response = env.client.post("/api/chat", json={"message": "Replace before with after in state.txt, then delete it."},
                               buffered=False)
    assert response.status_code == 200
    stream = SSEStream(response).start()
    conversation_id = stream.wait_for("meta")["conversation_id"]
    assert stream.wait_for("approval")["tool"] == "file_manage"
    assert target.read_text() == "after"
    assert env.new_client().post("/api/chat/stop", json={"conversation_id": conversation_id}).status_code == 200
    events = stream.finish()

    assert events[-1][0] == "done" and events[-1][1]["aborted"]
    results = [(payload["id"], payload["ok"]) for event, payload in events if event == "tool_result"]
    assert results == [("write", True), ("delete", False), ("later", False)]
    messages = _tool_messages(env)
    assert [message["tool_call_id"] for message in messages] == ["write", "delete", "later"]
    assert messages[0]["meta"]["ok"] is True
    assert all(message["content"].startswith("Cancelled before execution") for message in messages[1:])
    assert target.read_text() == "after" and not (env.workspace / "later.txt").exists()


SHELF = ("The top shelf holds 14 jars of plum jam, 9 jars of honey and 3 bottles of elderflower syrup. The lower "
         "shelf holds 22 tins of tomatoes, 17 bags of rice and 6 boxes of oat biscuits, counted on Monday morning. ")
SHELF_CHANGED = SHELF.replace("17 bags", "71 bags")


def test_a_reply_said_again_in_one_paragraph_is_a_loop() -> None:
    """Seen live: a 4B model restated a measurement over and over as one paragraph, with small changes
    between the copies, until the output limit."""
    loop = SHELF + SHELF_CHANGED + SHELF + SHELF
    assert inference.is_degenerate(loop)
    assert inference.trim_degenerate(loop) == SHELF.strip()
    assert not inference.is_degenerate(SHELF + SHELF_CHANGED)  # a text next to its corrected copy
    items = "".join(f"- On {day}, open the shed, count the jars on both shelves, write the totals in the book and "
                    f"lock the door behind you before you leave for the day.\n" for day in ("Monday", "Tuesday", "Friday", "Sunday"))
    assert not inference.is_degenerate(items)  # a list whose items share their wording
    answer = ("- Plum jam: 14 jars on the top shelf.\n- Honey: 9 jars beside it.\n- Syrup: 3 bottles at the back.\n"
              "- Tomatoes: 22 tins below.\n- Rice: 17 bags in the corner.\n- Biscuits: 6 boxes by the door.\n"
              "- Tea: 4 tins above the sink.\n- Flour: 2 sacks under it.\n- Salt: 5 tubs on the window sill.\n"
              "- Candles: 12 in the drawer by the stove.\n")
    assert not inference.is_degenerate(answer * 2)
    assert inference.is_degenerate(answer * 3)  # an answer in lines, begun for the third time
    assert inference.trim_degenerate(answer * 3) == answer.strip()
    fenced = "```\n" + SHELF * 4 + "\n```\n"
    assert not inference.is_degenerate(fenced)  # code repeats legitimately


def test_a_reply_that_says_itself_again_ends_with_what_it_said_first(env: Env, monkeypatch) -> None:
    """No second request: the answer was given, and sampling against repetition would garble its figures."""
    calls: list[dict] = []
    noted: list[str] = []

    def stream_chat(**kwargs):
        calls.append(kwargs)
        text = SHELF + SHELF_CHANGED + SHELF * 6
        for start in range(0, len(text), 9):  # streamed in small pieces, as a model server does
            kwargs["on_delta"](text[start:start + 9])
        return ChatResult(content=text)

    monkeypatch.setattr(env.svc.llm, "stream_chat", stream_chat)
    monkeypatch.setattr(env.svc.db, "note_trait", lambda model, kind: noted.append(kind))
    monkeypatch.setattr(env.svc.search, "web_search", lambda query, max_results=5: [])  # nothing to check the reply against
    events, _raw = chat_events(env, "Say what the two pantry shelves hold.")
    assert len(calls) == 1
    stored = [m for m in env.svc.db.list_messages(env.svc.db.list_conversations()[0]["id"]) if m["role"] == "assistant"]
    assert stored[-1]["content"] == SHELF.strip()
    rewrites = [payload["text"] for event, payload in events if event == "rewrite"]
    assert rewrites and rewrites[-1] == SHELF.strip()
    assert "loop" not in noted
