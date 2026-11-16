"""The turn looks at a page it wrote before it ends: what Kiwibro says itself, what the model is asked to fix,
when it stops, and the same loop, harder, for a message that starts with /goal."""

from __future__ import annotations

import json
from typing import Any

import pytest
from test_backend import Env, chat_events, tool_call_result
from test_pagecheck import CLEAN, Fake, script_error

from kiwibro import delegate, pagecheck, pageloop, prompts
from kiwibro.llm import ChatResult

PAGE = "<!doctype html><html><body><canvas></canvas><script>\nspin();\n</script></body></html>\n"
FIXED = "<!doctype html><html><body><canvas></canvas><script>\nconsole.log(1);\n</script></body></html>\n"
ASK = "make an html page with an animation of a bouncing ball"
SPIN = "make a page with a spinning cube that I can turn"


def messages(env: Env, role: str | None = None) -> list[dict[str, Any]]:
    stored = env.svc.db.list_messages(env.svc.db.list_conversations()[0]["id"])
    return [m for m in stored if role is None or m["role"] == role]


def shown(events: list[tuple[str, dict[str, Any]]]) -> str:
    """The reply as the screen ends up showing it: deltas appended, a retract taking back the text written last."""
    text = ""
    for name, data in events:
        if name == "delta":
            text += data["text"]
        elif name == "retract" and data.get("now"):
            text = text.removesuffix(data["text"])
    return text


def looks(env: Env) -> list[dict[str, Any]]:
    return [m for m in messages(env, "tool") if m["tool_name"] == "look_at_page"]


def write(path: str, content: str, call_id: str) -> ChatResult:
    return tool_call_result("write_file", {"path": path, "content": content}, call_id)


@pytest.fixture
def chrome(monkeypatch: pytest.MonkeyPatch):
    def use(*reports: dict[str, Any], **kwargs: Any) -> Fake:
        fake = Fake(*reports, **kwargs)
        monkeypatch.setattr(pagecheck, "chrome_render", fake)
        return fake

    return use


def test_a_script_error_is_fixed_on_the_second_look(env: Env, chrome) -> None:
    fake = chrome(script_error(2), CLEAN)
    env.fake.stream_script = [
        write("index.html", PAGE, "call_1"),
        ChatResult(content="Here is your page with a bouncing ball."),
        write("index.html", FIXED, "call_2"),
        ChatResult(content="I fixed the script."),
    ]
    events, _raw = chat_events(env, ASK)
    text = shown(events)
    ordered = [
        prompts.LOOK_FIRST.format(file="index.html"),
        prompts.LOOK_BAD.format(problems='a script error stops the page ("Uncaught ReferenceError: spin is not defined", line 2)'),
        prompts.LOOK_AGAIN,
        "It is good now: it loads without errors, it draws and it moves.",
        "I fixed the script.",
    ]
    cursor = 0
    for part in ordered:
        found = text.find(part, cursor)
        assert found >= 0, f"{part!r} not in order in {text!r}"
        cursor = found + len(part)
    assert "Here is your page with a bouncing ball." not in text  # the claim made before the look was taken back
    assert any(name == "retract" and data.get("now") for name, data in events)
    seen = looks(env)
    assert len(seen) == 2 and "script error" in seen[0]["content"] and "No problems found" in seen[1]["content"]
    assert fake.looks == 1 and (env.workspace / "screenshots" / "look-index.png").is_file()
    assert (env.workspace / "index.html").read_text() == FIXED
    # every request kept the shape the model is trained on: no two replies of its own in a row
    last = env.fake.stream_calls[-1]["messages"]
    roles = [m["role"] for m in last if m["role"] != "system"]
    assert all(not (a == b == "assistant") for a, b in zip(roles, roles[1:]))
    assert prompts.LOOK_BAD.split("{")[0] in json.dumps(last)


def test_the_limit_ends_with_the_i_stopped_line(env: Env, chrome) -> None:
    chrome(script_error(2))
    env.fake.stream_script = [write("index.html", PAGE, "call_1"), ChatResult(content="Done.")]
    for number in range(2, 6):
        env.fake.stream_script += [write("index.html", PAGE + f"<!-- {number} -->", f"call_{number}"), ChatResult(content="Done again.")]
    events, _raw = chat_events(env, ASK)
    assert len(looks(env)) == pageloop.LOOKS == 3
    last = messages(env, "assistant")[-1]["content"]
    assert last.startswith("I stopped after 3 tries. Still wrong: a script error stops the page")
    assert shown(events).rstrip().endswith(last)
    assert len(env.fake.stream_script) == 4  # six model rounds ran; none was spent after the third look


def test_a_reply_that_changes_nothing_after_a_problem_ends_the_turn(env: Env, chrome) -> None:
    fake = chrome(script_error(2))
    env.fake.stream_script = [write("index.html", PAGE, "call_1"), ChatResult(content="Done."), ChatResult(content="It is fine.")]
    events, _raw = chat_events(env, ASK)
    assert fake.looks == 0 and len(looks(env)) == 1
    last = messages(env, "assistant")[-1]["content"]
    assert "I stopped after 1 try. Still wrong: a script error stops the page" in last and "It is fine." not in last
    assert "It is fine." not in shown(events)


def test_a_page_that_was_not_touched_in_the_turn_is_never_looked_at(env: Env, chrome) -> None:
    fake = chrome(script_error(2))
    (env.workspace / "old.html").write_text(PAGE, encoding="utf-8")
    env.fake.stream_script = [tool_call_result("read_file", {"path": "old.html"}), ChatResult(content="It calls spin().")]
    chat_events(env, "what does old.html do?")
    assert looks(env) == [] and fake.sizes == []


def test_a_page_written_twice_is_looked_at_once_after_the_last_write(env: Env, chrome) -> None:
    fake = chrome(CLEAN)
    env.fake.stream_script = [write("index.html", PAGE, "call_1"), write("index.html", FIXED, "call_2"), ChatResult(content="Done.")]
    chat_events(env, ASK)
    assert len(looks(env)) == 1 and fake.looks == 0


def test_a_script_written_after_the_page_makes_the_page_due_again(env: Env, chrome) -> None:
    chrome(CLEAN)
    page = '<!doctype html><html><body><canvas></canvas><script src="app.js"></script></body></html>\n'
    env.fake.stream_script = [
        write("index.html", page, "call_1"), ChatResult(content="Made it."),
        write("app.js", "console.log(1);\n", "call_2"), ChatResult(content="Added the script."),
    ]
    chat_events(env, ASK)
    seen = looks(env)
    assert len(seen) == 2 and "app.js" in seen[0]["content"] and "No problems found" in seen[1]["content"]


def test_a_finished_kit_that_passes_ends_with_good_now_then_the_kit_line(env: Env, chrome) -> None:
    chrome(CLEAN)
    use = {"action": "use", "name": "web/scene-3d", "path": "fox", "values": {"scene": "fox walks to campfire"}}
    env.fake.stream_script = [tool_call_result("component", use, "call_1")]
    events, _raw = chat_events(env, "make a 3d scene where a fox walks to a campfire")
    assert len(env.fake.stream_calls) == 1  # the model was asked once: the look and the last line are Kiwibro's
    last = messages(env, "assistant")[-1]
    assert last["content"] == "It is good now: it loads without errors, it draws and it moves.\n\n" + prompts.KIT_MADE.format(files="fox/index.html")
    assert last["meta"]["kit"] == "web/scene-3d" and last["meta"]["files"] == ["fox/index.html"]
    assert shown(events).rstrip().endswith(prompts.KIT_MADE.format(files="fox/index.html"))
    assert len(looks(env)) == 1


def test_a_kit_page_that_failed_its_look_is_made_again_by_the_kit(env: Env, chrome) -> None:
    chrome(script_error(40), CLEAN)
    use = {"action": "use", "name": "web/scene-3d", "path": "fox", "values": {"scene": "fox walks to campfire"}}
    again = {**use, "values": {"scene": "fox walks to a tree", "sky": "night"}}
    env.fake.stream_script = [tool_call_result("component", use, "call_1"), tool_call_result("component", again, "call_2")]
    chat_events(env, "make a 3d scene where a fox walks to a campfire")
    seen = looks(env)
    assert "script error" in seen[0]["content"] and "replace_in_file" not in seen[0]["content"]
    assert "Call component again with name web/scene-3d and path fox, and other values for the blanks." in seen[0]["content"]
    assert "The lines at the top of fox/index.html now read:" in seen[0]["content"] and "sky day" in seen[0]["content"]
    page = (env.workspace / "fox" / "index.html").read_text()
    assert "fox walks to a tree" in page and "sky night" in page and "fox walks to campfire" not in page  # written again over the first page
    assert not any(m["content"].startswith("Error:") for m in messages(env, "tool"))
    said = [m["content"] for m in messages(env, "assistant")]
    assert prompts.LOOK_AGAIN in " ".join(said) and len(seen) == 2 and "No problems found" in seen[1]["content"]
    assert said[-1].endswith(prompts.KIT_MADE.format(files="fox/index.html"))


def test_a_page_in_a_turn_with_a_plan_is_looked_at_too(env: Env, chrome) -> None:
    chrome(script_error(2), CLEAN)
    env.fake.stream_script = [
        write("box.html", PAGE, "call_1"), ChatResult(content="Saved box.html."),
        write("box.html", FIXED, "call_2"), ChatResult(content="Fixed."),
    ]
    events, _raw = chat_events(env, "write a file box.html yourself with write_file: a page where a red square moves from left to right forever")
    assert [e for e in events if e[0] == "plan"], "the message made a plan"
    seen = looks(env)
    assert len(seen) == 2 and "script error" in seen[0]["content"] and "No problems found" in seen[1]["content"]
    assert "It is good now" in messages(env, "assistant")[-1]["content"]


def test_a_kit_page_that_passed_or_another_file_is_never_written_over(env: Env, chrome) -> None:
    chrome(CLEAN)
    use = {"action": "use", "name": "web/scene-3d", "path": "fox", "values": {}}
    env.fake.stream_script = [
        write("fox/index.html", PAGE, "call_1"),  # a page the model wrote itself, in the folder the kit wants
        tool_call_result("component", use, "call_2"),
        ChatResult(content="Made it."),
    ]
    chat_events(env, "make a 3d scene where a fox walks to a campfire")
    refused = [m["content"] for m in messages(env, "tool") if m["tool_name"] == "component"]
    assert refused and refused[0].startswith("Error: fox/index.html already exists")
    assert "spin()" in (env.workspace / "fox" / "index.html").read_text()


def test_a_browser_that_is_missing_is_said_and_the_page_is_left_alone(env: Env, chrome) -> None:
    chrome(error="Chrome did not start")
    env.fake.stream_script = [write("index.html", FIXED, "call_1"), ChatResult(content="Here it is.")]
    events, _raw = chat_events(env, ASK)
    last = messages(env, "assistant")[-1]["content"]
    assert last == prompts.LOOK_BLIND.format(file="index.html", why="Chrome did not start") + "\n\nHere it is."
    assert len(env.fake.stream_script) == 0 and "It is good now" not in shown(events)


# --- /goal -----------------------------------------------------------------------------------------


def test_a_goal_is_stored_without_its_command_and_marked(env: Env, chrome) -> None:
    chrome(CLEAN)
    env.fake.stream_script = [write("index.html", FIXED, "call_1"), ChatResult(content="Made it.")]
    chat_events(env, "/goal a page with a bouncing ball")
    user = messages(env, "user")[0]
    assert user["content"] == "a page with a bouncing ball" and user["meta"]["goal"] is True
    assert "a page with a bouncing ball" in json.dumps(env.fake.stream_calls[0]["messages"]) and "/goal" not in json.dumps(env.fake.stream_calls[0]["messages"])


@pytest.mark.parametrize("message", ["/goal", "/GOAL  ", " /goal"])
def test_a_bare_goal_is_answered_without_asking_a_model(env: Env, message: str) -> None:
    events, _raw = chat_events(env, message)
    assert env.fake.stream_calls == []
    assert messages(env, "assistant")[-1]["content"] == prompts.GOAL_USAGE
    assert shown(events) == prompts.GOAL_USAGE and events[-1][0] == "done"


def test_a_goal_looks_at_phone_width_and_up_to_eight_times(env: Env, chrome) -> None:
    fake = chrome(script_error(2))
    env.fake.stream_script = [write("index.html", PAGE, "call_1"), ChatResult(content="Done.")]
    for number in range(2, 12):
        env.fake.stream_script += [write("index.html", PAGE + f"<!-- {number} -->", f"call_{number}"), ChatResult(content="Done again.")]
    events, _raw = chat_events(env, "/goal a page with a spinning 3d cube that I can turn with the mouse")
    assert len(looks(env)) == pageloop.GOAL_LOOKS == 8
    assert pagecheck.PHONE[0] in fake.widths
    texts = [m["content"] for m in messages(env, "assistant")]
    assert any(t.startswith("I stopped after 8 tries.") for t in texts)
    assert "I stopped after 8 tries." in texts[-1]
    assert not any("Still failing" in t for t in texts)  # the line that stopped already said what is wrong


def test_a_goal_page_that_does_not_look_like_the_goal_is_not_good(env: Env, chrome, monkeypatch: pytest.MonkeyPatch) -> None:
    chrome(CLEAN)
    answers = iter(["No. It shows a blue square.", "Yes. A cube on a dark background."])
    asked: list[str] = []

    def describe(llm, workspace, path, question="", **kwargs):
        asked.append(question)
        return {"text": next(answers), "path": path, "question": question}

    monkeypatch.setattr(delegate, "describe_image", describe)
    env.fake.stream_script = [
        write("index.html", FIXED, "call_1"), ChatResult(content="Made it."),
        write("index.html", FIXED + "<!-- cube -->", "call_2"), ChatResult(content="Made a cube."),
    ]
    events, _raw = chat_events(env, "/goal a page with a spinning 3d cube")
    assert "a page with a spinning 3d cube" in asked[0]
    seen = looks(env)
    assert "It does not look like what was asked: It shows a blue square." in seen[0]["content"]
    assert "No problems found" in seen[1]["content"] and "it looks like what was asked" in seen[1]["content"]
    last = [m["content"] for m in messages(env, "assistant")]
    assert last[-1].startswith("It is good now: it loads without errors")
    assert not any(t.startswith("Checked ") for t in last)  # "It is good now" was said: nothing is left to report


def test_a_goal_line_reports_only_what_no_earlier_line_said(env: Env, chrome) -> None:
    chrome(CLEAN)
    env.fake.stream_script = [
        write("index.html", FIXED, "call_1"), write("words.py", "def f(x):\n    return x\n", "call_2"),
        ChatResult(content="Made both."), ChatResult(content="Fine."), ChatResult(content="DONE"),
    ]
    chat_events(env, "/goal a page with a spinning cube")
    said = [m["content"] for m in messages(env, "assistant")]
    assert any(t.startswith("It is good now") for t in said)
    assert said[-1] == prompts.GOAL_CHECKED.format(files="words.py")  # the page was already reported


def test_a_goal_without_a_page_is_checked_against_the_goal_and_its_files_are_parsed(env: Env, chrome) -> None:
    env.fake.stream_script = [
        write("words.py", "def f(x):\n    return x\n", "call_1"), ChatResult(content="Saved words.py."),
        ChatResult(content="Everything is in."), ChatResult(content="DONE"),
    ]
    events, _raw = chat_events(env, "/goal a python file words.py with a function f")
    review = prompts.GOAL_REVIEW.format(goal="a python file words.py with a function f")
    sent = [review in json.dumps(call["messages"][-1]["content"]) for call in env.fake.stream_calls]
    assert sent == [False, False, True, True]  # asked after each reply that is not DONE, and never kept in the chat
    assert review not in json.dumps([m["content"] for m in messages(env)])
    assert len(env.fake.stream_calls) == 4
    assert messages(env, "assistant")[-1]["content"] == prompts.GOAL_CHECKED.format(files="words.py")
    assert "DONE" not in shown(events)


def test_a_goal_review_is_sent_three_times_at_most(env: Env, chrome) -> None:
    env.fake.stream_script = [write("words.py", "x = 1\n", "call_1")] + [ChatResult(content=f"Still working {n}.") for n in range(6)]
    chat_events(env, "/goal a python file words.py")
    assert len(env.fake.stream_calls) == 5  # the write, the reply, and three reviews


def test_a_goal_that_made_nothing_checkable_says_so(env: Env, chrome) -> None:
    env.fake.stream_script = [tool_call_result("list_files", {"path": "."}), ChatResult(content="Nothing is in the folder.")]
    chat_events(env, "/goal tell me what is in this folder")
    assert messages(env, "assistant")[-1]["content"] == prompts.GOAL_UNCHECKED
