"""The note tool: save, append, read and search; private chats; notes in the turn context."""

from __future__ import annotations

import pytest
from test_backend import Env, chat_events, tool_call_result

from kiwibro import config, guidance, notes, prompts, tools
from kiwibro.llm import ChatResult
from kiwibro.sandbox import Workspace


def _run(env: Env, args: dict, conversation_id: str = "c_test", notes=True):
    context = tools.ToolContext(
        workspace=Workspace(env.workspace),
        memory=env.svc.memory,
        search=env.svc.search,
        conversation_id=conversation_id,
        notes=env.svc.notes if notes else None,
    )
    return tools.execute_tool("note", tools.validate_args("note", args), context)


def _private_chat(env: Env) -> str:
    response = env.client.post("/api/conversations", json={"title": "Secret", "private": True})
    assert response.status_code == 201
    return response.get_json()["id"]


def test_save_then_read_returns_the_body(env: Env):
    saved = _run(env, {"action": "save", "title": "Wi-Fi", "content": "The password is on the fridge door"})
    assert saved.ok is True and saved.content == "Saved note “Wi-Fi”."
    read = _run(env, {"action": "read", "title": "Wi-Fi"})
    assert read.ok is True
    assert read.content == "# Wi-Fi\nThe password is on the fridge door"


def test_append_adds_text_and_a_missing_note_fails(env: Env):
    _run(env, {"action": "save", "title": "Wi-Fi", "content": "The password is on the fridge door"})
    added = _run(env, {"action": "append", "title": "Wi-Fi", "content": "The router is in the hall."})
    assert added.ok is True and added.content == "Added to note “Wi-Fi”."
    assert "The router is in the hall." in _run(env, {"action": "read", "title": "Wi-Fi"}).content

    missing = _run(env, {"action": "append", "title": "Groceries", "content": "milk"})
    assert missing.ok is False
    assert missing.content == "No note called “Groceries”. Use action save to create it."
    assert env.svc.db.count_notes() == 1


def test_read_of_a_missing_note_suggests_the_closest_titles(env: Env):
    _run(env, {"action": "save", "title": "Trip to Izmir", "content": "Ferry times and a hotel"})
    assert _run(env, {"action": "read", "title": "Izmir"}).ok is True  # find() matches the shared word
    _run(env, {"action": "save", "title": "Holiday", "content": "Ferry to Bursa on Friday"})
    missed = _run(env, {"action": "read", "title": "Bursa"})  # only the body mentions it
    assert missed.ok is False
    assert missed.content == "No note called “Bursa”.\nClosest notes: Holiday"


def test_search_lists_matching_titles_or_says_none_match(env: Env):
    _run(env, {"action": "save", "title": "Trip to Izmir", "content": "Ferry times\nand a hotel"})
    _run(env, {"action": "save", "title": "Recipes", "content": "Lentil soup"})
    found = _run(env, {"action": "search", "content": "izmir"})
    assert found.ok is True and found.content == "- Trip to Izmir: Ferry times"
    none = _run(env, {"action": "search", "content": "kayak"})
    assert none.ok is True and none.content == "No notes match."


def test_unknown_action_fails_and_lists_the_four_actions(env: Env):
    result = _run(env, {"action": "delete", "title": "Wi-Fi"})
    assert result.ok is False
    assert result.content == "Unknown action “delete”. Use one of: save, append, read, search."


def test_note_tool_without_a_notebook_says_so(env: Env):
    result = _run(env, {"action": "save", "content": "milk"}, notes=False)
    assert result.ok is False and result.content == "Notes are not available here."


def test_save_is_refused_in_a_private_chat(env: Env):
    cid = _private_chat(env)
    env.fake.stream_script = [
        tool_call_result("note", {"action": "save", "title": "Code", "content": "The gate code is 4321"}),
        ChatResult(content="Ok."),
    ]
    events, _raw = chat_events(env, "take a note that the gate code is 4321", conversation_id=cid)
    result = next(payload for event, payload in events if event == "tool_result")
    assert result["ok"] is False
    assert result["preview"] == "Private chat: not saved"
    assert env.svc.db.count_notes() == 0


def test_a_chat_turn_that_calls_note_stores_the_note(env: Env):
    env.fake.stream_script = [
        tool_call_result("note", {"action": "save", "title": "Wi-Fi", "content": "The password is on the fridge"}),
        ChatResult(content="Noted."),
    ]
    chat_events(env, "take a note that the wifi password is on the fridge")
    (hit,) = env.svc.notes.search("fridge")
    assert hit["title"] == "Wi-Fi" and hit["body"] == "The password is on the fridge"


def test_a_note_saved_from_the_chat_says_who_saved_it(env: Env):
    _run(env, {"action": "save", "title": "Wi-Fi", "content": "The password is on the fridge door"})
    [note] = env.svc.db.list_notes()
    assert note["origin"] == "kiwibro" and note["source_conversation_id"] == "c_test"


def test_a_dictated_note_is_saved_even_when_the_model_only_talks(env: Env):
    env.fake.stream_script = [ChatResult(content="I'll save the note for you."),  # says so, calls nothing
                              ChatResult(content="Saved your note.")]
    events, _raw = chat_events(env, "save a note: buy oat milk and call the dentist on Friday")
    [note] = env.svc.db.list_notes()
    assert note["body"] == "buy oat milk and call the dentist on Friday"
    assert len(env.fake.stream_calls) == 1  # Kiwibro kept the note, so it says so itself: the model is not asked again
    conversation_id = events[0][1]["conversation_id"]
    shown = [m["content"] for m in env.svc.db.list_messages(conversation_id) if m["role"] == "assistant"]
    assert shown == ["", f"Saved note “{note['title']}”."]  # the empty promise is gone; the first message carries the call
    assert [payload for event, payload in events if event == "done"][0]["inference"]["acted_instead"] == "note"


def test_save_that_as_a_note_keeps_the_last_reply(env: Env):
    env.fake.stream_script = [ChatResult(content="Rowi are the rarest kiwi: about 600 birds, all near Ōkārito.")]
    events, _raw = chat_events(env, "tell me about the rarest kiwi")
    conversation_id = events[0][1]["conversation_id"]
    env.fake.stream_script = [ChatResult(content="Sure, noted!"), ChatResult(content="It's in your notes.")]
    chat_events(env, "save that as a note", conversation_id=conversation_id)
    [note] = env.svc.db.list_notes()
    assert note["body"] == "Rowi are the rarest kiwi: about 600 birds, all near Ōkārito."


def test_a_model_that_saves_the_note_itself_is_left_alone(env: Env):
    env.fake.stream_script = [tool_call_result("note", {"action": "save", "title": "Shopping", "content": "oat milk"}),
                              ChatResult(content="Saved.")]
    events, _raw = chat_events(env, "save a note: oat milk and bread")
    assert [note["title"] for note in env.svc.db.list_notes()] == ["Shopping"]
    assert "acted_instead" not in ([payload for event, payload in events if event == "done"][0]["inference"] or {})


def test_only_a_note_given_word_for_word_is_saved_as_typed():
    from kiwibro import notes

    assert notes.dictated("note: wifi password is kiwi-1234") == "wifi password is kiwi-1234"
    assert notes.dictated("take a note that the car service is due in March") == "the car service is due in March"
    assert notes.dictated('Could you save this note: "Meeting moved to 3pm"') == "Meeting moved to 3pm"
    assert notes.dictated("add this to my notes please") == "" and notes.dictated("note that down") == ""
    for composed in ("make a note about the numbers above", "write down the steps for making bread",
                     "what's in my notes?", "save that", "notes on quantum computing please"):
        assert notes.dictated(composed) is None


@pytest.mark.parametrize(("message", "body"), [
    ("note: buy milk tomorrow", "buy milk tomorrow"),
    ("take a note: meeting with Ali at 3pm", "meeting with Ali at 3pm"),
    ("write this down: call the bank on monday", "call the bank on monday"),
    ("save this as a note", ""),
    ("add to my notes that the server rotates keys monthly", "the server rotates keys monthly"),
    ("jot down call mom", "call mom"),
    ("keep a note of this: the api limit is 60 per minute", "the api limit is 60 per minute"),
    ("note down the gate code 4471", "the gate code 4471"),
    ("jot this down: dentist on friday", "dentist on friday"),
    ("put in my notes: renew passport", "renew passport"),
    ("add to notes: renew passport", "renew passport"),
    ("save to my notes that the router is in the attic", "the router is in the attic"),
    ("new note: idea for a kiwi logo", "idea for a kiwi logo"),
    ("remember this as a note: oil change at 90k", "oil change at 90k"),
    ("can you note down that the wifi is on channel 6?", "the wifi is on channel 6"),
    ("create a note called groceries with eggs, milk and bread", "eggs, milk and bread"),
    ("take a note called Ideas: kiwi logo, dark mode", "kiwi logo, dark mode"),
    ("make a note titled Trip with flights on the 12th", "flights on the 12th"),
])
def test_dictated_notes_keep_the_words_the_user_said(message, body):
    from kiwibro import notes

    assert notes.dictated(message) == body


@pytest.mark.parametrize("message", [
    "make a note about quantum computing",
    "write a note explaining how DNS works",
    "write down the steps to install docker",
    "what notes do I have?",
    "read my note about groceries",
    "search my notes for wifi",
    "I noted that the build failed",
    "note how the function handles errors",
    "create a note called groceries",
    "note: groceries",
    "note-taking apps are useful",
])
def test_a_note_the_model_writes_or_no_note_is_not_dictated(message):
    from kiwibro import notes

    assert notes.dictated(message) is None
    assert notes.dictation(message) is None


@pytest.mark.parametrize(("message", "found"), [
    ("create a note called groceries with eggs, milk and bread", ("groceries", "eggs, milk and bread")),
    ("make a note titled Trip with flights on the 12th", ("Trip", "flights on the 12th")),
    ('make a note called "Gate codes": 4471 north', ("Gate codes", "4471 north")),
    ("take a note called Ideas: kiwi logo, dark mode", ("Ideas", "kiwi logo, dark mode")),
    ("note: buy milk tomorrow", ("", "buy milk tomorrow")),
    ("note - buy milk tomorrow", ("", "buy milk tomorrow")),
    ("write it down: the wifi code is 99", ("", "the wifi code is 99")),
    ("save this as a note", ("", "")),
])
def test_dictation_gives_the_title_and_the_body(message, found):
    from kiwibro import notes

    assert notes.dictation(message) == found


def test_a_long_title_is_cut_to_the_title_limit():
    from kiwibro import notes

    title, body = notes.dictation("create a note called " + "t" * 300 + " with eggs")
    assert len(title) == notes.MAX_TITLE and body == "eggs"


def test_a_relevant_note_rides_on_the_newest_message_only(env: Env):
    env.svc.notes.save("Ferry to Bursa on Friday, hotel booked.", "Trip to Izmir")
    env.fake.stream_script = [ChatResult(content="Hello!")]
    _events, _raw = chat_events(env, "hi")
    conversation_id = env.svc.db.list_conversations()[0]["id"]
    assert "Trip to Izmir" not in env.fake.stream_calls[-1]["messages"][-1]["content"]  # small talk: none

    env.fake.stream_script = [ChatResult(content="Pack light.")]
    chat_events(env, "what should I pack for my trip to Izmir", conversation_id=conversation_id)
    messages = env.fake.stream_calls[-1]["messages"]
    assert "Trip to Izmir" in messages[-1]["content"]
    assert "Ferry to Bursa" in messages[-1]["content"]
    assert "Trip to Izmir" not in messages[0]["content"]  # never in message 0 (KV-cache prefix)


def test_guidance_asks_for_the_note_tool_when_notes_are_mentioned():
    assert "notes" in guidance.select("take a note that the wifi password is on the fridge")
    assert "notes" in guidance.select("show me my notes from the trip")
    assert "notes" not in guidance.select("what is the average of 3, 9 and 12")
    assert "notes" not in guidance.select("take a note", tools=False)


def test_note_is_auto_by_default_and_offered_in_the_full_profile(env: Env):
    assert config.DEFAULT_PERMISSIONS["note"] == "auto"
    assert tools.get_spec("note").risk == tools.get_spec("remember").risk
    assert "note" in tools.select_tool_names("auto")
    assert any(item["function"]["name"] == "note" for item in tools.tool_defs_for("full"))


def test_save_that_with_nothing_said_before_says_so_instead_of_pretending(env: Env):
    env.fake.stream_script = [ChatResult(content="I've saved the note in your notes.")]
    events, _raw = chat_events(env, "save that as a note")
    assert env.svc.db.list_notes() == []
    conversation_id = events[0][1]["conversation_id"]
    shown = [m["content"] for m in env.svc.db.list_messages(conversation_id) if m["role"] == "assistant"]
    assert shown == ["There is nothing earlier in this chat to save yet. Tell me what the note should say and I'll keep it."]


def test_a_dictated_note_is_saved_when_the_model_reaches_for_the_wrong_tool(env: Env):
    env.fake.stream_script = [tool_call_result("remember", {"content": "renew passport before March", "kind": "preference"}),
                              ChatResult(content="Your note has been saved successfully."),
                              ChatResult(content="It is in your notes now.")]
    events, _raw = chat_events(env, "save a note: renew passport before March")
    assert [note["body"] for note in env.svc.db.list_notes()] == ["renew passport before March"]
    assert [payload for event, payload in events if event == "done"][0]["inference"]["acted_instead"] == "note"
    conversation_id = events[0][1]["conversation_id"]
    shown = [m["content"] for m in env.svc.db.list_messages(conversation_id) if m["role"] == "assistant" and m["content"]]
    assert shown == [f"Saved note “{env.svc.db.list_notes()[0]['title']}”."]  # said by Kiwibro, in the tool's own words


@pytest.mark.parametrize(("message", "topic"), [
    ("make a note about the meeting tomorrow at 10 with the design team", "the meeting tomorrow at 10 with the design team"),
    ("write a note explaining how DNS works", "how DNS works"),
    ("summarize the article above and save it as a note", "summarize the article above"),
    ("can you make me a short note on python decorators?", "python decorators"),
])
def test_a_note_to_be_written_is_recognised(message: str, topic: str):
    assert notes.written_request(message) == topic


@pytest.mark.parametrize("message", ["note: buy milk", "write a story about a cat", "save this as a note", "what notes do I have?"])
def test_other_messages_ask_for_no_written_note(message: str):
    assert notes.written_request(message) is None


def test_a_note_the_model_wrote_but_never_saved_is_saved_for_it(env: Env):
    text = "Decorators wrap a function: @cache memoises results, @wraps keeps the name and docstring."
    env.fake.stream_script = [ChatResult(content=text), ChatResult(content="Saved it to your notes.")]
    events, _raw = chat_events(env, "make a note about python decorators")
    [note] = env.svc.db.list_notes()
    assert note["title"] == "Python decorators" and note["body"] == text
    assert len(env.fake.stream_calls) == 1
    conversation_id = events[0][1]["conversation_id"]
    shown = [m["content"] for m in env.svc.db.list_messages(conversation_id) if m["role"] == "assistant"]
    assert shown == [text, "Saved note “Python decorators”."]  # the note stays in the chat as well


def test_a_titled_dictated_note_keeps_its_title(env: Env):
    env.fake.stream_script = [ChatResult(content="Okay!"), ChatResult(content="Saved.")]
    chat_events(env, "create a note called groceries with eggs, milk and bread")
    [note] = env.svc.db.list_notes()
    assert note["title"] == "groceries" and note["body"] == "eggs, milk and bread"


def test_a_question_about_the_users_own_things_is_shown_the_note_this_chat_kept(env: Env):
    env.fake.stream_script = [tool_call_result("note", {"action": "save", "title": "Groceries", "content": "milk, eggs, bread"}),
                              ChatResult(content="Saved."), ChatResult(content="Milk, eggs and bread.")]
    events, _raw = chat_events(env, "note titled Groceries: milk, eggs, bread")
    conversation_id = events[0][1]["conversation_id"]
    chat_events(env, "what did i ask you to buy", conversation_id)
    assert "milk, eggs, bread" in env.fake.stream_calls[-1]["messages"][-1]["content"]  # no word shared, still shown
