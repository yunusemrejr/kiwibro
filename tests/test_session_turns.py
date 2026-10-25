"""Agent turn behaviours: retrying an unanswered message, re-titling chats, earlier-chats notes.

Every turn runs for real through POST /api/chat against the in-process fake LM Studio
from test_backend (no network, temporary data dir).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from test_backend import Env, chat_events

from kiwibro.agent import is_small_talk_title
from kiwibro.llm import ChatResult, LMStudioError


def _roles(env: Env, conversation_id: str) -> list[str]:
    return [m["role"] for m in env.svc.db.list_messages(conversation_id)]


def _user_texts(env: Env, conversation_id: str) -> list[str]:
    return [m["content"] for m in env.svc.db.list_messages(conversation_id) if m["role"] == "user"]


def _only_chat_id(env: Env) -> str:
    conversations = env.svc.db.list_conversations()
    assert len(conversations) == 1, [c["id"] for c in conversations]
    return conversations[0]["id"]


def _title(env: Env, conversation_id: str) -> str:
    return env.svc.db.get_conversation(conversation_id)["title"]


def _request_text(env: Env, index: int) -> str:
    """Everything the fake model received for its ``index``-th streamed request, as one string."""
    return json.dumps(env.fake.stream_calls[index]["messages"], ensure_ascii=False)


# ---------------------------------------------------------------------------
# 1. A repeated message after an unanswered one is not stored twice
# ---------------------------------------------------------------------------


def test_retry_after_a_failed_turn_reuses_the_stored_message(env: Env) -> None:
    env.fake.stream_script = [LMStudioError("stream", "The connection to LM Studio was interrupted.")]
    events, _raw = chat_events(env, "find PHP news")
    assert events[-1][0] == "error"
    cid = _only_chat_id(env)
    assert _roles(env, cid) == ["user"]  # the failed turn left only the question behind

    env.fake.stream_script = [ChatResult(content="Here is the PHP news.")]
    events, _raw = chat_events(env, "find PHP news", conversation_id=cid)

    assert events[-1][0] == "done"
    assert _user_texts(env, cid) == ["find PHP news"]
    assert _roles(env, cid) == ["user", "assistant"]
    assert env.svc.db.list_messages(cid)[-1]["content"] == "Here is the PHP news."
    retry_request = _request_text(env, 1)
    assert retry_request.count("find PHP news") == 1  # the model saw the question once


def test_resending_an_unanswered_message_stores_it_once(env: Env) -> None:
    cid = env.svc.db.create_conversation()["id"]
    env.svc.db.add_message(cid, "user", "find PHP news")

    events, _raw = chat_events(env, "find PHP news", conversation_id=cid)

    assert events[-1][0] == "done"
    assert _user_texts(env, cid) == ["find PHP news"]
    assert _roles(env, cid) == ["user", "assistant"]


def test_whitespace_differences_still_count_as_the_same_message(env: Env) -> None:
    cid = env.svc.db.create_conversation()["id"]
    env.svc.db.add_message(cid, "user", "find PHP news")

    events, _raw = chat_events(env, "find   PHP\n news ", conversation_id=cid)

    assert events[-1][0] == "done"
    assert _user_texts(env, cid) == ["find PHP news"]


def test_repeating_an_answered_message_stores_a_second_one(env: Env) -> None:
    env.fake.stream_script = [ChatResult(content="First answer."), ChatResult(content="Second answer.")]
    chat_events(env, "find PHP news")
    cid = _only_chat_id(env)

    events, _raw = chat_events(env, "find PHP news", conversation_id=cid)

    assert events[-1][0] == "done"
    assert _user_texts(env, cid) == ["find PHP news", "find PHP news"]
    assert _roles(env, cid) == ["user", "assistant", "user", "assistant"]


def test_different_text_after_an_unanswered_message_is_stored(env: Env) -> None:
    cid = env.svc.db.create_conversation()["id"]
    env.svc.db.add_message(cid, "user", "find PHP news")

    events, _raw = chat_events(env, "find Python news", conversation_id=cid)

    assert events[-1][0] == "done"
    assert _user_texts(env, cid) == ["find PHP news", "find Python news"]
    assert _roles(env, cid) == ["user", "user", "assistant"]


# ---------------------------------------------------------------------------
# 2. Re-titling away from small-talk titles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("title", ["Hi", "How's it going", "Good morning"])
def test_small_talk_titles_are_recognised(title: str) -> None:
    assert is_small_talk_title(title)


@pytest.mark.parametrize("title", ["PHP zerodays", "Write a small Python script"])
def test_topic_titles_are_not_small_talk(title: str) -> None:
    assert not is_small_talk_title(title)


def test_chat_opened_with_hi_is_titled_hi(env: Env) -> None:
    env.fake.stream_script = [ChatResult(content="Hello!")]
    chat_events(env, "hi")
    assert _title(env, _only_chat_id(env)) == "Hi"


def test_short_follow_up_does_not_retitle_a_small_talk_chat(env: Env) -> None:
    env.fake.stream_script = [ChatResult(content="Hello!"), ChatResult(content="Sure.")]
    chat_events(env, "hi")
    cid = _only_chat_id(env)

    chat_events(env, "search better.", conversation_id=cid)

    assert _title(env, cid) == "Hi"


def test_first_real_request_replaces_a_small_talk_title(env: Env) -> None:
    env.fake.stream_script = [ChatResult(content="Hello!"), ChatResult(content="Here is what I found.")]
    chat_events(env, "hi")
    cid = _only_chat_id(env)

    events, _raw = chat_events(
        env,
        "can search the web to find what happened yesterday about PHP zerodays",
        conversation_id=cid,
    )

    assert _title(env, cid) == "PHP zerodays"
    title_events = [payload for name, payload in events if name == "title"]
    assert title_events and title_events[-1]["title"] == "PHP zerodays"


def test_a_topic_title_is_never_replaced_by_later_messages(env: Env) -> None:
    env.fake.stream_script = [ChatResult(content="A script."), ChatResult(content="Sure.")]
    chat_events(env, "Write a small Python script")
    cid = _only_chat_id(env)
    assert _title(env, cid) == "Write a small Python script"

    chat_events(env, "find PHP zerodays news this week", conversation_id=cid)

    assert _title(env, cid) == "Write a small Python script"


@pytest.mark.parametrize("user_title", ["Mine", "Hi"])
def test_a_title_the_user_set_is_never_replaced(env: Env, user_title: str) -> None:
    env.fake.stream_script = [ChatResult(content="Hello!"), ChatResult(content="Here is what I found.")]
    cid = env.svc.db.create_conversation()["id"]
    patched = env.client.patch(f"/api/conversations/{cid}", json={"title": user_title})
    assert patched.status_code == 200
    assert env.svc.db.get_conversation(cid)["title_auto"] in (0, False)

    chat_events(env, "hi", conversation_id=cid)
    chat_events(env, "can search the web to find what happened yesterday about PHP zerodays", conversation_id=cid)

    assert _title(env, cid) == user_title


# ---------------------------------------------------------------------------
# 3. Earlier-chats note in the turn context
# ---------------------------------------------------------------------------


def _lisbon_chat(env: Env) -> tuple[str, str]:
    """Chat A: a planning exchange that a later chat can refer back to. Returns (id, title)."""
    env.fake.stream_script = [ChatResult(content="Budget is 900 euros for flights and hotel.")]
    chat_events(env, "let's plan the Lisbon trip budget")
    cid = _only_chat_id(env)
    return cid, _title(env, cid)


def test_question_about_an_earlier_chat_gets_the_recent_chats_note(env: Env) -> None:
    chat_a, title_a = _lisbon_chat(env)
    env.fake.stream_script = [ChatResult(content="You settled on 900 euros.")]
    before = len(env.fake.stream_calls)

    chat_events(env, "what budget did we decide yesterday?")  # a new chat, chat B

    chat_b = next(c["id"] for c in env.svc.db.list_conversations() if c["id"] != chat_a)
    assert chat_b != chat_a
    request = _request_text(env, before)
    assert "Your other recent chats" in request
    assert title_a in request
    assert "900 euros" in request


def test_an_unrelated_question_gets_no_earlier_chats_note(env: Env) -> None:
    _lisbon_chat(env)
    env.fake.stream_script = [ChatResult(content="4.")]
    before = len(env.fake.stream_calls)

    chat_events(env, "what is 2+2")  # a new chat, chat C

    assert len(env.svc.db.list_conversations()) == 2
    assert "Your other recent chats" not in _request_text(env, before)
