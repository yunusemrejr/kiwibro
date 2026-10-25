"""Private chats: nothing from them is remembered, and other chats never see them."""

from __future__ import annotations

from test_backend import Env, chat_events, tool_call_result

from kiwibro import sessions
from kiwibro.llm import ChatResult


def _private_chat(env: Env, title: str = "Secret") -> str:
    response = env.client.post("/api/conversations", json={"title": title, "private": True})
    assert response.status_code == 201
    assert response.get_json()["private"] is True
    return response.get_json()["id"]


def test_a_new_chat_can_start_private_and_nothing_is_captured(env: Env):
    response = env.client.post("/api/chat", json={"message": "My name is Ali and I live in Izmir.", "private": True})
    events = [line for line in response.get_data(as_text=True).splitlines() if line.startswith("event:")]
    assert "event: done" in events
    (conversation,) = env.svc.db.list_conversations()
    assert conversation["private"] is True
    assert env.svc.db.list_memories() == []  # code capture skipped
    listed = env.client.get("/api/conversations").get_json()["conversations"]
    assert listed[0]["private"] is True


def test_the_same_message_in_a_normal_chat_is_captured(env: Env):
    chat_events(env, "My name is Ali and I live in Izmir.")
    assert env.svc.db.list_memories()  # the control case: capture works outside private chats


def test_remember_tool_refuses_in_a_private_chat(env: Env):
    cid = _private_chat(env)
    env.fake.stream_script = [tool_call_result("remember", {"content": "User lives in Izmir"}), ChatResult(content="Ok.")]
    events, _raw = chat_events(env, "i live in izmir", conversation_id=cid)
    result = next(payload for event, payload in events if event == "tool_result")
    assert result["ok"] is False and "private" in result["preview"].lower()
    assert env.svc.db.list_memories() == []


def test_extraction_never_reads_a_private_chat_and_moves_its_watermark(env: Env):
    cid = _private_chat(env)
    env.svc.db.add_message(cid, "user", "I am allergic to peanuts")
    env.svc.db.add_message(cid, "assistant", "Noted.")
    # What the model would return if it were asked: proves it never is.
    env.fake.memory_ops = {"ops": [{"op": "add", "kind": "fact", "content": "User is allergic to peanuts"}]}
    counts = env.svc.memory.extract_and_apply(cid, "fake-model")
    assert counts == {"added": 0, "updated": 0, "removed": 0}
    assert env.svc.db.list_memories() == []
    assert env.svc.db.get_conversation(cid)["extracted_through_seq"] == 2
    # Made public later, the private part is still never learned from.
    env.client.patch(f"/api/conversations/{cid}", json={"private": False})
    assert env.svc.memory._pending_window(cid)[0] == []


def test_private_chats_stay_out_of_other_chats(env: Env):
    secret = _private_chat(env, "Surprise party")
    env.svc.db.add_message(secret, "user", "plan the surprise party budget for Elif")
    env.svc.db.add_message(secret, "assistant", "The party budget is 400 euros.")
    public = env.svc.db.create_conversation(title="Trip")["id"]
    env.svc.db.add_message(public, "user", "plan the Lisbon trip budget")
    env.svc.db.add_message(public, "assistant", "The trip budget is 900 euros.")
    current = env.svc.db.create_conversation()["id"]
    note = sessions.recent_note(env.svc.db, current, "what budget did we decide yesterday?")
    assert "Trip" in note and "900 euros" in note
    assert "Surprise party" not in note and "400 euros" not in note
    items = env.svc.memory.recall("party budget", conversation_id=current)
    assert all(item.get("conversation_id") != secret for item in items)


def test_privacy_is_patched_and_validated(env: Env):
    cid = env.svc.db.create_conversation(title="Chat")["id"]
    bad = env.client.patch(f"/api/conversations/{cid}", json={"private": "yes"})
    assert bad.status_code == 400
    ok = env.client.patch(f"/api/conversations/{cid}", json={"private": True}).get_json()
    assert ok["conversation"]["private"] is True
    assert env.client.get(f"/api/conversations/{cid}").get_json()["conversation"]["private"] is True
    assert env.client.post("/api/conversations", json={"private": 1}).status_code == 400
    assert env.client.post("/api/chat", json={"message": "hi", "private": "no"}).status_code == 400


def test_a_fork_of_a_private_chat_stays_private(env: Env):
    cid = _private_chat(env)
    env.svc.db.add_message(cid, "user", "secret plans")
    fork = env.client.post(f"/api/conversations/{cid}/fork", json={}).get_json()["conversation"]
    assert fork["private"] is True
