"""Automatic permissions, model-edited memory and the warm-up prefix contract."""

from __future__ import annotations

from test_backend import Env, SSEStream, chat_events, tool_call_result

from kiwibro import tools
from kiwibro.llm import ChatResult


# --- warm-up prefix == the next turn's prefix ---------------------------------------


def _assert_prefix(warm, sent) -> None:
    messages, tool_list, extra = warm
    # The warm-up ends on a placeholder user message; everything before it must be
    # exactly what the turn sends, or LM Studio cannot reuse the cache.
    assert sent["messages"][: len(messages) - 1] == messages[:-1]
    assert sent["messages"][len(messages) - 1]["role"] == "user"
    assert sent["tools"] == tool_list
    assert (sent["extra"] or {}) == (extra or {})


def test_warm_prefix_matches_a_new_chats_first_request(env: Env):
    warm = env.svc.agent.warm_prefix("fake-1")
    chat_events(env, "hello there")
    _assert_prefix(warm, env.fake.stream_calls[0])


def test_warm_prefix_matches_the_next_turn_of_an_open_chat(env: Env):
    env.fake.stream_script = [
        tool_call_result("list_files", {"path": "."}),
        ChatResult(content="The folder is empty."),
    ]
    events, _raw = chat_events(env, "what is in my folder?")
    conversation_id = next(payload for event, payload in events if event == "meta")["conversation_id"]
    warm = env.svc.agent.warm_prefix("fake-1", conversation_id)
    env.fake.stream_calls.clear()
    chat_events(env, "thanks, and now?", conversation_id=conversation_id)
    _assert_prefix(warm, env.fake.stream_calls[0])


# --- automatic permissions -----------------------------------------------------------


def test_automatic_mode_ignores_manual_per_tool_choices(env: Env):
    env.settings_patch({"permissions": {"write_file": "deny"}})  # set while in Manual, then hidden
    env.fake.stream_script = [
        tool_call_result("write_file", {"path": "auto.txt", "content": "ok"}),
        ChatResult(content="Written."),
    ]
    events, _raw = chat_events(env, "write auto.txt")
    assert "approval" not in [event for event, _ in events]
    assert (env.workspace / "auto.txt").read_text(encoding="utf-8") == "ok"


def test_allow_in_this_chat_covers_later_risky_steps_of_that_chat_only(env: Env):
    env.fake.stream_script = [
        tool_call_result("run_command", {"command": "kill 999999"}),
        tool_call_result("run_command", {"command": "kill 999998"}, call_id="call_2"),
        ChatResult(content="Done."),
    ]
    response = env.client.post("/api/chat", json={"message": "stop those processes"}, buffered=False)
    stream = SSEStream(response).start()
    approval = stream.wait_for("approval")
    assert approval["always_scope"] == "chat"
    resolved = env.new_client().post(f"/api/approvals/{approval['approval_id']}", json={"decision": "always"})
    assert resolved.status_code == 200
    events = stream.finish()
    calls = [payload for event, payload in events if event == "tool_call"]
    assert [call["requires_approval"] for call in calls] == [True, False]
    assert "allowed for this chat" in calls[1]["reasons"]
    assert [event for event, _ in events].count("approval") == 1
    # Nothing persisted: another chat, or a restart, asks again.
    assert env.svc.settings_provider()["permissions"].get("run_command") != "auto"
    conversation_id = next(payload for event, payload in events if event == "meta")["conversation_id"]
    assert env.svc.agent._chat_allowed == {conversation_id: {"run_command"}}


def test_manual_mode_always_still_sets_the_tool_permission(env: Env):
    env.settings_patch({"autonomy": "supervised"})
    env.fake.stream_script = [
        tool_call_result("write_file", {"path": "m.txt", "content": "m"}),
        ChatResult(content="Written."),
    ]
    response = env.client.post("/api/chat", json={"message": "write m.txt"}, buffered=False)
    stream = SSEStream(response).start()
    approval = stream.wait_for("approval")
    assert approval["always_scope"] == "tool"
    env.new_client().post(f"/api/approvals/{approval['approval_id']}", json={"decision": "always"})
    stream.finish()
    assert env.svc.settings_provider()["permissions"]["write_file"] == "auto"


def test_system_prompt_describes_the_active_permission_mode(env: Env):
    chat_events(env, "hi")
    automatic = env.fake.stream_calls[-1]["messages"][0]["content"]
    assert "Tools run on their own" in automatic
    env.settings_patch({"autonomy": "supervised"})
    chat_events(env, "hi again")
    manual = env.fake.stream_calls[-1]["messages"][0]["content"]
    assert "asks the user for permission first" in manual


# --- the remember tool: save, update, forget, grounded --------------------------------


def _remember(env: Env, said: str, **args):
    env.fake.stream_script = [tool_call_result("remember", args), ChatResult(content="Noted.")]
    events, _raw = chat_events(env, said)
    return next(payload for event, payload in events if event == "tool_result")


def test_remember_saves_what_the_user_said(env: Env):
    result = _remember(env, "I live in Istanbul", content="User lives in Istanbul")
    assert result["ok"] is True
    (memory,) = env.svc.db.list_memories()
    assert memory["content"] == "User lives in Istanbul" and memory["origin"] == "chat"
    assert memory["source_conversation_id"]


def test_remember_refuses_facts_nobody_stated(env: Env):
    result = _remember(env, "Hi! What can you do?", content="Kiwibro is a local AI agent for Linux")
    assert result["ok"] is False
    result = _remember(env, "Hi! What can you do?", content="User prefers dark themes in every editor")
    assert result["ok"] is False
    assert env.svc.db.list_memories() == []


def test_remember_update_replaces_the_outdated_fact(env: Env):
    env.svc.memory.remember("User lives in Istanbul")
    result = _remember(env, "I moved to Ankara last month", content="User lives in Ankara",
                       action="update", old="lives in Istanbul")
    assert result["ok"] is True and result["preview"].startswith("Updated memory")
    contents = [m["content"] for m in env.svc.db.list_memories()]
    assert contents == ["User lives in Ankara"]


def test_remember_forget_removes_the_memory_but_keeps_pinned_ones(env: Env):
    plain = env.svc.memory.remember("User drinks oat milk")
    pinned = env.svc.memory.remember("User's cat is called Pamuk")
    env.svc.memory.update(pinned["id"], pinned=True)
    assert _remember(env, "forget the oat milk thing", content="oat milk", action="forget")["ok"] is True
    assert env.svc.db.get_memory(plain["id"]) is None
    assert _remember(env, "forget my cat Pamuk", content="cat Pamuk", action="forget")["ok"] is False
    assert env.svc.db.get_memory(pinned["id"]) is not None


def test_remember_action_aliases_are_repaired():
    args = tools.validate_args("remember", {"content": "x y z", "action": "delete"})
    assert args["action"] == "forget"
    assert tools.validate_args("remember", {"content": "x y z", "action": "bogus"})["action"] == "save"


# --- origins: the user's own memories are protected ------------------------------------


def test_extraction_marks_learned_memories_and_never_deletes_the_users(env: Env):
    mine = env.svc.memory.remember("User works as a nurse")  # Memory sheet: origin "you"
    assert mine["origin"] == "you"
    conversation = env.svc.db.create_conversation(title="x", model="fake-1")
    counts = env.svc.memory._apply_ops(
        [{"op": "delete", "id": mine["id"]}, {"op": "add", "kind": "fact", "content": "User has two dogs"}],
        "I have two dogs",
        conversation_id=conversation["id"],
    )
    assert counts == {"added": 1, "updated": 0, "removed": 0}
    assert env.svc.db.get_memory(mine["id"]) is not None
    learned = next(m for m in env.svc.db.list_memories() if "dogs" in m["content"])
    assert (learned["origin"], learned["source_conversation_id"]) == ("learned", conversation["id"])


def test_editing_a_learned_memory_makes_it_the_users(env: Env):
    learned = env.svc.db.create_memory("fact", "User likes tea", "user likes tea", origin="learned")
    updated = env.svc.memory.update(learned["id"], content="User likes green tea")
    assert updated["origin"] == "you"


def test_pruning_never_evicts_the_users_memories(env: Env):
    mine = env.svc.memory.remember("User is allergic to peanuts")
    for index in range(5):
        env.svc.db.create_memory("fact", f"learned fact number {index}", f"learned fact number {index}", origin="learned")
    removed = env.svc.db.prune_memories(cap=2)
    assert removed == 4
    assert env.svc.db.get_memory(mine["id"]) is not None


# --- the model's own window does not limit the chat -------------------------------------


def test_a_message_longer_than_the_window_is_paged_into_the_library(env: Env):
    env.fake.models[0].update(context=8192, max_context=8192)
    pasted = "\n".join(f"line {index}: the quick brown fox jumps over the lazy dog" for index in range(3000))
    events, _raw = chat_events(env, pasted)
    assert "error" not in [event for event, _ in events]
    sent = env.fake.stream_calls[0]["messages"][-1]["content"]
    assert "more characters stored as" in sent and "doc_read" in sent and len(sent) < len(pasted) // 3
    assert sent.startswith("line 0:")
    conversation_id = next(payload for event, payload in events if event == "meta")["conversation_id"]
    stored = next(m for m in env.svc.db.list_messages(conversation_id) if m["role"] == "user")
    assert stored["content"] == pasted  # the chat keeps the full text


def test_a_tiny_window_gets_the_minimal_tool_set(env: Env):
    env.fake.models[0].update(context=4096, max_context=4096)
    chat_events(env, "hi")
    names = {tool["function"]["name"] for tool in env.fake.stream_calls[0]["tools"] or []}
    assert names and names <= set(tools.select_tool_names("minimal"))
    env.fake.models[0].update(context=32768, max_context=32768)
    chat_events(env, "hi again")
    assert len(env.fake.stream_calls[-1]["tools"]) == len(tools.select_tool_names("auto"))


def test_tool_schemas_are_reserved_in_the_budget(env: Env):
    events, _raw = chat_events(env, "hello")
    budget = next(payload for event, payload in events if event == "meta")["context"]["budget"]
    full = env.svc.context.budget_for("fake-1")[1]
    assert budget < full - 2000  # the auto tool set is ~3.5k tokens


# --- measured small-model mistakes (lfm2.5-1.2b, live run) -----------------------------


def test_remember_update_with_swapped_arguments_is_put_right(env: Env):
    env.svc.memory.remember("User lives in Istanbul")
    result = _remember(env, "Quick update: I moved to Ankara last week.", action="update",
                       content="User lives in Istanbul", old="User lives in Ankara")
    assert result["ok"] is True
    assert [m["content"] for m in env.svc.db.list_memories()] == ["User lives in Ankara"]


def test_extraction_update_of_an_unrelated_memory_becomes_an_add(env: Env):
    other = env.svc.db.create_memory("fact", "User has a cat named Pamuk", "user has a cat named pamuk", origin="learned")
    counts = env.svc.memory._apply_ops(
        [{"op": "update", "id": other["id"], "content": "User lives in Istanbul"}], "I live in Istanbul"
    )
    assert counts["added"] == 1 and counts["updated"] == 0
    contents = sorted(m["content"] for m in env.svc.db.list_memories())
    assert contents == ["User has a cat named Pamuk", "User lives in Istanbul"]


def test_extraction_update_must_be_grounded_and_about_the_user(env: Env):
    city = env.svc.db.create_memory("fact", "User lives in Istanbul", "user lives in istanbul", origin="learned")
    env.svc.memory._apply_ops(
        [{"op": "update", "id": city["id"], "content": "User lives in Berlin"}], "I like the weather today"
    )
    env.svc.memory._apply_ops(
        [{"op": "update", "id": city["id"], "content": "Kiwibro is a local AI agent that lives in Istanbul"}],
        "Kiwibro, I live in Istanbul",
    )
    assert env.svc.db.get_memory(city["id"])["content"] == "User lives in Istanbul"
    env.svc.memory._apply_ops(
        [{"op": "update", "id": city["id"], "content": "User lives in Ankara"}], "I moved; I live in Ankara now"
    )
    assert env.svc.db.get_memory(city["id"])["content"] == "User lives in Ankara"


def test_a_fact_a_newer_memory_already_changed_is_not_learned(env: Env):
    conversation = env.svc.db.create_conversation(title="x", model="fake-1")
    stated = env.svc.db.add_message(conversation["id"], "user", "I live in Istanbul and I mostly write Python.")
    window = [stated]
    moved = env.svc.memory.remember(
        "User moved to Ankara", source_conversation_id=conversation["id"]
    )  # saved by the next turn, while this window was still being extracted
    assert moved["updated_at"] > stated["created_at"]
    counts = env.svc.memory._apply_ops(
        [{"op": "add", "content": "User lives in Istanbul"}, {"op": "add", "content": "User writes Python"}],
        stated["content"], conversation_id=conversation["id"], window=window,
    )
    assert counts["added"] == 1
    assert sorted(m["content"] for m in env.svc.db.list_memories()) == ["User moved to Ankara", "User writes Python"]


def test_related_needs_the_same_subject():
    from kiwibro.memory import related

    assert related("User lives in Istanbul", "User moved to Ankara")
    assert related("User works at a bank", "User's job: teacher") is False  # "User's job" is another subject
    assert not related("User's sister Ana lives in Porto", "User lives in Istanbul")
    assert not related("User has a cat named Pamuk", "User lives in Istanbul")


def test_stop_works_while_lm_studio_is_still_prefilling():
    """LM Studio answers a streaming request with headers at once and sends nothing until the
    first token (75 s for a 6k-token prompt on a laptop CPU); Stop must not wait for it."""
    import http.server
    import threading
    import time

    from kiwibro.llm import LMStudioClient, TurnAborted

    class Silent(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.flush()
            time.sleep(5)  # "prefilling"

        def log_message(self, *args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Silent)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = LMStudioClient(lambda: {"lmstudio_url": f"http://127.0.0.1:{server.server_port}"})
    abort = threading.Event()
    threading.Timer(0.3, abort.set).start()
    started = time.monotonic()
    try:
        client.stream_chat(model="m", messages=[{"role": "user", "content": "hi"}], abort=abort)
        raised = None
    except TurnAborted as exc:
        raised = exc
    finally:
        server.shutdown()
    assert isinstance(raised, TurnAborted)
    assert time.monotonic() - started < 2.0


def test_forget_never_takes_a_merely_similar_sentence(env: Env):
    jazz = env.svc.memory.remember("User likes jazz")
    assert env.svc.memory.find_target("User likes tea") is None
    assert env.svc.memory.forget("User likes tea") == (None, "missing")
    assert env.svc.db.get_memory(jazz["id"]) is not None
    assert env.svc.memory.find_target("jazz")["id"] == jazz["id"]


def test_measured_4b_batch_keeps_the_newest_fact_and_the_bundled_one(env: Env):
    """spark-x2.6-4b returned these two adds for "I'm Yunus. I live in Istanbul and I mostly
    write Python." followed by "Quick update: I moved to Ankara last week."; the first
    version of the newer-fact rule kept the bundle and dropped Ankara."""
    conversation = env.svc.db.create_conversation(title="x", model="fake-1")
    first = env.svc.db.add_message(conversation["id"], "user", "Hi! I'm Yunus. I live in Istanbul and I mostly write Python.")
    second = env.svc.db.add_message(conversation["id"], "user", "Quick update: I moved to Ankara last week.")
    window = [first, second]
    transcript = "\n".join(m["content"] for m in window)
    env.svc.memory._apply_ops(
        [{"op": "add", "content": "User is Yunus, lives in Istanbul and mostly writes Python."},
         {"op": "add", "content": "User moved to Ankara last week."}],
        transcript, conversation_id=conversation["id"], window=window,
    )
    contents = sorted(m["content"] for m in env.svc.db.list_memories())
    assert "User moved to Ankara last week." in contents  # the newest fact is never dropped
    assert any("Yunus" in c and "Python" in c for c in contents)  # bundled facts are not lost


def test_an_older_single_fact_loses_to_a_later_one_in_the_same_batch(env: Env):
    conversation = env.svc.db.create_conversation(title="x", model="fake-1")
    first = env.svc.db.add_message(conversation["id"], "user", "I live in Istanbul.")
    second = env.svc.db.add_message(conversation["id"], "user", "Actually I moved to Ankara.")
    window = [first, second]
    env.svc.memory._apply_ops(
        [{"op": "add", "content": "User lives in Istanbul"}, {"op": "add", "content": "User moved to Ankara"}],
        "\n".join(m["content"] for m in window), conversation_id=conversation["id"], window=window,
    )
    assert [m["content"] for m in env.svc.db.list_memories()] == ["User moved to Ankara"]
