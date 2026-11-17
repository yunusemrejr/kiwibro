"""Native Auto can supply verified inventory for an explicitly scoped repo inspection."""

from io import BytesIO
from dataclasses import replace

import pytest

from kiwibro import safemode
from kiwibro.llm import ChatResult
from kiwibro.mind import KIWIMIND
from kiwibro.sandbox import Workspace
import test_backend
from test_backend import Env, chat_events, collect_sse, tool_call_result
from test_native_prompt import INFO

env = test_backend.env
FOLDER = "packages/flow"
SOURCE = f"{FOLDER}/lib/queue.py"
ASK = f"In {FOLDER}/, trace how inbound work gets queued. Read the shared policy and explain its default."


@pytest.fixture
def native_repo(env: Env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(env.svc.llm, "resolve_model", lambda _requested=None: KIWIMIND)
    monkeypatch.setattr(env.svc.llm, "model_info", lambda _model: INFO)
    source = env.workspace / SOURCE
    source.parent.mkdir(parents=True)
    source.write_text("DEFAULT_SLOTS = 6\n", encoding="utf-8")
    env.fake.stream_script = [ChatResult(content="The model can select the source.")]


@pytest.mark.parametrize("request_text", [
    ASK,
    f"In `./{FOLDER}/`, inspect the queue wiring and describe it.",
    f"Please in {FOLDER}/, review the handler behavior. Diagnosis only.",
])
def test_native_repo_start_keeps_the_active_schema_and_model_source_read(env: Env, native_repo, request_text: str):
    env.fake.stream_script = [
        tool_call_result("read_file", {"path": SOURCE}, call_id="source"),
        ChatResult(content=f"The default is 6 in {SOURCE}."),
    ]
    events, _ = chat_events(env, request_text)
    assert [(data["name"], data["ok"]) for kind, data in events if kind == "tool_result"] == [
        ("list_files", True), ("read_file", True),
    ]
    listing = next(data for kind, data in events if kind == "tool_call" and data["name"] == "list_files")
    assert listing["args"] == {"path": FOLDER} and listing["requires_approval"] is False
    first = env.fake.stream_calls[0]
    ordinary = env.svc.agent._tool_list(KIWIMIND, env.svc.settings_provider(), INFO)
    assert first["tools"] == ordinary
    assert first["messages"][-1]["role"] == "tool" and f"file\t{SOURCE}\t" in first["messages"][-1]["content"]
    assert "DEFAULT_SLOTS = 6" not in first["messages"][-1]["content"]
    assert "DEFAULT_SLOTS = 6" in env.fake.stream_calls[1]["messages"][-1]["content"]


def test_native_repo_start_avoids_relisting_the_same_observed_ancestor(env: Env, native_repo):
    env.fake.stream_script = [
        tool_call_result("read_file", {"path": f"{FOLDER}/guess/queue.py"}, call_id="missing"),
        tool_call_result("read_file", {"path": SOURCE}, call_id="source"),
        ChatResult(content=f"The default is 6 in {SOURCE}."),
    ]
    events, _ = chat_events(env, ASK)
    assert [(data["name"], data["ok"]) for kind, data in events if kind == "tool_result"] == [
        ("list_files", True), ("read_file", False), ("read_file", True),
    ]


@pytest.mark.parametrize("request_text", [
    f"In {FOLDER}, inspect the routing.",
    f"In {FOLDER}/, inspect the routing in gateway/.",
    f"In {FOLDER}/, trace the bug and fix it.",
    f"In {FOLDER}/, review the routing. Then run the tests.",
    f"In {FOLDER}/, trace the wiring. Do not list files.",
    f"In {FOLDER}/, inspect the routing. Read only queue.py.",
    "In ../outside/, inspect the routing.",
    "In /unrelated/, review the routing.",
    "In missing/, review the routing.",
    "/goal " + ASK,
])
def test_native_repo_start_does_not_guess_scope_or_do_mixed_work(env: Env, native_repo, request_text: str):
    events, _ = chat_events(env, request_text)
    assert not any(kind == "tool_call" and data["name"] == "list_files" for kind, data in events)
    assert env.fake.stream_calls


@pytest.mark.parametrize(("profile", "permission"), [("full", "auto"), ("auto", "ask"), ("auto", "deny")])
def test_native_repo_start_respects_tool_profile_and_permission(env: Env, native_repo, profile: str, permission: str):
    env.settings_patch({"tool_profile": profile, "autonomy": "supervised", "permissions": {"list_files": permission}})
    events, _ = chat_events(env, ASK)
    assert not any(kind in ("tool_call", "approval") for kind, _ in events)
    assert env.fake.stream_calls


def test_native_repo_start_refuses_a_workspace_alias(env: Env, native_repo):
    (env.workspace / "alias").symlink_to(env.workspace / FOLDER, target_is_directory=True)
    events, _ = chat_events(env, "In alias/, trace the request routing.")
    assert not any(kind == "tool_call" and data["name"] == "list_files" for kind, data in events)


def test_native_repo_start_honors_safe_copy_refusal(env: Env, native_repo, monkeypatch: pytest.MonkeyPatch):
    gate = safemode.gate
    monkeypatch.setattr(safemode, "gate", lambda db, cid, tool: (
        safemode.Gate(active=True, refusal="Listing is unavailable for this safe copy.")
        if tool == "list_files" else gate(db, cid, tool)
    ))
    events, _ = chat_events(env, ASK)
    assert not any(kind == "tool_call" and data["name"] == "list_files" for kind, data in events)


def test_native_repo_start_leaves_attachment_turns_to_the_model(env: Env, native_repo):
    upload = env.client.post("/api/attachments", data={"file": (BytesIO(b"local fixture"), "note.txt")})
    assert upload.status_code == 200
    attachment = upload.get_json()["attachment"]
    response = env.client.post("/api/chat", json={"message": ASK, "attachments": [{"path": attachment["path"]}]},
                               buffered=False)
    assert response.status_code == 200
    events, _ = collect_sse(response)
    assert not any(kind == "tool_call" and data["name"] == "list_files" for kind, data in events)
    assert env.fake.stream_calls


def test_native_repo_start_respects_midturn_workspace_change(env: Env, native_repo, monkeypatch: pytest.MonkeyPatch,
                                                       tmp_path):
    original_workspace = env.svc.agent._workspace
    second = Workspace(tmp_path / "second")
    (second.root / FOLDER).mkdir(parents=True)
    prepare = env.svc.agent.context.prepare
    switched = False

    def prepare_and_switch(*args, **kwargs):
        nonlocal switched
        result = prepare(*args, **kwargs)
        switched = True
        return result

    monkeypatch.setattr(env.svc.agent.context, "prepare", prepare_and_switch)
    monkeypatch.setattr(env.svc.agent, "_workspace",
                        lambda conversation_id=None: second if switched else original_workspace(conversation_id))
    events, _ = chat_events(env, ASK)
    assert not any(kind == "tool_call" and data["name"] == "list_files" for kind, data in events)


@pytest.mark.parametrize("change", ["workspace", "permission", "profile"])
def test_native_repo_start_rechecks_scope_at_the_tool_gate(
        env: Env, native_repo, monkeypatch: pytest.MonkeyPatch, tmp_path, change: str,
):
    agent = env.svc.agent
    original_workspace = agent._workspace
    call = agent._call_first
    second = Workspace(tmp_path / "second")
    (second.root / FOLDER).mkdir(parents=True)
    switched = False

    def change_before_dispatch(turn, name, args, **kwargs):
        nonlocal switched
        assert name == "list_files"
        if change == "workspace":
            switched = True
        else:
            settings = {"tool_profile": "full"} if change == "profile" else {
                "autonomy": "supervised", "permissions": {"list_files": "ask"},
            }
            response = env.new_client().patch("/api/settings", json=settings)
            assert response.status_code == 200
        return call(turn, name, args, **kwargs)

    monkeypatch.setattr(agent, "_call_first", change_before_dispatch)
    monkeypatch.setattr(agent, "_workspace",
                        lambda conversation_id=None: second if switched else original_workspace(conversation_id))
    events, _ = chat_events(env, ASK)
    assert not any(kind == "approval" for kind, _ in events)
    results = [data for kind, data in events if kind == "tool_result"]
    assert len(results) == 1 and not results[0]["ok"]
    assert results[0]["preview"] == ("Workspace changed" if change == "workspace" else "Automatic check skipped")


def test_native_repo_start_puts_steering_in_the_first_generation(env: Env, native_repo, monkeypatch: pytest.MonkeyPatch):
    conversation = env.svc.db.create_conversation()
    prepare = env.svc.agent.context.prepare
    steering = "Skip the folder listing and explain what a work queue is."
    sent = False

    def prepare_and_steer(*args, **kwargs):
        nonlocal sent
        result = prepare(*args, **kwargs)
        if not sent:
            response = env.new_client().post("/api/chat/steer", json={
                "conversation_id": conversation["id"], "message": steering,
            })
            assert response.status_code == 200
            sent = True
        return result

    monkeypatch.setattr(env.svc.agent.context, "prepare", prepare_and_steer)
    events, _ = chat_events(env, ASK, conversation_id=conversation["id"])
    assert any(kind == "steering" and data["text"] == steering for kind, data in events)
    assert not any(kind == "tool_call" and data["name"] == "list_files" for kind, data in events)
    assert any(steering in (message.get("content") or "") for message in env.fake.stream_calls[0]["messages"])


@pytest.mark.parametrize("name", ["read_file", "read_files"])
def test_native_repo_finishes_after_a_round_of_exact_verified_reads(env: Env, native_repo, name: str):
    additional = f"{FOLDER}/lib/worker.py"
    (env.workspace / additional).write_text("from .queue import DEFAULT_SLOTS\n", encoding="utf-8")
    args = {"path": SOURCE, "start": 1} if name == "read_file" else {"paths": [SOURCE, additional], "start": 1}
    env.fake.stream_script = [
        tool_call_result(name, args, call_id="original"),
        tool_call_result(name, args, call_id="repeat"),
        ChatResult(content=f"The default is 6 in {SOURCE}."),
    ]
    events, _ = chat_events(env, ASK)
    assert [(data["name"], data["ok"]) for kind, data in events if kind == "tool_result"] == [
        ("list_files", True), (name, True), (name, False),
    ]
    assert len(env.fake.stream_calls) == 3
    final = env.fake.stream_calls[-1]
    assert final["tools"] is None
    assert events[-1][1]["inference"]["repo_repeat_finished"]
    assert any(row["role"] == "tool" and row["tool_call_id"] == "original" and "DEFAULT_SLOTS = 6" in row["content"]
               for row in final["messages"])
    assert not any(row.get("tool_call_id") == "repeat" for row in final["messages"])


def test_native_repo_repeat_keeps_a_new_line_range_and_new_read_in_the_batch(env: Env, native_repo):
    sibling = f"{FOLDER}/lib/worker.py"
    (env.workspace / SOURCE).write_text("DEFAULT_SLOTS = 6\nCAPACITY = 9\n", encoding="utf-8")
    (env.workspace / sibling).write_text("from .queue import DEFAULT_SLOTS\n", encoding="utf-8")
    first = {"path": SOURCE, "start": 1, "end": 1}
    repeat = tool_call_result("read_file", first, call_id="repeat")
    repeat.tool_calls.extend(tool_call_result("read_file", {"path": SOURCE, "start": 2}, "second").tool_calls)
    repeat.tool_calls.extend(tool_call_result("read_file", {"path": sibling}, "sibling").tool_calls)
    env.fake.stream_script = [
        tool_call_result("read_file", first, "original"), repeat,
        ChatResult(content=f"The default is 6 in {SOURCE}."),
    ]
    events, _ = chat_events(env, ASK)
    assert [data["ok"] for kind, data in events if kind == "tool_result"] == [True, True, False, True, True]
    assert env.fake.stream_calls[-1]["tools"] is not None
    assert any("CAPACITY = 9" in row.get("content", "") for row in env.fake.stream_calls[-1]["messages"])


@pytest.mark.parametrize("change", ["external", "workspace"])
def test_native_repo_repeat_rereads_after_a_mutation(env: Env, native_repo, monkeypatch: pytest.MonkeyPatch, change: str):
    tool_args = {"path": SOURCE}
    mutations = []
    if change == "external":
        stream = env.svc.llm.stream_chat

        def mutate(**kwargs):
            if len(env.fake.stream_calls) == 1:
                (env.workspace / SOURCE).write_text("DEFAULT_SLOTS = 11\n", encoding="utf-8")
            return stream(**kwargs)

        monkeypatch.setattr(env.svc.llm, "stream_chat", mutate)
    else:
        env.settings_patch({"autonomy": "supervised", "permissions": {"write_file": "auto"}})
        mutations = [tool_call_result("write_file", {"path": "notes.txt", "content": "Synthetic note."}, "save")]
    env.fake.stream_script = [
        tool_call_result("read_file", tool_args, "original"), *mutations,
        tool_call_result("read_file", tool_args, "after_change"),
        ChatResult(content=f"The evidence is in {SOURCE}."),
    ]
    events, _ = chat_events(env, ASK)
    assert [data["ok"] for kind, data in events if kind == "tool_result" and data["name"] == "read_file"] == [
        True, True,
    ]
    assert env.fake.stream_calls[-1]["tools"] is not None
    if change == "external":
        assert any("DEFAULT_SLOTS = 11" in row.get("content", "") for row in env.fake.stream_calls[-1]["messages"])


@pytest.mark.parametrize("condition", ["paged", "guarded", "outside_scope", "omitted"])
def test_native_repo_repeat_requires_usable_evidence_in_the_prompt(
        env: Env, native_repo, monkeypatch: pytest.MonkeyPatch, condition: str,
):
    path = "sibling.py" if condition == "outside_scope" else SOURCE
    if condition == "outside_scope":
        (env.workspace / path).write_text("SIBLING = True\n", encoding="utf-8")
    if condition == "paged":
        (env.workspace / path).write_text("DEFAULT_SLOTS = 6\n" * 1250, encoding="utf-8")
    if condition == "guarded":
        token = "sk-proj-" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        (env.workspace / path).write_text(f"TOKEN = '{token}'\n", encoding="utf-8")
    if condition == "omitted":
        extend = env.svc.agent.context.extend

        def omit_old_result(*args, **kwargs):
            result = extend(*args, **kwargs)
            return replace(result, messages=[
                {**row, "content": "[earlier tool output omitted]"} if row.get("tool_call_id") == "original" else row
                for row in result.messages
            ])

        monkeypatch.setattr(env.svc.agent.context, "extend", omit_old_result)
    env.fake.stream_script = [
        tool_call_result("read_file", {"path": path}, "original"),
        tool_call_result("read_file", {"path": path}, "repeat"),
        ChatResult(content="The model kept its tools."),
    ]
    events, _ = chat_events(env, ASK)
    assert [data["ok"] for kind, data in events if kind == "tool_result" and data["name"] == "read_file"] == [
        True, True,
    ]
    assert env.fake.stream_calls[-1]["tools"] is not None


def test_native_repo_repeat_honors_an_explicit_reread_request(env: Env, native_repo):
    env.fake.stream_script = [
        tool_call_result("read_file", {"path": SOURCE}, "original"),
        tool_call_result("read_file", {"path": SOURCE}, "requested_reread"),
        ChatResult(content=f"The evidence is in {SOURCE}."),
    ]
    events, _ = chat_events(env, ASK + " Read the same file twice to check the source.")
    assert [data["ok"] for kind, data in events if kind == "tool_result"] == [True, True, True]
    assert env.fake.stream_calls[-1]["tools"] is not None
