"""Native Auto can supply verified inventory for an explicitly scoped repo inspection."""

from io import BytesIO

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
