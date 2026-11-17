"""Native Auto can supply verified inventory for an explicitly scoped repo inspection."""

from io import BytesIO
from dataclasses import replace
from threading import Event

import pytest

from kiwibro import safemode, safety
from kiwibro.llm import ChatResult, ContextLimitError, LLMError
from kiwibro.mind import KIWIMIND
from kiwibro.sandbox import Workspace
import test_backend
from test_backend import Env, SSEStream, chat_events, collect_sse, tool_call_result
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
    assert final["tools"] == env.fake.stream_calls[-2]["tools"]
    assert final["extra"]["tool_choice"] == "none" and not final["open_call"]
    assert events[-1][1]["inference"]["repo_repeat_finished"]
    assert any(row["role"] == "tool" and row["tool_call_id"] == "original" and "DEFAULT_SLOTS = 6" in row["content"]
               for row in final["messages"])
    assert not any(row.get("tool_call_id") == "repeat" for row in final["messages"])
    assert "workspace-relative paths" in final["messages"][-1]["content"]
    assert "successful read tool arguments" in final["messages"][-1]["content"]
    assert SOURCE not in final["messages"][-1]["content"]


@pytest.mark.parametrize("response", ["structured", "gemma_text", "tool_error", "context_limit"])
def test_native_repo_final_falls_back_to_plain_generation_without_executing_calls(env: Env, native_repo, response: str):
    raw_call = "<|tool_call>call:write_file{path:untouched.txt,content:bad}<tool_call|>"
    bad = {
        "structured": tool_call_result("write_file", {"path": "untouched.txt", "content": "bad"}, "unexpected"),
        "gemma_text": ChatResult(content=raw_call),
        "tool_error": LLMError("tools", "The model did not honor tool choice."),
        "context_limit": ContextLimitError(),
    }[response]
    answer = f"The default is 6 in {SOURCE}."
    env.fake.stream_script = [
        tool_call_result("read_file", {"path": SOURCE}, "original"),
        tool_call_result("read_file", {"path": SOURCE}, "repeat"),
        bad,
        ChatResult(content=answer),
    ]
    events, raw = chat_events(env, ASK)
    assert len(env.fake.stream_calls) == 4
    constrained, fallback = env.fake.stream_calls[-2:]
    assert constrained["tools"] == env.fake.stream_calls[1]["tools"] and constrained["extra"]["tool_choice"] == "none"
    assert fallback["tools"] is None and "tool_choice" not in (fallback["extra"] or {}) and not fallback["open_call"]
    assert [data["text"] for kind, data in events if kind == "delta"] == [answer]
    assert all(data["name"] != "write_file" for kind, data in events if kind == "tool_call")
    assert raw_call not in raw and not (env.workspace / "untouched.txt").exists()


def test_native_repo_final_stays_plain_when_the_model_state_changes(env: Env, native_repo, monkeypatch: pytest.MonkeyPatch):
    answer_anyway = env.svc.agent._answer_anyway
    helper = "synthetic/helper"

    def change_writer(turn, *args, **kwargs):
        if kwargs.get("repo_answer"):
            turn.model = helper
        return answer_anyway(turn, *args, **kwargs)

    monkeypatch.setattr(env.svc.agent, "_answer_anyway", change_writer)
    monkeypatch.setattr(env.svc.llm, "model_info", lambda model: INFO if model == KIWIMIND else {
        "id": model, "context": 8192, "tools": False,
    })
    env.fake.stream_script = [
        tool_call_result("read_file", {"path": SOURCE}, "original"),
        tool_call_result("read_file", {"path": SOURCE}, "repeat"),
        ChatResult(content=f"The default is 6 in {SOURCE}."),
    ]
    events, _ = chat_events(env, ASK)
    assert events[-1][0] == "done"
    final = env.fake.stream_calls[-1]
    assert final["model"] == helper and final["tools"] is None and not final["open_call"]
    assert "tool_choice" not in (final["extra"] or {})
    assert any(row["role"] == "tool" for row in final["messages"])


def test_native_repo_final_streams_prose_before_the_model_finishes(env: Env, native_repo, monkeypatch: pytest.MonkeyPatch):
    prefix = "The default is 6 "
    suffix = f"in {SOURCE}."
    release = Event()
    stream_chat = env.svc.llm.stream_chat

    def split_final(**kwargs):
        if (kwargs.get("extra") or {}).get("tool_choice") != "none":
            return stream_chat(**kwargs)
        result = stream_chat(**{**kwargs, "on_delta": None})
        emit = kwargs["on_delta"]
        emit(prefix)
        assert release.wait(timeout=5), "The visible answer waited for the whole generation"
        emit(suffix)
        return result

    monkeypatch.setattr(env.svc.llm, "stream_chat", split_final)
    env.fake.stream_script = [
        tool_call_result("read_file", {"path": SOURCE}, "original"),
        tool_call_result("read_file", {"path": SOURCE}, "repeat"),
        ChatResult(content=prefix + suffix),
    ]
    stream = SSEStream(env.client.post("/api/chat", json={"message": ASK}, buffered=False)).start()
    try:
        assert stream.wait_for("delta", timeout=4)["text"] == prefix
        assert not stream.done.is_set()
    finally:
        release.set()
    events = stream.finish(timeout=4)
    assert [data["text"] for kind, data in events if kind == "delta"] == [prefix, suffix]


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


DEPENDENCY_REQUEST = (
    f"In {FOLDER}/, trace why the router and the worker enforce the same limit. "
    "Read their shared implementation and name its source."
)
SIBLING = f"{FOLDER}/lib/worker.py"
DEPENDENCY = f"{FOLDER}/core/limits/concurrency.py"


@pytest.fixture
def repo_imports(env: Env, native_repo):
    source = "from core.limits.concurrency import allowed_jobs\n"
    (env.workspace / SOURCE).write_text(source + "ROUTER_MAX = allowed_jobs()\n", encoding="utf-8")
    (env.workspace / SIBLING).write_text(source + "WORKER_MAX = allowed_jobs()\n", encoding="utf-8")
    dependency = env.workspace / DEPENDENCY
    dependency.parent.mkdir(parents=True)
    dependency.write_text("def allowed_jobs():\n    return 13\n", encoding="utf-8")
    return [
        tool_call_result("read_file", {"path": SOURCE}, "router"),
        tool_call_result("read_file", {"path": SIBLING}, "worker"),
        tool_call_result("read_file", {"path": SOURCE}, "repeat"),
    ]


@pytest.mark.parametrize("target", ["module", "package"])
def test_native_repo_repeat_reads_one_shared_python_implementation(env: Env, repo_imports, target: str):
    path = DEPENDENCY
    if target == "package":
        path = f"{DEPENDENCY.removesuffix('.py')}/__init__.py"
        (env.workspace / path).parent.mkdir()
        (env.workspace / DEPENDENCY).rename(env.workspace / path)
    env.fake.stream_script = [*repo_imports, ChatResult(content=f"The shared limit is 13 in {path}.")]
    events, _ = chat_events(env, DEPENDENCY_REQUEST)
    calls = [data for kind, data in events if kind == "tool_call"]
    assert [(data["name"], data["ok"]) for kind, data in events if kind == "tool_result"] == [
        ("list_files", True), ("read_file", True), ("read_file", True), ("read_file", False), ("read_file", True),
    ]
    assert calls[-1]["args"]["path"] == path and calls[-1]["requires_approval"] is False
    assert len(env.fake.stream_calls) == 4
    final = env.fake.stream_calls[-1]
    assert final["extra"]["tool_choice"] == "none"
    assert any(row.get("tool_call_id") == calls[-1]["id"] and "return 13" in row["content"]
               for row in final["messages"])


@pytest.mark.parametrize("limit", [
    "unrequested", "conditional", "ambiguous", "stale", "paged", "omitted", "alias", "range", "seen", "denied",
])
def test_native_repo_repeat_does_not_read_an_unverified_dependency(
        env: Env, repo_imports, monkeypatch: pytest.MonkeyPatch, limit: str,
):
    if limit == "ambiguous":
        alternate = env.workspace / f"{DEPENDENCY.removesuffix('.py')}/__init__.py"
        alternate.parent.mkdir()
        alternate.write_text("def allowed_jobs():\n    return 99\n", encoding="utf-8")
    if limit == "stale":
        stream = env.svc.llm.stream_chat

        def change_consumer(**kwargs):
            if len(env.fake.stream_calls) == 2:
                (env.workspace / SIBLING).write_text("WORKER_MAX = 25\n", encoding="utf-8")
            return stream(**kwargs)

        monkeypatch.setattr(env.svc.llm, "stream_chat", change_consumer)
    if limit == "paged":
        worker = env.workspace / SIBLING
        worker.write_text(worker.read_text(encoding="utf-8") + "# synthetic source filler\n" * 550, encoding="utf-8")
    if limit == "omitted":
        extend = env.svc.agent.context.extend

        def omit_consumer(*args, **kwargs):
            result = extend(*args, **kwargs)
            return replace(result, messages=[
                {**row, "content": "[earlier tool output omitted]"} if row.get("tool_call_id") == "worker" else row
                for row in result.messages
            ])

        monkeypatch.setattr(env.svc.agent.context, "extend", omit_consumer)
    if limit == "alias":
        (env.workspace / DEPENDENCY).rename(env.workspace / "outside.py")
        (env.workspace / DEPENDENCY).symlink_to(env.workspace / "outside.py")
    if limit == "range":
        repo_imports[1] = tool_call_result("read_file", {"path": SIBLING, "end": 1}, "worker")
    if limit == "seen":
        repo_imports.insert(2, tool_call_result("read_file", {"path": DEPENDENCY}, "implementation"))
    if limit == "denied":
        decide = safety.decide

        def deny_dependency(settings, name, args, **kwargs):
            result = decide(settings, name, args, **kwargs)
            return replace(result, mode="deny") if name == "read_file" and args["path"] == DEPENDENCY else result

        monkeypatch.setattr(safety, "decide", deny_dependency)
    env.fake.stream_script = [*repo_imports, ChatResult(content="The checked sources use the same import.")]
    request = ASK if limit == "unrequested" else DEPENDENCY_REQUEST
    if limit == "conditional":
        request = request.replace("and name its source", "only after I approve")
    events, _ = chat_events(env, request)
    assert [data["id"] for kind, data in events if kind == "tool_call" and data["args"].get("path") == DEPENDENCY] == (
        ["implementation"] if limit == "seen" else []
    )
    assert env.fake.stream_calls[-1]["extra"]["tool_choice"] == "none"


def test_native_repo_dependency_approval_runs_in_a_normal_model_round(
        env: Env, repo_imports, monkeypatch: pytest.MonkeyPatch,
):
    decide = safety.decide

    def ask_for_dependency(settings, name, args, **kwargs):
        result = decide(settings, name, args, **kwargs)
        return replace(result, mode="ask") if name == "read_file" and args["path"] == DEPENDENCY else result

    monkeypatch.setattr(safety, "decide", ask_for_dependency)
    env.fake.stream_script = [
        *repo_imports, tool_call_result("read_file", {"path": DEPENDENCY}, "reviewed"),
        ChatResult(content=f"The sources share the implementation in {DEPENDENCY}."),
    ]
    response = env.client.post("/api/chat", json={"message": DEPENDENCY_REQUEST}, buffered=False)
    stream = SSEStream(response).start()
    approval = stream.wait_for("approval", timeout=4)
    assert approval["args"]["path"] == DEPENDENCY
    assert env.new_client().post(f"/api/approvals/{approval['approval_id']}", json={"decision": "deny"}).status_code == 200
    events = stream.finish(timeout=4)
    assert [data["id"] for kind, data in events if kind == "tool_call" and data["args"].get("path") == DEPENDENCY] == [
        "reviewed",
    ]
    followup = env.fake.stream_calls[3]
    assert followup["tools"] is not None and "tool_choice" not in (followup["extra"] or {})
    assert DEPENDENCY in followup["messages"][-1]["content"]


def test_native_repo_dependency_rechecks_steering_at_dispatch(env: Env, repo_imports, monkeypatch: pytest.MonkeyPatch):
    call_first = env.svc.agent._call_first
    note = "Leave the shared implementation unread and summarize what you saw so far."

    def steer_before_read(turn, name, args, **kwargs):
        if args.get("path") == DEPENDENCY:
            assert env.svc.agent.steer(turn.conversation_id, note)
        return call_first(turn, name, args, **kwargs)

    monkeypatch.setattr(env.svc.agent, "_call_first", steer_before_read)
    env.fake.stream_script = [*repo_imports, ChatResult(content="Both checked consumers use the same import.")]
    events, _ = chat_events(env, DEPENDENCY_REQUEST)
    assert [(data["name"], data["ok"]) for kind, data in events if kind == "tool_result"] == [
        ("list_files", True), ("read_file", True), ("read_file", True), ("read_file", False), ("read_file", False),
    ]
    assert any(kind == "steering" and data["text"] == note for kind, data in events)
    assert all("return 13" not in row["content"] for row in env.fake.stream_calls[-1]["messages"] if row.get("content"))
    assert env.fake.stream_calls[-1]["tools"] is not None
    assert note in env.fake.stream_calls[-1]["messages"][-1]["content"]
