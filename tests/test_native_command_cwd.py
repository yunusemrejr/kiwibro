"""An omitted native command cwd can follow one exact user-scoped command."""

from pathlib import Path

import pytest

from kiwibro import safety, tools
from kiwibro.llm import ChatResult
from kiwibro.mind import KIWIMIND
from kiwibro.sandbox import Workspace
import test_backend
from test_backend import Env, SSEStream, chat_events, tool_call_result
from test_native_prompt import INFO

env = test_backend.env
COMMAND = "python3 -B -m smoke_runner --all"


@pytest.fixture
def native_command(env: Env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(env.svc.llm, "resolve_model", lambda _requested=None: KIWIMIND)
    monkeypatch.setattr(env.svc.llm, "model_info", lambda _model: INFO)
    env.settings_patch({"autonomy": "supervised", "permissions": {"run_command": "auto"}})
    for relative in ("worktrees/unit.v2", "other", "QA fixtures"):
        (env.workspace / relative).mkdir(parents=True)
    calls = []

    def run(workspace: Workspace, command: str, cwd: str | None = None):
        calls.append((command, cwd))
        actual = workspace.public(workspace.resolve(cwd or ".", must_exist=True))
        return {"command": command, "cwd": actual, "exit_code": 0, "output": "Smoke check passed.",
                "duration_ms": 1, "timed_out": False}

    monkeypatch.setattr(Workspace, "run_command", run)
    return calls


def script(env: Env, **args):
    env.fake.stream_script = [
        tool_call_result("run_command", args, "call_run"),
        ChatResult(content="The requested check finished."),
    ]


@pytest.mark.parametrize(("wording", "command", "expected"), [
    (f"Keep reports untouched. Run {COMMAND} in worktrees/unit.v2.", COMMAND, "worktrees/unit.v2"),
    ("Run `printf 'a  b'` with 'QA fixtures' as the working directory.", "printf 'a  b'", "QA fixtures"),
    (f"Run {COMMAND} in .", COMMAND, "."),
])
def test_native_cwd_is_bound_before_validation_and_approval(
        env: Env, native_command, monkeypatch: pytest.MonkeyPatch, wording: str, command: str, expected: str,
):
    env.settings_patch({"autonomy": "supervised", "permissions": {"run_command": "ask"}})
    validated, assessed = [], []
    validate, decide = tools.validate_args, safety.decide

    def validate_args(name, args):
        if name == "run_command":
            validated.append(dict(args))
        return validate(name, args)

    def decision(settings, name, args, **kwargs):
        if name == "run_command":
            assessed.append(dict(args))
        return decide(settings, name, args, **kwargs)

    monkeypatch.setattr(tools, "validate_args", validate_args)
    monkeypatch.setattr(safety, "decide", decision)
    script(env, command=command)
    stream = SSEStream(env.client.post("/api/chat", json={"message": wording}, buffered=False)).start()
    approval = stream.wait_for("approval", timeout=4)
    args = {"command": command, "cwd": expected}
    assert approval["args"] == args and validated == assessed == [args]
    assert not native_command
    assert env.new_client().post(f"/api/approvals/{approval['approval_id']}", json={"decision": "allow"}).status_code == 200
    events = stream.finish(timeout=4)
    assert next(data for kind, data in events if kind == "tool_call")["args"] == args
    assert any(kind == "tool_result" and data["ok"] for kind, data in events)
    assert native_command == [(command, expected)]


@pytest.mark.parametrize(("cwd", "expected"), [("other", "other"), (".", "."), (None, None), ("", None)])
def test_native_cwd_does_not_override_a_supplied_value(env: Env, native_command, cwd, expected):
    script(env, command=COMMAND, cwd=cwd)
    chat_events(env, f"Run {COMMAND} in worktrees/unit.v2.")
    assert native_command == [(COMMAND, expected)]


@pytest.mark.parametrize(("wording", "command"), [
    ("Run the smoke checks in worktrees/unit.v2.", COMMAND),
    (f"Run {COMMAND}.", COMMAND),
    (f"Run {COMMAND} in worktrees/unit.v2.", COMMAND + " --quiet"),
    (f"Compare these instructions.\n```text\nRun {COMMAND} in worktrees/unit.v2.\n```", COMMAND),
    ("Run `printf 'a  b'` in worktrees/unit.v2.", "printf 'a b'"),
])
def test_only_the_exact_direct_command_can_supply_cwd(env: Env, native_command, wording: str, command: str):
    script(env, command=command)
    chat_events(env, wording)
    assert native_command == [(command, None)]


@pytest.mark.parametrize("wording", [
    f"Run {COMMAND} in worktrees/unit.v2 or other.",
    f"Run {COMMAND} in worktrees/unit.v2. Then run {COMMAND} in other.",
    f"- Run {COMMAND} in worktrees/unit.v2.\n- Run {COMMAND} in other.",
    f"Run {COMMAND} in worktrees/unit.v2. Run {COMMAND}.",
    f"Run {COMMAND} in.",
    f"Run {COMMAND} in missing.",
    f"Run {COMMAND} in /elsewhere.",
    f"Run {COMMAND} in ../elsewhere.",
    f"Run {COMMAND} in alias.",
])
def test_native_cwd_refuses_instead_of_using_the_root_when_scope_is_invalid(env: Env, native_command, wording: str):
    (env.workspace / "alias").symlink_to(env.workspace / "other", target_is_directory=True)
    script(env, command=COMMAND)
    events, _ = chat_events(env, wording)
    assert [data["ok"] for kind, data in events if kind == "tool_result"] == [False]
    assert not native_command and not any(kind == "approval" for kind, _ in events)


@pytest.mark.parametrize("change", ["steering", "moved_directory"])
def test_native_cwd_is_rechecked_after_approval(env: Env, native_command, tmp_path: Path, change: str):
    env.settings_patch({"autonomy": "supervised", "permissions": {"run_command": "ask"}})
    conversation = env.svc.db.create_conversation()
    script(env, command=COMMAND)
    stream = SSEStream(env.client.post("/api/chat", json={
        "message": f"Run {COMMAND} in worktrees/unit.v2.", "conversation_id": conversation["id"],
    }, buffered=False)).start()
    approval = stream.wait_for("approval", timeout=4)
    if change == "steering":
        update = env.new_client().post("/api/chat/steer", json={
            "message": "Leave that command alone.", "conversation_id": conversation["id"],
        })
        assert update.status_code == 200
    else:
        folder = env.workspace / "worktrees/unit.v2"
        folder.rename(tmp_path / "moved")
        folder.symlink_to(env.workspace / "other", target_is_directory=True)
    assert env.new_client().post(f"/api/approvals/{approval['approval_id']}", json={"decision": "allow"}).status_code == 200
    events = stream.finish(timeout=4)
    assert [data["ok"] for kind, data in events if kind == "tool_result"] == [False]
    assert not native_command
