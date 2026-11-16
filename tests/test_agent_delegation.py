"""Direct requests for an exact helper are enforced by /api/chat."""

from unittest.mock import Mock

import pytest

from kiwibro import delegate
from kiwibro.llm import ChatResult, Completion
from test_backend import Env, MODEL, SSEStream, chat_events, env, tool_call_result  # noqa: F401


NAMED = "local/explicit-reviewer"
OTHER = "local/rapid"
REQUEST = (f"Ask only {NAMED} for a second opinion on the naming rule “Osier-v2”. "
           "If that helper is unavailable, tell me instead of substituting a model.")
QUESTION = "What is your opinion on the naming rule “Osier-v2”?"


@pytest.mark.parametrize("permission", ["auto", "ask"])
def test_unavailable_exact_delegate_cannot_substitute_a_helper(
        env: Env, monkeypatch: pytest.MonkeyPatch, permission: str) -> None:
    env.fake.models.append({**MODEL, "id": OTHER})
    helper = Mock(side_effect=AssertionError("Model delegation should not run"))
    monkeypatch.setattr(delegate, "run_helper", helper)
    env.settings_patch({"autonomy": "supervised", "tool_profile": "full",
                        "permissions": {"model_info": "auto", "delegate": permission}})
    env.fake.stream_script = [
        tool_call_result("model_info", {}),
        tool_call_result("delegate", {"task": QUESTION, "model": OTHER}, call_id="call_2"),
        ChatResult(content=f"{NAMED} is unavailable."),
    ]

    response = env.client.post("/api/chat", json={"message": REQUEST}, buffered=False)
    stream = SSEStream(response).start()
    events = stream.finish(timeout=4)
    assert "approval" not in stream.names() and events[-1][0] == "done"
    call = next(body for event, body in events if event == "tool_call" and body["name"] == "delegate")
    assert call["requires_approval"] is False
    result = next(body for event, body in events if event == "tool_result" and body["name"] == "delegate")
    assert not result["ok"] and NAMED in result["preview"]
    observation = env.fake.stream_calls[2]["messages"][-1]
    assert observation["role"] == "tool" and f"restricted to {NAMED}" in observation["content"]
    helper.assert_not_called()


def test_omitted_delegate_model_is_pinned_before_approval(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    env.fake.models.extend([{**MODEL, "id": OTHER}, {**MODEL, "id": NAMED}])
    completed = Mock(return_value=Completion("The naming rule is clear."))
    monkeypatch.setattr(env.svc.llm, "complete", completed)
    env.settings_patch({"autonomy": "supervised", "tool_profile": "full", "permissions": {"delegate": "ask"}})
    env.fake.stream_script = [
        tool_call_result("delegate", {"task": QUESTION}),
        ChatResult(content="The requested reviewer says the rule is clear."),
    ]

    message = f"Could you use the exact helper `{NAMED}` to review the naming rule “Osier-v2”?"
    response = env.client.post("/api/chat", json={"message": message}, buffered=False)
    stream = SSEStream(response).start()
    approval = stream.wait_for("approval", timeout=4)
    shown = stream.wait_for("tool_call", timeout=4)
    assert approval["tool"] == "delegate" and approval["args"]["model"] == NAMED
    assert shown["args"]["model"] == NAMED and shown["requires_approval"]
    assert all(call.kwargs["model"] != NAMED for call in completed.call_args_list)
    accepted = env.new_client().post(f"/api/approvals/{approval['approval_id']}", json={"decision": "allow"})
    assert accepted.status_code == 200
    events = stream.finish(timeout=4)
    result = next(body for event, body in events if event == "tool_result" and body["name"] == "delegate")
    assert result["ok"]
    assert [call.kwargs["model"] for call in completed.call_args_list if call.kwargs["model"].startswith("local/")] == [NAMED]


@pytest.mark.parametrize(("message", "model"), [
    ("Ask the model for a second opinion on the naming rule “Osier-v2”.", None),
    (f"Ask {NAMED} for a second opinion. If that helper is unavailable, use another available local model.", OTHER),
    (f"Ask {NAMED} for a second opinion. Otherwise, use {OTHER}.", OTHER),
    (f"Ask {NAMED} for a second opinion, but use {OTHER} if unavailable.", OTHER),
    (f"Ask {NAMED} for a second opinion. You may use another model.", OTHER),
])
def test_helper_delegation_still_works_when_user_allows_selection(
        env: Env, monkeypatch: pytest.MonkeyPatch, message: str, model: str | None) -> None:
    env.fake.models.append({**MODEL, "id": OTHER})
    helper = Mock(return_value={"model": OTHER, "text": "The naming rule is clear."})
    monkeypatch.setattr(delegate, "run_helper", helper)
    env.settings_patch({"autonomy": "supervised", "tool_profile": "full", "permissions": {"delegate": "auto"}})
    env.fake.stream_script = [
        tool_call_result("delegate", {"task": QUESTION, "model": model}),
        ChatResult(content="The reviewer says the rule is clear."),
    ]

    events, _ = chat_events(env, message)
    result = next(body for event, body in events if event == "tool_result" and body["name"] == "delegate")
    assert result["ok"]
    assert helper.call_args.kwargs["model"] == model
