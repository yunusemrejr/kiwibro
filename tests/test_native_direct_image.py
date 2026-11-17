"""The native request can carry one uploaded image without a separate caption."""

from io import BytesIO

import pytest
from PIL import Image

from kiwibro import delegate
from kiwibro.agent import DIRECT_IMAGE_ENV, DIRECT_IMAGE_TOKENS
from kiwibro.llm import ADAPTER_TASK, ChatResult, ContextLimitError
from kiwibro.mind import KIWIMIND
import test_backend
from test_backend import Env, chat_events, collect_sse, tool_call_result
from test_native_prompt import INFO


env = test_backend.env
VISION = {**INFO, "vision": True, "arch": "gemma4", "version": "gemma4-fixture", "size_bytes": 100}
REQUEST = "Read the attached image and explain how to encode its status in HTML."


@pytest.fixture
def direct(env: Env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(DIRECT_IMAGE_ENV, "1")
    monkeypatch.setattr(env.svc.llm, "resolve_model", lambda _requested=None: KIWIMIND)
    monkeypatch.setattr(env.svc.llm, "model_info", lambda _model: VISION)
    monkeypatch.setattr(env.svc.llm, "has_adapters", lambda _model: True)
    captions = []

    def describe(_llm, _workspace, item, **_kwargs):
        captions.append(item["path"])
        return {"model": KIWIMIND, "text": "The stored caption."}

    monkeypatch.setattr(delegate, "see_attachment", describe)
    env.settings_patch({"memory_enabled": False})
    return captions


def upload(env: Env, content: bytes | None = None) -> dict:
    if content is None:
        image = BytesIO()
        Image.new("RGB", (80, 60), "navy").save(image, format="PNG")
        content = image.getvalue()
    response = env.client.post("/api/attachments", data={"file": (BytesIO(content), "snapshot.png")})
    assert response.status_code == 200
    return response.get_json()["attachment"]


def image_chat(env: Env, attachments: list[dict], conversation_id: str | None = None):
    response = env.client.post("/api/chat", json={
        "message": REQUEST, "attachments": attachments, "conversation_id": conversation_id, "model": KIWIMIND,
    }, buffered=False)
    assert response.status_code == 200
    return collect_sse(response)[0]


def image_parts(call: dict) -> list[list[dict]]:
    return [row["content"] for row in call["messages"] if isinstance(row.get("content"), list)]


def test_native_image_stays_on_the_turn_and_reuses_tools_and_adapter(env: Env, direct, monkeypatch: pytest.MonkeyPatch):
    attachment = upload(env)
    (env.workspace / "state.txt").write_text("Ready\n")
    preparation = env.svc.agent.context.prepare
    reserves = []

    def prepare(*args, **kwargs):
        if kwargs.get("anchor_user_id"):
            reserves.append(kwargs["reserve_tokens"])
        return preparation(*args, **kwargs)

    monkeypatch.setattr(env.svc.agent.context, "prepare", prepare)
    env.fake.stream_script = [
        tool_call_result("read_file", {"path": "state.txt"}, "read"),
        ChatResult(content="The status is Ready."),
    ]
    events = image_chat(env, [attachment])
    assert events[0][0] == "meta" and events[-1][0] == "done"
    conversation_id = events[0][1]["conversation_id"]
    first, second = env.fake.stream_calls
    assert first["model"] == second["model"] == KIWIMIND and first["tools"] == second["tools"]
    assert first["extra"][ADAPTER_TASK] == second["extra"][ADAPTER_TASK] == "code"
    assert reserves[0] == env.svc.agent._tool_tokens(KIWIMIND, first["tools"], VISION) + DIRECT_IMAGE_TOKENS
    assert image_parts(first) == image_parts(second)
    (parts,) = image_parts(first)
    assert parts[0]["type"] == "text" and REQUEST in parts[0]["text"]
    assert "The uploaded image is included in this request." in parts[0]["text"]
    assert "a vision model or by code" not in parts[0]["text"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert not direct
    assert "base64," not in str(env.svc.db.list_messages(conversation_id))
    env.fake.stream_script = [ChatResult(content="Here is the next answer.")]
    chat_events(env, "Say hello.", conversation_id=conversation_id)
    followup = env.fake.stream_calls[-1]
    assert not image_parts(followup)
    assert any(attachment["path"] in row.get("content", "") for row in followup["messages"])


def test_native_image_reinjects_on_the_authored_row_after_steering_and_context_recovery(
        env: Env, direct, monkeypatch: pytest.MonkeyPatch,
):
    attachment = upload(env)
    conversation_id = env.svc.db.create_conversation()["id"]
    env.svc.db.add_message(conversation_id, "user", "An older use of the same file.", meta={"attachments": [attachment]})
    env.svc.db.add_message(conversation_id, "assistant", "Earlier image discussion.")
    for index in range(6):
        env.svc.db.add_message(conversation_id, "user", f"Older prompt {index}")
        env.svc.db.add_message(conversation_id, "assistant", f"Older answer {index}")
    (env.workspace / "state.txt").write_text("Ready\n")
    note = "Keep the current answer brief."

    def read_and_steer():
        assert env.svc.agent.steer(conversation_id, note)
        return tool_call_result("read_file", {"path": "state.txt"}, "read")

    monkeypatch.setattr(env.svc.agent.context, "recover_aggressive", lambda *_args, **_kwargs: False)
    env.fake.stream_script = [read_and_steer, ContextLimitError(), ChatResult(content="Ready.")]
    events = image_chat(env, [attachment], conversation_id)
    assert events[-1][0] == "done" and not direct
    assert any(kind == "steering" and data["text"] == note for kind, data in events)
    first, before_recovery, retry = env.fake.stream_calls
    for call in (first, before_recovery, retry):
        (parts,) = image_parts(call)
        assert REQUEST in parts[0]["text"] and "An older use" not in parts[0]["text"]
        assert parts[1] == image_parts(first)[0][1]
        assert all("_anchor_user" not in row for row in call["messages"])
    assert any(isinstance(row.get("content"), str) and note in row["content"] for row in retry["messages"])
    assert env.svc.db.get_conversation(conversation_id)["summary_through_seq"] > 0


@pytest.mark.parametrize("reason", ["off", "no_projector", "other_model", "multiple", "corrupt"])
def test_native_image_ineligible_requests_keep_caption_path(env: Env, direct, monkeypatch: pytest.MonkeyPatch, reason: str):
    if reason == "off":
        monkeypatch.delenv(DIRECT_IMAGE_ENV)
    if reason in ("no_projector", "other_model"):
        info = {**VISION, "vision": reason != "no_projector", "arch": "other" if reason == "other_model" else "gemma4"}
        monkeypatch.setattr(env.svc.llm, "model_info", lambda _model: info)
    attachments = [upload(env, content=b"not a PNG" if reason == "corrupt" else None)]
    if reason == "multiple":
        attachments.append(upload(env))
    env.fake.stream_script = [ChatResult(content="I read the stored caption.")]
    events = image_chat(env, attachments)
    assert events[-1][0] == "done" and len(direct) == len(attachments)
    assert not image_parts(env.fake.stream_calls[0])
    assert any("a vision model or by code" in row.get("content", "") for row in env.fake.stream_calls[0]["messages"])


def test_native_image_stops_before_dispatch_after_encoding_is_cancelled(env: Env, direct, monkeypatch: pytest.MonkeyPatch):
    conversation_id = env.svc.db.create_conversation()["id"]
    encode = delegate.image_data_url

    def cancel(*args, **kwargs):
        image = encode(*args, **kwargs)
        assert env.svc.agent.stop(conversation_id)
        return image

    monkeypatch.setattr(delegate, "image_data_url", cancel)
    events = image_chat(env, [upload(env)], conversation_id)
    assert events[-1][0] == "done" and events[-1][1]["aborted"]
    assert not env.fake.stream_calls and not direct


def test_native_image_cannot_follow_a_different_pack_midturn(env: Env, direct, monkeypatch: pytest.MonkeyPatch):
    (env.workspace / "state.txt").write_text("Ready\n")
    env.fake.stream_script = [tool_call_result("read_file", {"path": "state.txt"}, "read")]
    monkeypatch.setattr(env.svc.llm, "model_info", lambda _model: (
        {**VISION, "version": "replacement"} if env.fake.stream_calls else VISION
    ))
    events = image_chat(env, [upload(env)])
    assert len(env.fake.stream_calls) == 1 and not direct
    assert any(kind == "error" and data["code"] == "vision" for kind, data in events)
