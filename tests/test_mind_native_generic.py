from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from kiwibro.llm import LLMClient, LLMError, TurnAborted
from kiwibro.mind.native_generic import NATIVE_FORMAT, NATIVE_MODEL, NativeConductor, describe_pack
from kiwibro.mind.pack import MANIFEST, TRUNK3, Pack, PackError

MESSAGES = [{"role": "user", "content": "Read the README."}]
TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
VISION = [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
    {"type": "text", "text": "Describe this."},
]}]


def pack_at(tmp_path: Path, *, base: str = "Qwen/Qwen3.8-27B", vision: bool = True,
            effort: bool = True) -> Pack:
    pack = Pack(tmp_path, subdir="candidate")
    pack.dir.mkdir(parents=True)
    payloads = {TRUNK3: b"model"}
    if vision:
        payloads["vision.gguf"] = b"projector"
    for name, payload in payloads.items():
        pack.path(name).write_bytes(payload)
    manifest = {"engine": "llama", "format": NATIVE_FORMAT, "base": base, "ctx": 8192, "quant": "Q4_K_M",
                "capabilities": {"tools": True, "thinking": True, "reasoning_effort": effort},
                "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in payloads.items()}}
    if vision:
        manifest["vision"] = {"file": "vision.gguf"}
    pack.path(MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return pack


class Reply:
    status_code = 200

    def __init__(self, *chunks: dict[str, Any]) -> None:
        self.chunks = chunks

    def __enter__(self) -> Reply:
        return self

    def __exit__(self, *_exc: Any) -> None:
        pass

    def iter_lines(self, *, decode_unicode: bool) -> Any:
        yield from (b"data: " + json.dumps(chunk).encode() for chunk in self.chunks)
        yield b"data: [DONE]"


def wire(monkeypatch: pytest.MonkeyPatch, reply: Reply) -> tuple[LLMClient, list[dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []

    def post(url: str, *, json: dict[str, Any], **_kwargs: Any) -> Reply:
        assert url == "http://127.0.0.1:9999/v1/chat/completions"
        payloads.append(json)
        return reply

    monkeypatch.setattr("kiwibro.llm.requests.post", post)
    return LLMClient(lambda: {"server_url": "http://127.0.0.1:9999"}), payloads


def text_reply(text: str) -> Reply:
    return Reply({"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": "stop"}]})


def test_native_manifest_rejects_wrong_format_and_damaged_weights(tmp_path: Path) -> None:
    pack = pack_at(tmp_path)
    described = describe_pack(pack)
    assert described.model.id == NATIVE_MODEL and described.model.tools and described.model.thinking
    assert described.model.kind == "vlm" and not described.model.deletable
    assert described.model.size_bytes == len(b"modelprojector")
    pack.path(TRUNK3).write_bytes(b"stale")
    with pytest.raises(PackError, match="damaged"):
        describe_pack(pack)
    manifest = pack.manifest()
    del manifest["format"]
    pack.path(MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PackError, match="not a native"):
        describe_pack(pack)


def test_wire_forces_a_required_native_tool_and_parses_streamed_arguments(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reply = Reply(
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_a", "function": {"name": "read_file", "arguments": '{"path":'}},
        ]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '"README.md"}'}},
        ]}, "finish_reason": "tool_calls"}]},
    )
    client, payloads = wire(monkeypatch, reply)
    conductor = NativeConductor(client, describe_pack(pack_at(tmp_path)))
    result = conductor.stream(messages=MESSAGES, tools=TOOLS, temperature=0, max_tokens=100, open_call=True)
    body = payloads[0]
    assert body["tool_choice"] == "required" and body["parallel_tool_calls"] is False
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["tools"] == TOOLS and result.finish_reason == "tool_calls"
    call = result.tool_calls[0]
    assert call["id"] == "call_a" and call["function"]["name"] == "read_file"
    assert json.loads(call["function"]["arguments"]) == {"path": "README.md"}


def test_failed_required_call_does_not_display_an_answer(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _payloads = wire(monkeypatch, text_reply("I read it."))
    conductor = NativeConductor(client, describe_pack(pack_at(tmp_path)))
    visible: list[str] = []
    with pytest.raises(LLMError, match="required tool call"):
        conductor.stream(messages=MESSAGES, tools=TOOLS, temperature=0, max_tokens=100,
                         open_call=True, on_delta=visible.append)
    assert not visible


@pytest.mark.parametrize("choice", ["none", {"type": "function", "function": {"name": "read_file"}}])
def test_a_tool_rejected_by_the_callers_choice_never_leaves_native_transport(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, choice: str | dict[str, Any]) -> None:
    reply = Reply({"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "call_a", "function": {"name": "list_files", "arguments": "{}"}},
    ]}, "finish_reason": "tool_calls"}]})
    client, payloads = wire(monkeypatch, reply)
    conductor = NativeConductor(client, describe_pack(pack_at(tmp_path)))
    with pytest.raises(LLMError) as error:
        conductor.stream(messages=MESSAGES, tools=TOOLS, temperature=0, max_tokens=100,
                         extra={"tool_choice": choice})
    assert error.value.kind == "tools" and payloads[0]["tool_choice"] == choice


@pytest.mark.parametrize(("base", "effort"), [("Qwen/Qwen3.8-27B", True), ("google/gemma-4-26B-A4B-it", False)])
def test_vision_and_reasoning_options_use_the_native_template(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base: str, effort: bool) -> None:
    client, payloads = wire(monkeypatch, text_reply("An image."))
    conductor = NativeConductor(client, describe_pack(pack_at(tmp_path, base=base, effort=effort)))
    shown: list[str] = []
    result = conductor.stream(messages=VISION, tools=None, temperature=0, max_tokens=100,
                              on_delta=shown.append, extra={"reasoning_effort": "medium"})
    assert result.content == "An image." and shown == ["An image."]
    assert payloads[0]["messages"] == VISION
    assert payloads[0]["chat_template_kwargs"] == {
        "enable_thinking": True, **({"reasoning_effort": "medium"} if effort else {}),
    }


def test_text_only_and_cancelled_requests_do_not_contact_the_runtime(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, payloads = wire(monkeypatch, text_reply("unused"))
    conductor = NativeConductor(client, describe_pack(pack_at(tmp_path, vision=False)))
    with pytest.raises(LLMError, match="vision projector"):
        conductor.stream(messages=VISION, tools=None, temperature=0, max_tokens=100)
    stop = threading.Event()
    stop.set()
    with pytest.raises(TurnAborted):
        conductor.complete(MESSAGES, temperature=0, max_tokens=100, abort=stop)
    assert not payloads


def test_native_utility_requests_stream_schema_and_turn_off_thinking(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, payloads = wire(monkeypatch, text_reply('{"ops": []}'))
    conductor = NativeConductor(client, describe_pack(pack_at(tmp_path)))
    schema = {"type": "json_schema", "json_schema": {"name": "memory", "schema": {"type": "object"}}}
    answer = conductor.complete(MESSAGES, temperature=0, max_tokens=100, response_format=schema)
    assert json.loads(answer.text) == {"ops": []}
    body = payloads[0]
    assert body["stream"] is True and body["response_format"] == schema
    assert body["reasoning_effort"] == "none" and body["chat_template_kwargs"]["enable_thinking"] is False


def test_public_chat_prefill_and_memory_share_the_authenticated_local_engine(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from kiwibro.llm import bind_turn
    from kiwibro.mind import KIWIMIND
    from kiwibro.mind.little import background

    monkeypatch.setenv("KIWIMIND_CANDIDATE", "1")
    monkeypatch.setenv("KIWIBRO_SERVER_KEY", "external-auth-must-not-leak")
    pack_at(tmp_path, base="google/gemma-4-26B-A4B-it", effort=False)
    client = LLMClient(lambda: {"server_url": "https://outside.example", "model": "explicit-helper"}, data_dir=tmp_path)
    engine = client.direct_engine()
    mind = client.mind()
    monkeypatch.setattr(mind, "runtime", lambda: object())
    calls: list[tuple[str, dict[str, Any]]] = []
    acquired: list[str] = []

    def acquire(model_id: str) -> Any:
        acquired.append(model_id)
        assert engine.find(model_id).id == NATIVE_MODEL
        return SimpleNamespace(url="http://127.0.0.1:9999")

    def post(url: str, *, json: dict[str, Any], headers: dict[str, str], **_kwargs: Any) -> Reply:
        assert headers["Authorization"] == f"Bearer {engine.api_key}"
        calls.append((url, json))
        return text_reply('{"ops": []}' if "response_format" in json else "Native answer.")

    monkeypatch.setattr(engine, "acquire", acquire)
    monkeypatch.setattr(engine, "release", lambda server: None)
    monkeypatch.setattr("kiwibro.llm.requests.post", post)

    events: list[tuple[str, dict[str, Any]]] = []
    bind_turn("chat1", lambda name, detail: events.append((name, detail)))
    assert client.stream_chat(model=KIWIMIND, messages=VISION, max_tokens=96).content == "Native answer."
    client.prefill(model=KIWIMIND, messages=MESSAGES)
    with background("memory"):
        assert client.complete_json(model=KIWIMIND, messages=MESSAGES) == {"ops": []}

    assert acquired == [NATIVE_MODEL] * 3
    assert all(url == "http://127.0.0.1:9999/v1/chat/completions" for url, _body in calls)
    assert calls[0][1]["messages"] == VISION
    assert calls[1][1]["max_tokens"] == 1
    assert calls[2][1]["response_format"]["json_schema"]["name"] == "memory_operations"
    assert events == [("brain", {"brain": "kiwimind", "skill": None, "hard": False})]
    assert not mind.host.alive() and not mind.legacy_host.alive() and not mind.little.alive()
    assert mind.describe()["active"] == "candidate"


def test_candidate_mode_requires_its_pack_without_exposing_old_downloads(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mind_pack import make_pack

    from kiwibro.mind import KIWIMIND

    monkeypatch.setenv("KIWIMIND_CANDIDATE", "1")
    make_pack(tmp_path)
    client = LLMClient(lambda: {}, data_dir=tmp_path)
    mind = client.mind()
    assert not mind.usable() and mind.pack.check().ok
    assert client.direct_engine().find(NATIVE_MODEL) is None
    assert mind.download_pack("primary") is mind.candidate_pack and mind.download_pack("legacy") is None
    with pytest.raises(LLMError, match="not installed"):
        client.resolve_model()
    with pytest.raises(LLMError, match="native Kiwi Mind"):
        client.stream_chat(model=KIWIMIND, messages=MESSAGES, max_tokens=100)
    assert not mind.host.alive()


@pytest.mark.parametrize(("active", "exposed"), [(4, True), (33, False), (True, False)])
def test_candidate_status_exposes_only_validated_architecture_counts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, active: int, exposed: bool) -> None:
    monkeypatch.setenv("KIWIMIND_CANDIDATE", "1")
    pack = pack_at(tmp_path, base="google/gemma-4-26B-A4B-it", effort=False)
    manifest = pack.manifest()
    manifest["arch"] = "gemma4"
    summary = {"experts_before": 128, "experts_after": 32, "top_k_before": 8, "top_k_after": active}
    manifest["expert_bank"] = {**summary, "kept_original_experts": {"0": [4, 12]}}
    pack.path(MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")

    mind = LLMClient(lambda: {}, data_dir=tmp_path).mind()
    monkeypatch.setattr(mind, "runtime", lambda: object())
    status = mind.describe()
    assert status["base"] == "google/gemma-4-26B-A4B-it" and status["architecture"] == "gemma4"
    assert status["quant"] == "Q4_K_M" and status["ctx"] == 8192 and status["vision"]
    assert status["capabilities"] == {"tools": True, "thinking": True, "reasoning_effort": False}
    assert status["expert_bank"] == (summary if exposed else None)

    pack.path(TRUNK3).write_bytes(b"stale")
    status = mind.describe()
    assert not status["usable"] and not status["vision"]
    assert "expert_bank" not in status and "base" not in status


def test_candidate_reserves_one_full_local_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIWIMIND_CANDIDATE", "1")
    pack_at(tmp_path, base="google/gemma-4-26B-A4B-it", effort=False)
    engine = LLMClient(lambda: {}, data_dir=tmp_path).direct_engine()
    monkeypatch.setattr(engine, "_start", lambda model, context, slots: (model.id, context, slots))
    assert engine.ensure(NATIVE_MODEL) == (NATIVE_MODEL, engine.find(NATIVE_MODEL).context_length, 1)
