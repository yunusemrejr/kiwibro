"""Kiwi Mind 3: the route the model writes before each reply, and the parts the service wires for that pack.

The conductor is driven through a scripted client, as tests/test_mind_service.py does for the 2.x one: no host
process, no network, no llama.cpp. The wiring tests build a small pack on disk and check what ``Mind`` makes of it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import pytest
from mind_pack import make_pack

from kiwibro import routing
from kiwibro.llm import ChatResult, ContextLimitError, LLMError
from kiwibro.mind import KIWIMIND
from kiwibro.mind.client import MindRequestError
from kiwibro.mind.conductor import Conductor, Settings
from kiwibro.mind.conductor3 import Conductor3
from kiwibro.mind.llama import NO_RUNTIME, LlamaRuntime
from kiwibro.mind.service import Mind

SYSTEM = {"role": "system", "content": "s"}
PICTURE = "data:image/png;base64,QUJD"
TOOLS = [{"type": "function", "function": {"name": "read_file",
                                           "parameters": {"type": "object",
                                                          "properties": {"path": {"type": "string"}}}}}]
HELPER = {"id": "big-model", "name": "Big Model", "vision": False}
ROUTE = {"skill": "code", "heavy": False, "act": False, "heavy_share": 0.1, "ms": 1.0}


# ------------------------------------------------------------------------------------------- fakes


class FakeClient:
    """Stands in for LlamaClient: a scripted route, scripted pieces and the request bodies it was given."""

    def __init__(self, route: dict[str, Any] | None = None) -> None:
        self.route_reply = dict(route or ROUTE)
        self.pieces = ["hello"]
        self.tokens = 50
        self.route_calls: list[dict[str, Any]] = []
        self.bodies: list[dict[str, Any]] = []
        self.warm_calls: list[dict[str, Any]] = []
        self.route_error: MindRequestError | None = None
        self.closed = False

    def tokenize(self, text: str) -> int:
        return self.tokens

    def route(self, prompt: str, slot: str, *, skill: str | None = None, weigh: bool = True,
              images: Any = ()) -> dict[str, Any]:
        self.route_calls.append({"prompt": prompt, "slot": slot, "skill": skill, "weigh": weigh,
                                 "images": list(images)})
        if self.route_error is not None:
            raise self.route_error
        answer = dict(self.route_reply)
        if skill:
            answer["skill"] = skill  # a forced skill is the one the grammar allows
        return answer

    def generate(self, body: dict[str, Any], abort: Any = None) -> Iterator[dict[str, Any]]:
        self.bodies.append(body)
        try:
            for piece in self.pieces:
                yield {"text": piece}
            yield {"done": True, "finish": "stop",
                   "usage": {"prompt_tokens": 10, "cached_tokens": 8, "completion_tokens": 3},
                   "timings": {"upper_ms": 1.0, "decode_ms": 2.0, "predicted_per_second": 5.0}}
        finally:
            self.closed = True

    def warm(self, prompt: str, slot: str, skill: str = "none", images: Any = ()) -> dict[str, Any]:
        self.warm_calls.append({"prompt": prompt, "slot": slot, "skill": skill, "images": list(images)})
        return {}


class Helper:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> ChatResult:
        self.calls.append(kwargs)
        kwargs["on_delta"]("helper words")
        return ChatResult(content="helper words", finish_reason="stop")


def conductor(client: FakeClient, *, helper: dict[str, Any] | None = None, helper_stream: Any = None,
              heavy_at: float | None = None, sees: bool = False, ctx: int = 4096) -> Conductor3:
    settings = Settings(lambda: ctx, lambda: helper, helper_stream, lambda: None, lambda: heavy_at, lambda: sees)
    return Conductor3(client, settings)


def ask(cond: Conductor3, text: str = "hi", *, messages: list[dict[str, Any]] | None = None, slot: str = "c1",
        tools: Any = None, events: list | None = None, open_call: bool = False,
        max_tokens: int = 200) -> ChatResult:
    if messages is None:
        messages = [SYSTEM, {"role": "user", "content": text}]
    log = events if events is not None else []
    return cond.stream(messages=messages, tools=tools, temperature=0.5, max_tokens=max_tokens, on_delta=None,
                       abort=None, conversation=slot, events=lambda name, data: log.append((name, data)),
                       open_call=open_call)


def picture_message(text: str = "what is this") -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text},
                                        {"type": "image_url", "image_url": {"url": PICTURE}}]}


# ----------------------------------------------------------------------------------- the first reply


def test_a_first_reply_routes_once_then_writes_under_that_skill() -> None:
    client, events = FakeClient(), []
    result = ask(conductor(client), events=events)
    assert len(client.route_calls) == 1
    assert client.route_calls[0]["skill"] is None and client.route_calls[0]["weigh"] is True
    body = client.bodies[0]
    assert body["skill"] == "code"
    assert body["prompt"].endswith("<think>code light say</think>")
    assert ("brain", {"brain": "kiwimind", "skill": "code", "hard": False}) in events
    assert result.content == "hello"


def test_a_reply_that_acts_is_opened_as_a_call_held_to_the_grammar() -> None:
    client = FakeClient({**ROUTE, "act": True})
    client.pieces = ['{"name": "read_file", "arguments": {"path": "a.txt"}}\n</tool_call>']
    result = ask(conductor(client), tools=TOOLS)
    body = client.bodies[0]
    assert body["prompt"].endswith("</think><tool_call>\n")
    assert isinstance(body["grammar"], str) and body["grammar"].startswith("root ::= call")
    assert "read_file" in body["grammar"]
    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["function"]["name"] == "read_file"
    assert json.loads(result.tool_calls[0]["function"]["arguments"]) == {"path": "a.txt"}
    assert result.finish_reason == "tool_calls"


def test_act_is_ignored_when_no_tool_is_on_offer() -> None:
    client = FakeClient({**ROUTE, "act": True})
    result = ask(conductor(client), tools=None)
    body = client.bodies[0]
    assert "grammar" not in body
    assert body["prompt"].endswith("<think>code light say</think>")
    assert result.content == "hello" and result.tool_calls == []


def test_open_call_opens_the_call_even_when_the_route_says_say() -> None:
    client = FakeClient({**ROUTE, "act": False})
    client.pieces = ['{"name": "read_file", "arguments": {"path": "a.txt"}}\n</tool_call>']
    result = ask(conductor(client), tools=TOOLS, open_call=True)
    assert client.bodies[0]["prompt"].endswith("</think><tool_call>\n")
    assert "grammar" in client.bodies[0]
    assert result.tool_calls[0]["function"]["name"] == "read_file"


# ------------------------------------------------------------------------------------- the steps


def test_a_later_step_keeps_the_first_steps_skill_and_is_not_weighed_again() -> None:
    client = FakeClient()
    cond = conductor(client)
    ask(cond)
    call = {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}}
    ask(cond, messages=[SYSTEM, {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "", "tool_calls": [call]},
                        {"role": "tool", "tool_call_id": "c1", "content": "some text"}], tools=TOOLS)
    assert len(client.route_calls) == 2
    second = client.route_calls[1]
    assert second["skill"] == "code" and second["weigh"] is False
    assert client.bodies[1]["skill"] == "code"


@pytest.mark.parametrize(("tools", "open_call", "action"), [(None, False, "say"), (TOOLS, True, "act")])
def test_a_retry_with_a_fixed_action_reuses_the_skill_without_routing(tools, open_call, action) -> None:
    client = FakeClient()
    cond = conductor(client)
    ask(cond)
    ask(cond, tools=tools, open_call=open_call)
    body = client.bodies[-1]
    assert len(client.route_calls) == 1
    assert body["skill"] == "code"
    assert f"<think>code light {action}</think>" in body["prompt"]
    assert bool(body.get("grammar")) == open_call


def test_a_new_user_message_is_weighed_again() -> None:
    client = FakeClient()
    cond = conductor(client)
    ask(cond)
    ask(cond, messages=[SYSTEM, {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello"},
                        {"role": "user", "content": "and now?"}])
    assert client.route_calls[1]["skill"] is None and client.route_calls[1]["weigh"] is True


# ------------------------------------------------------------------------------------- pictures


def test_a_message_that_brings_a_picture_is_answered_in_words_by_the_see_skill() -> None:
    client = FakeClient()
    ask(conductor(client, sees=True), messages=[SYSTEM, picture_message()], tools=TOOLS)
    assert client.route_calls == []  # the see skill learned to describe: there is no route to choose and no call to open
    body = client.bodies[0]
    assert body["skill"] == "see" and body["images"] == ["QUJD"]
    assert body["prompt"].endswith("<think>see light say</think>")


def test_a_picture_is_dropped_when_neither_the_pack_nor_a_helper_can_see() -> None:
    client = FakeClient()
    ask(conductor(client, sees=False), messages=[SYSTEM, picture_message()])
    assert client.route_calls[0]["skill"] is None and client.route_calls[0]["images"] == []
    body = client.bodies[0]
    assert "images" not in body
    assert "<__media__>" not in body["prompt"]


# --------------------------------------------------------------------------------- the heavy share


def test_a_heavy_share_over_the_mark_hands_the_words_to_the_helper() -> None:
    client, helper = FakeClient({**ROUTE, "heavy_share": 0.9}), Helper()
    result = ask(conductor(client, helper=HELPER, helper_stream=helper, heavy_at=0.5))
    assert helper.calls and client.bodies == []
    assert helper.calls[0]["model"] == "big-model" and helper.calls[0]["tools"] is None
    assert result.content == "helper words"


def test_a_share_under_the_mark_stays_with_kiwi_mind() -> None:
    client, helper = FakeClient({**ROUTE, "heavy_share": 0.2}), Helper()
    ask(conductor(client, helper=HELPER, helper_stream=helper, heavy_at=0.5))
    assert helper.calls == []
    assert client.bodies and client.bodies[0]["skill"] == "code"


def test_without_a_mark_the_routes_own_word_decides() -> None:
    heavy_client, heavy = FakeClient({**ROUTE, "heavy": True}), Helper()
    ask(conductor(heavy_client, helper=HELPER, helper_stream=heavy, heavy_at=None))
    assert heavy.calls and heavy_client.bodies == []
    light_client, light = FakeClient({**ROUTE, "heavy": False}), Helper()
    ask(conductor(light_client, helper=HELPER, helper_stream=light, heavy_at=None))
    assert light.calls == [] and light_client.bodies


def test_a_heavy_request_that_acts_stays_with_kiwi_mind() -> None:
    client, helper = FakeClient({**ROUTE, "heavy": True, "act": True}), Helper()
    ask(conductor(client, helper=HELPER, helper_stream=helper, heavy_at=None), tools=TOOLS)
    assert helper.calls == [] and client.bodies


# ------------------------------------------------------------------------------- remembered routes


def test_an_earlier_replys_route_is_written_before_it_in_the_next_prompt() -> None:
    client = FakeClient()
    cond = conductor(client)
    first = ask(cond)
    ask(cond, messages=[SYSTEM, {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": first.content},
                        {"role": "user", "content": "and now?"}])
    prompt = client.route_calls[1]["prompt"]
    assert "<think>code light say</think>hello" in prompt
    assert "chat light say" not in prompt  # the fallback for a reply whose route is not known


# --------------------------------------------------------------------------- complete and warm


def test_complete_writes_a_plain_answer_under_the_chat_skill() -> None:
    client = FakeClient()
    client.pieces = ["a title"]
    done = conductor(client).complete([{"role": "system", "content": "Name it."},
                                       {"role": "user", "content": "cats"}], max_tokens=50, temperature=0.0)
    body = client.bodies[0]
    assert body["slot"] == "util" and body["skill"] == "chat"
    assert body["prompt"].endswith("<think>chat light say</think>")
    assert done.text == "a title"


def test_complete_reads_a_picture_under_the_see_skill() -> None:
    client = FakeClient()
    conductor(client, sees=True).complete([{"role": "user", "content": picture_message()["content"]}],
                                          max_tokens=50, temperature=0.0)
    body = client.bodies[0]
    assert body["skill"] == "see" and body["images"] == ["QUJD"]
    assert body["prompt"].endswith("<think>see light say</think>")


def test_warm_reads_a_prompt_that_stops_at_the_route_layer() -> None:
    client = FakeClient()
    conductor(client).warm([SYSTEM, {"role": "user", "content": "hi"}], None, "c1")
    assert len(client.warm_calls) == 1
    assert client.warm_calls[0]["prompt"].endswith("<think>")
    assert client.warm_calls[0]["slot"] == "c1"


# ------------------------------------------------------------------------------------ route errors


def test_a_route_that_fails_is_answered_under_the_chat_skill() -> None:
    client = FakeClient()
    client.route_error = MindRequestError("the route layer wrote nothing")
    result = ask(conductor(client))
    assert result.content == "hello"
    assert client.bodies[0]["skill"] == "chat"
    assert client.bodies[0]["prompt"].endswith("<think>chat light say</think>")


def test_a_route_context_error_is_the_agents_context_limit() -> None:
    client = FakeClient()
    client.route_error = MindRequestError("the prompt does not fit", 400, context=True)
    with pytest.raises(ContextLimitError):
        ask(conductor(client))


# --------------------------------------------------------------------------------- the pack on disk


SKILLS3 = ["chat", "web", "code", "terminal", "computer", "tools", "write", "see"]


def make_pack3(root: Path, *, vision: bool = True, quant: str = "Q4_K_M") -> Path:
    """Write a 3.x pack under ``<root>/mind``: GGUF files and the manifest that lists them."""
    pack = root / "mind"
    files: dict[str, dict[str, Any]] = {}

    def put(name: str, data: bytes) -> None:
        path = pack / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    put("kiwimind-trunk.gguf", b"trunk")
    put("route.gguf", b"route")
    for name in SKILLS3:
        put(f"skills/{name}.gguf", b"skill-" + name.encode("ascii"))
    if vision:
        put("vision.gguf", b"vision")
    manifest: dict[str, Any] = {
        "name": "kiwimind", "engine": "llama", "version": "3.0.0", "ctx": 8192, "quant": quant, "skills": SKILLS3,
        "route": {"file": "route.gguf", "heavy_at": 0.62}, "files": files,
    }
    if vision:
        manifest["vision"] = {"file": "vision.gguf", "tokens": 256}
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return pack


def runtime() -> LlamaRuntime:
    return LlamaRuntime("/bin/true", frozenset(), None, {})


def mind_for(root: Path, *, runtime: Any = runtime) -> Mind:
    return Mind(root, helper=lambda: None, helper_stream=lambda **_kwargs: ChatResult(), runtime=runtime)


def test_a_3x_pack_is_served_by_llama_cpp_with_the_3_conductor(tmp_path: Path) -> None:
    make_pack3(tmp_path)
    mind = mind_for(tmp_path)
    try:
        assert mind.engine == "llama"
        assert isinstance(mind.conductor, Conductor3)
        assert mind.usable() is True
        assert mind.sees() is True
        assert mind.heavy_at() == 0.62
        info = mind.model_info()
        assert info["vision"] is True and info["quantization"] == "Q4_K_M"
    finally:
        mind.stop()


def test_without_llama_cpp_the_mind_says_so(tmp_path: Path) -> None:
    make_pack3(tmp_path)
    mind = mind_for(tmp_path, runtime=lambda: None)
    try:
        assert mind.reason() == NO_RUNTIME
        assert mind.usable() is False
        described = mind.describe()
        assert described["engine"] == "llama" and described["vision"] is True and described["node"] is None
    finally:
        mind.stop()


def test_a_pack_without_vision_layers_cannot_see(tmp_path: Path) -> None:
    make_pack3(tmp_path, vision=False)
    mind = mind_for(tmp_path)
    try:
        assert mind.engine == "llama" and mind.sees() is False
        assert mind.model_info()["vision"] is False
    finally:
        mind.stop()


def test_a_2x_pack_keeps_the_engine_and_conductor_it_had(tmp_path: Path) -> None:
    make_pack(tmp_path)
    mind = mind_for(tmp_path)
    try:
        assert mind.engine == "wasm"
        assert type(mind.conductor) is Conductor
    finally:
        mind.stop()


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("error_kind", ["connection", "stream"])
def test_spark_failure_switches_to_legacy_only_before_any_text(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, partial: bool, error_kind: str) -> None:
    primary = make_pack3(tmp_path / "app")
    make_pack(tmp_path / "source", version="2.0.1").rename(primary / "legacy")
    mind = Mind(tmp_path / "app", helper=lambda: None, helper_stream=lambda **_: ChatResult(),
                runtime=runtime, node="/bin/true")
    shown: list[str] = []

    def fail(**kwargs: Any) -> ChatResult:
        if partial:
            kwargs["on_delta"]("Started.")
        raise LLMError(error_kind, "engine stopped")

    def succeed(**kwargs: Any) -> ChatResult:
        kwargs["on_delta"]("Legacy.")
        return ChatResult(content="Legacy.")

    monkeypatch.setattr(mind.conductor, "stream", fail)
    monkeypatch.setattr(mind.legacy_conductor, "stream", succeed)
    kwargs = {"messages": [{"role": "user", "content": "Hello."}], "on_delta": shown.append}
    try:
        assert mind.describe()["active"] == "primary"
        if partial:
            with pytest.raises(LLMError, match="engine stopped"):
                mind.stream(**kwargs)
            assert shown == ["Started."] and mind.describe()["active"] == "primary"
        else:
            assert mind.stream(**kwargs).content == "Legacy."
            assert shown == ["Legacy."] and mind.describe()["active"] == "legacy"
    finally:
        mind.stop()


def test_mind_sees_reads_the_vision_flag_of_the_model_record() -> None:
    class LLM:
        def __init__(self, record: Any) -> None:
            self.record = record
            self.asked: list[str] = []

        def model_info(self, model: str) -> Any:
            self.asked.append(model)
            return self.record

    seeing = LLM({"id": KIWIMIND, "vision": True})
    assert routing._mind_sees(seeing) is True and seeing.asked == [KIWIMIND]
    assert routing._mind_sees(LLM({"vision": False})) is False
    assert routing._mind_sees(LLM(None)) is False
    assert routing._mind_sees(object()) is False
