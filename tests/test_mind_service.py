"""Kiwi Mind inside Kiwibro: pack, host process, conductor rules, routing and the context-note layout."""

from __future__ import annotations

import hashlib
import http.server
import importlib.util
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import pytest
from mind_pack import HEAD_NAMES, ROOT, make_pack, node_path

from kiwibro import context, inference, ledger
from kiwibro.llm import ChatResult, ContextLimitError, LLMClient, LLMError, TurnAborted, bind_turn
from kiwibro.mind import KIWIMIND, render
from kiwibro.mind.client import HostGone, MindRequestError
from kiwibro.mind.conductor import Conductor, Holdback, Settings
from kiwibro.mind.pack import Pack

needs_node = pytest.mark.skipif(node_path() is None, reason="node is not installed")


# ---------------------------------------------------------------------------------------------- fakes


class FakeClient:
    """Stands in for MindClient: scripted control logits and generated pieces."""

    def __init__(self) -> None:
        self.logits = {name: 0.0 for name in HEAD_NAMES}
        self.pieces: list[str] = ["Hello."]
        self.tokens = 50
        self.control_calls = 0
        self.bodies: list[dict[str, Any]] = []
        self.closed = False
        self.control_error: MindRequestError | None = None

    def tokenize(self, text: str) -> int:
        return self.tokens

    def control(self, prompt: str, slot: str) -> dict[str, Any]:
        self.control_calls += 1
        if self.control_error:
            raise self.control_error
        return {"names": HEAD_NAMES, "logits": [self.logits[name] for name in HEAD_NAMES]}

    def generate(self, body: dict[str, Any], abort: Any = None) -> Iterator[dict[str, Any]]:
        self.bodies.append(body)
        try:
            for piece in self.pieces:
                yield {"text": piece}
            yield {"done": True, "finish": "stop", "usage": {"prompt_tokens": 50, "cached_tokens": 40,
                                                              "completion_tokens": len(self.pieces)},
                   "timings": {"lower_ms": 5, "upper_ms": 7, "switch_ms": 0, "decode_ms": 10,
                               "predicted_per_second": 100}}
        finally:
            self.closed = True


def conductor(client: FakeClient, helper: dict[str, Any] | None = None, helper_stream: Any = None,
              ctx: int = 1000) -> Conductor:
    return Conductor(client, Settings(lambda: ctx, lambda: helper, helper_stream))  # type: ignore[arg-type]


def ask(cond: Conductor, text: str = "hi", *, slot: str = "c1", events: list | None = None, on_delta: Any = None,
        abort: Any = None, extra_messages: list | None = None, tools: Any = None, temperature: float = 0.6,
        max_tokens: int = 400) -> ChatResult:
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": text}, *(extra_messages or [])]
    log = events if events is not None else []
    return cond.stream(messages=messages, tools=tools, temperature=temperature, max_tokens=max_tokens,
                       on_delta=on_delta, abort=abort, conversation=slot, events=lambda n, d: log.append((n, d)))


# ---------------------------------------------------------------------------------- skill choice rules


def test_first_message_takes_the_top_skill_and_reports_it() -> None:
    client = FakeClient()
    client.logits["code"] = 3.0
    events: list = []
    ask(conductor(client), events=events)
    assert client.bodies[0]["skill"] == "code"
    assert events == [("brain", {"brain": "kiwimind", "skill": "code", "hard": False})]


def test_a_running_conversation_switches_skill_only_for_a_clear_gain() -> None:
    client, cond = FakeClient(), conductor(FakeClient())
    client = cond.client  # type: ignore[assignment]
    client.logits["code"] = 3.0
    ask(cond, "write a function")
    # the web skill is slightly ahead: softmax gap below 0.15, so the conversation stays on code
    client.logits.update({"code": 2.0, "web": 2.3})
    ask(cond, "and now?", extra_messages=[{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}])
    assert [b["skill"] for b in client.bodies] == ["code", "code"]
    # a clear gain switches
    client.logits.update({"code": 0.0, "web": 4.0})
    ask(cond, "x", extra_messages=[{"role": "assistant", "content": "ok"}, {"role": "user", "content": "a"},
                                   {"role": "assistant", "content": "ok"}, {"role": "user", "content": "b"}])
    assert client.bodies[-1]["skill"] == "web"


def test_tool_steps_of_one_turn_keep_skill_and_ask_the_head_once() -> None:
    cond = conductor(FakeClient())
    client = cond.client
    client.logits.update({"terminal": 3.0, "hard": 2.0})  # type: ignore[attr-defined]
    call = {"id": "c", "type": "function", "function": {"name": "x", "arguments": "{}"}}
    ask(cond)
    ask(cond, extra_messages=[{"role": "assistant", "content": "", "tool_calls": [call]},
                              {"role": "tool", "tool_call_id": "c", "content": "done"}])
    assert client.control_calls == 1  # type: ignore[attr-defined]
    assert [b["skill"] for b in client.bodies] == ["terminal", "terminal"]  # type: ignore[attr-defined]


def test_sampling_follows_the_rules() -> None:
    client = FakeClient()
    cond = conductor(client, ctx=100)
    client.tokens = 60
    ask(cond, temperature=1.2, max_tokens=900)
    body = client.bodies[0]
    assert body["temperature"] == 0.7 and body["top_k"] == 20 and body["top_p"] == 0.8
    assert body["repeat_penalty"] == 1.05 and body["stop"] == list(render.STOP)
    assert body["max_tokens"] == 39  # the room left: 100 - 60 - 1
    ask(cond, "again", slot="c2", temperature=0.0)
    assert client.bodies[1]["temperature"] == 0.0


# ----------------------------------------------------------------------- tool calls and the chat text


CALL = '<tool_call>\n{"name": "read_files", "arguments": {"paths": ["a.txt"]}}\n</tool_call>'


def test_a_tool_call_never_reaches_the_chat_as_answer_text() -> None:
    client = FakeClient()
    client.pieces = ["Let me look", " at it.\n<", "tool", "_call>\n", '{"name": "read_files", ',
                     '"arguments": {"paths": ["a.txt"]}}', "\n</tool_call>"]
    shown: list[str] = []
    result = ask(conductor(client), on_delta=shown.append)
    assert "".join(shown) == "Let me look at it.\n"
    assert "<" not in "".join(shown).replace("\n", "")
    assert result.content == "Let me look at it."
    assert result.finish_reason == "tool_calls"
    call = result.tool_calls[0]
    assert call["type"] == "function" and call["id"].startswith("call_")
    assert call["function"]["name"] == "read_files"
    assert json.loads(call["function"]["arguments"]) == {"paths": ["a.txt"]}


def test_text_that_only_looks_like_a_tag_is_released() -> None:
    hold = Holdback()
    out = hold.feed("a <") + hold.feed("b") + hold.feed(" and <tool") + hold.finish()
    assert out == "a <b and <tool"
    hold = Holdback()
    assert hold.feed("x<tool_c") == "x" and hold.feed("all>y") == "" and hold.finish() == ""


def test_result_carries_usage_and_timings_like_the_llama_path() -> None:
    result = ask(conductor(FakeClient()))
    assert result.usage == {"prompt_tokens": 50, "completion_tokens": 1, "total_tokens": 51}
    assert result.timings is not None and result.timings["cache_n"] == 40 and result.timings["prompt_n"] == 10
    assert result.finish_reason == "stop" and result.content == "Hello."


# ------------------------------------------------------------------------------------- helper hand-off


def hard_client() -> FakeClient:
    client = FakeClient()
    client.logits.update({"chat": 2.0, "hard": 1.0})
    client.pieces = [f"word{i} " for i in range(30)]
    return client


class Helper:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = fail

    def __call__(self, **kwargs: Any) -> ChatResult:
        self.calls.append(kwargs)
        if self.fail:
            raise LLMError("engine", "The model could not be loaded.")
        kwargs["on_delta"]("helper words")
        return ChatResult(content="helper words", finish_reason="stop")


HELPER = {"id": "big-model", "name": "Big Model", "vision": False}


def test_a_hard_question_goes_to_the_helper_without_tools() -> None:
    client, helper, events, shown = hard_client(), Helper(), [], []
    result = ask(conductor(client, HELPER, helper), events=events, on_delta=shown.append,
                 tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}])
    assert result.content == "helper words" and shown == ["helper words"]  # Kiwi Mind's first words were held back
    assert helper.calls[0]["tools"] is None and helper.calls[0]["model"] == "big-model"
    assert [e[1]["brain"] for e in events] == ["kiwimind", "helper"]
    assert events[1][1]["name"] == "Big Model" and events[1][1]["hard"] is True
    assert client.closed  # Kiwi Mind's own generation was stopped


def test_no_hand_off_when_the_reply_opens_a_tool_call() -> None:
    client, helper = hard_client(), Helper()
    client.pieces = ["<tool", "_call>\n", '{"name": "a", "arguments": {}}', "\n</tool_call>", *["x"] * 10]
    result = ask(conductor(client, HELPER, helper))
    assert helper.calls == [] and result.tool_calls[0]["function"]["name"] == "a"


def test_no_hand_off_without_a_helper_or_when_not_hard() -> None:
    helper = Helper()
    shown: list[str] = []
    ask(conductor(hard_client(), None, helper), on_delta=shown.append)
    assert helper.calls == [] and "".join(shown).startswith("word0 word1")
    client = hard_client()
    client.logits["hard"] = -1.0
    ask(conductor(client, HELPER, helper))
    assert helper.calls == []


def test_a_failing_helper_leaves_the_answer_to_kiwi_mind_and_says_so() -> None:
    client, events, shown = hard_client(), [], []
    helper = Helper(fail=True)
    result = ask(conductor(client, HELPER, helper), events=events, on_delta=shown.append)
    assert len(helper.calls) == 1
    assert result.content.startswith("word0 word1") and "".join(shown).startswith("word0 word1")
    assert ("status" in [e[0] for e in events]) and events[-1][1]["brain"] == "kiwimind"
    assert "Kiwi Mind is answering" in [e[1] for e in events if e[0] == "status"][0]["label"]


def test_a_turn_with_an_image_goes_to_a_helper_that_sees() -> None:
    helper = Helper()
    seeing = {**HELPER, "vision": True}
    messages = [{"role": "user", "content": [{"type": "text", "text": "what is this"},
                                             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]
    cond = conductor(FakeClient(), seeing, helper)
    result = cond.stream(messages=messages, tools=[], temperature=0.5, max_tokens=50, on_delta=None, abort=None,
                         conversation="c", events=None)
    assert result.content == "helper words" and helper.calls[0]["tools"] is None


# --------------------------------------------------------------------------------- errors and abort


def test_a_prompt_that_does_not_fit_is_the_agents_context_error() -> None:
    client = FakeClient()
    client.tokens = 2000
    with pytest.raises(ContextLimitError):
        ask(conductor(client, ctx=1000))
    client = FakeClient()
    client.control_error = MindRequestError("the prompt has 2000 tokens but the context holds 1000", 400, context=True)
    with pytest.raises(ContextLimitError):
        ask(conductor(client))


def test_abort_stops_the_generation() -> None:
    client = FakeClient()
    client.pieces = ["a"] * 50
    abort = threading.Event()

    def stop_after_three(_text: str) -> None:
        if len(client.bodies) and not abort.is_set():
            abort.set()

    with pytest.raises(TurnAborted):
        ask(conductor(client), abort=abort, on_delta=stop_after_three)
    assert client.closed


def test_a_dead_host_is_a_connection_error() -> None:
    class Dead(FakeClient):
        def tokenize(self, text: str) -> int:
            raise HostGone("Kiwi Mind stopped answering. Try again.")

    with pytest.raises(LLMError) as raised:
        ask(conductor(Dead()))
    assert raised.value.kind == "connection"


# ------------------------------------------------------------------------------- LLMClient routing


def llm_with_mind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, usable: bool, helper: dict | None,
                  fake: FakeClient | None = None) -> LLMClient:
    settings = {"server_url": "http://127.0.0.1:9", "model": helper["id"] if helper else None}
    client = LLMClient(lambda: settings, data_dir=tmp_path)
    mind = client.mind()
    monkeypatch.setattr(type(mind), "usable", lambda self: usable)
    monkeypatch.setattr(type(mind), "reason", lambda self: "" if usable else "Kiwi Mind is not installed.")
    monkeypatch.setattr(client, "helper", lambda: helper)
    mind.helper = lambda: helper
    if fake is not None:
        mind.conductor.client = fake  # type: ignore[assignment]
        mind.conductor.cfg = Settings(lambda: 1000, lambda: helper, client.stream_chat)
    return client


def test_kiwi_mind_leads_every_turn_when_usable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = llm_with_mind(tmp_path, monkeypatch, usable=True, helper=None)
    assert client.resolve_model("anything") == KIWIMIND
    info = client.model_info(KIWIMIND)
    assert info is not None and info["context"] == 8192 and info["tools"] is True and info["vision"] is False
    assert not inference.model_profile(KIWIMIND, info).reasoning  # no thinking, no budget


def test_not_usable_says_so_once_and_the_selected_model_answers(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    helper = {"id": "big-model", "name": "Big Model"}
    client = llm_with_mind(tmp_path, monkeypatch, usable=False, helper=helper)
    monkeypatch.setattr(client, "models_status", lambda force=False: (True, [helper], None))
    monkeypatch.setattr(client, "model_info", lambda model: helper)
    assert client.resolve_model() == "big-model"
    events: list = []
    client._announce_helper("big-model", lambda n, d: events.append((n, d)))
    client._announce_helper("big-model", lambda n, d: events.append((n, d)))
    labels = [d["label"] for n, d in events if n == "status"]
    assert labels == ["Kiwi Mind is not installed: Big Model is answering."]
    assert [d["brain"] for n, d in events if n == "brain"] == ["helper", "helper"]


def test_with_neither_the_no_model_message_offers_kiwi_mind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = llm_with_mind(tmp_path, monkeypatch, usable=False, helper=None)
    monkeypatch.setattr(client, "models_status", lambda force=False: (True, [], None))
    monkeypatch.setattr(client, "direct_on", lambda: True)
    with pytest.raises(LLMError) as raised:
        client.resolve_model()
    assert raised.value.kind == "no_model" and raised.value.message.startswith("Kiwi Mind is not installed.")


def test_utility_calls_use_kiwi_mind_plain_without_a_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeClient()
    fake.pieces = ["a title"]
    client = llm_with_mind(tmp_path, monkeypatch, usable=True, helper=None, fake=fake)
    done = client.complete(model=KIWIMIND, messages=[{"role": "system", "content": "Name it."},
                                                      {"role": "user", "content": "chat about cats"}])
    body = fake.bodies[0]
    assert done.text == "a title" and body["slot"] == "util" and body["skill"] == "none"
    assert body["prompt"] == render.render_plain([{"role": "system", "content": "Name it."},
                                                  {"role": "user", "content": "chat about cats"}])
    assert "Tools:" not in body["prompt"]


def test_utility_calls_use_the_helper_when_one_is_selected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = {"id": "big-model", "name": "Big Model"}
    fake = FakeClient()
    client = llm_with_mind(tmp_path, monkeypatch, usable=True, helper=helper, fake=fake)
    seen: dict[str, Any] = {}

    class Reply:
        status_code = 200

        @staticmethod
        def json() -> dict[str, Any]:
            return {"choices": [{"message": {"content": "from helper"}, "finish_reason": "stop"}]}

    def post(url: str, json: dict[str, Any], **_kw: Any) -> Reply:  # noqa: A002
        seen.update(url=url, model=json["model"])
        return Reply()

    monkeypatch.setattr("kiwibro.llm.requests.post", post)
    done = client.complete(model=KIWIMIND, messages=[{"role": "user", "content": "x"}])
    assert done.text == "from helper" and seen["model"] == "big-model" and fake.bodies == []


# ------------------------------------------------------------------------------------------ the pack


def test_pack_check_is_cached_and_catches_damage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_pack(tmp_path)
    pack = Pack(tmp_path)
    calls = []
    real = hashlib.sha256

    import kiwibro.mind.pack as pack_module

    def counting(path: Path, stop: Any = None) -> str:
        calls.append(path.name)
        return pack_module.__dict__["_real_sha"](path, stop)

    pack_module._real_sha = pack_module._sha256  # type: ignore[attr-defined]
    monkeypatch.setattr(pack_module, "_sha256", counting)
    assert pack.check().ok and len(calls) == 10
    again = Pack(tmp_path)
    assert again.check().ok and len(calls) == 10  # the saved result is trusted while size and time are unchanged
    del real
    trunk = tmp_path / "mind" / "kiwimind-trunk.fkm"
    data = bytearray(trunk.read_bytes())
    data[100] ^= 0xFF
    trunk.write_bytes(bytes(data))
    state = Pack(tmp_path).check()
    assert not state.ok and "damaged" in state.reason
    trunk.unlink()
    assert "missing" in Pack(tmp_path).check().reason


class FakeHub:
    """A tiny local server for the pack files with Range support."""

    def __init__(self, files: dict[str, bytes], *, break_after: int | None = None) -> None:
        self.files = files
        self.break_after = break_after
        self.ranges: list[str | None] = []
        hub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                name = self.path.split("/resolve/main/", 1)[-1]
                body = hub.files.get(name)
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                rng = self.headers.get("Range")
                hub.ranges.append(rng)
                start = int(rng.split("=")[1].rstrip("-")) if rng else 0
                self.send_response(206 if rng else 200)
                self.send_header("Content-Length", str(len(body) - start))
                self.end_headers()
                chunk = body[start:]
                if hub.break_after is not None and len(chunk) > hub.break_after:
                    self.wfile.write(chunk[:hub.break_after])
                    self.wfile.flush()
                    self.connection.close()
                    return
                self.wfile.write(chunk)

            def log_message(self, *_args: Any) -> None:
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()


def pack_files(tmp_path: Path) -> dict[str, bytes]:
    source = make_pack(tmp_path / "source")
    return {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()
            and p.name != ".verified.json"}


def wait_download(pack: Pack, states: tuple[str, ...] = ("done", "error", "cancelled")) -> dict[str, Any]:
    deadline = time.time() + 30
    while time.time() < deadline:
        state = pack.download_state()
        if state and state["state"] in states:
            return state
        time.sleep(0.05)
    raise AssertionError("download did not finish")


def test_download_installs_a_checked_pack_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kiwibro.mind.pack.CHUNK", 16_384)  # the pack is tiny: small blocks, as the real one has 1 MB
    files = pack_files(tmp_path)
    hub = FakeHub(files, break_after=60_000)
    try:
        target = tmp_path / "app"
        pack = Pack(target, base_url=hub.url)
        pack.start_download()
        first = wait_download(pack)
        assert first["state"] == "error" and not pack.check().ok  # the server cut the connection
        assert not (target / "mind" / "kiwimind-trunk.fkm").exists()  # never a half file under its real name
        assert (target / "mind" / "kiwimind-trunk.fkm.part").stat().st_size > 0
        hub.break_after = None
        pack.start_download()
        state = wait_download(pack)
        assert state["state"] == "done" and state["done_bytes"] == state["total_bytes"] > 0
        assert any(r and r.startswith("bytes=") for r in hub.ranges)  # it asked for the rest, not everything
        assert Pack(target).check().ok
    finally:
        hub.close()


def test_download_rejects_a_wrong_checksum_and_can_be_cancelled(tmp_path: Path) -> None:
    files = pack_files(tmp_path)
    trunk = bytearray(files["kiwimind-trunk.fkm"])
    trunk[50] ^= 1
    hub = FakeHub({**files, "kiwimind-trunk.fkm": bytes(trunk)})
    try:
        pack = Pack(tmp_path / "app", base_url=hub.url)
        pack.start_download()
        state = wait_download(pack)
        assert state["state"] == "error" and "checksum" in state["error"]
        assert not (tmp_path / "app" / "mind" / "kiwimind-trunk.fkm").exists()
        assert not (tmp_path / "app" / "mind" / "kiwimind-trunk.fkm.part").exists()
    finally:
        hub.close()
    missing = Pack(tmp_path / "none", base_url="http://127.0.0.1:9")
    missing.start_download()
    assert wait_download(missing)["state"] == "error"
    assert missing.cancel_download() is False


# ------------------------------------------------------------------------------- the real host


@needs_node
def test_host_serves_the_tiny_pack_and_comes_back_after_a_kill(tmp_path: Path) -> None:
    make_pack(tmp_path)
    client = LLMClient(lambda: {"model": None}, data_dir=tmp_path)
    mind = client.mind()
    assert mind.usable() and mind.describe()["running"] is False
    events: list = []
    began = time.monotonic()
    bind_turn("chat-1", lambda name, data: events.append(name))
    result = client.stream_chat(
        model=KIWIMIND, messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "hello there"}],
        tools=None, max_tokens=12, temperature=0.0,
    )
    first_reply_s = time.monotonic() - began
    assert result.usage and result.usage["completion_tokens"] > 0 and events == ["brain"]
    pid = mind.host.describe()["pid"]
    assert pid and mind.describe()["running"] and mind.health()["pid"] == pid
    print(f"\nhost ready in {mind.host.describe()['ready_ms']} ms; first full reply {first_reply_s * 1000:.0f} ms")
    os.kill(pid, 9)
    time.sleep(0.3)
    bind_turn("chat-1")
    again = client.stream_chat(model=KIWIMIND, messages=[{"role": "user", "content": "and again"}], tools=None,
                               max_tokens=4, temperature=0.0)
    assert again.usage and mind.host.describe()["pid"] not in (None, pid)
    bind_turn("chat-2")
    client.prefill(model=KIWIMIND, messages=[{"role": "user", "content": "warm this"}], tools=None)
    client.stop_idle(0.0)  # the idle rule stops the host
    assert not mind.host.alive()
    mind.stop()


@needs_node
def test_a_pack_that_is_damaged_never_starts_a_process(tmp_path: Path) -> None:
    pack_dir = make_pack(tmp_path)
    (pack_dir / "head.kmh").write_bytes(b"x" * 10)
    mind = LLMClient(lambda: {}, data_dir=tmp_path).mind()
    assert not mind.usable() and "head.kmh" in mind.reason()
    assert not mind.host.alive()


def test_a_node_that_is_too_old_is_refused_with_a_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    from kiwibro.mind import host

    monkeypatch.setattr(host, "node_candidates", lambda: ["/fake/node"])
    monkeypatch.setattr(host, "node_version", lambda path: (16, 20, 0))
    path, why = host.find_node()
    assert path is None and "too old" in why and "18.14" in why
    monkeypatch.setattr(host, "node_candidates", lambda: [])
    assert host.find_node()[1] == "Kiwi Mind needs Node.js, which is not installed."


# --------------------------------------------------------- the context note: app against training


def load_train() -> Any:
    spec = importlib.util.spec_from_file_location("kiwimind_train_for_tests", ROOT / "scripts" / "mind" / "train.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def real_user_turn(env: Any, *, long: bool, steps: bool = False) -> tuple[str, str]:
    """(the user message as the app sends it, the time in its clock line), built by the app's own code."""
    from kiwibro.agent import Turn

    question = "hello there"
    chat = env.svc.db.create_conversation()["id"]
    if long:  # a session that has scrolled: the ledger block is attached
        env.svc.db.update_conversation(chat, summary_through_seq=4)
        if steps:
            ledger.record_user(env.svc.db, chat, 1, "Build a to-do API.")
    note = env.svc.agent._turn_context(Turn(conversation_id=chat, message=question, requested_model=None), {})
    sent = context.place_turn_context([{"role": "user", "content": question}], note)[-1]["content"]
    when = next(line for line in note.splitlines() if line.startswith("Local time: ")).removeprefix("Local time: ")
    return sent, when


@pytest.mark.parametrize(("is_offline", "long"), [(False, False), (True, False), (False, True), (True, True)])
def test_the_context_note_has_the_layout_the_skills_were_trained_on(
    env: Any, request: pytest.FixtureRequest, is_offline: bool, long: bool
) -> None:
    if is_offline:
        request.getfixturevalue("offline")
    train = load_train()
    sent, when = real_user_turn(env, long=long)
    assert sent == train.with_note("hello there", when, offline=is_offline, long=long)
    prompt = render.render([{"role": "system", "content": "s"}, {"role": "user", "content": sent}], [])
    assert sent in prompt  # the model reads it as it was trained: nothing in the renderer reshapes the note


def test_a_session_with_recorded_steps_adds_lines_the_training_notes_do_not_have(env: Any) -> None:
    """A known difference, kept visible: the app lists what happened after the header, the training note is the
    header alone. The shared part is the same; the model has to cope with the extra lines."""
    train = load_train()
    sent, when = real_user_turn(env, long=True, steps=True)
    trained = train.with_note("hello there", when, long=True)
    assert sent.startswith(trained) and sent != trained
    assert sent[len(trained):].startswith("\nAsked: ")


# ------------------------------------------------------------------------------------------- the API


@pytest.fixture
def with_mind(env: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The test app, with a real Kiwi Mind (its own data folder) in place of the scripted model's lack of one."""
    real = LLMClient(lambda: {"model": None}, data_dir=tmp_path / "mind-data")
    monkeypatch.setattr(env.svc.llm, "mind", real.mind, raising=False)
    env.mind = real.mind()
    yield env
    env.mind.stop()


@needs_node
def test_the_api_describes_an_installed_kiwi_mind(with_mind: Any, tmp_path: Path) -> None:
    make_pack(tmp_path / "mind-data")
    shown = with_mind.client.get("/api/mind").get_json()
    assert shown["usable"] is True and shown["installed"] is True and shown["reason"] == ""
    assert shown["version"] == "test-1" and shown["bytes"] > 0 and shown["node"]
    assert shown["skills"] == ["chat", "web", "code", "terminal", "computer", "tools", "write"]
    assert shown["running"] is False and shown["pid"] is None and shown["helper"] is None
    assert shown["download"] is None
    assert with_mind.client.get("/health").get_json()["mind"] == {"usable": True, "running": False, "pid": None}
    chosen = with_mind.client.post("/api/models/select", json={"id": "none"})  # Kiwi Mind alone
    assert chosen.status_code == 200 and with_mind.svc.settings_provider()["model"] == "none"


def test_the_api_says_when_kiwi_mind_is_not_installed(with_mind: Any) -> None:
    shown = with_mind.client.get("/api/mind").get_json()
    assert shown["usable"] is False and shown["installed"] is False and shown["reason"] == "Kiwi Mind is not installed."
    assert with_mind.client.get("/health").get_json()["mind"]["usable"] is False
    assert with_mind.client.post("/api/models/select", json={"id": "none"}).status_code == 400  # nothing to lead
    assert with_mind.client.post("/api/mind/download/cancel").get_json()["ok"] is False


def test_the_api_downloads_the_pack_and_reports_progress(with_mind: Any, tmp_path: Path) -> None:
    hub = FakeHub(pack_files(tmp_path))
    try:
        with_mind.mind.pack.base_url = hub.url
        started = with_mind.client.post("/api/mind/download").get_json()
        assert started["ok"] is True and started["download"]["state"] in ("running", "done")
        wait_download(with_mind.mind.pack)
        shown = with_mind.client.get("/api/mind").get_json()
        assert shown["download"]["state"] == "done" and shown["download"]["done_bytes"] == shown["bytes"]
        assert shown["installed"] is True and (shown["usable"] is True or "Node.js" in shown["reason"])
    finally:
        hub.close()


# ---------------------------------------------------------- the tool list and the conversation binding


def test_kiwi_mind_gets_the_whole_catalog_in_the_apps_order(env: Any) -> None:
    from kiwibro import tools

    agent = env.svc.agent
    settings = env.svc.settings_provider()
    info = {"tools": True}
    given = agent._tool_list(KIWIMIND, settings, info)
    assert given == tools.tool_defs_for("full", settings=settings)
    assert [item["function"]["name"] for item in given] == tools.tool_names()
    assert len(given) > len(tools.tool_defs_for(settings=settings))  # more than the auto set any other model gets
    assert agent._tool_list("fake-1", settings, info) == tools.tool_defs_for(settings=settings)  # others unchanged
    denied = {**settings, "autonomy": "supervised", "permissions": {"browser": "deny"}}  # a tool the user switched off
    assert "browser" not in [item["function"]["name"] for item in agent._tool_list(KIWIMIND, denied, info)]
    assert agent._tool_list(KIWIMIND, {**settings, "agent_enabled": False}, info) is None


def test_the_system_text_is_the_same_for_every_skill() -> None:
    """The lower-layer cache is keyed by this text, so a skill switch must not touch it."""
    from kiwibro import tools

    catalog = tools.tool_defs_for("full", settings={})
    prompts_seen = []
    for skill in ("code", "web"):
        client = FakeClient()
        client.logits[skill] = 4.0
        ask(conductor(client), tools=catalog)
        assert client.bodies[0]["skill"] == skill
        prompts_seen.append(client.bodies[0]["prompt"])
    start = [text.split("<|im_start|>user")[0] if "<|im_start|>user" in text else text for text in prompts_seen]
    assert start[0] == start[1] and all(item["function"]["name"] in start[0] for item in catalog)


def test_a_bound_turn_is_used_once_and_never_reaches_a_request(tmp_path: Path) -> None:
    from kiwibro import llm

    bind_turn("chat-7", lambda *_: None)
    assert llm._take_turn()[0] == "chat-7"
    assert llm._take_turn() == (None, None)  # the next call is not bound unless it says so
    assert not any(key.startswith("_") for key in LLMClient(lambda: {}, data_dir=tmp_path)._payload(
        "m", [], None, 0.0, 1, stream=False, extra={"top_k": 5}))


def test_the_window_is_charged_for_the_manual_kiwi_mind_reads_not_for_the_json_schemas(env: Any) -> None:
    """The 47 JSON schemas are about 8,200 tokens, more than the whole 8192 window: charged like that, the room
    for the conversation collapsed to its floor. Kiwi Mind reads one line per tool, about 700 tokens."""
    import json

    from kiwibro import tools
    from kiwibro.context import compute_budget, estimate_tokens

    catalog = tools.tool_defs_for("full", settings={})
    schemas = estimate_tokens(json.dumps(catalog, ensure_ascii=False))
    mine = env.svc.agent._tool_tokens(KIWIMIND, catalog)
    assert schemas > 8000 and 400 < mine < 1000
    assert env.svc.agent._tool_tokens("fake-1", catalog) == schemas  # every other model: as before
    assert env.svc.agent._tool_tokens(KIWIMIND, None) == 0
    assert compute_budget(8192, 1024) - mine > 4500
