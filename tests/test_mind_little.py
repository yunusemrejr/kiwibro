"""The little CPU helper: which jobs it takes, when it leaves them alone, and what happens when it is not there."""

from __future__ import annotations

import http.server
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiwibro.db import Database
from kiwibro.llm import Completion, LLMClient, TurnAborted
from kiwibro.memory import MemoryEngine, lines_to_ops
from kiwibro.mind import KIWIMIND, little as little_module
from kiwibro.mind.little import Little, LittleError, YieldedToChat, background, server_args
from kiwibro.mind.llama import LlamaRuntime
from kiwibro.mind.service import Mind

MESSAGES = [{"role": "system", "content": "Keep notes."}, {"role": "user", "content": "I live in Bern."}]


# ---------------------------------------------------------------------------------------------- fakes


class FakeLittle:
    """Stands in for Little inside Kiwi Mind."""

    def __init__(self, error: Exception | None = None, *, text: str = "little answer") -> None:
        self.error = error
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: Any, **kwargs: Any) -> Completion:
        self.calls.append({"messages": messages, **kwargs})
        if self.error is not None:
            raise self.error
        return Completion(text=self.text, finish_reason="stop")

    def stop(self) -> None:
        pass

    def available(self) -> bool:
        return True


class FakeConductor:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: Any, **_kwargs: Any) -> Completion:
        self.calls += 1
        return Completion(text="kiwi mind answer", finish_reason="stop")


def client_with(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, helper: dict[str, Any] | None = None,
                little: FakeLittle | None = None) -> tuple[LLMClient, Any, FakeLittle, FakeConductor]:
    settings = {"server_url": "http://127.0.0.1:9", "model": helper["id"] if helper else "none"}
    client = LLMClient(lambda: settings, data_dir=tmp_path)
    mind = client.mind()
    monkeypatch.setattr(type(mind), "usable", lambda self: True)
    monkeypatch.setattr(client, "helper", lambda: helper)
    fake_little, conductor = little or FakeLittle(), FakeConductor()
    mind.little = fake_little
    mind.conductor = conductor
    return client, mind, fake_little, conductor


class FakeProcess:
    pid = 4242

    def __init__(self) -> None:
        self.alive = True
        self.terminated = False

    def poll(self) -> int | None:
        return None if self.alive else 0

    def terminate(self) -> None:
        self.terminated, self.alive = True, False

    def wait(self, _timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.alive = False


def running_little(port: int = 1, **kwargs: Any) -> tuple[Little, FakeProcess]:
    little = Little(None, lambda: None, **kwargs)
    process = FakeProcess()
    little._running = little_module._Running(process, port, "key", time.time(), 5, threading.Event())  # type: ignore[arg-type]
    return little, process


# -------------------------------------------------------------------------------- which jobs it takes


def test_a_background_job_goes_to_the_little_helper_when_no_helper_is_chosen(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _mind, little, conductor = client_with(tmp_path, monkeypatch)
    with background("summary"):
        done = client.complete(model=KIWIMIND, messages=MESSAGES, max_tokens=300)
    assert done.text == "little answer" and conductor.calls == 0
    assert little.calls[0]["max_tokens"] == 300 and little.calls[0]["defer"] is False


def test_memory_is_deferred_json_work_and_hands_its_abort_event_on(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _mind, little, _conductor = client_with(tmp_path, monkeypatch)
    stop = threading.Event()
    seen: dict[str, Any] = {}

    def spy(messages: Any, **kw: Any) -> Completion:
        seen.update(kw)
        return Completion(text='{"ops": []}', finish_reason="stop")

    little.complete = spy  # type: ignore[method-assign]
    with background("memory", defer=True, as_json=True):
        assert client.complete_json(model=KIWIMIND, messages=MESSAGES, abort=stop) == {"ops": []}
    assert seen["defer"] is True and seen["as_json"] is True and seen["abort"] is stop


def test_calls_that_are_no_background_job_stay_on_kiwi_mind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _mind, little, conductor = client_with(tmp_path, monkeypatch)
    assert client.complete(model=KIWIMIND, messages=MESSAGES).text == "kiwi mind answer"  # a title, a side question
    picture = [{"role": "user", "content": [{"type": "text", "text": "what is this"},
                                           {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]
    with background("summary"):
        assert client.complete(model=KIWIMIND, messages=picture).text == "kiwi mind answer"
    assert little.calls == [] and conductor.calls == 2


def test_a_helper_model_the_user_chose_keeps_the_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = {"id": "big-model", "name": "Big Model"}
    client, _mind, little, conductor = client_with(tmp_path, monkeypatch, helper=helper)
    seen: dict[str, Any] = {}

    class Reply:
        status_code = 200

        @staticmethod
        def json() -> dict[str, Any]:
            return {"choices": [{"message": {"content": "from helper"}, "finish_reason": "stop"}]}

    def post(url: str, json: dict[str, Any], **_kw: Any) -> Reply:  # noqa: A002
        seen["model"] = json["model"]
        return Reply()

    monkeypatch.setattr("kiwibro.llm.requests.post", post)
    with background("summary"):
        assert client.complete(model=KIWIMIND, messages=MESSAGES).text == "from helper"
    assert seen["model"] == "big-model" and little.calls == [] and conductor.calls == 0


def test_a_little_helper_that_cannot_take_a_job_leaves_it_to_kiwi_mind(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _mind, little, conductor = client_with(tmp_path, monkeypatch, little=FakeLittle(LittleError("no model")))
    with background("summary"):
        assert client.complete(model=KIWIMIND, messages=MESSAGES).text == "kiwi mind answer"
    assert len(little.calls) == 1 and conductor.calls == 1


def test_trained_memory_expert_gets_the_short_prompt_and_explicit_abstention(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, mind, stock, conductor = client_with(tmp_path, monkeypatch)
    trained = FakeLittle(text="none")
    mind.memory_expert = trained
    prompt = [MESSAGES[0], {"role": "assistant", "content": "example"}, MESSAGES[-1]]
    with background("memory", messages=prompt, convert=lambda _text: '{"ops": []}'):
        assert client.complete_json(model=KIWIMIND, messages=MESSAGES) == {"ops": []}
    assert trained.calls[0]["messages"] == MESSAGES
    assert not stock.calls and conductor.calls == 0


@pytest.mark.parametrize("error", [LittleError("no server"), None])
def test_unavailable_or_bad_expert_uses_stock_memory_prompt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception | None) -> None:
    client, mind, stock, conductor = client_with(tmp_path, monkeypatch)
    trained = FakeLittle(error)
    mind.memory_expert = trained
    prompt = [MESSAGES[0], {"role": "assistant", "content": "example"}, MESSAGES[-1]]
    with background("memory", messages=prompt, convert=lambda _text: '{"ops": []}'):
        assert client.complete_json(model=KIWIMIND, messages=MESSAGES) == {"ops": []}
    assert trained.calls[0]["messages"] == MESSAGES
    assert stock.calls[0]["messages"] == prompt and conductor.calls == 0


def test_a_reply_that_came_first_reaches_the_memory_job_as_cancelled(tmp_path: Path) -> None:
    class Llm:
        def complete_json(self, **_kw: Any) -> Any:
            raise YieldedToChat()

    db = Database(tmp_path / "data" / "kiwibro.db")
    db.initialize()
    memory = MemoryEngine(db, Llm(), lambda: {})  # type: ignore[arg-type]
    window = [{"role": "user", "content": "I live in Bern.", "seq": 1}]
    assert memory._bounded_extract("c1", KIWIMIND, window=window) == (None, "cancelled")  # no abort flag was set


# ------------------------------------------------------------------------ not there, or not starting


def test_without_the_model_file_a_job_is_given_back_and_it_is_said_once(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    little = Little(tmp_path, lambda: None)
    caplog.set_level(logging.INFO, logger="kiwibro.mind.little")
    for _ in range(3):
        with pytest.raises(LittleError, match="model file"):
            little.complete(MESSAGES, max_tokens=50)
    said = [r for r in caplog.records if "little helper is not used" in r.getMessage()]
    assert len(said) == 1
    assert little.describe()["available"] is False and little.describe()["running"] is False


def test_a_model_file_put_in_place_later_is_found_without_a_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    little = Little(tmp_path, lambda: None)
    with pytest.raises(LittleError, match="model file"):
        little.complete(MESSAGES, max_tokens=50)
    path = little.path
    assert path is not None
    path.parent.mkdir(parents=True)
    path.write_bytes(b"gguf")
    with pytest.raises(LittleError, match="llama.cpp was not found"):  # the file is seen now; the engine is not
        little.complete(MESSAGES, max_tokens=50)
    assert little.describe()["available"] is True


def test_a_server_that_does_not_start_is_not_tried_again_for_a_while(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    little = Little(tmp_path, lambda: None)
    assert little.path is not None
    little.path.parent.mkdir(parents=True)
    little.path.write_bytes(b"gguf")
    starts = []

    def fail() -> Any:
        starts.append(1)
        raise LittleError("the little helper did not start")

    monkeypatch.setattr(little, "_start", fail)
    caplog.set_level(logging.INFO, logger="kiwibro.mind.little")
    for _ in range(3):
        with pytest.raises(LittleError):
            little.complete(MESSAGES, max_tokens=50)
    assert len(starts) == 1
    little._retry_at = 0.0  # the wait is over
    with pytest.raises(LittleError):
        little.complete(MESSAGES, max_tokens=50)
    assert len(starts) == 2
    assert len([r for r in caplog.records if "little helper is not used" in r.getMessage()]) == 1


def test_a_job_that_does_not_fit_the_window_goes_to_kiwi_mind() -> None:
    little, _process = running_little()
    big = [{"role": "user", "content": "word " * 9000}]
    with pytest.raises(LittleError, match="holds"):
        little.complete(big, max_tokens=300)


# ---------------------------------------------------------------------------- the command line it runs


def test_the_server_is_cpu_only_two_threads_one_slot() -> None:
    gpu_build = LlamaRuntime("/x/llama-server", frozenset(), True, {})
    args = server_args(gpu_build, Path("/m/q.gguf"), port=9999)
    pairs = dict(zip(args, args[1:]))
    assert pairs["--device"] == "none" and pairs["--gpu-layers"] == "0"
    assert pairs["--threads"] == "2" and pairs["--parallel"] == "1" and pairs["--ctx-size"] == str(little_module.CTX)
    assert pairs["--host"] == "127.0.0.1" and pairs["--port"] == "9999" and "--jinja" in args
    cpu_build = LlamaRuntime("/x/llama-server", frozenset({"--model", "--gpu-layers"}), None, {})
    plain = server_args(cpu_build, Path("/m/q.gguf"), port=1)
    assert "--device" not in plain and "--jinja" not in plain and "--gpu-layers" in plain  # only what the build lists


# -------------------------------------------------------------------------------- idle and in use


def test_it_leaves_only_after_a_quiet_spell_and_never_in_the_middle_of_a_job() -> None:
    little, process = running_little()
    running = little._acquire()  # a job starts
    assert running.port == 1 and little.stop_idle(0.0) == 0 and process.alive
    little._release()
    little._last_used = time.monotonic() - 10
    assert little.stop_idle(60.0) == 0 and process.alive  # not quiet for long enough
    assert little.stop_idle(5.0) == 1 and process.terminated
    assert little.describe()["running"] is False


def test_the_state_names_the_process_and_the_port() -> None:
    little, _process = running_little(port=8123)
    assert little.health() == {"available": False, "running": True, "pid": 4242, "port": 8123}


def test_kiwi_mind_reports_and_stops_its_little_helper(tmp_path: Path) -> None:
    mind = Mind(tmp_path, helper=lambda: None, helper_stream=lambda *a, **k: None, runtime=lambda: None)  # type: ignore[arg-type]
    assert mind.health()["little"] == {"available": False, "running": False, "pid": None, "port": None}
    assert mind.describe()["little"]["model"] == "Qwen3-0.6B Q8_0"
    process = FakeProcess()
    mind.little._running = little_module._Running(process, 7, "k", time.time(), 1, threading.Event())  # type: ignore[arg-type]
    assert mind.health()["little"]["running"] is True
    mind.stop()
    assert process.terminated and mind.health()["little"]["running"] is False


# ------------------------------------------------------------ a reply first: waiting and letting go


def test_a_deferred_job_waits_while_a_reply_is_written() -> None:
    busy = [True]
    little, _process = running_little(chat_busy=lambda: busy[0])
    done = threading.Event()

    def job() -> None:
        little._wait_for_chat(None)
        done.set()

    threading.Thread(target=job, daemon=True).start()
    assert not done.wait(0.6)
    busy[0] = False
    assert done.wait(2.0)


def test_a_deferred_job_gives_way_when_the_reply_goes_on_or_the_caller_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    little, _process = running_little(chat_busy=lambda: True)
    monkeypatch.setattr(little_module, "CHAT_WAIT_S", 0.3)
    with pytest.raises(YieldedToChat):
        little._wait_for_chat(None)
    abort = threading.Event()
    abort.set()
    with pytest.raises(TurnAborted) as raised:
        little._wait_for_chat(abort)
    assert not isinstance(raised.value, YieldedToChat)


class Chat:
    """A llama-server's /v1/chat/completions: streams a first piece, then either ends or stalls."""

    def __init__(self, stall: bool = False, cut: bool = False) -> None:
        self.stall, self.cut, self.bodies, self.closed = stall, cut, [], threading.Event()
        chat = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                chat.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                piece = {"choices": [{"delta": {"content": "<think>\n\n</think>\n\nBern"}}]}
                self.wfile.write(f"data: {json.dumps(piece)}\n\n".encode())
                self.wfile.flush()
                if chat.stall:
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline:
                        time.sleep(0.05)
                        try:
                            self.wfile.write(b": still working\n\n")
                            self.wfile.flush()
                        except OSError:
                            chat.closed.set()
                            return
                    return
                if chat.cut:  # hangs up after the first piece
                    return
                end = {"choices": [{"delta": {}, "finish_reason": "stop"}],
                       "timings": {"prompt_n": 30, "predicted_n": 4}}
                self.wfile.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = int(self.server.server_address[1])
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_a_job_is_asked_without_thinking_and_the_answer_is_cleaned() -> None:
    chat = Chat(stall=False)
    try:
        little, _process = running_little(port=chat.port)
        done = little.complete(MESSAGES, max_tokens=64, as_json=True)
    finally:
        chat.stop()
    body = chat.bodies[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["stream"] is True
    assert body["max_tokens"] == 64 and body["response_format"] == {"type": "json_object"}
    assert done.text == "Bern" and done.finish_reason == "stop" and done.usage == {"prompt_tokens": 30, "completion_tokens": 4}
    assert little.served == 1


def test_a_reply_that_starts_cancels_a_deferred_job_in_flight() -> None:
    chat = Chat(stall=True)
    busy = [False]
    try:
        little, _process = running_little(port=chat.port, chat_busy=lambda: busy[0])
        threading.Timer(0.5, lambda: busy.__setitem__(0, True)).start()
        began = time.monotonic()
        with pytest.raises(YieldedToChat):
            little._ask(little._running, MESSAGES, 64, 0.0, None, True, False)  # type: ignore[arg-type]
        assert time.monotonic() - began < 4
        assert chat.closed.wait(3)  # the connection was dropped, so llama-server stops working on it
    finally:
        chat.stop()


def test_a_reply_does_not_cancel_a_job_somebody_is_waiting_for() -> None:
    chat = Chat(stall=False)
    try:
        little, _process = running_little(port=chat.port, chat_busy=lambda: True)
        assert little._ask(little._running, MESSAGES, 64, 0.0, None, False, False).text == "Bern"  # type: ignore[arg-type]
    finally:
        chat.stop()


def test_the_caller_can_abort_a_job_in_flight() -> None:
    chat = Chat(stall=True)
    abort = threading.Event()
    try:
        little, _process = running_little(port=chat.port)
        threading.Timer(0.5, abort.set).start()
        with pytest.raises(TurnAborted) as raised:
            little._ask(little._running, MESSAGES, 64, 0.0, abort, False, False)  # type: ignore[arg-type]
        assert not isinstance(raised.value, YieldedToChat)
        assert chat.closed.wait(3)
    finally:
        chat.stop()


def test_a_server_that_hangs_up_early_is_a_failed_job_not_a_short_answer() -> None:
    chat = Chat(cut=True)
    try:
        little, _process = running_little(port=chat.port)
        with pytest.raises(LittleError, match="stopped answering"):
            little._ask(little._running, MESSAGES, 64, 0.0, None, False, False)  # type: ignore[arg-type]
    finally:
        chat.stop()


# ------------------------------------------------------------------------- its own prompt and its answer


def test_a_job_may_bring_the_little_helper_its_own_prompt_and_reading(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _mind, little, _conductor = client_with(tmp_path, monkeypatch)
    simple = [{"role": "user", "content": "plain version"}]
    with background("memory", defer=True, messages=simple, convert=lambda text: text.upper()):
        done = client.complete(model=KIWIMIND, messages=MESSAGES, max_tokens=100)
    assert little.calls[0]["messages"] is simple
    assert done.text == "LITTLE ANSWER"


def test_a_line_that_names_something_the_user_never_wrote_is_dropped() -> None:
    answer = "\n".join([
        "fact | User's name is Selin",
        "fact | User lives in Izmir.",
        "person | User's son Omar is nine",  # copied from a worked example
        "Australia is Sydney",  # not about the user, no kind
        "preference | User wants answers in French",  # copied: French is not in what was written
        "- fact | User has two cats",
    ])
    source = "User: Hi, I'm Selin, I live in Izmir and I have two cats."
    ops = json.loads(lines_to_ops(answer, source))["ops"]
    assert [(op["kind"], op["content"]) for op in ops] == [
        ("fact", "User's name is Selin"), ("fact", "User lives in Izmir."), ("fact", "User has two cats")]
    assert all(op["op"] == "add" for op in ops)


def test_none_and_empty_answers_become_no_operations() -> None:
    assert json.loads(lines_to_ops("none", "User: how do I sort a list"))["ops"] == []
    assert json.loads(lines_to_ops("", ""))["ops"] == []


def test_the_little_memory_prompt_is_worked_examples_then_the_transcript(tmp_path: Path) -> None:
    from kiwibro import prompts

    engine = MemoryEngine(Database(tmp_path / "m.db"), None, lambda: {})  # type: ignore[arg-type]
    messages = engine._little_extraction_messages((["User: I keep bees.", "Assistant: Nice."], []))
    assert messages[0] == {"role": "system", "content": prompts.LITTLE_MEMORY_SYSTEM}
    assert len(messages) == 2 + 2 * len(prompts.LITTLE_MEMORY_EXAMPLES)
    assert messages[-1] == {"role": "user", "content": "I keep bees."}  # only what the user wrote
    assert "Already remembered" not in messages[-1]["content"]
