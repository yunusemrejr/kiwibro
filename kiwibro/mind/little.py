"""A CPU-only Qwen3-0.6B worker for small utility jobs.

Memory extraction and conversation summaries used to run on Kiwi Mind's 4B model, on the graphics chip that
writes the replies. The little helper has a llama-server of its own, started on the first such job, on two CPU
threads, with thinking off. Kiwi Mind can point a worker at a trained GGUF or use the stock local model.
Workers leave after a few idle minutes and with the app. Replies and route decisions stay with the conductor.

It is an optional extra: with no model file, no llama.cpp or a server that will not start, ``Little`` says so
once and the callers keep using Kiwi Mind, as they did before.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import secrets
import subprocess
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import requests

from ..llm import CONNECT_TIMEOUT, USER_AGENT, Completion, TurnAborted, _socket_of, strip_reasoning
from .llama import LlamaRuntime, _auth

log = logging.getLogger(__name__)

# Under <data>/models: where the file is put by hand, and where a download from the Models panel lands.
MODEL_FILES = (
    Path("Qwen") / "Qwen3-0.6B-Q8_0.gguf",
    Path("Qwen") / "Qwen3-0.6B-GGUF" / "Qwen3-0.6B-Q8_0.gguf",
)
MODEL_NAME = "Qwen3-0.6B Q8_0"
THREADS = 2
CTX = 8192
IDLE_STOP_S = 300.0  # the server leaves after this long without a job
RETRY_AFTER_S = 600.0  # after a failed start, not again for this long
READY_TIMEOUT_S = 120.0
REQUEST_LIMIT_S = 300.0  # one job that takes longer is given back to Kiwi Mind
CHAT_WAIT_S = 120.0  # a background job waits this long for the chat to be quiet, then gives way
NICE = 10  # the chat is never slowed by a background job
STOP_WAIT_S = 5.0
LOG_LIMIT = 256 * 1024
PROMPT_OVERHEAD = 48  # chat template tokens around a prompt


class LittleError(Exception):
    """The little helper cannot take this job; the job goes to Kiwi Mind."""


class YieldedToChat(TurnAborted):
    """A reply started, so the little helper let go of a background job. The job is tried again later."""


@dataclass
class _Job:
    kind: str  # "memory" | "summary"
    defer: bool  # waits for a quiet chat and gives way to one
    as_json: bool  # the answer is JSON
    messages: list[dict[str, Any]] | None  # the prompt for the little helper, when it differs from Kiwi Mind's
    convert: Callable[[str], str] | None  # turns the little helper's answer into the form the caller reads


_JOB: ContextVar[_Job | None] = ContextVar("kiwibro_little_job", default=None)


@contextlib.contextmanager
def background(
    kind: str, *, defer: bool = False, as_json: bool = False, messages: list[dict[str, Any]] | None = None,
    convert: Callable[[str], str] | None = None,
) -> Iterator[None]:
    """Mark the model calls made inside as a background job the little helper may take.

    ``defer``: the job is not needed by anyone waiting (memory), so it waits while a reply is being written and is
    let go when one starts. A job somebody waits for (a summary made while a turn is prepared) is not deferred.
    ``messages`` and ``convert``: a smaller model needs a simpler prompt and answers in a simpler form; the
    little helper gets ``messages`` instead of the caller's and its answer goes through ``convert``."""
    token = _JOB.set(_Job(kind, defer, as_json, messages, convert))
    try:
        yield
    finally:
        _JOB.reset(token)


def current_job() -> _Job | None:
    return _JOB.get()


@dataclass
class _Running:
    process: subprocess.Popen[bytes]
    port: int
    key: str
    started_at: float
    ready_ms: int
    gone: threading.Event

    @property
    def pid(self) -> int:
        return self.process.pid


class Little:
    """Starts, uses and stops the little helper's llama-server. One per app; nothing starts before the first job."""

    def __init__(
        self, data_dir: str | Path | None, runtime: Callable[[], LlamaRuntime | None],
        chat_busy: Callable[[], bool] = lambda: False, *, idle_stop_s: float = IDLE_STOP_S,
        model_file: Callable[[], Path | None] | None = None, model_name: str = MODEL_NAME,
        log_name: str = "little.log",
    ) -> None:
        self.data_dir = Path(data_dir).expanduser() if data_dir is not None else None
        self._model_file = model_file
        self._model_name = model_name
        self._log_name = log_name
        self._runtime = runtime
        self._chat_busy = chat_busy
        self._idle_stop_s = idle_stop_s
        self._lock = threading.RLock()
        self._running: _Running | None = None
        self._in_flight = 0
        self._last_used = time.monotonic()
        self._retry_at = 0.0
        self._why = ""
        self._said = False
        self.served = 0

    # --- state -------------------------------------------------------------------

    @property
    def path(self) -> Path | None:
        if self._model_file is not None:
            return self._model_file()
        if self.data_dir is None:
            return None
        found = [self.data_dir / "models" / name for name in MODEL_FILES]
        return next((file for file in found if file.is_file()), found[0])

    @property
    def log_path(self) -> Path | None:
        return self.data_dir / "mind" / self._log_name if self.data_dir is not None else None

    def available(self) -> bool:
        """Whether the model file is there (a server that failed to start is a separate matter: see ``describe``)."""
        path = self.path
        return path is not None and path.is_file()

    def alive(self) -> bool:
        running = self._running
        return running is not None and running.process.poll() is None

    def describe(self) -> dict[str, Any]:
        running = self._running if self.alive() else None
        return {
            "available": self.available(),
            "model": self._model_name,
            "running": running is not None,
            "pid": running.pid if running else None,
            "port": running.port if running else None,
            "served": self.served,
            "why": self._why,
        }

    def health(self) -> dict[str, Any]:
        running = self._running if self.alive() else None
        return {"available": self.available(), "running": running is not None,
                "pid": running.pid if running else None, "port": running.port if running else None}

    # --- one job -------------------------------------------------------------------

    def complete(
        self, messages: Sequence[dict[str, Any]], *, max_tokens: int, temperature: float = 0.0,
        abort: threading.Event | None = None, defer: bool = False, as_json: bool = False,
    ) -> Completion:
        """The answer to one utility prompt. Raises LittleError when the job should go to Kiwi Mind and
        YieldedToChat (a TurnAborted) when a reply came first; ``abort`` raises TurnAborted."""
        self._fits(messages, max_tokens)
        if defer:
            self._wait_for_chat(abort)
        running = self._acquire()
        try:
            done = self._ask(running, messages, max_tokens, temperature, abort, defer, as_json)
        finally:
            self._release()
        self.served += 1
        return done

    def _fits(self, messages: Sequence[dict[str, Any]], max_tokens: int) -> None:
        from ..context import estimate_tokens  # not at the top: context imports this module

        need = sum(estimate_tokens(str(m.get("content") or "")) + 4 for m in messages) + PROMPT_OVERHEAD + max_tokens
        if need > CTX:
            raise LittleError(f"the job needs about {need} tokens and the little helper holds {CTX}")

    def _wait_for_chat(self, abort: threading.Event | None) -> None:
        deadline = time.monotonic() + CHAT_WAIT_S
        while self._chat_busy():
            if abort is not None and abort.is_set():
                raise TurnAborted()
            if time.monotonic() > deadline:
                raise YieldedToChat()
            time.sleep(0.25)
        if abort is not None and abort.is_set():
            raise TurnAborted()

    def _ask(
        self, running: _Running, messages: Sequence[dict[str, Any]], max_tokens: int, temperature: float,
        abort: threading.Event | None, defer: bool, as_json: bool,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": "little", "messages": list(messages), "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": True,
        }
        if as_json:
            payload["response_format"] = {"type": "json_object"}
        stop = threading.Event()  # set by the watcher: why is in ``cause``
        cause: list[str] = []
        finished = threading.Event()
        text: list[str] = []
        finish: str | None = None
        usage: dict[str, int] | None = None
        try:
            with requests.post(
                f"http://127.0.0.1:{running.port}/v1/chat/completions", json=payload, stream=True,
                timeout=(CONNECT_TIMEOUT, REQUEST_LIMIT_S),
                headers={**_auth(running.key), "User-Agent": USER_AGENT, "Accept": "text/event-stream"},
            ) as response:
                threading.Thread(
                    target=self._watch, args=(response, abort, defer, stop, cause, finished),
                    name="kiwibro-little-watch", daemon=True,
                ).start()
                if response.status_code != 200:
                    raise LittleError(f"the little helper refused the job ({response.status_code}): {response.text[:200]}")
                for raw in response.iter_lines(decode_unicode=False):
                    if stop.is_set():
                        break
                    if not raw.startswith(b"data:"):
                        continue
                    data = raw[5:].strip()
                    if data == b"[DONE]":
                        break
                    try:
                        chunk = json.loads(data.decode("utf-8"))
                    except ValueError:
                        continue
                    choice = (chunk.get("choices") or [{}])[0]
                    text.append(str((choice.get("delta") or {}).get("content") or ""))
                    finish = choice.get("finish_reason") or finish
                    timings = chunk.get("timings")
                    if isinstance(timings, dict):
                        usage = {"prompt_tokens": int(timings.get("prompt_n") or 0),
                                 "completion_tokens": int(timings.get("predicted_n") or 0)}
        except requests.RequestException as exc:
            if not stop.is_set():
                raise LittleError(f"the little helper did not answer: {exc}") from exc
        finally:
            finished.set()
        if finish is None:  # not finished: somebody cut it off, or the server went away
            if cause and cause[0] == "abort":
                raise TurnAborted()
            if cause and cause[0] == "chat":
                raise YieldedToChat()
            raise LittleError("the job took too long" if cause else "the little helper stopped answering")
        return Completion(text=strip_reasoning("".join(text)).strip(), usage=usage, finish_reason=finish)

    def _watch(
        self, response: requests.Response, abort: threading.Event | None, defer: bool, stop: threading.Event,
        cause: list[str], finished: threading.Event,
    ) -> None:
        """Close the connection when the caller aborts, when a reply starts (a deferred job) or when the job takes
        too long: llama-server drops the work at its next batch, also during the prompt."""
        deadline = time.monotonic() + REQUEST_LIMIT_S
        while not finished.wait(0.1):
            if abort is not None and abort.is_set():
                cause.append("abort")
            elif defer and self._chat_busy():
                cause.append("chat")
            elif time.monotonic() > deadline:
                cause.append("late")
            else:
                continue
            stop.set()
            sock = _socket_of(response)
            if sock is not None:
                try:
                    sock.shutdown(2)  # SHUT_RDWR
                except OSError:
                    pass
            return

    # --- the server ------------------------------------------------------------------

    def _acquire(self) -> _Running:
        """The running server, started if need be, counted as in use until ``_release``."""
        with self._lock:
            if not self.alive():
                if self._running is not None:
                    self._note(f"the little helper (pid {self._running.pid}) stopped; it starts again")
                    self._running = None
                self._running = self._start_checked()
            self._in_flight += 1
            assert self._running is not None
            return self._running

    def _release(self) -> None:
        with self._lock:
            self._in_flight -= 1
            self._last_used = time.monotonic()

    def _start_checked(self) -> _Running:
        if not self.available():  # looked at every time: the file may be put in place while Kiwibro runs
            self._why = "the model file is not there"
            self._note(f"the little helper is not used, background jobs stay on Kiwi Mind: {self._why}")
            raise LittleError(self._why)
        if time.monotonic() < self._retry_at:
            raise LittleError(self._why)
        try:
            running = self._start()
            self._why = ""
            return running
        except LittleError as exc:
            self._why = str(exc)
            self._retry_at = time.monotonic() + RETRY_AFTER_S
            self._note(f"the little helper is not used, background jobs stay on Kiwi Mind: {exc}")
            raise

    def _note(self, message: str) -> None:
        if not self._said:  # said once; the reason stays in describe()
            self._said = True
            log.info(message)
        else:
            log.debug(message)

    def _start(self) -> _Running:
        from ..engine import _SPAWNER, _die_with_parent, _free_port

        path = self.path
        if path is None or not path.is_file():
            raise LittleError("the model file is not there")
        runtime = self._runtime()
        if runtime is None:
            raise LittleError("llama.cpp was not found")
        port, key = _free_port(), secrets.token_urlsafe(24)
        args = server_args(runtime, path, port=port)
        env = {**runtime.env, "LLAMA_API_KEY": key}
        log_file = self._open_log()

        def child() -> None:  # pragma: no cover - runs in the forked child
            _die_with_parent()
            try:
                os.nice(NICE)
            except OSError:
                pass

        def spawn() -> subprocess.Popen[bytes]:
            return subprocess.Popen(
                args, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT, env=env,
                cwd=str(Path(runtime.path).parent), preexec_fn=child, close_fds=True,
            )

        began = time.monotonic()
        try:
            process = _SPAWNER.run(spawn)
        except OSError as exc:
            raise LittleError(f"llama.cpp could not start: {exc}") from exc
        finally:
            log_file.close()
        try:
            self._wait_ready(process, port, key)
        except LittleError:
            self._kill(process)
            raise
        elapsed = int((time.monotonic() - began) * 1000)
        running = _Running(process, port, key, time.time(), elapsed, threading.Event())
        threading.Thread(target=self._reap, args=(running,), name="kiwibro-little-idle", daemon=True).start()
        log.info("little helper started (%s, pid %d, port %d, %d threads, CPU) in %d ms",
                 self._model_name, process.pid, port, THREADS, elapsed)
        return running

    def _open_log(self) -> Any:
        path = self.log_path
        if path is None:
            return open(os.devnull, "ab")  # noqa: SIM115 - the child owns it from here
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if path.exists() and path.stat().st_size > LOG_LIMIT:
                path.write_bytes(path.read_bytes()[-LOG_LIMIT // 4:])
        except OSError:
            pass
        return path.open("ab")

    def _wait_ready(self, process: subprocess.Popen[bytes], port: int, key: str) -> None:
        deadline = time.monotonic() + READY_TIMEOUT_S
        while process.poll() is None and time.monotonic() < deadline:
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", headers=_auth(key), timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.25)
        raise LittleError(f"the little helper did not start (log: {self.log_path})")

    def _reap(self, running: _Running) -> None:
        """Stop the server once it has had nothing to do for ``idle_stop_s``."""
        while not running.gone.wait(5.0):
            if running.process.poll() is not None:
                return
            self.stop_idle(self._idle_stop_s)

    # --- stopping --------------------------------------------------------------------

    def stop_idle(self, min_idle_s: float) -> int:
        with self._lock:
            if not self.alive() or self._in_flight or time.monotonic() - self._last_used < min_idle_s:
                return 0
        self.stop()
        return 1

    def stop(self) -> None:
        with self._lock:
            running, self._running = self._running, None
        if running is not None:
            running.gone.set()
            self._kill(running.process)
            log.info("little helper stopped (pid %d)", running.pid)

    @staticmethod
    def _kill(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(STOP_WAIT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(STOP_WAIT_S)


def server_args(runtime: LlamaRuntime, model: Path, *, port: int) -> list[str]:
    """The llama-server command line: CPU only, two threads, one slot, a small window (options the build lacks are
    left out)."""
    def has(flag: str) -> bool:
        return not runtime.flags or flag in runtime.flags

    args = [runtime.path, "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
            "--ctx-size", str(CTX), "--threads", str(THREADS), "--parallel", "1"]
    if has("--threads-batch"):
        args += ["--threads-batch", str(THREADS)]
    if runtime.offload is not None and has("--device"):  # a GPU build: keep it off the chip that writes the replies
        args += ["--device", "none"]
    args += ["--gpu-layers", "0"] if has("--gpu-layers") else ["--n-gpu-layers", "0"]
    for flag in ("--jinja", "--no-webui", "--no-warmup"):
        if has(flag):
            args.append(flag)
    if has("--cache-ram"):
        args += ["--cache-ram", "0"]  # a one-slot server that drops a prompt would otherwise keep it in memory
    return args
