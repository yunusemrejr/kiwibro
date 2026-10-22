"""Deterministic fake LM Studio server for Kiwibro end-to-end tests.

It speaks the same OpenAI-compatible wire format the real LM Studio server does
(``GET /api/v0/models``, ``POST /v1/chat/completions`` with SSE streaming), but
the model's behaviour is scripted by markers in the test prompt:

    "hello [[scenario:plain]]"        -> a canned streamed answer
    "[[scenario:tool_write]]"         -> streams a write_file tool call,
                                         then a final answer after the result
    "[[scenario:context_400_once]]"   -> first stream returns an HTTP 400 that
                                         looks like an LM Studio context-limit
                                         error, the retry succeeds

Non-streaming requests (title generation, memory extraction, summarisation) are
answered by inspecting the system prompt, like a well-behaved model would.

Run standalone for browser tests::

    python3 tests/e2e/fake_lm.py --port 0 --print-url
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

SCENARIO_RE = re.compile(r"\[\[scenario:([a-z0-9_]+)\]\]")

DEFAULT_MODELS: list[dict[str, Any]] = [
    {
        "id": "qwen3.8-2b",
        "object": "model",
        "type": "llm",
        "publisher": "fake-workshop",
        "arch": "qwen3",
        "quantization": "Q4_K_M",
        "state": "loaded",
        "max_context_length": 262144,
        "loaded_context_length": 32768,
        "capabilities": ["tool_use"],
    },
    {
        "id": "spark-x2.6-4b",
        "object": "model",
        "type": "llm",
        "publisher": "fake-workshop",
        "arch": "spark",
        "quantization": "IQ2_M",
        "state": "not-loaded",
        "max_context_length": 1048576,
        "capabilities": ["tool_use"],
    },
    {
        "id": "tiny-context",
        "object": "model",
        "type": "llm",
        "publisher": "fake-workshop",
        "arch": "tiny",
        "quantization": "Q8_0",
        "state": "loaded",
        "max_context_length": 4096,
        "loaded_context_length": 2048,
        "capabilities": ["tool_use"],
    },
    {
        "id": "embedding-kiwi",
        "object": "model",
        "type": "embeddings",
        "publisher": "fake-workshop",
        "state": "loaded",
        "max_context_length": 8192,
        "capabilities": [],
    },
]

MODELS_NO_TOOLS: list[dict[str, Any]] = [
    {
        "id": "plain-model",
        "object": "model",
        "type": "llm",
        "publisher": "fake-workshop",
        "state": "loaded",
        "max_context_length": 8192,
        "loaded_context_length": 8192,
        "capabilities": [],
    }
]

MODELS_TINY_CONTEXT: list[dict[str, Any]] = [
    {
        "id": "tiny-context",
        "object": "model",
        "type": "llm",
        "publisher": "fake-workshop",
        "state": "loaded",
        "max_context_length": 4096,
        "loaded_context_length": 2048,
        "capabilities": ["tool_use"],
    }
]

MARKDOWN_REPLY = (
    "Here is a **bold** answer with `inline code`.\n\n"
    "## Steps\n"
    "1. First step\n"
    "2. Second step and a [link](https://example.com/kiwi)\n\n"
    "```python\nprint('kiwi')\n```\n\n"
    "- a bullet\n"
    "> a quote\n"
)

UTF8_REPLY = "Grüße — 世界 🥝 café ✓ 100% done."

# scenario -> (tool name, arguments, final message after the tool result)
TOOL_SCENARIOS: dict[str, tuple[str, dict[str, Any], str]] = {
    "tool_write": (
        "write_file",
        {"path": "notes/hello.txt", "content": "Kiwi says hi 🥝\n"},
        "I wrote the file `notes/hello.txt`.",
    ),
    "tool_write_deny": (
        "write_file",
        {"path": "denied.txt", "content": "this must never be written"},
        "I could not write the file.",
    ),
    "tool_write_traversal": (
        "write_file",
        {"path": "../../pwned.txt", "content": "escape"},
        "I could not write outside the workspace.",
    ),
    "tool_run": (
        "run_command",
        {"command": "printf 'kiwi\\n'"},
        "The command finished.",
    ),
    "tool_run_blocklist": (
        "run_command",
        {"command": "sudo rm -rf /"},
        "The command was refused.",
    ),
    "tool_read_traversal": (
        "read_file",
        {"path": "../../../etc/passwd"},
        "I could not read that file.",
    ),
    "tool_fetch_ssrf": (
        "fetch_page",
        {"url": "http://169.254.169.254/latest/meta-data/"},
        "I could not fetch that page.",
    ),
    "tool_unknown": ("bogus_tool", {"x": 1}, "I used an unknown tool."),
    "tool_bad_args": ("web_search", {"query": ""}, "The search argument was empty."),
    "tool_remember": (
        "remember",
        {"content": "The QA kiwi is named Pip and likes concise answers.", "kind": "fact"},
        "I will remember that.",
    ),
}

ALWAYS_SCENARIOS = {"tool_always_flow"}

CONTEXT_SCENARIOS = {"context_400_once": 1, "context_400_twice": 2, "context_400_always": 99}


def _chunk(text: str, size: int = 7) -> list[str]:
    if not text:
        return []
    return [text[index : index + size] for index in range(0, len(text), size)]


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages or []):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    str(part.get("text", "")) for part in content if isinstance(part, dict)
                )
    return ""


def _scenario(payload: dict[str, Any]) -> str:
    match = SCENARIO_RE.search(_last_user_text(payload.get("messages") or []))
    return match.group(1) if match else "plain"


def _tool_stage(payload: dict[str, Any]) -> int:
    return sum(1 for message in payload.get("messages") or [] if message.get("role") == "tool")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "FakeLMStudio/1.0"

    # these are set by the factory
    fake: "FakeLM"

    def log_message(self, *args: Any) -> None:  # silence test output
        return

    # -- routing -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/v0/models":
            self._json(200, {"object": "list", "data": self.fake.models})
        elif path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": m["id"]} for m in self.fake.models]})
        elif path == "/health":
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except ValueError:
            self._json(400, {"error": {"message": "invalid json"}})
            return
        if not isinstance(payload, dict):
            self._json(400, {"error": {"message": "invalid body"}})
            return
        self.fake.record(payload)
        scenario = _scenario(payload)
        self.fake.count_scenario(scenario)
        if not payload.get("stream"):
            self._non_stream(payload, scenario)
            return
        self._stream(payload, scenario)

    # -- helpers -----------------------------------------------------------

    def _json(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def _sse_start(self) -> None:
        self.send_response(200)
        # Deliberately no charset, exactly like LM Studio's event stream.
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

    def _sse(self, data: Any) -> bool:
        line = "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"
        try:
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def _sse_done(self) -> None:
        self._sse("[DONE]")
        self.close_connection = True

    # -- non-streaming -----------------------------------------------------

    def _non_stream(self, payload: dict[str, Any], scenario: str) -> None:
        content = self._utility_text(payload) or "Fake completion text."
        body = {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model") or "fake",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 21, "completion_tokens": 9, "total_tokens": 30},
        }
        self._json(200, body)

    @staticmethod
    def _utility_text(payload: dict[str, Any]) -> str | None:
        """The answer to a title / memory / summary prompt (streamed or not), else None."""
        messages = payload.get("messages") or []
        system = ""
        for message in messages:
            if message.get("role") == "system":
                system = str(message.get("content") or "")
                break
        user_text = ""
        for message in messages:
            if message.get("role") == "user":
                user_text += str(message.get("content") or "")
        content = None
        lowered = system.lower()
        if "write chat titles" in lowered or "name conversations" in lowered:
            if "[[scenario:title_echo]]" in user_text:
                content = user_text  # a bad model echoing the prompt back
            else:
                content = "Kiwi Test Title"
        elif "memory keeper" in lowered or "extract durable memory" in lowered:
            content = json.dumps(
                {
                    "ops": [
                        {
                            "op": "add",
                            "kind": "preference",
                            "content": "Prefers brief German answers — Grüße 🥝",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        elif "rolling summary" in lowered:
            content = "Summary: the user talked about kiwis, files and the fake model."
        return content

    # -- streaming ---------------------------------------------------------

    def _stream(self, payload: dict[str, Any], scenario: str) -> None:
        utility = self._utility_text(payload)
        if utility is not None:  # Kiwibro streams cancellable utility calls (memory extraction)
            self._sse_start()
            for part in _chunk(utility, 12):
                if not self._sse({"choices": [{"index": 0, "delta": {"content": part}}]}):
                    return
            self._sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            self._sse_done()
            return
        self.fake.record_scenario_request(scenario, payload)
        model = payload.get("model") or "fake"

        # Simulated context-limit failures recoverable by the backend.
        limit = CONTEXT_SCENARIOS.get(scenario)
        if limit is not None:
            attempts = self.fake.scenario_count(scenario)
            if attempts <= limit:
                self._json(
                    400,
                    {
                        "error": {
                            "message": "The prompt is too long: context length exceeded. "
                            "Please reduce the input tokens.",
                            "type": "invalid_request_error",
                            "code": "context_length_exceeded",
                        }
                    },
                )
                return

        # Simulated "tools unsupported" 400 for the first tool-bearing request.
        if scenario == "tools_reject_once" and payload.get("tools"):
            self._json(
                400,
                {"error": {"message": "tools are not supported by this model"}},
            )
            return

        self._sse_start()
        delay = self.fake.stream_delay
        chunks: list[tuple[str, Any]] = []
        usage = {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49}
        finish_reason = "stop"

        if scenario in ("stop_slow", "slow_then_answer"):
            delay = 0.05 if scenario == "stop_slow" else 0.03
            reps = 80 if scenario == "stop_slow" else 30
            for index in range(reps):
                chunks.append(("content", f"tick {index} "))
        elif scenario == "utf8":
            for part in _chunk(UTF8_REPLY, 4):
                chunks.append(("content", part))
        elif scenario == "markdown":
            for part in _chunk(MARKDOWN_REPLY, 9):
                chunks.append(("content", part))
        elif scenario == "reasoning":
            for part in _chunk("Let me think about the kiwi first. ", 6):
                chunks.append(("reasoning", part))
            for part in _chunk("The kiwi is a bird.", 5):
                chunks.append(("content", part))
        elif scenario == "empty":
            usage = None
        elif scenario == "no_usage":
            for part in _chunk("No usage information here.", 5):
                chunks.append(("content", part))
            usage = None
        elif scenario in TOOL_SCENARIOS or scenario in ALWAYS_SCENARIOS:
            stage = _tool_stage(payload)
            call = self._choose_tool_call(scenario, stage)
            if call is None:
                text = self._final_text(scenario, stage)
                for part in _chunk(text, 6):
                    chunks.append(("content", part))
                finish_reason = "stop"
            else:
                name, arguments = call
                chunks.append(("tool", (f"call_fake_{stage}", name, arguments)))
                finish_reason = "tool_calls"
        else:  # plain / context recovery / tools_reject_once second attempt
            if scenario.startswith("context_400"):
                text = "Recovered after context compaction."
            elif scenario == "tools_reject_once":
                text = "Text answer without tools."
            else:
                text = "Hello from the fake model."
            for part in _chunk(text, 6):
                chunks.append(("content", part))

        if scenario == "slow_then_answer" and self.fake.stream_delay > 0:
            delay = 0.03

        for kind, value in chunks:
            if kind == "content":
                okay = self._sse(
                    {
                        "id": "chatcmpl-fake",
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": value}, "finish_reason": None}],
                    }
                )
            elif kind == "reasoning":
                okay = self._sse(
                    {
                        "id": "chatcmpl-fake",
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [
                            {"index": 0, "delta": {"reasoning_content": value}, "finish_reason": None}
                        ],
                    }
                )
            else:  # tool call: emit id/name first, then split the arguments
                call_id, name, arguments = value
                okay = self._sse(
                    {
                        "id": "chatcmpl-fake",
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": call_id,
                                            "type": "function",
                                            "function": {"name": name, "arguments": ""},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                argument_text = json.dumps(arguments, ensure_ascii=False)
                for fragment in _chunk(argument_text, 9):
                    if not okay:
                        break
                    okay = self._sse(
                        {
                            "id": "chatcmpl-fake",
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "function": {"arguments": fragment},
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
            if not okay:
                return
            if delay:
                time.sleep(delay)

        self._sse(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
        )
        if usage is not None:
            self._sse(
                {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [],
                    "usage": usage,
                }
            )
        self._sse_done()

    # -- scenario scripts --------------------------------------------------

    @staticmethod
    def _choose_tool_call(scenario: str, stage: int) -> tuple[str, dict[str, Any]] | None:
        if scenario == "tool_always_flow":
            if stage == 0:
                return "write_file", {"path": "always-1.txt", "content": "first"}
            if stage == 1:
                return "write_file", {"path": "always-2.txt", "content": "second"}
            return None
        name, arguments, _final = TOOL_SCENARIOS[scenario]
        if stage == 0:
            return name, arguments
        return None

    @staticmethod
    def _final_text(scenario: str, stage: int) -> str:
        if scenario == "tool_always_flow":
            return "Both files written with your permission."
        name = TOOL_SCENARIOS.get(scenario)
        if name is None:
            return "Done."
        final = name[2]
        if scenario == "tool_write_traversal":
            return final + " The workspace sandbox refused it."
        return final


class FakeLM:
    """Threaded fake LM Studio server. ``url`` is usable as KIWIBRO_LMSTUDIO_URL."""

    def __init__(
        self,
        models: list[dict[str, Any]] | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        stream_delay: float = 0.003,
    ) -> None:
        self.models = copy.deepcopy(models if models is not None else DEFAULT_MODELS)
        self.stream_delay = stream_delay
        self.requests: list[dict[str, Any]] = []
        self.scenario_requests: list[dict[str, Any]] = []
        self._scenario_counts: dict[str, int] = {}
        self._lock = threading.Lock()

        handler = type("Handler", (_Handler,), {"fake": self})
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self.httpd.block_on_close = False
        self.port = int(self.httpd.server_address[1])
        self.url = f"http://{host}:{self.port}"
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "FakeLM":
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="fake-lm", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        try:
            self.httpd.shutdown()
        except Exception:
            pass
        try:
            self.httpd.server_close()
        except Exception:
            pass

    def __enter__(self) -> "FakeLM":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- bookkeeping -------------------------------------------------------

    def record(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.requests.append(payload)
            if len(self.requests) > 500:
                del self.requests[:100]

    def record_scenario_request(self, scenario: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self.scenario_requests.append({"scenario": scenario, "payload": payload})

    def count_scenario(self, scenario: str) -> None:
        with self._lock:
            self._scenario_counts[scenario] = self._scenario_counts.get(scenario, 0) + 1

    def scenario_count(self, scenario: str) -> int:
        with self._lock:
            return self._scenario_counts.get(scenario, 0)

    def reset(self) -> None:
        with self._lock:
            self.requests.clear()
            self.scenario_requests.clear()
            self._scenario_counts.clear()

    # -- queries used by checks -------------------------------------------

    def chat_requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return [payload for payload in self.requests if payload.get("stream")]

    def completion_requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return [payload for payload in self.requests if not payload.get("stream")]

    def last_chat_request(self) -> dict[str, Any] | None:
        chats = self.chat_requests()
        return chats[-1] if chats else None

    def last_system_prompt(self) -> str:
        request = self.last_chat_request()
        if not request:
            return ""
        for message in request.get("messages") or []:
            if message.get("role") == "system":
                return str(message.get("content") or "")
        return ""

    def last_request_has_tools(self) -> bool | None:
        request = self.last_chat_request()
        if request is None:
            return None
        return bool(request.get("tools"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fake LM Studio server for Kiwibro e2e tests.")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--models",
        choices=("default", "no-tools", "tiny-context"),
        default="default",
    )
    parser.add_argument("--print-url", action="store_true")
    parser.add_argument("--delay", type=float, default=0.003)
    args = parser.parse_args()
    models = {
        "default": DEFAULT_MODELS,
        "no-tools": MODELS_NO_TOOLS,
        "tiny-context": MODELS_TINY_CONTEXT,
    }[args.models]
    fake = FakeLM(models, host=args.host, port=args.port, stream_delay=args.delay).start()
    if args.print_url:
        print(json.dumps({"url": fake.url, "port": fake.port}), flush=True)
    else:
        print(f"fake LM Studio on {fake.url}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        fake.stop()


if __name__ == "__main__":
    main()
