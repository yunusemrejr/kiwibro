"""Tests for the direct engine (kiwibro.engine), its GGUF header reader and the
direct-mode parts of kiwibro.llm and kiwibro.config.

Hermetic: LM Studio's folders, the llama.cpp search path and KIWIBRO_LLAMA_SERVER
are all replaced, and the only "llama-server" is a fake Python script that serves
the OpenAI-style routes on 127.0.0.1. No network, no real LM Studio, no bubblewrap.
"""

from __future__ import annotations

import copy
import json
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

import pytest

from kiwibro import config, engine, gguf
from kiwibro.engine import DirectEngine, LocalModel, Runtime, adapt_payload, find_runtimes, server_args
from kiwibro.llm import LMStudioClient, LMStudioError

# --- GGUF writer ------------------------------------------------------------------------

T_UINT32, T_FLOAT32, T_STRING, T_ARRAY = 4, 6, 8, 9

CHAT_TEMPLATE = (
    "{% if tools %}<tool_list>{{ tools | tojson }}</tool_list>{% endif %}"
    "{% if enable_thinking is defined %}<think>{% endif %}"
    "{% for message in messages %}{{ message['content'] }}{% endfor %}"
)

# key -> (value type, value): a small chat model with thinking and tools.
TINY_KV: dict[str, tuple[int, Any]] = {
    "general.architecture": (T_STRING, "llama"),
    "general.name": (T_STRING, "Tiny"),
    "general.size_label": (T_STRING, "1B"),
    "general.file_type": (T_UINT32, 15),  # LLAMA_FTYPE_MOSTLY_Q4_K_M
    "llama.context_length": (T_UINT32, 4096),
    "tokenizer.chat_template": (T_STRING, CHAT_TEMPLATE),
    "tokenizer.ggml.tokens": (T_ARRAY, (T_STRING, ["<s>", "hello", "world"])),
    "tokenizer.ggml.scores": (T_ARRAY, (T_FLOAT32, [0.0, -1.0, -2.0])),
}


def _gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf_value(kind: int, value: Any) -> bytes:
    if kind == T_STRING:
        return _gguf_string(value)
    if kind == T_UINT32:
        return struct.pack("<I", value)
    if kind == T_FLOAT32:
        return struct.pack("<f", value)
    if kind == T_ARRAY:
        item_kind, items = value
        return struct.pack("<IQ", item_kind, len(items)) + b"".join(_gguf_value(item_kind, i) for i in items)
    raise AssertionError(f"unsupported test value type {kind}")


def write_gguf(
    path: Path,
    *,
    overrides: dict[str, tuple[int, Any]] | None = None,
    omit: tuple[str, ...] = (),
    version: int = 3,
) -> Path:
    """A minimal GGUF header (no tensors). ``overrides`` replace or add keys, ``omit`` drops them."""
    kv = {key: spec for key, spec in TINY_KV.items() if key not in omit}
    kv.update(overrides or {})
    body = bytearray(b"GGUF" + struct.pack("<IQQ", version, 0, len(kv)))
    for key, (kind, value) in kv.items():
        body += _gguf_string(key) + struct.pack("<I", kind) + _gguf_value(kind, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(body))
    return path


def pad(path: Path, extra: int) -> None:
    """Grow a file with trailing bytes (tensor data stand-in); the header reader ignores them."""
    with open(path, "ab") as handle:
        handle.write(b"\0" * extra)


def write_script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(0o755)
    return path


def install_backend(root: Path, folder: str, name: str, version: str, server_source: Path | None = None) -> Path:
    """One LM Studio llama.cpp engine: an executable llama-server plus its manifest."""
    engine_dir = root / "extensions" / "backends" / folder
    engine_dir.mkdir(parents=True, exist_ok=True)
    server = engine_dir / "llama-server"
    if server_source is None:
        server.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    else:
        shutil.copyfile(server_source, server)
    server.chmod(0o755)
    (engine_dir / "backend-manifest.json").write_text(json.dumps({"name": name, "version": version}), encoding="utf-8")
    return server


# --- a fake llama-server ----------------------------------------------------------------

HELP_TEXT = (
    "usage: llama-server [options]\n"
    "  --api-key KEY       API key for the server\n"
    "  --alias NAME        model name reported by the API\n"
    "  --jinja             use the model's own chat template\n"
    "  --parallel N        number of slots\n"
    "  --kv-unified        one KV pool shared by the slots\n"
    "  --threads-batch N   threads for prompt processing\n"
    "  --no-webui          disable the web UI\n"
    "  --ctx-size N        prompt context size\n"
)

_FAKE_SERVER_CODE = r'''
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SSE_LINES = [
    '{"choices":[{"delta":{"content":"Hello"}}]}',
    '{"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}',
    '{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}',
]
USAGE = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
API_KEY = ""
ALIAS = ""


def record(value):
    path = os.environ.get("FAKE_LLAMA_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(value) + "\n")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ok"})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": "not found"})
            return
        if self.headers.get("Authorization") != "Bearer " + API_KEY:
            self.send_json(401, {"error": "invalid api key"})
            return
        record(body)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for line in SSE_LINES:
                self.wfile.write(("data: " + line + "\n\n").encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self.send_json(200, {
            "model": ALIAS,
            "choices": [{"message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
            "usage": USAGE,
        })


def main():
    global API_KEY, ALIAS
    argv = sys.argv[1:]
    if "--help" in argv:
        sys.stdout.write(HELP)
        return 0

    def option(name, default=""):
        return argv[argv.index(name) + 1] if name in argv else default

    port = int(option("--port"))
    API_KEY = option("--api-key")
    ALIAS = option("--alias")
    record(sys.argv)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

FAKE_LLAMA_BODY = "HELP = " + repr(HELP_TEXT) + "\n" + _FAKE_SERVER_CODE

# Like an old llama.cpp build: it answers --help, then dies while loading any model.
CRASHING_BODY = f"""
import sys
if "--help" in sys.argv:
    sys.stdout.write({HELP_TEXT!r})
    sys.exit(0)
print("error: unknown model architecture", file=sys.stderr)
sys.exit(1)
"""


def read_calls(path: Path) -> list[Any]:
    """Lines the fake server appended: argv lists (one per start) and request bodies."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- fixtures ----------------------------------------------------------------------------


@pytest.fixture
def models_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolation: no LM Studio roots, no extra folders, no PATH llama-server, one tmp models folder."""
    folder = tmp_path / "models"
    folder.mkdir()
    monkeypatch.setattr(engine, "LMSTUDIO_ROOTS", ())
    monkeypatch.setattr(engine, "EXTRA_MODEL_DIRS", ())
    monkeypatch.setenv("KIWIBRO_MODELS_DIR", str(folder))
    monkeypatch.delenv("KIWIBRO_LLAMA_SERVER", raising=False)
    monkeypatch.setattr(engine.shutil, "which", lambda *args, **kwargs: None)
    return folder


@pytest.fixture
def fake_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log = tmp_path / "fake-calls.log"
    monkeypatch.setenv("FAKE_LLAMA_LOG", str(log))
    return log


def add_tiny_model(models: Path) -> None:
    """pub/tiny: a 32k-context chat model (LM Studio's JIT-style start uses the full trained window)."""
    write_gguf(
        models / "Pub" / "Tiny-GGUF" / "Tiny-Q4_K_M.gguf",
        overrides={"llama.context_length": (T_UINT32, 32768)},
    )


@pytest.fixture
def direct_client(tmp_path: Path, models_root: Path, fake_calls: Path):
    add_tiny_model(models_root)
    script = write_script(tmp_path / "bin" / "llama-server", FAKE_LLAMA_BODY)
    settings = {"engine": "direct", "direct_runtime": str(script), "model": None}
    client = LMStudioClient(lambda: settings, data_dir=tmp_path / "data")
    try:
        yield client
    finally:
        client.stop_direct()  # no fake server outlives the test, even when an assert failed


def scan() -> list[LocalModel]:
    return DirectEngine(lambda: {}, None).models(force=True)


# --- 1. GGUF header reader -------------------------------------------------------------


def test_gguf_header_is_read_and_summarised(tmp_path: Path):
    path = write_gguf(tmp_path / "tiny.gguf")

    meta = gguf.read_metadata(path)
    assert meta["gguf.version"] == 3
    assert meta["general.architecture"] == "llama"
    assert meta["llama.context_length"] == 4096
    assert meta["tokenizer.ggml.tokens"] == 3  # arrays come back as their element count
    assert meta["tokenizer.ggml.scores"] == 3

    facts = gguf.summarize(meta)
    assert facts["arch"] == "llama"
    assert facts["context_length"] == 4096
    assert facts["size_label"] == "1B"
    assert facts["name"] == "Tiny"
    assert facts["quantization"] == "Q4_K_M"
    assert facts["tools"] is True
    assert facts["thinking"] is True
    assert facts["thinking_toggle"] is True
    assert facts["embeddings"] is False
    assert facts["projector"] is False


def test_non_gguf_file_raises(tmp_path: Path):
    path = tmp_path / "notes.gguf"
    path.write_bytes(b"this is a text file, not a model at all")
    with pytest.raises(gguf.GGUFError):
        gguf.read_metadata(path)


def test_truncated_header_raises(tmp_path: Path):
    full = write_gguf(tmp_path / "full.gguf").read_bytes()
    cut = tmp_path / "cut.gguf"
    cut.write_bytes(full[:20])  # magic, version, tensor count, and half of the key count
    with pytest.raises(gguf.GGUFError):
        gguf.read_metadata(cut)


def test_header_truncated_inside_a_skipped_array_raises(tmp_path: Path):
    full = write_gguf(tmp_path / "full.gguf").read_bytes()
    cut = tmp_path / "cut.gguf"
    cut.write_bytes(full[:-10])  # the last key is the float32 scores array
    with pytest.raises(gguf.GGUFError):
        gguf.read_metadata(cut)


# --- 2. scanning LM Studio's folders ---------------------------------------------------------


def test_single_quant_model_gets_publisher_id(models_root: Path):
    write_gguf(models_root / "Pub" / "Some-Model-GGUF" / "Some-Model-Q4_K_M.gguf")

    models = scan()

    assert [m.id for m in models] == ["pub/some-model"]
    assert models[0].kind == "llm"
    assert models[0].quantization == "Q4_K_M"


def test_two_quantisations_in_one_folder_get_quant_suffixes(models_root: Path):
    folder = models_root / "Pub" / "X"
    write_gguf(folder / "X-Q4_K_M.gguf")
    write_gguf(folder / "X-Q8_0.gguf", overrides={"general.file_type": (T_UINT32, 7)})  # LLAMA_FTYPE_MOSTLY_Q8_0

    models = scan()

    assert sorted(m.id for m in models) == ["pub/x@q4_k_m", "pub/x@q8_0"]
    assert {m.quantization for m in models} == {"Q4_K_M", "Q8_0"}


def test_split_shards_are_one_model_sized_as_the_sum(models_root: Path):
    folder = models_root / "Pub" / "Big"
    first = write_gguf(folder / "big-00001-of-00002.gguf")
    second = write_gguf(folder / "big-00002-of-00002.gguf")
    pad(first, 1000)
    pad(second, 3000)

    models = scan()

    assert [m.id for m in models] == ["pub/big"]
    assert Path(models[0].path).name == "big-00001-of-00002.gguf"
    assert models[0].size_bytes == first.stat().st_size + second.stat().st_size


def test_projector_makes_a_vlm_and_is_not_listed_as_a_model(models_root: Path):
    folder = models_root / "Pub" / "Vis"
    write_gguf(folder / "Vis-Q4_K_M.gguf")
    projector = write_gguf(
        folder / "mmproj-Vis-F16.gguf",
        overrides={"general.architecture": (T_STRING, "clip")},
        omit=("tokenizer.chat_template",),
    )

    models = scan()

    assert [m.id for m in models] == ["pub/vis"]
    assert models[0].kind == "vlm"
    assert Path(models[0].projector).name == projector.name


def test_embedding_model_has_kind_embeddings(models_root: Path):
    write_gguf(
        models_root / "Pub" / "Embed" / "Embed-F16.gguf",
        overrides={"llama.pooling_type": (T_UINT32, 1)},
        omit=("tokenizer.chat_template",),
    )

    models = scan()

    assert [m.id for m in models] == ["pub/embed"]
    assert models[0].embeddings is True
    assert models[0].kind == "embeddings"


def test_unreadable_gguf_file_is_skipped(models_root: Path):
    folder = models_root / "Pub" / "Some-Model"
    write_gguf(folder / "Some-Model-Q4_K_M.gguf")
    (folder / "broken-Q2_K.gguf").write_bytes(b"not really a model")

    assert [m.id for m in scan()] == ["pub/some-model"]


def test_split_shard_size_does_not_absorb_another_quant(models_root: Path):
    folder = models_root / "Pub" / "Big"
    for quant, file_type in (("Q2_K", 10), ("Q2_K_S", 21)):  # LLAMA_FTYPE_MOSTLY_Q2_K / _Q2_K_S
        for index in (1, 2):
            name = f"big-{quant}-{index:05d}-of-00002.gguf"
            write_gguf(folder / name, overrides={"general.file_type": (T_UINT32, file_type)})
    shard = (folder / "big-Q2_K-00001-of-00002.gguf").stat().st_size

    sizes = {m.quantization: m.size_bytes for m in scan()}

    assert sizes == {"Q2_K": 2 * shard, "Q2_K_S": 2 * shard}


def test_split_model_in_flat_folder_keeps_its_plain_id(models_root: Path):
    write_gguf(models_root / "big-00001-of-00002.gguf")
    write_gguf(models_root / "big-00002-of-00002.gguf")

    assert [m.id for m in scan()] == ["big"]


# --- 3. find() and canonical() ---------------------------------------------------------------


def test_find_resolves_exact_id_alias_and_name_part(models_root: Path):
    write_gguf(models_root / "Pub" / "Some-Model" / "Some-Model-Q4_K_M.gguf")
    eng = DirectEngine(lambda: {}, None)

    assert eng.find("pub/some-model").id == "pub/some-model"
    assert eng.find("PUB/Some-Model").id == "pub/some-model"  # case-insensitive
    assert eng.find("some-model").id == "pub/some-model"  # repo name without the publisher
    assert eng.find("otherpublisher/some-model").id == "pub/some-model"  # name part of an unknown publisher
    assert eng.canonical("otherpublisher/some-model") == "pub/some-model"
    assert eng.find("nothing-like-this") is None
    assert eng.find(None) is None
    assert eng.find("") is None


def test_ambiguous_repo_name_does_not_resolve_by_name_alone(models_root: Path):
    write_gguf(models_root / "Pub" / "Some-Model" / "Some-Model-Q4_K_M.gguf")
    write_gguf(models_root / "Other" / "Some-Model" / "Some-Model-Q4_K_M.gguf")
    eng = DirectEngine(lambda: {}, None)

    assert sorted(m.id for m in eng.models()) == ["other/some-model", "pub/some-model"]
    assert eng.find("pub/some-model").id == "pub/some-model"
    assert eng.find("third/some-model") is None
    assert eng.canonical("third/some-model") is None


# --- 4. adapt_payload -------------------------------------------------------------------


def test_adapt_payload_drops_lmstudio_only_keys_and_sets_prompt_cache():
    payload = {"model": "m", "messages": [], "ttl": 300, "draft_model": "d", "temperature": 0.2}
    original = copy.deepcopy(payload)

    out = adapt_payload(payload)

    assert "ttl" not in out and "draft_model" not in out
    assert out["cache_prompt"] is True
    assert out["temperature"] == 0.2
    assert payload == original  # the caller's dict is not mutated


def test_adapt_payload_reasoning_none_switches_thinking_off():
    out = adapt_payload({"model": "m", "reasoning_effort": "none"})

    assert out["reasoning_effort"] == "none"
    assert out["chat_template_kwargs"] == {"enable_thinking": False}


def test_adapt_payload_other_effort_becomes_thinking_on():
    out = adapt_payload({"model": "m", "reasoning_effort": "medium"})

    assert "reasoning_effort" not in out
    assert out["chat_template_kwargs"] == {"enable_thinking": True}


def test_adapt_payload_keeps_existing_template_kwargs():
    out = adapt_payload({"model": "m", "reasoning_effort": "none", "chat_template_kwargs": {"foo": 1}})

    assert out["chat_template_kwargs"] == {"foo": 1, "enable_thinking": False}


# --- 5. server_args -------------------------------------------------------------------------

RUNTIME = Runtime(path="/opt/llama/llama-server", version="2.10.0", kind="cpu", source="lmstudio")
ALL_OPTIONAL_FLAGS = frozenset({
    "--api-key", "--alias", "--jinja", "--parallel", "--kv-unified", "--threads-batch", "--no-webui",
    "--ctx-size", "--flash-attn", "--reasoning-format", "--cache-reuse", "--cache-ram", "--metrics", "--mmproj",
})


def chat_model(**changes: Any) -> LocalModel:
    values: dict[str, Any] = {
        "id": "pub/some-model", "path": "/models/pub/some-model/Some-Model-Q4_K_M.gguf",
        "size_bytes": 1, "publisher": "pub", "repo": "Some-Model",
    }
    values.update(changes)
    return LocalModel(**values)


def build(model: LocalModel | None = None, flags: frozenset[str] = frozenset(), **changes: Any) -> list[str]:
    options: dict[str, Any] = {
        "port": 8123, "api_key": "secret", "context_length": 8192, "parallel": 4,
        "threads": 6, "cache_ram_mb": None,
    }
    options.update(changes)
    return server_args(RUNTIME, flags, model or chat_model(), **options)


def value_after(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_server_args_with_unknown_help_includes_every_optional_flag():
    args = build(chat_model(projector="/models/mmproj.gguf"), frozenset(), cache_ram_mb=2048)

    for flag in ("--threads-batch", "--api-key", "--kv-unified", "--jinja", "--flash-attn",
                 "--reasoning-format", "--cache-reuse", "--cache-ram", "--no-webui", "--metrics", "--mmproj"):
        assert flag in args, flag


def test_server_args_omits_flags_the_runtime_does_not_list():
    flags = ALL_OPTIONAL_FLAGS - {"--kv-unified", "--cache-ram"}

    args = build(flags=flags, cache_ram_mb=2048)

    assert "--kv-unified" not in args
    assert "--cache-ram" not in args
    assert "--jinja" in args and "--flash-attn" in args


def test_server_args_never_adds_kv_unified_for_one_slot():
    args = build(flags=ALL_OPTIONAL_FLAGS, parallel=1)

    assert "--kv-unified" not in args
    assert value_after(args, "--parallel") == "1"


def test_server_args_adds_mmproj_for_a_projector():
    args = build(chat_model(projector="/models/pub/mmproj-some.gguf"), ALL_OPTIONAL_FLAGS)

    assert value_after(args, "--mmproj") == "/models/pub/mmproj-some.gguf"


def test_server_args_for_embeddings_is_single_slot_embedding_server():
    args = build(chat_model(embeddings=True), ALL_OPTIONAL_FLAGS)

    assert "--embeddings" in args
    assert value_after(args, "--parallel") == "1"
    assert args.count("--parallel") == 1
    assert "--kv-unified" not in args and "--jinja" not in args


def test_server_args_carries_key_alias_context_and_binding():
    model = chat_model()
    args = build(model, ALL_OPTIONAL_FLAGS, api_key="s3cret", context_length=8192, port=9001)

    assert args[0] == RUNTIME.path
    assert value_after(args, "--model") == model.path
    assert value_after(args, "--host") == "127.0.0.1"
    assert value_after(args, "--port") == "9001"
    assert value_after(args, "--api-key") == "s3cret"
    assert value_after(args, "--alias") == model.id
    assert value_after(args, "--ctx-size") == "8192"


# --- 6. find_runtimes ----------------------------------------------------------------------


@pytest.fixture
def fake_lmstudio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, models_root: Path) -> Path:
    root = tmp_path / "lmstudio"
    monkeypatch.setattr(engine, "LMSTUDIO_ROOTS", (str(root),))
    install_backend(root, "llama.cpp-linux-x86_64-avx2-2.10.0", "llama.cpp-linux-x86_64-avx2", "2.10.0")
    install_backend(root, "llama.cpp-linux-x86_64-avx2-2.20.0", "llama.cpp-linux-x86_64-avx2", "2.20.0")
    install_backend(root, "llama.cpp-linux-x86_64-vulkan-avx2-2.30.0",
                    "llama.cpp-linux-x86_64-vulkan-avx2", "2.30.0")
    return root


def test_find_runtimes_newest_version_first_without_preferences(fake_lmstudio: Path):
    runtimes = find_runtimes()

    assert [r.version for r in runtimes] == ["2.30.0", "2.20.0", "2.10.0"]
    assert runtimes[0].kind == "vulkan"
    assert all(r.source == "lmstudio" for r in runtimes)


def test_find_runtimes_prefers_the_engine_lmstudio_is_set_to_use(fake_lmstudio: Path):
    (fake_lmstudio / ".internal").mkdir()
    (fake_lmstudio / ".internal" / "backend-preferences-v1.json").write_text(json.dumps([
        {"model_format": "gguf", "name": "llama.cpp-linux-x86_64-avx2", "version": "2.10.0"},
    ]), encoding="utf-8")

    runtimes = find_runtimes()

    assert runtimes[0].version == "2.10.0"
    assert runtimes[0].path.endswith("llama.cpp-linux-x86_64-avx2-2.10.0/llama-server")
    assert [r.version for r in runtimes[1:]] == ["2.30.0", "2.20.0"]


def test_custom_runtime_ranks_before_lmstudio_engines(fake_lmstudio: Path, tmp_path: Path):
    custom = write_script(tmp_path / "custom-build" / "llama-server", "exit 0\n")

    runtimes = find_runtimes(str(custom))

    assert runtimes[0].path == str(custom)
    assert runtimes[0].source == "custom"
    assert [r.version for r in runtimes[1:]] == ["2.30.0", "2.20.0", "2.10.0"]


def test_custom_runtime_that_is_not_executable_is_ignored(fake_lmstudio: Path, tmp_path: Path):
    plain = tmp_path / "not-a-server"
    plain.write_text("just text", encoding="utf-8")
    plain.chmod(0o644)

    runtimes = find_runtimes(str(plain))

    assert all(r.source == "lmstudio" for r in runtimes)


# --- 7. end to end with the fake llama-server ----------------------------------------------------


def test_direct_engine_serves_chat_and_completion_then_loads_and_unloads(direct_client, fake_calls):
    client = direct_client

    listed = {m["id"]: m for m in client.list_models(force=True)}
    assert listed["pub/tiny"]["state"] == "not-loaded"

    result = client.stream_chat(model="pub/tiny", messages=[{"role": "user", "content": "hi"}])
    assert result.content == "Hello world"

    listed = {m["id"]: m for m in client.list_models(force=True)}
    assert listed["pub/tiny"]["state"] == "loaded"

    chat_bodies = [entry for entry in read_calls(fake_calls) if isinstance(entry, dict)]
    assert chat_bodies, "the fake server logged no request body"
    assert all(body["cache_prompt"] is True for body in chat_bodies)

    completion = client.complete(model="pub/tiny", messages=[{"role": "user", "content": "ping"}])
    assert completion.text == "pong"

    server = client.direct_engine().loaded("pub/tiny")
    assert server is not None
    process = server.process
    assert client.unload_model("pub/tiny") is True
    assert process.poll() is not None  # the llama-server child has exited
    assert client.direct_engine().loaded("pub/tiny") is None

    assert client.load_model("pub/tiny", 8192, parallel=2) is True
    last_start = [entry for entry in read_calls(fake_calls) if isinstance(entry, list)][-1]
    assert value_after(last_start, "--ctx-size") == "8192"
    assert value_after(last_start, "--parallel") == "2"


def test_unknown_model_id_raises_engine_error(direct_client):
    with pytest.raises(LMStudioError) as caught:
        direct_client.stream_chat(model="nobody/missing", messages=[{"role": "user", "content": "hi"}])

    assert caught.value.kind == "engine"


# --- 8. runtime fallback --------------------------------------------------------------------------


def test_load_falls_back_to_the_next_runtime_when_one_crashes(
    tmp_path: Path, models_root: Path, fake_calls: Path, monkeypatch: pytest.MonkeyPatch,
):
    add_tiny_model(models_root)
    working = write_script(tmp_path / "working" / "llama-server", FAKE_LLAMA_BODY)
    crashing = write_script(tmp_path / "crashing" / "llama-server", CRASHING_BODY)
    lmstudio = tmp_path / "lmstudio"
    installed = install_backend(
        lmstudio, "llama.cpp-linux-x86_64-avx2-2.10.0", "llama.cpp-linux-x86_64-avx2", "2.10.0",
        server_source=working,
    )
    monkeypatch.setattr(engine, "LMSTUDIO_ROOTS", (str(lmstudio),))
    settings = {"engine": "direct", "direct_runtime": str(crashing), "model": None}
    client = LMStudioClient(lambda: settings, data_dir=tmp_path / "data")
    try:
        assert client.load_model("pub/tiny") is True
        server = client.direct_engine().loaded("pub/tiny")
        assert server is not None
        assert server.runtime.path == str(installed)
    finally:
        client.stop_direct()


def test_server_handed_out_by_acquire_is_never_evicted(tmp_path: Path, models_root: Path):
    for name in ("a", "b", "c"):
        write_gguf(models_root / "Pub" / f"{name}-GGUF" / f"{name}-Q4_K_M.gguf")
    script = write_script(tmp_path / "bin" / "llama-server", FAKE_LLAMA_BODY)
    eng = DirectEngine(lambda: {"direct_runtime": str(script)}, tmp_path / "data")
    try:
        held = eng.acquire("pub/a")  # handed out already marked busy
        eng.ensure("pub/b")
        eng.ensure("pub/c")  # MAX_RESIDENT reached: only an idle server may be stopped
        assert held.alive()
        eng.release(held)
    finally:
        eng.shutdown()


def test_unload_does_not_stop_a_server_with_a_request_in_flight(tmp_path: Path, models_root: Path):
    add_tiny_model(models_root)
    script = write_script(tmp_path / "bin" / "llama-server", FAKE_LLAMA_BODY)
    eng = DirectEngine(lambda: {"direct_runtime": str(script)}, tmp_path / "data")
    try:
        server = eng.acquire("pub/tiny")  # a chat reply is streaming from this server
        assert eng.unload("pub/tiny") is False
        assert server.alive()
        eng.release(server)
        assert eng.unload("pub/tiny") is True
    finally:
        eng.shutdown()


# --- 9. config.validate_settings_patch ---------------------------------------------------------


def test_engine_setting_accepts_only_known_engines():
    assert config.validate_settings_patch({"engine": "direct"}, {}) == {"engine": "direct"}
    with pytest.raises(ValueError):
        config.validate_settings_patch({"engine": "x"}, {})


@pytest.mark.parametrize("value", [0, 8])
def test_direct_threads_accepts_zero_and_in_range(value: int):
    assert config.validate_settings_patch({"direct_threads": value}, {}) == {"direct_threads": value}


@pytest.mark.parametrize("value", [-1, 300, True, "4"])
def test_direct_threads_rejects_out_of_range_and_non_integers(value: Any):
    with pytest.raises(ValueError):
        config.validate_settings_patch({"direct_threads": value}, {})


def test_direct_runtime_empty_means_automatic():
    assert config.validate_settings_patch({"direct_runtime": ""}, {}) == {"direct_runtime": ""}


def test_direct_runtime_must_be_an_absolute_path():
    with pytest.raises(ValueError):
        config.validate_settings_patch({"direct_runtime": "bin/llama-server"}, {})


def test_direct_runtime_must_be_executable(tmp_path: Path):
    plain = tmp_path / "llama-server"
    plain.write_text("text", encoding="utf-8")
    plain.chmod(0o644)
    with pytest.raises(ValueError):
        config.validate_settings_patch({"direct_runtime": str(plain)}, {})


def test_direct_runtime_accepts_an_executable_file(tmp_path: Path):
    server = write_script(tmp_path / "llama-server", "exit 0\n")

    cleaned = config.validate_settings_patch({"direct_runtime": str(server)}, {})

    assert cleaned == {"direct_runtime": str(server)}


def test_direct_models_dir_must_be_an_existing_folder(tmp_path: Path):
    assert config.validate_settings_patch({"direct_models_dir": str(tmp_path)}, {}) == {
        "direct_models_dir": str(tmp_path),
    }
    with pytest.raises(ValueError):
        config.validate_settings_patch({"direct_models_dir": str(tmp_path / "missing")}, {})
