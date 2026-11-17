"""Managed native LoRA registration, startup fallback, and request routing."""

from __future__ import annotations

import csv
import hashlib
import json
from types import SimpleNamespace

import pytest

from kiwibro.engine import DirectEngine, EngineError, Runtime
from kiwibro.llm import ADAPTERS_OFF, LLMClient
from kiwibro.mind import KIWIMIND
from kiwibro.mind.native_generic import NATIVE_MODEL, describe_pack
from kiwibro.mind.pack import MANIFEST, TRUNK3, Pack, PackError
from scripts.mind.stage_native_pack import ADAPTER_FILE, stage
from tests.test_engine import T_STRING, T_UINT32, write_gguf
from tests.test_mind_native_generic import pack_at, text_reply


@pytest.fixture
def artifacts(tmp_path):
    trunk = write_gguf(tmp_path / "model.gguf", overrides={
        "general.architecture": (T_STRING, "gemma4"),
        "gemma4.context_length": (T_UINT32, 8192),
        "gemma4.embedding_length": (T_UINT32, 2816),
        "gemma4.expert_count": (T_UINT32, 80),
        "gemma4.expert_used_count": (T_UINT32, 6),
    })
    vision = write_gguf(tmp_path / "projector.gguf", overrides={
        "general.architecture": (T_STRING, "clip"),
        "clip.vision.projection_dim": (T_UINT32, 2816),
    })
    lora = write_gguf(tmp_path / "repo-lora.gguf", overrides={
        "general.architecture": (T_STRING, "gemma4"),
        "general.name": (T_STRING, "Repository helper"),
        "general.type": (T_STRING, "adapter"),
        "adapter.type": (T_STRING, "lora"),
    })
    return trunk, vision, lora


@pytest.fixture
def adapter_pack(tmp_path, artifacts):
    trunk, vision, lora = artifacts
    pack = Pack(tmp_path, subdir="candidate")
    stage(trunk, vision, pack.dir, version="test", context=8192, adapter=lora, adapter_tasks=("code",),
          adapter_scale=0.8, adapter_trunk_digest=hashlib.sha256(trunk.read_bytes()).hexdigest())
    return pack


def test_stage_binds_exact_trunk_and_verifies_the_adapter_payload(adapter_pack, artifacts, tmp_path):
    trunk, vision, lora = artifacts
    with pytest.raises(ValueError, match="exact SHA-256"):
        stage(trunk, vision, tmp_path / "other", version="test", context=8192,
              adapter=lora, adapter_tasks=("code",), adapter_trunk_digest="0" * 64)
    assert not (tmp_path / "other").exists()
    manifest = adapter_pack.manifest()
    assert manifest["lora"]["trunk_sha256"] == manifest["files"][TRUNK3]["sha256"]
    assert manifest["files"][ADAPTER_FILE]["sha256"] == hashlib.sha256(lora.read_bytes()).hexdigest()
    (adapter,) = describe_pack(adapter_pack).model.pack_adapters
    assert (adapter.path, adapter.arch, adapter.tasks, adapter.scale) == (
        str(adapter_pack.path(ADAPTER_FILE)), "gemma4", ("code",), 0.8,
    )
    # Keep the exact byte count so this uses the checksum verifier.
    raw = lora.read_bytes()
    lora.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(PackError, match="damaged"):
        describe_pack(adapter_pack)


@pytest.mark.parametrize(("edit", "message"), [
    ({"file": "unregistered.gguf"}, "checksummed"),
    ({"file": TRUNK3}, "separate"),
    ({"trunk_sha256": "0" * 64}, "exact trunk"),
    ({"tasks": ["code", "other"]}, "task labels"),
    ({"scale": True}, "scale"),
    ({"scale": float("nan")}, "scale"),
])
def test_native_pack_rejects_bad_adapter_registration(adapter_pack, edit, message):
    manifest = adapter_pack.manifest()
    manifest["lora"].update(edit)
    adapter_pack.path(MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PackError, match=message):
        describe_pack(adapter_pack)


@pytest.mark.parametrize("overrides", [
    {"general.architecture": (T_STRING, "llama"), "general.type": (T_STRING, "adapter"),
     "adapter.type": (T_STRING, "lora")},
    {"general.architecture": (T_STRING, "gemma4"), "general.type": (T_STRING, "model")},
])
def test_stage_rejects_wrong_adapter_type_or_architecture(tmp_path, artifacts, overrides):
    trunk, vision, lora = artifacts
    write_gguf(lora, overrides=overrides)
    with pytest.raises(ValueError, match="GGUF adapter"):
        stage(trunk, vision, tmp_path / "staged", version="test", context=8192,
              adapter=lora, adapter_tasks=("code",),
              adapter_trunk_digest=hashlib.sha256(trunk.read_bytes()).hexdigest())


def test_managed_server_uses_only_pack_adapter_and_retries_trunk(adapter_pack, tmp_path, monkeypatch):
    model = describe_pack(adapter_pack).model
    engine = DirectEngine(lambda: {}, tmp_path, managed={NATIVE_MODEL: lambda: model})
    runtime = Runtime("/unused/llama-server", "b12443", "vulkan", "custom")
    monkeypatch.setattr(engine, "_runtimes", lambda: [runtime])

    def unexpected_discovery():
        pytest.fail("A managed pack attempted loose adapter discovery")

    monkeypatch.setattr(engine, "adapters", unexpected_discovery)
    launches = []

    def launch(runtime, model, context, parallel, speculative, loras):
        launches.append(loras)
        if loras:
            raise EngineError("adapter startup timed out")
        return SimpleNamespace(adapters=())

    monkeypatch.setattr(engine, "_launch", launch)
    monkeypatch.setattr(engine, "_kill", lambda _server: None)
    try:
        assert engine.ensure(NATIVE_MODEL).adapters == ()
        assert launches == [model.pack_adapters, ()]
        assert engine.adapters_for(model) == ()
    finally:
        engine.shutdown()


@pytest.mark.parametrize(("subdir", "flags", "enabled"), [
    ("candidate", {"--lora-scaled"}, True),
    ('with,"punctuation"', {"--lora-scaled"}, True),
    ("with:colon", {"--lora-scaled"}, False),
    ("candidate", {"--lora", "--lora-init-without-apply"}, False),
    ("candidate", set(), False),
], ids=["global-zero", "csv-path", "colon-path", "legacy-flags", "unknown-flags"])
def test_managed_launch_exposes_only_adapters_loaded_at_global_zero(
    adapter_pack, tmp_path, monkeypatch, subdir, flags, enabled,
):
    pack = Pack(tmp_path, subdir=subdir)
    if pack.dir != adapter_pack.dir:
        adapter_pack.dir.rename(pack.dir)
    model = describe_pack(pack).model
    engine = DirectEngine(lambda: {"direct_threads": 2}, tmp_path)
    runtime = Runtime("/unused/llama-server", "b12443", "cpu", "custom")
    engine._flags[runtime.path] = frozenset(flags)
    process = SimpleNamespace(pid=1)
    launches = []

    def popen(args, **_kwargs):
        launches.append(args)
        return process

    monkeypatch.setattr("kiwibro.engine.gpu_devices", lambda _runtime: [])
    monkeypatch.setattr("kiwibro.engine._SPAWNER", SimpleNamespace(run=lambda spawn: spawn()))
    monkeypatch.setattr("kiwibro.engine.subprocess.Popen", popen)
    monkeypatch.setattr(engine, "_write_pid", lambda _server: None)
    monkeypatch.setattr(engine, "_wait_ready", lambda _server: None)
    server = engine._launch(runtime, model, 8192, 1, loras=model.pack_adapters)

    (args,) = launches
    assert "--lora" not in args and "--lora-init-without-apply" not in args
    if enabled:
        field = args[args.index("--lora-scaled") + 1]
        assert next(csv.reader([field])) == [f"{pack.path(ADAPTER_FILE)}:0"]
        assert server.adapters == model.pack_adapters
    else:
        assert "--lora-scaled" not in args
        assert server.adapters == ()


def test_public_native_alias_routes_adapter_and_keeps_utility_requests_unadapted(adapter_pack, tmp_path, monkeypatch):
    monkeypatch.setenv("KIWIMIND_CANDIDATE", "1")
    client = LLMClient(lambda: {"server_url": "http://127.0.0.1:9998"}, data_dir=tmp_path)
    runner = client.direct_engine()
    monkeypatch.setattr(client.mind(), "runtime", lambda: object())
    expected = describe_pack(adapter_pack).model.pack_adapters
    server = SimpleNamespace(url="http://127.0.0.1:9999", adapters=expected, alive=lambda: True)
    monkeypatch.setattr(runner, "_servers", {NATIVE_MODEL: server})
    monkeypatch.setattr(runner, "acquire", lambda model: runner._servers[model])
    monkeypatch.setattr(runner, "release", lambda server: None)
    monkeypatch.setattr(runner, "_kill", lambda _server: None)
    requests = []

    def post(url, *, json, **_kwargs):
        assert url == "http://127.0.0.1:9999/v1/chat/completions"
        requests.append(json)
        return text_reply("Done.")

    monkeypatch.setattr("kiwibro.llm.requests.post", post)
    code = [{"role": "user", "content": "Fix the bug in this Python function."}]
    writing = [{"role": "user", "content": "Draft a short poem."}]
    assert client.has_adapters(KIWIMIND)
    assert client.adapters_on(KIWIMIND, code) == list(expected)
    assert not client.has_adapters("attached-model")
    client.stream_chat(model=KIWIMIND, messages=code)
    client.stream_chat(model=KIWIMIND, messages=writing)
    client.stream_chat(model=KIWIMIND, messages=code, extra={ADAPTERS_OFF: True})
    client.complete(model=KIWIMIND, messages=code)
    client.blame_adapters(list(expected))
    client.blame_adapters(list(expected))
    client.stream_chat(model=KIWIMIND, messages=code)
    assert [row["lora"] for row in requests] == [
        [{"id": 0, "scale": scale}] for scale in (0.8, 0.0, 0.0, 0.0, 0.4)
    ]
    assert all(ADAPTERS_OFF not in row for row in requests)


def test_adapter_free_pack_ignores_loose_adapter_files(tmp_path, artifacts, monkeypatch):
    model = describe_pack(pack_at(tmp_path / "old")).model
    engine = DirectEngine(lambda: {}, tmp_path, managed={NATIVE_MODEL: lambda: model})
    monkeypatch.setenv("KIWIBRO_ADAPTERS_DIR", str(artifacts[2].parent))
    assert model.pack_adapters == engine.adapters_for(model) == ()
