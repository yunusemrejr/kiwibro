"""Native E4B drafter registration and launch policy."""

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from kiwibro import engine
from kiwibro.mind import native_generic as native
from kiwibro.mind.pack import MANIFEST, TRUNK3, PackError
from tests.test_engine import T_STRING, T_UINT32, write_gguf
from tests.test_mind_native_generic import pack_at

DRAFT = "mtp.gguf"
FLAGS = frozenset({
    "--device", "--mmproj", "--no-mmproj-offload", "--spec-type", "--spec-draft-model",
    "--spec-draft-n-max", "--spec-draft-ngl", "--spec-draft-device", "--lora-scaled",
})
RUNTIME = engine.Runtime("/unused/llama-server", "b12443", "cpu", "kiwibro")
ATTENTION = {
    "head_count_kv": 2, "key_length": 512, "value_length": 512,
    "key_length_swa": 256, "value_length_swa": 256, "sliding_window": 512,
}


def file_entry(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def mtp_pack(tmp_path, monkeypatch):
    pack = pack_at(tmp_path, base=native.E4B_BASE)
    trunk = write_gguf(pack.path(TRUNK3), overrides={
        "general.architecture": (T_STRING, "gemma4"),
        "gemma4.block_count": (T_UINT32, 42),
        "gemma4.embedding_length": (T_UINT32, 2560),
        "gemma4.attention.shared_kv_layers": (T_UINT32, 18),
        **{f"gemma4.attention.{key}": (T_UINT32, value) for key, value in ATTENTION.items()},
        "tokenizer.ggml.tokens": (T_UINT32, 262144),
    })
    draft = write_gguf(pack.path(DRAFT), overrides={
        "general.architecture": (T_STRING, "gemma4-assistant"),
        "general.type": (T_STRING, "model"),
        "gemma4-assistant.embedding_length_out": (T_UINT32, 2560),
        **{f"gemma4-assistant.attention.{key}": (T_UINT32, value) for key, value in ATTENTION.items()},
        "tokenizer.ggml.tokens": (T_UINT32, 262144),
    })
    manifest = pack.manifest()
    manifest.update(arch="gemma4", base_revision=native.E4B_BASE_REVISION)
    manifest["files"][TRUNK3] = file_entry(trunk)
    manifest["files"][DRAFT] = file_entry(draft)
    manifest["mtp"] = {"file": DRAFT, "trunk_sha256": manifest["files"][TRUNK3]["sha256"]}
    pack.path(MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(native, "E4B_MTP_SHA256", manifest["files"][DRAFT]["sha256"])
    monkeypatch.setattr(native, "E4B_MTP_BYTES", manifest["files"][DRAFT]["bytes"])
    return pack


def build(model, *, runtime=RUNTIME, flags=FLAGS, offload=None, speculative=True, parallel=1, adapters=()):
    return engine.server_args(runtime, flags, model, port=24355, api_key="test", context_length=8192,
                              parallel=parallel, threads=6, cache_ram_mb=0, speculative=speculative,
                              offload=offload, adapters=adapters)


def test_verified_mtp_precedes_ngram_and_keeps_the_projector(mtp_pack, monkeypatch):
    monkeypatch.setenv("KIWIMIND_CPU_DRAFT", "1")
    model = native.describe_pack(mtp_pack).model
    assert model.pack_mtp == str(mtp_pack.path(DRAFT))
    flags = FLAGS | {flag for flag, _ in engine.NATIVE_CPU_DRAFT}
    args = build(model, flags=flags)
    assert args[args.index("--spec-type") + 1] == "draft-mtp"
    for flag, value in (("--spec-draft-model", model.pack_mtp), ("--spec-draft-n-max", "2"),
                        ("--spec-draft-ngl", "0"), ("--spec-draft-device", "none"), ("--mmproj", model.projector)):
        assert args[args.index(flag) + 1] == value
    assert "--no-mmproj-offload" in args
    assert all(flag not in args for flag, _ in engine.NATIVE_CPU_DRAFT)
    gpu_build_cpu_args = build(model, runtime=replace(RUNTIME, kind="vulkan"), offload=False)
    assert gpu_build_cpu_args[gpu_build_cpu_args.index("--device") + 1] == "none"
    assert "--spec-draft-model" in gpu_build_cpu_args


@pytest.mark.parametrize("options", [
    {"runtime": replace(RUNTIME, version="b12442")},
    {"runtime": replace(RUNTIME, source="custom")},
    {"runtime": replace(RUNTIME, kind="vulkan")},
    {"runtime": replace(RUNTIME, kind="vulkan"), "offload": True},
    {"flags": FLAGS - {"--spec-draft-n-max"}},
    {"flags": FLAGS - {"--no-mmproj-offload"}},
    {"flags": frozenset()},
    {"parallel": 2},
    {"speculative": False},
    {"adapters": (SimpleNamespace(path="/task-adapter.gguf", scale=0.8),)},
])
def test_mtp_is_skipped_outside_the_verified_profile(mtp_pack, monkeypatch, options):
    monkeypatch.delenv("KIWIMIND_CPU_DRAFT", raising=False)
    args = build(native.describe_pack(mtp_pack).model, **options)
    assert "--spec-draft-model" not in args and "--spec-type" not in args
    assert "--mmproj" in args


def test_zero_scale_lora_and_prior_text_only_defaults(mtp_pack):
    model = native.describe_pack(mtp_pack).model
    zero = SimpleNamespace(path="/task-adapter.gguf", scale=0.0)
    assert "--spec-draft-model" in build(model, adapters=(zero,))
    args = build(replace(model, projector=None, pack_mtp=None))
    assert args[args.index("--spec-type") + 1] == "ngram-mod"
    assert "--no-mmproj-offload" not in args


@pytest.mark.parametrize(("key", "value", "problem"), [
    ("mtp", {"file": "loose.gguf"}, "checksummed"),
    ("mtp", {"file": TRUNK3}, "separate"),
    ("mtp", {"file": "vision.gguf"}, "separate"),
    ("mtp", {"file": DRAFT, "trunk_sha256": "0" * 64}, "exact SHA-256"),
    ("base", "google/gemma-4-E2B-it", "pinned Gemma 4 E4B"),
    ("base_revision", "0" * 40, "pinned Gemma 4 E4B"),
])
def test_invalid_mtp_manifest_is_rejected(mtp_pack, key, value, problem):
    manifest = mtp_pack.manifest()
    manifest[key] = value
    mtp_pack.path(MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PackError, match=problem):
        native.describe_pack(mtp_pack)


def test_mtp_rejects_wrong_artifact_or_interface(mtp_pack, monkeypatch):
    model = native.describe_pack(mtp_pack).model
    monkeypatch.setattr(native, "E4B_MTP_SHA256", "0" * 64)
    with pytest.raises(PackError, match="official E4B Q8_0"):
        native.describe_pack(mtp_pack)
    monkeypatch.setattr(native, "E4B_MTP_SHA256", file_entry(mtp_pack.path(DRAFT))["sha256"])
    manifest = mtp_pack.manifest()
    trunk = write_gguf(mtp_pack.path(TRUNK3), overrides={
        "general.architecture": (T_STRING, model.arch),
        "gemma4.embedding_length": (T_UINT32, 1536),
    })
    manifest["files"][TRUNK3] = file_entry(trunk)
    manifest["mtp"]["trunk_sha256"] = manifest["files"][TRUNK3]["sha256"]
    mtp_pack.path(MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PackError, match="shared-state interface"):
        native.describe_pack(mtp_pack)


@pytest.mark.parametrize("failure", [engine._Crashed, engine.EngineError])
def test_mtp_startup_failure_retries_without_a_draft(mtp_pack, tmp_path, monkeypatch, failure):
    model = native.describe_pack(mtp_pack).model
    direct = engine.DirectEngine(lambda: {"direct_threads": 6}, tmp_path)
    direct._flags[RUNTIME.path] = FLAGS
    monkeypatch.setattr(direct, "_runtimes", lambda: [RUNTIME])
    monkeypatch.setattr(direct, "adapters_for", lambda _model: ())
    monkeypatch.setattr(direct, "_write_pid", lambda _server: None)
    monkeypatch.setattr("kiwibro.engine._SPAWNER", SimpleNamespace(run=lambda spawn: spawn()))
    launches, killed = [], []
    monkeypatch.setattr(direct, "_kill", killed.append)

    def popen(args, **_kwargs):
        launches.append(args)
        return SimpleNamespace()

    def wait_ready(_server):
        if len(launches) == 1:
            raise failure("draft startup failed")

    monkeypatch.setattr("kiwibro.engine.subprocess.Popen", popen)
    monkeypatch.setattr(direct, "_wait_ready", wait_ready)
    try:
        assert direct._start(model, 8192, 1).model is model
        assert len(launches) == 2 and len(killed) == 1
        assert "--spec-draft-model" in launches[0]
        assert "--spec-type" not in launches[1] and "--mmproj" in launches[1]
    finally:
        direct.shutdown()
