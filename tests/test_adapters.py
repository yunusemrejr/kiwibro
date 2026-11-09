"""Task adapters: LoRA discovery, fit, task detection and the per-request scales."""

from __future__ import annotations

import json
from pathlib import Path

from kiwibro import adapters
from kiwibro.engine import DirectEngine, LocalModel, Runtime, find_runtimes, model_dirs, server_args
from tests.test_engine import T_STRING, write_gguf, write_script


def lora(path: Path, name: str = "", base: str = "") -> Path:
    extra = {"general.type": (T_STRING, "adapter"), "adapter.type": (T_STRING, "lora")}
    if name:
        extra["general.name"] = (T_STRING, name)
    if base:
        extra["general.base_model.0.name"] = (T_STRING, base)
    return write_gguf(path, overrides=extra)


def test_classify_reads_the_task_from_the_users_words():
    assert adapters.classify("fix the bug in this python function") == "code"
    assert adapters.classify("translate this into English") == "translate"
    assert adapters.classify("summarize the key points") == "summarize"
    assert adapters.classify("hello there") == ""


def test_follow_up_keeps_the_task_and_the_turn_note_is_ignored():
    chat = [{"role": "user", "content": "write a python script that sorts files"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "ok do it\n\n[note] translate summarize search essay"}]
    assert adapters.task_of(chat) == "code"
    assert adapters.task_of([{"role": "user", "content": "hi"}]) == ""


def test_scan_finds_adapters_and_names_their_tasks(tmp_path: Path):
    lora(tmp_path / "tiny-coder-lora.gguf")
    lora(tmp_path / "mystery.gguf")
    lora(tmp_path / "styled.gguf")
    (tmp_path / "styled.json").write_text(json.dumps({"tasks": ["writing", "bogus"], "scale": 0.5}))
    write_gguf(tmp_path / "model.gguf")  # a model, not an adapter
    found = {Path(a.path).name: a for a in adapters.scan([tmp_path])}
    assert set(found) == {"tiny-coder-lora.gguf", "mystery.gguf", "styled.gguf"}
    assert found["tiny-coder-lora.gguf"].tasks == ("code",)
    assert found["mystery.gguf"].tasks == ()
    assert (found["styled.gguf"].tasks, found["styled.gguf"].scale) == (("writing",), 0.5)


def test_fit_needs_the_same_architecture_and_base(tmp_path: Path):
    adapter = adapters.read_adapter(lora(tmp_path / "a-code.gguf", base="Spark X2.6"))
    assert adapter is not None
    assert adapters.fits(adapter, adapter.arch, "spark-x2.6-4b")
    assert not adapters.fits(adapter, adapter.arch, "qwen3.8-2b")
    assert not adapters.fits(adapter, "other-arch", "spark-x2.6-4b")


def test_request_scales_switch_one_adapter_on():
    loaded = (adapters.Adapter("/a", "coder", "x", tasks=("code",)), adapters.Adapter("/b", "writer", "x", tasks=("writing",), scale=0.7),
              adapters.Adapter("/c", "house style", "x", tasks=("*",)), adapters.Adapter("/d", "unknown", "x"))
    assert adapters.request_scales(loaded, "writing") == [
        {"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.7}, {"id": 2, "scale": 1.0}, {"id": 3, "scale": 0.0}]
    assert [s["scale"] for s in adapters.request_scales(loaded, "")] == [0.0, 0.0, 1.0, 0.0]


def test_server_args_load_adapters_unapplied_and_sleep_when_idle():
    model = LocalModel("m", "/m.gguf", 1, "p", "r")
    runtime = Runtime("/bin/llama-server", "1", "cpu", "custom")
    loaded = (adapters.Adapter("/a.gguf", "coder", "x", tasks=("code",)),)
    args = server_args(runtime, frozenset(), model, port=1, api_key="k", context_length=4096, parallel=2, threads=2,
                       cache_ram_mb=None, adapters=loaded, idle_sleep_s=600)
    assert args[args.index("--lora") + 1] == "/a.gguf" and "--lora-init-without-apply" in args
    assert args[args.index("--sleep-idle-seconds") + 1] == "600"
    old = server_args(runtime, frozenset({"--jinja"}), model, port=1, api_key="k", context_length=4096, parallel=2,
                      threads=2, cache_ram_mb=None, adapters=loaded, idle_sleep_s=600)
    assert "--lora" not in old and "--sleep-idle-seconds" not in old


def test_kiwibro_owns_a_models_and_a_runtime_folder(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("KIWIBRO_MODELS_DIR", raising=False)
    monkeypatch.delenv("KIWIBRO_LLAMA_SERVER", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nopath"))
    data = tmp_path / "data"
    engine = DirectEngine(lambda: {}, data)
    assert (data / "models").is_dir() and (data / "adapters").is_dir() and (data / "runtime").is_dir()
    assert (data / "models").resolve() in model_dirs(None, data)
    assert find_runtimes(data_dir=data) == []
    server = write_script(data / "runtime" / "llama-server", "print('ok')\n")
    assert [r.path for r in find_runtimes(data_dir=data)] == [str(server)]
    write_gguf(data / "models" / "tiny.gguf")
    lora(data / "adapters" / "tiny-code.gguf")
    assert len(engine.models(force=True)) == 1  # the adapter is not listed as a model
    model = engine.models()[0]
    assert [a.path for a in engine.adapters_for(model)] == [str(data / 'adapters' / 'tiny-code.gguf')]
    assert engine.lora_for(model.id, [{"role": "user", "content": "fix this code"}]) is None  # not running
    engine.shutdown()
