"""Task adapters: LoRA discovery, fit, task detection and the per-request scales."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiwibro import adapters, sense
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


# --- adapters that adjust themselves ---------------------------------------------------


@pytest.mark.skipif(sense.shared() is None, reason="Kiwi Sense is not available")
def test_a_message_without_task_words_is_read_by_meaning():
    assert adapters.classify("why does my loop never end") == "code"
    assert adapters.classify("make this paragraph sound friendlier") == "writing"
    for small_talk in ("hello", "how are you doing today", "thanks a lot", "I feel tired", "tell me a joke"):
        assert adapters.classify(small_talk) == ""  # a wrong adapter is worse than none


def test_an_adapter_that_spoils_replies_is_turned_down_then_off(tmp_path: Path):
    coder = adapters.Adapter("/a", "coder", "x", tasks=("code",), scale=0.8)
    house = adapters.Adapter("/b", "house style", "x", tasks=("*",))
    health = adapters.Health(tmp_path / "health.json")
    scales = lambda: [s["scale"] for s in adapters.request_scales((coder, house), "code", health.counts())]  # noqa: E731
    assert scales() == [0.8, 1.0]
    health.note("/a")
    assert scales() == [0.8, 1.0]  # once can be chance
    assert health.note("/a") == adapters.TROUBLE_HALF
    assert scales() == [0.4, 1.0]
    health.note("/a")
    assert health.note("/a") == adapters.TROUBLE_OFF
    assert scales() == [0.0, 1.0]
    assert adapters.Health(tmp_path / "health.json").trouble("/a") == 4  # kept across restarts
    health.reset("/a")
    assert scales() == [0.8, 1.0] and adapters.Health(tmp_path / "health.json").counts() == {}
    assert [s["scale"] for s in adapters.request_scales((coder, house), "code", off=True)] == [0.0, 0.0]


def test_adapter_trouble_ages_out(tmp_path: Path, monkeypatch):
    health = adapters.Health(tmp_path / "health.json")
    health.note("/a")
    health.note("/a")
    assert health.trouble("/a") == 2
    later = adapters.time.time() + (adapters.TROUBLE_DAYS + 1) * 86400
    monkeypatch.setattr(adapters.time, "time", lambda: later)
    assert health.trouble("/a") == 0 and health.counts() == {}
    assert health.note("/a") == 1  # the old ones are dropped, not carried


def test_library_lists_adapters_and_can_reset_and_delete_them(env, monkeypatch):
    svc = env.svc
    monkeypatch.setattr(svc.llm, "direct_on", lambda: True)
    engine = svc.llm.direct_engine()
    own = lora(svc.data_dir / "adapters" / "tiny-code.gguf", name="Tiny coder")
    (svc.data_dir / "adapters" / "tiny-code.json").write_text(json.dumps({"tasks": ["code"], "scale": 0.8}))
    engine.adapters(force=True)
    listed = env.client.get("/api/library").get_json()["adapters"]
    assert [(a["name"], a["tasks"], a["scale"], a["strength"], a["deletable"]) for a in listed] == [
        ("Tiny coder", ["code"], 0.8, 1.0, True)]
    assert listed[0]["size_bytes"] == own.stat().st_size

    engine.adapter_health.note(str(own))
    engine.adapter_health.note(str(own))
    turned_down = env.client.get("/api/library").get_json()["adapters"][0]
    assert (turned_down["trouble"], turned_down["strength"]) == (2, 0.5)
    assert env.client.post("/api/library/adapters/reset", json={"path": str(own)}).get_json() == {"ok": True}
    assert env.client.get("/api/library").get_json()["adapters"][0]["strength"] == 1.0

    assert env.client.delete("/api/library/adapters", json={"path": "/etc/passwd"}).status_code == 404
    removed = env.client.delete("/api/library/adapters", json={"path": str(own)})
    assert removed.get_json() == {"ok": True, "freed_bytes": listed[0]["size_bytes"]}
    assert not own.exists() and not (svc.data_dir / "adapters" / "tiny-code.json").exists()
    assert env.client.get("/api/library").get_json()["adapters"] == []
