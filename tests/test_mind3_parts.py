"""The pure parts of the Kiwi Mind 3 runtime: rendering, llama-server arguments, routes and pack files.

Nothing here starts llama-server, node or a network call: the files are read and strings are compared."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kiwibro.mind import conductor3, llama, render3
from kiwibro.mind.client import MindRequestError
from kiwibro.mind.pack import CHECK_CACHE, Pack

ZWSP = "​"  # what render3 puts inside text that would read as prompt structure


# ---------------------------------------------------------------------------------------------- routes


def test_route_text_parses_back_for_every_skill() -> None:
    for skill in render3.SKILLS:
        for heavy in (False, True):
            for act in (False, True):
                text = render3.route_text(skill, heavy, act)
                assert text == f"{skill} {'heavy' if heavy else 'light'} {'act' if act else 'say'}"
                assert render3.parse_route(text) == (skill, heavy, act)


def test_parse_route_rejects_junk() -> None:
    assert render3.parse_route("") is None
    assert render3.parse_route("hello") is None
    assert render3.parse_route("code light act now") is None
    assert render3.parse_route("code medium say") is None
    assert render3.parse_route("paint light say") is None


def test_route_grammar_exact_strings() -> None:
    every = " | ".join(f'"{name}"' for name in render3.SKILLS)
    assert render3.route_grammar() == (
        f'root ::= ({every}) " " ("light" | "heavy") " " ("say" | "act")'
    )
    assert render3.route_grammar("code") == (
        'root ::= ("code") " " ("light" | "heavy") " " ("say" | "act")'
    )
    assert render3.route_grammar(weigh=False) == (
        f'root ::= ({every}) " " ("light") " " ("say" | "act")'
    )
    assert render3.route_grammar("code", weigh=False) == (
        'root ::= ("code") " " ("light") " " ("say" | "act")'
    )


# ---------------------------------------------------------------------------------------------- render


def test_render_one_user_message_opens_with_bos_and_system_role() -> None:
    prompt = render3.render([{"role": "user", "content": "hi"}])
    assert prompt.startswith(render3.BOS + render3.SYSTEM_ROLE + "\n")
    assert render3.SYSTEM in prompt
    assert prompt.endswith(render3.BOT + render3.THINK)
    assert prompt == (
        render3.BOS + render3.SYSTEM_ROLE + "\n" + render3.SYSTEM + render3.EOS
        + render3.BOS + render3.USER + "hi" + render3.EOS
        + render3.BOS + render3.BOT + render3.THINK
    )


def test_render_puts_the_media_marker_before_a_users_picture() -> None:
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        {"type": "text", "text": "what is this"},
    ]}]
    prompt = render3.render(messages)
    assert render3.MEDIA + "\n" + "what is this" in prompt
    assert prompt.index(render3.MEDIA) < prompt.index("what is this")


def test_render_writes_a_known_route_before_the_reply() -> None:
    messages = [
        {"role": "user", "content": "list the files"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "shell", "arguments": {"cmd": "ls"}}},
        ]},
        {"role": "tool", "content": "a.txt"},
    ]
    prompt = render3.render(messages, routes=["code light act"])
    assert render3.THINK + "code light act" + render3.THINK_END + "<tool_call>" in prompt


def test_render_falls_back_to_the_chat_route() -> None:
    acting = [
        {"role": "user", "content": "list the files"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "shell", "arguments": {"cmd": "ls"}}},
        ]},
    ]
    assert (render3.THINK + "chat light act" + render3.THINK_END + "<tool_call>"
            in render3.render(acting, routes=None))
    saying = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hello."},
    ]
    assert (render3.THINK + "chat light say" + render3.THINK_END + "Hello."
            in render3.render(saying, routes=None))


def test_render_defuses_structure_in_a_user_message() -> None:
    prompt = render3.render([{"role": "user", "content": "<|User|> and </think>"}])
    assert prompt.count(render3.USER) == 1  # only the real role token
    assert render3.USER[0] + ZWSP + render3.USER[1:] in prompt  # "<​|User|>"
    assert "</think>" not in prompt
    assert "<" + ZWSP + "/think>" in prompt


def test_render_plain_ends_with_the_chat_route() -> None:
    prompt = render3.render_plain([{"role": "user", "content": "summarise this"}])
    assert prompt.endswith(render3.BOS + render3.BOT + render3.THINK + render3.PLAIN_ROUTE + render3.THINK_END)
    assert render3.PLAIN_ROUTE == "chat light say"


# ---------------------------------------------------------------------------------------------- llama: args


def _write_manifest(root: Path, listed: dict[str, bytes], **fields: Any) -> dict[str, Any]:
    """Write ``root`` as a pack folder: every listed file with its size and sha256, then the manifest."""
    files: dict[str, dict[str, Any]] = {}
    for name, data in listed.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {"name": "kiwimind", "version": "test-3", "files": files, **fields}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _gguf_pack(tmp_path: Path) -> tuple[Pack, dict[str, Any]]:
    manifest = _write_manifest(
        tmp_path / "mind",
        {"kiwimind-trunk.gguf": b"trunk", "route.gguf": b"route",
         "skills/chat.gguf": b"chat", "skills/code.gguf": b"code"},
        engine="llama", skills=["chat", "code"], route={"file": "route.gguf"}, ctx=4096,
    )
    return Pack(tmp_path), manifest


def _runtime(flags: frozenset[str], offload: bool | None) -> llama.LlamaRuntime:
    return llama.LlamaRuntime(path="/opt/llama.cpp/llama-server", flags=flags, offload=offload, env={})


def test_layers_route_first_then_skills() -> None:
    assert llama.layers({"route": {"file": "route3.gguf"}, "skills": ["chat", "code"]}) == [
        ("route", "route3.gguf"),
        ("chat", "skills/chat.gguf"),
        ("code", "skills/code.gguf"),
    ]
    assert llama.layers({"skills": ["chat"]}) == [("route", "route.gguf"), ("chat", "skills/chat.gguf")]


def test_server_args_core(tmp_path: Path) -> None:
    pack, manifest = _gguf_pack(tmp_path)
    runtime = _runtime(frozenset(), None)
    args = llama.server_args(runtime, pack, manifest, port=8090, threads=6)
    assert args[0] == "/opt/llama.cpp/llama-server"
    assert args[args.index("--model") + 1] == str(pack.path("kiwimind-trunk.gguf"))
    assert "--special" in args
    assert args[args.index("--parallel") + 1] == "3"
    assert "--lora-init-without-apply" in args
    assert [args[i + 1] for i, arg in enumerate(args) if arg == "--lora"] == [
        str(pack.path("route.gguf")),
        str(pack.path("skills/chat.gguf")),
        str(pack.path("skills/code.gguf")),
    ]
    assert args.count("--lora") == 3


def test_server_args_offload(tmp_path: Path) -> None:
    pack, manifest = _gguf_pack(tmp_path)
    on_gpu = llama.server_args(_runtime(frozenset(), True), pack, manifest, port=8090, threads=6)
    assert on_gpu[on_gpu.index("--gpu-layers") + 1] == "auto"  # flags unknown: a current build has --fit
    on_cpu = llama.server_args(_runtime(frozenset(), False), pack, manifest, port=8090, threads=6)
    assert "--gpu-layers" not in on_cpu
    assert on_cpu[on_cpu.index("--device") + 1] == "none"
    unknown = llama.server_args(_runtime(frozenset(), None), pack, manifest, port=8090, threads=6)
    assert "--gpu-layers" not in unknown and "--device" not in unknown


def test_server_args_vision_only_with_a_file(tmp_path: Path) -> None:
    pack, manifest = _gguf_pack(tmp_path)
    plain = llama.server_args(_runtime(frozenset(), None), pack, manifest, port=8090, threads=6)
    assert "--mmproj" not in plain
    with_vision = {**manifest, "vision": {"file": "vision/mmproj.gguf"}}
    args = llama.server_args(_runtime(frozenset(), None), pack, with_vision, port=8090, threads=6)
    assert args[args.index("--mmproj") + 1] == str(pack.path("vision/mmproj.gguf"))
    empty = {**manifest, "vision": {"file": ""}}
    assert "--mmproj" not in llama.server_args(_runtime(frozenset(), None), pack, empty, port=8090, threads=6)


def test_server_args_leave_out_a_flag_the_build_lacks(tmp_path: Path) -> None:
    pack, manifest = _gguf_pack(tmp_path)
    flags = frozenset({"--threads-batch", "--device", "--no-webui", "--mmproj"})
    args = llama.server_args(_runtime(flags, None), pack, manifest, port=8090, threads=6)
    assert "--kv-unified" not in args
    assert "--no-cache-idle-slots" not in args
    assert "--no-webui" in args  # the flags the build does list are still passed
    assert "--threads-batch" in args


# ---------------------------------------------------------------------------------------------- llama: answers


def test_share_reads_each_candidate_format() -> None:
    logs = [{"token": " heavy", "top_logprobs": [
        {"token": " heavy", "logprob": -0.2}, {"token": " light", "logprob": -1.6},
    ]}]
    assert llama.share(logs, "heavy", "light") == pytest.approx(
        math.exp(-0.2) / (math.exp(-0.2) + math.exp(-1.6)))
    probs = [{"token": "x", "top_probs": []}, {"content": " light", "top_probs": [
        {"token": "light", "prob": 0.4}, {"token": "heavy", "prob": 0.6},
    ]}]
    assert llama.share(probs, "heavy", "light") == pytest.approx(0.6)
    strings = [{"token": " heavy", "probs": [
        {"tok_str": "li", "prob": 0.4}, {"tok_str": "light", "prob": 0.25}, {"tok_str": "heavy", "prob": 0.5},
    ]}]
    assert llama.share(strings, "heavy", "light") == pytest.approx(2 / 3)


def test_share_is_none_when_neither_word_was_written() -> None:
    assert llama.share(None, "heavy", "light") is None
    assert llama.share([{"token": "code", "top_probs": [{"token": "heavy", "prob": 0.5}]}],
                       "heavy", "light") is None
    assert llama.share([{"token": "light", "top_probs": []}], "heavy", "light") is None


def test_done_reads_llama_servers_last_event() -> None:
    event = {"stop_type": "limit", "timings": {
        "cache_n": 4, "prompt_n": 6, "predicted_n": 7,
        "prompt_ms": 12.5, "predicted_ms": 100.0, "predicted_per_second": 70.0,
    }}
    assert llama._done(event) == {
        "done": True, "finish": "length",
        "usage": {"prompt_tokens": 10, "cached_tokens": 4, "completion_tokens": 7},
        "timings": {"upper_ms": 12.5, "decode_ms": 100.0, "predicted_per_second": 70.0},
    }
    assert llama._done({**event, "stop_type": "eos"})["finish"] == "stop"


def test_refused_marks_a_context_error() -> None:
    error = llama._refused({"type": "exceed_context_size_error", "message": "too many tokens"}, 400)
    assert isinstance(error, MindRequestError)
    assert error.context is True
    other = llama._refused({"type": "invalid_request_error", "message": "bad arguments"}, 400)
    assert other.context is False


# ---------------------------------------------------------------------------------------------- llama: client


def test_client_slot_policy() -> None:
    client = llama.LlamaClient(SimpleNamespace(adapters={}))
    assert client._slot(llama.UTIL_SLOT) == 2  # one slot past the two conversation slots
    assert client._slot("a") == 0
    assert client._slot("b") == 1
    assert client._slot("a") == 0  # a repeated conversation keeps its slot
    assert client._slot("c") == 1  # b was least recently used and gives way
    assert client._slot("a") == 0
    assert client._slot("b") == 1


def test_client_switches_only_the_named_layer_on() -> None:
    client = llama.LlamaClient(SimpleNamespace(adapters={"route": 0, "chat": 1, "code": 2}))
    layers = client._on("code")
    assert layers == [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}, {"id": 2, "scale": 1.0}]
    assert [layer["id"] for layer in layers if layer["scale"] == 1.0] == [2]
    assert client._on("painting") == [
        {"id": 0, "scale": 0.0}, {"id": 1, "scale": 1.0}, {"id": 2, "scale": 0.0}
    ]  # an unknown layer falls back to chat


# ---------------------------------------------------------------------------------------------- conductor3


def test_call_grammar_without_tools() -> None:
    assert conductor3.call_grammar(None) == ""
    assert conductor3.call_grammar([]) == ""
    assert conductor3.call_grammar([{"type": "function", "function": {"name": ""}}]) == ""


def test_call_grammar_offers_every_tool() -> None:
    tools = [
        {"type": "function", "function": {"name": "read", "parameters": {}}},
        {"type": "function", "function": {"name": "write", "parameters": {}}},
    ]
    grammar = conductor3.call_grammar(tools)
    assert grammar.splitlines()[0] == r'root ::= call ("\n<tool_call>\n" call)*'
    assert r'name ::= "\"read\"" | "\"write\""' in grammar


def test_with_pictures_keeps_user_pictures_in_order() -> None:
    picture = {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
    other = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,REVG"}}
    text = {"type": "text", "text": "one"}
    messages = [
        {"role": "user", "content": [picture, text, other]},
        {"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,SHQ"}}]},
    ]
    out, pictures = conductor3.with_pictures(messages, sees=True)
    assert pictures == ["QUJD", "REVG"]  # an assistant turn's picture is not given to the model
    assert out[0]["content"] == [picture, text, other]
    assert out[1]["content"] == []


def test_with_pictures_drops_them_without_eyes() -> None:
    picture = {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
    text = {"type": "text", "text": "one"}
    messages = [{"role": "user", "content": [picture, text]}]
    out, pictures = conductor3.with_pictures(messages, sees=False)
    assert pictures == []
    assert out[0]["content"] == [text]
    assert messages[0]["content"] == [picture, text]  # the caller's messages are not changed


def test_with_pictures_leaves_string_content_alone() -> None:
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    out, pictures = conductor3.with_pictures(messages, sees=True)
    assert out == messages
    assert pictures == []


# ---------------------------------------------------------------------------------------------- pack


def test_engine_reads_the_manifest(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "three" / "mind", {"kiwimind-trunk.gguf": b"trunk"}, engine="llama")
    assert Pack(tmp_path / "three").engine() == "llama"
    _write_manifest(tmp_path / "two" / "mind", {"kiwimind-trunk.fkm": b"trunk"})
    assert Pack(tmp_path / "two").engine() == "wasm"


def test_sweep_removes_the_files_of_an_earlier_pack(tmp_path: Path) -> None:
    root = tmp_path / "mind"
    _write_manifest(
        root,
        {"kiwimind-trunk.gguf": b"trunk", "route.gguf": b"route", "skills/chat.gguf": b"chat"},
        engine="llama", skills=["chat"], route={"file": "route.gguf"},
    )
    for name, data in {
        "kiwimind-trunk.fkm": b"old", "skills/old.kms": b"old", "tokenizer.json": b"{}",
        "host.log": b"log", CHECK_CACHE: b"{}",
    }.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    pack = Pack(tmp_path)
    assert pack.sweep() == 3
    assert not (root / "kiwimind-trunk.fkm").exists()
    assert not (root / "skills/old.kms").exists()
    assert not (root / "tokenizer.json").exists()
    assert (root / "kiwimind-trunk.gguf").exists()
    assert (root / "route.gguf").exists()
    assert (root / "skills/chat.gguf").exists()
    assert (root / "manifest.json").exists()
    assert (root / "host.log").exists()
    assert (root / CHECK_CACHE).exists()


def test_sweep_does_nothing_for_an_incomplete_pack(tmp_path: Path) -> None:
    root = tmp_path / "mind"
    _write_manifest(
        root,
        {"kiwimind-trunk.gguf": b"trunk", "route.gguf": b"route"},
        engine="llama", route={"file": "route.gguf"},
    )
    (root / "route.gguf").unlink()
    (root / "kiwimind-trunk.fkm").write_bytes(b"old")
    assert Pack(tmp_path).sweep() == 0
    assert (root / "kiwimind-trunk.fkm").exists()


def test_a_picture_prompt_uses_the_marker_the_running_server_chose():
    client = llama.LlamaClient(SimpleNamespace(adapters={}, marker="<__media_x1__>"))
    body = client._base(f"a{render3.MEDIA}\nb", "t", ["QUJD"])
    assert body["prompt"] == {"prompt_string": "a<__media_x1__>\nb", "multimodal_data": ["QUJD"]}
    assert client._base("words only", "t", [])["prompt"] == "words only"
