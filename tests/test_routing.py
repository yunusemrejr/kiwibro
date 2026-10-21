"""Capability routing tests: helper picking + bounded helper lifecycle.

Offline: pure ranking over model dicts and a fake LLM client that records
load/complete/unload calls. The contracts pinned here are the seam other
lanes code against (`routing.run_helper/pick_helper/capabilities_of`).
"""

from __future__ import annotations

from typing import Any

import pytest

from kiwibro import routing
from kiwibro.llm import Completion, LMStudioError

MODELS: list[dict[str, Any]] = [
    {"id": "tiny-chat-0.8b", "type": "llm", "state": "loaded", "context": 8192,
     "capabilities": ["tool_use"], "quantization": "Q4_K_M"},
    {"id": "big-chat-27b", "type": "llm", "state": "not-loaded", "context": 32768,
     "capabilities": ["tool_use"], "quantization": "Q4_K_M"},
    {"id": "qwen-vl-2b", "type": "vlm", "state": "not-loaded", "context": 16384,
     "capabilities": [], "quantization": "Q4_K_M"},
    {"id": "qwen-vl-7b", "type": "vlm", "state": "not-loaded", "context": 32768,
     "capabilities": [], "quantization": "Q4_K_M"},
    {"id": "deepseek-r1-1.5b", "type": "llm", "state": "not-loaded", "context": 65536,
     "capabilities": [], "quantization": "Q4_K_M"},
    {"id": "text-embedding-nomic", "type": "embeddings", "state": "not-loaded",
     "context": 2048, "capabilities": [], "quantization": ""},
]


def test_capabilities_of_detects_vision_and_soft_needs():
    vlm = next(m for m in MODELS if m["id"] == "qwen-vl-2b")
    assert "vision" in routing.capabilities_of(vlm)
    reasoner = next(m for m in MODELS if m["id"] == "deepseek-r1-1.5b")
    caps = routing.capabilities_of(reasoner)
    assert "reasoning" in caps and "long_context" in caps
    tiny = next(m for m in MODELS if m["id"] == "tiny-chat-0.8b")
    assert "vision" not in routing.capabilities_of(tiny)
    assert "tools" in routing.capabilities_of(tiny)


def test_pick_helper_smallest_vision_and_never_current():
    helper = routing.pick_helper(MODELS, "vision", current="tiny-chat-0.8b")
    assert helper is not None and helper["id"] == "qwen-vl-2b"  # smallest adequate
    assert routing.pick_helper(MODELS, "audio") is None
    with pytest.raises(ValueError):
        routing.pick_helper(MODELS, "telepathy")


def test_pick_helper_loaded_first_and_long_context():
    helper = routing.pick_helper(MODELS, "fast", current="big-chat-27b")
    assert helper is not None and helper["id"] == "tiny-chat-0.8b"  # loaded + tiny
    long_ctx = routing.pick_helper(MODELS, "long_context", current="tiny-chat-0.8b")
    assert long_ctx is not None and long_ctx["id"] == "deepseek-r1-1.5b"


class FakeLLM:
    def __init__(self, models: list[dict[str, Any]]) -> None:
        self.models = models
        self.calls: list[tuple[str, str]] = []

    def catalog(self, force: bool = False):
        return [dict(m) for m in self.models]

    def list_models(self, force: bool = False):
        return [dict(m) for m in self.models if m["type"] in ("llm", "vlm")]

    def load_model(self, model: str, context_length: int | None = None) -> bool:
        self.calls.append(("load", model))
        return True

    def unload_model(self, model: str) -> bool:
        self.calls.append(("unload", model))
        return True

    def complete(self, *, model, messages, max_tokens=600, temperature=0.0, retries=0, **kw):
        self.calls.append(("complete", model))
        return Completion(f"answer from {model}")

    def complete_vision(self, *, model, prompt, image_data_uris, max_tokens=900, temperature=0.0, **kw):
        self.calls.append(("vision", model))
        return Completion("a kiwi bird next to a harbor gate")

    def embed(self, *, model, texts, **kw):
        self.calls.append(("embed", model))
        return [[0.1, 0.2] for _ in texts]


def test_run_helper_vision_lifecycle_loads_and_unloads():
    llm = FakeLLM(MODELS)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }]
    result = routing.run_helper(llm, "vision", messages, current_model="tiny-chat-0.8b")
    assert result["model"] == "qwen-vl-2b"
    assert "kiwi" in result["text"]
    assert llm.calls == [("load", "qwen-vl-2b"), ("vision", "qwen-vl-2b"), ("unload", "qwen-vl-2b")]


def test_run_helper_never_unloads_loaded_or_main_model():
    llm = FakeLLM(MODELS)
    result = routing.run_helper(
        llm, "fast", [{"role": "user", "content": "hi"}], current_model="big-chat-27b"
    )
    assert result["model"] == "tiny-chat-0.8b"  # already loaded
    assert ("unload", "tiny-chat-0.8b") not in llm.calls
    assert ("load", "tiny-chat-0.8b") not in llm.calls


def test_run_helper_embeddings_and_missing_capability():
    llm = FakeLLM(MODELS)
    result = routing.run_helper(
        llm, "embeddings", [{"role": "user", "content": "embed me"}], current_model="tiny-chat-0.8b"
    )
    assert result["embeddings"] == [[0.1, 0.2]]
    with pytest.raises(LMStudioError) as exc:
        routing.run_helper(llm, "audio", [{"role": "user", "content": "listen"}])
    assert exc.value.kind == "no_model"


def test_run_helper_unloads_even_when_call_fails():
    class FailingLLM(FakeLLM):
        def complete_vision(self, **kw):  # noqa: ANN003 - test double
            self.calls.append(("vision", kw["model"]))
            raise LMStudioError("server", "boom")

    llm = FailingLLM(MODELS)
    messages = [{
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}],
    }]
    with pytest.raises(LMStudioError):
        routing.run_helper(llm, "vision", messages, current_model="tiny-chat-0.8b")
    assert llm.calls[-1] == ("unload", "qwen-vl-2b")
