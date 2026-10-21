"""Capability catalog + LM Studio serving primitives (session 01a16089-71d5).

Model payloads mirror a live LM Studio instance captured 2026-10-22:
``capabilities`` only ever lists ``tool_use``; vision is ``type == "vlm"``;
the load/unload endpoints do not exist on that build.
"""

from __future__ import annotations

import pytest

import kiwibro.llm as llm_module
from kiwibro.capabilities import (
    catalog,
    catalog_summary,
    est_memory_gb,
    infer_capabilities,
    parse_params_b,
    quant_bits,
    size_key,
)
from kiwibro.llm import LMStudioClient, LMStudioError

RAW_MODELS = [
    {"id": "spark-x2.6-4b", "type": "llm", "state": "loaded", "capabilities": ["tool_use"],
     "quantization": "IQ2_M", "max_context_length": 1048576, "loaded_context_length": 32768},
    {"id": "liquid/lfm2.5-1.2b", "type": "llm", "state": "not-loaded", "capabilities": ["tool_use"],
     "quantization": "Q4_K_M", "max_context_length": 128000},
    {"id": "qwen3.8-27b", "type": "vlm", "state": "not-loaded", "capabilities": ["tool_use"],
     "quantization": "IQ2_XXS", "max_context_length": 262144},
    {"id": "text-embedding-nomic-embed-text-v1.5", "type": "embeddings", "state": "not-loaded",
     "capabilities": None, "quantization": "Q4_K_M", "max_context_length": 2048},
    {"id": "minicpm5-2b", "type": "llm", "state": "not-loaded", "capabilities": None,
     "quantization": "Q4_K_M", "max_context_length": 131072},
]


class FakeResponse:
    def __init__(self, payload, status=200, text=""):
        self._payload = payload
        self.status_code = status
        self.text = text

    def json(self):
        return self._payload


def make_client() -> LMStudioClient:
    return LMStudioClient(lambda: {"lmstudio_url": "http://127.0.0.1:1234", "model": None})


def test_parse_params_and_quant():
    assert parse_params_b("qwen3.8-2b") == 2.0
    assert parse_params_b("liquid/lfm2.5-1.2b") == 1.2
    assert parse_params_b("qwen3.5-0.8b-distilled") == 0.8
    assert parse_params_b("qwen3.8_4b_distilled_gguf") == 4.0
    assert parse_params_b("text-embedding-nomic") is None
    assert quant_bits("Q4_K_M") == pytest.approx(4.5)
    assert quant_bits("IQ2_XXS") == pytest.approx(2.3)
    assert quant_bits("") is None


def test_infer_capabilities_server_facts_first():
    by_id = {m["id"]: m for m in catalog(RAW_MODELS)}
    assert by_id["qwen3.8-27b"]["vision"] is True
    assert by_id["qwen3.8-27b"]["vision_source"] == "server"
    assert by_id["spark-x2.6-4b"]["vision"] is False
    assert by_id["text-embedding-nomic-embed-text-v1.5"]["embeddings"] is True
    assert by_id["text-embedding-nomic-embed-text-v1.5"]["chat"] is False
    assert by_id["minicpm5-2b"]["tool_use"] is False  # capabilities: None on the wire
    assert by_id["liquid/lfm2.5-1.2b"]["params_b"] == pytest.approx(1.2)
    assert by_id["liquid/lfm2.5-1.2b"]["est_memory_gb"] < by_id["spark-x2.6-4b"]["est_memory_gb"]


def test_infer_name_heuristic_is_tagged():
    guessed = infer_capabilities({"id": "acme/llava-1.5-7b", "type": "llm"})
    assert guessed["vision"] is True and guessed["vision_source"] == "name-heuristic"


def test_size_key_sorts_smallest_first_unknown_last():
    ordered = sorted(catalog(RAW_MODELS), key=size_key)
    assert ordered[0]["id"] == "liquid/lfm2.5-1.2b"
    assert ordered[-1]["id"] == "text-embedding-nomic-embed-text-v1.5"  # unknown size last


def test_est_memory_scales_with_quant():
    q2 = est_memory_gb({"id": "m-4b", "quantization": "Q2_K"})
    q8 = est_memory_gb({"id": "m-4b", "quantization": "Q8_0"})
    assert q2 < q8


def test_catalog_includes_embeddings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        llm_module.requests, "get", lambda *a, **k: FakeResponse({"data": RAW_MODELS})
    )
    entries = make_client().catalog()
    assert len(entries) == 5
    embeddings = [e for e in entries if e["embeddings"]]
    assert embeddings and embeddings[0]["id"].startswith("text-embedding")
    assert any(e["vision"] for e in entries)


def test_catalog_summary_is_compact():
    summary = catalog_summary(RAW_MODELS)
    lines = summary.splitlines()
    assert len(lines) == 5
    assert "vision" in lines[2] and "qwen3.8-27b" in lines[2]
    assert len(summary) < 600


def test_complete_vision_builds_image_parts(monkeypatch: pytest.MonkeyPatch):
    seen: dict = {}

    def fake_post(url, json=None, timeout=None, headers=None, **kw):
        seen.update(json or {})
        return FakeResponse({"choices": [{"message": {"content": "a kiwi"}}], "usage": {}})

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    result = make_client().complete_vision(
        model="qwen3.8-27b", prompt="What is this?",
        image_data_uris=["data:image/png;base64,iVBORw0KGgo="],
    )
    assert result.text == "a kiwi"
    parts = seen["messages"][-1]["content"]
    assert parts[0] == {"type": "text", "text": "What is this?"}
    assert parts[1]["type"] == "image_url"


def test_complete_vision_rejects_missing_images():
    with pytest.raises(LMStudioError):
        make_client().complete_vision(model="m", prompt="?", image_data_uris=["http://x/y.png"])


def test_embed_orders_by_index(monkeypatch: pytest.MonkeyPatch):
    payload = {"data": [{"index": 1, "embedding": [0.3]}, {"index": 0, "embedding": [0.1, 0.2]}]}
    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: FakeResponse(payload))
    assert make_client().embed(model="e", texts=["a", "b"]) == [[0.1, 0.2], [0.3]]
    assert make_client().embed(model="e", texts=[]) == []


def test_load_unload_feature_detects_missing_endpoint(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def fake_post(url, json=None, timeout=None, headers=None, **kw):
        calls.append(url)
        return FakeResponse({}, text='{"error":"Unexpected endpoint or method. (POST /api/v0/models/load)"}')

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    client = make_client()
    assert client.load_model("x") is False
    assert client.unload_model("x") is False
    assert len(calls) == 1  # support flag cached: no second doomed request


def test_complete_passes_response_format(monkeypatch: pytest.MonkeyPatch):
    seen: dict = {}

    def fake_post(url, json=None, timeout=None, headers=None, **kw):
        seen.update(json or {})
        return FakeResponse({"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    make_client().complete(
        model="m", messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )
    assert seen["response_format"] == {"type": "json_object"}
