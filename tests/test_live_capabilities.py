"""Opt-in LIVE checks against a running LM Studio (skipped unless KIWIBRO_LIVE=1).

Exercises the real capability-delegation path end to end: catalog discovery,
smallest-capable helper selection, JIT load, vision inference, lease release.
Expect the vision test to take minutes when the VLM must load cold.
"""

from __future__ import annotations

import base64
import os
import struct
import time
import zlib

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("KIWIBRO_LIVE") != "1", reason="live LM Studio checks are opt-in (KIWIBRO_LIVE=1)"
)


def _png_data_uri() -> str:
    """96x96 white PNG with a centered red square (stdlib only)."""
    width = height = 96
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            red = 24 <= x < 72 and 24 <= y < 72
            row += bytes((255, 0, 0)) if red else bytes((255, 255, 255))
        rows.append(bytes(row))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


@pytest.fixture(scope="module")
def client():
    from kiwibro.llm import LMStudioClient

    url = os.environ.get("KIWIBRO_LMSTUDIO_URL", "http://127.0.0.1:1234")
    return LMStudioClient(lambda: {"lmstudio_url": url, "model": None})


def test_live_catalog_has_capabilities(client):
    from kiwibro import capabilities as caps

    entries = client.catalog()
    assert entries
    assert any(e["chat"] for e in entries)
    assert all(e["id"] for e in entries)
    summary = caps.catalog_summary(entries)
    assert "loaded" in summary or "idle" in summary


def test_live_embed(client):
    entries = client.catalog()
    model = next((e["id"] for e in entries if e["embeddings"]), None)
    if model is None:
        pytest.skip("no embeddings model installed")
    vectors = client.embed(model=model, texts=["kiwi", "agent"])
    assert len(vectors) == 2 and len(vectors[0]) > 100


def test_live_vision_delegation(client):
    """Main model lacks vision -> harness runs the smallest VLM temporarily."""
    import kiwibro.routing as routing

    entries = client.catalog()
    if not any(e["vision"] for e in entries):
        pytest.skip("no vision model installed")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What color is the square in this image? Answer with one word."},
            {"type": "image_url", "image_url": {"url": _png_data_uri()}},
        ],
    }]
    started = time.perf_counter()
    result = routing.run_helper(client, "vision", messages, current_model=None, max_tokens=200)
    elapsed = time.perf_counter() - started
    print(f"\nvision via {result['model']} (temporary={result.get('temporary')}) "
          f"in {elapsed:.0f}s: {result['text']!r}")
    assert result["text"]
    assert "red" in result["text"].lower()
    assert routing.temporary_models() == []  # lease released after the call
