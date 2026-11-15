"""The trainer of the control head's "heavy" output (scripts/mind/heavy.py): what it asks the host, and what it
writes into a head and into a pack. Made-up states stand in for the engine's."""

from __future__ import annotations

import hashlib
import http.server
import importlib.util
import json
import threading

import numpy as np
import pytest
from mind_pack import ROOT, load_formats

NAMES = ["chat", "web", "hard"]
WIDTH = 24


def load_heavy():
    spec = importlib.util.spec_from_file_location("kiwimind_heavy", ROOT / "scripts" / "mind" / "heavy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _logits(head: dict, x: np.ndarray) -> np.ndarray:
    normed = x / np.sqrt((x * x).mean(axis=1, keepdims=True) + head["eps"])
    return np.maximum(0, normed @ head["w1"].T + head["b1"]) @ head["w2"].T + head["b2"]


@pytest.fixture
def trained(tmp_path):
    """A random head, and states whose first two numbers say whether the request is heavy."""
    formats, rng = load_formats(), np.random.default_rng(3)
    old = tmp_path / "old.kmh"
    formats.write_head(old, rng.normal(size=(6, WIDTH)).astype(np.float32), rng.normal(size=6).astype(np.float32),
                       rng.normal(size=(3, 6)).astype(np.float32), rng.normal(size=3).astype(np.float32), NAMES, layer=2)
    x = rng.normal(size=(600, WIDTH)).astype(np.float32)
    heavy = x[:, 0] + x[:, 1] > 0.8
    states = tmp_path / "states.npz"
    np.savez_compressed(
        states, x=x, heavy=heavy, second=np.arange(600) % 3 == 0, ask=np.array([f"made-up request {i}" for i in range(600)]),
        source=np.array(["made-up"] * 600), logits=_logits(formats.read_head(old), x).astype(np.float32),
        names=np.array(NAMES))
    return formats, old, states, x, heavy


def test_the_heavy_output_is_learned_and_the_other_outputs_keep_their_values(trained, tmp_path):
    formats, old, states, x, heavy = trained
    new, report = tmp_path / "new.kmh", tmp_path / "report.json"
    assert load_heavy().main(["train", "--states", str(states), "--head", str(old), "--hidden", "8",
                              "--out", str(new), "--report", str(report)]) == 0
    before, after = formats.read_head(old), formats.read_head(new)
    assert after["w1"].shape == (6 + 8, WIDTH) and after["names"] == NAMES and after["layer"] == 2
    was, now = _logits(before, x), _logits(after, x)
    assert np.allclose(now[:, :2], was[:, :2], atol=1e-5)
    said = now[:, 2] > 0
    assert (said & heavy).sum() / said.sum() > 0.8
    assert (said & heavy).sum() / heavy.sum() > 0.5
    figures = json.loads(report.read_text())
    assert figures["test"]["n"] == figures["test_first_message"]["n"] + figures["test_later_message"]["n"] > 0


def test_a_pack_gets_the_new_head_and_a_manifest_that_matches_it(trained, tmp_path):
    _formats, old, states, _x, _heavy = trained
    pack = tmp_path / "mind"
    pack.mkdir()
    (pack / "head.kmh").write_bytes(old.read_bytes())
    (pack / "manifest.json").write_text(json.dumps({
        "version": "9.0.0", "files": {"head.kmh": {"bytes": 1, "sha256": "old"}, "other.bin": {"bytes": 5, "sha256": "kept"}},
        "head_report": {"skill_accuracy": 0.9, "hard_precision": 1.0, "hard_recall": 0.5, "hard_share": 0.003}}))
    assert load_heavy().main(["train", "--states", str(states), "--head", str(old), "--hidden", "8",
                              "--pack", str(pack), "--version", "9.0.1"]) == 0
    manifest, written = json.loads((pack / "manifest.json").read_text()), (pack / "head.kmh").read_bytes()
    assert written != old.read_bytes()
    assert manifest["version"] == "9.0.1"
    assert manifest["files"] == {"head.kmh": {"bytes": len(written), "sha256": hashlib.sha256(written).hexdigest()},
                                 "other.bin": {"bytes": 5, "sha256": "kept"}}
    assert manifest["head_report"]["skill_accuracy"] == 0.9
    assert manifest["head_report"]["hard_recall"] == manifest["head_report"]["hard_unit"]["test"]["recall"]


def test_a_folder_without_a_manifest_is_not_taken_for_a_pack(trained, tmp_path):
    _formats, old, states, _x, _heavy = trained
    with pytest.raises(SystemExit, match="manifest"):
        load_heavy().main(["train", "--states", str(states), "--head", str(old), "--pack", str(tmp_path)])
    assert not (tmp_path / "head.kmh").exists()


def test_reading_one_side_asks_with_the_prompts_of_a_full_run(tmp_path):
    heavy, asked = load_heavy(), []

    class Host(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/control" and body["state"] is True and self.headers["Authorization"] == "Bearer made-up-key"
            asked.append(body["prompt"])
            reply = json.dumps({"names": NAMES, "logits": [0.0, 0.0, 0.0], "state": [0.5] * WIDTH}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args) -> None:
            pass

    labels, replies, prefix = tmp_path / "labels.jsonl", tmp_path / "replies.jsonl", tmp_path / "prefix.txt"
    labels.write_text("".join(json.dumps({"ask": f"made-up request {i}", "heavy": i % 4 == 0, "source": "made-up"}) + "\n"
                              for i in range(60)))
    replies.write_text(json.dumps({"ask": "hello", "reply": "Hello."}) + "\n")
    prefix.write_text("<|im_start|>system\nA made-up system turn.<|im_end|>\n")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Host)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        common = ["states", "--host", f"http://127.0.0.1:{server.server_address[1]}", "--key", "made-up-key",
                  "--prefix", str(prefix), "--labels", str(labels), "--replies", str(replies)]
        assert heavy.main([*common, "--out", str(tmp_path / "all.npz")]) == 0
        everything = list(asked)
        asked.clear()
        assert heavy.main([*common, "--only", "test", "--out", str(tmp_path / "test.npz")]) == 0
    finally:
        server.shutdown()
        server.server_close()
    full, side = np.load(tmp_path / "all.npz"), set(np.load(tmp_path / "test.npz")["ask"].tolist())
    assert asked == [prompt for prompt, ask in zip(everything, full["ask"].tolist()) if ask in side]
    assert 0 < len(asked) < len(everything) == 60 + int(60 * 0.35)
    assert all(prompt.startswith("<|im_start|>system\nA made-up system turn.") and prompt.endswith("<|im_start|>assistant\n")
               for prompt in everything)
