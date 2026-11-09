"""Kiwi Sense: the reading model beside the language model."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from kiwibro import sense
from kiwibro.sense import blob as blobfile
from kiwibro.sense.pyengine import PyEngine

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "static" / "sense"

TEXTS = [
    "What is the capital of France?",
    "Paris is the capital of France.",
    "I'm sorry, but as an AI language model I can't browse the internet.",
    "I am ChatGPT, a large language model trained by OpenAI.",
    "I've saved the file to your desktop.",
    "Let me search the web for that.",
    "The user is asking about the weather.",
    "Türkiye'nin başkenti Ankara'dır, İstanbul değil.",
    "日本語のテキストと emoji 🙂 mixed in",
    "naïve café déjà-vu — “quoted”… end",
    "the cat sat on the mat . " * 30,
    "x" * 150,
    "word " * 400,
    "",
    "   ",
    "?",
    "def reverse(items):\n    return items[::-1]  # O(n)",
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
]


@pytest.fixture(scope="module")
def reader() -> sense.Sense:
    return sense.Sense(engine="python")


# ---- text shaping ----

def test_normalize_lowercases_strips_accents_and_splits_punctuation():
    assert sense.normalize("Héllo,  WORLD!\n(ok)") == "hello , world ! ( ok )"
    assert sense.normalize("İstanbul'da") == "istanbul ' da"
    assert sense.normalize("\x00a\tb​c") == "a bc"
    assert sense.normalize("") == ""


def test_normalize_puts_cjk_characters_apart():
    assert sense.normalize("日本") == "日 本"


def test_prose_drops_code_and_markup():
    text = "## Title\nUse `rm -rf` **carefully**.\n```sh\nI am ChatGPT\n```\nSee [docs](http://x.y)."
    cleaned = sense.prose(text)
    assert "ChatGPT" not in cleaned and "rm -rf" not in cleaned
    assert "carefully" in cleaned and "docs" in cleaned and "http" not in cleaned


def test_sentences_split_on_ends_and_lines():
    assert sense.sentences("One. Two!\nThree? Four") == ["One.", "Two!", "Three?", "Four"]
    assert len(sense.sentences("A. " * 40, limit=5)) == 5


# ---- data file ----

def test_blob_round_trip_keeps_sections_aligned():
    packed = blobfile.pack({"AAAA": b"abc", "BBBB": b"\x01" * 9})
    sections = blobfile.unpack(packed)
    assert bytes(sections["AAAA"]) == b"abc" and bytes(sections["BBBB"]) == b"\x01" * 9
    offsets = [int.from_bytes(packed[12 + 12 * i + 4:12 + 12 * i + 8], "little") for i in range(2)]
    assert all(offset % 8 == 0 for offset in offsets)


def test_blob_rejects_other_files():
    with pytest.raises(ValueError):
        blobfile.unpack(b"not a sense file at all")
    with pytest.raises(ValueError):
        blobfile.unpack(b"KSNS" + (9).to_bytes(4, "little") + (0).to_bytes(4, "little"))


def test_vocab_section_round_trip():
    tokens = ["[PAD]", "the", "##ing", "é", "a"]
    section = memoryview(blobfile.vocab_section(tokens))
    assert blobfile.vocab_tokens(section) == [token.encode() for token in tokens]
    with pytest.raises(ValueError):
        blobfile.vocab_section(["a", "a"])


def test_shipped_data_file_describes_itself(reader):
    assert reader.labels[0] == "fine" and "denial" in reader.labels and "other_identity" in reader.labels
    probe = reader.meta["probe"]
    assert 0.5 <= probe["threshold"] < 1
    assert reader.meta["embeddings"]["licence"] == "MIT"
    assert (DATA / "sense.wasm").stat().st_size < 64_000  # the core stays tiny


# ---- meaning ----

def test_related_texts_score_above_unrelated(reader):
    question = "my dog is sick"
    assert reader.similarity(question, "You should take your puppy to the vet.") > 0.3
    assert reader.similarity(question, "The stock market fell sharply on Tuesday.") < 0.25
    assert reader.similarity("What is the capital of France?", "Paris is the capital of France.") > 0.7


def test_similarity_is_none_when_nothing_can_be_read(reader):
    assert reader.similarity("", "hello") is None
    assert reader.embed("") is None
    assert reader.coverage("") == 0.0


def test_rank_puts_the_closest_text_first(reader):
    texts = ["Bananas are yellow.", "Use reversed() to flip a Python list.", "The train leaves at noon."]
    order = [index for index, _ in reader.rank("how do I reverse a list in python", texts)]
    assert order[0] == 1


def test_lexical_overlap_works_without_english(reader):
    same = reader.lexical("Türkiye'nin başkenti neresi?", "Türkiye'nin başkenti Ankara'dır.")
    other = reader.lexical("Türkiye'nin başkenti neresi?", "Bugün hava çok güzel.")
    assert same > 0.5 > other


def test_embedding_is_a_unit_vector(reader):
    vector = reader.embed("A context window is the text a model can see at once.")
    assert len(vector) == 64
    assert abs(sum(x * x for x in vector) - 1.0) < 1e-9


# ---- behaviour ----

@pytest.mark.parametrize("sentence, label", [
    ("As an AI language model, I cannot browse the internet.", "denial"),
    ("I don't have access to real-time information.", "denial"),
    ("I am ChatGPT, a large language model trained by OpenAI.", "other_identity"),
    ("I'm Qwen, developed by Alibaba Cloud.", "other_identity"),
    ("I've saved the file to your desktop.", "action_claim"),
    ("Let me search the web for that.", "action_promise"),
    ("The user is asking about the weather.", "meta_leak"),
])
def test_probe_names_the_behaviour(reader, sentence, label):
    scores = reader.probe(sentence)
    assert max(scores, key=scores.get) == label
    assert scores[label] >= reader.meta["probe"]["threshold"]


@pytest.mark.parametrize("sentence", [
    "Paris is the capital of France.",
    "ChatGPT is a chatbot made by OpenAI.",
    "I'm Kiwibro, an assistant that runs on your computer.",
    "You can save the file with Ctrl+S.",
    "Let me know if you need anything else.",
    "The user account must have admin rights.",
    "I can search the web for that if you like.",
])
def test_probe_leaves_ordinary_talk_alone(reader, sentence):
    scores = reader.probe(sentence)
    worst = max((value for label, value in scores.items() if label != "fine"), default=0.0)
    assert worst < reader.meta["probe"]["threshold"]


def test_probe_scores_are_probabilities(reader):
    scores = reader.probe("Hello there.")
    assert set(scores) == set(reader.labels)
    assert abs(sum(scores.values()) - 1.0) < 1e-9
    assert reader.probe("") == {}


def test_handwritten_sentences_keep_their_score(reader):
    """The honest test set: written by hand, never trained on."""
    sys.path.insert(0, str(ROOT / "scripts" / "sense"))
    try:
        import corpus
    finally:
        sys.path.pop(0)
    threshold = reader.meta["probe"]["threshold"]
    right = fine_flagged = 0
    for sentence, label in corpus.HELD_OUT:
        scores = reader.probe(sentence)
        best = max(scores, key=scores.get)
        right += best == corpus.CLASSES[label]
        fine_flagged += label == corpus.FINE and best != "fine" and scores[best] >= threshold
    recorded = reader.meta["probe"]["handwritten_test"]
    assert recorded["sentences"] == len(corpus.HELD_OUT)
    assert right / len(corpus.HELD_OUT) == pytest.approx(recorded["accuracy"], abs=1e-4)
    assert right / len(corpus.HELD_OUT) >= 0.9
    assert fine_flagged == recorded["fine_flagged"] == 0


def test_repeat_sees_loops_and_ignores_normal_text(reader):
    assert reader.repeat("I will check that for you right now. " * 10) > 0.8
    assert reader.repeat("A quick brown fox jumps over the lazy dog while the sun sets behind the quiet hills.") == 0.0
    assert reader.repeat("short text") == 0.0


def test_one_reader_serves_many_threads(reader):
    wanted = reader.probe("I am ChatGPT.")
    results: list[dict[str, float]] = []

    def work() -> None:
        for _ in range(20):
            reader.embed("Paris is the capital of France.")
            results.append(reader.probe("I am ChatGPT."))

    threads = [threading.Thread(target=work) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 120 and all(result == wanted for result in results)


def test_shared_reader_loads_once():
    first = sense.shared()
    assert first is not None and sense.shared() is first


# ---- the WebAssembly core and its Python twin give the same numbers ----

def _cases() -> dict:
    texts = [sense.normalize(text) for text in TEXTS]
    pairs = [[texts[i], texts[(i * 7 + 3) % len(texts)]] for i in range(len(texts))]
    return {"texts": texts, "pairs": pairs}


def _python_results(engine) -> dict:
    cases = _cases()
    out = {"texts": [], "pairs": []}
    for text in cases["texts"]:
        raw = text.encode("utf-8")
        vector, known = engine.embed(raw)
        words = engine.last_words
        out["texts"].append({
            "known": known, "words": words, "vector": list(vector),
            "scores": engine.probe(raw), "repeat": engine.repeat(raw),
        })
    for a, b in cases["pairs"]:
        out["pairs"].append({
            "similarity": engine.similarity(a.encode("utf-8"), b.encode("utf-8")),
            "lexsim": engine.lexsim(a.encode("utf-8"), b.encode("utf-8")),
        })
    return out


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the WebAssembly file")
def test_wasm_core_matches_the_python_twin_exactly(tmp_path):
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps(_cases()), encoding="utf-8")
    done = subprocess.run(
        ["node", str(ROOT / "tests" / "sense_wasm.mjs"), str(DATA / "sense.wasm"), str(DATA / "sense.bin"), str(cases_file)],
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    from_wasm = json.loads(done.stdout)
    from_python = _python_results(PyEngine((DATA / "sense.bin").read_bytes()))
    assert from_wasm["dims"] == 64 and from_wasm["classes"] == 6
    assert from_wasm["texts"] == from_python["texts"]
    assert from_wasm["pairs"] == from_python["pairs"]


def test_wasmtime_host_matches_the_python_twin_exactly():
    pytest.importorskip("wasmtime")
    from kiwibro.sense.wasmengine import WasmEngine

    blob = (DATA / "sense.bin").read_bytes()
    assert _python_results(WasmEngine(DATA / "sense.wasm", blob)) == _python_results(PyEngine(blob))
    assert sense.Sense(engine="wasm").engine == "wasm"


# ---- the browser's reader (static/js/sense.js) agrees with the server's ----

@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the browser module")
def test_browser_reader_matches_the_python_reader(tmp_path, reader):
    texts = [text for text in TEXTS if text.strip()] + ["Ünïcödé “quotes” — and… dashes", "  spaced\tout text here "]
    texts_file = tmp_path / "texts.json"
    texts_file.write_text(json.dumps(texts), encoding="utf-8")
    done = subprocess.run(
        ["node", str(ROOT / "tests" / "sense_browser.mjs"), str(ROOT / "static" / "js" / "sense.js"),
         str(DATA / "sense.wasm"), str(DATA / "sense.bin"), str(texts_file)],
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    seen = json.loads(done.stdout)
    assert seen["labels"] == reader.labels
    assert seen["threshold"] == reader.meta["probe"]["threshold"]
    for text, got in zip(texts, seen["texts"]):
        assert got["normalized"] == sense.normalize(text), text
        wanted = reader.probe(text)
        assert set(got["probe"]) == set(wanted)
        for label, value in wanted.items():
            assert got["probe"][label] == pytest.approx(value, abs=1e-9), (text, label)
        vector = reader.embed(text)
        if vector is None:
            assert got["vector"] is None
        else:
            assert got["vector"] == pytest.approx(list(vector), abs=1e-12)
        closeness = reader.similarity(text, texts[0])
        if closeness is None:
            assert got["similarity"] is None
        else:
            assert got["similarity"] == pytest.approx(closeness, abs=1e-9)
