import json
import struct

import pytest

np = pytest.importorskip("numpy")

from scripts.mind.compact_e4b import INDEX, REPORT, export

BEFORE, AFTER, HIDDEN = 512, 256, 4
LAYER = "model.language_model.layers.0.mlp."
VISION = "model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight"


def make_shard(path, values):
    payload = bytearray()
    header = {"__metadata__": {"format": "pt"}}
    for name, array in values.items():
        raw = array.tobytes()
        header[name] = {"dtype": "BF16", "shape": list(array.shape),
                        "data_offsets": [len(payload), len(payload) + len(raw)]}
        payload.extend(raw)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def unpack(path, key):
    data = path.read_bytes()
    size, = struct.unpack_from("<Q", data)
    header = json.loads(data[8:8 + size])
    item = header[key]
    left, right = item["data_offsets"]
    return np.frombuffer(data[8 + size + left:8 + size + right], dtype="<u2").reshape(item["shape"])


def bf16(array):
    return (array.astype("<f4").view("<u4") >> 16).astype("<u2")


def fixture(root, *, sharded):
    root.mkdir()
    c = np.arange(BEFORE, dtype=np.float32) + 1
    c[BEFORE - AFTER:] *= 4
    lane = np.arange(HIDDEN, dtype=np.float32) + 1
    values = {
        LAYER + "gate_proj.weight": bf16(c[:, None] * lane),
        LAYER + "up_proj.weight": bf16(c[:, None] * (lane + 1.5)),
        LAYER + "down_proj.weight": bf16((lane[:, None] + 3.5) * c),
        VISION: bf16(np.arange(21).reshape(3, 7)),
    }
    original_map = {key: "model.safetensors" for key in values}
    if sharded:
        original_map.update({
            LAYER + "up_proj.weight": "model-2.safetensors",
            LAYER + "down_proj.weight": "model-3.safetensors",
        })
        (root / INDEX).write_text(json.dumps({"weight_map": original_map, "metadata": {"fixture": True}}))
    for shard in set(original_map.values()):
        make_shard(root / shard, {k: v for k, v in values.items() if original_map[k] == shard})
    config = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {"num_hidden_layers": 1, "hidden_size": HIDDEN,
                        "intermediate_size": BEFORE, "enable_moe_block": False,
                        "use_double_wide_mlp": False, "hidden_size_per_layer_input": 2},
        "vision_config": {"hidden_size": 7}, "audio_config": {"hidden_size": 2},
    }
    (root / "config.json").write_text(json.dumps(config))
    for name in ("tokenizer.json", "processor_config.json", "chat_template.jinja"):
        (root / name).write_text("original " + name + "\n")
    return config, original_map, values


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("activation", [False, True])
def test_export(sharded, activation, tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    config, original_map, values = fixture(source, sharded=sharded)
    if activation:
        score_file = tmp_path / "scores.npy"
        selected = np.sort(np.r_[np.arange(126), np.arange(382, BEFORE)])
        scores = np.ones((1, BEFORE), dtype=np.float32)
        scores[0, selected] = 10
        np.save(score_file, scores)
    else:
        score_file = None
        selected = np.arange(BEFORE - AFTER, BEFORE)

    report = export(source, output, scores=score_file, width=AFTER, expected_shape=(1, HIDDEN, BEFORE))

    assert json.loads((output / REPORT).read_text()) == report
    assert report["kept_original_channels"] == {"0": selected.tolist()}
    assert bool(report["activation_scores_sha256"]) is activation
    assert report["selection"] == ("activation_scores" if activation else "weight_l2_product_fallback")
    config["text_config"]["intermediate_size"] = AFTER
    assert json.loads((output / "config.json").read_text()) == config
    for name in ("tokenizer.json", "processor_config.json", "chat_template.jinja"):
        assert (output / name).read_bytes() == (source / name).read_bytes()
    index = json.loads((output / INDEX).read_text())
    assert index["weight_map"] == original_map
    assert index["metadata"].get("fixture", False) is sharded
    total = 0
    for name, original in values.items():
        expected = original
        if name.startswith(LAYER):
            expected = original[:, selected] if "down_proj" in name else original[selected]
        actual = unpack(output / original_map[name], name)
        np.testing.assert_array_equal(actual, expected)
        total += actual.nbytes
    assert index["metadata"]["total_size"] == total
    assert report["removed_weights"] == 3 * (BEFORE - AFTER) * HIDDEN


def test_invalid_scores_create_no_output(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    fixture(source, sharded=False)
    scores = tmp_path / "scores.npy"
    np.save(scores, np.full((1, BEFORE), np.nan))
    with pytest.raises(ValueError, match="finite, nonnegative"):
        export(source, output, scores=scores, width=AFTER, expected_shape=(1, HIDDEN, BEFORE))
    assert not output.exists()
