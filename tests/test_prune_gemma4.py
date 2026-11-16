import json
import struct

import pytest

from scripts.mind.prune_gemma4 import BASE_REVISION, INDEX, SLICED, _read_header, choose, export


def make_checkpoint(root):
    root.mkdir()
    weights = {}
    names = ["model.visual.weight", "model.language_model.layers.0.router.scale"]
    names += sorted(f"model.language_model.layers.0.{key}" for key in SLICED)
    payload = bytearray()
    for number, name in enumerate(names):
        values = b"".join(struct.pack("<H", number * 10 + item) for item in range(5))
        weights[name] = {"dtype": "BF16", "shape": [5], "data_offsets": [len(payload), len(payload) + len(values)]}
        payload += values
    header = json.dumps(weights).encode()
    header += b" " * (-len(header) % 8)
    (root / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (root / INDEX).write_text(json.dumps({"weight_map": {name: "weights.safetensors" for name in weights}}))
    config = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {"enable_moe_block": True, "num_experts": 5, "top_k_experts": 2, "num_hidden_layers": 1},
        "vision_config": {"hidden_size": 5},
    }
    (root / "config.json").write_text(json.dumps(config))


def test_pruning_keeps_rare_workflow_experts_and_preserves_tensor_alignment(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "small"
    make_checkpoint(source)
    profile = {
        "base_revision": BASE_REVISION,
        "layers": {"0": {"repository": [98, 0, 2, 0, 0], "vision": [0, 1, 0, 9, 0]}},
    }
    report = export(source, dest, profile, retain=2, top_k=1)
    assert report["kept_original_experts"] == {0: [0, 3]}
    assert report["retained_router_mass"] == {0: {"repository": 0.98, "vision": 0.9}}
    old = _tensors(source / "weights.safetensors")
    new = _tensors(dest / "weights.safetensors")
    for name, (shape, values) in new.items():
        expected = old[name][1]
        if any(name.endswith(f".{name_part}") for name_part in SLICED):
            assert shape == [2]
            assert values == expected[:2] + expected[6:8]
        else:
            assert values == expected
    text = json.loads((dest / "config.json").read_text())["text_config"]
    assert (text["num_experts"], text["top_k_experts"]) == (2, 1)


def _tensors(shard):
    with shard.open("rb") as handle:
        header, start, _ = _read_header(handle)
        handle.seek(start)
        values = handle.read()
    return {
        name: (item["shape"], values[slice(*item["data_offsets"])]) for name, item in header.items()
    }


def test_bad_calibration_and_mismatched_revision_are_rejected(tmp_path):
    source = tmp_path / "source"
    make_checkpoint(source)
    with pytest.raises(ValueError, match="different checkpoint"):
        export(source, tmp_path / "small", {"base_revision": "wrong"}, retain=2, top_k=1)
    with pytest.raises(ValueError, match="No router observations"):
        choose({"empty": [0] * 5}, 5, 2)
    assert not (tmp_path / "small").exists()
