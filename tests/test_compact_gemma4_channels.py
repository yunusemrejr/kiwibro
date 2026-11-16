import base64
import json
import struct

import numpy as np
import pytest

from scripts.mind.compact_gemma4_channels import DOWN, GATE_UP, export
from scripts.mind.prune_gemma4 import BASE_REVISION, INDEX, _read_header


LAYER = "model.language_model.layers.0."
BEFORE, AFTER, HIDDEN, EXPERTS = 704, 512, 4, 2
FIRST = np.r_[np.arange(448), np.arange(640, BEFORE)]
SECOND = np.arange(AFTER)


def bf16(values):
    return (values.astype("<f4").view("<u4") >> 16).astype("<u2")


def write_shard(path, weights):
    metadata = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name, array in weights.items():
        raw = array.tobytes()
        metadata[name] = {
            "dtype": "BF16", "shape": list(array.shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload.extend(raw)
    header = json.dumps(metadata).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def read_tensor(path, name):
    with path.open("rb") as handle:
        header, start, _ = _read_header(handle)
        info = header[name]
        left, right = info["data_offsets"]
        handle.seek(start + left)
        value = np.frombuffer(handle.read(right - left), dtype="<u2").reshape(info["shape"])
    return value


def make_checkpoint(root, *, split, bad_shape=False, auxiliary=False):
    root.mkdir()
    rng = np.random.default_rng(42)
    gate_up = rng.choice([-1, 1], (EXPERTS, 2 * BEFORE, HIDDEN))
    down = rng.choice([-1, 1], (EXPERTS, HIDDEN, BEFORE))
    gate_up[0, FIRST[0::3], :] *= 4
    gate_up[0, BEFORE + FIRST[1::3], :] *= 4
    down[0, :, FIRST[2::3]] *= 4
    weights = {
        "a.safetensors": {
            LAYER + GATE_UP: bf16(gate_up),
            LAYER + "router.scale": bf16(np.arange(HIDDEN)),
            LAYER + "router.per_expert_scale": bf16(np.arange(EXPERTS) + 1),
        },
        "vision.safetensors": {"model.vision_tower.proj.weight": bf16(np.ones((2, 3)))},
    }
    down_shard = weights.setdefault("b.safetensors", {}) if split else weights["a.safetensors"]
    down_shard[LAYER + DOWN] = bf16(down[..., :-1] if bad_shape else down)
    if auxiliary:
        down_shard[LAYER + DOWN + ".weight_scale"] = bf16(np.ones((EXPERTS, HIDDEN, BEFORE)))
    for shard, tensors in weights.items():
        write_shard(root / shard, tensors)
    index = {"metadata": {"fixture": True}, "weight_map": {
        name: shard for shard, tensors in weights.items() for name in tensors
    }}
    config = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "vision_config": {"hidden_size": 3},
        "text_config": {
            "enable_moe_block": True, "num_experts": EXPERTS, "top_k_experts": 1,
            "num_hidden_layers": 1, "hidden_size": HIDDEN, "moe_intermediate_size": BEFORE,
            "intermediate_size": 1_024,
        },
    }
    architecture = {"base_revision": BASE_REVISION, "kept_original_experts": {"0": [7, 9]}}
    for filename, value in (("config.json", config), (INDEX, index), ("architecture.json", architecture)):
        (root / filename).write_text(json.dumps(value))
    (root / "training.json").write_text('{"steps":2}')
    return config, index, weights


@pytest.mark.parametrize("split", [False, True])
def test_compaction_keeps_coupled_channels_scales_and_expert_order(tmp_path, split):
    source, output = tmp_path / "source", tmp_path / "compact"
    config, index, weights = make_checkpoint(source, split=split)
    report = export(source, output)
    new_config = json.loads((output / "config.json").read_text())
    config["text_config"]["moe_intermediate_size"] = AFTER
    assert new_config == config
    new_index = json.loads((output / INDEX).read_text())
    assert new_index["weight_map"] == index["weight_map"]
    assert new_index["metadata"]["fixture"]

    for name, shard in new_index["weight_map"].items():
        original = weights[shard][name]
        actual = read_tensor(output / shard, name)
        if name == LAYER + GATE_UP:
            assert actual.shape == (EXPERTS, 2 * AFTER, HIDDEN)
            for expert, selected in enumerate((FIRST, SECOND)):
                np.testing.assert_array_equal(actual[expert, :AFTER], original[expert, selected])
                np.testing.assert_array_equal(actual[expert, AFTER:], original[expert, BEFORE + selected])
        elif name == LAYER + DOWN:
            assert actual.shape == (EXPERTS, HIDDEN, AFTER)
            for expert, selected in enumerate((FIRST, SECOND)):
                np.testing.assert_array_equal(actual[expert], original[expert][:, selected])
        else:
            np.testing.assert_array_equal(actual, original)
    assert (output / "vision.safetensors").read_bytes() == (source / "vision.safetensors").read_bytes()
    assert (output / "training.json").read_bytes() == (source / "training.json").read_bytes()
    assert report["kept_original_experts"] == {"0": [7, 9]}
    details = report["channel_compaction"]
    assert details["activation_calibration"] is None
    for encoded, expected in zip(details["layers"][0]["kept_channels"], (FIRST, SECOND), strict=True):
        packed = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        selected = np.flatnonzero(np.unpackbits(packed, bitorder="little")[:BEFORE])
        np.testing.assert_array_equal(selected, expected)
    sizes = []
    for p in output.glob("*.safetensors"):
        with p.open("rb") as handle:
            sizes.append(_read_header(handle)[-1])
    assert report["weight_bytes"] == new_index["metadata"]["total_size"] == sum(sizes)
    with pytest.raises(FileExistsError):
        export(source, output)


@pytest.mark.parametrize(
    ("fixture", "message"),
    [({"bad_shape": True}, "Unexpected BF16 expert shape"), ({"auxiliary": True}, "auxiliary expert tensors")],
)
def test_malformed_checkpoint_leaves_no_partial_output(tmp_path, fixture, message):
    source, output = tmp_path / "source", tmp_path / "compact"
    make_checkpoint(source, split=True, **fixture)
    with pytest.raises(ValueError, match=message):
        export(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".gemma-channels-*"))
