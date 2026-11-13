"""A NumPy forward pass of the Kiwi Mind trunk, skills and control head, for checking the C engine.

It reads the same files the engine reads (scripts/mind/formats.py) and does the arithmetic with float
weights (the dequantised values) in float64 sums. With emulate=True (the default) it also repeats the engine's
three deliberate roundings, which is what makes a tight comparison possible:

  * the input of every integer matrix product is quantised to int8 per group of 32 values (round half away
    from zero, scale = max|x| / 127), then used as q * scale;
  * the KV cache holds int8 with one scale per 32 values (the first 4 positions stay exact floats);
  * the query is rounded to int16 (scale = max|q| / 32767) for the product with the int8 keys.

With emulate=False none of that happens: the result is the model's exact float answer, and the difference to
the engine measures what the quantisation costs.

    python reference.py DIR CASES.json OUT.json
CASES.json: {"model": "tiny.fkm", "emulate": true, "cases": [{"tokens": [...], "skill": "alpha.kms" | null,
"head": "head.kmh" | null, "state_layers": [2]}]}. OUT.json: per case "logits" (last position), "states"
(hidden state at the input of each listed layer, every position) and "control" (head output, last position).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import formats  # noqa: E402

F32 = np.float32
SINK = 4


def quantize_act(x: np.ndarray) -> np.ndarray:
    """What the engine's int8 activation quantisation turns x [T, cols] into (as floats)."""
    t, cols = x.shape
    g = x.reshape(t, cols // 32, 32).astype(F32)
    d = np.abs(g).max(-1, keepdims=True) / F32(127)
    inv = np.where(d > 0, F32(1) / np.where(d > 0, d, F32(1)), F32(0)).astype(F32)
    v = g * inv
    q = np.trunc(v + np.copysign(F32(0.5), v))
    return (q * d).reshape(t, cols).astype(F32)


def quantize_kv(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x [T, heads, head_dim] -> int8 codes [T, heads, groups, 32] and scales [T, heads, groups]."""
    t, h, hd = x.shape
    g = x.reshape(t, h, hd // 32, 32).astype(F32)
    d = np.abs(g).max(-1, keepdims=True) / F32(127)
    inv = np.where(d > 0, F32(1) / np.where(d > 0, d, F32(1)), F32(0)).astype(F32)
    v = g * inv
    return np.trunc(v + np.copysign(F32(0.5), v)).astype(np.int32), d[..., 0]


def rms(x: np.ndarray, w: np.ndarray | None, eps: float) -> np.ndarray:
    x = x.astype(F32)
    r = F32(1) / np.sqrt(np.mean(x * x, axis=-1, keepdims=True, dtype=F32) + F32(eps))
    y = x * r
    return (y * w).astype(F32) if w is not None else y.astype(F32)


def rope_table(config: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    hd = config["head_dim"]
    inv = np.array([np.float32(1.0 / (float(config["rope_theta"]) ** (2 * i / hd))) for i in range(hd // 2)], dtype=F32)
    ang = np.arange(n, dtype=F32)[:, None] * inv[None, :]
    return np.cos(ang.astype(np.float64)).astype(F32), np.sin(ang.astype(np.float64)).astype(F32)


class Reference:
    def __init__(self, trunk: Path, emulate: bool = True):
        self.config, self.w, self.types = formats.read_trunk(trunk)
        self.emulate = emulate

    def matrix(self, layer: int, name: str) -> tuple[np.ndarray, int]:
        key = f"model.layers.{layer}.{formats.MATRIX_PARTS[name]}"
        return self.w[key], self.types[key]

    def linear(self, x: np.ndarray, w: np.ndarray, typ: int, delta=None) -> np.ndarray:
        xin = quantize_act(x) if self.emulate and typ != formats.F32 else x
        y = xin.astype(np.float64) @ w.astype(np.float64).T
        if delta is not None:
            a, b, scale = delta
            y = y + scale * ((x.astype(np.float64) @ a.astype(np.float64).T) @ b.astype(np.float64).T)
        return y.astype(F32)

    def attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        """q [T, NH, HD], k/v [T, KV, HD] after norm and rope; causal full attention over the sequence."""
        c = self.config
        t, nh, hd = q.shape
        kv = c["n_kv_heads"]
        out = np.zeros((t, nh, hd), dtype=F32)
        inv = F32(1) / np.sqrt(F32(hd))
        if self.emulate:
            kq, ks = quantize_kv(k)
            vq, vs = quantize_kv(v)
            k_deq = (kq * ks[..., None]).reshape(t, kv, hd).astype(F32)
            v_deq = (vq * vs[..., None]).reshape(t, kv, hd).astype(F32)
        else:
            k_deq, v_deq = k, v
        for h in range(nh):
            kh = h // (nh // kv)
            scores = np.zeros((t, t))
            if self.emulate:
                amax = np.abs(q[:, h]).max(-1, keepdims=True)
                qinv = np.where(amax > 0, F32(32767) / np.where(amax > 0, amax, F32(1)), F32(0)).astype(F32)
                q16 = np.trunc(q[:, h] * qinv + np.where(q[:, h] < 0, F32(-0.5), F32(0.5))).astype(np.int64)
                groups = hd // 32
                dots = np.einsum("tgi,pgi->tpg", q16.reshape(t, groups, 32), kq[:, kh].astype(np.int64))
                scores = (dots * ks[:, kh][None, :, :]).sum(-1) * (amax / F32(32767))
                exact = q[:, h].astype(np.float64) @ k[:SINK, kh].astype(np.float64).T
                scores[:, :SINK] = exact  # the first positions are kept exactly
            else:
                scores = q[:, h].astype(np.float64) @ k_deq[:, kh].astype(np.float64).T
            scores = scores * float(inv)
            scores = np.where(np.tril(np.ones((t, t), dtype=bool)), scores, -np.inf)
            p = np.exp(scores - scores.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            vh = v_deq[:, kh].astype(np.float64).copy()
            if self.emulate:
                vh[:SINK] = v[:SINK, kh]
            out[:, h] = (p @ vh).astype(F32)
        return out

    def forward(self, tokens: list[int], skill: dict | None = None, state_layers: tuple[int, ...] = ()) -> dict:
        """Logits of the last position and the hidden state at the input of each of state_layers (all positions)."""
        c = self.config
        d, nh, kv, hd, eps = c["dim"], c["n_heads"], c["n_kv_heads"], c["head_dim"], c["norm_eps"]
        deltas = formats.skill_deltas(skill, c) if skill else {}
        t = len(tokens)
        cos, sin = rope_table(c, t)
        x = self.w["embed"][tokens].astype(F32)
        states = {}
        for layer in range(c["n_layers"]):
            if layer in state_layers:
                states[layer] = x.copy()
            p = f"model.layers.{layer}."
            h = rms(x, self.w[p + "input_layernorm"].reshape(-1), eps)
            proj = {}
            for m in ("q", "k", "v"):
                w, typ = self.matrix(layer, m)
                proj[m] = self.linear(h, w, typ, deltas.get((layer, m)))
            q = rms(proj["q"].reshape(t, nh, hd), self.w[p + "self_attn.q_norm"].reshape(-1), eps)
            k = rms(proj["k"].reshape(t, kv, hd), self.w[p + "self_attn.k_norm"].reshape(-1), eps)
            for arr in (q, k):
                a, b = arr[..., : hd // 2].copy(), arr[..., hd // 2:].copy()
                arr[..., : hd // 2] = a * cos[:, None, :] - b * sin[:, None, :]
                arr[..., hd // 2:] = b * cos[:, None, :] + a * sin[:, None, :]
            att = self.attention(q, k, proj["v"].reshape(t, kv, hd)).reshape(t, nh * hd)
            w, typ = self.matrix(layer, "o")
            x = (x + self.linear(att, w, typ, deltas.get((layer, "o")))).astype(F32)
            h = rms(x, self.w[p + "post_attention_layernorm"].reshape(-1), eps)
            wg, tg = self.matrix(layer, "gate")
            wu, tu = self.matrix(layer, "up")
            g = self.linear(h, wg, tg, deltas.get((layer, "gate"))).astype(np.float64)
            u = self.linear(h, wu, tu, deltas.get((layer, "up"))).astype(np.float64)
            ff = (g / (1.0 + np.exp(-g)) * u).astype(F32)
            w, typ = self.matrix(layer, "down")
            x = (x + self.linear(ff, w, typ, deltas.get((layer, "down")))).astype(F32)
        if c["n_layers"] in state_layers:
            states[c["n_layers"]] = x.copy()
        last = rms(x[-1:], self.w["model.norm"].reshape(-1), eps)
        logits = self.linear(last, self.w["embed"], self.types["embed"])[0]
        return {"logits": logits, "states": states}


def control(head: dict, state: np.ndarray) -> np.ndarray:
    x = state.astype(np.float64)
    x = x / np.sqrt(np.mean(x * x) + head["eps"])
    h = np.maximum(head["w1"].astype(np.float64) @ x + head["b1"], 0.0)
    return head["w2"].astype(np.float64) @ h + head["b2"]


def main(argv: list[str]) -> int:
    root, cases_path, out_path = Path(argv[1]), Path(argv[2]), Path(argv[3])
    spec = json.loads(cases_path.read_text())
    ref = Reference(root / spec["model"], emulate=spec.get("emulate", True))
    results = []
    for case in spec["cases"]:
        skill = formats.read_skill(root / case["skill"]) if case.get("skill") else None
        head = formats.read_head(root / case["head"]) if case.get("head") else None
        layers = set(case.get("state_layers", []))
        head_layer = None
        if head:
            head_layer = head["layer"] if head["layer"] is not None else (skill["first_layer"] if skill else
                                                                           case.get("split", ref.config["n_layers"]))
            layers.add(head_layer)
        out = ref.forward(case["tokens"], skill, tuple(sorted(layers)))
        result = {"logits": out["logits"].tolist(),
                  "states": {str(l): out["states"][l].tolist() for l in case.get("state_layers", [])}}
        if head:
            result["control"] = control(head, out["states"][head_layer][-1]).tolist()
        results.append(result)
    out_path.write_text(json.dumps({"cases": results}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
