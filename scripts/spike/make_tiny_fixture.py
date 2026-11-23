#!/usr/bin/env python3
"""Erzeugt ein winziges Zufalls-Llama-Modell + Referenz-Logits als Testfixture.

Die Referenz ist eine unabhängige NumPy-Implementierung in Matrixform (ganze Sequenz,
kausale Maske, kein KV-Cache) nach dem Hugging-Face-Llama-Code. Sie prüft die TS-Engine
(Token-für-Token mit KV-Cache, GQA, RoPE) auf Indexfehler. Den Abgleich mit dem echten
PyTorch/transformers-Modell macht der Spike-Workflow in GitHub Actions.

    python scripts/spike/make_tiny_fixture.py   # schreibt test/fixtures/tiny-llama.*
"""
import json
import pathlib

import numpy as np
from safetensors.numpy import save_file

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "fixtures"

CFG = dict(
    vocab_size=64,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=48,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    max_position_embeddings=64,
    tie_word_embeddings=True,
)
TOKENS = [1, 17, 5, 42, 9, 33, 2, 60, 11, 7]


def make_weights(rng):
    d, V = CFG["hidden_size"], CFG["vocab_size"]
    H, KV = CFG["num_attention_heads"], CFG["num_key_value_heads"]
    hd = d // H
    ff = CFG["intermediate_size"]
    w = {"model.embed_tokens.weight": rng.normal(0, 0.5, (V, d))}
    for i in range(CFG["num_hidden_layers"]):
        p = f"model.layers.{i}."
        w[p + "input_layernorm.weight"] = 1 + rng.normal(0, 0.1, d)
        w[p + "self_attn.q_proj.weight"] = rng.normal(0, 0.3, (H * hd, d))
        w[p + "self_attn.k_proj.weight"] = rng.normal(0, 0.3, (KV * hd, d))
        w[p + "self_attn.v_proj.weight"] = rng.normal(0, 0.3, (KV * hd, d))
        w[p + "self_attn.o_proj.weight"] = rng.normal(0, 0.3, (d, H * hd))
        w[p + "post_attention_layernorm.weight"] = 1 + rng.normal(0, 0.1, d)
        w[p + "mlp.gate_proj.weight"] = rng.normal(0, 0.3, (ff, d))
        w[p + "mlp.up_proj.weight"] = rng.normal(0, 0.3, (ff, d))
        w[p + "mlp.down_proj.weight"] = rng.normal(0, 0.3, (d, ff))
    w["model.norm.weight"] = 1 + rng.normal(0, 0.1, d)
    return {k: v.astype(np.float32) for k, v in w.items()}


def rms(x, g, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g


def rope(x, pos, theta):
    # x: (T, heads, hd); HF: rotate_half auf (erste Hälfte, zweite Hälfte)
    hd = x.shape[-1]
    inv = 1.0 / theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = np.outer(pos, inv)  # (T, hd/2)
    cos = np.concatenate([np.cos(ang)] * 2, -1)[:, None, :]
    sin = np.concatenate([np.sin(ang)] * 2, -1)[:, None, :]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    rot = np.concatenate([-x2, x1], -1)
    return x * cos + rot * sin


def forward(w, tokens):
    c = CFG
    d, H, KV = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"]
    hd = d // H
    T = len(tokens)
    x = w["model.embed_tokens.weight"][tokens].astype(np.float64)
    pos = np.arange(T)
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        h = rms(x, w[p + "input_layernorm.weight"], c["rms_norm_eps"])
        q = (h @ w[p + "self_attn.q_proj.weight"].T).reshape(T, H, hd)
        k = (h @ w[p + "self_attn.k_proj.weight"].T).reshape(T, KV, hd)
        v = (h @ w[p + "self_attn.v_proj.weight"].T).reshape(T, KV, hd)
        q, k = rope(q, pos, c["rope_theta"]), rope(k, pos, c["rope_theta"])
        k = np.repeat(k, H // KV, axis=1)
        v = np.repeat(v, H // KV, axis=1)
        s = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd) + mask
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", s, v).reshape(T, H * hd)
        x = x + o @ w[p + "self_attn.o_proj.weight"].T
        h = rms(x, w[p + "post_attention_layernorm.weight"], c["rms_norm_eps"])
        g = h @ w[p + "mlp.gate_proj.weight"].T
        u = h @ w[p + "mlp.up_proj.weight"].T
        x = x + ((g / (1 + np.exp(-g))) * u) @ w[p + "mlp.down_proj.weight"].T
    x = rms(x, w["model.norm.weight"], c["rms_norm_eps"])
    return x @ w["model.embed_tokens.weight"].T  # gebundene Embeddings


def main():
    rng = np.random.default_rng(1234)
    w = make_weights(rng)
    logits = forward(w, TOKENS)
    OUT.mkdir(parents=True, exist_ok=True)
    save_file(w, str(OUT / "tiny-llama.safetensors"))
    (OUT / "tiny-llama.config.json").write_text(json.dumps(CFG, indent=2) + "\n")
    ref = {"tokens": TOKENS, "logits": [[round(float(v), 6) for v in row] for row in logits]}
    (OUT / "tiny-llama.reference.json").write_text(json.dumps(ref) + "\n")
    print(f"fixture: {sum(v.size for v in w.values())} Parameter, {len(TOKENS)} Tokens")


if __name__ == "__main__":
    main()
