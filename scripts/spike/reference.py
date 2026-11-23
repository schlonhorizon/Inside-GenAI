#!/usr/bin/env python3
"""Spike P1a, Schritt 1: Referenzwerte mit PyTorch/transformers (läuft in GitHub Actions).

Lädt SmolLM2-135M (Base) und SmolLM2-135M-Instruct vom Hugging Face Hub, rechnet in fp32
und schreibt nach spike-out/:
  reference.json          Revisionen, Configs, Prompts mit Token-IDs, Top-10, Greedy-Fortsetzung
  ref-<prompt>.f32        komplette Logits nach dem letzten Prompt-Token (für Korrelation/Fehler)
"""
import json
import pathlib

import numpy as np
import torch
from huggingface_hub import model_info, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = pathlib.Path("spike-out")
OUT.mkdir(exist_ok=True)

MODELS = {
    "base": "HuggingFaceTB/SmolLM2-135M",
    "instruct": "HuggingFaceTB/SmolLM2-135M-Instruct",
}

# id, Modell, Art, Text. "golden" = Korrektheitsvergleich, "candidate" = Leitbeispiel-Kandidat.
PROMPTS = [
    ("capital", "base", "golden", "The capital of France is"),
    ("centralbank", "base", "golden", "The central bank raised interest rates because"),
    ("story", "base", "candidate", "Once upon a time, there was a"),
    ("stocks", "base", "candidate", "The stock market fell sharply after"),
    ("chat-rates", "instruct", "chat", "Why do central banks raise interest rates?"),
    ("chat-capital", "instruct", "chat", "What is the capital of France?"),
]
GREEDY_STEPS = 16
TOP = 10

torch.set_grad_enabled(False)
result = {"models": {}, "prompts": []}
loaded = {}

for key, repo in MODELS.items():
    sha = model_info(repo).sha
    path = snapshot_download(repo, revision=sha, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
    cfg = json.loads((pathlib.Path(path) / "config.json").read_text())
    st_files = sorted(p.name for p in pathlib.Path(path).glob("*.safetensors"))
    result["models"][key] = {"repo": repo, "sha": sha, "path": path, "config": cfg, "safetensors": st_files}
    loaded[key] = (tok, model)
    print(f"[ref] {repo}@{sha[:10]} geladen ({', '.join(st_files)})")


def topk(logits, tok):
    probs = torch.softmax(logits, -1)
    p, idx = torch.topk(probs, TOP)
    return [
        {"id": int(i), "tok": tok.decode([int(i)]), "p": float(pp), "logit": float(logits[int(i)])}
        for pp, i in zip(p, idx)
    ]


for pid, mkey, kind, text in PROMPTS:
    tok, model = loaded[mkey]
    if kind == "chat":
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
    else:
        ids = tok(text)["input_ids"]
    x = torch.tensor([ids])
    logits = model(x).logits[0, -1].float()
    logits.numpy().astype("<f4").tofile(OUT / f"ref-{pid}.f32")
    # Greedy-Fortsetzung, Schritt für Schritt (genau wie die Engine)
    seq = list(ids)
    greedy = []
    for _ in range(GREEDY_STEPS):
        nxt = int(model(torch.tensor([seq])).logits[0, -1].argmax())
        greedy.append(nxt)
        seq.append(nxt)
    result["prompts"].append(
        {
            "id": pid,
            "model": mkey,
            "kind": kind,
            "text": text,
            "ids": ids,
            "rendered": tok.decode(ids),
            "top": topk(logits, tok),
            "greedy_ids": greedy,
            "greedy_text": tok.decode(greedy),
        }
    )
    print(f"[ref] {pid}: {len(ids)} Tokens → {tok.decode(greedy)!r}")

(OUT / "reference.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print("[ref] fertig")
