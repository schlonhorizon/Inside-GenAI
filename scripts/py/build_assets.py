#!/usr/bin/env python3
"""Erzeugt vorberechnete Assets (aufgerufen von .github/workflows/assets.yml).

Ziele:
  tokenizers  – tokenizer.json von GPT-2 und SmolLM2 (gepinnt) → public/tokenizers/,
                Lizenztexte → public/licenses/, Referenz-Tokenisierungen → test/fixtures/
  toy         – synthetisches deutsches Mini-Korpus, Training von zwei winzigen Llama-Modellen
                (mit RoPE und ohne Positionssignal) → public/data/toy/, public/data/toy-nopos/,
                Referenz-Logits → test/fixtures/
  golden      – (P2) Referenzwerte für die Golden-Tests
  embeddings  – (P3) fastText-Teilmenge
"""
import argparse
import json
import pathlib
import random
import re
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "public"
FIXTURES = ROOT / "test" / "fixtures"

# ─────────────────────────────── Tokenizer ────────────────────────────────

TOKENIZERS = {
    "gpt2": {"repo": "openai-community/gpt2", "revision": None, "license": "MIT"},
    "smollm2": {
        "repo": "HuggingFaceTB/SmolLM2-135M",
        "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        "license": "Apache-2.0",
    },
}

TEST_STRINGS = [
    "The capital of France is",
    "The central bank raised interest rates because",
    "Once upon a time, there was a",
    "Die Hauptstadt von Frankreich ist Paris.",
    "Die Zentralbank erhöht den Leitzins, weil die Inflation steigt.",
    "Unbegreiflichkeit",
    "Donaudampfschifffahrtsgesellschaftskapitän",
    "Größe, Übermaß, Äpfel und Öl – süß!",
    "strawberry",
    "How many r's are in strawberry?",
    "Incomprehensibility",
    "Hello world",
    " Hello world",
    "Hello  world   with   spaces",
    "Zeile eins\nZeile zwei\n\n\tTab",
    "1234567890 und 3.14159 sowie 1,5 %",
    "Der Kurs fiel um 12,7 % auf 1.024,50 €.",
    "it's, they're, I'll, we've, she'd",
    "emoji: 🙂🚀 ok",
    "日本語のテキスト",
    "naïve café résumé",
    "C++ und Python: print('hi')",
    "URL: https://example.com/a?b=c",
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>",
    "<|endoftext|>",
    "",
    "   ",
    "a",
    "Tokenisierung zerlegt Text in Tokens.",
    "Large language models predict the next token.",
]


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def build_tokenizers() -> None:
    from huggingface_hub import hf_hub_download, model_info
    from tokenizers import Tokenizer

    out = PUBLIC / "tokenizers"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    reference = {"strings": TEST_STRINGS, "tokenizers": {}}
    for key, t in TOKENIZERS.items():
        rev = t["revision"] or model_info(t["repo"]).sha
        path = hf_hub_download(t["repo"], "tokenizer.json", revision=rev)
        data = pathlib.Path(path).read_bytes()
        (out / f"{key}.json").write_bytes(data)
        manifest[key] = {"repo": t["repo"], "revision": rev, "license": t["license"], "file": f"{key}.json", "bytes": len(data)}
        tok = Tokenizer.from_file(path)
        rows = []
        for s in TEST_STRINGS:
            enc = tok.encode(s, add_special_tokens=False)
            rows.append({"ids": enc.ids, "tokens": enc.tokens, "decoded": tok.decode(enc.ids, skip_special_tokens=False)})
        reference["tokenizers"][key] = rows
        print(f"[tokenizers] {key}: {t['repo']}@{rev[:10]} ({len(data)//1024} KB)")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "tokenizer-reference.json").write_text(json.dumps(reference, ensure_ascii=False, indent=1) + "\n")

    lic = PUBLIC / "licenses"
    lic.mkdir(parents=True, exist_ok=True)
    (lic / "gpt2-LICENSE.txt").write_bytes(fetch("https://raw.githubusercontent.com/openai/gpt-2/master/LICENSE"))
    (lic / "smollm2-Apache-2.0.txt").write_bytes(fetch("https://www.apache.org/licenses/LICENSE-2.0.txt"))
    print("[tokenizers] Lizenztexte geschrieben")


# ──────────────────────────────── Toy-Modell ────────────────────────────────

SPECIALS = ["<unk>", "<bos>", "<eos>", "<pad>"]
WORD_RE = re.compile(r"[a-zäöüß]+|[.,!?]")


def words(s: str) -> list[str]:
    return WORD_RE.findall(s.lower())


def corpus(n: int, rng: random.Random) -> list[str]:
    capitals = {"frankreich": "paris", "deutschland": "berlin", "italien": "rom", "spanien": "madrid", "österreich": "wien"}
    out = []
    for _ in range(n):
        kind = rng.choices(["capital", "capital_rev", "bank", "market", "story"], [0.32, 0.08, 0.25, 0.15, 0.20])[0]
        if kind == "capital":
            land = rng.choice(list(capitals))
            tail = rng.choices(
                [f"{capitals[land]} .", "schön .", "sehr groß .", "eine alte stadt .", f"{capitals[land]} und sehr schön ."],
                [0.55, 0.12, 0.1, 0.1, 0.13],
            )[0]
            out.append(f"die hauptstadt von {land} ist {tail}")
        elif kind == "capital_rev":
            land = rng.choice(list(capitals))
            out.append(f"{capitals[land]} ist die hauptstadt von {land} .")
        elif kind == "bank":
            subj = rng.choice(["die bank", "die zentralbank"])
            if rng.random() < 0.6:
                why = rng.choices(["die inflation steigt", "die preise steigen", "die wirtschaft stark ist"], [0.55, 0.3, 0.15])[0]
                out.append(f"{subj} erhöht den zins weil {why} .")
            else:
                why = rng.choices(["die wirtschaft schwach ist", "die inflation sinkt", "die preise sinken"], [0.5, 0.3, 0.2])[0]
                out.append(f"{subj} senkt den zins weil {why} .")
        elif kind == "market":
            out.append(
                rng.choice(
                    [
                        "wenn der zins steigt , sinken die kurse .",
                        "wenn der zins sinkt , steigen die kurse .",
                        "die aktie steigt heute stark .",
                        "die aktie fällt heute leicht .",
                        "die aktie steigt heute leicht .",
                        "die kurse fallen heute stark .",
                    ]
                )
            )
        else:
            animal = rng.choice(["ein kleiner hund", "eine kluge katze", "ein alter fuchs", "eine kleine maus"])
            out.append(
                rng.choice(
                    [
                        f"es war einmal {animal} .",
                        "der hund spielt im garten .",
                        "die katze schläft auf dem sofa .",
                        "der kleine hund bellt laut .",
                        "die maus läuft schnell in den garten .",
                        "der fuchs ist sehr klug .",
                    ]
                )
            )
    return out


TOY_PROMPTS = ["die hauptstadt von frankreich ist", "die bank erhöht den zins weil", "es war einmal"]


def train_toy(sentences, vocab, rope: bool, outdir: pathlib.Path, seed: int = 0):
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM
    import transformers.models.llama.modeling_llama as llama_mod

    original = llama_mod.apply_rotary_pos_emb
    if not rope:
        # Kein Positionssignal: Rotation durch Identität ersetzen (cos = 1, sin = 0)
        llama_mod.apply_rotary_pos_emb = lambda q, k, cos, sin, *a, **kw: (q, k)
    try:
        torch.manual_seed(seed)
        stoi = {w: i for i, w in enumerate(vocab)}
        max_len = 24
        seqs = []
        for s in sentences:
            ids = [stoi["<bos>"]] + [stoi[w] for w in words(s)] + [stoi["<eos>"]]
            seqs.append(ids[:max_len])
        cfg = LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            tie_word_embeddings=True,
            bos_token_id=stoi["<bos>"],
            eos_token_id=stoi["<eos>"],
            pad_token_id=stoi["<pad>"],
        )
        model = LlamaForCausalLM(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
        pad = stoi["<pad>"]
        steps, bs = 3000, 64
        for step in range(steps):
            batch = random.Random(seed * 100000 + step).sample(seqs, bs)
            L = max(len(b) for b in batch)
            x = torch.full((bs, L), pad)
            labels = torch.full((bs, L), -100)
            for i, b in enumerate(batch):
                x[i, : len(b)] = torch.tensor(b)
                labels[i, : len(b)] = torch.tensor(b)
            loss = model(input_ids=x, labels=labels).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 500 == 0 or step == steps - 1:
                print(f"[toy{'' if rope else '-nopos'}] step {step} loss {loss.item():.4f}")
        model.eval()
        outdir.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
        save_file(state, str(outdir / "model.safetensors"))
        conf = cfg.to_dict()
        conf["rope"] = rope
        conf["final_loss"] = round(float(loss.item()), 4)
        (outdir / "config.json").write_text(json.dumps(conf, indent=2) + "\n")
        (outdir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1) + "\n")
        # Referenz-Logits (alle Positionen) für den Engine-Test
        ref = {"prompts": []}
        with torch.no_grad():
            for p in TOY_PROMPTS:
                ids = [stoi["<bos>"]] + [stoi[w] for w in words(p)]
                logits = model(input_ids=torch.tensor([ids])).logits[0]
                ref["prompts"].append({"text": p, "ids": ids, "logits": [[round(float(v), 5) for v in row] for row in logits]})
        return ref
    finally:
        llama_mod.apply_rotary_pos_emb = original


def build_toy() -> None:
    rng = random.Random(42)
    sentences = corpus(8000, rng)
    counts: dict[str, int] = {}
    for s in sentences:
        for w in words(s):
            counts[w] = counts.get(w, 0) + 1
    vocab = SPECIALS + sorted(counts, key=lambda w: (-counts[w], w))
    print(f"[toy] {len(sentences)} Sätze, Vokabular {len(vocab)}")
    info = {
        "sentences": len(sentences),
        "vocab_size": len(vocab),
        "examples": sentences[:20],
        "word_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    }
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for rope, name in [(True, "toy"), (False, "toy-nopos")]:
        outdir = PUBLIC / "data" / name
        ref = train_toy(sentences, vocab, rope, outdir)
        (outdir / "corpus-info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n")
        (FIXTURES / f"{name}-reference.json").write_text(json.dumps(ref) + "\n")


KNOWN = {"tokenizers": build_tokenizers, "toy": build_toy, "golden": None, "embeddings": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="tokenizers,toy")
    args = parser.parse_args()
    for t in [x.strip() for x in args.targets.split(",") if x.strip()]:
        if t not in KNOWN:
            raise SystemExit(f"Unbekanntes Ziel: {t} (bekannt: {', '.join(KNOWN)})")
        fn = KNOWN[t]
        if fn is None:
            print(f"[assets] {t}: noch nicht implementiert (folgt später)")
            continue
        fn()


if __name__ == "__main__":
    main()
