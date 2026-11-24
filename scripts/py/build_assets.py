#!/usr/bin/env python3
"""Erzeugt vorberechnete Assets (aufgerufen von .github/workflows/assets.yml).

Ziele:
  tokenizers  – tokenizer.json von GPT-2 und SmolLM2 (gepinnt) → public/tokenizers/,
                Lizenztexte → public/licenses/, Referenz-Tokenisierungen → test/fixtures/
  toy         – synthetisches deutsches Mini-Korpus, Training von zwei winzigen Llama-Modellen
                (mit RoPE und ohne Positionssignal) → public/data/toy/, public/data/toy-nopos/,
                Referenz-Logits → test/fixtures/
  golden      – Referenzwerte für die Golden-Tests: SmolLM2-135M Base + Instruct (gepinnt), je
                Prompt die IDs, die vollen Logits nach dem letzten Token (f32) und eine
                Greedy-Fortsetzung über 16 Tokens → test/fixtures/golden/
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


def corpus_v1(n: int, rng: random.Random) -> list[str]:
    """Korpus der ersten Toy-Version (P1b). Nur noch für den Vorher-Vergleich."""
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


# ── Korpus v2: gewichtete, endliche Grammatik ─────────────────────────────────
# Knoten: str (Wörter), S(...) = Folge, C((gewicht, knoten), ...) = Auswahl.
# Daraus: (a) exakte Aufzählung aller Sätze mit Wahrscheinlichkeit, (b) Sampling des Korpus,
# (c) exakte Soll-Verteilung des nächsten Wortes für jedes Präfix.


class S(tuple):
    def __new__(cls, *parts):
        return super().__new__(cls, parts)


class C(tuple):
    def __new__(cls, *opts):
        return super().__new__(cls, opts)


LANDS = [("frankreich", "paris"), ("deutschland", "berlin"), ("italien", "rom"), ("spanien", "madrid"), ("österreich", "wien")]


def capital_tail(city: str):
    return C(
        (48, f"{city} ."),
        (12, "schön ."),
        (8, "sehr schön ."),
        (8, "sehr groß ."),
        (8, "eine alte stadt ."),
        (6, "eine schöne stadt ."),
        (5, f"{city} und sehr schön ."),
        (5, "schön und alt ."),
    )


OBJ = C((5, "den zins"), (3, "den leitzins"), (2, "die zinsen"))
PLACE = C((4, "im garten"), (3, S("auf dem", C((3, "sofa"), (2, "bett")))), (3, "im haus"))


def animal_vp(barks: bool):
    opts = [
        (3, S("spielt", PLACE)),
        (3, S("schläft", PLACE)),
        (2, S("läuft", C((1, "schnell"), (1, S("schnell", PLACE))))),
    ]
    if barks:
        opts.append((2, S("bellt", C((1, "laut"), (1, "leise")))))
    return C(*opts)


GRAMMAR = C(
    (
        32,
        C(
            (60, S("die hauptstadt von", C(*[(1, S(land, "ist", capital_tail(city))) for land, city in LANDS]))),
            (20, S("die größte stadt von", C(*[(1, S(land, "ist", C((70, f"{city} ."), (30, "sehr groß .")))) for land, city in LANDS]))),
            (20, C(*[(1, S(city, "ist", C((6, f"die hauptstadt von {land} ."), (2, "sehr schön ."), (2, "eine alte stadt .")))) for land, city in LANDS])),
        ),
    ),
    (
        26,
        S(
            C((5, "die bank"), (3, "die zentralbank"), (2, "die notenbank")),
            C(
                (
                    45,
                    S(
                        "erhöht",
                        OBJ,
                        C(
                            (55, S("weil", C((5, S("die inflation", C((6, "steigt"), (2, "weiter steigt"), (2, "hoch ist")))), (3, "die preise steigen"), (2, "die wirtschaft stark ist")))),
                            (20, "heute"),
                            (15, "erneut"),
                            (10, S("denn", C((6, "die inflation steigt"), (4, "die preise steigen")))),
                        ),
                    ),
                ),
                (
                    35,
                    S(
                        "senkt",
                        OBJ,
                        C(
                            (55, S("weil", C((5, S("die wirtschaft", C((7, "schwach ist"), (3, "kaum wächst")))), (3, "die inflation sinkt"), (2, "die preise sinken")))),
                            (20, "heute"),
                            (15, "erneut"),
                            (10, S("denn", C((6, "die wirtschaft schwach ist"), (4, "die inflation sinkt")))),
                        ),
                    ),
                ),
                (20, S("hält", OBJ, C((50, "stabil"), (30, "heute stabil"), (20, S("weil die inflation", C((1, "sinkt"), (1, "steigt"))))))),
            ),
            ".",
        ),
    ),
    (
        20,
        C(
            (
                35,
                S(
                    C((6, "die aktie"), (4, "der kurs")),
                    C((4, "steigt"), (4, "fällt"), (2, "schwankt")),
                    C((4, S("heute", C((4, "stark"), (3, "leicht"), (3, "")))), (3, "stark"), (3, "leicht"), (1, "kaum")),
                    ".",
                ),
            ),
            (35, S("die kurse", C((1, "steigen"), (1, "fallen")), C((4, S("heute", C((1, "stark"), (1, "leicht"), (1, "")))), (3, "stark"), (3, "leicht")), ".")),
            (
                30,
                S(
                    "wenn der zins",
                    C(
                        (1, S("steigt ,", C((6, "sinken"), (4, "fallen")), "die", C((6, "kurse"), (4, "preise")))),
                        (1, S("sinkt , steigen die", C((6, "kurse"), (4, "preise")))),
                    ),
                    ".",
                ),
            ),
        ),
    ),
    (
        22,
        C(
            (
                35,
                S(
                    "es war einmal",
                    C(
                        (3, S("ein", C((4, "kleiner"), (3, "alter"), (3, "kluger")), C((6, "hund"), (4, "fuchs")))),
                        (2, S("eine", C((4, "kleine"), (3, "alte"), (3, "kluge")), C((5, "katze"), (5, "maus")))),
                    ),
                    ".",
                ),
            ),
            (35, S("der", C((6, S("hund", animal_vp(True))), (4, S("fuchs", animal_vp(False)))), ".")),
            (30, S("die", C((5, "katze"), (5, "maus")), animal_vp(False), ".")),
        ),
    ),
)


def enumerate_grammar(node) -> dict[tuple, float]:
    if isinstance(node, str):
        return {tuple(words(node)): 1.0}
    if isinstance(node, S):
        acc = {(): 1.0}
        for part in node:
            sub = enumerate_grammar(part)
            nxt: dict[tuple, float] = {}
            for a, pa in acc.items():
                for b, pb in sub.items():
                    nxt[a + b] = nxt.get(a + b, 0.0) + pa * pb
            acc = nxt
        return acc
    if isinstance(node, C):
        total = sum(w for w, _ in node)
        out: dict[tuple, float] = {}
        for w, child in node:
            for k, p in enumerate_grammar(child).items():
                out[k] = out.get(k, 0.0) + p * w / total
        return out
    raise TypeError(type(node))


def target_next(dist: dict[tuple, float], prefix: tuple) -> dict[str, float]:
    """Exakte Soll-Verteilung des nächsten Wortes nach `prefix` (nach Satzende: <eos>)."""
    n = len(prefix)
    out: dict[str, float] = {}
    for sent, p in dist.items():
        if sent[:n] == prefix:
            nxt = sent[n] if n < len(sent) else "<eos>"
            out[nxt] = out.get(nxt, 0.0) + p
    z = sum(out.values())
    if z == 0:
        raise ValueError(f"Präfix kommt in der Grammatik nicht vor: {' '.join(prefix)}")
    return {k: v / z for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


def empirical_next(sentences: list[str], prefix: tuple) -> dict[str, float]:
    """Empirische bedingte Verteilung im (alten) Korpus – Soll-Ersatz für das Vorher-Modell."""
    n = len(prefix)
    out: dict[str, float] = {}
    for s in sentences:
        ws = tuple(words(s))
        if ws[:n] == prefix:
            nxt = ws[n] if n < len(ws) else "<eos>"
            out[nxt] = out.get(nxt, 0.0) + 1
    z = sum(out.values()) or 1
    return {k: v / z for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


def sample_corpus(dist: dict[tuple, float], n: int, rng: random.Random) -> list[str]:
    sents = list(dist)
    weights = [dist[s] for s in sents]
    return [" ".join(s) for s in rng.choices(sents, weights, k=n)]


# Feste Prüfsätze: jedes Wort wird nach seinem Präfix vorhergesagt (wie in P1b gemessen)
EVAL_SENTENCES = [
    "die hauptstadt von frankreich ist paris .",
    "die hauptstadt von deutschland ist sehr groß .",
    "die bank erhöht den zins weil die inflation steigt .",
    "die zentralbank senkt den zins weil die wirtschaft schwach ist .",
    "wenn der zins steigt , sinken die kurse .",
    "die aktie steigt heute stark .",
    "es war einmal ein kleiner hund .",
    "der hund spielt im garten .",
    "die katze schläft auf dem sofa .",
    "rom ist die hauptstadt von italien .",
]
GOALS = {"below80_share": 0.5, "entropy_bits": 1.0}
LEIT_PREFIX = "die hauptstadt von frankreich ist"


def eval_positions():
    for s in EVAL_SENTENCES:
        ws = words(s)
        for k in range(len(ws)):
            yield s, tuple(ws[:k]), ws[k]


def dist_stats(d: dict[str, float]) -> tuple[float, float]:
    import math

    top = max(d.values())
    h = -sum(p * math.log2(p) for p in d.values() if p > 0)
    return top, h


def summarize(rows: list[dict], key: str) -> dict:
    n = len(rows)
    below = sum(1 for r in rows if r[key]["top1"] < 0.8)
    return {
        "positions": n,
        "below80": below,
        "below80_share": round(below / n, 4),
        "entropy_bits": round(sum(r[key]["entropy_bits"] for r in rows) / n, 4),
    }


TOY_PROMPTS = ["die hauptstadt von frankreich ist", "die bank erhöht den zins weil", "es war einmal"]


def model_next_probs(model, stoi, vocab, prefix: tuple):
    import torch

    ids = [stoi["<bos>"]] + [stoi.get(w, stoi["<unk>"]) for w in prefix]
    with torch.no_grad():
        p = torch.softmax(model(input_ids=torch.tensor([ids])).logits[0, -1], -1)
    return {vocab[i]: float(p[i]) for i in range(len(vocab))}


def evaluate(model, stoi, vocab, targets: dict) -> tuple[list[dict], float]:
    import math

    rows, kl_sum = [], 0.0
    for sent, prefix, actual in eval_positions():
        soll = targets[prefix]
        ist = model_next_probs(model, stoi, vocab, prefix)
        kl = sum(p * math.log2(p / max(ist.get(w, 0.0), 1e-12)) for w, p in soll.items() if p > 0)
        kl_sum += kl
        st, sh = dist_stats(soll)
        it, ih = dist_stats(ist)
        rows.append(
            {
                "sentence": sent,
                "prefix": " ".join(prefix),
                "actual": actual,
                "kl_bits": round(kl, 5),
                "soll": {"top1": round(st, 4), "entropy_bits": round(sh, 4), "top5": [[w, round(p, 4)] for w, p in list(soll.items())[:5]]},
                "ist": {
                    "top1": round(it, 4),
                    "entropy_bits": round(ih, 4),
                    "top5": [[w, round(p, 4)] for w, p in sorted(ist.items(), key=lambda kv: -kv[1])[:5]],
                },
            }
        )
    return rows, kl_sum / len(rows)


def load_existing_model(outdir: pathlib.Path):
    """Lädt das bisher eingecheckte Toy-Modell (für den Vorher-Vergleich)."""
    import torch
    from safetensors.torch import load_file
    from transformers import LlamaConfig, LlamaForCausalLM

    conf = json.loads((outdir / "config.json").read_text())
    vocab = json.loads((outdir / "vocab.json").read_text())
    keep = {k: v for k, v in conf.items() if k not in ("rope", "final_loss", "corpus_version", "training")}
    model = LlamaForCausalLM(LlamaConfig(**keep))
    model.load_state_dict(load_file(str(outdir / "model.safetensors")), strict=False)
    model.tie_weights()
    model.eval()
    return model, conf, vocab


def train_toy(train, val, vocab, rope: bool, outdir: pathlib.Path, seed: int = 0):
    import copy

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
        pad = stoi["<pad>"]

        def encode(sentences):
            return [[stoi["<bos>"]] + [stoi[w] for w in words(s)] + [stoi["<eos>"]] for s in sentences]

        def batchify(seqs):
            L = max(len(b) for b in seqs)
            x = torch.full((len(seqs), L), pad)
            y = torch.full((len(seqs), L), -100)
            for i, b in enumerate(seqs):
                x[i, : len(b)] = torch.tensor(b)
                y[i, : len(b)] = torch.tensor(b)
            return x, y

        tr, va = encode(train), encode(val)
        vx, vy = batchify(va)
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
            attention_dropout=0.1,
            tie_word_embeddings=True,
            bos_token_id=stoi["<bos>"],
            eos_token_id=stoi["<eos>"],
            pad_token_id=pad,
        )
        model = LlamaForCausalLM(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
        max_steps, bs, every, patience = 3000, 64, 100, 5
        best = (float("inf"), 0, None)
        bad = 0
        history = []
        rng = random.Random(seed)
        for step in range(1, max_steps + 1):
            model.train()
            x, y = batchify(rng.sample(tr, bs))
            loss = model(input_ids=x, labels=y).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % every == 0:
                model.eval()
                with torch.no_grad():
                    vloss = float(model(input_ids=vx, labels=vy).loss)
                history.append({"step": step, "train": round(float(loss), 4), "val": round(vloss, 4)})
                if vloss < best[0] - 1e-4:
                    best = (vloss, step, copy.deepcopy(model.state_dict()))
                    bad = 0
                else:
                    bad += 1
                print(f"[toy{'' if rope else '-nopos'}] step {step} train {float(loss):.4f} val {vloss:.4f}")
                if bad >= patience:
                    break
        model.load_state_dict(best[2])
        model.eval()
        outdir.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
        save_file(state, str(outdir / "model.safetensors"))
        conf = cfg.to_dict()
        conf["rope"] = rope
        conf["corpus_version"] = 2
        conf["training"] = {"best_step": best[1], "best_val_loss": round(best[0], 4), "stopped_at": history[-1]["step"], "history": history}
        (outdir / "config.json").write_text(json.dumps(conf, indent=2) + "\n")
        (outdir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1) + "\n")
        ref = {"prompts": []}
        with torch.no_grad():
            for p in TOY_PROMPTS:
                ids = [stoi["<bos>"]] + [stoi[w] for w in words(p)]
                logits = model(input_ids=torch.tensor([ids])).logits[0]
                ref["prompts"].append({"text": p, "ids": ids, "logits": [[round(float(v), 5) for v in row] for row in logits]})
        return model, stoi, ref, conf["training"]
    finally:
        llama_mod.apply_rotary_pos_emb = original


def build_toy() -> None:
    dist = enumerate_grammar(GRAMMAR)
    targets = {prefix: target_next(dist, prefix) for _, prefix, _ in eval_positions()}

    # 1) Erfüllt schon die Soll-Verteilung die Hilfs-Zielwerte? Sonst Korpus anpassen, nicht Training verschlechtern.
    soll_rows = [{"soll": dict(zip(("top1", "entropy_bits"), dist_stats(targets[p])))} for _, p, _ in eval_positions()]
    soll_summary = summarize(soll_rows, "soll")
    print(f"[toy] Grammatik: {len(dist)} verschiedene Sätze; Soll: {soll_summary}")
    if soll_summary["below80_share"] < GOALS["below80_share"] or soll_summary["entropy_bits"] < GOALS["entropy_bits"]:
        raise SystemExit("[toy] Die Soll-Verteilung erfüllt die Zielwerte nicht – Grammatik anpassen.")

    rng = random.Random(42)
    sentences = sample_corpus(dist, 12000, rng)
    n_val = len(sentences) // 10
    val, train = sentences[:n_val], sentences[n_val:]
    counts: dict[str, int] = {}
    for s in dist:
        for w in s:
            counts[w] = counts.get(w, 0) + 1
    sample_counts: dict[str, int] = {}
    for s in sentences:
        for w in words(s):
            sample_counts[w] = sample_counts.get(w, 0) + 1
    vocab = SPECIALS + sorted(counts, key=lambda w: (-sample_counts.get(w, 0), w))
    print(f"[toy] Korpus: {len(train)} Training + {len(val)} Validierung, Vokabular {len(vocab)}")

    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "toy-eval.json").write_text(
        json.dumps(
            {
                "sentences": EVAL_SENTENCES,
                "goals": GOALS,
                "positions": [{"prefix": " ".join(p), "soll": targets[p]} for _, p, _ in eval_positions()],
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n"
    )

    info = {
        "corpus_version": 2,
        "grammar_sentences": len(dist),
        "sentences": len(sentences),
        "vocab_size": len(vocab),
        "examples": sentences[:20],
        "word_counts": dict(sorted(sample_counts.items(), key=lambda kv: -kv[1])),
    }
    for rope, name in [(True, "toy"), (False, "toy-nopos")]:
        outdir = PUBLIC / "data" / name
        # Vorher: altes Modell gegen die empirische Verteilung seines eigenen (alten) Korpus
        before = None
        old_eval = outdir / "eval.json"
        if (outdir / "config.json").exists():
            old_conf = json.loads((outdir / "config.json").read_text())
            if old_conf.get("corpus_version", 1) == 1:
                old_model, _, old_vocab = load_existing_model(outdir)
                old_sents = corpus_v1(8000, random.Random(42))
                old_targets = {p: empirical_next(old_sents, p) for _, p, _ in eval_positions()}
                rows, kl = evaluate(old_model, {w: i for i, w in enumerate(old_vocab)}, old_vocab, old_targets)
                before = {
                    "note": "Modell v1 gegen die empirische Verteilung seines eigenen Korpus (v1 war keine Grammatik)",
                    "model": summarize(rows, "ist"),
                    "soll": summarize(rows, "soll"),
                    "kl_bits": round(kl, 4),
                    "leitbeispiel": next(r for r in rows if r["prefix"] == LEIT_PREFIX),
                }
            elif old_eval.exists():
                before = json.loads(old_eval.read_text()).get("before")
        model, stoi, ref, training = train_toy(train, val, vocab, rope, outdir)
        rows, kl = evaluate(model, stoi, vocab, targets)
        after = {"model": summarize(rows, "ist"), "soll": summarize(rows, "soll"), "kl_bits": round(kl, 4)}
        leit = next(r for r in rows if r["prefix"] == LEIT_PREFIX)
        report = {
            "variant": name,
            "goals": GOALS,
            "before": before,
            "after": after,
            "leitbeispiel": leit,
            "training": {k: v for k, v in training.items() if k != "history"},
            "label_smoothing": 0,
            "positions": rows,
        }
        (outdir / "eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
        (outdir / "corpus-info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n")
        (FIXTURES / f"{name}-reference.json").write_text(json.dumps(ref) + "\n")
        print(f"[toy{'' if rope else '-nopos'}] vorher: {before and before['model']} KL {before and before['kl_bits']}")
        print(f"[toy{'' if rope else '-nopos'}] nachher: {after}")
        print(f"[toy{'' if rope else '-nopos'}] Leitbeispiel Soll {leit['soll']['top5']} · Ist {leit['ist']['top5']}")


# ─────────────────────────────── Golden-Referenz ──────────────────────────

GOLDEN_MODELS = {
    "base": {"repo": "HuggingFaceTB/SmolLM2-135M", "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"},
    "instruct": {"repo": "HuggingFaceTB/SmolLM2-135M-Instruct", "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac"},
}
GOLDEN_LONG = (
    "In the early morning the harbour was quiet. Fishermen prepared their boats, gulls circled above the water, "
    "and the smell of coffee drifted from the small cafe near the pier. Anna walked along the old stone wall, "
    "counting the ships that had arrived during the night. She wrote each name into her notebook, because her "
    "grandfather had taught her that every ship has a story worth remembering. When the sun finally rose over the hills,"
)
GOLDEN_BASE = [
    ("capital", "The capital of France is"),
    ("centralbank", "The central bank raised interest rates because"),
    ("story", "Once upon a time, there was a"),
    ("stocks", "The stock market fell sharply after"),
    ("german", "Die Hauptstadt von Deutschland ist"),
    ("numbers", "1, 2, 3, 4, 5,"),
    ("code", "def fibonacci(n):\n    "),
    ("coffee", "To make a good cup of coffee, you need"),
    ("umlaut", "Unbegreiflichkeit ist ein langes Wort. Größe, Übermut und Äpfel:"),
    ("long", GOLDEN_LONG),
]
GOLDEN_CHATS = [
    ("chat-capital", [{"role": "user", "content": "What is the capital of France?"}]),
    ("chat-rates", [{"role": "user", "content": "Why do central banks raise interest rates?"}]),
    ("chat-haiku", [{"role": "user", "content": "Write a haiku about the sea."}]),
    ("chat-translate", [{"role": "user", "content": "Translate 'good morning' into German."}]),
    ("chat-math", [{"role": "user", "content": "What is 12 times 7?"}]),
    ("chat-tips", [{"role": "user", "content": "Give me three tips for learning a language."}]),
    ("chat-pirate", [{"role": "system", "content": "You are a pirate."}, {"role": "user", "content": "Hello!"}]),
    (
        "chat-memory",
        [
            {"role": "user", "content": "My name is Anna."},
            {"role": "assistant", "content": "Nice to meet you, Anna!"},
            {"role": "user", "content": "What is my name?"},
        ],
    ),
    ("chat-photo", [{"role": "user", "content": "Explain photosynthesis in one sentence."}]),
    ("chat-german", [{"role": "user", "content": "Wie geht es dir?"}]),
]
GOLDEN_GREEDY = 16


def build_golden() -> None:
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    out = FIXTURES / "golden"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.f32"):
        old.unlink()
    result = {"models": GOLDEN_MODELS, "greedy_steps": GOLDEN_GREEDY, "dtype": "float32", "prompts": []}
    for key, m in GOLDEN_MODELS.items():
        path = snapshot_download(m["repo"], revision=m["revision"], allow_patterns=["*.json", "*.safetensors", "*.txt"])
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
        items = GOLDEN_BASE if key == "base" else GOLDEN_CHATS
        for pid, payload in items:
            entry = {"id": pid, "model": key}
            if key == "base":
                ids = tok(payload, add_special_tokens=False)["input_ids"]
                entry["text"] = payload
            else:
                rendered = tok.apply_chat_template(payload, tokenize=False, add_generation_prompt=True)
                ids = tok(rendered, add_special_tokens=False)["input_ids"]
                entry["messages"] = payload
                entry["rendered"] = rendered
            logits = model(torch.tensor([ids])).logits[0, -1].float()
            logits.numpy().astype("<f4").tofile(out / f"{pid}.f32")
            seq = list(ids)
            greedy = []
            for _ in range(GOLDEN_GREEDY):
                nxt = int(model(torch.tensor([seq])).logits[0, -1].argmax())
                greedy.append(nxt)
                seq.append(nxt)
            probs = torch.softmax(logits, -1)
            p, idx = torch.topk(probs, 10)
            entry.update(
                {
                    "ids": ids,
                    "greedy_ids": greedy,
                    "greedy_text": tok.decode(greedy),
                    "top": [{"id": int(i), "tok": tok.decode([int(i)]), "p": float(pp)} for pp, i in zip(p, idx)],
                }
            )
            result["prompts"].append(entry)
            print(f"[golden] {pid}: {len(ids)} Tokens → {tok.decode(greedy)!r}")
    (out / "golden.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    size = sum(f.stat().st_size for f in out.iterdir())
    print(f"[golden] {len(result['prompts'])} Prompts, {size / 1e6:.1f} MB in {out.relative_to(ROOT)}")


KNOWN = {"tokenizers": build_tokenizers, "toy": build_toy, "golden": build_golden, "embeddings": None}


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
