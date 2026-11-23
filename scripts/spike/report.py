#!/usr/bin/env python3
"""Spike P1a, Schritt 3: Vergleich Engine ↔ PyTorch und Bericht.

Liest spike-out/reference.json, engine.json und die .f32-Logits. Je Prompt und Gewichtsformat:
Top-1/Top-10, KL-Divergenz, größte Abweichung einer Wahrscheinlichkeit (Δp, mit Token),
relativer L2-Fehler, Korrelation, Übereinstimmung der Greedy-Fortsetzung.
Dazu: Ablation (welcher Teil des Modells verursacht den int8-Fehler?), vorab festgelegte
Entscheidungsregel, Pages-Budget. Schreibt spike-out/results.json und spike-report.md.

Umgebungsvariable DIST_MB: Größe der gebauten Seite ohne Modelle (aus dem Workflow).
"""
import json
import os
import pathlib

import numpy as np
from transformers import AutoTokenizer

OUT = pathlib.Path("spike-out")
ref = json.loads((OUT / "reference.json").read_text())
eng = json.loads((OUT / "engine.json").read_text())
toks = {k: AutoTokenizer.from_pretrained(v["path"]) for k, v in ref["models"].items()}
PARTS = ["attn", "mlp", "down", "emb"]
PART_LABEL = {"attn": "Attention (q/k/v/o)", "mlp": "MLP gate/up", "down": "MLP down", "emb": "Embedding = LM-Head"}
TOLERANCE_PP = 0.5  # Entscheidungsregel: höchstens 0,5 pp schlechter als die genaueste int8-Variante
PAGES_LIMIT_MB = 800


def load(name):
    return np.fromfile(OUT / name, dtype="<f4").astype(np.float64)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def parse(spec):
    body, *flags = spec.split("+")
    if ":" not in body:
        return {p: body for p in PARTS}, "s16" in flags
    d = {p: "f32" for p in PARTS}
    for kv in body.split(","):
        k, v = kv.split(":")
        d[k] = v
    return d, "s16" in flags


def describe(spec):
    parts, s16 = parse(spec)
    vals = set(parts.values())
    suffix = "" if s16 or not any(v.startswith("q8") for v in vals) else " (Skalen f32)"
    if len(vals) == 1:
        return {"f16": "fp16 (exakt)", "f32": "fp32"}.get(spec, spec.replace("+s16", "")) + suffix, "main"
    q = [p for p in PARTS if parts[p].startswith("q8")]
    fmt = parts[q[0]] if q else "f16"
    if len(q) == 1:
        return f"nur {PART_LABEL[q[0]]} {fmt}, Rest fp16", "only"
    keep = [p for p in PARTS if not parts[p].startswith("q8")]
    return f"{fmt}, aber {PART_LABEL[keep[0]]} fp16", "mixed"


rows = []
for run in eng["runs"]:
    for ep in run["prompts"]:
        rp = next(p for p in ref["prompts"] if p["id"] == ep["id"])
        r = load(f"ref-{rp['id']}.f32")
        t = load(f"ts-{run['precision']}-{rp['id']}.f32")
        pr, pt = softmax(r), softmax(t)
        top_r, top_t = list(np.argsort(-r)[:10]), list(np.argsort(-t)[:10])
        worst = int(np.argmax(np.abs(pr - pt)))
        gm = 0
        for a, b in zip(ep["greedy_ids"], rp["greedy_ids"]):
            if a != b:
                break
            gm += 1
        rows.append(
            {
                "model": run["model"],
                "precision": run["precision"],
                "prompt": rp["id"],
                "top1": bool(top_r[0] == top_t[0]),
                "top10_identical": bool(top_r == top_t),
                "top10_overlap": len(set(top_r) & set(top_t)),
                "kl": float(np.sum(pr * (np.log(pr + 1e-30) - np.log(pt + 1e-30)))),
                "max_dp": float(abs(pr[worst] - pt[worst])),
                "worst_token": toks[run["model"]].decode([worst]),
                "worst_p_ref": float(pr[worst]),
                "worst_p_eng": float(pt[worst]),
                "rel_l2": float(np.linalg.norm(t - r) / np.linalg.norm(r)),
                "corr": float(np.corrcoef(t, r)[0, 1]),
                "greedy_match": gm,
                "greedy_len": len(rp["greedy_ids"]),
                "top5": [{"id": int(i), "p_ref": float(pr[i]), "p_eng": float(pt[i])} for i in top_r[:5]],
            }
        )

runs_by = {(r["model"], r["precision"]): r for r in eng["runs"]}
variants = list(dict.fromkeys(r["precision"] for r in rows))


def agg(spec):
    rs = [r for r in rows if r["precision"] == spec]
    w = max(rs, key=lambda r: r["max_dp"])
    base = runs_by.get(("base", spec), {})
    label, kind = describe(spec)
    return {
        "spec": spec,
        "label": label,
        "kind": kind,
        "top1": f"{sum(r['top1'] for r in rs)}/{len(rs)}",
        "top10": f"{sum(r['top10_identical'] for r in rs)}/{len(rs)}",
        "min_overlap": min(r["top10_overlap"] for r in rs),
        "max_kl": max(r["kl"] for r in rs),
        "max_dp": w["max_dp"],
        "worst": f"{w['model']}/{w['prompt']}: `{w['worst_token'].strip() or repr(w['worst_token'])}` {w['worst_p_ref']*100:.1f} % → {w['worst_p_eng']*100:.1f} %",
        "min_corr": min(r["corr"] for r in rs),
        "min_greedy": min(r["greedy_match"] for r in rs),
        "download_mb": base.get("downloadMB"),
        "ram_mb": base.get("weightMB"),
        "decode_ms": base.get("perf", {}).get("decodeMsPerToken"),
        "instruct_download_mb": runs_by.get(("instruct", spec), {}).get("downloadMB"),
    }


A = {v: agg(v) for v in variants}

# Entscheidungsregel (vorab festgelegt): Kandidaten = int8-Hauptvarianten mit fp16-Skalen + gemischte Varianten
cands = [a for a in A.values() if a["spec"] != "f16" and a["spec"].endswith("+s16") and a["kind"] in ("main", "mixed")]
best_dp = min(a["max_dp"] for a in cands)
within = [a for a in cands if a["max_dp"] <= best_dp + TOLERANCE_PP / 100]
# Größe auf 2 MB gerundet vergleichen; bei Gleichstand die schnellere (gemessene) Variante
winner = sorted(within, key=lambda a: (round(a["download_mb"] / 2), a["decode_ms"] if a["decode_ms"] is not None else 1e9))[0]

# Pages-Budget
dist_mb = float(os.environ.get("DIST_MB", "0") or 0)
f16_base = A["f16"]["download_mb"]
f16_inst = A["f16"]["instruct_download_mb"] or 0
q_base, q_inst = winner["download_mb"], winner["instruct_download_mb"] or 0
scen_a = dist_mb + q_base + q_inst + f16_base + f16_inst
scen_b = dist_mb + q_base + q_inst + f16_base
fp16_scope = "Base und Instruct" if scen_a <= PAGES_LIMIT_MB else "nur Base"

results = {"rows": rows, "aggregates": A, "winner": winner, "best_dp": best_dp,
           "budget": {"dist_mb": dist_mb, "scenario_a": scen_a, "scenario_b": scen_b, "fp16_scope": fp16_scope}, "engine": eng}
(OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

L = ["# Spike P1a – Ergebnisse (automatisch erzeugt)\n"]
for k, m in ref["models"].items():
    L.append(f"- **{k}:** `{m['repo']}` @ `{m['sha']}`")
L.append(f"- Node {eng['node']} ({eng['cpu']}), ein Thread. Referenz: PyTorch fp32.")
L.append("- Rechengenauigkeit der Engine: Akkumulation in JS-`number` (f64), Aktivierungen/Norms in f32/f64, Softmax in f64; int8 symmetrisch ±127, Rundung `Math.round`.\n")


def table(items, title):
    L.append(f"## {title}\n")
    L.append("| Variante | Spezifikation | Download Base | RAM | Decode | Top-1 | Top-10 ident. | min. Überl. | max. KL | max. Δp | schlechtestes Δp bei | min. Greedy |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in items:
        mark = " **← gewählt**" if a is winner else ""
        L.append(
            f"| {a['label']}{mark} | `{a['spec']}` | {a['download_mb']} MB | {a['ram_mb']} MB | "
            f"{str(a['decode_ms']) + ' ms' if a['decode_ms'] else '–'} | {a['top1']} | {a['top10']} | {a['min_overlap']}/10 | "
            f"{a['max_kl']:.2e} | {a['max_dp']*100:.2f} pp | {a['worst']} | {a['min_greedy']}/16 |"
        )
    L.append("")


table([a for a in A.values() if a["kind"] == "main"], "Hauptvergleich: Gewichtsformate")
table([a for a in A.values() if a["kind"] == "only"], "Ablation A: nur ein Teil int8, Rest fp16")
table([a for a in A.values() if a["kind"] == "mixed"], "Ablation B: alles int8 außer einem Teil (gemischte Varianten)")

L.append("## Entscheidung (Regel vorab festgelegt)\n")
L.append(f"Kandidaten: int8-Varianten mit fp16-Skalen und gemischte Varianten. Genaueste: max. Δp = {best_dp*100:.2f} pp. "
         f"Zulässig sind Varianten bis {best_dp*100 + TOLERANCE_PP:.2f} pp; davon die kleinste (bei Gleichstand die schnellere).\n")
L.append(f"**Gewählt: {winner['label']}** (`{winner['spec']}`): max. Δp {winner['max_dp']*100:.2f} pp, KL {winner['max_kl']:.2e}, "
         f"Download Base {winner['download_mb']} MB / Instruct {winner['instruct_download_mb']} MB.\n")

L.append("## Pages-Budget\n")
L.append(f"- Seite ohne Modelle (gemessen, inkl. 30-MB-Test-Shard): {dist_mb:.0f} MB")
L.append(f"- (a) int8 Base + int8 Instruct + fp16 Base + fp16 Instruct: **{scen_a:.0f} MB**")
L.append(f"- (b) int8 Base + int8 Instruct + fp16 nur Base: **{scen_b:.0f} MB**")
L.append(f"- Regel: über {PAGES_LIMIT_MB} MB → fp16 nur für Base. **Ergebnis: fp16-Fassung für {fp16_scope}.**\n")

L.append("## Wahrscheinlichkeiten im Vergleich (Top-5)\n")
show = ["f16", "q8+s16", winner["spec"]]
show = list(dict.fromkeys(show))
for pid in dict.fromkeys(r["prompt"] for r in rows):
    sub = {r["precision"]: r for r in rows if r["prompt"] == pid}
    base = sub["f16"]
    tz = toks[base["model"]]
    L.append(f"**{pid}**\n")
    L.append("| Token | PyTorch | " + " | ".join(A[s]["label"] for s in show) + " |")
    L.append("|---|---|" + "---|" * len(show))
    for i, t in enumerate(base["top5"]):
        name = tz.decode([t["id"]]).replace("\n", "↵").replace("|", "\\|")
        L.append(f"| `{name}` | {t['p_ref']*100:.1f} % | " + " | ".join(f"{sub[s]['top5'][i]['p_eng']*100:.1f} %" for s in show) + " |")
    L.append("")

L.append("## Leistung (Base-Modell)\n")
L.append("| Variante | Aufbereiten | RAM | Download | Prefill 16 / 64 / 128 | Decode |")
L.append("|---|---|---|---|---|---|")
for run in eng["runs"]:
    if run["model"] == "base" and "perf" in run:
        pf = run["perf"]["prefillMs"]
        L.append(f"| {A[run['precision']]['label']} | {run['loadMs']/1000:.1f} s | {run['weightMB']} MB | {run['downloadMB']} MB | "
                 f"{pf['16']/1000:.1f} / {pf['64']/1000:.1f} / {pf['128']/1000:.1f} s | {run['perf']['decodeMsPerToken']} ms/Token |")

L.append("\n## Leitbeispiel-Kandidaten (PyTorch, fp32)\n")
for p in ref["prompts"]:
    L.append(f"### `{p['id']}` ({p['model']}, {p['kind']}): `{p['text']}`\n")
    L.append(" · ".join(f"`{t['tok']}` {t['p']*100:.1f} %" for t in p["top"][:6]).replace("\n", "↵"))
    L.append(f"\nGreedy: `{p['greedy_text'].strip()}`\n".replace("\n<", " <"))

md = "\n".join(L) + "\n"
(OUT / "spike-report.md").write_text(md)
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(md)
print(md)
