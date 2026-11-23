#!/usr/bin/env python3
"""Spike P1a, Schritt 3: Vergleich Engine ↔ PyTorch und Bericht.

Liest spike-out/reference.json, engine.json und die .f32-Logits, berechnet je Prompt und
Präzision: Top-10 identisch, Top-1 gleich, relativer L2-Fehler, Korrelation, max. Abweichung,
Übereinstimmung der Greedy-Fortsetzung. Schreibt spike-out/results.json und spike-report.md
(und hängt den Bericht an die Job-Zusammenfassung von GitHub Actions an).
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

# Startkriterium aus PLAN.md (P1a); endgültige Toleranzen legt docs/spike.md fest
CRIT = {"top10": True, "rel_l2": 1e-2, "corr": 0.999}


def load(name):
    return np.fromfile(OUT / name, dtype="<f4").astype(np.float64)


rows = []
for run in eng["runs"]:
    for ep in run["prompts"]:
        rp = next(p for p in ref["prompts"] if p["id"] == ep["id"])
        r = load(f"ref-{rp['id']}.f32")
        t = load(f"ts-{run['precision']}-{rp['id']}.f32")
        top_r = list(np.argsort(-r)[:10])
        top_t = list(np.argsort(-t)[:10])
        greedy_match = 0
        for a, b in zip(ep["greedy_ids"], rp["greedy_ids"]):
            if a != b:
                break
            greedy_match += 1
        row = {
            "model": run["model"],
            "precision": run["precision"],
            "prompt": rp["id"],
            "kind": rp["kind"],
            "top1": bool(top_r[0] == top_t[0]),
            "top10_identical": bool(top_r == top_t),
            "top10_overlap": len(set(top_r) & set(top_t)),
            "rel_l2": float(np.linalg.norm(t - r) / np.linalg.norm(r)),
            "corr": float(np.corrcoef(t, r)[0, 1]),
            "max_abs": float(np.max(np.abs(t - r))),
            "greedy_match": greedy_match,
            "greedy_len": len(rp["greedy_ids"]),
            "engine_greedy_text": toks[run["model"]].decode(ep["greedy_ids"]),
            "prefillMs": ep["prefillMs"],
            "decodeMsPerToken": ep["decodeMsPerToken"],
        }
        row["pass_start_criterion"] = bool(
            row["top10_identical"] and row["rel_l2"] < CRIT["rel_l2"] and row["corr"] > CRIT["corr"]
        )
        rows.append(row)

(OUT / "results.json").write_text(json.dumps({"criterion": CRIT, "rows": rows, "engine": eng}, indent=2) + "\n")

L = []
L.append("# Spike P1a – Ergebnisse (automatisch erzeugt)\n")
for k, m in ref["models"].items():
    L.append(f"- **{k}:** `{m['repo']}` @ `{m['sha']}`")
L.append(f"- Node {eng['node']} ({eng['cpu']}), ein Thread\n")

L.append("## Korrektheit: Engine vs. PyTorch (fp32-Referenz)\n")
L.append("| Modell | Präzision | Prompt | Top-1 | Top-10 identisch | Überlappung | rel. L2 | Korrelation | max. Abw. | Greedy gleich | Start-Kriterium |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    L.append(
        f"| {r['model']} | {r['precision']} | {r['prompt']} | {'✓' if r['top1'] else '✗'} | "
        f"{'✓' if r['top10_identical'] else '✗'} | {r['top10_overlap']}/10 | {r['rel_l2']:.2e} | {r['corr']:.6f} | "
        f"{r['max_abs']:.3e} | {r['greedy_match']}/{r['greedy_len']} | {'✓' if r['pass_start_criterion'] else '✗'} |"
    )

L.append("\n## Zusammenfassung je Präzision (schlechtester Wert über alle Prompts)\n")
L.append("| Präzision | Top-1 alle | Top-10 identisch | min. Überlappung | max. rel. L2 | min. Korrelation | min. Greedy gleich |")
L.append("|---|---|---|---|---|---|---|")
for prec in dict.fromkeys(r["precision"] for r in rows):
    rs = [r for r in rows if r["precision"] == prec]
    L.append(
        f"| {prec} | {sum(r['top1'] for r in rs)}/{len(rs)} | {sum(r['top10_identical'] for r in rs)}/{len(rs)} | "
        f"{min(r['top10_overlap'] for r in rs)}/10 | {max(r['rel_l2'] for r in rs):.2e} | {min(r['corr'] for r in rs):.6f} | "
        f"{min(r['greedy_match'] for r in rs)}/{rs[0]['greedy_len']} |"
    )

L.append("\n## Leistung (Base-Modell)\n")
L.append("| Präzision | Laden + Aufbereiten | RAM Gewichte | Download ≈ | Prefill 16 / 64 / 128 Tokens | Decode | RSS |")
L.append("|---|---|---|---|---|---|---|")
for run in eng["runs"]:
    if run["model"] != "base" or "perf" not in run:
        continue
    pf = run["perf"]["prefillMs"]
    L.append(
        f"| {run['precision']} | {run['loadMs']/1000:.1f} s | {run['weightMB']} MB | {run['downloadMB']} MB | "
        f"{pf['16']/1000:.1f} / {pf['64']/1000:.1f} / {pf['128']/1000:.1f} s | {run['perf']['decodeMsPerToken']} ms/Token | "
        f"{run['memory']['rssMB']} MB |"
    )

L.append("\n## Leitbeispiel-Kandidaten (PyTorch, fp32)\n")
for p in ref["prompts"]:
    L.append(f"### `{p['id']}` ({p['model']}, {p['kind']})\n")
    L.append(f"Prompt: `{p['text']}` · {len(p['ids'])} Tokens\n")
    L.append("| # | Token | p | Logit |")
    L.append("|---|---|---|---|")
    for i, t in enumerate(p["top"], 1):
        shown = t["tok"].replace("\n", "↵").replace("|", "\\|")
        L.append(f"| {i} | `{shown}` | {t['p']*100:.1f} % | {t['logit']:.2f} |")
    L.append(f"\nGreedy-Fortsetzung: `{p['greedy_text'].strip()}`\n")

md = "\n".join(L) + "\n"
(OUT / "spike-report.md").write_text(md)
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(md)
print(md)
