# Changelog

## P1a: Engine-Spike (2026-11-23)

- Eigene Inferenz-Engine in TypeScript (`src/engine`): Llama-Architektur mit RMSNorm, RoPE, Grouped-Query-Attention, SwiGLU, KV-Cache; safetensors-Leser; int8-Quantisierung (pro Zeile und in Gruppen)
- Test gegen eine unabhängige NumPy-Referenz an einem Mini-Modell (`test/engine.test.ts`)
- Spike-Workflow `spike.yml`: Vergleich mit PyTorch am echten SmolLM2-135M (Base + Instruct), Zeitmessung, Leitbeispiel-Kandidaten
- Ergebnis und festgelegte Toleranzen: `docs/spike.md` (fp32/fp16 exakt, int8 ≤ 3 pp Abweichung, Decode ≈ 165 ms/Token)

## P0: Fundament (2026-11-23)

- Seitengerüst: Hash-Routing (`#/kapitel/<slug>`, `#/glossar`, `#/credits`, `#/diagnose`, `#/styleguide`), Kapitelnavigation mit Fortschrittsanzeige, Mobilmenü
- Alle 16 Kapitel als MDX-Platzhalter mit Lernzielen; Formeln über KaTeX, Fachbegriffe mit Hover-Karte und Verweis auf das Glossar
- Einstellungen (Modus, Mathe-Ebene, Bewegung) und Lernfortschritt werden im Browser gespeichert
- `<Fact>` für Modellzahlen: Die Werte werden beim Build aus der Modell-Config berechnet und nie im Text fest eingetragen (Werte ab P2)
- `#/diagnose`: Download-Test (30-MB-Test-Shard), Cache API, WebGPU, Rechen-Benchmark, kopierbarer Bericht
- `#/credits` mit allen Lizenzen; Styleguide (vormals Design-Probe) nur noch mit den gewählten Schriften
- CI: Tests (Farbregeln, Lizenzen, Kapitel/Glossar, Router, Modellzahlen), Screenshots aller Routen in Hell und Dunkel als Artifact, Smoke-Test; Gerüst für den manuellen Asset-Workflow

## D0: Design-Probe (2026-11-23)

- Style Tile mit Farbprüfung (Kontrast, ΔE2000, Farbfehlsichtigkeit), Schrift-Kandidaten, Line-Art-Pipeline, Reglern und Stempeln
- Entscheidung: Fraunces / Inter / JetBrains Mono, Dunkelmodus minimal kälter (`docs/design-decisions.md`)
- Deploy auf GitHub Pages bei jedem Push auf `main`
