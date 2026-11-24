# Changelog

## P2a: Echtes Modell im Browser (in Arbeit)

- **SmolLM2-135M (Base + Instruct) rechnet im Browser**, in einem Web Worker (Comlink). Laden auf Klick mit Fortschritt und Abbrechen, Ablage im Cache Storage (zweiter Besuch ohne Download), Prüfsummen je Shard
- **Shards beim Build:** `scripts/prepare-models.ts` lädt die gepinnten Revisionen im CI, bereitet die Gewichte mit derselben Funktion auf wie die Engine und schreibt Shards ≤ 32 MB mit Manifest; Fassungen: int8 (Base, Instruct) und fp16 (nur Base). Cache in `ci.yml` und `deploy.yml`, Größenprüfung < 800 MB
- **Engine:** Batch-Prefill (bitgleich mit Token für Token, im Benchmark ≈ 2× schneller), KV-Cache als eigenes Objekt, Kürzen auf den gemeinsamen Anfang (Verlängerung, Abzweigung, Verkürzung), Sitzungen pro Widget mit LRU, Stopp-Tokens
- **Kapitel 7, 8, Mini-Playground:** Umschalter Spielzeug / Echt · Base / Echt · Instruct, Stempel folgt dem Modell, freie Texteingabe mit BPE-Chips, „Live nachrechnen“ für das Leitbeispiel, Chat-Modus mit Kürzung auf das Prompt-Limit
- „Neueste Anfrage gewinnt“: überholte Antworten werden verworfen, Generierungen bei neuer Eingabe abgebrochen
- Höchstens ein echtes Modell im Worker; Download- und RAM-Angabe an den Ladeknöpfen; `#/diagnose` zeigt das geladene Modell, den Speicher und einen Batch-Benchmark
- **Golden-Tests** gegen PyTorch (10 Base-Prompts, 10 Chats, volle Logits) mit den Toleranzen aus dem Spike, in PR-Check und Deploy
- Playwright prüft im CI: Laden mit Fortschritt, Cache beim zweiten Besuch, blockierten Download, Instruct-Stopp, schnelle Eingabefolgen
- Credits: SmolLM2-Gewichte (Apache-2.0 mit Änderungshinweis), Comlink

## Toy-Modell v2 (2026-11-24)

- Korpus als gewichtete Grammatik mit Weggabelungen an vielen Stellen; Training per Sampling, Early Stopping auf Validierungsdaten, kein Label Smoothing
- Exakte Soll-Verteilung je Präfix; Auswertung in `eval.json` (Soll vs. Modell, KL) und `docs/toy-model.md`
- Top-1 < 80 % jetzt an 63 % statt 38 % der Prüfpositionen, mittlere Entropie 1,05 statt 0,64 bit, KL(Soll ‖ Modell) 0,025 bit
- Kapitel 7: Abschnitt „Warum ist das Spielzeugmodell an manchen Stellen so sicher?“ mit Werten aus den Daten statt fester Zahlen

## P1b: MVP (2026-11-23)

- **Kapitel 0 (Karte):** klickbare Pipeline mit Fortschrittsmarken und Datenfluss-Puls
- **Kapitel 2 (Tokenisierung):** echte Tokenizer von GPT-2 und SmolLM2 im Browser (eigene BPE-Implementierung, identisch zur Python-Referenz), Deutsch vs. Englisch, BPE live, „strawberry“
- **Kapitel 7 (Next-Token & Sampling):** echte Verteilung des Leitbeispiels (fp16, vorberechnet) und interaktive Verteilung am Spielzeugmodell mit Temperatur, Top-k, Top-p, Min-p, Seed, Würfeln und 1000× würfeln
- **Kapitel 8 (Generierungs-Loop):** Schritt-Modus, Token erzwingen, Antwort-Baum mit Ast-Wahrscheinlichkeiten, KV-Cache-Grafik
- **Mini-Playground** (`#/playground`) mit Wortliste und Autovervollständigung
- **Spielzeugmodelle:** zwei winzige Llama-Modelle (mit RoPE / ohne Positionssignal), trainiert per `assets.yml`, gegen PyTorch getestet
- Interaktionsprüfung in der CI (Stempel, Regler, Seed, Erzwingen, Navigation ohne Netz)

## P1a: Engine-Spike (2026-11-23)

- Eigene Inferenz-Engine in TypeScript (`src/engine`): Llama-Architektur mit RMSNorm, RoPE, Grouped-Query-Attention, SwiGLU, KV-Cache; safetensors-Leser; int8-Quantisierung (pro Zeile und in Gruppen)
- Test gegen eine unabhängige NumPy-Referenz an einem Mini-Modell (`test/engine.test.ts`)
- Spike-Workflow `spike.yml`: Vergleich mit PyTorch am echten SmolLM2-135M (Base + Instruct), Zeitmessung, Leitbeispiel-Kandidaten
- Nachtrag: Vergleich von int8 pro Zeile und in Gruppen (128/64/32/16), fp16-Skalen, Ablation je Modellteil, Pages-Budget
- Ergebnis (`docs/spike.md`): fp16 exakt; Hauptquelle des int8-Fehlers sind die Attention-Projektionen → Standard wird die „int8-Fassung“ (Attention fp16, Rest Gruppen-int8 zu 32): ≈ 160 MB, max. 1,3 pp Abweichung; fp16-Fassung nur für das Base-Modell (Pages-Budget 611 MB)
- Leitbeispiel festgelegt: „The capital of France is“
- Neuer Stempel „Echt · int8“ (zweigeteilt, Unterscheidung über die Form)

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
