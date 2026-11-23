# Changelog

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
