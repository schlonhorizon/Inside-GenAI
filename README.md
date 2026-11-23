# Inside GenAI

[PAGE](https://schlonhorizon.github.io/Inside-GenAI/)

Ein interaktives Lernkompendium, das Schritt für Schritt zeigt, wie Sprachmodelle (LLMs) aus einem Prompt eine Antwort machen: Tokenisierung, Embeddings, Attention, Sampling, Training, Assistant-Tuning und mehr. Die Seite ist rein statisch und läuft komplett im Browser.

- **Plan & Konzept:** [PLAN.md](PLAN.md)
- **Aktueller Stand:** P1b (MVP): Kapitel 0, 2, 7 und 8 sind fertig, dazu der Mini-Playground mit Spielzeugmodell. Änderungen siehe [CHANGELOG.md](CHANGELOG.md)

## Texte bearbeiten

Alle Inhalte liegen in `content/` und lassen sich direkt im GitHub-Webeditor ändern:

- `content/chapters/NN-slug.mdx`: Kapiteltexte (Markdown; Formeln mit `$…$` bzw. `$$…$$`; Fachbegriffe mit `<Term id="token">Tokens</Term>`)
- `content/glossary.yaml`: Glossar
- `content/credits.yaml`: Quellen & Lizenzen

Nach dem Commit auf `main` prüft GitHub Actions die Inhalte (Tests) und veröffentlicht die Seite neu.

## Entwicklung

Ohne lokale Umgebung: Jeder Push auf `main` baut die Seite per GitHub Actions und deployt sie auf GitHub Pages (`.github/workflows/deploy.yml`).

Lokal (optional, Node 22):

```bash
npm install
npm run dev
```

```bash
npm test
```

## Lizenz

Code unter [MIT](LICENSE). Schriften (SIL OFL 1.1), Bibliotheken, Modelle und Daten stehen unter ihren eigenen Lizenzen (siehe PLAN.md, Abschnitt 8.3).

_Privates Lernprojekt, nicht mit Anthropic verbunden._
