# Inside GenAI

[PAGE](https://schlonhorizon.github.io/Inside-GenAI/)

Ein interaktives Lernkompendium, das Schritt für Schritt zeigt, wie Sprachmodelle (LLMs) aus einem Prompt eine Antwort machen: Tokenisierung, Embeddings, Attention, Sampling, Training, Assistant-Tuning und mehr. Die Seite ist rein statisch und läuft komplett im Browser.

- **Plan & Konzept:** [PLAN.md](PLAN.md)
- **Aktueller Stand:** P2a: Kapitel 0, 2, 7 und 8 sind fertig, dazu der Mini-Playground. Das echte Modell SmolLM2-135M (Base und Instruct) lässt sich per Klick laden und rechnet im Browser; ohne Laden rechnet ein Spielzeugmodell. Änderungen siehe [CHANGELOG.md](CHANGELOG.md)

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

Das echte Modell wird nur im CI aus den Hugging-Face-Dateien aufbereitet (`npm run prepare-models`, braucht Zugang zu huggingface.co). Ohne diese Dateien läuft die Seite lokal mit dem Spielzeugmodell. Um Laden, Worker und Oberfläche trotzdem lokal zu testen, gibt es ein kleines Zufallsmodell mit dem echten Vokabular:

```bash
npm run prepare-models:fake
```

```bash
npm run build && npx vite preview
```

```bash
npm run check:ui -- http://localhost:4173/ --models
```

## Lizenz

Code unter [MIT](LICENSE). Schriften (SIL OFL 1.1), Bibliotheken, Modelle und Daten stehen unter ihren eigenen Lizenzen (siehe PLAN.md, Abschnitt 8.3).

_Privates Lernprojekt, nicht mit Anthropic verbunden._
