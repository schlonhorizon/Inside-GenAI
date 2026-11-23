# Inside GenAI

[PAGE](https://schlonhorizon.github.io/Inside-GenAI/)

Ein interaktives Lernkompendium, das Schritt für Schritt zeigt, wie Sprachmodelle (LLMs) aus einem Prompt eine Antwort machen: Tokenisierung, Embeddings, Attention, Sampling, Training, Assistant-Tuning und mehr. Die Seite ist rein statisch und läuft komplett im Browser.

- **Plan & Konzept:** [PLAN.md](PLAN.md)
- **Aktueller Stand:** D0, eine Design-Probe (Style Tile), auf der Schriften und Farben ausgewählt werden

## Entwicklung

Ohne lokale Umgebung: Jeder Push auf `main` baut die Seite per GitHub Actions und deployt sie auf GitHub Pages (`.github/workflows/deploy.yml`).

Lokal (optional, Node 22):

```bash
npm install
npm run dev
```

## Lizenz

Code unter [MIT](LICENSE). Schriften (SIL OFL 1.1), Bibliotheken, Modelle und Daten stehen unter ihren eigenen Lizenzen (siehe PLAN.md, Abschnitt 8.3).

_Privates Lernprojekt, nicht mit Anthropic verbunden._
