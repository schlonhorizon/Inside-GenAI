# Spielzeugmodell: Korpus, Training, Auswertung

Stand: 2026-11-24 · Erzeugt von `scripts/py/build_assets.py` (Ziel `toy`) über `.github/workflows/assets.yml`
Daten: `public/data/toy/` und `public/data/toy-nopos/` (je `eval.json` mit allen Prüfpositionen)

## Warum eine Nachbesserung?

Die erste Fassung (P1b) war an vielen Stellen fast sicher. Dadurch wirkten Temperatur, Top-p und der Antwort-Baum in
Kapitel 7/8 kaum. Ursache war **das Korpus, nicht das Training**: Wenige Satzvorlagen, Weggabelungen fast nur nach „ist“.
Ein Modell, das solche Daten gut lernt, *muss* sicher sein.

## Korpus v2: gewichtete Grammatik

Das Korpus ist synthetisch und als **endliche, gewichtete Grammatik** beschrieben: Wortfolgen, Auswahlen mit Gewichten,
verschachtelt. Daraus folgt dreierlei:

1. **Exakte Satzverteilung:** Alle 372 möglichen Sätze werden mit ihrer Wahrscheinlichkeit aufgezählt (Produkt der Gewichte).
2. **Trainingskorpus per Sampling:** 12 000 Sätze werden aus dieser Verteilung gezogen (Seed 42), 10 % davon als Validierung zurückgehalten. Die Häufigkeiten im Korpus entsprechen damit den Gewichten.
3. **Soll-Verteilung:** Für jedes Präfix ist die wahre Verteilung des nächsten Wortes exakt bekannt. Dazu werden die Wahrscheinlichkeiten aller Sätze mit diesem Präfix aufsummiert, gruppiert nach dem nächsten Wort.

Weggabelungen gibt es jetzt an vielen Stellen: Subjekt (bank/zentralbank/notenbank), Verb (erhöht/senkt/hält),
Objekt, Satzende, Adverbien, Tiere, Orte. Das Leitbeispiel bleibt erhalten: Nach „die hauptstadt von frankreich ist“
folgt mit 53 % „paris“, daneben „schön“, „sehr …“ und „eine … stadt“. Das Vokabular umfasst 94 Wörter inklusive
`<unk>`, `<bos>`, `<eos>`, `<pad>`.

## Training

- Winziges Llama-Modell (2 Layer, d = 32, 2 Heads), gebundene Embeddings; einmal mit RoPE, einmal ohne Positionssignal.
- AdamW, Lernrate 3·10⁻³, Batch 64, `attention_dropout` 0,1.
- **Early Stopping** auf dem Validierungs-Loss: Auswertung alle 100 Schritte, Geduld 5, bestes Modell behalten.
- **Kein Label Smoothing.** Das Training wird nicht künstlich verschlechtert, um weichere Verteilungen zu erzwingen.

## Auswertung

10 feste Prüfsätze, jedes Wort wird nach seinem Präfix vorhergesagt: 78 Positionen.

**Maßstab ist die Soll-Verteilung.** Das eigentliche Ziel ist ein Modell, das die Soll-Verteilung trifft, also eine kleine
KL(Soll ‖ Modell). Die Werte „Top-1 < 80 %“ und „mittlere Entropie“ sind **Hilfsgrößen**, damit die Regler in Kapitel 7/8
sichtbar wirken. Erfüllt schon die Soll-Verteilung sie nicht, wird das **Korpus** angepasst, nicht das Training. Das
Skript prüft das vor dem Training und bricht sonst ab.

Vorab festgelegte Hilfs-Zielwerte: Top-1 < 80 % an ≥ 50 % der Positionen, mittlere Entropie ≥ 1,0 bit.

| | Top-1 < 80 % | mittlere Entropie | KL(Soll ‖ Modell) |
|---|---|---|---|
| **vorher** (v1-Modell)* | 30/78 (38 %) | 0,64 bit | 0,013 bit |
| **Soll** (Grammatik v2) | 48/78 (62 %) | 1,03 bit | – |
| **nachher** `toy` | **49/78 (63 %)** | **1,05 bit** | **0,025 bit** |
| **nachher** `toy-nopos` | 49/78 (63 %) | 1,06 bit | 0,024 bit |

\* Das v1-Korpus war keine Grammatik. Als Soll-Ersatz dient deshalb die empirische bedingte Verteilung des alten
Korpus. Die kleine KL zeigt: **Das alte Modell hatte seine Daten gut gelernt.** Die Übersicherheit kam von den
gleichförmigen Daten, genau deshalb wurde das Korpus geändert und nicht das Training.

**Leitbeispiel** „die hauptstadt von frankreich ist“:

| | paris | schön | sehr | eine |
|---|---|---|---|---|
| Soll | 53 % | 17 % | 16 % | 14 % |
| `toy` | 58 % | 16 % | 11 % | 14 % |
| `toy-nopos` | 52 % | 17 % | 12 % | 17 % |

**Nebenbefund für Kapitel 4:** Die Variante ohne Positionssignal ist kaum schlechter. Die kausale Maske verrät dem Modell
bereits einiges über die Reihenfolge, weil jedes Token nur die Tokens davor sieht. Für die Positions-Demo in Kapitel 4
braucht es deshalb Beispiele, bei denen die Wortreihenfolge *innerhalb* des sichtbaren Kontexts entscheidet.

## Absicherung

- `test/toy.test.ts`: beide Varianten Position für Position gegen die PyTorch-Logits.
- `test/toy.test.ts` (Regressionstest): Hilfs-Zielwerte erfüllt und KL(Soll ‖ Modell) unter dem 1,5-Fachen des
  Trainingslaufs, gerechnet mit der eigenen TS-Engine.
- `test/content.test.ts`: Kapiteltexte enthalten keine fest eingetragenen Prozentwerte des Spielzeugmodells. Werte
  kommen aus `eval.json` über `<Fact m="toy" k="…" />`.
