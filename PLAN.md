# PLAN.md: „Inside GenAI“

> Interaktives Lernkompendium: Wie macht ein LLM aus einem Prompt eine Antwort?
> Stand: 2026-11-23 · Revision 3 (Entscheidungen aus Runde 1 + 2 eingearbeitet) · Status: freigegeben, Umsetzung startet mit D0

## 1. Zusammenfassung

„Inside GenAI“ ist eine rein statische Lernseite (Vite + React + TypeScript, gehostet auf GitHub Pages). Sie führt in 16 Kapiteln vom Prompt bis zur Antwort: zuerst die Grundlagen neuronaler Netze, dann Tokenisierung, Embeddings, Attention, Transformer-Layer, Sampling, Generierung, Training, Assistant-Tuning, Kontext, In-Context Learning, Grenzen sowie Reasoning, Tools und RAG. Herzstück ist eine **eigene, gläserne Inferenz-Engine in TypeScript**, die mit demselben Code zwei Modelle rechnet: ein **winziges, selbst trainiertes Spielzeug-Transformer-Modell** (deutsches Mini-Vokabular, jede Zahl sichtbar, immer verfügbar) und das **echte kleine Open-Source-LLM SmolLM2-135M** in Base- *und* Instruct-Variante. Weil der Forward-Pass selbst ausgeführt wird, sind echte Attention-Gewichte, der Residual Stream, die „Logit Lens“ und alle Wahrscheinlichkeitsverteilungen sichtbar und manipulierbar. Die Modellgewichte werden im CI nach gepinnter Revision geladen, quantisiert, gecacht und **same-origin über GitHub Pages** ausgeliefert. Das funktioniert auch in Firmennetzen, die huggingface.co blockieren. Ohne echtes Modell bleibt die Seite vollständig nutzbar. Jede Visualisierung trägt einen Stempel: **ECHT**, **ECHT · TOY**, **VORBERECHNET** oder **ILLUSTRATIV**. Die Optik ist warm und ruhig (Off-White bzw. warmes Grau, Serif-Headlines, feine Line Art). Farbe wird nur mit didaktischer Bedeutung eingesetzt. Vor dem Bau gibt es zwei Absicherungen: eine **Design-Probe** (Style Tile) und einen **Engine-Spike**, der Korrektheit und Geschwindigkeit des echten Modells misst.

## 2. Didaktisches Konzept

### 2.1 Lernpfad (16 Kapitel)
Der Weg folgt der Datenflussrichtung, danach kommt der Blick zurück auf das Training und nach außen:

```
 [0 Karte] → 1 Neuronale Netze (Grundlagen)
          → 2 Tokens → 3 Embeddings → 4 Position → 5 Attention → 6 Block/Layer
          → 7 Next-Token → 8 Loop                     (Inferenz: „Was passiert bei einer Anfrage?“)
          → 9 Training → 10 Base→Assistant              (Herkunft: „Woher kommen die Gewichte?“)
          → 11 Kontext → 12 In-Context Learning → 13 Grenzen → 14 Reasoning/Tools/RAG
                                                        (Praxis: „Was heißt das für mich?“)
          → 15 Playground „Full Trace“                  (Synthese: alles zusammen, A/B-Vergleich)
```

Die Kapitelnummern sind nur Anzeige. Routen nutzen stabile Slugs (`#/kapitel/attention`), daher bleiben Links bei späterem Umsortieren gültig.

**Roter Faden:** Ein Leitbeispiel taucht in jedem Kapitel wieder auf. Kandidaten für das echte Modell sind `"The central bank raised interest rates because"`, `"The capital of France is"` und 1–2 weitere; für das Toy-Modell z. B. `"die bank erhöht den zins weil"`. **Das Leitbeispiel wird erst nach dem Engine-Spike festgelegt** (P1a). Dort wird geprüft, welcher Prompt beim echten Modell eine aussagekräftige Verteilung liefert.

### 2.2 Struktur: Kapitel-Unterseiten mit linearem Pfad *(entschieden)*
Jedes Kapitel ist eine eigene Hash-Route und intern eine kurze **Scroll-Story**. Die Karte (Kapitel 0) dient als Hub; Prev/Next, eine Fortschrittsleiste und ein „Weiter“-Block am Kapitelende erzeugen das Gefühl eines durchgehenden Pfads. Vorteile: Code-Splitting pro Kapitel, keine Konflikte zwischen Scroll-Animationen und Widgets, stabile Deep-Links, messbarer Fortschritt.

### 2.3 Intuition → Tiefe: drei Ebenen pro Kapitel
1. **Intuition** (immer sichtbar): Alltagsanalogie plus kleine Line-Art-Illustration bzw. Animation, 2–3 Minuten.
2. **Ausprobieren** (immer sichtbar): das zentrale Widget mit einer konkreten Aufgabe („Ändere die Temperatur so, dass das Top-Token unter 50 % fällt“).
3. **Unter der Haube** (aufklappbar, plus „Mathe-Ebene“-Toggle): Formeln (KaTeX), Tensor-Shapes als Bemaßungen, echte Zahlen aus dem Toy-Modell, Verweise auf Papers.

Dazu kommen pro Kapitel: **„Was du jetzt weißt“** (3 Bullets), ein **Quiz** (3–5 Fragen inkl. Vorhersage-Fragen: erst tippen, dann rechnet das Modell nach) und ein **Stellschrauben-Kasten**, der festhält, welcher Hebel hier wirkt und ob er zur Trainings- oder zur Inferenzzeit greift.

### 2.4 Ehrlichkeits-Stempel (verbindlich für jede Abbildung)
| Stempel | Bedeutung |
|---|---|
| `ECHT` | live im Browser mit dem echten LLM berechnet (SmolLM2) |
| `ECHT · TOY` | live und exakt mit dem Spielzeug-Transformer bzw. Mini-Netz berechnet |
| `VORBERECHNET` | echte Werte, aber vorab berechnet (z. B. Embedding-Projektion, Referenz-Logits) |
| `ILLUSTRATIV` | vereinfacht bzw. geskriptet, zeigt das Prinzip und keine echten Werte |

Aussagen zu Claude oder ChatGPT bekommen zusätzlich die Marker **„öffentlich dokumentiert (Quelle, Datum)“** oder **„allgemeines LLM-Wissen / Annahme“**.

## 3. Kapitel für Kapitel

Aufwand: **S** ≈ 0,5–1 Tag · **M** ≈ 2–3 Tage · **L** ≈ 4+ Tage (Claude-Sessions, grobe Schätzung).

### Kapitel 0: Die große Karte · **M**
- **Lernziel:** Gesamtbild in 60 Sekunden, Orientierung für alles Weitere.
- **Interaktion:** Ein Prompt läuft als animierter Puls durch das Schema. Hover auf einen Block zeigt eine Kurzbeschreibung samt Tensor-Shape, Klick öffnet das Kapitel. Blöcke zeigen den Fortschritt (unbesucht / gelesen / Quiz ✓).
- **Visualisierung:**
```
 "Die Bank erhöht…"  ┌─────────┐   ┌──────────┐   ┌──────────────── × N ─┐   ┌────────┐   ┌─────────┐
  ────────────────▶ │TOKENIZER│──▶│EMBEDDING │──▶│ ATTENTION ─▶ MLP     │──▶│ LOGITS │──▶│SAMPLING │──▶ "weil"
                    └─────────┘   │+POSITION │   │  └─ residual stream ─┘│   │SOFTMAX │   └────┬────┘
                     [2]          └──────────┘   └──────────────────────┘   └────────┘        │
                                   [3][4]          [5][6]   (Bausteine: [1])  [7]        [8] ◀─┘ (Loop)
        ┄┄┄┄ Training [9] / Fine-Tuning [10] formen die Gewichte (gestrichelt, von unten) ┄┄┄┄
```
- **Echt vs. simuliert:** `ILLUSTRATIV`. Optional zeigt der Modus „mit echtem Modell“ die echten Shapes und Tokens.

### Kapitel 1: Grundlagen: Was ist ein neuronales Netz? · **S–M** *(neu)*
- **Lernziel:** Das Handwerkszeug für alle folgenden Kapitel. Ein Neuron ist eine gewichtete Summe plus Aktivierungsfunktion. Eine Matrixmultiplikation sind „viele Neuronen gleichzeitig“. Parameter sind die Gewichte. Das Skalarprodukt dient als Ähnlichkeitsmaß (wird für Attention gebraucht).
- **Interaktionen:**
  - **Ein Neuron:** 3 Eingänge mit Reglern für Gewichte und Bias; die Rechnung `w₁x₁ + w₂x₂ + w₃x₃ + b` steht live daneben. Die Aktivierungsfunktion ist umschaltbar (ReLU / GELU / Sigmoid), der Ausgang wird als Punkt auf der Funktionskurve gezeigt.
  - **Eine Schicht = Matrix:** 3 Neuronen nebeneinander werden animiert zu `W · x + b` „zusammengeklappt“. Jede Zeile von W gehört zu einem Neuron (Hover verbindet Zeile und Neuron).
  - **Mini-Netz entscheidet:** 2 Eingänge → 4 versteckte Neuronen → 1 Ausgang trennt zwei Punktwolken. Die Entscheidungsfläche ist eingefärbt, Gewichte lassen sich von Hand verstellen oder mit „Trainieren“ automatisch finden (Vorgriff auf Kapitel 9).
  - **Skalarprodukt als Ähnlichkeit:** zwei Pfeile drehen, der Wert von `a·b` und `cos θ` ändern sich live.
  - **Parameterzähler:** „Dieses Netz: `{toy.params}` Parameter. SmolLM2-135M: `{smollm2.params}`. Große Modelle: Hunderte Milliarden (Schätzungen, nicht offiziell).“ Die Zahlen werden aus der jeweiligen Modell-Config berechnet (siehe 5.5), nicht im Text hart codiert.
- **Visualisierung:**
```
  x₁ ─(w₁=0.8)─╮                         W (3×3)        x      b       h
  x₂ ─(w₂=-0.3)┼─▶ Σ + b ─▶ ReLU ─▶ y   ┌ .8 -.3 .5 ┐ ┌x₁┐  ┌b₁┐   ┌h₁┐
  x₃ ─(w₃=0.5)─╯   └ 0.71 ┘    0.71      │ .1  .9 -.2│·│x₂│ +│b₂│ = │h₂│  ← je Zeile ein Neuron
                                         └-.4  .2  .7┘ └x₃┘  └b₃┘   └h₃┘
```
- **Echt:** `ECHT · TOY` (exakte Rechnung, echtes Mini-Training), Analogien `ILLUSTRATIV`.

### Kapitel 2: Tokenisierung · **M**
- **Lernziel:** Modelle sehen keine Buchstaben oder Wörter, sondern Token-IDs. BPE entsteht aus Häufigkeiten. Daraus folgen Kosten, Sprachunterschiede und die Schwäche beim Buchstabenzählen.
- **Interaktionen:**
  - Freitext → Token-Chips mit ID, Token-Zählung, Zeichen/Token-Verhältnis.
  - **Tokenizer-Vergleich** nebeneinander (GPT-2, SmolLM2, optional ein multilingualer Tokenizer). Hinweis: Claudes Tokenizer ist nicht öffentlich, Token-Zählung gibt es nur über die API.
  - **Deutsch vs. Englisch:** derselbe Satz in beiden Sprachen, Token-Anzahl als Bemaßungslinie.
  - **BPE live:** Schritt für Schritt die häufigsten Paare mergen, an einem Mini-Korpus oder am eigenen Text.
  - **„Wie viele r in strawberry?“** zeigt, dass das Modell `str|aw|berry` sieht, nicht 10 Buchstaben.
- **Visualisierung:**
```
  Text:   U n b e g r e i f l i c h k e i t
  Tokens: [Un][beg][re][if][lich][keit]      ├──── 6 Tokens ────┤
  IDs:     3118 1350 260 361 2742 9354        Englisch: [Incomprehensibility] → 4
```
- **Echt:** `ECHT`. Eigene TS-BPE-Implementierung mit den echten `tokenizer.json`-Dateien (je ~2–3 MB), **auch ohne Modell-Download** verfügbar. Der BPE-Lernmodus ist `ECHT · TOY`.

### Kapitel 3: Embeddings · **M**
- **Lernziel:** Token-ID → Vektor. Ähnliche Bedeutung heißt nahe Vektoren, Richtungen tragen Bedeutung. Dazu der Unterschied zwischen statischem Embedding und kontextuellem Hidden State.
- **Interaktionen:**
  - 2D-Karte eines Bedeutungsraums (ca. 2–5k Wörter), zoom- und suchbar, mit Clustern (Finanzen, Tiere, Städte …).
  - Ein Wort anklicken zeigt die nächsten Nachbarn mit Cosinus-Ähnlichkeit (Rückbezug auf das Skalarprodukt aus Kapitel 1).
  - **Vektorarithmetik:** `König − Mann + Frau = ?`, als Pfeile in 2D animiert, das Ergebnis aber aus dem vollen Raum.
  - „Lookup-Tabelle“-Animation: Die Token-ID wählt eine Zeile der Embedding-Matrix (Toy, alle Zahlen sichtbar).
  - Mit echtem Modell: nächste Nachbarn in der SmolLM2-Embedding-Matrix.
- **Visualisierung:**
```
        ·Paris  ·Berlin            ↑ Dim 2 (PCA)
     ·Rom   ·Madrid     ·Aktie  ·Anleihe
                           ·Dividende
   König ──────▶ Königin         (gleicher Pfeil wie)
   Mann  ──────▶ Frau
```
- **Echt:** `VORBERECHNET` (deutsche fastText-Vektoren, Teilmenge, PCA/UMAP; Hinweis „statische Wortvektoren ≠ LLM-Interna“) · Lookup `ECHT · TOY` · Nachbarn im echten Modell `ECHT`.

### Kapitel 4: Position · **S–M**
- **Lernziel:** Attention ist von sich aus reihenfolgeblind („Hund beißt Mann“ = „Mann beißt Hund“). Darum braucht das Modell Positionsinformation: absolut gelernt (GPT-2), sinusförmig (Original-Transformer) oder rotiert (RoPE, z. B. SmolLM2/Llama).
- **Interaktionen:** Wörter per Drag umsortieren. Ohne Positionssignal bleibt die Toy-Ausgabe gleich, mit Positionssignal ändert sie sich (echtes Ablationsexperiment). Dazu das Sinus-Muster als „Strichcode“ pro Position und RoPE als rotierender Zeiger; das Skalarprodukt zweier Zeiger hängt nur vom Abstand ab.
- **Echt:** Toy-Experiment `ECHT · TOY`, Sinus/RoPE-Formeln exakt, Story `ILLUSTRATIV`.

### Kapitel 5: Attention · **L**
- **Lernziel:** Q/K/V intuitiv und mathematisch verstehen. Jedes Token sammelt gewichtete Information von früheren Tokens (Causal Mask). Verschiedene Heads haben verschiedene „Aufgaben“.
- **Interaktionen:**
  - **Rechenweg am Toy-Modell:** Token anklicken, dann Schritt für Schritt q·kᵀ → /√d → Maske → Softmax → gewichtete Summe von V, jede Matrix mit Zahlen.
  - **Attention-Viewer am echten Modell:** Layer- und Head-Auswahl, Linien- und Heatmap-Ansicht. Klick auf ein Token zeigt, worauf es achtet.
  - **Head-Galerie:** Mini-Heatmaps aller Heads eines Layers mit typischen Mustern („vorheriges Token“, „erstes Token / Attention Sink“, „Induction Head“).
  - **Induction-Head-Demo:** Zufällige Tokenfolge zweimal wiederholen; im zweiten Durchlauf springt die Attention auf „das Token nach dem letzten Vorkommen“. Wird in Kapitel 12 (In-Context Learning) wieder aufgegriffen.
  - **Head ausschalten (Ablation):** live sehen, wie sich die Next-Token-Verteilung ändert.
- **Visualisierung:**
```
   Query: "weil"   Layer 7 · Head 3                       Heatmap (kausal)
   die ────────── 0.02                                    die  ■
   bank ━━━━━━━━━ 0.41   ◀── Bemaßung: Gewicht           bank ■ ■
   erhöht ━━━━━── 0.18                                    erh. ▪ ■ ■
   den ────────── 0.04                                    den  · ▪ ■ ■
   zins ━━━━━━━━━ 0.35                                    zins ▪ ■ ▪ · ■
```
- **Echt:** Rechenweg `ECHT · TOY`, Viewer/Galerie/Ablation `ECHT`, Analogie `ILLUSTRATIV`.

### Kapitel 6: Der Transformer-Block & Residual Stream · **L**
- **Lernziel:** Ein Block besteht aus Norm → Attention → (+) → Norm → MLP → (+). Der Residual Stream ist ein „Förderband“, auf das jede Schicht etwas addiert. N Blöcke werden gestapelt, und die Vorhersage wird Layer für Layer schärfer.
- **Interaktionen:** Ein animierter Block-Schaltplan mit Puls zeigt bei Hover die Shapes als Bemaßungen (`[T × 576]`). Die **Logit Lens** (echt) zeigt pro Layer den Top-5-Tipp fürs nächste Token. Dazu die Norm des Residual Streams pro Layer, der Beitrag von Attention vs. MLP und die MLP-Intuition als „Key-Value-Speicher“ mit echten Neuronen-Aktivierungen (Rückbezug auf Kapitel 1).
- **Visualisierung:**
```
  x ═══╦═══════════════╦═══════════════╦═══▶  (residual stream, × 30 Layer)
       ║   ┌──────┐    ║    ┌─────┐    ║
       ╚══▶│NORM→ │═(+)╚═══▶│NORM→│═(+)
           │ATTN  │         │ MLP │
           └──────┘         └─────┘
  Logit Lens:  L1 "the"  L8 "of"  L15 "the"  L22 "inflation"  L30 "inflation" 34 %
```
- **Echt:** `ECHT` (Logit Lens, Normen, Aktivierungen), Schaltplan `ILLUSTRATIV` mit echten Shapes.

### Kapitel 7: Next-Token-Prediction & Sampling · **M** (Kernkapitel)
- **Lernziel:** Das Modell gibt **eine Verteilung** aus, keine Antwort. Logits → Softmax → Wahrscheinlichkeiten → Sampling. Jede Sampling-Stellschraube wird verstanden.
- **Interaktionen:**
  - Balkendiagramm der Top-20-Tokens (Logit / Wahrscheinlichkeit umschaltbar), live animiert.
  - Regler **Temperatur** (0–2), **Top-k**, **Top-p**, optional **Min-p**, Repetition Penalty, **Seed**. Abgeschnittene Tokens werden schraffiert, die Formel aktualisiert sich live (Mathe-Ebene).
  - **Würfeln:** Die Zufallszahl fällt auf kumulierte Balken. **1000× würfeln:** Histogramm der Samples gegen die theoretische Verteilung.
  - Temperatur 0 vs. 1 vs. 1.5 auf denselben Prompt, nebeneinander.
  - **Mini-Playground (P1):** freier Prompt, alle Sampling-Regler, Generierung mehrerer Tokens. Er sitzt als Abschluss-Widget von Kapitel 7/8 und ist zusätzlich als eigene Route `#/playground` erreichbar (in P5 zum vollen Playground ausgebaut).
- **Visualisierung:**
```
  Logits ──softmax(z/T)──▶  " rising"     ████████████████ 31.2%
                            " inflation"  ███████████ 21.7%
          Top-p = 0.9 ┄┄┄┄  " the"        ██████ 11.0%
                            " of"         ███ 5.4%      ├── Σ = 0.90 ──┤
                            ░░░░ (abgeschnitten)
  T [━━━━●━━━━] 0.8   top-k [━━━━━●━] 40   top-p [━━━━━━●━] 0.90   seed 42
```
- **Echt:** `ECHT` mit echtem Modell, sonst `ECHT · TOY`. Die Sampling-Mathematik ist immer exakt.

### Kapitel 8: Der Generierungs-Loop & Antwort-Baum · **M–L**
- **Lernziel:** Die Generierung ist autoregressiv: Token anhängen, neu vorhersagen. Frühe Entscheidungen bestimmen den Rest. Der KV-Cache erklärt die Geschwindigkeit.
- **Interaktionen:** Play / Pause / Step / Zurück, pro Schritt eine Mini-Verteilung unter jedem erzeugten Token. **Token erzwingen** erzeugt einen neuen Ast im **Antwort-Baum** (D3-Tree, Äste mit kumulierter Wahrscheinlichkeit beschriftet). Dazu Stopp-Kriterien (EOS, max. Länge) und eine KV-Cache-Animation.
- **Visualisierung:**
```
  The central bank raised interest rates because ─┬─ " inflation"(0.31) ─ " was" ─ " rising" ─ …
                                                  ├─ " of"(0.18) ─ " the" ─ " economy" ─ …   ◀ erzwungen
                                                  └─ " it"(0.07) ─ …
```
- **Echt:** `ECHT` bzw. `ECHT · TOY`, KV-Cache-Animation `ILLUSTRATIV`.

### Kapitel 9: Training · **L**
- **Lernziel:** Training heißt „Next-Token-Vorhersage üben“. Dazu gehören Loss (Cross-Entropy), Gradient Descent und Lernrate, Überanpassung und Skalierung (Daten, Parameter, Compute).
- **Interaktionen:**
  - **Gradient Descent in 2D:** Kugel auf einer Loss-Landschaft, Lernraten-Regler (zu klein / gut / zu groß → divergiert).
  - **Live-Training im Browser** auf einem wählbaren Text (Vorlagen: Goethe, Finanznachrichten, eigener Text): (1) **Bigram-Modell**, bei dem Zählmatrix und gelerntes Modell zum Gleichen konvergieren, mit Loss-Kurve live; (2) **Mini-MLP/Char-Modell** mit Kontext 3–8 Zeichen und regelmäßigen Textsamples („vom Buchstabensalat zu fast-Wörtern“).
  - Regler für Lernrate, Batchgröße, Kontextlänge, Embedding-Dim und Train/Validation-Split, damit Overfitting sichtbar wird.
  - Einordnung mit öffentlichen Zahlen (Toy vs. SmolLM2-135M vs. Frontier-Modelle; Claude-Details nicht öffentlich, so gekennzeichnet).
- **Echt:** `ECHT · TOY` (echtes Training, handgeschriebener Backprop im Web Worker), Skalierung `ILLUSTRATIV` mit Quellen.

### Kapitel 10: Vom Base-Model zum Assistenten · **M–L**
- **Lernziel:** Pretraining erzeugt einen „Text-Fortsetzer“, SFT und Präferenz-Training (RLHF, Constitutional AI) machen daraus einen Assistenten. Dazu das Chat-Template mit Special Tokens und die Rolle des System-Prompts.
- **Interaktionen:** **Base vs. Instruct echt nebeneinander** (gleicher Input, SmolLM2-135M vs. -Instruct). Das **Chat-Template** wird sichtbar: Die Konversation erscheint als Rohtext mit `<|im_start|>system … <|im_end|>`. Ein **System-Prompt-Regler** zeigt die Verteilung des ersten Antwort-Tokens bei verschiedenen System-Prompts. Eine **Präferenz-Demo** zeigt RLHF im Prinzip; Constitutional AI wird nach dem öffentlichen Anthropic-Paper erklärt.
- **Echt:** Base/Instruct, System-Prompt und Template `ECHT`; RLHF/CAI `ILLUSTRATIV`.

### Kapitel 11: Kontextfenster & „Gedächtnis“ · **M**
- **Lernziel:** Kein Gedächtnis zwischen Aufrufen. Die ganze Konversation wird bei jedem Schritt als ein Text neu eingelesen. Kontext ist endlich, Abschneiden oder Zusammenfassen hat Folgen. Lange Kontexte kosten Geld und Latenz.
- **Interaktionen:** Chat-Simulation mit wachsendem Token-Band und Füllstandsbemaßung. Ein **künstlich kleines Fenster** (z. B. 64 Tokens) zeigt mit dem echten Modell, wie Information „herausfällt“. Strategien im Vergleich (Abschneiden, Zusammenfassen, Retrieval → Kap. 14), veröffentlichte Kontextgrößen (mit Datum und Quelle), „Lost in the Middle“.
- **Echt:** Fenster-Experiment und Tokenzählung `ECHT`, Strategien `ILLUSTRATIV`.

### Kapitel 12: In-Context Learning & Prompting · **M** *(neu, eigenes Kapitel)*
- **Warum ein eigenes Kapitel?** ICL verbindet Mechanik (Induction Heads, Kap. 5) mit Praxis (Prompting) und ist der Grund, warum Prompt-Gestaltung überhaupt wirkt. Es passt nach dem Kontextfenster, denn die Beispiele *sind* Kontext, und vor die Grenzen, denn Prompting-Tricks erklären auch manche Grenze.
- **Lernziel:** Ein Modell „lernt“ aus Beispielen im Prompt, ohne dass sich Gewichte ändern. Warum das funktioniert und welche praktischen Lehren daraus folgen.
- **Interaktionen:**
  - **0 / 1 / 3 Beispiele, echt gemessen:** Aufgabe wählen (z. B. Land → Hauptstadt, Wort → Gegenteil, Satz → Sentiment-Label, oder eine **erfundene Zuordnung** wie „positiv → blau, negativ → rot“, die das Modell nicht aus dem Training kennen kann). Gezeigt wird die Wahrscheinlichkeit des Ziel-Tokens als Balken je Anzahl Beispiele.
  - **Attention-Blick:** Ein Induction- bzw. Kopier-Head achtet beim Antwort-Token auf die passenden Stellen in den Beispielen (Verknüpfung zu Kap. 5).
  - **Reihenfolge & Format:** Beispiele per Drag umsortieren oder das Format ändern (Labels, Trennzeichen) und sehen, wie stark die Wahrscheinlichkeit schwankt (Order Sensitivity, Label-Bias), echt berechnet.
  - **Prompting-Lehren** als Takeaways: klare Formate, repräsentative und ausgewogene Beispiele, Anweisung + Beispiel kombinieren, Rollen bzw. System-Prompt (Rückbezug auf Kap. 10).
- **Visualisierung:**
```
  P("Paris")   0-shot ██████ 0.21
               1-shot ████████████████ 0.58
               3-shot ███████████████████████ 0.83     (Ziel-Token-Wahrscheinlichkeit, echt)
  Prompt:  Germany → Berlin ⏎ Italy → Rome ⏎ Spain → Madrid ⏎ France → ▮
                         ╰────── Induction-Head-Attention ──────╯
```
- **Echt:** `ECHT` (Messungen am Base- und Instruct-Modell), Prompting-Lehren `ILLUSTRATIV` mit Quellen.

### Kapitel 13: Grenzen · **M**
- **Lernziel:** Halluzinationen sind plausibles Weiterschreiben ohne Faktencheck. Dazu Wissens-Cutoff, Sycophancy und Nichtdeterminismus.
- **Interaktionen:** **Gleicher Prompt, 20 Läufe** (echte Samples, gruppiert). Ein **Halluzinations-Labor** fragt nach einer erfundenen Firma, die Konfidenz-Einfärbung zeigt pro Token die Unsicherheit. Sycophancy und Cutoff werden geskriptet und mit Erklärung gezeigt.
- **Echt:** Sampling/Konfidenz `ECHT`, Sycophancy/Cutoff `ILLUSTRATIV`.

### Kapitel 14: Über das Modell hinaus: Reasoning, Tool Use, Agents, RAG · **L**
Drei vollwertige Abschnitte:

**14.1 Reasoning & Test-Time Compute** *(aufgewertet)*
- **Lernziel:** Warum „mehr Tokens = mehr Rechenschritte“ gilt: Jedes Token bekommt einen Forward-Pass fester Tiefe (z. B. 30 Layer). Schwierige Probleme brauchen mehr serielle Schritte, und Zwischenergebnisse im Kontext wirken als externer Notizblock. Unterschied zwischen **„denk Schritt für Schritt“ (Prompting)** und **trainiertem Reasoning** (Modelle, die per Reinforcement Learning auf überprüfbaren Aufgaben lernen, lange Gedankenketten zu erzeugen).
- **Interaktionen:**
  - **„Rechentiefe“-Animation:** feste Layer-Zahl pro Token vs. viele Tokens hintereinander → Rechenbudget wächst mit der Länge.
  - **Vorgegebene Gedankenkette, echt gemessen:** Bei einer Aufgabe (z. B. kleine Textaufgabe) misst das echte Modell die Wahrscheinlichkeit der richtigen Antwort direkt und nach einer *von uns vorgegebenen* Zwischenrechnung im Kontext. Ehrlich gekennzeichnet: Das 135M-Modell kann nicht selbst gut „denken“, aber man sieht, dass Zwischenschritte im Kontext die Antwort erleichtern.
  - **Thinking-Budget-Regler** (qualitativ): mehr Denk-Tokens → bessere Ergebnisse bei schweren Aufgaben, aber mehr Kosten und Latenz, nach öffentlichen Berichten.
  - **Was ist öffentlich bekannt?** Claude: „Extended Thinking“ mit einstellbarem Token-Budget, Thinking-Blöcke in der API (laut Anthropic-Dokumentation). OpenAI: Reasoning-Modelle (o-Serie / „Thinking“-Modi) mit internen Reasoning-Tokens, trainiert mit Reinforcement Learning (laut OpenAI-Veröffentlichungen). Offene Referenz: das DeepSeek-R1-Paper mit Details zu RL mit überprüfbaren Belohnungen. Alles mit Quelle und Datum, Trainingsdetails von Claude/ChatGPT als „nicht öffentlich“ markiert.
- **Echt:** Gedankenketten-Messung `ECHT`, Rechentiefe/Budget `ILLUSTRATIV`.

**14.2 Tool Use & Agents:** Agent-Loop-Animation (Modell → `tool_use`-JSON → Tool → `tool_result` zurück in den Kontext → Modell …) mit dem rohen Kontext-Text daneben. `ILLUSTRATIV`, Format nach öffentlicher Tool-Use-Doku.

**14.3 RAG:** Kleine Dokumentensammlung (z. B. fiktive Geschäftsberichte). Frage → **echtes BM25-Retrieval im Browser** mit Scores → Chunks im Prompt → Antwort (Instruct-Modell oder geskriptet). Retrieval `ECHT`.

### Kapitel 15: Playground „Full Trace“ · **L**
- **Lernziel:** Synthese. Alle Stellschrauben auf einer Seite, und du verstehst, *warum* genau dieser Output kommt.
- **Interaktionen:** Prompt (+ System-Prompt, Chat-Modus), Modellwahl (Toy / Base / Instruct), alle Sampling-Parameter. Ein kompakter **Pipeline-Trace** (Tokens → Attention-Mini-Map → Logit-Lens-Streifen → Verteilung → generierte Tokens), aufklappbar zu den Detail-Widgets. **A/B-Vergleich** mit Token-Diff, KL-Divergenz pro Schritt und markiertem Abzweig-Punkt. **„Warum dieses Token?“** per Occlusion-Attribution. Zustand im URL-Hash (teilbar, reproduzierbar inkl. Seed).
- **Echt:** `ECHT` bzw. `ECHT · TOY`.

### Querschnitt · **M**
**Glossar** (Seite + Hover-Karten über `<Term>`), **Quiz** pro Kapitel, **Fortschritt** in `localStorage`, **Mathe-Ebene-Toggle**, **Spickzettel** (alle Stellschrauben in einer Matrix: Hebel × Wirkung × Trainings-/Inferenzzeit), **Diagnose-Seite** `#/diagnose` (siehe 4.4), **Credits-Seite** `#/credits` (alle Quellen und Lizenzen, siehe 8.3).

## 4. Tech-Stack

| Bereich | Wahl | Begründung |
|---|---|---|
| Build | **Vite** | schnell, statischer Output, `base`-Pfad für Pages |
| UI | **React + TypeScript** | „Kapitel = Komponente“, großes Ökosystem |
| Routing | **Hash-Routing** (React Router `createHashRouter`) | Pages hat kein SPA-Fallback; Reload und Deep-Links funktionieren |
| Inhalte | **MDX** + `remark-math`/`rehype-katex` | Texte als Markdown, im GitHub-Webeditor bearbeitbar, Widgets einbettbar |
| Formeln | **KaTeX** (lokal) | schnell, Build-Zeit-Rendering, kein CDN |
| State | **Zustand** (+ `persist`) | schlank; Einstellungen, Fortschritt, Modellstatus |
| Animation | **Motion** (ehem. Framer Motion) + CSS | deklarativ, `useReducedMotion`, `pathLength` fürs Linienzeichnen, `useInView` |
| Datenviz | **D3-Module** (scale, shape, hierarchy, array), Rendering per React-SVG | D3 rechnet, React rendert |
| Worker | **Web Worker + Comlink** | Inferenz und Training blockieren nie die UI |
| Schriften | **@fontsource** (siehe 6.2) | lokal gebündelt, nur Open Source |
| Tests | **Vitest** (Ops, Sampler, BPE, Golden-Tests) + **Playwright** (Screenshots, Smoke-Test) | Ehrlichkeit und Optik im CI prüfbar |
| Modell-Aufbereitung | **Node/TS-Skript** (safetensors lesen → int8 → Shards) | Deploy braucht kein Python |
| Referenzdaten | **Python** (PyTorch/transformers), **nur** im manuellen Workflow | Golden-Werte, Embeddings, Toy-Training |

### 4.1 Echtes Modell im Browser *(entschieden)*
- **Eigene TS-Engine** (Llama-Architektur: RMSNorm, RoPE, GQA, SwiGLU) mit Hooks an jeder Stelle (`onAttention`, `onResidual`, `onLogits`, `patchHead`). Das Toy-Modell nutzt denselben Code.
- **SmolLM2-135M Base + Instruct** (Apache-2.0): 30 Layer, d=576, 9 Heads (3 KV-Heads), Vokabular 49.152. Diese Werte dienen nur als Planungsgrundlage. Im Seitentext werden sie aus `config.json` der gepinnten Revision gelesen (siehe 5.5).
- **int8** mit Skalierung pro Zeile, **~140 MB pro Modell** in Shards à ≤ 32 MB, Laden **nur auf Klick** mit Größenangabe und Fortschritt, danach in der **Cache Storage API**.
- Erwartung (wird im Spike P1a gemessen): ~0,1–0,5 s/Token mit KV-Cache, Prefill von 64 Tokens einige Sekunden. Prompt-Limit im UI 128–256 Tokens. WebGPU-Turbo ist optional für P6 bzw. früher, falls der Spike es nahelegt.
- Pages setzt keine COOP/COEP-Header, also kein `SharedArrayBuffer`. Die reine JS-Engine braucht das nicht.
- **Prompts am echten Modell auf Englisch**, Toy-Modell und UI auf Deutsch, mit Erklärung warum (Trainingsdaten).
- **Fallback:** Das Toy-Modell (≈100–300 KB, im Bundle) ist immer da, Widgets schalten automatisch um, ein Banner informiert. Die Tokenizer sind immer echt.

### 4.2 Spielzeugmodell
Wort-Level-Vokabular mit ~64–128 deutschen Tokens, d_model = 16, 2 Layer, 2 Heads, Kontext 16. Trainiert auf einem synthetischen Mini-Korpus mit klaren Mustern, dadurch sind die Attention-Muster interpretierbar. Training über `scripts/train_toy.py` im manuellen Workflow; das Ergebnis `toy-model.json` wird committet. Zusätzlich gibt es eine Variante ohne Positionssignal (Kap. 4).

### 4.3 Workflows & Deployment auf GitHub Pages

Drei Workflows mit klar getrennten Aufgaben. Der normale Deploy ist **schlank und Python-frei**:

```
deploy.yml   (push auf main, workflow_dispatch)                         — Node only
  1. actions/cache  key = models-<modell-revision>-<hash(prepare-models.ts)>
  2. bei Cache-Miss: scripts/prepare-models.ts → lädt gepinnte Revision vom HF Hub,
     int8-Quantisierung, Shards + manifest.json   (sonst: Shards aus dem Cache)
  3. npm ci → vitest (inkl. Golden-Tests gegen committete Referenzwerte) → vite build
     (BASE_PATH aus actions/configure-pages) → Shards + Test-Shard nach dist/
  4. actions/upload-pages-artifact → actions/deploy-pages (environment: github-pages)
  permissions: contents: read, pages: write, id-token: write · concurrency: "pages"

ci.yml       (pull_request)                                              — Node only
  1. actions/cache restore mit DEMSELBEN Key wie deploy.yml
     (bei Cache-Miss: scripts/prepare-models.ts erzeugt die Shards und speichert sie im Cache)
  2. npm ci → vitest INKL. Golden-Tests gegen die echten Shards → ein Engine-Fehler wird schon im PR rot
  3. vite build → vite preview → Playwright: Screenshots aller Kapitel in Light + Dark
     (Toy-Modus, kein Modell-Download im Browser) → actions/upload-artifact "screenshots"
     + Smoke-Test: jede Route rendert ohne Konsolenfehler

assets.yml   (nur workflow_dispatch, manuell)                            — Python + PyTorch
  scripts/golden.py            → public/data/golden/*.json   (Referenz-Logits/Attention, wenige KB)
  scripts/export_embeddings.py → public/data/embeddings-de.bin (fastText-Teilmenge: nur die ersten
                                  N häufigsten Wörter werden gestreamt, kein Komplett-Download)
  scripts/train_toy.py         → public/data/toy-model*.json
  → committet auf Branch assets/update-<datum> und öffnet einen Pull Request
```

**Base-Path:** `vite.config.ts` → `base: process.env.BASE_PATH ?? '/'`, im Workflow aus `actions/configure-pages` (`base_path`). Alle Asset-URLs laufen über `import.meta.env.BASE_URL`.

**Was du in den Repo-Settings einstellen musst:**
1. **Repo öffentlich machen** *(entschieden)*. Pages für private Repos bräuchte einen bezahlten Plan.
2. **Settings → Pages → Build and deployment → Source: „GitHub Actions“**.
3. **Settings → Actions → General → Workflow permissions:** Haken bei **„Allow GitHub Actions to create and approve pull requests“**. Den braucht `assets.yml`, um den PR mit neuen Referenzdaten zu öffnen.
4. **Settings → Environments → `github-pages`** (entsteht automatisch beim ersten Deploy): Deployment-Branch-Regel erlaubt `main`. Nur prüfen, falls du eigene Regeln nutzt.
5. Ergebnis: `https://schlonhorizon.github.io/<repo>/`. Screenshots eines PRs findest du im PR unter „Checks“ → Workflow-Lauf → Artifacts.

**Arbeiten ohne lokale Node-Umgebung:** Entwicklung über Claude-Code-Sessions und den GitHub-Webeditor (für Texte). Der PR-Check samt Screenshots ersetzt den lokalen Build und die Sichtkontrolle.

### 4.4 Diagnose-Seite `#/diagnose` (versteckt, ab P0)
Ziel: am Arbeitslaptop in 30 Sekunden sehen, was geht.
- **Download-Test:** lädt einen **30-MB-Test-Shard** (Zufallsbytes, beim Build erzeugt, nicht committet) von derselben Origin und zeigt Dauer, MB/s und ob der Download vollständig war (Proxy-Abbrüche).
- **Cache API:** Schreiben/Lesen eines Testeintrags, `navigator.storage.estimate()` (verfügbarer Speicher), `persist()`-Status.
- **WebGPU:** verfügbar ja/nein, Adapter-Info, Limits.
- **Rechen-Benchmark:** kurzer int8-Matmul-Test im Worker, hochgerechnet auf „geschätzt x ms pro Token für SmolLM2-135M“.
- **Umgebung:** Browser, `deviceMemory`, `hardwareConcurrency`, `prefers-reduced-motion`, Farbschema.
- Optionaler Button „Externe Erreichbarkeit testen“ (huggingface.co), rein informativ, nur auf Klick.
- Ergebnis als kopierbarer Text-Report.

## 5. Architektur

### 5.1 Ordnerstruktur
```
/
├─ .github/workflows/        deploy.yml, ci.yml, assets.yml
├─ scripts/
│  ├─ prepare-models.ts      (Node) Download gepinnte Revision, int8, Shards, manifest
│  ├─ make-testshard.ts      (Node) 30-MB-Test-Shard für #/diagnose
│  ├─ spike/                 (Node) Engine-Spike: run-spike.ts → docs/spike.md
│  └─ py/                    (Python, nur assets.yml) golden.py, export_embeddings.py,
│                            train_toy.py, requirements.txt
├─ content/                  ← HIER bearbeitest du Texte
│  ├─ chapters/              00-karte.mdx … 15-playground.mdx (Frontmatter: Titel, Slug, Lernziele, Quiz)
│  ├─ glossary.yaml
│  ├─ sources.yaml           Quellen + Datum für alle Claude/ChatGPT-Aussagen
│  └─ examples.yaml          Leitbeispiel, Kandidaten-Prompts, Trainingstexte, RAG-Dokumente, ICL-Aufgaben
├─ docs/                     spike.md, design-decisions.md
├─ licenses/                 Apache-2.0 (SmolLM2) + NOTICE, MIT (GPT-2-Tokenizer, KaTeX …), OFL (Schriften),
│                            CC-BY-SA-3.0 (fastText-abgeleitete Embeddings); Quelle für #/credits
├─ LICENSE                   MIT (Projektcode)
├─ public/
│  ├─ tokenizers/            gpt2.json, smollm2.json
│  ├─ fonts/                 (nur falls nicht über @fontsource)
│  └─ data/                  toy-model*.json, embeddings-de.bin (+ embeddings-de.LICENSE.txt), golden/*.json
├─ src/
│  ├─ app/                   Router, Layout, Navigation, Karten-Hub, Kapitelrahmen
│  ├─ chapters/<nn-slug>/    index.tsx (lädt MDX) + kapitelspezifische Widgets
│  ├─ pages/                 Diagnose, Glossar, Spickzettel, Credits, StyleTile (Design-Probe)
│  ├─ generated/             model-facts.json (beim Build aus den Modell-Configs erzeugt, nicht editieren)
│  ├─ components/
│  │  ├─ figure/             Figure, Callout, Dimension, Wire, PipelineNode, Stamp, Illustration
│  │  ├─ controls/           Slider, Toggle, Segmented, StepControls, TokenInput
│  │  ├─ viz/                TokenChips, BarDistribution, Heatmap, AttentionLines, VectorPlot,
│  │  │                      TreeView, Sparkline, MatrixView, NeuronDiagram
│  │  └─ learn/              Term, Quiz, MathBlock, DeepDive, Takeaways, KnobBox
│  ├─ engine/
│  │  ├─ tokenizer/          bpe.ts, loadTokenizer.ts
│  │  ├─ core/               ops.ts (matmul int8/f32, softmax, rmsnorm, rope, silu …), prng.ts
│  │  ├─ model/              transformer.ts (generisch, Hooks), config.ts, weights.ts (Shards + Cache API)
│  │  ├─ sampling.ts         temperature, top-k, top-p, min-p, repetition penalty, seeded
│  │  ├─ trace.ts
│  │  ├─ training/           bigram.ts, mlp.ts, tinynet.ts (Kap. 1)
│  │  ├─ retrieval/          bm25.ts
│  │  └─ worker.ts           Comlink-API: tokenize, forward, generate, train, ablate, benchmark
│  ├─ state/                 settings.ts, progress.ts, model.ts
│  ├─ hooks/                 useModel, useTrace, useGeneration, useReducedMotionPref
│  └─ styles/                tokens.css, global.css, katex-overrides.css
└─ PLAN.md, README.md
```

### 5.2 State-Management
- **`settings`** (persistiert): Theme (hell/dunkel/System), Mathe-Ebene, Animationen (voll/reduziert/aus; Default aus `prefers-reduced-motion`), bevorzugtes Modell.
- **`progress`** (persistiert): pro Kapitel `visited`, `completed`, Quiz-Ergebnis, letzter Abschnitt.
- **`model`** (flüchtig): Status je Modell (`idle | downloading(p) | ready | error`), aktives Modell, Fallback-Flag.
- Widget-lokaler State bleibt in React; der Playground serialisiert ihn zusätzlich in den URL-Hash.

### 5.3 Anbindung Modell ↔ Visualisierung
```
 Widget ──useTrace(prompt, opts)──▶ TraceCache (Key = Modell+Tokens+Patches) ──miss──▶ Worker (Comlink)
    ▲                                         │                                         │
    └──────────── Trace (immutable) ◀─────────┘◀──────── transformer.forward(tokens, {record}) ┘
```
- Einheitliches Interface **`LanguageModel`** (Toy und Real): `tokenize`, `forward(tokens, {record, patches})`, `generateStep`.
- **Trace** = `{ source, tokens, layers: [{ attn[h][q][k], residNorm, attnOut, mlpOut, logitLens }], logits }`. Aufgezeichnet wird nur, was angefordert ist.
- Sampling-Regler rechnen **im Main-Thread** auf gecachten Logits und reagieren sofort, ohne neuen Forward-Pass.
- Widgets leiten den **Stempel automatisch** aus `trace.source` ab. Damit ist Ehrlichkeit Teil der Architektur.

### 5.4 Korrektheit
- **Golden-Tests:** `golden.py` (manueller Workflow) erzeugt Referenz-Logits und -Attention für 5 Prompts mit PyTorch. Vitest vergleicht die TS-Engine dagegen, und zwar **in `ci.yml` (jeder PR) und in `deploy.yml`**. Das Kriterium ist robust statt einer absoluten Logit-Schwelle: Top-10-Tokens identisch, relative Abweichung der Logits < 1e-2 und Korrelation der Logit-Vektoren > 0,999. Die endgültigen Toleranzen für fp32 und int8 legt der Spike (P1a) fest, und `docs/spike.md` dokumentiert sie.
- Unit-Tests für Ops, Sampler (Top-p-Grenzfälle, Seed-Reproduzierbarkeit) und BPE (gegen Referenz-Tokenisierungen).

### 5.5 Modellzahlen aus der Config (keine festen Zahlen im Text)
- Ein Build-Schritt (`scripts/model-facts.ts`) liest `config.json` der gepinnten Revisionen (SmolLM2 Base/Instruct) und die Toy-Config. Er berechnet abgeleitete Größen (Parameterzahl je Komponente, Layer, d_model, Heads, KV-Heads, Vokabular, Kontextlänge, Download-Größe der Shards) und schreibt `src/generated/model-facts.json`.
- In MDX werden diese Werte als Variable eingesetzt, z. B. `<Fact m="smollm2" k="params" format="de" />` oder `{facts.smollm2.layers}`. Auch Bemaßungen in Diagrammen (`[T × d_model]`) lesen aus derselben Quelle.
- Ein Lint-Check im CI meldet hart codierte Modellzahlen in `content/` (Heuristik: bekannte Werte wie 576, 49152, 30 Layer in Prosa).

## 6. Design-System *(überarbeitet)*

**Leitidee:** Tinte auf warmem Papier. Ruhig, viel Weißraum, feine Line Art. Die Farbwelt ist angelehnt an die warme Claude-Oberfläche, ohne Logos, Markennamen oder proprietäre Schriften zu übernehmen. Der Footer trägt den Hinweis „Privates Lernprojekt, nicht mit Anthropic verbunden“. **Alle Werte unten sind Startwerte.** Endgültig festgelegt werden sie nach der Design-Probe (Phase D0), inklusive Kontrast- und Farbfehlsichtigkeits-Check.

### 6.1 Neutrale Grundpalette
| Token | Hell (warmes Off-White) | Dunkel (warmes Grau) | Verwendung |
|---|---|---|---|
| `--bg` | `#FAF8F3` | `#262523` | Seitenhintergrund (cremig, kein reines Weiß / kein reines Schwarz) |
| `--surface` | `#F3F0E8` | `#2F2E2B` | Panels, Abbildungsflächen |
| `--surface-raised` | `#FFFDF9` | `#383633` | Karten, Popover, Tooltips |
| `--border` | `#E3DED3` | `#45423D` | Trennlinien, Panelränder |
| `--ink` | `#2A2722` | `#EEEAE2` | Text, Hauptlinien der Diagramme |
| `--ink-muted` | `#6B655B` | `#ADA79C` | Sekundärtext, Beschriftungen, Bemaßungen (AA-geprüft) |
| `--ink-faint` | `#A39C90` | `#77716A` | nur Nicht-Text: Hilfslinien, inaktive Elemente |

### 6.2 Farbe → Bedeutung (seitenweit verbindlich)
Grundsätzlich monochrom. Farbe tritt **nur** in diesen Rollen auf und bedeutet überall dasselbe:

| Rolle | Hell | Dunkel | Bedeutung / Einsatz | Regeln |
|---|---|---|---|---|
| **Akzent** (Terrakotta) | `#B5562F` | `#E08A63` | **aktiv / ausgewählt / Datenfluss**: angeklicktes Token, gesampeltes Token, laufender Puls, Fokus-Ring, aktiver Regler | exklusiv; erscheint nie in einer anderen Rolle |
| **Sequenziell** (Sepia-Tinte, 5 Stufen) | `#EFE6D6 → #D6C6AD → #B39A78 → #87704F → #574632` | `#3A3631 → #5E5446 → #8C7A60 → #BCA585 → #E6D5B8` | **Größe eines Werts**: Attention-Gewichte, Wahrscheinlichkeiten, Aktivierungen, Heatmaps, Balken | „mehr“ = mehr Kontrast zum Hintergrund; Balken der Verteilung in Sepia, das gewählte Token in Akzent |
| **Divergierend** | Oliv `#5E7A3A` ↔ Mitte `#DCD5C8` ↔ Pflaume `#8A4F72` | Oliv `#9DB873` ↔ Mitte `#4A4640` ↔ Pflaume `#C48AAE` | **Richtung einer Änderung**: erhöht (+) / senkt (−) eine Wahrscheinlichkeit; Attribution, Ablations-Effekt, Logit-Differenz | immer redundant mit ▲/▼ bzw. ± |
| **A/B-Vergleich** | A = Petrol `#3E6D85` · B = Ocker `#A38419` | A = `#7FB0C8` · B = `#E0B84A` | Lauf A vs. Lauf B (Playground, Temperaturvergleich, Base vs. Instruct) | zusätzlich durchgezogen (A) vs. gestrichelt (B); farbfehlsichtig-sicheres Paar (Blau/Gelb-Achse) |
| **Kategorial** (3 Töne, D0) | Teal `#1A8574` · Lavendel `#8A7CC4` · Indigo `#3F467F` | `#62C4AC` · `#BDB3F2` · `#6A72C4` | **nur** zur Unterscheidung von **Attention-Heads** und **Embedding-Clustern** | höchstens 3 gleichzeitig hervorgehoben; weitere Heads bzw. Cluster bleiben Tinte bzw. `--ink-faint`; enthält keinen Ton einer anderen Rolle |
| **Chip-Abgrenzung** (neutral) | `#F1ECE3` / `#E6DFD2` abwechselnd | `#34322E` / `#3D3A35` abwechselnd | Flächen benachbarter **Token-Chips**: reine Abgrenzung, **keine Bedeutung** | keine bunten Chips; ein ausgewählter Chip bekommt den Akzent-Rahmen |
| **Quiz-Feedback** | Tinte + Form | Tinte + Form | richtig: ✓ mit durchgezogenem Rahmen · falsch: ✗ mit gestricheltem Rahmen, plus Erklärtext | bewusst **ohne eigene Farbe**, also keine Kollision mit Oliv oder Akzent |
| **Special Tokens** | Tinte, gestrichelter Rahmen, keine Fläche | entsprechend | `<|im_start|>`, `<eos>` usw. | Form statt Farbe |

Die Ehrlichkeits-Stempel sind ebenfalls monochrom und unterscheiden sich nur über die Form: `ECHT` als gefülltes Label, `ECHT · TOY` als durchgezogener Rahmen, `VORBERECHNET` als gepunkteter Rahmen, `ILLUSTRATIV` als gestrichelter Rahmen.

**Prüfregeln für das Farbsystem:**
1. **Eindeutigkeit:** Jeder Farbwert ist genau **einer** Rolle zugeordnet. Kein Hex-Wert erscheint in zwei Zeilen der Tabelle. Das sichert ein Test über `tokens.css` ab: Er schlägt fehl, wenn ein Wert doppelt vorkommt.
2. **Unterscheidbarkeit:** Zwischen allen bunten Rollenfarben eines Themes gilt ein Mindestabstand von ΔE2000 ≥ 15. Die Design-Probe zeigt die ΔE-Matrix und eine Simulation für Protanopie/Deuteranopie/Tritanopie. Fällt ein Paar durch, wird der Ton verschoben oder die Palette reduziert. *D0-Ergebnis:* Mit 4 kategorialen Tönen fielen Waldgrün ↔ Oliv (ΔE ≈ 10) und Schiefer ↔ Petrol (ΔE ≈ 10) durch, daher sind es jetzt 3 Töne. Ocker B wurde aufgehellt, damit er sich bei Deuteranopie vom Terrakotta-Akzent abhebt.
3. **Kontrast:** Text-Paare WCAG AA (≥ 4,5 : 1), grafische Elemente und Rollenfarben gegen `--bg` ≥ 3 : 1.
4. **Keine Bedeutung allein über Farbe:** Jede Farbrolle hat eine redundante Kodierung (Form, Linienstil, Symbol oder Beschriftung).

### 6.3 Typografie (nur Open Source, lokal über @fontsource)
- **Headlines: Serif, leicht verspielt.** Kandidaten für die Design-Probe:
  1. **Fraunces** (Variable Font mit `opsz`, `SOFT`, `WONK`), Startwert `SOFT 50`, `WONK 1` nur in Display-Größen. *Favorit:* warm, eigenständig, fein dosierbar.
  2. **Instrument Serif**: elegant, kondensiert, nur Regular/Italic.
  3. **Young Serif**: freundlich, kräftig, nur ein Schnitt.
- **Fließtext: Sans-Serif.** **Inter** (Variable) oder **IBM Plex Sans**, ebenfalls in der Probe. 17–18 px, Zeilenhöhe 1,6, Satzbreite ~68 Zeichen.
- **Monospace:** **IBM Plex Mono** oder **JetBrains Mono** für Tokens, IDs, Zahlen, Shapes, Bemaßungen, Code und Formel-Beschriftungen. Zahlen mit `tabular-nums`.
- Kapitelkopf: Serif-Headline + darüber eine Mono-Metazeile in Kapitälchen-Optik (`KAPITEL 05 · ATTENTION · ≈ 15 MIN`).
- Skala (Startwert): 13 / 15 / 17 / 21 / 28 / 40 / 56 px.

### 6.4 Grafiken: Line Art
- **Linien:** 1 px Hilfslinien, **1,25 px Standard**, 1,5 px Betonung, 2 px aktiver Datenfluss (Akzent). `stroke-linecap: round`, `stroke-linejoin: round`.
- **Formen:** Knoten mit 6 px Eckenradius (minimal verspielt), offene Chevron-Pfeilspitzen, keine Schatten, keine Verläufe.
- **Gestrichelt** (`3 4`, runde Enden) = optional, verborgen oder illustrativ.
- **Bemaßungen und nummerierte Callouts** bleiben als Stilmittel, aber dezent: in `--ink-muted`, feine Endstriche, Werte in Mono.
- **Kleine Line-Art-Illustrationen** auf der Intuitions-Ebene, selbst gezeichnet als SVG mit gleicher Strichstärke (z. B. Wörterbuch für Tokenizer, Bibliothekar mit Karteikarten für Attention, Förderband für Residual Stream, Würfel für Sampling, Notizblock für Reasoning).
- **Kein Rasterhintergrund.** Höchstens ein extrem dezentes Punktraster (3–4 % Deckkraft) innerhalb von Diagrammflächen, wo es beim Ablesen hilft.

### 6.5 Animationsprinzipien
1. **Bewegung erklärt Kausalität.** Ein Puls läuft entlang einer Verbindung, *weil* dort Daten fließen. Nichts bewegt sich nur zur Dekoration.
2. **Zeichnen beim Erscheinen** (`pathLength`, 600–900 ms, gestaffelt), einmalig und dezent.
3. **Parameter → sanfter Übergang:** Balken, Linienstärken und Zahlen tweenen in 200–300 ms.
4. **Schritt-Modus** überall, wo Sequenzen erklärt werden.
5. **Reduced Motion** (`prefers-reduced-motion` + eigener Schalter): keine Pulse, keine Zeichen-Animation, Crossfades ≤ 100 ms. Alle Informationen bleiben ohne Animation verständlich.
6. Easing `cubic-bezier(.2,.7,.2,1)`, linear nur für Datenpulse.

### 6.6 Wiederverwendbare Komponenten
`ChapterHeader` · `Figure` (Abbildungsrahmen + „Abb. 5.2“ + Stempel) · `Callout` (nummerierter Kreis + Hinweislinie) · `Dimension` (Bemaßung) · `PipelineNode` + `Wire` (mit optionalem Puls) · `Stamp` · `Illustration` · `Slider` (feine Skala mit Teilstrichen, Wert in Mono, Akzent-Griff) · `Toggle`, `Segmented` · `StepControls` · `TokenChips` · `MatrixView` · `Heatmap` · `BarDistribution` · `NeuronDiagram` · `Term` · `MathBlock` · `DeepDive` · `Quiz` · `KnobBox`.

## 7. Umsetzung in Phasen (mit Definition of Done)

Nach jeder Phase wird deployt und ist nutzbar. Am Ende jeder Phase stehen eine kurze Abnahme durch dich und ein Changelog-Eintrag.

### D0: Design-Probe (Style Tile) · **S–M**
Minimales Vite-Gerüst + `deploy.yml` (wird in P0 weiterverwendet, nicht weggeworfen) + `LICENSE` (MIT). Eine einzelne Seite zeigt: Farbpalette (neutral + alle Rollen aus 6.2) in beiden Modi mit Kontrastwerten und ΔE-Matrix, **3 Headline-Kandidaten** nebeneinander (Fraunces / Instrument Serif / Young Serif) mit **2 Body-Kandidaten** (Inter / IBM Plex Sans) und 2 Mono-Kandidaten, ein Fließtext-Beispiel mit Formel, ein animiertes Line-Art-Diagramm (Mini-Pipeline mit Puls und sich zeichnenden Linien), einen Regler, Token-Chips inkl. Special Token, alle vier Stempel, ein Callout mit Bemaßung, eine kleine Line-Art-Illustration und den Footer-Hinweis.
**DoD:**
- [ ] Seite ist unter der Pages-URL erreichbar, Reload funktioniert.
- [ ] Light/Dark-Umschalter funktioniert, System-Präferenz wird beim ersten Besuch übernommen.
- [ ] Alle Schriften lokal: Im Netzwerk-Tab gibt es keine Requests an fremde Domains.
- [ ] Kontrastwerte aller Text/Hintergrund-Paare und die ΔE-Matrix der Rollenfarben stehen auf der Seite; Verstöße gegen die Prüfregeln aus 6.2 sind markiert.
- [ ] Mit Reduced Motion laufen keine Animationen.
→ **Entscheidung durch dich:** Schriften, Feinjustierung der Farben, danach Festschreibung in `docs/design-decisions.md`.

### P0: Fundament · **M**
Hash-Routing mit allen 16 Kapitel-Routen (Platzhalter), MDX + KaTeX, Design-Tokens final, Basiskomponenten (ChapterHeader, Figure, Callout, Stamp, Slider, Toggle), Settings/Progress-Stores, Glossar-Mechanik, Mathe-Toggle, `<Fact>`-Mechanik für Modellzahlen (5.5), **`#/diagnose`**, **`#/credits`**, `ci.yml` mit Playwright-Screenshots, Gerüst für `assets.yml`.
**DoD:**
- [ ] `#/credits` listet alle bis dahin eingebundenen Drittkomponenten mit Lizenz (Schriften/OFL, KaTeX, React usw.); ein Test prüft, dass jede Datei in `licenses/` auf der Seite verlinkt ist.
- [ ] Ein Test über `tokens.css` prüft die Eindeutigkeitsregel aus 6.2 (kein Farbwert in zwei Rollen).
- [ ] Deep-Link `…/#/kapitel/attention` lädt direkt, auch nach Reload.
- [ ] Eine Textänderung in einer MDX-Datei über den GitHub-Webeditor erscheint nach dem Push automatisch live.
- [ ] Ein PR erzeugt ein Screenshot-Artifact aller Routen in Light und Dark.
- [ ] `#/diagnose` zeigt Download-Rate des 30-MB-Test-Shards, Cache-API-Status, WebGPU-Status und Rechen-Benchmark; der Report ist kopierbar.
- [ ] Mathe-Toggle blendet Formeln seitenweit ein und aus; die Einstellung überlebt einen Reload.

### P1a: Engine-Spike · **M**
1. Prüfen, ob die Claude-Session huggingface.co erreicht. Falls nicht, läuft der Spike als GitHub-Actions-Job (Trigger: Push auf den Spike-Branch), und alles Modellbezogene läuft dauerhaft nur über Actions.
2. Minimale TS-Engine (Llama-Architektur) lädt SmolLM2-135M (Base), Forward-Pass in Node.
3. Vergleich der Top-5-Logits mit PyTorch-Referenz für 2 Prompts, in fp32 **und** int8.
4. Messung: ms/Token (mit KV-Cache), Prefill-Zeit für 16/64/128 Tokens, Speicherbedarf.
5. **Leitbeispiel-Validierung:** Top-10-Fortsetzungen (Base und Instruct) für 3–4 Kandidaten-Prompts (Finanz-Prompt, „The capital of France is“, 1–2 weitere aus `examples.yaml`).
6. Bericht `docs/spike.md` mit Empfehlung: weiter wie geplant / WebGPU vorziehen / Alternativen.
**DoD:**
- [ ] fp32: Top-10-Tokens identisch zur Referenz, relative Logit-Abweichung < 1e-2, Korrelation der Logit-Vektoren > 0,999 (Startkriterium).
- [ ] int8: dieselben Kennzahlen gemessen und dokumentiert. Die **endgültigen Toleranzen** für fp32 und int8 sind in `docs/spike.md` festgelegt und begründet; die Golden-Tests übernehmen sie.
- [ ] Laufzeit- und Speicherwerte gemessen und mit der Zielgröße verglichen (≤ 0,5 s/Token in Node als Richtwert; Browser-Werte folgen über `#/diagnose`).
- [ ] Kandidaten-Prompts mit Verteilungen im Bericht → **Entscheidung durch dich: Leitbeispiel**.
- [ ] Klarer Go/No-Go-Vorschlag für P2.

### P1b: MVP · **L**
Toy-Modell (Training + Engine-Kern + Sampler), echte Tokenizer, **Kapitel 0, 2, 7, 8** + **Mini-Playground** (Prompt, Sampling-Regler, Generierung; `#/playground`).
**DoD:**
- [ ] Kap. 2: Eigener Text wird mit GPT-2- und SmolLM2-Tokenizer tokenisiert; die IDs stimmen mit den Referenz-Tokenisierungen der Tests überein.
- [ ] Kap. 7: Temperatur/Top-k/Top-p/Seed ändern die Verteilung ohne spürbare Verzögerung; gleicher Seed → gleiches Sample (Test).
- [ ] Kap. 8: Step-Modus und Token-Erzwingen erzeugen einen sichtbaren Antwort-Baum.
- [ ] Mini-Playground generiert mit dem Toy-Modell, alle Regler wirken.
- [ ] Alles funktioniert offline nach dem ersten Laden (kein echtes Modell nötig), alle Abbildungen tragen den richtigen Stempel.

### P2: Echtes Modell + Grundlagen fürs Innenleben · **L**
`prepare-models.ts` + Cache im Deploy, int8-Loader + Cache API, Modell-Lade-UI + Fallback-Banner, Golden-Tests; Kap. 2/7/8 + Mini-Playground auf echtes Modell heben; **Kapitel 1 (Neuronale Netze), 5 (Attention), 6 (Block)**.
**DoD:**
- [ ] Deploy **und** PR-Check nutzen bei unveränderter Revision die gecachten Shards (sichtbar im Log, kein Download).
- [ ] Golden-Tests laufen in `ci.yml` und `deploy.yml` grün; ein Test-PR mit absichtlich falscher Engine-Änderung wird **schon im PR-Check** rot.
- [ ] Alle Modellzahlen im Seitentext kommen aus `model-facts.json`; der Lint-Check findet keine hart codierten Werte.
- [ ] `#/credits` enthält SmolLM2 (Apache-2.0, Lizenz + Hinweis mitgeliefert) und den GPT-2-Tokenizer (MIT).
- [ ] Modell-Download zeigt Fortschritt; der zweite Besuch lädt aus dem Cache (kein Netzwerk-Download).
- [ ] Bei blockiertem Download (simuliert) bleibt die Seite voll nutzbar mit Toy-Banner.
- [ ] Kap. 5: Attention-Viewer zeigt echte Gewichte für alle Layer/Heads; Head-Ablation verändert die Verteilung; Kap. 6: Logit Lens für das Leitbeispiel.

### P3: Grundlagen komplett · **M–L**
**Kapitel 3 (Embeddings), 4 (Position), 9 (Training)**; `assets.yml` produktiv (Embeddings, Toy-Modelle).
**DoD:**
- [ ] `assets.yml` läuft manuell durch und öffnet einen PR mit den Datendateien (< 5 MB gesamt).
- [ ] Kap. 3: `König − Mann + Frau` liefert „Königin“ unter den Top-3 (vorberechnete Vektoren).
- [ ] `embeddings-de.bin` wird mit Namensnennung (Grave et al., fastText) und unter CC BY-SA 3.0 ausgeliefert (Lizenzdatei daneben + Eintrag in `#/credits`).
- [ ] Kap. 4: Toy-Modell ohne Positionssignal liefert für permutierte Sätze identische Ausgaben, mit Signal unterschiedliche.
- [ ] Kap. 9: Live-Training auf eigenem Text senkt den Loss sichtbar, UI bleibt flüssig (Worker), Overfitting-Demo zeigt Train/Val-Schere.

### P4: Vom Modell zum Produkt · **L**
Instruct-Modell, **Kapitel 10, 11, 12, 13, 14**.
**DoD:**
- [ ] Kap. 10: Base und Instruct antworten nebeneinander auf denselben Input; Chat-Template mit Special Tokens sichtbar.
- [ ] Kap. 11: Mit dem 64-Token-Fenster „vergisst“ das Modell nachweislich eine frühe Information.
- [ ] Kap. 12: Ziel-Token-Wahrscheinlichkeit für 0/1/3 Beispiele wird echt gemessen und angezeigt; Umsortieren der Beispiele aktualisiert die Werte.
- [ ] Kap. 14: Gedankenketten-Messung, BM25-Retrieval mit Scores und Agent-Loop-Animation funktionieren; alle Claude/ChatGPT-Aussagen haben Quelle + Datum in `sources.yaml`.

### P5: Synthese · **L**
**Kapitel 15** (voller Playground: Full Trace, A/B-Diff, Attribution, URL-Zustand), Quizzes für alle Kapitel, Spickzettel, Glossar-Feinschliff.
**DoD:**
- [ ] Zwei Läufe mit unterschiedlichen Parametern zeigen Token-Diff, KL-Sparkline und Abzweig-Punkt.
- [ ] Ein geteilter Playground-Link reproduziert exakt denselben Lauf (inkl. Seed).
- [ ] Jedes Kapitel hat ≥ 3 Quizfragen; der Fortschritt erscheint auf der Karte und überlebt einen Reload.
- [ ] Jeder Fachbegriff ist beim ersten Auftreten pro Kapitel als `<Term>` verlinkt (Lint-Check im CI).

### P6: Politur · **M**
Optional WebGPU-Matmul, Performance, Tablet-Layout, A11y, inhaltliches Review.
**DoD:**
- [ ] Alle Kapitel auf 1024 px Breite (Tablet quer) ohne horizontales Scrollen bedienbar.
- [ ] Alle Widgets per Tastatur bedienbar; Diagramme haben Textalternativen.
- [ ] Lighthouse-Accessibility ≥ 95 auf jeder Route.
- [ ] Falls WebGPU umgesetzt: gleiche Golden-Test-Ergebnisse wie der JS-Pfad, messbar schneller.

## 8. Entscheidungen, offene Fragen & Risiken

### 8.1 Entscheidungen aus Runde 1
| # | Thema | Entscheidung |
|---|---|---|
| 1 | Repo-Sichtbarkeit | öffentlich |
| 2 | Echtes Modell | SmolLM2-135M Base + Instruct |
| 3 | Inferenz | eigene TS-Engine |
| 4 | Modell-Hosting | im CI vom HF Hub → gecacht → same-origin über Pages |
| 5 | Prompt-Sprache echtes Modell | Englisch (UI und Toy Deutsch) |
| 6 | Struktur | Kapitel-Unterseiten mit Scroll-Story |
| 7 | Leitbeispiel | Finanz-Prompt bevorzugt, **endgültig nach Spike** |
| 8 | Inhaltsformat | MDX |
| 9 | Live-Training | Bigram + Mini-MLP |
| 10 | Download-Verhalten | echtes Modell nur auf Klick |
| 11 | Scope | MVP zuerst, dann iterativ |
| + | Playground | Mini-Playground in P1b, voller Playground in P5 |
| + | Design | warme Claude-angelehnte Palette, Serif-Headlines, Line Art (Abschnitt 6) |
| + | CI | Python nur im manuellen `assets.yml`; Deploy und PR-Check Node-only; Golden-Tests in **beiden** |
| + | Spike-Toleranz | robustes Kriterium (Top-10, relative Abweichung, Korrelation); endgültig im Spike |
| + | Farbsystem | jede Farbe genau eine Rolle; Token-Chips neutral; kategorial nur Heads/Cluster (3 Töne nach D0-Prüfung); Quiz ohne Farbe |
| + | Lizenzen | Projektcode MIT; `#/credits` mit allen Quellen; fastText-Derivat unter CC BY-SA 3.0 |
| + | Modellzahlen | aus Config gelesen bzw. berechnet, nie hart im Text |

### 8.2 Entscheidungen aus Runde 2 (vormals offene Fragen)
| # | Thema | Entscheidung |
|---|---|---|
| A | Kapitelnummerierung | neu durchnummeriert (0–15), Routen über Slugs |
| B | In-Context Learning | eigenes Kapitel 12 |
| C | Kapitel 1 (Neuronale Netze) | in P2, zusammen mit Attention |
| D | Referenzdaten aus `assets.yml` | automatischer PR |
| E | Screenshots im PR-Check | nur Toy-Modus |
| F | Footer-Hinweis | ja: „Privates Lernprojekt, nicht mit Anthropic verbunden“ |

### 8.3 Lizenzen & Quellen
| Komponente | Lizenz | Pflichten / Umsetzung |
|---|---|---|
| Projektcode | **MIT** (`LICENSE`) | – |
| SmolLM2-135M / -Instruct (Gewichte, Tokenizer) | Apache-2.0 | Lizenztext + Hinweis auf Herkunft und Änderungen (int8-Quantisierung) in `licenses/` und `#/credits` |
| GPT-2-Tokenizer (`vocab`/`merges`) | MIT (OpenAI) | Copyright-Hinweis + Lizenztext |
| Schriften (Fraunces, Instrument Serif, Young Serif, Inter, IBM Plex, JetBrains Mono) | SIL OFL 1.1 | Lizenztexte mitliefern (über @fontsource enthalten), Nennung in `#/credits` |
| fastText-Wortvektoren (Deutsch) | CC BY-SA 3.0 | Namensnennung (Grave et al. 2018); **abgeleitete `embeddings-de.bin` unter derselben Lizenz** mit eigener Lizenzdatei |
| npm-Abhängigkeiten (React, Motion, D3, KaTeX …) | überwiegend MIT/ISC/BSD | automatisch generierte Liste (`license-checker` o. ä. im Build) auf `#/credits` |
| Weitere Datenquellen (Trainingstexte, RAG-Dokumente, Beispiele) | je Quelle | **vor dem Einbinden prüfen**; bevorzugt gemeinfreie Texte (z. B. Goethe) oder selbst geschriebene/fiktive Inhalte; jede Quelle mit Lizenz in `content/sources.yaml` |

### 8.4 Risiken
- **Performance auf dem Arbeitslaptop:** Die JS-Inferenz über 30 Layer kann träge sein. Gegenmaßnahmen: Spike misst früh, `#/diagnose` misst am echten Gerät, Prompt-Limit, Schritt-Modus, WebGPU als Option.
- **Firmennetz:** Wenn `*.github.io` selbst blockiert ist, hilft nur Codespaces oder ein privates Gerät. Gedrosselte Downloads werden über Shards, Cache und die Diagnose-Seite sichtbar.
- **HF Hub im CI nicht erreichbar oder Revision verschwindet:** Die Revision ist gepinnt, und der Cache bleibt bis zu 7 Tage ohne Zugriff erhalten. Als Notfall-Option lassen sich die Shards zusätzlich als Release-Asset im Repo ablegen (Release-Assets dürfen bis 2 GB groß sein).
- **Qualität des 135M-Modells:** Halluziniert stark, schwach im Reasoning. Das wird offen kommuniziert und didaktisch genutzt (Kap. 13, 14.1), nicht versteckt.
- **Claude/ChatGPT-Fakten veralten:** Sie liegen zentral in `content/sources.yaml` mit Datum und lassen sich leicht aktualisieren.
- **Design-Nähe zur Claude-Oberfläche:** nur Farbwelt und Anmutung, keine Logos, Namen oder proprietären Schriften; dazu der Footer-Hinweis.
- **Scope Creep:** feste Abnahme nach D0, P1a und P1b.

## 9. Eigene Ideen (zusätzlich, priorisiert)

1. **„Warum dieses Token?“-Erklärer** (in Kap. 15 eingeplant): Occlusion-Attribution, echt berechnet. ⭐
2. **Vorhersage-Spiel:** Du tippst das nächste Token samt Wahrscheinlichkeit, ein Brier-Score misst, wie gut deine Intuition kalibriert ist (passt zu Finance). ⭐
3. **Head-Ablation / „Gehirnchirurgie“** (Kap. 5/6). ⭐
4. **Finance-Track:** optionale Finanzbeispiele in jedem Kapitel, dazu ein Token-Kosten-Rechner (Prompt-Länge × öffentliche API-Preise, mit Datum).
5. **Sampling-Monte-Carlo** (Kap. 13): 100 Antworten als Verteilung der Antworten.
6. **Guided Tour vs. freier Modus.**
7. **Spickzettel-Poster** als druckbares A3-SVG.
8. **Toy-Gewichte editierbar** (Mechanistic-Interpretability-Gefühl).
9. **Tastaturkürzel** und **Präsentationsmodus**.
10. **Später:** Steering-Vektoren, Speculative Decoding, Mixture of Experts, Multimodalität als Ausblickskapitel.
