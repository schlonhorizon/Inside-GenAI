# Engine-Spike (P1a): Bericht

Stand: 2026-11-23 · Workflow: `.github/workflows/spike.yml` · Rohdaten des letzten Laufs: [`spike-raw.md`](spike-raw.md)

## Kurzfassung

| Frage | Antwort |
|---|---|
| Rechnet die eigene TS-Engine richtig? | **Ja.** In fp32/fp16 stimmen die Logits mit PyTorch praktisch exakt überein (KL ≈ 10⁻¹¹, rel. Fehler ≈ 2·10⁻⁶), die Greedy-Fortsetzungen sind 16/16 identisch. Das gilt für Base und Instruct. |
| Ist fp16-Auslieferung verlustfrei? | **Ja.** Die Originalgewichte liegen in bf16 vor; fp16 bildet sie ohne messbaren Unterschied ab. |
| Bringt Gruppen-int8 (128/64/32/16) mehr Genauigkeit? | **Allein kaum.** Die Gewichte werden um 10–25 % genauer rekonstruiert, an den Wahrscheinlichkeiten bleibt die Abweichung bei 2,4–3,4 pp. Die Reihenfolge der Gruppengrößen ist dabei nicht einmal monoton. |
| Woher kommt der int8-Fehler? | **Hauptquelle sind die Attention-Projektionen (q/k/v/o)**, danach MLP-down. Embedding/LM-Head und MLP gate/up tragen wenig bei (Details unten). |
| Welche Variante wird Standard? | **Gemischt: Attention in fp16, alles andere int8 mit Gruppen zu 32** (`attn:f16,mlp:q8g32,down:q8g32,emb:q8g32+s16`). Max. Δp **1,31 pp** statt 2,38 pp, ≈ **160 MB** Download pro Modell. |
| Passt alles auf GitHub Pages? | int8 für beide Modelle + fp16 für beide wären 868 MB, also über der 800-MB-Grenze. **Darum gibt es fp16 nur für das Base-Modell:** 611 MB gesamt. |
| Ist es schnell genug? | **Generieren ja:** ≈ 100–170 ms pro Token, je nach Runner. **Prompt-Einlesen noch nicht:** Token für Token ≈ 80–130 ms pro Token. Batch-Prefill ist feste P2-Aufgabe. |
| Go / No-Go für P2? | **Go.** |

## Aufbau

1. **Referenz (PyTorch, fp32):** `scripts/spike/reference.py` speichert Token-IDs, komplette Logits nach dem letzten Prompt-Token, Top-10 und eine Greedy-Fortsetzung über 16 Tokens für 6 Prompts (4 Base, 2 Instruct/Chat).
2. **Eigene Engine:** `scripts/spike/run-engine.ts` rechnet dieselben Prompts in 23 Gewichtsformaten. Das Format lässt sich pro Modellteil wählen (`attn`, `mlp`, `down`, `emb`).
3. **Vergleich:** `scripts/spike/report.py` berechnet KL, die größte Abweichung einer Wahrscheinlichkeit (Δp, mit Token), Top-1/Top-10 und die Greedy-Übereinstimmung. Es wendet die vorab festgelegte Entscheidungsregel an und rechnet das Pages-Budget aus der echten Build-Größe.

**Modelle (gepinnt für P2):**
- `HuggingFaceTB/SmolLM2-135M` @ `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f77366fa6b3b4b768ec3050bf629380bac`

**Umgebung:** GitHub-Actions-Runner (ubuntu-latest, x64), Node 22, ein Thread. Die Runner-Hardware schwankt zwischen Läufen; absolute Zeiten sind daher nur Richtwerte. Werte für deinen Laptop liefert `#/diagnose`.

Zusätzlich prüft `test/engine.test.ts` die Engine bei jedem Push gegen eine unabhängige NumPy-Referenz an einem Mini-Modell.

## Gewichtsformate im Vergleich

Schlechtester Wert über alle 6 Prompts. Download = Base-Modell, Skalen als fp16 (wie ausgeliefert).

| Variante | Download | RAM | max. Δp | max. KL | schlechtestes Δp bei |
|---|---|---|---|---|---|
| fp16 (exakt) | 257 MB | 513 MB | 0,00 pp | 7·10⁻¹² | – |
| int8 pro Zeile | 129 MB | 129 MB | 2,38 pp | 5,8·10⁻³ | Base/stocks: ` the` 50,1 % → 47,7 % |
| int8 Gruppen 128 | 129 MB | 130 MB | 3,11 pp | 5,8·10⁻³ | Base/stocks: ` the` 50,1 % → 47,0 % |
| int8 Gruppen 64 | 132 MB | 136 MB | 3,31 pp | 3,5·10⁻³ | Instruct/chat-rates: `Central` 76,0 % → 72,7 % |
| int8 Gruppen 32 | 136 MB | 144 MB | 2,65 pp | 2,5·10⁻³ | Instruct/chat-rates: `Central` 76,0 % → 78,7 % |
| int8 Gruppen 16 | 144 MB | 160 MB | 3,40 pp | 4,1·10⁻³ | Instruct/chat-rates: `Central` 76,0 % → 79,4 % |
| int8 pro Zeile, aber Attention fp16 | 154 MB | 205 MB | 2,18 pp | 5,5·10⁻³ | Base/stocks: ` the` 50,1 % → 47,9 % |
| **int8 Gruppen 32, aber Attention fp16** ✓ | **160 MB** | **217 MB** | **1,31 pp** | **1,3·10⁻³** | Instruct/chat-rates: `Central` 76,0 % → 74,7 % |
| int8 Gruppen 32, aber Embedding fp16 | 162 MB | 222 MB | 2,51 pp | 2,3·10⁻³ | Instruct/chat-rates: `Central` 76,0 % → 78,5 % |

Alle Varianten treffen das wahrscheinlichste Token in 6/6 Fällen, die Top-10 überlappen zu ≥ 9/10.

**Am Leitbeispiel** (`The capital of France is`):

| Token | PyTorch | fp16 | int8 pro Zeile | gewählt (Attention fp16, Rest Gruppen-int8) |
|---|---|---|---|---|
| ` the` | 26,2 % | 26,2 % | 26,5 % | 25,2 % |
| ` Paris` | 9,4 % | 9,4 % | 9,8 % | 9,7 % |
| ` located` | 7,3 % | 7,3 % | 6,8 % | 7,5 % |
| ` called` | 4,4 % | 4,4 % | 4,5 % | 4,3 % |
| ` a` | 3,9 % | 3,9 % | 3,6 % | 3,8 % |

## Woher kommt die Abweichung? (Ursachenanalyse)

**1. Wo tritt das schlechteste Δp auf?** Immer bei einem Token mit **hoher Wahrscheinlichkeit**: `Central` (76 %, Instruct-Chat) bzw. ` the` (50 %, Base/stocks). Das ist erwartbar, denn eine kleine Logit-Verschiebung Δz ändert die Wahrscheinlichkeit um etwa p·(1−p)·Δz. Bei p = 0,5–0,8 wirkt sie am stärksten. Die gemessenen 2–3 pp entsprechen Logit-Abweichungen von nur ≈ 0,1–0,2.

**2. Ablation: Welcher Teil verursacht den Fehler?** Jeweils nur ein Teil int8, alles andere fp16:

| nur dieser Teil int8 | Anteil am Download | max. Δp (pro Zeile) | max. Δp (Gruppen 32) | max. KL (Gruppen 32) |
|---|---|---|---|---|
| Attention (q/k/v/o) | ≈ 20 % | 2,28 pp | **3,72 pp** | 4,6·10⁻³ |
| MLP down | ≈ 20 % | 1,73 pp | 1,79 pp | 1,3·10⁻³ |
| MLP gate/up | ≈ 40 % | 1,46 pp | 0,50 pp | 4,6·10⁻⁴ |
| Embedding = LM-Head | ≈ 20 % | 0,88 pp | 0,56 pp | 3,5·10⁻⁴ |

**Hauptquelle sind die Attention-Projektionen.** Allein quantisiert erzeugen sie so viel Abweichung wie das ganze Modell in int8. Eine plausible Erklärung: Fehler in q und k werden über das Skalarprodukt q·k und die anschließende Softmax (Exponentialfunktion) verstärkt und ändern, worauf ein Token achtet. Das wirkt dann in allen folgenden Layern nach.

Die umgekehrte Ablation (alles int8 außer einem Teil) bestätigt das: Den größten Gewinn bringt es, die Attention in fp16 zu lassen. Mit Gruppen-int8 im Rest sinkt der Fehler dann von 2,65 pp auf 1,31 pp.

**3. Verliert die Zwischenrechnung unnötig an Genauigkeit?**
- **Akkumulation:** JavaScript-Zahlen sind f64. Summen in Matrixprodukten, RMSNorm und Softmax laufen also in doppelter Genauigkeit. Das ist unkritisch.
- **Aktivierungen:** Sie werden zwischen den Schritten als f32 gespeichert und nicht weiter verkleinert.
- **Skalen-Format:** int8 pro Zeile mit f32-Skalen ergibt 2,97 pp, mit fp16-Skalen 2,38 pp. Der Unterschied liegt im Rauschen; fp16-Skalen schaden also nicht und werden so ausgeliefert.
- **Quantisierung:** symmetrisch ±127 mit `Math.round` (Standardverfahren).

**4. Warum helfen kleinere Gruppen allein kaum?** Die Gewichte von SmolLM2 haben wenige Ausreißer. Kleinere Gruppen verbessern die Rekonstruktion jedes Gewichts nur um 10–25 % (siehe `test/engine.test.ts`). Über 30 Layer addieren sich die verbleibenden Fehler eher zufällig. Darum schwanken die Messwerte zwischen den Gruppengrößen stärker, als sie sich systematisch verbessern; mit nur 6 Prompts ist die Rangfolge unter den reinen Gruppenvarianten nicht belastbar. Belastbar ist das Ablationsergebnis, weil es bei beiden Quantisierungsarten in dieselbe Richtung zeigt.

## Entscheidung

**Regel (vor der Messung festgelegt):** Kandidaten sind alle int8-Varianten mit fp16-Skalen und die gemischten Varianten. Maßgeblich ist das schlechteste max. Δp über alle Prompts. Gewählt wird die kleinste (bei Gleichstand die schnellere) Variante, die höchstens 0,5 pp über der genauesten liegt.

**Ergebnis:**
- Die genaueste Variante (1,31 pp) ist zugleich die einzige im Toleranzband: **Attention fp16, MLP und Embedding int8 mit Gruppen zu 32, Skalen fp16.**
- Download ≈ 160 MB pro Modell (statt 129 MB bei int8 pro Zeile), RAM ≈ 220 MB.
- Attention in f32 rechnet in JS etwas schneller als int8; die Geschwindigkeit liegt damit zwischen fp16 und reinem int8.

**Kennzeichnung:** Diese Fassung trägt den Stempel **„Echt · int8“**, die fp16-Fassung den Stempel **„Echt“**.

## Pages-Budget

| Szenario | Größe |
|---|---|
| Seite ohne Modelle (gemessen, inkl. 30-MB-Test-Shard) | 34 MB |
| (a) int8 Base + int8 Instruct + fp16 Base + fp16 Instruct | **868 MB** über 800 MB |
| (b) int8 Base + int8 Instruct + fp16 nur Base | **611 MB** |

**Entscheidung nach deiner Regel:**
- Die exakte fp16-Fassung gibt es **nur für das Base-Modell**; das Instruct-Modell nur in der int8-Fassung.
- Kapitel 10 (Base vs. Instruct) vergleicht deshalb beide int8-Fassungen; der Stempel zeigt das an.
- Das weiche Bandbreitenlimit von Pages (≈ 100 GB/Monat) ist bei privater Nutzung unkritisch.

## Leistung (Base-Modell, letzter Lauf)

| Variante | Download | RAM | Decode | Prefill 16 / 64 / 128 Tokens |
|---|---|---|---|---|
| fp16 (im Speicher f32) | 257 MB | 513 MB | 99 ms/Token | 1,3 / 5,4 / 10,5 s |
| int8 pro Zeile | 129 MB | 129 MB | 129 ms/Token | 1,7 / 6,7 / 13,6 s |
| int8 Gruppen 32 | 136 MB | 144 MB | 122 ms/Token | 1,6 / 6,4 / 12,7 s |

Frühere Läufe auf langsameren Runnern ergaben ≈ 160–170 ms/Token. Prefill läuft derzeit Token für Token. **Batch-Prefill** (alle Prompt-Tokens als Matrix×Matrix, jede Gewichtszeile wird nur einmal gelesen) kommt in P2. Bis dahin gilt ein Prompt-Limit von 128 Tokens.

## Leitbeispiel (entschieden)

- **Roter Faden:** `The capital of France is`. Das Base-Modell setzt am wahrscheinlichsten mit ` the` fort (26 %), ` Paris` liegt erst auf Platz 2 (9 %). Ein LLM ist eine Verteilung über Fortsetzungen, keine Faktendatenbank. Das Instruct-Modell antwortet „The capital of France is Paris.“
- **Kapitel 13 (Grenzen):** `The central bank raised interest rates because`. Die Greedy-Fortsetzung „it wanted to stimulate the economy“ klingt plausibel, ist ökonomisch aber falsch.
- **Kapitel 7 (Sampling):** `Once upon a time, there was a`. Die Verteilung ist flach (` little` 13 %, ` curious` 9 %, ` wise` 9 % …).

**Nebenbefund für P1b/P2:** Nach `<|im_end|>` schreibt das Instruct-Modell weiter Template-Tokens. Die Generierung muss am End-Token stoppen.

## Festgelegte Toleranzen für die Golden-Tests

| Pfad | Kriterien (alle müssen erfüllt sein) | gemessen (schlechtester Wert) |
|---|---|---|
| **exakt** (fp16) | Top-10 identisch · rel. L2 < 10⁻⁴ · KL < 10⁻⁶ · Greedy 16/16 | 2,4·10⁻⁶ · 7·10⁻¹² · 16/16 |
| **int8-Fassung** (gewählte Variante) | Top-1 identisch · Top-10-Überlappung ≥ 8/10 · KL < 4·10⁻³ · max. Δp < 3,5 pp | 6/6 · 9/10 · 1,3·10⁻³ · 1,31 pp |

Die int8-Grenzen lassen etwa den Faktor 2,5–3 Spielraum. Echte Fehler (falsche Indizes, vertauschte Gewichte) erzeugen KL-Werte im Bereich > 1 und fallen sofort auf. In P2 werden die Golden-Tests auf ≈ 20 Prompts erweitert, damit die Werte statistisch stabiler sind.

## Aufgaben für P2 aus dem Spike

1. Shards beim Build erzeugen: int8-Fassung (Attention fp16, Rest q8g32, Skalen fp16) für Base und Instruct, fp16 nur für Base; gepinnte Revisionen.
2. Batch-Prefill in der Engine.
3. Stopp-Tokens aus der Tokenizer-Config.
4. Golden-Tests mit den Toleranzen oben in `ci.yml` und `deploy.yml`, ≈ 20 Prompts.
5. Download-Knopf „Exakte Fassung laden (≈ 260 MB)“ nur beim Base-Modell; Stempel „Echt · int8“ bzw. „Echt“ automatisch je nach geladener Fassung.
