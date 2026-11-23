# Engine-Spike (P1a): Bericht

Stand: 2026-11-23 · Workflow: `.github/workflows/spike.yml` · Rohdaten: [`spike-raw.md`](spike-raw.md)

## Kurzfassung

| Frage | Antwort |
|---|---|
| Rechnet die eigene TS-Engine richtig? | **Ja.** In fp32 stimmen die Logits mit PyTorch praktisch exakt überein (relativer Fehler ≈ 2·10⁻⁶, KL ≈ 10⁻¹¹), die Greedy-Fortsetzungen sind 16/16 identisch. Das gilt für Base und Instruct. |
| Ist fp16-Auslieferung verlustfrei? | **Ja.** Die Originalgewichte liegen in bf16 vor; fp16 bildet sie ohne messbaren Unterschied ab (gleiche Werte wie fp32). |
| Wie gut ist int8? | **Brauchbar, aber nicht exakt.** Das wahrscheinlichste Token stimmt immer (6/6), die Top-10 überlappen zu ≥ 9/10, Wahrscheinlichkeiten weichen um bis zu **3 Prozentpunkte** ab. Längere Greedy-Texte laufen nach 6–16 Tokens auseinander. |
| Hilft feinere Quantisierung (Gruppen, fp16-Ausgabeschicht)? | **Kaum.** Gruppen-int8 ist nur wenig genauer, aber 10–20 % langsamer und etwas größer. Verworfen. |
| Ist es schnell genug? | **Beim Generieren ja:** ≈ 160–170 ms pro Token (Ziel ≤ 500 ms). **Beim Einlesen des Prompts nein:** Token für Token ≈ 130 ms pro Token, also ≈ 17 s für 128 Tokens. Das muss in P2 schneller werden (Batch-Prefill). |
| Go / No-Go für P2? | **Go**, mit zwei Entscheidungen (unten) und Batch-Prefill als fester P2-Aufgabe. |

## Aufbau

1. **Referenz (PyTorch, fp32):** `scripts/spike/reference.py` lädt die Modelle vom Hugging Face Hub und speichert Token-IDs, komplette Logits nach dem letzten Prompt-Token, Top-10 und eine Greedy-Fortsetzung über 16 Tokens.
2. **Eigene Engine:** `scripts/spike/run-engine.ts` rechnet dieselben Prompts mit `src/engine` in mehreren Gewichtsformaten und misst die Zeiten.
3. **Vergleich:** `scripts/spike/report.py` berechnet Top-1/Top-10, relativen L2-Fehler, Korrelation, KL-Divergenz, die größte Abweichung einer Wahrscheinlichkeit (Δp) und die Übereinstimmung der Greedy-Texte.

**Modelle (gepinnt für P2):**
- `HuggingFaceTB/SmolLM2-135M` @ `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f77366fa6b3b4b768ec3050bf629380bac`

**Umgebung:** GitHub-Actions-Runner (ubuntu-latest, x64), Node 22, ein Thread. Werte auf deinem Laptop folgen über `#/diagnose`.

Zusätzlich prüft `test/engine.test.ts` die Engine bei jedem Push gegen eine unabhängige NumPy-Referenz an einem Mini-Modell (läuft ohne Hugging Face).

## Korrektheit

Schlechtester Wert über alle 6 Prompts (4 Base, 2 Instruct/Chat):

| Format | Top-1 | Top-10 identisch | min. Überlappung | max. rel. L2 | max. KL (nats) | max. Δp | Greedy gleich (min.) |
|---|---|---|---|---|---|---|---|
| f32 | 6/6 | 6/6 | 10/10 | 3,0·10⁻⁶ | 9·10⁻¹² | 0,00 pp | 16/16 |
| f16 | 6/6 | 6/6 | 10/10 | 2,4·10⁻⁶ | 7·10⁻¹² | 0,00 pp | 16/16 |
| int8 pro Zeile | 6/6 | 1/6 | 9/10 | 1,1·10⁻¹ | 6,1·10⁻³ | 2,97 pp | 6/16 |
| int8 Gruppen (64) + fp16-Ausgabe | 6/6 | 2/6 | 9/10 | 8,5·10⁻² | 3,9·10⁻³ | 3,06 pp | 7/16 |

So sieht der Unterschied im Balkendiagramm aus (Prompt `The capital of France is`):

| Token | PyTorch | f32 / f16 | int8 pro Zeile |
|---|---|---|---|
| ` the` | 26,2 % | 26,2 % | 26,4 % |
| ` Paris` | 9,4 % | 9,4 % | 9,7 % |
| ` located` | 7,3 % | 7,3 % | 6,8 % |
| ` called` | 4,4 % | 4,4 % | 4,5 % |
| ` a` | 3,9 % | 3,9 % | 3,7 % |

**Einordnung:** Bei int8 sammelt sich der Rundungsfehler über 30 Layer an. Am Bild der Verteilung ändert das wenig, die Reihenfolge der Top-Tokens bleibt meist gleich. Für eine Seite, die „echt berechnet“ verspricht, muss int8 aber als **„int8-Fassung“** gekennzeichnet werden. Wer exakte Werte will, braucht fp16.

**Startkriterium aus PLAN.md** (Top-10 identisch, rel. L2 < 10⁻², Korrelation > 0,999): von f32/f16 klar erfüllt, von int8 nicht. Das Kriterium bleibt für den exakten Pfad und wird für int8 durch eigene Toleranzen ersetzt (siehe unten).

## Leistung (Base-Modell)

| Format | Download ≈ | RAM Gewichte | Decode | Prefill 16 / 64 / 128 Tokens | Aufbereiten |
|---|---|---|---|---|---|
| f32 (als fp16 ausgeliefert) | 257 MB | 513 MB | 162 ms/Token | 2,1 / 8,3 / 16,7 s | 0,4 s |
| int8 pro Zeile | 129 MB | 129 MB | 171 ms/Token | 2,2 / 8,7 / 17,7 s | 2,0 s* |
| int8 Gruppen + fp16-Ausgabe | 158 MB | 216 MB | 189 ms/Token | – | 3,3 s* |

\* Quantisieren im Spike zur Laufzeit; in P2 passiert das beim Build, der Browser lädt fertige int8-Shards.

- **Decode** (ein neues Token mit KV-Cache): deutlich unter dem Ziel von 500 ms. Ein Schritt-Modus mit ~6 Tokens pro Sekunde fühlt sich flüssig an.
- **Prefill** (Prompt einlesen) läuft derzeit Token für Token und ist der Engpass: Ein Chat-Prompt mit Template (≈ 38 Tokens) braucht ≈ 4,6 s. Gegenmaßnahme in P2: **Batch-Prefill**, also alle Prompt-Tokens gemeinsam als Matrix×Matrix. Jede Gewichtszeile wird dann einmal statt T-mal gelesen; erwartet wird ein Faktor 2–4. Bis dahin gilt ein Prompt-Limit von 128 Tokens im UI.
- **Speicher:** Der Spike hält die komplette safetensors-Datei plus Umrechnungen im Speicher (RSS 1,1–1,6 GB). Im Browser lädt die Engine fertige Shards direkt in das Zielformat; für int8 sind ≈ 200–300 MB realistisch, für fp16→f32 ≈ 600–700 MB.
- int8 ist in reinem JS **nicht schneller** als f32 (die Umrechnung int8×f32 kostet genauso viel). Der Vorteil von int8 ist allein der halbierte Download und der kleinere Speicher.

## Leitbeispiel: Kandidaten

| Prompt | Top-Tokens (Base) | Greedy-Fortsetzung | Eignung |
|---|---|---|---|
| `The capital of France is` | ` the` 26 %, **` Paris` 9 %**, ` located` 7 % | „the capital of the country. The capital of France is …“ | **Sehr gut als roter Faden:** Jeder kennt die Antwort, und das Modell sagt sie *nicht* am wahrscheinlichsten. Daran sieht man, dass ein Modell eine Verteilung über Fortsetzungen ist und keine Faktendatenbank. Das Instruct-Modell antwortet „The capital of France is Paris.“, ein starker Kontrast für Kapitel 10. |
| `The central bank raised interest rates because` | ` it` 28 %, ` the` 18 %, ` of` 12 % … ` inflation` 1,4 % | „it wanted to stimulate the economy.“ | **Gut als Finanz-Beispiel für Grenzen:** Die Fortsetzung klingt plausibel, ist ökonomisch aber falsch (Zinserhöhungen bremsen). Eine Halluzination wie aus dem Lehrbuch für Kapitel 13. Als roter Faden zu unscharf. |
| `Once upon a time, there was a` | ` little` 13 %, ` curious` 9 %, ` wise` 9 %, ` man` 6 % … | „little girl named Lily. She lived in a big house …“ | **Ideal für Sampling/Temperatur** (Kapitel 7): flache Verteilung mit vielen plausiblen Optionen. |
| `The stock market fell sharply after` | ` the` 50 % | „the crash of 1929, and the stock market crash of“ | Wenig ergiebig (ein dominantes Token). |
| Chat: „Why do central banks raise interest rates?“ | `Central` 76 % | „Central banks raise interest rates to manage the risk of inflation …“ | Gutes Instruct-Beispiel für Kapitel 10. |

**Nebenbefund für P1b/P2:** Nach `<|im_end|>` schreibt das Instruct-Modell weiter Template-Tokens. Die Generierung muss am End-Token stoppen (Stopp-Tokens aus der Tokenizer-Config).

## Festgelegte Toleranzen für die Golden-Tests

| Pfad | Kriterien (alle müssen erfüllt sein) | gemessen (schlechtester Wert) |
|---|---|---|
| **exakt** (f32/fp16) | Top-10 identisch · rel. L2 < 10⁻⁴ · KL < 10⁻⁶ · Greedy 16/16 identisch | 2,4·10⁻⁶ · 9·10⁻¹² · 16/16 |
| **int8** | Top-1 identisch · Top-10-Überlappung ≥ 8/10 · Korrelation ≥ 0,997 · KL < 2·10⁻² · max. Δp < 5 pp | 6/6 · 9/10 · 0,9987 · 6,1·10⁻³ · 2,97 pp |

Die int8-Grenzen lassen etwa den Faktor 2–3 Spielraum über dem gemessenen schlechtesten Wert. Echte Fehler (falsche Indizes, vertauschte Gewichte) erzeugen KL-Werte im Bereich > 1 und fallen sofort auf.

## Empfehlung und offene Entscheidungen

1. **Gewichtsformat im Browser**
   - **Empfehlung:** Standard ist **int8 pro Zeile** (≈ 130 MB, ≈ 250 MB RAM), gekennzeichnet als „Echt · int8-Fassung“. Optional gibt es einen Knopf **„Exakte Fassung laden (fp16, ≈ 260 MB)“** für exakt PyTorch-gleiche Werte.
   - Alternative: nur fp16, also immer exakt, aber doppelter Download und rund doppelter Speicher. Kritisch im Firmennetz und bei 8-GB-Laptops.
2. **Leitbeispiel**
   - **Empfehlung:** `The capital of France is` als roter Faden.
   - Dazu `The central bank raised interest rates because` als Finanz- und Halluzinationsbeispiel (Kapitel 13) und `Once upon a time, there was a` für Sampling (Kapitel 7).
3. **P2-Aufgaben aus dem Spike:**
   - Batch-Prefill
   - Stopp-Tokens
   - Modell-Revisionen pinnen
   - int8- und fp16-Shards beim Build erzeugen
   - Golden-Tests mit den Toleranzen oben in `ci.yml` und `deploy.yml`
