# Spike P1a – Ergebnisse (automatisch erzeugt)

- **base:** `HuggingFaceTB/SmolLM2-135M` @ `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- **instruct:** `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- Node v22.23.5 (x64), ein Thread. Referenz: PyTorch fp32.
- Rechengenauigkeit der Engine: Akkumulation in JS-`number` (f64), Aktivierungen/Norms in f32/f64, Softmax in f64; int8 symmetrisch ±127, Rundung `Math.round`.

## Hauptvergleich: Gewichtsformate

| Variante | Spezifikation | Download Base | RAM | Decode | Top-1 | Top-10 ident. | min. Überl. | max. KL | max. Δp | schlechtestes Δp bei | min. Greedy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 (exakt) | `f16` | 257 MB | 513 MB | 99 ms | 6/6 | 6/6 | 10/10 | 7.49e-12 | 0.00 pp | instruct/chat-rates: `Central` 76.0 % → 76.0 % | 16/16 |
| q8 (Skalen f32) | `q8` | 129 MB | 129 MB | – | 6/6 | 1/6 | 9/10 | 6.14e-03 | 2.97 pp | instruct/chat-rates: `Central` 76.0 % → 79.0 % | 6/16 |
| q8 | `q8+s16` | 129 MB | 129 MB | 129 ms | 6/6 | 1/6 | 9/10 | 5.83e-03 | 2.38 pp | base/stocks: `the` 50.1 % → 47.7 % | 1/16 |
| q8g128 | `q8g128+s16` | 129 MB | 130 MB | – | 6/6 | 1/6 | 9/10 | 5.80e-03 | 3.11 pp | base/stocks: `the` 50.1 % → 47.0 % | 1/16 |
| q8g64 | `q8g64+s16` | 132 MB | 136 MB | 118 ms | 6/6 | 1/6 | 9/10 | 3.53e-03 | 3.31 pp | instruct/chat-rates: `Central` 76.0 % → 72.7 % | 8/16 |
| q8g32 | `q8g32+s16` | 136 MB | 144 MB | 122 ms | 6/6 | 1/6 | 9/10 | 2.50e-03 | 2.65 pp | instruct/chat-rates: `Central` 76.0 % → 78.7 % | 7/16 |
| q8g16 | `q8g16+s16` | 144 MB | 160 MB | – | 6/6 | 1/6 | 9/10 | 4.08e-03 | 3.40 pp | instruct/chat-rates: `Central` 76.0 % → 79.4 % | 9/16 |

## Ablation A: nur ein Teil int8, Rest fp16

| Variante | Spezifikation | Download Base | RAM | Decode | Top-1 | Top-10 ident. | min. Überl. | max. KL | max. Δp | schlechtestes Δp bei | min. Greedy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nur Attention (q/k/v/o) q8, Rest fp16 | `attn:q8,mlp:f16,down:f16,emb:f16+s16` | 231 MB | 437 MB | – | 6/6 | 2/6 | 9/10 | 2.22e-03 | 2.28 pp | instruct/chat-rates: `Central` 76.0 % → 78.3 % | 2/16 |
| nur MLP gate/up q8, Rest fp16 | `attn:f16,mlp:q8,down:f16,emb:f16+s16` | 206 MB | 362 MB | – | 6/6 | 1/6 | 9/10 | 7.84e-04 | 1.46 pp | base/stocks: `the` 50.1 % → 48.6 % | 7/16 |
| nur MLP down q8, Rest fp16 | `attn:f16,mlp:f16,down:q8,emb:f16+s16` | 231 MB | 437 MB | – | 6/6 | 0/6 | 9/10 | 5.76e-03 | 1.73 pp | base/story: `curious` 9.3 % → 11.0 % | 6/16 |
| nur Embedding = LM-Head q8, Rest fp16 | `attn:f16,mlp:f16,down:f16,emb:q8+s16` | 230 MB | 432 MB | – | 6/6 | 1/6 | 9/10 | 1.33e-03 | 0.88 pp | base/story: `curious` 9.3 % → 8.4 % | 1/16 |
| nur Attention (q/k/v/o) q8g32, Rest fp16 | `attn:q8g32,mlp:f16,down:f16,emb:f16+s16` | 233 MB | 440 MB | – | 6/6 | 4/6 | 9/10 | 4.64e-03 | 3.72 pp | instruct/chat-rates: `Central` 76.0 % → 79.7 % | 16/16 |
| nur MLP gate/up q8g32, Rest fp16 | `attn:f16,mlp:q8g32,down:f16,emb:f16+s16` | 209 MB | 368 MB | – | 6/6 | 3/6 | 9/10 | 4.57e-04 | 0.50 pp | instruct/chat-capital: `The` 83.1 % → 82.6 % | 7/16 |
| nur MLP down q8g32, Rest fp16 | `attn:f16,mlp:f16,down:q8g32,emb:f16+s16` | 233 MB | 440 MB | – | 6/6 | 2/6 | 9/10 | 1.31e-03 | 1.79 pp | base/capital: `the` 26.2 % → 24.4 % | 9/16 |
| nur Embedding = LM-Head q8g32, Rest fp16 | `attn:f16,mlp:f16,down:f16,emb:q8g32+s16` | 231 MB | 436 MB | – | 6/6 | 0/6 | 9/10 | 3.49e-04 | 0.56 pp | base/capital: `the` 26.2 % → 26.7 % | 7/16 |

## Ablation B: alles int8 außer einem Teil (gemischte Varianten)

| Variante | Spezifikation | Download Base | RAM | Decode | Top-1 | Top-10 ident. | min. Überl. | max. KL | max. Δp | schlechtestes Δp bei | min. Greedy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q8, aber Attention (q/k/v/o) fp16 | `attn:f16,mlp:q8,down:q8,emb:q8+s16` | 154 MB | 205 MB | – | 6/6 | 1/6 | 9/10 | 5.45e-03 | 2.18 pp | base/stocks: `the` 50.1 % → 47.9 % | 1/16 |
| q8, aber MLP gate/up fp16 | `attn:q8,mlp:f16,down:q8,emb:q8+s16` | 179 MB | 281 MB | – | 6/6 | 2/6 | 9/10 | 5.95e-03 | 2.67 pp | instruct/chat-rates: `Central` 76.0 % → 78.7 % | 1/16 |
| q8, aber MLP down fp16 | `attn:q8,mlp:q8,down:f16,emb:q8+s16` | 154 MB | 205 MB | – | 6/6 | 1/6 | 9/10 | 2.65e-03 | 2.15 pp | base/stocks: `the` 50.1 % → 47.9 % | 6/16 |
| q8, aber Embedding = LM-Head fp16 | `attn:q8,mlp:q8,down:q8,emb:f16+s16` | 156 MB | 210 MB | – | 6/6 | 2/6 | 9/10 | 6.90e-03 | 1.85 pp | base/stocks: `the` 50.1 % → 48.2 % | 6/16 |
| q8g32, aber Attention (q/k/v/o) fp16 **← gewählt** | `attn:f16,mlp:q8g32,down:q8g32,emb:q8g32+s16` | 160 MB | 217 MB | – | 6/6 | 2/6 | 9/10 | 1.25e-03 | 1.31 pp | instruct/chat-rates: `Central` 76.0 % → 74.7 % | 9/16 |
| q8g32, aber MLP gate/up fp16 | `attn:q8g32,mlp:f16,down:q8g32,emb:q8g32+s16` | 184 MB | 290 MB | – | 6/6 | 0/6 | 9/10 | 2.96e-03 | 2.92 pp | instruct/chat-rates: `Central` 76.0 % → 78.9 % | 7/16 |
| q8g32, aber MLP down fp16 | `attn:q8g32,mlp:q8g32,down:f16,emb:q8g32+s16` | 160 MB | 217 MB | – | 6/6 | 2/6 | 9/10 | 4.46e-03 | 3.60 pp | instruct/chat-rates: `Central` 76.0 % → 79.6 % | 12/16 |
| q8g32, aber Embedding = LM-Head fp16 | `attn:q8g32,mlp:q8g32,down:q8g32,emb:f16+s16` | 162 MB | 222 MB | – | 6/6 | 2/6 | 9/10 | 2.25e-03 | 2.51 pp | instruct/chat-rates: `Central` 76.0 % → 78.5 % | 7/16 |

## Entscheidung (Regel vorab festgelegt)

Kandidaten: int8-Varianten mit fp16-Skalen und gemischte Varianten. Genaueste: max. Δp = 1.31 pp. Zulässig sind Varianten bis 1.81 pp; davon die kleinste (bei Gleichstand die schnellere).

**Gewählt: q8g32, aber Attention (q/k/v/o) fp16** (`attn:f16,mlp:q8g32,down:q8g32,emb:q8g32+s16`): max. Δp 1.31 pp, KL 1.25e-03, Download Base 160 MB / Instruct 160 MB.

## Pages-Budget

- Seite ohne Modelle (gemessen, inkl. 30-MB-Test-Shard): 34 MB
- (a) int8 Base + int8 Instruct + fp16 Base + fp16 Instruct: **868 MB**
- (b) int8 Base + int8 Instruct + fp16 nur Base: **611 MB**
- Regel: über 800 MB → fp16 nur für Base. **Ergebnis: fp16-Fassung für nur Base.**

## Wahrscheinlichkeiten im Vergleich (Top-5)

**capital**

| Token | PyTorch | fp16 (exakt) | q8 | q8g32, aber Attention (q/k/v/o) fp16 |
|---|---|---|---|---|
| ` the` | 26.2 % | 26.2 % | 26.5 % | 25.2 % |
| ` Paris` | 9.4 % | 9.4 % | 9.8 % | 9.7 % |
| ` located` | 7.3 % | 7.3 % | 6.8 % | 7.5 % |
| ` called` | 4.4 % | 4.4 % | 4.5 % | 4.3 % |
| ` a` | 3.9 % | 3.9 % | 3.6 % | 3.8 % |

**centralbank**

| Token | PyTorch | fp16 (exakt) | q8 | q8g32, aber Attention (q/k/v/o) fp16 |
|---|---|---|---|---|
| ` it` | 27.8 % | 27.8 % | 28.4 % | 27.0 % |
| ` the` | 17.8 % | 17.8 % | 16.8 % | 17.4 % |
| ` of` | 12.3 % | 12.3 % | 11.3 % | 13.5 % |
| ` they` | 9.4 % | 9.4 % | 9.0 % | 9.7 % |
| ` its` | 2.4 % | 2.4 % | 2.4 % | 2.4 % |

**story**

| Token | PyTorch | fp16 (exakt) | q8 | q8g32, aber Attention (q/k/v/o) fp16 |
|---|---|---|---|---|
| ` little` | 12.8 % | 12.8 % | 13.4 % | 12.7 % |
| ` curious` | 9.3 % | 9.3 % | 10.1 % | 9.3 % |
| ` wise` | 8.5 % | 8.5 % | 8.9 % | 8.5 % |
| ` man` | 5.9 % | 5.9 % | 5.0 % | 5.7 % |
| ` group` | 5.6 % | 5.6 % | 5.7 % | 5.9 % |

**stocks**

| Token | PyTorch | fp16 (exakt) | q8 | q8g32, aber Attention (q/k/v/o) fp16 |
|---|---|---|---|---|
| ` the` | 50.1 % | 50.1 % | 47.7 % | 49.3 % |
| ` a` | 4.1 % | 4.1 % | 4.3 % | 4.0 % |
| ` ` | 2.5 % | 2.5 % | 2.4 % | 2.5 % |
| ` that` | 2.3 % | 2.3 % | 2.5 % | 2.3 % |
| ` it` | 1.6 % | 1.6 % | 1.7 % | 1.6 % |

**chat-rates**

| Token | PyTorch | fp16 (exakt) | q8 | q8g32, aber Attention (q/k/v/o) fp16 |
|---|---|---|---|---|
| `Central` | 76.0 % | 76.0 % | 78.0 % | 74.7 % |
| `The` | 6.9 % | 6.9 % | 5.8 % | 7.5 % |
| `When` | 5.9 % | 5.9 % | 5.9 % | 6.1 % |
| `As` | 2.5 % | 2.5 % | 1.9 % | 2.7 % |
| `A` | 1.0 % | 1.0 % | 0.8 % | 1.1 % |

**chat-capital**

| Token | PyTorch | fp16 (exakt) | q8 | q8g32, aber Attention (q/k/v/o) fp16 |
|---|---|---|---|---|
| `The` | 83.1 % | 83.1 % | 84.2 % | 83.0 % |
| `Paris` | 4.1 % | 4.1 % | 3.5 % | 4.3 % |
| `As` | 1.8 % | 1.8 % | 1.6 % | 2.1 % |
| `Bon` | 1.5 % | 1.5 % | 1.4 % | 1.3 % |
| `I` | 1.4 % | 1.4 % | 1.5 % | 1.4 % |

## Leistung (Base-Modell)

| Variante | Aufbereiten | RAM | Download | Prefill 16 / 64 / 128 | Decode |
|---|---|---|---|---|---|
| fp16 (exakt) | 6.1 s | 513 MB | 257 MB | 1.3 / 5.4 / 10.5 s | 99 ms/Token |
| q8 | 1.6 s | 129 MB | 129 MB | 1.7 / 6.7 / 13.6 s | 129 ms/Token |
| q8g64 | 1.8 s | 136 MB | 132 MB | 1.5 / 6.0 / 12.2 s | 118 ms/Token |
| q8g32 | 2.1 s | 144 MB | 136 MB | 1.6 / 6.4 / 12.7 s | 122 ms/Token |

## Leitbeispiel-Kandidaten (PyTorch, fp32)

### `capital` (base, golden): `The capital of France is`

` the` 26.2 % · ` Paris` 9.4 % · ` located` 7.3 % · ` called` 4.4 % · ` a` 3.9 % · ` situated` 3.6 %

Greedy: `the capital of the country.

The capital of France is the capital of`

### `centralbank` (base, golden): `The central bank raised interest rates because`

` it` 27.8 % · ` the` 17.8 % · ` of` 12.3 % · ` they` 9.4 % · ` its` 2.4 % · ` there` 2.4 %

Greedy: `it wanted to stimulate the economy.

The central bank raised interest rates because`

### `story` (base, candidate): `Once upon a time, there was a`

` little` 12.8 % · ` curious` 9.3 % · ` wise` 8.5 % · ` man` 5.9 % · ` group` 5.6 % · ` young` 5.6 %

Greedy: `little girl named Lily. She lived in a big house with her family, but`

### `stocks` (base, candidate): `The stock market fell sharply after`

` the` 50.1 % · ` a` 4.1 % · ` ` 2.5 % · ` that` 2.3 % · ` it` 1.6 % · ` Trump` 1.6 %

Greedy: `the crash of 1929, and the stock market crash of`

### `chat-rates` (instruct, chat): `Why do central banks raise interest rates?`

`Central` 76.0 % · `The` 6.9 % · `When` 5.9 % · `As` 2.5 % · `A` 1.0 % · `At` 0.9 %

Greedy: `Central banks raise interest rates to manage the risk of inflation and inflation risk. When`

### `chat-capital` (instruct, chat): `What is the capital of France?`

`The` 83.1 % · `Paris` 4.1 % · `As` 1.8 % · `Bon` 1.5 % · `I` 1.4 % · `France` 0.9 %

Greedy: `The capital of France is Paris.<|im_end|> <|im_end|><|im_start|>system <|im_end|> <|im_start|>`

