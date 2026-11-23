# Spike P1a – Ergebnisse (automatisch erzeugt)

- **base:** `HuggingFaceTB/SmolLM2-135M` @ `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- **instruct:** `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- Node v22.23.5 (x64), ein Thread

## Korrektheit: Engine vs. PyTorch (fp32-Referenz)

| Modell | Präzision | Prompt | Top-1 | Top-10 identisch | Überlappung | rel. L2 | Korrelation | max. Abw. | Greedy gleich | Start-Kriterium |
|---|---|---|---|---|---|---|---|---|---|---|
| base | f32 | capital | ✓ | ✓ | 10/10 | 9.73e-07 | 1.000000 | 1.955e-05 | 16/16 | ✓ |
| base | f32 | centralbank | ✓ | ✓ | 10/10 | 2.98e-06 | 1.000000 | 3.004e-05 | 16/16 | ✓ |
| base | f32 | story | ✓ | ✓ | 10/10 | 8.76e-07 | 1.000000 | 1.907e-05 | 16/16 | ✓ |
| base | f32 | stocks | ✓ | ✓ | 10/10 | 1.68e-06 | 1.000000 | 3.815e-05 | 16/16 | ✓ |
| base | f16 | capital | ✓ | ✓ | 10/10 | 8.72e-07 | 1.000000 | 1.812e-05 | 16/16 | ✓ |
| base | f16 | centralbank | ✓ | ✓ | 10/10 | 2.37e-06 | 1.000000 | 2.694e-05 | 16/16 | ✓ |
| base | f16 | story | ✓ | ✓ | 10/10 | 7.26e-07 | 1.000000 | 1.907e-05 | 16/16 | ✓ |
| base | f16 | stocks | ✓ | ✓ | 10/10 | 1.04e-06 | 1.000000 | 3.433e-05 | 16/16 | ✓ |
| base | q8 | capital | ✓ | ✓ | 10/10 | 3.25e-02 | 0.999317 | 6.320e-01 | 6/16 | ✗ |
| base | q8 | centralbank | ✓ | ✗ | 10/10 | 1.06e-01 | 0.999456 | 1.027e+00 | 12/16 | ✗ |
| base | q8 | story | ✓ | ✗ | 9/10 | 5.51e-02 | 0.999397 | 8.554e-01 | 16/16 | ✗ |
| base | q8 | stocks | ✓ | ✗ | 9/10 | 1.60e-02 | 0.999453 | 6.095e-01 | 8/16 | ✗ |
| base | q8g64-hp | capital | ✓ | ✗ | 10/10 | 8.51e-02 | 0.999578 | 7.639e-01 | 16/16 | ✗ |
| base | q8g64-hp | centralbank | ✓ | ✓ | 10/10 | 2.31e-02 | 0.999819 | 3.737e-01 | 16/16 | ✗ |
| base | q8g64-hp | story | ✓ | ✗ | 9/10 | 5.47e-02 | 0.999763 | 6.630e-01 | 9/16 | ✗ |
| base | q8g64-hp | stocks | ✓ | ✗ | 9/10 | 1.14e-02 | 0.999818 | 3.763e-01 | 16/16 | ✗ |
| instruct | f32 | chat-rates | ✓ | ✓ | 10/10 | 6.62e-07 | 1.000000 | 3.910e-05 | 16/16 | ✓ |
| instruct | f32 | chat-capital | ✓ | ✓ | 10/10 | 7.65e-07 | 1.000000 | 4.959e-05 | 16/16 | ✓ |
| instruct | f16 | chat-rates | ✓ | ✓ | 10/10 | 6.16e-07 | 1.000000 | 3.386e-05 | 16/16 | ✓ |
| instruct | f16 | chat-capital | ✓ | ✓ | 10/10 | 8.61e-07 | 1.000000 | 4.530e-05 | 16/16 | ✓ |
| instruct | q8 | chat-rates | ✓ | ✗ | 9/10 | 1.54e-02 | 0.999003 | 8.504e-01 | 7/16 | ✗ |
| instruct | q8 | chat-capital | ✓ | ✗ | 10/10 | 3.16e-02 | 0.998682 | 9.822e-01 | 7/16 | ✗ |
| instruct | q8g64-hp | chat-rates | ✓ | ✓ | 10/10 | 1.77e-02 | 0.999610 | 6.441e-01 | 7/16 | ✗ |
| instruct | q8g64-hp | chat-capital | ✓ | ✗ | 9/10 | 1.00e-02 | 0.999601 | 5.999e-01 | 16/16 | ✗ |

## Zusammenfassung je Präzision (schlechtester Wert über alle Prompts)

| Präzision | Top-1 alle | Top-10 identisch | min. Überlappung | max. rel. L2 | min. Korrelation | max. KL (nats) | max. Δp | min. Greedy gleich |
|---|---|---|---|---|---|---|---|---|
| f32 | 6/6 | 6/6 | 10/10 | 2.98e-06 | 1.000000 | 9.15e-12 | 0.00 pp | 16/16 |
| f16 | 6/6 | 6/6 | 10/10 | 2.37e-06 | 1.000000 | 7.49e-12 | 0.00 pp | 16/16 |
| q8 | 6/6 | 1/6 | 9/10 | 1.06e-01 | 0.998682 | 6.14e-03 | 2.97 pp | 6/16 |
| q8g64-hp | 6/6 | 2/6 | 9/10 | 8.51e-02 | 0.999578 | 3.93e-03 | 3.06 pp | 7/16 |

## Wahrscheinlichkeiten im Vergleich (Top-5, PyTorch vs. Engine)

**capital**

| Token | PyTorch | f32 | f16 | q8 | q8g64-hp |
|---|---|---|---|---|---|
| ` the` | 26.2 % | 26.2 % | 26.2 % | 26.4 % | 24.8 % |
| ` Paris` | 9.4 % | 9.4 % | 9.4 % | 9.7 % | 8.7 % |
| ` located` | 7.3 % | 7.3 % | 7.3 % | 6.8 % | 7.7 % |
| ` called` | 4.4 % | 4.4 % | 4.4 % | 4.5 % | 4.1 % |
| ` a` | 3.9 % | 3.9 % | 3.9 % | 3.7 % | 4.4 % |

**centralbank**

| Token | PyTorch | f32 | f16 | q8 | q8g64-hp |
|---|---|---|---|---|---|
| ` it` | 27.8 % | 27.8 % | 27.8 % | 28.3 % | 28.4 % |
| ` the` | 17.8 % | 17.8 % | 17.8 % | 16.7 % | 17.1 % |
| ` of` | 12.3 % | 12.3 % | 12.3 % | 11.5 % | 12.4 % |
| ` they` | 9.4 % | 9.4 % | 9.4 % | 9.1 % | 9.2 % |
| ` its` | 2.4 % | 2.4 % | 2.4 % | 2.4 % | 2.5 % |

**story**

| Token | PyTorch | f32 | f16 | q8 | q8g64-hp |
|---|---|---|---|---|---|
| ` little` | 12.8 % | 12.8 % | 12.8 % | 13.4 % | 12.6 % |
| ` curious` | 9.3 % | 9.3 % | 9.3 % | 10.1 % | 9.8 % |
| ` wise` | 8.5 % | 8.5 % | 8.5 % | 8.9 % | 9.0 % |
| ` man` | 5.9 % | 5.9 % | 5.9 % | 4.9 % | 5.2 % |
| ` group` | 5.6 % | 5.6 % | 5.6 % | 5.7 % | 6.1 % |

**stocks**

| Token | PyTorch | f32 | f16 | q8 | q8g64-hp |
|---|---|---|---|---|---|
| ` the` | 50.1 % | 50.1 % | 50.1 % | 47.9 % | 48.9 % |
| ` a` | 4.1 % | 4.1 % | 4.1 % | 4.3 % | 4.2 % |
| ` ` | 2.5 % | 2.5 % | 2.5 % | 2.5 % | 2.5 % |
| ` that` | 2.3 % | 2.3 % | 2.3 % | 2.5 % | 2.4 % |
| ` it` | 1.6 % | 1.6 % | 1.6 % | 1.7 % | 1.5 % |

**chat-rates**

| Token | PyTorch | f32 | f16 | q8 | q8g64-hp |
|---|---|---|---|---|---|
| `Central` | 76.0 % | 76.0 % | 76.0 % | 79.0 % | 73.0 % |
| `The` | 6.9 % | 6.9 % | 6.9 % | 5.4 % | 8.1 % |
| `When` | 5.9 % | 5.9 % | 5.9 % | 5.6 % | 6.2 % |
| `As` | 2.5 % | 2.5 % | 2.5 % | 1.8 % | 2.9 % |
| `A` | 1.0 % | 1.0 % | 1.0 % | 0.8 % | 1.2 % |

**chat-capital**

| Token | PyTorch | f32 | f16 | q8 | q8g64-hp |
|---|---|---|---|---|---|
| `The` | 83.1 % | 83.1 % | 83.1 % | 84.4 % | 84.0 % |
| `Paris` | 4.1 % | 4.1 % | 4.1 % | 3.5 % | 3.6 % |
| `As` | 1.8 % | 1.8 % | 1.8 % | 1.5 % | 1.9 % |
| `Bon` | 1.5 % | 1.5 % | 1.5 % | 1.3 % | 1.5 % |
| `I` | 1.4 % | 1.4 % | 1.4 % | 1.5 % | 1.3 % |

## Leistung (Base-Modell)

| Präzision | Laden + Aufbereiten | RAM Gewichte | Download ≈ | Prefill 16 / 64 / 128 Tokens | Decode | RSS |
|---|---|---|---|---|---|---|
| f32 | 0.4 s | 513 MB | 257 MB | 2.1 / 8.3 / 16.7 s | 162 ms/Token | 1140 MB |
| q8 | 2.0 s | 129 MB | 129 MB | 2.2 / 8.7 / 17.7 s | 171 ms/Token | 1569 MB |

## Leitbeispiel-Kandidaten (PyTorch, fp32)

### `capital` (base, golden)

Prompt: `The capital of France is` · 5 Tokens

| # | Token | p | Logit |
|---|---|---|---|
| 1 | ` the` | 26.2 % | 14.02 |
| 2 | ` Paris` | 9.4 % | 13.00 |
| 3 | ` located` | 7.3 % | 12.75 |
| 4 | ` called` | 4.4 % | 12.24 |
| 5 | ` a` | 3.9 % | 12.12 |
| 6 | ` situated` | 3.6 % | 12.04 |
| 7 | ` in` | 1.6 % | 11.24 |
| 8 | ` ` | 1.2 % | 10.93 |
| 9 | ` known` | 1.0 % | 10.78 |
| 10 | ` not` | 0.9 % | 10.70 |

Greedy-Fortsetzung: `the capital of the country.

The capital of France is the capital of`

### `centralbank` (base, golden)

Prompt: `The central bank raised interest rates because` · 7 Tokens

| # | Token | p | Logit |
|---|---|---|---|
| 1 | ` it` | 27.8 % | 16.73 |
| 2 | ` the` | 17.8 % | 16.28 |
| 3 | ` of` | 12.3 % | 15.91 |
| 4 | ` they` | 9.4 % | 15.64 |
| 5 | ` its` | 2.4 % | 14.27 |
| 6 | ` there` | 2.4 % | 14.27 |
| 7 | `,` | 1.7 % | 13.92 |
| 8 | ` inflation` | 1.4 % | 13.76 |
| 9 | ` investors` | 1.3 % | 13.68 |
| 10 | ` interest` | 1.2 % | 13.56 |

Greedy-Fortsetzung: `it wanted to stimulate the economy.

The central bank raised interest rates because`

### `story` (base, candidate)

Prompt: `Once upon a time, there was a` · 8 Tokens

| # | Token | p | Logit |
|---|---|---|---|
| 1 | ` little` | 12.8 % | 15.15 |
| 2 | ` curious` | 9.3 % | 14.84 |
| 3 | ` wise` | 8.5 % | 14.75 |
| 4 | ` man` | 5.9 % | 14.39 |
| 5 | ` group` | 5.6 % | 14.33 |
| 6 | ` young` | 5.6 % | 14.33 |
| 7 | ` small` | 3.3 % | 13.80 |
| 8 | ` beautiful` | 3.2 % | 13.76 |
| 9 | ` magical` | 2.1 % | 13.33 |
| 10 | ` girl` | 2.0 % | 13.28 |

Greedy-Fortsetzung: `little girl named Lily. She lived in a big house with her family, but`

### `stocks` (base, candidate)

Prompt: `The stock market fell sharply after` · 6 Tokens

| # | Token | p | Logit |
|---|---|---|---|
| 1 | ` the` | 50.1 % | 9.80 |
| 2 | ` a` | 4.1 % | 7.30 |
| 3 | ` ` | 2.5 % | 6.80 |
| 4 | ` that` | 2.3 % | 6.71 |
| 5 | ` it` | 1.6 % | 6.37 |
| 6 | ` Trump` | 1.6 % | 6.37 |
| 7 | ` President` | 1.3 % | 6.17 |
| 8 | ` its` | 1.0 % | 5.87 |
| 9 | ` this` | 0.9 % | 5.78 |
| 10 | ` New` | 0.6 % | 5.36 |

Greedy-Fortsetzung: `the crash of 1929, and the stock market crash of`

### `chat-rates` (instruct, chat)

Prompt: `Why do central banks raise interest rates?` · 38 Tokens

| # | Token | p | Logit |
|---|---|---|---|
| 1 | `Central` | 76.0 % | 32.89 |
| 2 | `The` | 6.9 % | 30.49 |
| 3 | `When` | 5.9 % | 30.32 |
| 4 | `As` | 2.5 % | 29.47 |
| 5 | `A` | 1.0 % | 28.56 |
| 6 | `At` | 0.9 % | 28.42 |
| 7 | `central` | 0.5 % | 27.90 |
| 8 | `For` | 0.4 % | 27.63 |
| 9 | `Our` | 0.4 % | 27.52 |
| 10 | `Ra` | 0.3 % | 27.33 |

Greedy-Fortsetzung: `Central banks raise interest rates to manage the risk of inflation and inflation risk. When`

### `chat-capital` (instruct, chat)

Prompt: `What is the capital of France?` · 37 Tokens

| # | Token | p | Logit |
|---|---|---|---|
| 1 | `The` | 83.1 % | 31.33 |
| 2 | `Paris` | 4.1 % | 28.33 |
| 3 | `As` | 1.8 % | 27.51 |
| 4 | `Bon` | 1.5 % | 27.35 |
| 5 | `I` | 1.4 % | 27.25 |
| 6 | `France` | 0.9 % | 26.78 |
| 7 | `B` | 0.6 % | 26.42 |
| 8 | `"` | 0.6 % | 26.32 |
| 9 | `In` | 0.4 % | 25.91 |
| 10 | `F` | 0.3 % | 25.59 |

Greedy-Fortsetzung: `The capital of France is Paris.<|im_end|>
<|im_end|><|im_start|>system
<|im_end|>
<|im_start|>`

