# Design-Entscheidungen (nach D0)

Stand: 2026-11-23 · Grundlage: Design-Probe (Style Tile) · Quelle der Werte: `src/styles/tokens.css`

## Schriften

| Rolle | Schrift | Einstellungen |
|---|---|---|
| Headlines | **Fraunces** (Variable) | `SOFT 50`, `WONK 1` in Display-Größen, `font-optical-sizing: auto` |
| Fließtext | **Inter** (Variable) | 17–18 px, Zeilenhöhe 1,6, Satzbreite ~68 Zeichen |
| Monospace | **JetBrains Mono** | Ligaturen aus (`font-variant-ligatures: none`), `tabular-nums` |

Alle Schriften stehen unter der SIL Open Font License 1.1 und werden lokal über @fontsource eingebunden.
Verworfen wurden Instrument Serif, Young Serif, IBM Plex Sans und IBM Plex Mono. Sie fliegen in P0 aus dem Bundle.

**Warum Ligaturen aus?** JetBrains Mono setzt `<|` standardmäßig als ◁. Tokens wie `<|im_start|>` müssen aber wörtlich erscheinen.

## Farben

### Neutral

| Token | Hell | Dunkel |
|---|---|---|
| `--bg` | `#FAF8F3` | `#252523` |
| `--surface` | `#F3F0E8` | `#2E2D2B` |
| `--surface-raised` | `#FFFDF9` | `#373634` |
| `--border` | `#E3DED3` | `#44423F` |
| `--ink` | `#2A2722` | `#EDEAE4` |
| `--ink-muted` | `#6B655B` | `#ABA79F` |
| `--ink-faint` | `#A39C90` | `#76726D` |

**Dunkelmodus minimal kälter (Runde 4):** Der Dunkelmodus ging ganz leicht ins Braune. Deshalb ist die Buntheit (Lab-Chroma) der dunklen Neutraltöne um etwa 25–40 % reduziert, bei praktisch unveränderter Helligkeit (ΔL ≤ 0,4). Die Kontraste bleiben gleich (Tinte/Hintergrund 12,8 : 1, gedämpfte Tinte/Hintergrund 6,4 : 1).

### Rollen (Farbe → Bedeutung)

| Rolle | Hell | Dunkel | Bedeutung |
|---|---|---|---|
| Akzent | `#B5562F` | `#E08A63` | aktiv / ausgewählt / Datenfluss |
| Sequenziell | `#EFE6D6 → #D6C6AD → #B39A78 → #87704F → #574632` | `#3A3631 → #5E5446 → #8C7A60 → #BCA585 → #E6D5B8` | Größe eines Werts |
| Divergierend | `#5E7A3A` ↔ `#DCD5C8` ↔ `#8A4F72` | `#9DB873` ↔ `#484642` ↔ `#C48AAE` | erhöht (+) / senkt (−), immer mit ▲/▼ |
| A/B | `#3E6D85` / `#A38419` | `#7FB0C8` / `#E0B84A` | Lauf A (durchgezogen) / Lauf B (gestrichelt) |
| Kategorial | `#1A8574` · `#8A7CC4` · `#3F467F` | `#62C4AC` · `#BDB3F2` · `#6A72C4` | nur Attention-Heads und Embedding-Cluster, max. 3 gleichzeitig |
| Chips | `#F1ECE3` / `#E6DFD2` | `#33322F` / `#3C3A37` | nur Abgrenzung, keine Bedeutung |

### Ergebnisse der D0-Prüfung

- **Kategorial auf 3 Töne reduziert:** Mit 4 Tönen lagen Waldgrün ↔ Oliv und Schiefer ↔ Petrol bei ΔE2000 ≈ 10 (Ziel ≥ 15).
- **Ocker B aufgehellt** (`#A0711F` → `#A38419`, dunkel `#D9A94F` → `#E0B84A`): Bei Deuteranopie war er vorher kaum vom Terrakotta-Akzent zu unterscheiden.
- **Bekannte Grenze:** Bei Deuteranopie liegen im Dunkelmodus Akzent und Oliv nah beieinander (min ΔE ≈ 3,5). Abgefangen wird das durch die Regel, dass Farbe nie allein Bedeutung trägt (▲/▼, Linienstile, Rahmen).
