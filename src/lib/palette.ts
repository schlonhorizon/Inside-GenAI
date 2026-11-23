/**
 * Metadaten zu den Farbtokens (Werte stehen ausschließlich in tokens.css).
 * Jede Zeile = eine Rolle aus PLAN.md, Abschnitt 6.2.
 */

export interface TokenInfo {
  name: string; // CSS-Variable
  label: string;
}

export interface RoleGroup {
  id: string;
  title: string;
  meaning: string;
  rule: string;
  tokens: TokenInfo[];
  /** bunte Rollenfarbe → fließt in ΔE-Matrix und CVD-Prüfung ein */
  chromatic: boolean;
  /** Mindestkontrast gegen --bg (Text 4,5 · Grafik 3,0 · keiner = undefined) */
  minContrast?: number;
}

export const ROLE_GROUPS: RoleGroup[] = [
  {
    id: 'neutral',
    title: 'Neutral',
    meaning: 'Papier, Flächen, Tinte. Trägt den Großteil der Seite',
    rule: 'Text nur in --ink oder --ink-muted; --ink-faint ausschließlich für Hilfslinien.',
    chromatic: false,
    tokens: [
      { name: '--bg', label: 'Hintergrund' },
      { name: '--surface', label: 'Fläche' },
      { name: '--surface-raised', label: 'Karte' },
      { name: '--border', label: 'Rand' },
      { name: '--ink', label: 'Tinte' },
      { name: '--ink-muted', label: 'Tinte, gedämpft' },
      { name: '--ink-faint', label: 'Hilfslinie' },
    ],
  },
  {
    id: 'accent',
    title: 'Akzent',
    meaning: 'aktiv / ausgewählt / Datenfluss',
    rule: 'Exklusiv. Kommt in keiner anderen Rolle vor.',
    chromatic: true,
    minContrast: 3,
    tokens: [{ name: '--accent', label: 'Terrakotta' }],
  },
  {
    id: 'seq',
    title: 'Sequenziell',
    meaning: 'Größe eines Werts (Gewichte, Wahrscheinlichkeiten, Aktivierungen)',
    rule: 'Mehr = mehr Kontrast zum Hintergrund.',
    chromatic: false,
    tokens: [1, 2, 3, 4, 5].map((i) => ({ name: `--seq-${i}`, label: `Stufe ${i}` })),
  },
  {
    id: 'div',
    title: 'Divergierend',
    meaning: 'Richtung einer Änderung: erhöht (+) ↔ senkt (−)',
    rule: 'Immer zusammen mit ▲/▼ bzw. ±.',
    chromatic: true,
    minContrast: 3,
    tokens: [
      { name: '--div-pos', label: '+ Oliv' },
      { name: '--div-neg', label: '− Pflaume' },
    ],
  },
  {
    id: 'ab',
    title: 'A/B-Vergleich',
    meaning: 'Lauf A vs. Lauf B',
    rule: 'A durchgezogen, B gestrichelt.',
    chromatic: true,
    minContrast: 3,
    tokens: [
      { name: '--ab-a', label: 'A Petrol' },
      { name: '--ab-b', label: 'B Ocker' },
    ],
  },
  {
    id: 'cat',
    title: 'Kategorial',
    meaning: 'nur Attention-Heads und Embedding-Cluster',
    rule: 'Höchstens 3 gleichzeitig hervorgehoben; der Rest bleibt Tinte.',
    chromatic: true,
    minContrast: 3,
    tokens: [
      { name: '--cat-1', label: 'Teal' },
      { name: '--cat-2', label: 'Lavendel' },
      { name: '--cat-3', label: 'Indigo' },
    ],
  },
  {
    id: 'chip',
    title: 'Chip-Abgrenzung',
    meaning: 'trennt benachbarte Token-Chips, ohne Bedeutung',
    rule: 'Abwechselnd; Auswahl über Akzent-Rahmen.',
    chromatic: false,
    tokens: [
      { name: '--chip-a', label: 'Chip A' },
      { name: '--chip-b', label: 'Chip B' },
    ],
  },
];

/** Neben dem Divergenz-Mittelwert, der nicht als Swatch erscheint. */
export const EXTRA_TOKENS = ['--div-mid'];

export const ALL_TOKENS = [...ROLE_GROUPS.flatMap((g) => g.tokens.map((t) => t.name)), ...EXTRA_TOKENS];

export const CHROMATIC_TOKENS = ROLE_GROUPS.filter((g) => g.chromatic).flatMap((g) =>
  g.tokens.map((t) => ({ ...t, group: g.title })),
);

/** Text/Hintergrund-Paare, die WCAG AA (4,5 : 1) erfüllen müssen. */
export const TEXT_PAIRS: [string, string][] = [
  ['--ink', '--bg'],
  ['--ink', '--surface'],
  ['--ink', '--surface-raised'],
  ['--ink-muted', '--bg'],
  ['--ink-muted', '--surface'],
  ['--ink', '--chip-a'],
  ['--ink', '--chip-b'],
  ['--accent', '--bg'],
];
