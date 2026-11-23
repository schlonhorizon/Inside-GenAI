/**
 * BPE lernen – die didaktische Mini-Version (Kapitel 2, „BPE live“):
 * Start mit einzelnen Zeichen, dann immer das häufigste benachbarte Paar verschmelzen.
 * Gleichstand: das Paar, das im Text zuerst vorkommt (reproduzierbar).
 */

export interface BpeWord {
  symbols: string[];
  count: number;
}

export interface BpeState {
  words: BpeWord[];
  merges: { pair: [string, string]; count: number }[];
}

export function initBpe(text: string): BpeState {
  const counts = new Map<string, number>();
  for (const w of text.split(/\s+/).filter(Boolean)) counts.set(w, (counts.get(w) ?? 0) + 1);
  return { words: [...counts].map(([w, count]) => ({ symbols: Array.from(w), count })), merges: [] };
}

export function pairCounts(state: BpeState): { pair: [string, string]; count: number }[] {
  const counts = new Map<string, { pair: [string, string]; count: number; first: number }>();
  let pos = 0;
  for (const w of state.words) {
    for (let i = 0; i < w.symbols.length - 1; i++) {
      const pair: [string, string] = [w.symbols[i], w.symbols[i + 1]];
      const key = pair.join('\u0000');
      const e = counts.get(key);
      if (e) e.count += w.count;
      else counts.set(key, { pair, count: w.count, first: pos });
      pos++;
    }
  }
  return [...counts.values()].sort((a, b) => b.count - a.count || a.first - b.first).map(({ pair, count }) => ({ pair, count }));
}

/** Ein Lernschritt; liefert null, wenn nichts mehr zu verschmelzen ist. */
export function stepBpe(state: BpeState): BpeState | null {
  const best = pairCounts(state)[0];
  if (!best) return null;
  const [a, b] = best.pair;
  const words = state.words.map((w) => {
    const out: string[] = [];
    for (let i = 0; i < w.symbols.length; i++) {
      if (i < w.symbols.length - 1 && w.symbols[i] === a && w.symbols[i + 1] === b) {
        out.push(a + b);
        i++;
      } else out.push(w.symbols[i]);
    }
    return { symbols: out, count: w.count };
  });
  return { words, merges: [...state.merges, best] };
}

export function vocabulary(state: BpeState): string[] {
  const set = new Set<string>();
  for (const w of state.words) for (const s of w.symbols) set.add(s);
  return [...set].sort((x, y) => y.length - x.length || x.localeCompare(y, 'de'));
}

export function totalTokens(state: BpeState): number {
  return state.words.reduce((n, w) => n + w.symbols.length * w.count, 0);
}
