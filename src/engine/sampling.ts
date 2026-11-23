/**
 * Sampling: aus Logits eine Verteilung machen, filtern, ziehen.
 * Reihenfolge wie bei Hugging Face: Temperatur → Top-k → (renormieren) → Top-p → Min-p.
 */

export interface SamplingParams {
  /** 0 = greedy (immer das wahrscheinlichste Token) */
  temperature: number;
  /** 0 = aus */
  topK: number;
  /** 1 = aus */
  topP: number;
  /** 0 = aus; behält Tokens mit p ≥ minP · p_max */
  minP?: number;
}

export const DEFAULT_SAMPLING: SamplingParams = { temperature: 1, topK: 0, topP: 1, minP: 0 };

export interface Distribution {
  /** Wahrscheinlichkeiten nach Temperatur, vor dem Filtern */
  probs: Float64Array;
  /** 1 = bleibt im Rennen, 0 = abgeschnitten */
  kept: Uint8Array;
  /** renormierte Wahrscheinlichkeiten der behaltenen Tokens (daraus wird gezogen) */
  final: Float64Array;
}

export function argmax(x: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
  return best;
}

export function softmax(logits: ArrayLike<number>, temperature = 1): Float64Array {
  const n = logits.length;
  const out = new Float64Array(n);
  if (temperature <= 0) {
    out[argmax(logits)] = 1;
    return out;
  }
  let max = -Infinity;
  for (let i = 0; i < n; i++) max = Math.max(max, logits[i] / temperature);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(logits[i] / temperature - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

/** Indizes nach absteigender Wahrscheinlichkeit (bei Gleichstand kleinere ID zuerst). */
export function rankOrder(p: ArrayLike<number>): number[] {
  return Array.from({ length: p.length }, (_, i) => i).sort((a, b) => p[b] - p[a] || a - b);
}

export function distribution(logits: ArrayLike<number>, params: SamplingParams): Distribution {
  const probs = softmax(logits, params.temperature);
  const n = probs.length;
  const kept = new Uint8Array(n).fill(1);
  const order = rankOrder(probs);

  if (params.temperature <= 0) {
    kept.fill(0);
    kept[order[0]] = 1;
  } else {
    if (params.topK > 0 && params.topK < n) for (let r = params.topK; r < n; r++) kept[order[r]] = 0;
    if (params.topP < 1) {
      let mass = 0;
      for (const i of order) if (kept[i]) mass += probs[i];
      let acc = 0;
      let reached = false;
      for (const i of order) {
        if (!kept[i]) continue;
        if (reached) {
          kept[i] = 0;
          continue;
        }
        acc += probs[i] / mass;
        if (acc >= params.topP - 1e-12) reached = true;
      }
    }
    if (params.minP && params.minP > 0) {
      const pmax = probs[order[0]];
      for (let i = 0; i < n; i++) if (probs[i] < params.minP * pmax) kept[i] = 0;
      kept[order[0]] = 1;
    }
  }

  const final = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) if (kept[i]) sum += probs[i];
  for (let i = 0; i < n; i++) final[i] = kept[i] ? probs[i] / sum : 0;
  return { probs, kept, final };
}

/** Kleiner, schneller, seedbarer Zufallsgenerator (mulberry32). Liefert Zahlen in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Zieht einen Index gemäß Verteilung p mit Zufallszahl u ∈ [0, 1). */
export function sampleIndex(p: ArrayLike<number>, u: number): number {
  let acc = 0;
  let last = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] <= 0) continue;
    last = i;
    acc += p[i];
    if (u < acc) return i;
  }
  return last;
}
