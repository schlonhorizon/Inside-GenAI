/**
 * Modellzahlen (PLAN.md 5.5): werden NIE im Text hart codiert, sondern beim Build
 * aus den Modell-Configs berechnet (scripts/generate.mjs → src/generated/model-facts.json).
 */
import facts from '../generated/model-facts.json';

export interface ModelFacts {
  available: boolean;
  label: string;
  source?: string;
  values: Record<string, number | string>;
}

export const MODEL_FACTS = facts as Record<string, ModelFacts>;

export function getFact(model: string, key: string): number | string | undefined {
  const m = MODEL_FACTS[model];
  return m?.available ? m.values[key] : undefined;
}

export type FactFormat = 'de' | 'compact' | 'raw';

export function formatFact(v: number | string, format: FactFormat = 'de'): string {
  if (typeof v === 'string' || format === 'raw') return String(v);
  if (format === 'compact') {
    if (v >= 1e9) return `${(v / 1e9).toLocaleString('de-DE', { maximumFractionDigits: 1 })} Mrd.`;
    if (v >= 1e6) return `${(v / 1e6).toLocaleString('de-DE', { maximumFractionDigits: 1 })} Mio.`;
  }
  return v.toLocaleString('de-DE');
}
