import { describe, expect, it } from 'vitest';
import { distribution, mulberry32, sampleIndex, softmax } from '../src/engine/sampling';

const logits = [2.0, 1.0, 0.5, 0.0, -1.0];
const sum = (a: ArrayLike<number>) => Array.from(a).reduce((x, y) => x + y, 0);

describe('Sampling', () => {
  it('Softmax summiert zu 1, Temperatur verändert die Schärfe', () => {
    const p1 = softmax(logits, 1);
    const p05 = softmax(logits, 0.5);
    const p2 = softmax(logits, 2);
    expect(sum(p1)).toBeCloseTo(1, 12);
    expect(p05[0]).toBeGreaterThan(p1[0]);
    expect(p2[0]).toBeLessThan(p1[0]);
  });

  it('Temperatur 0 ist greedy', () => {
    const d = distribution(logits, { temperature: 0, topK: 0, topP: 1 });
    expect(Array.from(d.final)).toEqual([1, 0, 0, 0, 0]);
  });

  it('Top-k behält genau k Tokens', () => {
    const d = distribution(logits, { temperature: 1, topK: 2, topP: 1 });
    expect(Array.from(d.kept)).toEqual([1, 1, 0, 0, 0]);
    expect(sum(d.final)).toBeCloseTo(1, 12);
  });

  it('Top-p: kleinste Menge mit Masse ≥ p; das wahrscheinlichste Token bleibt immer', () => {
    const p = softmax(logits, 1);
    const d = distribution(logits, { temperature: 1, topK: 0, topP: p[0] + p[1] - 1e-9 });
    expect(Array.from(d.kept)).toEqual([1, 1, 0, 0, 0]);
    const tiny = distribution(logits, { temperature: 1, topK: 0, topP: 0.01 });
    expect(Array.from(tiny.kept)).toEqual([1, 0, 0, 0, 0]);
    const all = distribution(logits, { temperature: 1, topK: 0, topP: 1 });
    expect(Array.from(all.kept)).toEqual([1, 1, 1, 1, 1]);
  });

  it('Top-p wird nach Top-k auf der renormierten Verteilung angewandt', () => {
    const d = distribution(logits, { temperature: 1, topK: 3, topP: 0.999 });
    expect(Array.from(d.kept)).toEqual([1, 1, 1, 0, 0]);
  });

  it('Min-p behält Tokens mit p ≥ minP · p_max', () => {
    const p = softmax(logits, 1);
    const d = distribution(logits, { temperature: 1, topK: 0, topP: 1, minP: p[2] / p[0] - 1e-9 });
    expect(Array.from(d.kept)).toEqual([1, 1, 1, 0, 0]);
  });

  it('gleicher Seed → gleiche Folge, anderer Seed → andere Folge', () => {
    const d = distribution(logits, { temperature: 1.5, topK: 0, topP: 1 });
    const draw = (seed: number) => {
      const rng = mulberry32(seed);
      return Array.from({ length: 50 }, () => sampleIndex(d.final, rng()));
    };
    expect(draw(42)).toEqual(draw(42));
    expect(draw(42)).not.toEqual(draw(43));
  });

  it('Häufigkeiten nähern sich der Verteilung an', () => {
    const d = distribution(logits, { temperature: 1, topK: 0, topP: 1 });
    const rng = mulberry32(7);
    const counts = new Array(5).fill(0);
    const n = 20000;
    for (let i = 0; i < n; i++) counts[sampleIndex(d.final, rng())]++;
    counts.forEach((c, i) => expect(c / n).toBeCloseTo(d.final[i], 1));
  });
});
