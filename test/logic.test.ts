import { describe, expect, it } from 'vitest';
import { contrastRatio, deltaE2000 } from '../src/lib/color';
import { parseHash } from '../src/app/router';
// @ts-expect-error – reines ESM-Skript ohne Typen
import { llamaFacts } from '../scripts/model-facts-lib.mjs';

describe('Router', () => {
  it.each([
    ['', { name: 'home' }],
    ['#/', { name: 'home' }],
    ['#/kapitel/attention', { name: 'chapter', slug: 'attention', anchor: undefined }],
    ['#/kapitel/attention/rechenweg', { name: 'chapter', slug: 'attention', anchor: 'rechenweg' }],
    ['#/kapitel/attention/', { name: 'chapter', slug: 'attention', anchor: undefined }],
    ['#/glossar', { name: 'glossary', id: undefined }],
    ['#/glossar/token', { name: 'glossary', id: 'token' }],
    ['#/diagnose', { name: 'diagnose' }],
    ['#/credits', { name: 'credits' }],
    ['#/styleguide', { name: 'styleguide' }],
    ['#/playground', { name: 'playground' }],
    ['#/gibtsnicht', { name: 'notfound', path: '/gibtsnicht' }],
  ])('%s', (hash, expected) => {
    expect(parseHash(hash)).toEqual(expected);
  });
});

describe('Farbmathematik', () => {
  it('WCAG-Kontrast Schwarz/Weiß = 21', () => {
    expect(contrastRatio('#000000', '#FFFFFF')).toBeCloseTo(21, 5);
  });
  it('ΔE2000 ist 0 für gleiche Farben und symmetrisch', () => {
    expect(deltaE2000('#B5562F', '#B5562F')).toBe(0);
    expect(deltaE2000('#B5562F', '#3E6D85')).toBeCloseTo(deltaE2000('#3E6D85', '#B5562F'), 10);
  });
});

describe('Modellzahlen aus der Config', () => {
  it('Parameterzahl einer Llama-Config mit SmolLM2-135M-Form', () => {
    // Testfixture in der Form einer HF-config.json; die Seite liest die echte Datei.
    const facts = llamaFacts({
      hidden_size: 576,
      num_hidden_layers: 30,
      num_attention_heads: 9,
      num_key_value_heads: 3,
      intermediate_size: 1536,
      vocab_size: 49152,
      max_position_embeddings: 8192,
      tie_word_embeddings: true,
    });
    expect(facts.params).toBe(134_515_008);
    expect(facts.head_dim).toBe(64);
  });
});
