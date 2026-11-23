import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { ToyLanguageModel } from '../src/engine/lm';
import { distribution, mulberry32, sampleIndex } from '../src/engine/sampling';

/** Spielzeugmodelle (mit RoPE / ohne Positionssignal) gegen PyTorch-Referenzlogits. */

function load(variant: 'toy' | 'toy-nopos') {
  const dir = new URL(`../public/data/${variant}/`, import.meta.url);
  const buf = readFileSync(new URL('model.safetensors', dir));
  return new ToyLanguageModel(
    variant,
    variant,
    JSON.parse(readFileSync(new URL('config.json', dir), 'utf8')),
    JSON.parse(readFileSync(new URL('vocab.json', dir), 'utf8')),
    buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  );
}

for (const variant of ['toy', 'toy-nopos'] as const) {
  describe(`Toy-Modell ${variant}`, () => {
    const model = load(variant);
    const ref = JSON.parse(readFileSync(new URL(`./fixtures/${variant}-reference.json`, import.meta.url), 'utf8')) as {
      prompts: { text: string; ids: number[]; logits: number[][] }[];
    };

    it('Config-Flag rope passt zur Variante', () => {
      expect(model.cfg.rope).toBe(variant === 'toy');
    });

    it.each(ref.prompts.map((p) => [p.text, p] as const))('%s: Logits an jeder Position wie PyTorch', (_t, p) => {
      expect(model.promptIds(p.text)).toEqual(p.ids);
      for (let n = 1; n <= p.ids.length; n++) {
        const logits = model.logitsFor(p.ids.slice(0, n));
        let maxAbs = 0;
        for (let j = 0; j < logits.length; j++) maxAbs = Math.max(maxAbs, Math.abs(logits[j] - p.logits[n - 1][j]));
        expect(maxAbs, `Position ${n - 1}`).toBeLessThan(2e-3);
      }
    });
  });
}

describe('Toy-Generierung', () => {
  const model = load('toy');

  it('Leitbeispiel: „paris“ ist die wahrscheinlichste Fortsetzung, aber nicht die einzige', () => {
    const d = distribution(model.logitsFor(model.promptIds('die hauptstadt von frankreich ist')), { temperature: 1, topK: 0, topP: 1 });
    const paris = model.tokenizer.encode('paris')[0].id;
    const top = d.final.indexOf(Math.max(...d.final));
    expect(top).toBe(paris);
    expect(d.final[paris]).toBeLessThan(0.95);
  });

  it('gleicher Seed → gleicher Text', () => {
    const gen = (seed: number) => {
      const rng = mulberry32(seed);
      const ids = model.promptIds('die bank');
      for (let i = 0; i < 10; i++) {
        const d = distribution(model.logitsFor(ids), { temperature: 1, topK: 0, topP: 0.95 });
        const next = sampleIndex(d.final, rng());
        if (model.eosIds.includes(next)) break;
        ids.push(next);
      }
      return model.decode(ids);
    };
    expect(gen(1)).toBe(gen(1));
  });
});
