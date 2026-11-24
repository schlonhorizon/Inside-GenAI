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
    expect(d.final[paris]).toBeLessThan(0.7);
    const schoen = model.tokenizer.encode('schön')[0].id;
    const top3 = Array.from(d.final.keys()).sort((x, y) => d.final[y] - d.final[x]).slice(0, 3);
    expect(top3).toContain(schoen);
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

/** Regressionstest: Das Modell soll die Soll-Verteilung der Trainingsgrammatik treffen (docs/toy-model.md). */
describe('Toy-Modelle treffen die Soll-Verteilung', () => {
  const evalSet = JSON.parse(readFileSync(new URL('./fixtures/toy-eval.json', import.meta.url), 'utf8')) as {
    goals: { below80_share: number; entropy_bits: number };
    positions: { prefix: string; soll: Record<string, number> }[];
  };

  for (const variant of ['toy', 'toy-nopos'] as const) {
    it(`${variant}: Hilfs-Zielwerte erfüllt und KL(Soll ‖ Modell) klein`, () => {
      const model = load(variant);
      const report = JSON.parse(readFileSync(new URL(`../public/data/${variant}/eval.json`, import.meta.url), 'utf8'));
      let below = 0;
      let entropy = 0;
      let kl = 0;
      for (const pos of evalSet.positions) {
        const ids = model.promptIds(pos.prefix);
        const p = distribution(model.logitsFor(ids), { temperature: 1, topK: 0, topP: 1 }).final;
        const top = Math.max(...p);
        if (top < 0.8) below++;
        for (const x of p) if (x > 0) entropy -= x * Math.log2(x);
        for (const [w, q] of Object.entries(pos.soll)) {
          const id = w === '<eos>' ? model.tokenizer.eosId : model.tokenizer.encode(w)[0].id;
          kl += q * Math.log2(q / Math.max(p[id], 1e-12));
        }
      }
      const n = evalSet.positions.length;
      expect(below / n).toBeGreaterThanOrEqual(evalSet.goals.below80_share);
      expect(entropy / n).toBeGreaterThanOrEqual(evalSet.goals.entropy_bits);
      // gleiche Größenordnung wie im Trainingslauf gemessen (Faktor 1,5 Spielraum)
      expect(kl / n).toBeLessThan(report.after.kl_bits * 1.5);
    });
  }
});
