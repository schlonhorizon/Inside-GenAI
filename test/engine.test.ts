import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { parseSafeTensors } from '../src/engine/model/safetensors';
import { configFromHf, LlamaModel, loadLlamaWeights } from '../src/engine/model/llama';

/**
 * Engine gegen eine unabhängige NumPy-Referenz (scripts/spike/make_tiny_fixture.py):
 * winziges Zufalls-Llama mit GQA, RoPE, SwiGLU, gebundenen Embeddings.
 */

const dir = new URL('./fixtures/', import.meta.url);
const buf = readFileSync(new URL('tiny-llama.safetensors', dir));
const st = parseSafeTensors(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const cfg = configFromHf(JSON.parse(readFileSync(new URL('tiny-llama.config.json', dir), 'utf8')));
const ref = JSON.parse(readFileSync(new URL('tiny-llama.reference.json', dir), 'utf8')) as {
  tokens: number[];
  logits: number[][];
};

function relL2(a: ArrayLike<number>, b: ArrayLike<number>) {
  let num = 0;
  let den = 0;
  for (let i = 0; i < b.length; i++) {
    num += (a[i] - b[i]) ** 2;
    den += b[i] ** 2;
  }
  return Math.sqrt(num / den);
}

describe('Llama-Engine (Mini-Modell)', () => {
  it('fp32: Logits an jeder Position stimmen mit der Referenz überein', () => {
    const model = new LlamaModel(cfg, loadLlamaWeights(st, cfg, 'f32'), 32);
    ref.tokens.forEach((t, i) => {
      const logits = model.step(t)!;
      let maxAbs = 0;
      for (let j = 0; j < logits.length; j++) maxAbs = Math.max(maxAbs, Math.abs(logits[j] - ref.logits[i][j]));
      expect(maxAbs, `Position ${i}`).toBeLessThan(1e-3);
    });
  });

  it('int8: Logits bleiben nah an der Referenz (relativer L2-Fehler < 5 %)', () => {
    const model = new LlamaModel(cfg, loadLlamaWeights(st, cfg, 'q8'), 32);
    ref.tokens.forEach((t, i) => {
      const logits = model.step(t)!;
      expect(relL2(logits, ref.logits[i]), `Position ${i}`).toBeLessThan(0.05);
    });
  });

  it.each(['q8g64', 'q8g32', 'q8g64-hp'] as const)('%s: genauer oder gleich gut wie int8 pro Zeile', (precision) => {
    const row = new LlamaModel(cfg, loadLlamaWeights(st, cfg, 'q8'), 32);
    const grp = new LlamaModel(cfg, loadLlamaWeights(st, cfg, precision), 32);
    let errRow = 0;
    let errGrp = 0;
    ref.tokens.forEach((t, i) => {
      errRow += relL2(row.step(t)!, ref.logits[i]);
      errGrp += relL2(grp.step(t)!, ref.logits[i]);
    });
    expect(errGrp).toBeLessThanOrEqual(errRow * 1.05);
  });

  it('prefill + reset liefern dasselbe wie Einzelschritte', () => {
    const model = new LlamaModel(cfg, loadLlamaWeights(st, cfg, 'f32'), 32);
    const a = Array.from(model.prefill(ref.tokens));
    model.reset();
    const b = Array.from(model.prefill(ref.tokens));
    expect(b).toEqual(a);
  });
});
