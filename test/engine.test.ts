import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { parseSafeTensors } from '../src/engine/model/safetensors';
import { configFromHf, LlamaModel, loadLlamaWeights, parseFormat } from '../src/engine/model/llama';
import { quantizeGroups, quantizeRows } from '../src/engine/model/quantize';
import { linearRow, type Linear } from '../src/engine/core/ops';

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

  it('Gruppen-int8 rekonstruiert jedes Gewicht genauer als int8 pro Zeile', () => {
    for (const name of st.names().filter((n) => st.header[n].shape.length === 2)) {
      const { shape, data } = st.get(name);
      const [rows, cols] = shape;
      const err = (l: Linear) => {
        const out = new Float32Array(cols);
        let e = 0;
        for (let r = 0; r < rows; r++) {
          linearRow(l, r, out);
          for (let c = 0; c < cols; c++) e += (out[c] - data[r * cols + c]) ** 2;
        }
        return e;
      };
      expect(err(quantizeGroups(data, rows, cols, 16)), name).toBeLessThan(err(quantizeRows(data, rows, cols)));
    }
  });

  it.each(['q8g16', 'q8g16+s16', 'attn:q8,mlp:q8,down:q8,emb:f16', 'f16'])('%s: Logits nah an der Referenz', (precision) => {
    const model = new LlamaModel(cfg, loadLlamaWeights(st, cfg, precision), 32);
    ref.tokens.forEach((t, i) => {
      expect(relL2(model.step(t)!, ref.logits[i]), `Position ${i}`).toBeLessThan(0.05);
    });
  });

  it('Formatangaben werden korrekt gelesen', () => {
    expect(parseFormat('q8g64+s16')).toEqual({ attn: 'q8g64', mlp: 'q8g64', down: 'q8g64', emb: 'q8g64', scales16: true });
    expect(parseFormat('attn:q8,emb:f16')).toMatchObject({ attn: 'q8', mlp: 'f32', down: 'f32', emb: 'f16', scales16: false });
    expect(() => parseFormat('q4')).toThrow();
  });

  it('prefill + reset liefern dasselbe wie Einzelschritte', () => {
    const model = new LlamaModel(cfg, loadLlamaWeights(st, cfg, 'f32'), 32);
    const a = Array.from(model.prefill(ref.tokens));
    model.reset();
    const b = Array.from(model.prefill(ref.tokens));
    expect(b).toEqual(a);
  });
});
